import math
import numpy as np
from os.path import join

import torch
from torch import nn, randn
import torch.nn.functional as F

import clip
from clip.model import CLIP, convert_weights

from einops import rearrange


def get_model(opt, name='Model'):
    model = eval(name)(opt)
    model.cuda()
    model = nn.DataParallel(model)
    return model


def xcorr_depthwise(x, kernel):
    """
    depthwise cross correlation
    ref: https://github.com/JudasDie/SOTS/blob/SOT/lib/models/sot/head.py#L227
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2))
    kernel = kernel.view(batch*channel, 1, kernel.size(2))
    out = F.conv1d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2))
    return out


class MyCLIP(CLIP):
    def __init__(self, *args):
        super(MyCLIP, self).__init__(*args)

    def encode_text_2(self, text, truncation=10):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        hidden = x[torch.arange(x.shape[0]), :truncation] @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return hidden, x

    def encode_text_(self, text):
        device = text.device
        B, L = text.size()  # L=77 (i.e., context_length)

        # original token/embedding
        token = text.detach()
        embedding = self.token_embedding(text).type(self.dtype).detach()

        # new token/embedding
        prompt_token = torch.zeros(B, 77)
        text_embedding = self.embedding(torch.arange(77).to(device))[None, :].repeat(B, 1, 1)  # [batch_size, n_ctx, d_model]

        # write token/embedding
        prefix, postfix = 4, 4
        for i in range(B):
            ind = torch.argmax(token[i], -1)  # EoT
            prompt_token[i, 0] = token[i, 0]
            prompt_token[i, prefix+1:prefix+ind] = token[i, 1:ind]
            prompt_token[i, prefix+ind+postfix] = token[i, ind]
            text_embedding[i, 0] = embedding[i,0]
            text_embedding[i, prefix+1: prefix+ind] = embedding[i, 1:ind]
            text_embedding[i, prefix+ind+postfix] = embedding[i, ind]
        prompt_token.to(device)
        text_embedding.to(device)
        x, text = text_embedding, prompt_token

        # copy from the original codes
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def load_clip(model_path, input_resolution=None):
    state_dict = torch.jit.load(model_path).state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    if input_resolution is not None:
        if input_resolution != image_resolution:
            del state_dict['visual.attnpool.positional_embedding']
        image_resolution = input_resolution

    model = MyCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)

    return model


class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mlp(x)
        x = x + self.drop(y)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.clip = load_clip(
            join(opt.save_root, f'CLIP/{opt.clip_model}.pt'),
            input_resolution=224,
        )
        self.clip = self.clip.float()
        self.img_dim = 2048
        self.text_dim = 1024
        self.img_fc = self.get_img_fc(use_ln=False)
        self.text_fc = self.get_text_fc(use_ln=False)
        self._freeze_text_encoder()

        self.fusion_local_global = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0.,
        )

        local_reso = 7 * 7
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
        global_reso = 21 * 21
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso))

        if self.opt.kum_mode == 'cascade attention':
            self.fusion_visual_textual = nn.MultiheadAttention(
                embed_dim=self.img_dim,
                num_heads=4,
                dropout=0,
            )
            self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)
            self.fusion_ffn = FFN(self.img_dim, 0.1)
            self.fusion_drop = nn.Dropout(p=0.1)
        elif self.opt.kum_mode in ('cross correlation', 'text-first modulation'):
            self.fusion_conv1 = nn.Sequential(
                nn.Conv1d(self.text_dim, self.img_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.img_dim),
            )
            self.fusion_conv2 = nn.Sequential(
                nn.Conv1d(self.img_dim, self.img_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.img_dim),
            )
            self.fusion_drop = nn.Dropout(p=0.1)

    def _freeze_text_encoder(self):
        """
        These parameters are not frozen:
        - list(self.clip.token_embedding.parameters())
        - [self.clip.positional_embedding]
        """
        for p in list(self.clip.transformer.parameters()) + \
                 list(self.clip.ln_final.parameters()) + \
                 [self.clip.text_projection, ]:
            p.requires_grad = False

    def _init_weights_function(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, epoch=1e5):
        output = dict()
        textual_hidden, textual_feat = self.textual_encoding(x['exp'])
        if self.opt.kum_mode and (epoch >= self.opt.tg_epoch):
            if self.opt.kum_mode == 'cascade attention':
                visual_feat = self.visual_local_global(
                    x['local_img'], x['global_img'], textual_hidden, self.opt.kum_mode
                )
            elif self.opt.kum_mode in ['cross correlation', 'text-first modulation']:
                visual_feat = self.visual_local_global(
                    x['local_img'], x['global_img'], textual_feat, self.opt.kum_mode
                )
        else:
            visual_feat = self.visual_local_global(x['local_img'], x['global_img'])
        logits = F.cosine_similarity(visual_feat, textual_feat)
        output['logits'] = logits
        output['vis_feat'] = visual_feat
        output['text_feat'] = textual_feat
        return output

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [bt,c,l]->[bt,c]
        # temporal pooling
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [b,c]
        # projection
        feat = self.img_fc(feat)
        return feat

    def cross_modal_fusion(self, vis_feat, text_feat, b, t, mode):
        if mode == 'cascade attention':
            assert len(text_feat.size()) == 3
            # get textual embeddings
            text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
            text_feat = text_feat.repeat([1, t, 1, 1])
            text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
            text_feat = self.fusion_fc(text_feat)
            text_feat = rearrange(text_feat, 'bt l c -> l bt c')
            # fusion
            fused_feat = self.fusion_visual_textual(
                query=vis_feat,
                key=text_feat,
                value=text_feat,
            )[0]
            vis_feat = vis_feat * fused_feat
            vis_feat = rearrange(vis_feat, 'l bt c -> bt c l')
            return vis_feat
        elif mode == 'cross correlation':
            assert len(text_feat.size()) == 2
            # get textual embeddings
            text_feat = text_feat.unsqueeze(1)  # [b,c]->[b,1,c]
            text_feat = text_feat.repeat([1, t, 1])  # [b,t,c]
            text_feat = rearrange(text_feat, 'b t c -> (b t) c 1')  # [bt,c,1]
            text_feat = self.fusion_conv1(text_feat)  # [bt,c,1]
            # fusion
            vis_feat = rearrange(vis_feat, 'HW bt c -> bt c HW')  # [bt,c,l]
            fused_feat = xcorr_depthwise(vis_feat, kernel=text_feat)  # [bt,c,l]
            vis_feat = vis_feat + self.fusion_drop(fused_feat)
            vis_feat = self.fusion_conv2(vis_feat)
            return vis_feat
        elif mode == 'text-first modulation':
            assert len(text_feat.size()) == 2
            L, _, _ = vis_feat.size()
            # get textual embeddings
            text_feat = text_feat.unsqueeze(1)  # [b,c]->[b,1,c]
            text_feat = text_feat.repeat([1, t, 1])  # [b,t,c]
            text_feat = rearrange(text_feat, 'b t c -> (b t) c 1')  # [bt,c,1]
            text_feat = self.fusion_conv1(text_feat)  # [bt,c,1]
            text_feat = text_feat.repeat([1, 1, L])  # [bt,c,HW]
            # fusion
            vis_feat = rearrange(vis_feat, 'HW bt c -> bt c HW')
            out_feat = vis_feat * self.fusion_drop(text_feat)
            out_feat = rearrange(out_feat, 'bt c HW -> HW bt c')
            return out_feat

    def visual_local_global(self, local_img, global_img, text_feat=None, kum_mode=None):
        b, t = global_img.size()[:2]
        # spatial encoding
        local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
        local_feat = self.clip.visual(local_img, with_pooling=False)  # [bt,c,7,7]
        bt, c, h, w = local_feat.size()
        global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        global_feat = self.clip.visual(global_img, with_pooling=False)  # [bt,c,7,7]
        bt, c, H, W = global_feat.size()
        # rearrange
        local_feat = rearrange(local_feat, 'bt c h w -> bt c (h w)')
        global_feat = rearrange(global_feat, 'bt c H W -> bt c (H W)')
        local_feat = local_feat + self.pos_emb_local
        global_feat = global_feat + self.pos_emb_global
        local_feat = rearrange(local_feat, 'bt c hw -> hw bt c')
        global_feat = rearrange(global_feat, 'bt c HW -> HW bt c')
        # text-guided
        if kum_mode == 'text-first modulation':
            local_feat_2 = self.cross_modal_fusion(
                local_feat, text_feat, b, t, kum_mode
            )
            global_feat_2 = self.cross_modal_fusion(
                global_feat, text_feat, b, t, kum_mode
            )
            fusion_feat = self.fusion_local_global(
                query=local_feat_2,
                key=global_feat_2,
                value=global_feat,
            )[0]
        else:
            # cross-attention
            fusion_feat = self.fusion_local_global(
                query=local_feat,
                key=global_feat,
                value=global_feat,
            )[0]
        fusion_feat = fusion_feat + local_feat  # [HW,bt,c]
        # text-guided
        if kum_mode in ('cascade attention', 'cross correlation'):
            fusion_feat= self.cross_modal_fusion(
                fusion_feat, text_feat, b, t, kum_mode
            )
        else:
            fusion_feat = rearrange(fusion_feat, 'HW bt c -> bt c HW')
        fusion_feat = self.st_pooling(fusion_feat, bs=b)
        if self.training:
            return fusion_feat
        else:
            fusion_feat = F.normalize(fusion_feat, p=2, dim=-1)
            return fusion_feat

    def textual_encoding(self, tokens):
        x_hidden, x = self.clip.encode_text_2(tokens, self.opt.truncation)
        x = self.text_fc(x)
        if self.training:
            return x_hidden, x
        else:
            return x_hidden, F.normalize(x, p=2, dim=-1)

    def get_img_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.img_dim, self.opt.feature_dim),
                nn.LayerNorm(self.opt.feature_dim, eps=1e-12),
            )
        else:
            return nn.Linear(self.img_dim, self.opt.feature_dim)

    def get_text_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(),
                nn.Linear(self.text_dim, self.opt.feature_dim),
                nn.LayerNorm(self.opt.feature_dim, eps=1e-12),
            )
        else:
            return nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(),
                nn.Linear(self.text_dim, self.opt.feature_dim),
            )


if __name__ == '__main__':
    from opts import opt
    model = Model(opt)
    a = 1