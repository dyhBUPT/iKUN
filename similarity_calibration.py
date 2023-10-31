import os
import sys
import json
import itertools
import numpy as np
from tqdm import tqdm
from os.path import join
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

from clip import load

from opts import opt
from utils import VIDEOS, tokenize, expression_conversion


def encode_text():
    EXP_DIR = join(opt.data_root, 'expression')
    TEXT_FEAT_DICT = dict()
    clip, _ = load('ViT-B/32')
    clip.cuda()
    clip.eval()
    for mode in ('train', 'test'):
        TEXT_FEAT_DICT[mode] = defaultdict(
            lambda: {
                'feature': None,
                'bbox_num': 0,
                'probability': 0.,
            }
        )
        NUM_BBOX = 0
        for video in VIDEOS[mode]:
            VIDEO_DIR = join(EXP_DIR, video)
            for exp_file in os.listdir(VIDEO_DIR):
                exp = exp_file[:-5]
                exp = expression_conversion(exp)
                exp_gt = json.load(open(join(VIDEO_DIR, exp_file)))
                if mode == 'train':
                    # calculate the number of positive bboxes for each expression
                    bbox_num = sum(len(x) for x in exp_gt['label'].values())
                    TEXT_FEAT_DICT[mode][exp]['bbox_num'] += bbox_num
                    NUM_BBOX += bbox_num
                if TEXT_FEAT_DICT[mode][exp]['feature'] is None:
                    text = tokenize(exp).cuda()
                    feat = clip.encode_text(text)
                    feat = F.normalize(feat, p=2)
                    feat = feat.detach().cpu().tolist()[0]
                    TEXT_FEAT_DICT[mode][exp]['feature'] = feat
            if mode == 'train':
                for exp in TEXT_FEAT_DICT[mode]:
                    TEXT_FEAT_DICT[mode][exp]['probability'] = \
                        TEXT_FEAT_DICT[mode][exp]['bbox_num'] / NUM_BBOX

    json.dump(TEXT_FEAT_DICT, open(join(opt.save_root, 'text_feat_bboxNum.json'), 'w'))


def similarity_calibration(TEXT_FEAT_DICT, CLS_DICT, a, b, tau):
    fn = lambda x: a * x + b

    cls_dict = deepcopy(CLS_DICT)
    FEATS = np.array([x['feature'] for x in TEXT_FEAT_DICT['train'].values()])
    PROBS = np.array([x['probability'] for x in TEXT_FEAT_DICT['train'].values()])

    for video, video_value in cls_dict.items():
        for obj_id, obj_value in  video_value.items():
            for frame, frame_value in obj_value.items():
                for exp, exp_value in frame_value.items():
                    exp_new = expression_conversion(exp)
                    feat = np.array(TEXT_FEAT_DICT['test'][exp_new]['feature'])[None, :]
                    sim = (feat @ FEATS.T)[0]
                    sim = (sim - sim.min()) / (sim.max() - sim.min())
                    weight = np.exp(tau * sim) / np.exp(tau * sim).sum()
                    prob = (weight * PROBS).sum()
                    new_exp_value = [
                        x + fn(prob) for x in exp_value
                    ]
                    frame_value[exp] = new_exp_value

    return cls_dict
