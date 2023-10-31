import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

import warnings
warnings.filterwarnings('ignore')

# import `opts` first to set gpus
from opts import opt

from utils import *
from model import get_model
from dataloader import get_dataloader, get_transform
from similarity_calibration import similarity_calibration


def test_accuracy_v1(model, dataloader, save_img=False):
    model.eval()
    TP, FP, FN = 0, 0, 0
    assert dataloader.batch_size == 1
    if save_img:
        save_dir = join(opt.save_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)
        global_idx = 1
        un_norm = get_transform('unnorm', opt, -1)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
        # for batch_idx, data in enumerate(dataloader):
            # load
            expressions = data['target_expressions']
            expressions = expressions[0].split(',')
            labels = data['target_labels'][0]
            images = data['cropped_images']
            images = images.repeat_interleave(len(expressions), dim=0)
            # forward
            inputs = dict(
                img=images.cuda(),
                exp=tokenize(expressions).cuda(),
            )
            logits = model(inputs).cpu()
            # evaluate
            TP += ((logits >= 0) * (labels == 1)).sum()
            FP += ((logits >= 0) * (labels == 0)).sum()
            FN += ((logits < 0) * (labels == 1)).sum()
            # save images
            if save_img:
                imgs = un_norm(inputs['img'])
                for i in range(len(imgs)):
                    file_name = '{}_{}_{:.0f}_{:.2f}.jpg'.format(
                        global_idx,
                        expressions[i].replace(' ', '-'),
                        labels[i],
                        logits[i]
                    )
                    save_image(
                        imgs[i],
                        join(save_dir, file_name)
                    )
                    global_idx += 1

    PRECISION = TP / (TP + FP) * 100
    RECALL = TP / (TP + FN) * 100
    return PRECISION, RECALL


def test_accuracy(model, dataloader, save_img=False):
    model.eval()
    TP, FP, FN = 0, 0, 0
    assert dataloader.batch_size == 1
    if save_img:
        save_dir = join(opt.save_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)
        global_idx = 1
        un_norm = get_transform('unnorm', opt, -1)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
        # for batch_idx, data in enumerate(dataloader):
            # load
            expressions = data['target_expressions']
            expressions = expressions[0].split(',')
            labels = data['target_labels'][0]
            # forward
            inputs = dict(
                local_img=data['cropped_images'].cuda().repeat_interleave(len(expressions), dim=0),
                global_img=data['global_images'].cuda().repeat_interleave(len(expressions), dim=0),
                exp=tokenize(expressions).cuda(),
            )
            logits = model(inputs)['logits'].cpu()
            # evaluate
            TP += ((logits >= 0) * (labels == 1)).sum()
            FP += ((logits >= 0) * (labels == 0)).sum()
            FN += ((logits < 0) * (labels == 1)).sum()
            # save images
            if save_img:
                local_img = data['cropped_images'].squeeze(0)
                global_img = data['global_images'].squeeze(0)
                local_img = F.interpolate(local_img, global_img.size()[2:])
                imgs = un_norm(
                    torch.cat(
                        (local_img, global_img),
                        dim=0
                    )
                )
                imgs = imgs.repeat(len(expressions), 1, 1, 1, 1)
                for i in range(len(imgs)):
                    file_name = '{}_{}_{:.0f}_{:.2f}.jpg'.format(
                        global_idx,
                        expressions[i].replace(' ', '-'),
                        labels[i],
                        logits[i]
                    )
                    save_image(
                        imgs[i],
                        join(save_dir, file_name)
                    )
                    global_idx += 1

    PRECISION = TP / (TP + FP) * 100
    RECALL = TP / (TP + FN) * 100
    print(TP, FP, FN)
    return PRECISION, RECALL


def test_tracking(model, dataloader):
    print('========== Testing Tracking ==========')
    model.eval()
    OUTPUTS = multi_dim_dict(4, list)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            # forward
            inputs = dict(
                local_img=data['cropped_images'].cuda(),
                global_img=data['global_images'].cuda(),
                exp=tokenize(data['expression_new']).cuda(),
            )
            similarity = model(inputs)['logits'].cpu()
            for idx in range(len(data['video'])):
                for frame_id in range(data['start_frame'][idx], data['stop_frame'][idx] + 1):
                    frame_dict = OUTPUTS[data['video'][idx]][int(data['obj_id'][idx])][int(frame_id)]
                    frame_dict[data['expression_raw'][idx]].append(similarity[idx].cpu().numpy().tolist())
    return OUTPUTS


def generate_final_results(cls_dict, data_dir, track_dir, save_dir, thr_score=0.):
    """
    给定`test_tracking`输出的结果，生成最终跟踪结果
    - cls_dict: video->id->frame->exp->
    """
    template_dir = join(data_dir, 'gt_template')
    if exists(save_dir):
        shutil.rmtree(save_dir)
    for video in os.listdir(template_dir):
        if video not in cls_dict:
            continue
        video_dir_in = join(template_dir, video)
        video_dir_out = join(save_dir, video)
        MIN_FRAME, MAX_FRAME = FRAMES[video]
        # symbolic link for `gt.txt`
        for exp in os.listdir(video_dir_in):
            exp_dir_in = join(video_dir_in, exp)
            exp_dir_out = join(video_dir_out, exp)
            os.makedirs(exp_dir_out, exist_ok=True)
            gt_path_in = join(exp_dir_in, 'gt.txt')
            gt_path_out = join(exp_dir_out, 'gt.txt' )
            if not exists(gt_path_out):
                os.symlink(gt_path_in, gt_path_out)
        # load tracks
        # noinspection PyBroadException
        try:
            tracks = np.loadtxt(join(track_dir, video, 'all', 'gt.txt'), delimiter=',')
        except:
            tracks_1 = np.loadtxt(join(track_dir, video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0
            tracks_2 = np.loadtxt(join(track_dir, video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)
        # generate `predict.txt`
        video_dict = cls_dict[video]
        for obj_id, obj_dict in video_dict.items():
            for frame_id, frame_dict in obj_dict.items():
                for exp in EXPRESSIONS[video]:
                    if exp in EXPRESSIONS['dropped']:
                        continue
                    if exp not in frame_dict:  # TODO:可删
                        continue
                    exp_dir_out = join(video_dir_out, exp)
                    score = np.mean(frame_dict[exp])
                    with open(join(exp_dir_out, 'predict.txt'), 'a') as f:
                        if score > thr_score:
                            bbox = tracks[
                                (tracks[:, 0] == int(frame_id)) *
                                (tracks[:, 1] == int(obj_id))
                            ][0]
                            assert bbox.shape in ((9, ), (10, ))
                            if MIN_FRAME < bbox[0] < MAX_FRAME:  # TODO
                                # the min/max frame is not included in `gt.txt`
                                f.write(','.join(list(map(str, bbox))) + '\n')


if __name__ == '__main__':
    print(
        '========== Testing (Text-Guided {}) =========='
            .format('ON' if opt.kum_mode else 'OFF')
    )
    output_path = join(opt.save_root, opt.exp_name, f'results{opt.save_postfix}.json')

    if not exists(output_path):
        model = get_model(opt, 'Model')
        # noinspection PyBroadException
        try:
            model, _ = load_from_ckpt(model, join(opt.save_root, f'{opt.test_ckpt}'))
        except:
            print('The model is not loaded.')
        dataloader = get_dataloader('test', opt, 'Track_Dataset')
        output = test_tracking(model, dataloader)
        os.makedirs(join(opt.save_root, opt.exp_name), exist_ok=True)
        json.dump(
            output,
            open(output_path, 'w')
        )

    SAVE_DIR = join(opt.save_root, opt.exp_name, f'results{opt.save_postfix}')
    CLS_DICT = json.load(open(output_path))

    if opt.similarity_calibration:
        TEXT_FEAT_DICT = json.load(open(join(opt.save_root, 'textual_features.json')))
        CLS_DICT = similarity_calibration(
            TEXT_FEAT_DICT,
            CLS_DICT,
            a=8,
            b=-0.1,
            tau=100
        )

    generate_final_results(
        cls_dict=CLS_DICT,
        data_dir=opt.data_root,
        track_dir=opt.track_root,
        save_dir=SAVE_DIR,
    )