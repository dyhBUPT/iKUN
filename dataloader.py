import os
import json
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from utils import *
from opts import opt


def get_dataloader(mode, opt, dataset='RMOT_Dataset', show=False, **kwargs):
    dataset = eval(dataset)(mode, opt, **kwargs)
    if show:
        dataset.show_information()
    if mode == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=opt.train_bs,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers,
        )
    elif mode == 'test':
        dataloader = DataLoader(
            dataset,
            batch_size=opt.test_bs,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
        )
    return dataloader


def get_transform(mode, opt, idx):
    if mode == 'train':
        return T.Compose([
            SquarePad(),
            T.RandomResizedCrop(
                opt.img_hw[idx],
                ratio=opt.random_crop_ratio
            ),
            T.ToTensor(),
            T.Normalize(opt.norm_mean, opt.norm_std),
        ])
    elif mode == 'test':
        return T.Compose([
            SquarePad(),
            T.Resize(opt.img_hw[idx]),
            T.ToTensor(),
            T.Normalize(opt.norm_mean, opt.norm_std),
        ])
    elif mode == 'unnorm':
        mean = opt.norm_mean
        std = opt.norm_std
        return T.Normalize(
            [-mean[i]/std[i] for i in range(3)],
            [1/std[i] for i in range(3)],
        )


def filter_target_expressions(gt, target_expressions, exp_key, only_car):
    """
    给定“帧级标签”和“视频级exp"，得到帧级exps和对应labels
    """
    OUT_EXPS, OUT_LABELS = list(), list()
    GT_EXPRESSIONS = gt[exp_key]
    for tgt_exp in target_expressions:
        if only_car and ('car' not in tgt_exp):
            continue
        OUT_EXPS.append(tgt_exp)
        if tgt_exp in GT_EXPRESSIONS:
            OUT_LABELS.append(1)
        else:
            OUT_LABELS.append(0)
    return OUT_EXPS, OUT_LABELS


def filter_gt_expressions(gt_expressions, KEY=None):
    OUT_EXPS = list()
    for gt_exp in gt_expressions:
        if KEY is None:
            OUT_EXPS.append(gt_exp)
        else:
            for key in WORDS[KEY]:
                if key in gt_exp:
                    OUT_EXPS.append(gt_exp)
                    break
    return OUT_EXPS


class RMOT_Dataset(Dataset):
    """
    For the `car` + `color+direction+location` settings
    For the `car` + 'status' settings
    """
    def __init__(self, mode, opt, only_car=False):
        super().__init__()
        assert mode in ('train', 'test')
        self.opt = opt
        self.mode = mode
        self.only_car = only_car  # 选择类别
        self.transform = {idx: get_transform(mode, self.opt, idx) for idx in (0, 1, 2)}
        self.exp_key = 'expression_new'  # 经处理后的expression标签
        self.data = self._parse_data()
        self.data_keys = list(self.data.keys())
        self.exp2id = {exp: idx for idx, exp in ID2EXP.items()}

    def _parse_data(self):
        labels = json.load(open(join(self.opt.save_root, 'Refer-KITTI_labels.json')))
        data = multi_dim_dict(2, list)
        target_expressions = defaultdict(list)
        expression_dir = join(self.opt.data_root, 'expression')
        for video in VIDEOS[self.mode]:
            # load expressions
            for exp_file in os.listdir(join(expression_dir, video)):
                expression = exp_file.replace('.json', '')
                expression_new = expression_conversion(expression)
                if expression_new not in target_expressions[video]:
                    target_expressions[video].append(expression_new)
            # load data
            H, W = RESOLUTION[video]
            for obj_id, obj_label in labels[video].items():
                num = 0
                for value in obj_label.values():
                    if len(value['category']) > 0 \
                        and (
                            (self.only_car and (value['category'][0] == 'car'))
                            or (not self.only_car)
                        ):
                                num += 1
                if num <= self.opt.sample_frame_len:
                    continue
                if len(obj_label) <= self.opt.sample_frame_len:
                    continue
                obj_key = f'{video}_{obj_id}'
                pre_frame_id = -1
                curr_data = defaultdict(list)
                for frame_id, frame_label in obj_label.items():
                    # check that the `frame_id` is in order
                    frame_id = int(frame_id)
                    assert frame_id > pre_frame_id
                    pre_frame_id = frame_id
                    # get target exps
                    tgt_exps, tgt_labels = filter_target_expressions(
                        frame_label, target_expressions[video], self.exp_key, self.only_car
                    )
                    if len(tgt_exps) == 0:
                        continue
                    # load exp
                    exps = frame_label[self.exp_key]
                    exps = filter_gt_expressions(exps, None)
                    if len(exps) == 0:
                        continue
                    # load box
                    x, y, w, h = frame_label['bbox']
                    # save
                    curr_data['expression'].append(exps)
                    curr_data['target_expression'].append(tgt_exps)
                    curr_data['target_labels'].append(tgt_labels)
                    curr_data['bbox'].append([frame_id, x * W, y * H, (x + w) * W, (y + h) * H])
                if len(curr_data['bbox']) > self.opt.sample_frame_len:
                    data[obj_key] = curr_data.copy()
        return data

    def _crop_image(self, images, indices, data, mode):
        if mode == 'small':
            crops = torch.stack(
                [self.transform[0](
                    images[i].crop(data['bbox'][idx][1:])
                ) for i, idx in enumerate(indices)],
                dim=0
            )
        elif mode == 'big':
            X1, Y1, X2, Y2 = 1e5, 1e5, -1, -1
            for idx in indices:
                x1, y1, x2, y2 = data['bbox'][idx][1:]
                X1, Y1, X2, Y2 = min(X1, x1), min(Y1, y1), max(X2, x2), max(Y2, y2)
            crops = torch.stack(
                [self.transform[0](
                    image.crop([X1, Y1, X2, Y2])
                ) for image in images],
                dim=0
            )
        return crops

    def __getitem__(self, index):
        data_key = self.data_keys[index]
        video = data_key.split('_')[0]
        data = self.data[data_key]

        # sample frames
        data_len = len(data['bbox'])
        sample_len = self.opt.sample_frame_len
        sample_num = self.opt.sample_frame_num
        sampled_indices = list()
        if self.mode == 'train':
            # continuous random sampling
            start_idx = random.randint(0, data_len - sample_len)
            stop_idx = start_idx + sample_len
            # restricted random sampling
            step = sample_len // sample_num
            for idx in range(start_idx, stop_idx, step):
                sampled_indices.append(
                    random.randint(idx, idx + step - 1)
                )
        elif self.mode == 'test':
            # continuous sampling
            start_idx = index % (data_len - sample_len)
            stop_idx = start_idx + sample_len
            # restricted sampling
            step = sample_len // sample_num
            for idx in range(start_idx, stop_idx, step):
                sampled_indices.append(idx + step // 2)

        # load images
        images = [
            Image.open(
                join(
                    self.opt.data_root,
                    'KITTI/training/image_02/{}/{:0>6d}.png'
                        .format(video, data['bbox'][idx][0])
                )
            ) for idx in sampled_indices
        ]

        # load expressions
        expressions = list()
        for idx in sampled_indices:
            expressions.extend(data['expression'][idx])
        expressions = sorted(list(set(expressions)))

        # crop images
        cropped_images = self._crop_image(
            images, sampled_indices, data, 'small'
        )  # [T,C,H,W]

        # global images
        global_images = torch.stack([
            self.transform[2](image)
            for image in images
        ], dim=0)

        # sample target expressions
        if self.mode == 'train':
            idx = choice(sampled_indices, size=1)[0]
        elif self.mode == 'test':
            idx = sampled_indices[len(sampled_indices) // 2]
        target_expressions = data['target_expression'][idx]
        target_labels = data['target_labels'][idx]
        if self.mode == 'train':
            assert self.opt.sample_expression_num == 1
            sampled_target_idx = choice(
                range(len(target_expressions)),
                size=1,
                replace=False
            )
            sampled_target_exp = [
                target_expressions[i]
                for i in sampled_target_idx
            ]
            sampled_target_label = [
                target_labels[i]
                for i in sampled_target_idx
            ]
            exp_id = self.exp2id[sampled_target_exp[0]]
        elif self.mode == 'test':
            sampled_target_exp = target_expressions
            sampled_target_label = target_labels
            exp_id = -1

        sampled_target_label = torch.tensor(
            sampled_target_label,
            dtype=float
        )
        return dict(
            cropped_images=cropped_images,
            global_images=global_images,
            expressions=','.join(expressions),
            target_expressions=','.join(sampled_target_exp),
            target_labels=sampled_target_label,
            expression_id=exp_id,
            start_idx=start_idx,
            stop_idx=stop_idx,
            data_key=data_key,
        )

    def __len__(self):
        return len(self.data_keys)

    def show_information(self):
        print(
            f'===> Refer-KITTI ({self.mode}) <===\n'
            f"Number of identities: {len(self.data)}"
        )


class Track_Dataset(Dataset):
    def __init__(self, mode, opt):
        self.opt = opt
        self.mode = mode
        self.transform = {idx: get_transform(self.mode, self.opt, idx) for idx in (0, 1, 2)}
        self.data = self._parse_data()

    def _parse_data(self):
        sample_length = self.opt.sample_frame_len
        sample_stride = self.opt.sample_frame_stride
        DATA = list()
        for video in VIDEOS[self.mode]:
            # load tracks
            tracks_1 = np.loadtxt(join(self.opt.track_root, video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0
            tracks_2 = np.loadtxt(join(self.opt.track_root, video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)
            tracks = tracks[np.lexsort([tracks[:, 0], tracks[:, 1]])]  # ID->frame
            # parse tracks
            ids = set(tracks[:, 1])
            for obj_id in ids:
                tracks_id = tracks[tracks[:, 1] == obj_id]
                frame_min, frame_max = int(min(tracks_id[:, 0])), int(max(tracks_id[:, 0]))
                # 识别轨迹断点位置，从而方便对每个sub-tracklet单独处理
                frame_pairs, start_frame, stop_frame = list(), frame_min, -1
                previous_frame = start_frame - 1
                for frame_idx in list(tracks_id[:, 0]) + [1e5]:
                    if frame_idx != previous_frame + 1:
                        stop_frame = previous_frame
                        frame_pairs.append([int(start_frame), int(stop_frame)])
                        start_frame = frame_idx
                    previous_frame = frame_idx
                # 将tracklets按sample_stride划分为片段
                total_length = 0
                for f_min, f_max in frame_pairs:
                    total_length += (f_max - f_min + 1)
                    for f_idx in range(f_min, f_max + 1, sample_stride):
                        f_stop = min(f_max, f_idx + sample_length - 1)
                        f_start = max(f_min, f_stop - sample_length + 1)
                        tracklets = tracks_id[np.isin(
                            tracks_id[:, 0],
                            range(f_start, f_stop + 1)
                        )][:, :6]
                        tracklets[:, 4:6] += tracklets[:, 2:4]
                        tracklets = tracklets.astype(int)
                        assert (f_stop - f_start + 1) == len(tracklets)
                        for expression in EXPRESSIONS[video]:
                            DATA.append(dict(
                                video=video,
                                obj_id=int(obj_id),
                                start_frame=f_start,
                                stop_frame=f_stop,
                                tracklets=tracklets,
                                expression=expression,
                            ))
                        if f_stop == f_max:
                            break
                assert total_length == len(tracks_id)
        return DATA

    def __getitem__(self, index):
        video, obj_id, start_frame, stop_frame, tracklets, expression = self.data[index].values()
        assert (stop_frame - start_frame + 1) == len(tracklets)

        # expression conversion
        expression_converted = expression_conversion(expression)

        # frame sampling
        sampled_indices = np.linspace(
            0, len(tracklets),
            self.opt.sample_frame_num,
            endpoint=False, dtype=int
        )
        sampled_tracklets = tracklets[sampled_indices]

        # load images
        images = [
            Image.open(
                join(
                    self.opt.data_root,
                    'KITTI/training/image_02/{}/{:0>6d}.png'
                        .format(video, bbox[0])
                )
            ) for bbox in sampled_tracklets
        ]

        # crop images
        cropped_images = torch.stack(
            [self.transform[0](
                images[i].crop(bbox[2:6])
            ) for i, bbox in enumerate(sampled_tracklets)],
            dim=0
        )

        # global images
        global_images = torch.stack([
            self.transform[2](image)
            for image in images
        ], dim=0)

        return dict(
            video=video,
            obj_id=obj_id,
            start_frame=start_frame,
            stop_frame=stop_frame,
            cropped_images=cropped_images,
            global_images=global_images,
            expression_raw=expression,
            expression_new=expression_converted,
        )

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = RMOT_Dataset('train', opt)
    print(dataset.exp2id)