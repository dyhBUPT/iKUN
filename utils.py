import os
import math
import json
import random
import logging
import numpy as np
from os.path import join
from collections import defaultdict
import torchvision.transforms.functional as F

import clip
import torch


__all__ = [
    'VIDEOS', 'RESOLUTION', 'WORDS', 'EXPRESSIONS', 'FRAMES',
    'multi_dim_dict', 'expression_conversion', 'SquarePad',
    'save_configs', 'get_logger', 'get_lr', 'set_lr',
    'AverageMeter', 'ProgressMeter',
    'tokenize', 'load_from_ckpt', 'set_seed',
    'ID2EXP', 'EXP_TO_POS_BBOX_NUMS', 'ID_TO_POS_BBOX_NUMS'
]


VIDEOS = {
    'train': [
        '0001', '0002', '0003', '0004', '0006',
        '0007', '0008', '0009', '0010', '0012',
        '0014', '0015', '0016', '0018', '0020',
    ],
    'val': [
        '0005', '0011', '0013'
    ],
    'test': [
        '0005', '0011', '0013'
    ],
}


RESOLUTION = {
    '0001': (375, 1242), '0002': (375, 1242), '0003': (375, 1242), '0004': (375, 1242),
    '0005': (375, 1242), '0006': (375, 1242), '0007': (375, 1242), '0008': (375, 1242),
    '0009': (375, 1242), '0010': (375, 1242), '0011': (375, 1242), '0012': (375, 1242),
    '0013': (375, 1242), '0014': (370, 1224), '0015': (370, 1224), '0016': (370, 1224),
    '0018': (374, 1238), '0020': (376, 1241),
}


WORDS_MAPPING = {
    'cars': 'car',
    'vehicles': 'car',
    'people': 'pedestrian',
    'persons': 'pedestrian',
    'males': 'men',
    'females': 'women',
    'light-colors': 'light-color',
}


WORDS = {
    'dropped': [
        'in', 'the', 'with', 'direction', 'of',
        'ours', 'camera', 'and', 'which', 'are',
        'than', 'carrying', 'holding', 'a', 'bag',
        'pants', 'who', 'horizon',
    ],
    'category': ['car', 'pedestrian', 'men', 'women'],
    'color': ['black', 'red', 'silver', 'light-color', 'white'],
    'location': ['right', 'left', 'front'],
    'direction': ['same', 'counter'],
    'status': ['moving', 'turning', 'parking', 'braking', 'walking', 'standing'],
    'speed': ['faster', 'slower'],
}


EXPRESSIONS = {
    'train': [
        'black-cars', 'black-cars-in-right', 'black-cars-in-the-left', 'black-cars-with-the-counter-direction-of-ours',
        'black-moving-cars', 'black-moving-vehicles', 'black-vehicles', 'black-vehicles-in-right',
        'black-vehicles-in-the-left', 'black-vehicles-with-the-counter-direction-of-ours', 'cars', 'cars-in-black',
        'cars-in-front-of-ours', 'cars-in-front-of-the-camera', 'cars-in-horizon-direction', 'cars-in-left',
        'cars-in-light-color', 'cars-in-red', 'cars-in-right', 'cars-in-silver', 'cars-in-the-counter-direction',
        'cars-in-the-counter-direction-of-ours', 'cars-in-the-left', 'cars-in-the-same-direction-of-ours',
        'cars-which-are-braking', 'cars-which-are-in-the-left-and-turning', 'cars-which-are-parking',
        'cars-which-are-slower-than-ours', 'cars-which-are-turning', 'cars-with-the-counter-direction-of-ours',
        'cars-with-the-same-direction-of-ours', 'counter-direction-cars-in-the-left',
        'counter-direction-cars-in-the-right', 'counter-direction-vehicles-in-the-left',
        'counter-direction-vehicles-in-the-right', 'left-cars', 'left-cars-in-black', 'left-cars-in-light-color',
        'left-cars-in-red', 'left-cars-in-silver', 'left-cars-in-the-counter-direction-of-ours',
        'left-cars-in-the-same-direction-of-ours', 'left-cars-which-are-black', 'left-cars-which-are-in-light-colors',
        'left-cars-which-are-parking', 'left-moving-cars', 'left-moving-vehicles', 'left-vehicles',
        'left-vehicles-in-black', 'left-vehicles-in-light-color', 'left-vehicles-in-red', 'left-vehicles-in-silver',
        'left-vehicles-in-the-counter-direction-of-ours', 'left-vehicles-in-the-same-direction-of-ours',
        'left-vehicles-which-are-black', 'left-vehicles-which-are-in-light-colors', 'left-vehicles-which-are-parking',
        'light-color-cars', 'light-color-cars-in-the-left', 'light-color-cars-in-the-right',
        'light-color-cars-which-are-parking', 'light-color-cars-with-the-counter-direction-of-ours',
        'light-color-moving-cars', 'light-color-moving-vehicles', 'light-color-vehicles',
        'light-color-vehicles-in-the-left', 'light-color-vehicles-in-the-right',
        'light-color-vehicles-which-are-parking', 'light-color-vehicles-with-the-counter-direction-of-ours',
        'males-in-the-left', 'males-in-the-right', 'men-in-the-left', 'men-in-the-right', 'moving-black-cars',
        'moving-black-vehicles', 'moving-cars', 'moving-cars-in-the-same-direction-of-ours', 'moving-turning-cars',
        'moving-turning-vehicles', 'moving-vehicles', 'moving-vehicles-in-the-same-direction-of-ours', 'parking-cars',
        'parking-cars-in-the-left', 'parking-cars-in-the-right', 'parking-vehicles', 'parking-vehicles-in-the-left',
        'parking-vehicles-in-the-right', 'pedestrian', 'pedestrian-in-the-left', 'pedestrian-in-the-pants',
        'pedestrian-in-the-right', 'people', 'people-in-the-left', 'people-in-the-pants', 'people-in-the-right',
        'persons', 'persons-in-the-left', 'persons-in-the-pants', 'persons-in-the-right', 'red-cars-in-the-left',
        'red-cars-in-the-right', 'red-moving-cars', 'red-moving-vehicles', 'red-turning-cars', 'red-turning-vehicles',
        'red-vehicles-in-the-left', 'red-vehicles-in-the-right', 'right-cars-in-black', 'right-cars-in-light-color',
        'right-cars-in-red', 'right-cars-in-silver', 'right-cars-in-the-counter-direction-of-ours',
        'right-cars-in-the-same-direction-of-ours', 'right-cars-which-are-black',
        'right-cars-which-are-in-light-colors', 'right-cars-which-are-parking', 'right-moving-cars',
        'right-moving-vehicles', 'right-vehicles-in-black', 'right-vehicles-in-light-color', 'right-vehicles-in-red',
        'right-vehicles-in-silver', 'right-vehicles-in-the-counter-direction-of-ours',
        'right-vehicles-in-the-same-direction-of-ours', 'right-vehicles-which-are-black',
        'right-vehicles-which-are-in-light-colors', 'right-vehicles-which-are-parking',
        'same-direction-cars-in-the-left', 'same-direction-cars-in-the-right', 'same-direction-vehicles-in-the-left',
        'same-direction-vehicles-in-the-right', 'silver-cars-in-right', 'silver-cars-in-the-left',
        'silver-turning-cars', 'silver-turning-vehicles', 'silver-vehicles-in-right', 'silver-vehicles-in-the-left',
        'turning-cars', 'turning-vehicles', 'vehicles', 'vehicles-in-black', 'vehicles-in-front-of-ours',
        'vehicles-in-front-of-the-camera', 'vehicles-in-horizon-direction', 'vehicles-in-left',
        'vehicles-in-light-color', 'vehicles-in-red', 'vehicles-in-right', 'vehicles-in-silver',
        'vehicles-in-the-counter-direction', 'vehicles-in-the-counter-direction-of-ours', 'vehicles-in-the-left',
        'vehicles-in-the-same-direction-of-ours', 'vehicles-which-are-braking',
        'vehicles-which-are-in-the-left-and-turning', 'vehicles-which-are-parking',
        'vehicles-which-are-slower-than-ours', 'vehicles-which-are-turning',
        'vehicles-with-the-counter-direction-of-ours', 'vehicles-with-the-same-direction-of-ours',
        'walking-pedestrian-in-the-left', 'walking-pedestrian-in-the-right', 'walking-people-in-the-left',
        'walking-people-in-the-right', 'walking-persons-in-the-left', 'walking-persons-in-the-right'],
    'test': [
        'black-cars-in-right', 'black-cars-in-the-left', 'black-vehicles-in-right', 'black-vehicles-in-the-left',
        'cars-in-black', 'cars-in-front-of-ours', 'cars-in-left', 'cars-in-light-color', 'cars-in-right',
        'cars-in-silver', 'cars-in-the-counter-direction-of-ours', 'cars-in-the-left', 'cars-in-the-right',
        'cars-in-the-same-direction-of-ours', 'cars-in-white', 'cars-which-are-faster-than-ours',
        'counter-direction-cars-in-the-left', 'counter-direction-vehicles-in-the-left', 'females',
        'females-in-the-left', 'females-in-the-right', 'left-cars-in-black', 'left-cars-in-light-color',
        'left-cars-in-silver', 'left-cars-in-the-counter-direction-of-ours',
        'left-cars-in-the-same-direction-of-ours', 'left-cars-in-white', 'left-cars-which-are-parking',
        'left-pedestrian-who-are-walking', 'left-people-who-are-walking', 'left-persons-who-are-walking',
        'left-vehicles-in-black', 'left-vehicles-in-light-color', 'left-vehicles-in-silver',
        'left-vehicles-in-the-counter-direction-of-ours', 'left-vehicles-in-the-same-direction-of-ours',
        'left-vehicles-in-white', 'left-vehicles-which-are-parking', 'light-color-cars-in-the-left',
        'light-color-cars-in-the-right', 'light-color-vehicles-in-the-left', 'light-color-vehicles-in-the-right',
        'males', 'males-in-the-left', 'males-in-the-right', 'men', 'men-in-the-left', 'men-in-the-right',
        'moving-cars', 'moving-left-pedestrian', 'moving-pedestrian', 'moving-right-pedestrian', 'moving-vehicles',
        'parking-cars', 'parking-vehicles', 'pedestrian', 'pedestrian-in-the-left', 'pedestrian-in-the-right',
        'pedestrian-who-are-walking', 'people', 'people-in-the-left', 'people-in-the-right',
        'people-who-are-walking', 'persons', 'persons-in-the-left', 'persons-in-the-right',
        'persons-who-are-walking', 'right-cars-in-black', 'right-cars-in-light-color', 'right-cars-in-silver',
        'right-cars-in-white', 'right-cars-which-are-parking', 'right-pedestrian-who-are-walking',
        'right-people-who-are-walking', 'right-persons-who-are-walking', 'right-vehicles-in-black',
        'right-vehicles-in-light-color', 'right-vehicles-in-silver', 'right-vehicles-in-white',
        'right-vehicles-which-are-parking', 'same-direction-cars-in-the-left',
        'same-direction-vehicles-in-the-left', 'silver-cars-in-right', 'silver-cars-in-the-left',
        'silver-vehicles-in-right', 'silver-vehicles-in-the-left', 'standing-females', 'standing-males',
        'standing-men', 'standing-women', 'turning-cars', 'turning-vehicles', 'vehicles-in-black',
        'vehicles-in-front-of-ours', 'vehicles-in-left', 'vehicles-in-light-color', 'vehicles-in-right',
        'vehicles-in-silver', 'vehicles-in-the-counter-direction-of-ours', 'vehicles-in-the-left',
        'vehicles-in-the-right', 'vehicles-in-the-same-direction-of-ours', 'vehicles-in-white',
        'vehicles-which-are-faster-than-ours', 'walking-females', 'walking-males', 'walking-men',
        'walking-pedestrian', 'walking-women', 'white-cars-in-the-left', 'white-cars-in-the-right',
        'white-vehicles-in-the-left', 'white-vehicles-in-the-right', 'women', 'women-carrying-a-bag',
        'women-holding-a-bag', 'women-in-the-left', 'women-in-the-right'
    ],
    'dropped': [
        'women-back-to-the-camera', 'vehicles-which-are-braking', 'men-back-to-the-camera',
        'vehicles-in-horizon-direction', 'cars-which-are-braking', 'cars-in-horizon-direction',
        'males-back-to-the-camera', 'females-back-to-the-camera',
    ],  # these expressions are not evaluated as in TransRMOT
    '0005': [
        'left-cars-in-silver', 'same-direction-cars-in-the-left', 'counter-direction-cars-in-the-left',
        'left-vehicles-in-the-counter-direction-of-ours', 'silver-vehicles-in-the-left',
        'vehicles-in-front-of-ours', 'right-cars-in-light-color', 'counter-direction-vehicles-in-the-left',
        'same-direction-vehicles-in-the-left', 'light-color-cars-in-the-right', 'light-color-vehicles-in-the-right',
        'left-cars-in-the-counter-direction-of-ours', 'left-vehicles-in-light-color', 'vehicles-in-left',
        'left-vehicles-in-black', 'left-cars-in-black', 'cars-in-the-same-direction-of-ours',
        'left-cars-which-are-parking', 'cars-in-front-of-ours', 'cars-in-light-color', 'moving-vehicles',
        'vehicles-in-the-counter-direction-of-ours', 'cars-which-are-braking', 'left-vehicles-which-are-parking',
        'vehicles-which-are-braking', 'silver-vehicles-in-right', 'vehicles-in-right', 'vehicles-in-silver',
        'left-cars-in-light-color', 'left-vehicles-in-the-same-direction-of-ours', 'vehicles-in-black',
        'black-cars-in-the-left', 'right-cars-in-silver', 'black-vehicles-in-the-left', 'right-vehicles-in-silver',
        'cars-in-left', 'left-cars-in-the-same-direction-of-ours', 'right-vehicles-in-light-color', 'cars-in-black',
        'cars-in-silver', 'moving-cars', 'cars-in-the-counter-direction-of-ours',
        'vehicles-in-the-same-direction-of-ours', 'vehicles-in-light-color', 'cars-in-right',
        'silver-cars-in-right', 'light-color-cars-in-the-left', 'silver-cars-in-the-left',
        'left-vehicles-in-silver', 'light-color-vehicles-in-the-left'
    ],
    '0011': [
        'right-cars-in-black', 'black-vehicles-in-right', 'parking-vehicles', 'left-cars-in-white',
        'white-cars-in-the-right', 'counter-direction-cars-in-the-left',
        'left-vehicles-in-the-counter-direction-of-ours', 'right-cars-in-light-color',
        'counter-direction-vehicles-in-the-left', 'right-cars-which-are-parking',
        'light-color-cars-in-the-right', 'light-color-vehicles-in-the-right',
        'left-cars-in-the-counter-direction-of-ours', 'left-vehicles-in-light-color', 'pedestrian',
        'vehicles-in-left', 'white-vehicles-in-the-left', 'left-vehicles-in-black', 'moving-pedestrian',
        'vehicles-which-are-faster-than-ours', 'left-cars-in-black', 'cars-in-the-same-direction-of-ours',
        'persons-who-are-walking', 'left-cars-which-are-parking', 'white-vehicles-in-the-right',
        'turning-vehicles', 'parking-cars', 'left-vehicles-in-white', 'cars-in-light-color',
        'moving-vehicles', 'vehicles-in-the-counter-direction-of-ours', 'cars-in-horizon-direction',
        'left-vehicles-which-are-parking', 'black-cars-in-right', 'people-who-are-walking',
        'walking-pedestrian', 'vehicles-in-right', 'vehicles-in-white', 'turning-cars', 'right-cars-in-white',
        'right-vehicles-in-black', 'left-cars-in-light-color', 'vehicles-in-black', 'black-cars-in-the-left',
        'black-vehicles-in-the-left', 'right-vehicles-which-are-parking', 'cars-in-left',
        'vehicles-in-horizon-direction', 'right-vehicles-in-light-color', 'cars-which-are-faster-than-ours',
        'cars-in-black', 'people', 'cars-in-white', 'moving-cars', 'pedestrian-who-are-walking', 'persons',
        'cars-in-the-counter-direction-of-ours', 'vehicles-in-the-same-direction-of-ours',
        'vehicles-in-light-color', 'cars-in-right', 'right-vehicles-in-white', 'light-color-cars-in-the-left',
        'white-cars-in-the-left', 'light-color-vehicles-in-the-left'],
    '0013': [
        'walking-males', 'women-back-to-the-camera', 'walking-women', 'women', 'females',
        'left-people-who-are-walking', 'men-back-to-the-camera', 'right-people-who-are-walking', 'males',
        'people-in-the-left', 'persons-in-the-right', 'people-in-the-right', 'left-persons-who-are-walking',
        'females-back-to-the-camera', 'standing-males', 'men-in-the-left', 'women-carrying-a-bag',
        'women-in-the-left', 'men-in-the-right', 'standing-men', 'standing-women', 'vehicles-in-the-right',
        'moving-left-pedestrian', 'women-in-the-right', 'right-persons-who-are-walking', 'cars-in-the-right',
        'men', 'persons-in-the-left', 'males-in-the-left', 'pedestrian-in-the-left',
        'right-pedestrian-who-are-walking', 'women-holding-a-bag', 'males-back-to-the-camera',
        'vehicles-in-the-left', 'walking-men', 'standing-females', 'left-pedestrian-who-are-walking',
        'females-in-the-left', 'pedestrian-in-the-right', 'cars-in-the-left', 'moving-right-pedestrian',
        'walking-females', 'females-in-the-right', 'males-in-the-right'
    ],
}


ID2EXP = {
      0: 'left car which are parking',
      1: 'car in right',
      2: 'right car which are black',
      3: 'left car which are in light-color',
      4: 'men in the left',
      5: 'right car in light-color',
      6: 'right moving car',
      7: 'left car in black',
      8: 'car in front of the camera',
      9: 'right car which are in light-color',
      10: 'counter direction car in the right',
      11: 'same direction car in the right',
      12: 'pedestrian',
      13: 'counter direction car in the left',
      14: 'left moving car',
      15: 'black moving car',
      16: 'red moving car',
      17: 'light-color car in the left',
      18: 'light-color car with the counter direction of ours',
      19: 'silver car in right',
      20: 'left car in red',
      21: 'car in the same direction of ours',
      22: 'same direction car in the left',
      23: 'moving car in the same direction of ours',
      24: 'men in the right',
      25: 'right car in black',
      26: 'car in silver',
      27: 'black car',
      28: 'silver turning car',
      29: 'moving black car',
      30: 'left car in the same direction of ours',
      31: 'car with the counter direction of ours',
      32: 'car in horizon direction',
      33: 'black car in the left',
      34: 'parking car',
      35: 'car which are braking',
      36: 'right car in the counter direction of ours',
      37: 'red car in the right',
      38: 'right car which are parking',
      39: 'walking pedestrian in the right',
      40: 'moving turning car',
      41: 'left car',
      42: 'car with the same direction of ours',
      43: 'left car in silver',
      44: 'light-color car in the right',
      45: 'car in the counter direction',
      46: 'car in the counter direction of ours',
      47: 'silver car in the left',
      48: 'walking pedestrian in the left',
      49: 'black car with the counter direction of ours',
      50: 'car in light-color',
      51: 'left car in light-color',
      52: 'pedestrian in the right',
      53: 'car in front of ours',
      54: 'red turning car',
      55: 'left car in the counter direction of ours',
      56: 'car which are in the left and turning',
      57: 'car in the left',
      58: 'right car in silver',
      59: 'car in left',
      60: 'turning car',
      61: 'light-color car',
      62: 'light-color moving car',
      63: 'moving car',
      64: 'car which are turning',
      65: 'car',
      66: 'car in red',
      67: 'car in black',
      68: 'car which are parking',
      69: 'right car in the same direction of ours',
      70: 'parking car in the left',
      71: 'parking car in the right',
      72: 'car which are slower than ours',
      73: 'pedestrian in the pants',
      74: 'light-color car which are parking',
      75: 'black car in right',
      76: 'right car in red',
      77: 'pedestrian in the left',
      78: 'left car which are black',
      79: 'red car in the left'
 }


EXP_TO_POS_BBOX_NUMS = {
     'black car': 114,
     'black car in right': 1347,
     'black car in the left': 3758,
     'black car with the counter direction of ours': 104,
     'black moving car': 114,
     'car': 1456,
     'car in black': 6358,
     'car in front of ours': 1831,
     'car in front of the camera': 66,
     'car in horizon direction': 897,
     'car in left': 9304,
     'car in light-color': 6489,
     'car in red': 478,
     'car in right': 4877,
     'car in silver': 1841,
     'car in the counter direction': 72,
     'car in the counter direction of ours': 4843,
     'car in the left': 834,
     'car in the same direction of ours': 10230,
     'car which are braking': 839,
     'car which are in the left and turning': 52,
     'car which are parking': 647,
     'car which are slower than ours': 4138,
     'car which are turning': 289,
     'car with the counter direction of ours': 130,
     'car with the same direction of ours': 217,
     'counter direction car in the left': 4087,
     'counter direction car in the right': 368,
     'left car': 72,
     'left car in black': 3706,
     'left car in light-color': 3053,
     'left car in red': 240,
     'left car in silver': 817,
     'left car in the counter direction of ours': 4087,
     'left car in the same direction of ours': 4790,
     'left car which are black': 112,
     'left car which are in light-color': 112,
     'left car which are parking': 3987,
     'left moving car': 165,
     'light-color car': 223,
     'light-color car in the left': 3053,
     'light-color car in the right': 1464,
     'light-color car which are parking': 589,
     'light-color car with the counter direction of ours': 186,
     'light-color moving car': 223,
     'men in the left': 1827,
     'men in the right': 0,
     'moving black car': 217,
     'moving car': 8368,
     'moving car in the same direction of ours': 358,
     'moving turning car': 104,
     'parking car': 7877,
     'parking car in the left': 557,
     'parking car in the right': 61,
     'pedestrian': 115,
     'pedestrian in the left': 2419,
     'pedestrian in the pants': 1984,
     'pedestrian in the right': 253,
     'red car in the left': 240,
     'red car in the right': 187,
     'red moving car': 223,
     'red turning car': 223,
     'right car in black': 1347,
     'right car in light-color': 1464,
     'right car in red': 187,
     'right car in silver': 227,
     'right car in the counter direction of ours': 368,
     'right car in the same direction of ours': 2632,
     'right car which are black': 68,
     'right car which are in light-color': 68,
     'right car which are parking': 4089,
     'right moving car': 526,
     'same direction car in the left': 4790,
     'same direction car in the right': 2632,
     'silver car in right': 227,
     'silver car in the left': 817,
     'silver turning car': 66,
     'turning car': 388,
     'walking pedestrian in the left': 0,
     'walking pedestrian in the right': 0
}


ID_TO_POS_BBOX_NUMS = {
    idx: EXP_TO_POS_BBOX_NUMS[exp] for idx, exp in ID2EXP.items()
}


FRAMES = {
    '0005': (0, 296),
    '0011': (0, 372),
    '0013': (0, 339),
}  # 视频起止帧


def set_seed(seed):
    """
    ref: https://blog.csdn.net/weixin_44791964/article/details/131622957
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_dim_dict(n, types):
    if n == 0:
        return types()
    else:
        return defaultdict(lambda: multi_dim_dict(n-1, types))


def expression_conversion(expression):
    """expression => expression_new"""
    expression = expression.replace('-', ' ').replace('light color', 'light-color')
    words = expression.split(' ')
    expression_converted = ''
    for word in words:
        if word in WORDS_MAPPING:
            word = WORDS_MAPPING[word]
        expression_converted += f'{word} '
    expression_converted = expression_converted[:-1]
    return expression_converted


class SquarePad:
    """Reference:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


def save_configs(opt):
    configs = vars(opt)
    os.makedirs(opt.save_dir, exist_ok=True)
    json.dump(
        configs,
        open(join(opt.save_dir, 'config.json'), 'w'),
        indent=2
    )


def get_logger(save_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filename = join(save_dir, 'log.txt')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] %(message)s')
    # writting to file
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # display in terminal
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)
    return logger


def get_lr(opt, curr_epoch):
    if curr_epoch < opt.warmup_epoch:
        return (
            opt.warmup_start_lr
            + (opt.base_lr - opt.warmup_start_lr)
            * curr_epoch
            / opt.warmup_epoch
        )
    else:
        return (
            opt.cosine_end_lr
            + (opt.base_lr - opt.cosine_end_lr)
            * (
                math.cos(
                    math.pi * (curr_epoch - opt.warmup_epoch) / (opt.max_epoch - opt.warmup_epoch)
                )
                + 1.0
            )
            * 0.5
        )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", lr=0.):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches, lr)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches, lr):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + '] [lr: {:.2e}]'.format(lr)


def tokenize(text):
    token = clip.tokenize(text)
    return token


def load_from_ckpt(model, ckpt_path, model_name='model'):
    print(f'load from {ckpt_path}...')
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt[model_name])
    return model, epoch

