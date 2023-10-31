import os
import json
import argparse
from os.path import join


class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic settings
        self.parser.add_argument('--gpus', type=str, default='0,1')
        self.parser.add_argument('--seed', type=int, default=1000)
        self.parser.add_argument('--exp_name', type=str, default='iKUN')
        self.parser.add_argument('--save_root', type=str, default='/data1/dyh/results/RMOT/Git')
        self.parser.add_argument('--save_postfix', type=str, default='')

        # basic parameters
        self.parser.add_argument('--img_hw', nargs='+', type=tuple,
                                 default=[(224, 224), (448, 448), (672, 672)])
        self.parser.add_argument('--norm_mean', type=list, default=[0.48145466, 0.4578275, 0.40821073])
        self.parser.add_argument('--norm_std', type=list, default=[0.26862954, 0.26130258, 0.27577711])
        self.parser.add_argument('--num_workers', type=int, default=4)

        # model
        self.parser.add_argument('--clip_model', type=str, default='RN50')
        self.parser.add_argument('--feature_dim', type=int, default=1024)
        self.parser.add_argument('--truncation', type=int, default=10)
        self.parser.add_argument('--tg_epoch', type=int, default=0)
        self.parser.add_argument('--kum_mode', type=str, default=None)

        # dataset
        self.parser.add_argument('--sample_frame_len', type=int, default=8)
        self.parser.add_argument('--sample_frame_num', type=int, default=2)
        self.parser.add_argument('--sample_frame_stride', type=int, default=4)
        self.parser.add_argument('--sample_expression_num', type=int, default=1)

        # train
        self.parser.add_argument('--train_bs', type=int, default=8)
        self.parser.add_argument('--cosine_end_lr', type=float, default=0.)
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--warmup_epoch', type=int, default=0)
        self.parser.add_argument('--warmup_start_lr', type=float, default=1e-5)
        self.parser.add_argument('--random_crop_ratio', nargs='+', type=float, default=[0.8, 1.0])
        self.parser.add_argument('--base_lr', type=float, default=1e-5)
        self.parser.add_argument('--max_epoch', type=int, default=100)
        self.parser.add_argument('--train_print_freq', type=int, default=40)
        self.parser.add_argument('--eval_frequency', type=int, default=10)
        self.parser.add_argument('--save_frequency', type=int, default=100)

        self.parser.add_argument('--loss_rho', type=float, default=None)
        self.parser.add_argument('--loss_gamma', type=float, default=2.)
        self.parser.add_argument('--loss_reduction', type=str, default='sum')

        # resume
        self.parser.add_argument('--resume_path', type=str, default='')

        # test
        self.parser.add_argument('--test_bs', type=int, default=1)
        self.parser.add_argument('--test_ckpt', type=str, default='iKUN.pth')
        self.parser.add_argument('--similarity_calibration', action='store_true', default=False)


    def parse(self, args=''):
        if args:
            opt = self.parser.parse_args(args)
        else:
            opt = self.parser.parse_args()
        opt.save_dir = join(opt.save_root, opt.exp_name)
        opt.data_root = join(opt.save_root, 'Refer-KITTI')
        opt.track_root = join(opt.save_root, 'NeuralSORT')
        return opt


opt = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
