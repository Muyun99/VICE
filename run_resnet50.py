import argparse
import os
import time

from dywsss.tool import pyutils
from mmcv import Config, DictAction
from dywsss.pipeline.train_cam_resnet50 import train
from dywsss.pipeline.infer_multi_scale_resnet50 import infer_multi_scale
from dywsss.tool.torch_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a models')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--tag', help='the tag')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--weights', type=str, default='best.pth')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    args = cfg

    set_seed(cfg)

    # Output Path
    # time_suffix = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # cfg.session_name = cfg.session_name + f"_{time_suffix}"
    args.model_dir = os.path.join('work_dirs', args.session_name, "model")
    args.test_dir = os.path.join('work_dirs', args.session_name, "test")
    args.log_dir = os.path.join('work_dirs', args.session_name, "log")
    args.tensorboard_dir = os.path.join('work_dirs', args.session_name, "tensorboard")

    os.makedirs("work_dirs", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    print(vars(args))

    if args.train_multi_scale:

        timer = pyutils.Timer('train in multi-scale strategy:')
        train(args)

    if args.gen_mask_for_multi_crop:
        timer = pyutils.Timer('infer multi_scale cam and make rough mask:')

        args.weights = os.path.join(args.model_dir, args.weights)
        args.infer_list = os.path.join('voc12', args.infer_list)
        args.out_cam = os.path.join(args.test_dir,
                                    f'cam_{args.infer_list.split("/")[-1].split(".")[0]}_{args.weights.split("/")[-1].split(".")[0]}')
        args.out_crf = os.path.join(args.test_dir, 'train_mask')
        infer_multi_scale(args)
