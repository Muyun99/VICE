from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from tqdm import tqdm
import argparse
from mmcv import Config, DictAction
import os.path

import mmcv
import numpy as np
import torch

from dywsss.tool import pyutils, imutils
import dywsss.tool.data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from dywsss.tool.torch_utils import *
import timm
from ml_metric import Accuracy, F1Measure, F1Measure_sklearn, ECE_loss
cudnn.enabled = True

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def average_performance(pred, target, thr=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.
    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thr is None and k is None:
        thr = 0.5
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thr is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = pred >= thr

    else:
        # top-k labels will be predicted positive for any example
        sort_inds = np.argsort(-pred, axis=1)
        sort_inds_ = sort_inds[:, :k]
        inds = np.indices(sort_inds_.shape)
        pos_inds = np.zeros_like(pred)
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    return CP, CR, CF1, OP, OR, OF1


def average_precision(pred, target):
    r"""Calculate the average precision for a single class.
    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:
    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n
    Note that no approximation is involved since the curve is piecewise
    constant.
    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).
    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap

# from mmcls


def mAP(pred, target):
    """Calculate the mean average precision with respect of classes.
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    # exist_classes = range(num_classes)

    label_exist = np.where(pred > 0.5, 1, 0).sum(0) + target.sum(0)

    # exist_classes = np.argwhere(label_exist > 0)[0]
    exist_classes = (label_exist > 0).nonzero()[0].tolist()
    # print(f'num_exist_classes is {len(exist_classes)}, exist_classes is {exist_classes}')

    # exist_classes = pass

    for k in exist_classes:
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.
    return mean_ap


def Average_Precision(pred, target):
    N = len(target)
    for i in range(N):
        if max(target[i]) == 0 or min(target[i]) == 1:
            pass
    precision = 0
    for i in range(N):
        index = np.where(target[i] == 1)[0]
        score = pred[i][index]
        score = sorted(score)
        score_all = sorted(pred[i])
        precision_tmp = 0
        for item in score:
            tmp1 = score.index(item)
            tmp1 = len(score) - tmp1
            tmp2 = score_all.index(item)
            tmp2 = len(score_all) - tmp2
            precision_tmp += tmp1 / tmp2
        precision += precision_tmp / len(score)
    Average_Precision = precision / N
    return Average_Precision


def mean_avg_precision(pred, target):
    meanAP = metrics.average_precision_score(
        target, pred, average='macro', pos_label=1)
    return meanAP


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


def analyse_bin_10(prediction_batch, labels_batch):
    debug = 1
    for idx in range(labels_batch.shape[0]):
        y = np.argsort(prediction_batch[idx])[-1]
        x = np.sort(prediction_batch[idx])[-1]
        label = (labels_batch[idx] == 1).nonzero(as_tuple=True)[0]


# Multi-label ECE
def compute_Conf_ECE(args, dir_score, num_bin, metric_func, note):
    df_score = pd.read_csv(dir_score)

    # sort by confidence
    df_score.sort_values(by=['confidence'], inplace=True)
    conf_score = df_score['confidence'].to_list()

    prediction_all = torch.randn(0)
    labels_all = torch.randn(0)

    perfermance_bins = []
    count_bins = []
    images_list_bins = []

    # mAP_function = mean_avg_precision

    # mAP_function = Average_Precision

    model = timm.create_model(args.network, pretrained=True, num_classes=20)
    model.load_state_dict(torch.load(args.weights))
    print(f'Loading weights from {args.weights}')
    print('\nvalidating ... ', flush=True, end='')

    # devide miou_score to num_bin bins
    each_bin = 1/num_bin
    for bin in tqdm(range(num_bin)):

        bin_lower = bin * each_bin
        bin_upper = (bin + 1) * each_bin
        # find the name_images in bin
        idx_in_bin = (np.array(conf_score) <= bin_upper) * \
            (np.array(conf_score) >= bin_lower)
        img_name_list_in_bin = np.array(
            df_score['name_image'].tolist())[idx_in_bin]
        mIoU_in_bin = np.array(df_score['confidence'].tolist())[idx_in_bin]

        images_list_bins.append(img_name_list_in_bin)
        count_bins.append(len(img_name_list_in_bin))

        # print(f'In [{bin_lower, bin_upper}] bin have {len(img_name_list_in_bin)} samples')
        normalize = Normalize()
        val_dataset = dywsss.tool.data.VOC12ClsDataset(
            img_name_list_path='',
            img_name_list=img_name_list_in_bin,
            voc12_root=args.voc12_root,
            transform=transforms.Compose([
                np.asarray,
                normalize,
                imutils.CenterCrop(args.crop_size),
                imutils.HWC_to_CHW,
                torch.from_numpy,
            ])
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        model.eval()
        model = model.cuda()

        valid_dict = {}
        count = 0

        prediction_batch = torch.randn(0)
        labels_batch = torch.randn(0)

        with torch.no_grad():
            for pack in val_data_loader:
                names = pack[0]
                imgs = pack[1].cuda(non_blocking=True)
                labels = pack[2].cuda(non_blocking=True)
                x = model(imgs)
                prediction = torch.sigmoid(x)
                prediction_batch = torch.cat(
                    (prediction_batch, prediction.cpu()), 0)
                labels_batch = torch.cat((labels_batch, labels.cpu()), 0)

        # visualization
        case_idx = 4
        try:
            case_show(args, names[case_idx], img=imgs[case_idx], label=labels[case_idx], prediction=prediction[case_idx],
                      mIoU=mIoU_in_bin[case_idx], bin=bin,
                      dir_save=f'Fig/fig_ECE/{args.session_name}/case_show_batch_{case_idx}')
        except Exception as e:
            print(f'case show error! {e}')

        prediction_all = torch.cat((prediction_all, prediction_batch.cpu()), 0)
        labels_all = torch.cat((labels_all, labels_batch.cpu()), 0)

        # perfermance_bin = metric_func(pred=prediction_batch, target=labels_batch)
        try:
            # compute mAP pred, target
            perfermance_bin = metric_func(
                pred=prediction_batch, target=labels_batch)
        except:
            perfermance_bin = 0
        perfermance_bins.append(perfermance_bin)

    perfermance_overall = metric_func(pred=deepcopy(
        prediction_all), target=deepcopy(labels_all))
    ML_ECE = ECE_loss(pred=deepcopy(prediction_all), target=deepcopy(
        labels_all), num_bin=num_bin, network=args.fignote, save_path=f'Fig/fig_ECE/{args.session_name}')

    return perfermance_bins, count_bins, images_list_bins, perfermance_overall, ML_ECE


def compute_CAM_ECE(args, dir_score, num_bin):
    df_score = pd.read_csv(dir_score)
    df_score.sort_values(by=['miou'], inplace=True)
    miou_score = df_score['miou'].to_list()

    prediction_all = torch.randn(0)
    labels_all = torch.randn(0)

    mAP_bins = []
    count_bins = []
    images_list_bins = []

    # mAP_function = mean_avg_precision
    mAP_function = mAP
    # mAP_function = Average_Precision

    model = timm.create_model(args.network, pretrained=True, num_classes=20)
    model.load_state_dict(torch.load(args.weights))
    print(f'Loading weights from {args.weights}')
    print('\nvalidating ... ', flush=True, end='')

    each_bin = 1/num_bin
    for bin in range(num_bin):

        bin_lower = bin * each_bin
        bin_upper = (bin + 1) * each_bin

        idx_in_bin = (np.array(miou_score) <= bin_upper) * \
            (np.array(miou_score) >= bin_lower)
        img_name_list_in_bin = np.array(
            df_score['name_image'].tolist())[idx_in_bin]
        mIoU_in_bin = np.array(df_score['miou'].tolist())[idx_in_bin]

        images_list_bins.append(img_name_list_in_bin)
        count_bins.append(len(img_name_list_in_bin))

        normalize = Normalize()
        val_dataset = dywsss.tool.data.VOC12ClsDataset(
            img_name_list_path='',
            img_name_list=img_name_list_in_bin,
            voc12_root=args.voc12_root,
            transform=transforms.Compose([
                np.asarray,
                normalize,
                imutils.CenterCrop(args.crop_size),
                imutils.HWC_to_CHW,
                torch.from_numpy,
            ])
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        model.eval()
        model = model.cuda()

        valid_dict = {}
        count = 0

        prediction_batch = torch.randn(0)
        labels_batch = torch.randn(0)

        with torch.no_grad():
            for pack in val_data_loader:
                names = pack[0]
                imgs = pack[1].cuda(non_blocking=True)
                labels = pack[2].cuda(non_blocking=True)
                x = model(imgs)
                prediction = torch.sigmoid(x)
                prediction_batch = torch.cat(
                    (prediction_batch, prediction.cpu()), 0)
                labels_batch = torch.cat((labels_batch, labels.cpu()), 0)

        prediction_all = torch.cat((prediction_all, prediction_batch.cpu()), 0)
        labels_all = torch.cat((labels_all, labels_batch.cpu()), 0)

        try:
            mAP_bin = mAP_function(pred=prediction_batch, target=labels_batch)
        except:
            mAP_bin = 0
        mAP_bins.append(mAP_bin)

    mAP_overall = mAP_function(pred=prediction_all, target=labels_all)

    return mAP_bins, count_bins, images_list_bins, mAP_overall


def draw_plot_CAM(mAP_bins, count_bins, network, note):
    # count_bins = [count / 1449 * 100 for count in count_bins]
    num_bin = len(mAP_bins)
    # bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bins = np.linspace(start=0, stop=1-1/num_bin, num=num_bin).tolist()
    bins_hundred = np.linspace(start=0, stop=(
        1 - 1 / num_bin) * 100, num=num_bin).tolist()

    # plt.bar(bins, mAP_bins, width=1/num_bin-1/(num_bin*10), edgecolor='#4D79C8')
    # plt.bar(bins, bins_hundred, width=1 / num_bin, color='#FA7F6F', edgecolor='black', label='GAP')
    plt.bar(bins, mAP_bins, width=1 / num_bin,
            color='#82B0D2', edgecolor='black', label='mIoU')

    plt.title(f'{note} per bin {network}')
    plt.xlabel('CAM')
    plt.ylabel(f'{note} of the Multi-label Classification')
    plt.legend()
    plt.savefig(f'Fig/fig_ECE/{args.session_name}/{network}_{note}.png')
    # plt.show()
    plt.clf()

    # plt.bar(bins, count_bins, width=1/num_bin-1/(num_bin*10), edgecolor='#4D79C8')
    plt.bar(bins, count_bins, width=1 / num_bin,
            color='#FA7F6F', edgecolor='black', label='Count')
    plt.title(f'count per bin {network}')
    plt.savefig(f'Fig/fig_ECE/{args.session_name}/{network}_confidence_count.png')
    # plt.show()


def draw_plot_confidence(mAP_bins, count_bins, network, note):
    # count_bins = [count / 1449 * 100 for count in count_bins]
    num_bin = len(mAP_bins)
    # bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bins = np.linspace(start=0, stop=1-1/num_bin, num=num_bin).tolist()
    bins_hundred = np.linspace(start=0, stop=(
        1 - 1 / num_bin) * 100, num=num_bin).tolist()

    # plt.bar(bins, mAP_bins, width=1/num_bin-1/(num_bin*10), edgecolor='#4D79C8')
    # plt.bar(bins, bins_hundred, width=1 / num_bin, color='#FA7F6F', edgecolor='black', label='GAP')
    plt.bar(bins, mAP_bins, width=1 / num_bin,
            color='#82B0D2', edgecolor='black', label='mIoU')

    plt.title(f'{note} per bin {network}')
    plt.xlabel('confidence')
    plt.ylabel(f'{note} of the Multi-label Classification')
    plt.legend()
    plt.savefig(f'Fig/fig_ECE/{args.session_name}/{network}_{note}.png')
    # plt.show()
    plt.clf()

    # plt.bar(bins, count_bins, width=1/num_bin-1/(num_bin*10), edgecolor='#4D79C8')
    plt.bar(bins, count_bins, width=1 / num_bin,
            color='#FA7F6F', edgecolor='black', label='Count')
    plt.title(f'count per bin {network}')
    
    plt.savefig(f'Fig/fig_ECE/{args.session_name}/{network}_confidence_count.png')
    # plt.show()


def cam2mask(cam):
    h, w = list(cam.values())[0].shape
    tensor = np.zeros((21, h, w), np.float32)
    for key in cam.keys():
        tensor[key + 1] = cam[key]
    tensor[0, :, :] = 0.1
    mask = np.argmax(tensor, axis=0).astype(np.uint8)
    return mask


def compute_mIoU(prediction, gt):
    pass


def putpalette(mask):
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    r = mask.copy()
    g = mask.copy()
    b = mask.copy()

    for cls in range(21):
        r[mask == cls] = colormap[cls][0]
        g[mask == cls] = colormap[cls][1]
        b[mask == cls] = colormap[cls][2]

    # b[mask == cls] = self.colormap[color_cls][2]

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r

    return rgb.astype('uint8')


def concat_two_img(img1, img2):
    image = np.hstack((img1, img2))
    return image


def case_show(args, name, img, label, prediction, mIoU, bin, dir_save):
    dir_img = os.path.join(args.voc12_root, args.img_dir, name+'.jpg')
    dir_cam = os.path.join(args.out_cam, name+'.npy')
    args.gt_dir = gt_dir = "voc12/VOC2012/SegmentationClassAug"
    dir_gt = os.path.join(args.gt_dir, name+'.png')

    img = mmcv.imread(dir_img)
    cam = np.load(dir_cam, allow_pickle=True).item()
    gt = mmcv.imread(dir_gt, flag='grayscale')

    cam_mask = cam2mask(cam)
    # mIou = compute_mIoU(cam_mask, gt)
    cam_mask = putpalette(cam_mask)
    gt = putpalette(gt)

    vis_cam = cv2.addWeighted(img, 0.4, cam_mask, 0.6, gamma=0.1)
    vis_gt = cv2.addWeighted(img, 0.4, gt, 0.6, gamma=0.1)
    vis = concat_two_img(vis_cam, vis_gt)

    vis = mmcv.imresize(vis.copy(), (2048, 1024))

    #  prediction, label, mIoU
    label = np.argwhere(label.cpu())[0].tolist()
    prediction_classes = np.argwhere(prediction.cpu() > 0.5)[0].tolist()
    label = [categories[item] for item in label]

    prediction_classes = [categories[item] for item in prediction_classes]
    prediction_confidence = prediction[prediction > 0.5].mean().item()

    if np.isnan(prediction_confidence):
        prediction_confidence = prediction.max().item()
    # sample_accuracy = (prediction > 0.5).sum() + label
    # sample_loss = 0

    text_mIoU = f"JS score = {round(mIoU * 100, 2)}%"
    text_Confidence = f"ML-Confidence = {round(prediction_confidence * 100, 2)}%"
    text_label = f"label = {label}"
    text_prediction = f"prediction = {prediction_classes}"

    cv2.rectangle(vis, (1200, 0), (2048, 0 + 200), (255, 255, 255), -1)

    cv2.putText(vis, text_mIoU, (1225, 25),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(vis, text_Confidence, (1225, 75),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(vis, text_label, (1225, 125),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(vis, text_prediction, (1225, 175),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    mmcv.imwrite(vis, os.path.join(dir_save, f'{bin}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a models')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--csv', type=str)
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
    args = parser.parse_args()

    return args


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    args = cfg

    args.model_dir = os.path.join('work_dirs', args.session_name, "model")
    args.weights = os.path.join(args.model_dir, args.weights)
    args.test_dir = os.path.join('work_dirs', args.session_name, "test")
    args.out_cam = os.path.join(args.test_dir,
                                f'cam_{args.eval_list.split("/")[-1].split(".")[0]}_{args.weights.split("/")[-1].split(".")[0]}')
    
    if os.path.exists(f'Fig/fig_ECE/{args.session_name}') == False:
        os.makedirs(f'Fig/fig_ECE/{args.session_name}')
        
    dir_score = f'miou_loss_csv/miou_loss_{args.session_name}_{args.weights.split("/")[-1].split(".")[0]}_{args.eval_list.split("/")[-1].split(".")[0]}.csv'

    Accuracy_bins, Accuracy_count_bins, _, Accuracy_overall, ML_ECE = compute_Conf_ECE(
        args, dir_score, num_bin=20, metric_func=Accuracy, note='Accuracy')
    draw_plot_confidence(Accuracy_bins, Accuracy_count_bins, args.fignote, note='Accuracy')
    print(f'{args.session_name} Accuracy_overall is {Accuracy_overall}')
    print(f'{args.session_name} ML_ECE is {ML_ECE}')