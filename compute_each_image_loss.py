from PIL import Image
from math import log, e
from tqdm import tqdm
import argparse
from mmcv import Config, DictAction
import os
import timm
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import dywsss.tool.data
from dywsss.tool import pyutils, imutils
from dywsss.tool.torch_utils import *

cudnn.enabled = True


# the function to calculate entropy, you should use the probabilities as the parameters
def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


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


def compute_each_loss(args):
    normalize = Normalize()
    val_dataset = dywsss.tool.data.VOC12ClsDataset(
        args.eval_list,
        voc12_root=args.voc12_root,
        transform=transforms.Compose([
            np.asarray,
            normalize,
            imutils.CenterCrop(args.crop_size),
            imutils.HWC_to_CHW,
            torch.from_numpy])
    )
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = timm.create_model(args.network, pretrained=True, num_classes=20)
    model.load_state_dict(torch.load(args.weights))
    print(f'Loading weights from {args.weights}')
    print('\nvalidating ... ', flush=True, end='')

    model.eval()
    model = model.cuda()

    valid_dict = {}
    count = 0

    with torch.no_grad():
        for pack in tqdm(val_data_loader):
            names = pack[0]
            imgs = pack[1].cuda(non_blocking=True)
            labels = pack[2].cuda(non_blocking=True)

            x = model(imgs)
            loss = F.multilabel_soft_margin_loss(x, labels, reduction='none')

            for idx, name in enumerate(names):
                name_single = name
                loss_single = loss[idx].cpu().numpy()

                Prediction_threshold = 0.5
                prediction = F.sigmoid(x)[idx]
                prediction_single = np.argwhere(
                    prediction.cpu() > 0.5)[0].tolist()

                if len(prediction_single) != 0:
                    confidence_single = prediction[prediction_single].mean(
                    ).cpu().numpy().item()
                else:
                    confidence_single = prediction.max().cpu().numpy().item()

                entropy_single = entropy(
                    F.sigmoid(x)[idx].cpu().numpy().tolist())
                label_single = np.where(labels[idx].cpu().numpy() == 1)[0]

                valid_dict[count] = [name_single, loss_single, confidence_single,
                                     entropy_single, label_single, prediction_single]
                count += 1
                # break
            # break

    df_loss = pd.DataFrame.from_dict(valid_dict, orient='index')
    df_loss.columns = ['name_image', 'loss',
                       'confidence', 'entropy', 'label', 'prediction']
    df_loss.sort_values(by='name_image', inplace=True)
    return df_loss


def compute_each_mIoU(args):
    num_cls = 21
    df = pd.read_csv(args.eval_list, names=['filename'])
    name_list = df['filename'].values

    score_dict = {}

    for idx, name in tqdm(enumerate(name_list)):
        name = name_list[idx]
        if args.input_type == 'png':
            predict_file = os.path.join(args.out_cam, '%s.png' % name)
            # cv2.imread(predict_file)
            predict = np.array(Image.open(predict_file))
        elif args.input_type == 'npy':
            predict_file = os.path.join(args.out_cam, '%s.npy' % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in predict_dict.keys():
                tensor[key + 1] = predict_dict[key]
            tensor[0, :, :] = args.threshold
            predict = np.argmax(tensor, axis=0).astype(np.uint8)

        gt_file = os.path.join(args.gt_dir, '%s.png' % name)
        gt = np.array(Image.open(gt_file))
        cal = gt < 255

        mask = (predict == gt) * cal

        T_single = []
        P_single = []
        TP_single = []
        FN_single = []
        FP_single = []

        Precision_single = []
        Recall_single = []

        for i in range(num_cls):
            P_single.append(np.sum((predict == i) * cal))
            T_single.append(np.sum((gt == i) * cal))
            TP_single.append(np.sum((gt == i) * mask))

        IoU_single = []

        for i in range(num_cls):
            IoU_single.append(
                TP_single[i] / (T_single[i] + P_single[i] - TP_single[i] + 1e-10))

        miou_single = np.mean(np.array(IoU_single)[np.array(IoU_single) != 0])
        miou_exist = np.array(IoU_single)[np.array(IoU_single) != 0]
        score_dict[idx] = [name, miou_single, miou_exist]

    df_miou = pd.DataFrame.from_dict(score_dict, orient='index')
    # df_miou.columns = ['name_image', 'miou', 'miou_exist', 'F1_single', 'mFN(Under-activation)', 'mFP(over-activation)']
    df_miou.columns = ['name_image', 'miou', 'miou_exist']
    df_miou.sort_values(by='name_image', inplace=True)
    return df_miou


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
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    args = cfg

    args.model_dir = os.path.join('work_dirs', args.session_name, "model")
    args.test_dir = os.path.join('work_dirs', args.session_name, "test")
    args.log_dir = os.path.join('work_dirs', args.session_name, "log")
    args.tensorboard_dir = os.path.join(
        'work_dirs', args.session_name, "tensorboard")
    args.weights = os.path.join(args.model_dir, args.weights)
    args.eval_list = f'voc12/VOC2012/ImageSets/Segmentation/{args.eval_list}'

    df_loss = compute_each_loss(args)
    args.input_type = 'npy'
    args.threshold = 0.1
    args.out_cam = os.path.join(args.test_dir,
                                f'cam_{args.eval_list.split("/")[-1].split(".")[0]}_{args.weights.split("/")[-1].split(".")[0]}')

    args.gt_dir = 'voc12/VOC2012/SegmentationClassAug'
    df_miou = compute_each_mIoU(args)

    df_merge = pd.merge(df_loss, df_miou, how='inner', on=['name_image'])
    if os.path.exists('miou_loss_csv') is False:
        os.mkdir('miou_loss_csv')
    df_merge.to_csv(
        f'miou_loss_csv/miou_loss_{args.session_name}_{args.weights.split("/")[-1].split(".")[0]}_{args.eval_list.split("/")[-1].split(".")[0]}.csv',
        index=False)
