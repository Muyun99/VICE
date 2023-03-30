import os

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from dywsss.tool.ViT_explanation_generator import LRP
import torchvision.transforms as transforms
from dywsss.network.vit_builder import build_model
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from mmcv import DictAction, Config
import dywsss.tool.data
from dywsss.tool import imutils
from tqdm import tqdm
import argparse
from torchvision.transforms import Compose, Normalize, ToTensor


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


CLS2IDX = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car',
           7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
           15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(attribution_generator, original_image, class_index=None):
    # full
    # rollout
    # grad
    # last_layer
    # last_layer_attn
    # second_layer
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, transformer_attribution


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    return class_indices


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swint(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def preprocess_image(img: np.ndarray, mean, std) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return torch.tensor(preprocessing(img.copy())).unsqueeze(0)


def infer_multi_scale(args):

    print(vars(args))
    normalize = Normalize()
    model = build_model(args.network, num_classes=20, pretrained=False)
    model.load_state_dict(torch.load(args.weights))
    print(f'Loading weights from {args.weights}')
    model.eval()
    model.cuda()

    dywsss.tool.data.NUM_CLS = args.num_cls
    dywsss.tool.data.IMG_FOLDER_NAME = args.img_dir
    dywsss.tool.data.ANNOT_FOLDER_NAME = args.gt_dir
    dywsss.tool.data.CLS_LABEL = args.cls_label

    infer_dataset = dywsss.tool.data.VOC12ClsDatasetMSF(args.infer_list,
                                                        voc12_root=args.voc12_root,
                                                        # pseudo_gt=args.pseudo_list,
                                                        scales=[0.5, 1.0, 1.5, 2.0],
                                                        inter_transform=torchvision.transforms.Compose(
                                                     [np.asarray,
                                                      normalize,
                                                      imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    transform = transforms.Compose([
        transforms.Resize(size=(args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    crf_alpha = [int(i) for i in args.crf_threshs.split('_')]
    if 'swin' in args.network:
        # SwinT
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swint
    else:
        # ViT
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=reshape_transform)

    global iter
    for iter, (img_name, img_list, label) in tqdm(enumerate(infer_data_loader)):
        img_name = img_name[0]
        label = label[0]

        img_path = dywsss.tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = Image.open(img_path).convert('RGB')
        orig_img_size = np.asarray(orig_img).shape
        trans_img = transform(orig_img)
        input_tensor = trans_img.unsqueeze(0)
        eigen_smooth = True
        aug_smooth = True
        cam_dict = {}
        gt_cls_indices = np.argwhere(label == 1)[0].numpy().tolist()

        for target_category in gt_cls_indices:
            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                eigen_smooth=eigen_smooth,
                                aug_smooth=aug_smooth)

            # transform_show = transforms.Compose([
            #     transforms.Resize(size=(args.crop_size, args.crop_size)),
            #     transforms.ToTensor(),
            # ])
            # rgb_img = transform_show(orig_img).permute(1, 2, 0)
            # cam_image = grayscale_cam[0, :]
            # cam_image = show_cam_on_image(rgb_img, cam_image)
            # mmcv.imshow(cam_image)

            cam_output = grayscale_cam[0]
            heatmap = torch.tensor(cam_output)
            heatmap = torch.unsqueeze(heatmap, 0)
            heatmap = torch.unsqueeze(heatmap, 0)
            heatmap = F.upsample(heatmap, orig_img_size[:2], mode='bilinear', align_corners=False)[0][0].cpu().numpy()
            cam_dict[target_category] = heatmap


        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        def _crf_with_alpha(cam_dict, alpha, num_cls):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            orig_img_numpy = np.asarray(orig_img)
            crf_score = imutils.crf_inference(orig_img_numpy, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()
            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key + 1] = crf_score[i + 1]

            return n_crf_al

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t, args.num_cls)
                folder = args.out_crf
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)

        if iter % 500 == 0:
            print('iter: ' + str(iter), flush=True)

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

    # Output Path
    args.model_dir = os.path.join('work_dirs', args.session_name, "model")
    args.test_dir = os.path.join('work_dirs', args.session_name, "test")
    args.log_dir = os.path.join('work_dirs', args.session_name, "log")
    args.tensorboard_dir = os.path.join('work_dirs', args.session_name, "tensorboard")

    os.makedirs("work_dirs", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    args.infer_list = 'voc12/val.txt'
    args.weights = os.path.join(args.model_dir, 'best.pth')
    args.out_cam = os.path.join(args.test_dir, 'cam_gradcam')
    args.out_crf = os.path.join(args.test_dir, 'train_mask')
    infer_multi_scale(args)