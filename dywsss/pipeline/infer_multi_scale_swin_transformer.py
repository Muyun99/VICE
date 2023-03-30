import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from dywsss.tool.ViT_explanation_generator import LRP
from dywsss.network.build_network import build_model
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import dywsss.tool.data
from dywsss.tool import imutils
from tqdm import tqdm
import math


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

def get_cam_dict(args, cam_function, orig_img, orig_img_size, trans_img, class_indices):
    cam_dict = {}
    for cls in class_indices:
        # vis, heatmap = generate_visualization(attribution_generator, trans_img, class_index=cls)
        # grayscale_cam = cam_function(input_tensor=torch.unsqueeze(trans_img, 0), target_category=cls)
        grayscale_cam = cam_function(
            input_tensor=torch.unsqueeze(trans_img, 0).cuda(),
            target_category=cls,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth
        )[0]
        heatmap = torch.tensor(grayscale_cam)
        heatmap = torch.unsqueeze(heatmap, 0)
        heatmap = torch.unsqueeze(heatmap, 0)
        heatmap = F.upsample(heatmap, orig_img_size[:2], mode='bilinear', align_corners=False)[0][0].cpu().numpy()
        cam_dict[cls] = heatmap
    return cam_dict


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def build_reshape_transform(args, model):
    """build reshape_transform for `cam.activations_and_grads`, some neural
    networks such as SwinTransformer and VisionTransformer need an additional
    reshape operation.

    CNNs don't need, jush return `None`.
    """
    # ViT_based_Transformers have an additional clstoken in features
    if 'swin' in args.network:
        has_clstoken = False
    elif 'vit' in args.network:
        has_clstoken = True
    else:
        return None

    def _reshape_transform(tensor, has_clstoken=has_clstoken):
        """reshape_transform helper."""
        tensor = tensor[:, 1:, :] if has_clstoken else tensor
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        message = 'Only square input images are supported for Transformers.'
        assert height * height == heat_map_area, message
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return _reshape_transform

def get_layer(layer_str, model):
    """get model lyaer from given str."""
    cur_layer = model
    assert layer_str.startswith(
        'model'), "target-layer must start with 'model'"
    layer_items = layer_str.strip().split('.')
    assert not (layer_items[-1].startswith('relu')
                or layer_items[-1].startswith('bn')
                ), "target-layer can't be 'bn' or 'relu'"
    for item_str in layer_items[1:]:
        if hasattr(cur_layer, item_str):
            cur_layer = getattr(cur_layer, item_str)
        else:
            raise ValueError(
                f"model don't have `{layer_str}`, please use valid layers")
    return cur_layer

def infer_multi_scale_swin(args):
    print(vars(args))
    normalize = Normalize()
    model = build_model(args)
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

    global iter
    for iter, (img_name, img_list, label) in tqdm(enumerate(infer_data_loader)):
        img_name = img_name[0]
        label = label[0]

        img_path = dywsss.tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = Image.open(img_path).convert('RGB')
        orig_img_size = np.asarray(orig_img).shape
        trans_img = transform(orig_img)

        # build target layers
        target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [get_layer(layer_str, model) for layer_str in args.target_layers]
        assert len(args.target_layers) != 0

        # init a cam grad calculator
        use_cuda = True if 'cuda' in args.device else False
        reshape_transform = build_reshape_transform(args, model)

        # calculate cam grads and show|save the visualization image
        grayscale_cam = cam(
            input_tensor=data['img'],
            target_category=args.target_category,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth)


        gt_cls_indices = np.argwhere(label == 1)[0].numpy().tolist()
        target_layers = [model.layers[-1].blocks[-1].norm1]
        cam_function = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_transform)
        cam_dict = get_cam_dict(args, cam_function, orig_img, orig_img_size, trans_img, gt_cls_indices)

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