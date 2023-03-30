import os

import mmcv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from copy import copy



CLS2IDX = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car',
           7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
           15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

def show_cam_on_image(img, cam, dir_save, file_name):
    fig, axs = plt.subplots(1, len(cam.keys()) + 1)
    if 'vit' in dir_save:
        plt.suptitle('Vision Transformer')
    else:
        plt.suptitle('CNN')
    axs[0].imshow(img)
    height, width, _ = img.shape
    for idx, cls in enumerate(cam.keys()):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam[cls]), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(np.array(heatmap), cv2.COLOR_BGR2RGB)
        vis = cv2.addWeighted(img, 0.4, heatmap, 0.6, gamma=0.1)

        axs[idx + 1].imshow(vis)
        axs[idx + 1].set_title(f'{CLS2IDX[cls]}')
        # plt.show()
        plt.savefig(os.path.join(dir_save, f'{file_name}.png'))

    plt.close()
    # heatmap_array = np.array(heatmap_list, dtype='uint8')
    # heatmap = np.mean(heatmap_array, axis=0)
    # vis = cv2.addWeighted(img, 0.4, heatmap, 0.6, gamma=0.1)

    # vis = cv2.addWeighted(img, 0.4, heatmap_sum, 0.6, gamma=0.1)
    # mmcv.imshow(vis)
    # return vis

def concat_img_cam(dir_img, dir_cam, dir_save):

    files_list = os.listdir(dir_cam)
    for file_name in tqdm(files_list):
        file_name = file_name.split('.')[0]
        dir_single_img = os.path.join(dir_img, file_name + '.jpg')
        dir_single_cam_cnn = os.path.join(dir_cam, file_name + '.npy')
        img = mmcv.imread(dir_single_img, channel_order='rgb')
        cam = np.load(dir_single_cam_cnn, allow_pickle=True).item()
        show_cam_on_image(img, cam, dir_save, file_name)

if __name__ == '__main__':
    dir_img = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/voc12/VOC2012/JPEGImages'

    # dir_cam_cnn = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/PMM/data/results/test/voc12_wr38/cam'
    # dir_save_cnn = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/PMM/data/results/test/voc12_wr38/fig'
    # if not os.path.exists(dir_save_cnn): os.makedirs(dir_save_cnn)

    # dir_cam_resnet50 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/resnet50_Pytorch/test/cam'
    # dir_save_resnet50 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/resnet50_Pytorch/test/fig'
    # if not os.path.exists(dir_save_resnet50): os.makedirs(dir_save_resnet50)

    # dir_cam_transformer = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_small_patch16_224_timm_pretrain_lr_0.1/test/cam'
    # dir_save_transformer = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_small_patch16_224_timm_pretrain_lr_0.1/test/fig'
    # if not os.path.exists(dir_save_transformer): os.makedirs(dir_save_transformer)

    # dir_cam_resnet50_BYOL = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/resnet50_BYOL_pretrain_lr_0.01/test/cam'
    # dir_save_resnet50_BYOL = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/resnet50_BYOL_pretrain_lr_0.01/test/fig'
    # if not os.path.exists(dir_save_resnet50_BYOL): os.makedirs(dir_save_resnet50_BYOL)

    # dir_cam_vit_base = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_lr_0.01/test/cam'
    # dir_save_vit_base = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_lr_0.01/test/fig'
    # if not os.path.exists(dir_save_vit_base): os.makedirs(dir_save_vit_base)

    # dir_cam_vit_small = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_small_patch16_224_timm_pretrain_lr_0.01/test/cam'
    # dir_save_vit_small = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_small_patch16_224_timm_pretrain_lr_0.01/test/fig'
    # if not os.path.exists(dir_save_vit_small): os.makedirs(dir_save_vit_small)

    # dir_cam_swin_base = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/swin_base_patch4_window7_224_timm_pretrain_lr_0.01/test/cam'
    # dir_save_swin_base = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/swin_base_patch4_window7_224_timm_pretrain_lr_0.01/test/fig'
    # if not os.path.exists(dir_save_swin_base): os.makedirs(dir_save_swin_base)

    # dir_cam_vit_base_finetune = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01/test/cam_gradcam'
    # dir_save_vit_base_finetune = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01/test/fig_gradcam'
    # if not os.path.exists(dir_save_vit_base_finetune): os.makedirs(dir_save_vit_base_finetune)
    # concat_img_cam(dir_img=dir_img, dir_cam=dir_cam_vit_base_finetune, dir_save=dir_save_vit_base_finetune)
    #
    # dir_cam_swinT_base_finetune = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/swin_base_patch4_window7_224_timm_pretrain_lr_0.01/test/cam_gradcam'
    # dir_save_swinT_base_finetune = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/swin_base_patch4_window7_224_timm_pretrain_lr_0.01/test/fig_gradcam'
    # if not os.path.exists(dir_save_swinT_base_finetune): os.makedirs(dir_save_swinT_base_finetune)
    # concat_img_cam(dir_img=dir_img, dir_cam=dir_cam_swinT_base_finetune, dir_save=dir_save_swinT_base_finetune)

    # dir_cam_res2Net101 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/res2net101_26w_4s_PyTorch_pretrain_multi_epoch_lr_0.01/test/cam'
    # dir_save_res2Net101 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/res2net101_26w_4s_PyTorch_pretrain_multi_epoch_lr_0.01/test/fig'
    # if not os.path.exists(dir_save_res2Net101): os.makedirs(dir_save_res2Net101)
    # concat_img_cam(dir_img=dir_img, dir_cam=dir_cam_res2Net101, dir_save=dir_save_res2Net101)

    dir_cam_resnest269 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01/test/cam'
    dir_save_resnest269 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01/test/fig'
    if not os.path.exists(dir_save_resnest269): os.makedirs(dir_save_resnest269)
    concat_img_cam(dir_img=dir_img, dir_cam=dir_cam_resnest269, dir_save=dir_save_resnest269)