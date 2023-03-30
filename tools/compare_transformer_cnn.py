import os

import mmcv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

dir_cam_cnn = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/resnet50_BYOL_pretrain_lr_0.01/test/fig'
dir_cam_transformer = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/work_dirs/vit_base_patch16_224_timm_pretrain_lr_0.01/test/fig'

dir_save = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/TWSS/vis'

files_list = os.listdir(dir_cam_transformer)
for file_name in tqdm(files_list):
    cam_cnn = mmcv.imread(os.path.join(dir_cam_cnn, file_name))
    cam_transformer = mmcv.imread(os.path.join(dir_cam_transformer, file_name))
    image = np.vstack((cam_cnn, cam_transformer))
    mmcv.imwrite(image, os.path.join(dir_save, file_name))


# 原图


# ====使用numpy的数组矩阵合并concatenate======
# 纵向连接

# # 横向连接
# image = np.concatenate([img1, img2], axis=1)

# =============


# cv2.imshow('image', image)

