import os
# Pretrain
pretrain = None

# Environment
num_workers = os.cpu_count()//2
local_rank = 0
random_seed = 2022

# Dataset
dataset = "voc12"
csv = ""

# Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.
voc12_root = "voc12/VOC2012"
img_dir = "JPEGImages"
gt_dir = "voc12/VOC2012/SegmentationClass"
cls_label = "voc12/cls_labels.npy"
train_list = "voc12/train_aug.txt"
val_list = "voc12/val.txt"
infer_list = "voc12/val.txt"
eval_list = ""
# Train
finetune = False
batch_size = 16
warmup_epochs = 5
max_epoches = 20
network = "vit_small_patch16_224"

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = dict(type='MultiStepLR', milestones=[10, 15], gamma=0.1)

num_cls = 21
heatmap_root = ""
dilations = "1_1_1_1"
structure = "models/scalenet/structures/scalenet101.json"

# 目前是224
crop_size = 224
weights = ""
cut_p = 0
cut_s = 0.25
scales = "0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3"
stride = 300
session_name = ""


# Inference
out_cam = None
out_crf = None

# CRFs
crf_threshs = '24'
cam_thresh = 0.05
cv_scale = 0.3

# Evaluation
type = 'png'
curve = False
eval_thresh = 0.1
predict_dir = ""

# Pipline
nni_train_multi_scale = False
train_multi_scale = False
gen_mask_for_multi_crop = False
train_multi_crop = False
eval = False
gen_seg_mask = False
