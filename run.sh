# ResNet-50
# 1. Train a model
python run_resnet50.py config/resnet50.py --options "train_multi_scale=True" "network=resnet50" "session_name=resnet50_PyTorch_pretrain_lr_0.01" "optimizer.lr=0.01" 

# 2. Inference the CAMs 
python run_resnet50.py config/resnet50.py --options "gen_mask_for_multi_crop=True" "network=resnet50" "session_name=resnet50_PyTorch_pretrain_lr_0.01" "weights=best.pth" "infer_list=val.txt" 

# 3. Evaluate the Quality of CAMs (JI Score).
python evaluation.py \
    --list voc12/VOC2012/ImageSets/Segmentation/val.txt \
    --predict_dir work_dirs/resnet50_PyTorch_pretrain_lr_0.01/test/cam_val_best \
    --gt_dir voc12/VOC2012/SegmentationClassAug \
    --comment resnet50_PyTorch_pretrain_lr_0.01_cam_val_best \
    --type npy \
    --curve True

# 4. Compute Sample-wise Loss.
python compute_each_image_loss.py config/resnet50.py --options "network=resnet50" "session_name=resnet50_PyTorch_pretrain_lr_0.01" "weights=best.pth" "eval_list=val.txt" 

# 5. Compute ECE-ML and VICE and Draw the Realibility Diagrams of ECE-ML and VICE
python compute_ECE.py config/resnet50.py --options "network=resnet50" "session_name=resnet50_PyTorch_pretrain_lr_0.01" "weights=best.pth" "eval_list=val.txt" "fignote=ResNet-50"

python compute_VICE.py --dir_csv miou_loss_csv/miou_loss_resnet50_PyTorch_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50-PyTorch-Pretrain





# ViT-Small

# 1. Train a model

python run_transformer.py config/vision_transformer.py --options "train_multi_scale=True" "network=vit_small_patch16_224" "session_name=vit_small_patch16_224_timm_pretrain_lr_0.01" "crop_size=224"

# 2. Inference the CAMs 

python run_transformer.py config/vision_transformer.py --options "gen_mask_for_multi_crop=True" "network=vit_small_patch16_224" "session_name=vit_small_patch16_224_timm_pretrain_lr_0.01" "crop_size=224" "infer_list=val.txt" "weights=best.pth"

# 3. Evaluate the Quality of CAMs (JI Score).

python evaluation.py \
    --list voc12/VOC2012/ImageSets/Segmentation/val.txt \
    --predict_dir work_dirs/vit_small_patch16_224_timm_pretrain_lr_0.01/test/cam_val_best \
    --gt_dir voc12/VOC2012/SegmentationClassAug \
    --comment vit_small_patch16_224_timm_pretrain_lr_0.01_cam_val_best \
    --type npy \
    --curve True

# 4. Compute Sample-wise Loss.

python compute_each_image_loss.py config/vision_transformer.py --options "network=vit_small_patch16_224" "session_name=vit_small_patch16_224_timm_pretrain_lr_0.01" "weights=best.pth" "eval_list=val.txt" 

# 5. Compute ECE-ML and VICE and Draw the Realibility Diagrams of ECE-ML and VICE

python compute_ECE.py config/resnet50.py --options "network=vit_small_patch16_224" "session_name=vit_small_patch16_224_timm_pretrain_lr_0.01" "weights=best.pth" "eval_list=val.txt" "fignote=ViT-Small"

python compute_VICE.py --dir_csv miou_loss_csv/miou_loss_vit_small_patch16_224_timm_pretrain_lr_0.01_best_val.csv --plot_title ViT-Small
