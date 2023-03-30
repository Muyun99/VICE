python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_res2net50_48w_2s_lr_0.01_best_val.csv --plot_title Res2Net-50 --VICE 0.775
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_efficientnet_b4_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title EfficientNet-b4 --VICE 0.823

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnest14d_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNeSt-14 --VICE 0.850
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnest26d_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNeSt-26 --VICE 0.842
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnest50d_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNeSt-50 --VICE 0.928
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnest101e_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNeSt-101 --VICE 0.909

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet18_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNet-18 --VICE 0.674
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet34_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNet-34 --VICE 0.914
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNet-50 --VICE 0.859
# python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet101_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNet-101 --VICE

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_small_patch16_224_timm_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ViT-Small --VICE 0.978
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ViT-Base --VICE 0.878



python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_timm_finetune_lr_0.01_best_val.csv --plot_title ViT-Base-Timm-Finetuning --VICE 1.139
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_DINO_finetune_lr_0.01_best_val.csv --plot_title ViT-Base-DINO-Finetuning --VICE 1.112
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_MoCov3_finetune_lr_0.01_best_val.csv --plot_title ViT-Base-MoCov3-Finetuning --VICE 0.197
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_MoCov3_VOC_finetune_lr_0.01_best_val.csv --plot_title ViT-Base-MoCov3-VOC--Finetuning --VICE 0.463

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_timm_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ViT-Base-Timm-Pretrain --VICE 0.878
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_DINO_pretrain_lr_0.01_best_val.csv --plot_title ViT-Base-DINO-Pretrain --VICE 0.933
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_MoCov3_pretrain_lr_0.01_best_val.csv --plot_title ViT-Base-MoCov3-Pretrain --VICE 0.648
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_vit_base_patch16_224_MoCov3_VOC_pretrain_lr_0.01_best_val.csv --plot_title ViT-Base-MoCov3-VOC-Pretrain --VICE 0.471

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_PyTorch_pretrain_multi_epoch_lr_0.01_best_val.csv --plot_title ResNet-50-PyTorch-Pretrain --VICE 0.859
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_BYOL_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50-BYOL-Pretrain --VICE 0.513
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_DINO_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50-DINO-Pretrain --VICE 0.451
# python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov2_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50
# python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_DenseCL_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov3_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50-MoCov3-Pretrain --VICE 0.541
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov3_VOC_pretrain_lr_0.01_best_val.csv --plot_title ResNet-50-MoCov3-VOC-Pretrain --VICE 0.266

python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_Pytorch_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50-PyTorch-Finetuning --VICE 0.619
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_BYOL_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50-BYOL-Finetuning --VICE 0.219
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_DINO_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50-DINO-Finetuning --VICE 0.246
# python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov2_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50
# python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_DenseCL_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov3_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50-MoCov3-Finetuning --VICE 0.204
python tool_muyun/compute_pearson.py --dir_csv miou_loss_csv/miou_loss_resnet50_MoCov3_VOC_finetuning_lr_0.01_best_val.csv --plot_title ResNet-50-MoCov3-VOC-Finetuning --VICE 0.281


