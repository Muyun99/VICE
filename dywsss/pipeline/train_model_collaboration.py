import torch.nn.functional as F
from dywsss.tool import pyutils, imutils
import dywsss.tool.data
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.backends import cudnn
from dywsss.scheduler import build_scheduler
from dywsss.optimizer import build_optimizer, build_optimizer_for_two_model
from torch.utils.tensorboard import SummaryWriter
from dywsss.network.vit_builder import build_model
from dywsss.tool.torch_utils import *
from dywsss.tool.ViT_explanation_generator import LRP
from torch.nn import CrossEntropyLoss, SmoothL1Loss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

cudnn.enabled = True
import timm
import os
from dywsss.tool.load_pretrain import load_checkpoint

def forward_cam_cnn(model, img, orig_img_size):
    # model.fc.weight (20,2048)
    # feature (1,2048,6,8)

    feature = model.forward_features(img)
    weight = model.fc.weight
    weight = torch.unsqueeze(weight, 2)
    weight = torch.unsqueeze(weight, 2)

    cam = F.conv2d(feature, weight)
    cam = F.relu(cam)

    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)
    return cam

def forward_cam_vit(model, img):
    # model.fc.weight (20,2048)
    # feature (1,2048,6,8)

    feature = model.forward_features(img)
    weight = model.fc.weight
    weight = torch.unsqueeze(weight, 2)
    weight = torch.unsqueeze(weight, 2)

    cam = F.conv2d(feature, weight)
    cam = F.relu(cam)
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
    return transformer_attribution

def get_cam_dict(attribution_generator, orig_img_size, trans_img):
    cams_list = []
    for img in trans_img:
        cam_list = []
        for cls in range(20):
            # heatmap: ndarray:(224,224)
            # print(trans_img.shape)
            heatmap = generate_visualization(attribution_generator, img, class_index=cls)
            heatmap = torch.tensor(heatmap)
            heatmap = torch.unsqueeze(heatmap, 0)
            heatmap = torch.unsqueeze(heatmap, 0)
            # print(heatmap.size())
            # heatmap: Tensor (1,1,224,224)
            heatmap = F.upsample(heatmap, orig_img_size, mode='bilinear', align_corners=False)[0][0].cpu().numpy()
            cam_list.append(heatmap)
            # cam_dict[cls] = heatmap
        # cam_tensor = torch.tensor(np.array(cam_list))
        cams_list.append(cam_list)
    cams_tensor = torch.tensor(np.array(cams_list))

    return cams_tensor.cuda()

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

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


def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1].cuda(non_blocking=True)
            label = pack[2].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()
    val_loss = val_loss_meter.pop('loss')
    print('loss:', val_loss)

    return val_loss


def train(args):
    normalize = Normalize()
    pyutils.Logger(os.path.join(args.log_dir, args.session_name) + '.log')
    train_dataset = dywsss.tool.data.VOC12ClsDataset(
        args.train_list,
        voc12_root=args.voc12_root,
        transform=transforms.Compose([
            imutils.RandomResizeLong(448, 768),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            np.asarray,
            normalize,
            imutils.RandomCrop(args.crop_size),
            imutils.HWC_to_CHW,
            torch.from_numpy])
    )
    val_dataset = dywsss.tool.data.VOC12ClsDataset(
        args.val_list,
        voc12_root=args.voc12_root,
        transform=transforms.Compose([
            np.asarray,
            normalize,
            imutils.CenterCrop(224),
            imutils.HWC_to_CHW,
            torch.from_numpy])
    )
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    model_resnet50 = timm.create_model(args.cnn_network, pretrained=True, num_classes=20)
    model_vit = build_model(args.vit_network, num_classes=20, pretrained=True)

    # Loading weight
    if args.cnn_pretrain is None:
        print(f'{args.cnn_network} Using Pytorch weight')
    else:
        print(f'{args.cnn_network} Loading Pretrain Model from {args.cnn_pretrain}')
        msg = model_resnet50.load_state_dict(torch.load(args.cnn_pretrain), strict=False)
        print(msg)

    if args.vit_pretrain is None:
        print(f'{args.vit_network} Using Pytorch weight')
    else:
        print(f'{args.vit_network} Loading Pretrain Model from {args.vit_pretrain}')
        load_checkpoint(model_resnet50, args.pretrain, strict=False)

    model_vit.train()
    model_resnet50.train()
    model_vit = model_vit.cuda()
    model_resnet50 = model_resnet50.cuda()
    attribution_generator = LRP(model_vit)

    squeeze_block = FCUDown(inplanes=2048, outplanes=768, dw_stride=4)
    expand_block = FCUUp(inplanes=768, outplanes=2048, up_stride=4)

    writer = SummaryWriter(args.tensorboard_dir)
    optimizer = build_optimizer_for_two_model(cfg=args, model1=model_resnet50, model2=model_vit)
    scheduler = build_scheduler(args, optimizer)

    global_step = 0
    best_valid_loss_cnn = 99999999
    best_valid_loss_vit = 99999999

    avg_meter = pyutils.AverageMeter('loss_cnn', 'loss_vit', 'loss_cam', 'loss_kl', 'loss_all')

    timer = pyutils.Timer("Session started: ")
    KLLoss = MSELoss()
    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):

            img = pack[1].cuda(non_blocking=True)
            label = pack[2].cuda(non_blocking=True)

            # CNN cls loss
            x_cnn = model_resnet50(img)
            loss_cnn = F.multilabel_soft_margin_loss(x_cnn, label)
            avg_meter.add({'loss_cnn': loss_cnn.item()})

            # Transformer cls loss
            x_vit = model_vit(img)
            loss_vit = F.multilabel_soft_margin_loss(x_vit, label)
            avg_meter.add({'loss_vit': loss_vit.item()})

            # CAM loss
            # img1: {24, 3, 448, 448}
            # img2: {24, 3, 134, 134}
            # label: {24, 21, 1, 1}
            # cam1: {24, 21, 448, 448}
            # cam_rv1:{24, 21, 448, 448}
            # loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))

            # original_img_size = img.cpu().numpy().shape
            # cam_cnn = forward_cam_cnn(model_resnet50, img, orig_img_size=original_img_size[2:])
            # cam_vit = get_cam_dict(
            #     attribution_generator,
            #     orig_img_size=original_img_size[2:],
            #     trans_img=img)
            # loss_cam = None
            # loss_cam = torch.mean(torch.abs(cam_cnn - cam_vit))
            # avg_meter.add({'loss_cam': loss_cam.item()})

            feature_cnn = model_resnet50.forward_features(img)
            feature_vit = model_vit.forward_features(img)

            squeeze_block = nn.Conv2d(2048, 768, 1, bias=True).cuda()
            feature_cnn = squeeze_block(feature_cnn)
            feature_cnn = model_resnet50.global_pool(feature_cnn)

            loss_kl = KLLoss(feature_cnn, feature_vit) * 0.1

            avg_meter.add({'loss_kl': loss_kl.item()})

            loss_all = loss_cnn + loss_vit + loss_kl
            avg_meter.add({'loss_all': loss_all.item()})

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            global_step += 1

            if (global_step - 1) % 50 == 0:
                timer.update_progress(global_step / max_step)

                print('Iter:%5d/%5d' % (global_step - 1, max_step),
                      'Loss_cnn:%.4f' % (avg_meter.pop('loss_cnn')),
                      'Loss_vit:%.4f' % (avg_meter.pop('loss_vit')),
                      # 'Loss_cam:%.4f' % (avg_meter.pop('loss_cam')),
                      'loss_kl:%.4f' % (avg_meter.pop('loss_kl')),
                      'Loss_all:%.4f' % (avg_meter.pop('loss_all')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size /
                                     timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            writer.add_scalar('Train/loss_cnn', loss_cnn, global_step)
            writer.add_scalar('Train/loss_vit', loss_vit, global_step)
            # writer.add_scalar('Train/loss_cam', loss_cam, global_step)
            writer.add_scalar('Train/loss_kl', loss_kl, global_step)
            writer.add_scalar('Train/loss_all', loss_all, global_step)
            writer.add_scalar('Train/learning_rate', learning_rate, global_step)

        valid_loss_cnn = validate(model_resnet50, val_data_loader)
        valid_loss_vit = validate(model_vit, val_data_loader)

        if valid_loss_cnn < best_valid_loss_cnn:
            best_valid_loss_cnn = valid_loss_cnn
            torch.save(model_resnet50.state_dict(), os.path.join(args.model_dir, 'cnn_best.pth'))
        if valid_loss_vit < best_valid_loss_vit:
            best_valid_loss_vit = valid_loss_vit
            torch.save(model_vit.state_dict(), os.path.join(args.model_dir, 'vit_best.pth'))
        writer.add_scalar('Evaluation/valid_loss_cnn', valid_loss_cnn, global_step)
        writer.add_scalar('Evaluation/best_valid_loss_cnn', best_valid_loss_cnn, global_step)
        writer.add_scalar('Evaluation/valid_loss_vit', valid_loss_vit, global_step)
        writer.add_scalar('Evaluation/best_valid_loss_vit', best_valid_loss_vit, global_step)

        timer.reset_stage()
        scheduler.step()

    torch.save(model_resnet50.state_dict(), os.path.join(args.model_dir, 'cnn_last.pth'))
    torch.save(model_vit.state_dict(), os.path.join(args.model_dir, 'vit_last.pth'))