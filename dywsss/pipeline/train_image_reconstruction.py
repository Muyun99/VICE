import numpy as np
import timm
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import dywsss.tool.data
from dywsss.tool import pyutils, imutils, torchutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch import nn
from dywsss.scheduler import build_scheduler
from dywsss.optimizer import build_optimizer
from torch.utils.tensorboard import SummaryWriter
from dywsss.tool.torch_utils import *
from dywsss.network.build_network import build_model
from dywsss.network.dlinknet import UNet_ResNet50
from torch.nn import CrossEntropyLoss, SmoothL1Loss, MSELoss
from dywsss.loss import content_style_loss




class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def validate(model, data_loader, restruction_criterion):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss_cls', 'loss_restruction')

    model.eval()
    # restruction_criterion = SmoothL1Loss()
    with torch.no_grad():
        for pack in data_loader:
            img = pack[1].cuda(non_blocking=True)
            label = pack[2].cuda(non_blocking=True)

            cls_result, restruction_result = model(img)
            loss_cls = F.multilabel_soft_margin_loss(cls_result, label)
            loss_restruction = restruction_criterion.forward(restruction_result, img)
            val_loss_meter.add({'loss_cls': loss_cls.item()})
            val_loss_meter.add({'loss_restruction': loss_restruction.item()})

    model.train()

    val_loss_cls = val_loss_meter.pop('loss_cls')
    val_loss_restruction = val_loss_meter.pop('loss_restruction')
    print('loss:', val_loss_cls, val_loss_restruction)

    return val_loss_cls, val_loss_restruction

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def get_parameter_groups(model):
    from_scratch_layers = ["fc"]
    groups = ([], [], [], [])

    for m in model.modules():

        if isinstance(m, nn.Conv2d):

            if m.weight.requires_grad:
                if m in from_scratch_layers:
                    groups[2].append(m.weight)
                else:
                    groups[0].append(m.weight)

            if m.bias is not None and m.bias.requires_grad:

                if m in from_scratch_layers:
                    groups[3].append(m.bias)
                else:
                    groups[1].append(m.bias)
    return groups

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def train(args):
    normalize = Normalize()
    pyutils.Logger(os.path.join(args.log_dir, args.session_name) + '.log')

    model_resnet = build_model(args)
    model = UNet_ResNet50(resnet_model=model_resnet)

    restruction_criterion = None
    if args.reconstruction_loss == 'MSELoss':
        print('Using MSELoss')
        restruction_criterion = MSELoss()
    elif args.reconstruction_loss == 'SmoothL1Loss':
        print('Using SmoothL1Loss')
        restruction_criterion = SmoothL1Loss()
    elif args.reconstruction_loss == 'content_style_loss':
        print('Using content_style_loss')
        restruction_criterion = content_style_loss()

    writer = SummaryWriter(args.tensorboard_dir)
    optimizer = build_optimizer(cfg=args, model=model)
    scheduler = build_scheduler(args, optimizer)
    global_step = 0
    best_valid_loss = 99999999

    # dywsss.tool.data.NUM_CLS = args.num_cls
    # dywsss.tool.data.IMG_FOLDER_NAME = args.img_dir
    # dywsss.tool.data.ANNOT_FOLDER_NAME = args.gt_dir
    # dywsss.tool.data.CLS_LABEL = args.cls_label
    #
    if args.heatmap_root == '':
        train_dataset = dywsss.tool.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                                         # pseudo_gt=args.pseudo_list,
                                                         transform=transforms.Compose([
                            imutils.RandomResizeLong(448, 768),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            np.asarray,
                            normalize,
                            imutils.RandomCrop(args.crop_size),
                            imutils.HWC_to_CHW,
                            torch.from_numpy
                        ]))
    else:
        train_dataset = dywsss.tool.data.VOC12ClsHeatCropDataset(args.train_list, voc12_root=args.voc12_root,
                                                                 heatmap_root=args.heatmap_root, heat_type='npy', scale=(0.04, 1), ratio=(3. / 5., 5. / 3.),
                                                                 label_match_thresh=0.1, cut_scale=(0.04, args.cut_s), cut_p=args.cut_p,
                                                                 crop_scales=[float(i) for i in args.scales.split(',')] if args.scales != "" else [],
                                                                 crop_size=448, stride=args.stride,
                                                                 transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            transforms.Resize((args.crop_size, args.crop_size)),
                            np.asarray,
                            normalize,
                            imutils.HWC_to_CHW,
                            torch.from_numpy
                        ]))
    val_dataset = dywsss.tool.data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
                                                   transform=transforms.Compose([
                                                np.asarray,
                                                normalize,
                                                imutils.CenterCrop(224),
                                                imutils.HWC_to_CHW,
                                                torch.from_numpy,
                                            ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    # param_groups = get_parameter_groups(model)
    # optimizer = torchutils.PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    # ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_restruction')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1].cuda(non_blocking=True)
            label = pack[2].cuda(non_blocking=True)

            cls_result, restruction_result = model(img)

            loss_cls = F.multilabel_soft_margin_loss(cls_result, label)
            # img = torch.tensor(img, dtype=torch.long)
            loss_restruction = restruction_criterion.forward(restruction_result, img)
            loss = loss_cls + loss_restruction

            avg_meter.add({'loss_cls': loss_cls.item()})
            avg_meter.add({'loss_restruction': loss_restruction.item()})
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if (global_step - 1) % 50 == 0:
                timer.update_progress(global_step / max_step)

                print('Iter:%5d/%5d' % (global_step - 1, max_step),
                      'Loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_restruction:%.4f' % (avg_meter.pop('loss_restruction')),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            writer.add_scalar('Train/loss', loss, global_step)
            writer.add_scalar('Train/loss_cls', loss_cls, global_step)
            writer.add_scalar('Train/loss_restruction', loss_restruction, global_step)
            writer.add_scalar('Train/learning_rate', learning_rate, global_step)

        val_loss_cls, val_loss_restruction = validate(model, val_data_loader, restruction_criterion)
        valid_loss = val_loss_cls + val_loss_restruction
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.module.state_dict(), os.path.join(args.model_dir, 'best.pth'))
        writer.add_scalar('Evaluation/val_loss_cls', val_loss_cls, global_step)
        writer.add_scalar('Evaluation/val_loss_restruction', val_loss_restruction, global_step)
        writer.add_scalar('Evaluation/best_valid_loss', best_valid_loss, global_step)
        timer.reset_stage()
        scheduler.step()

    torch.save(model.module.state_dict(), os.path.join(args.model_dir, 'last.pth'))