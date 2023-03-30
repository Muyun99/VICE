
import torch.nn.functional as F
from dywsss.tool import pyutils, imutils
import dywsss.tool.data
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.backends import cudnn
from dywsss.scheduler import build_scheduler
from dywsss.optimizer import build_optimizer
from torch.utils.tensorboard import SummaryWriter
from dywsss.tool.torch_utils import *

cudnn.enabled = True
import timm
import os
from dywsss.tool.load_pretrain import load_checkpoint


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

def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
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

    train_dataset = dywsss.tool.data.VOC12ClsDataset(args.train_list,
                                                     voc12_root=args.voc12_root,
                                                     # pseudo_gt=args.pseudo_list,
                                                     transform=transforms.Compose([
                                                  imutils.RandomResizeLong(
                                                      448, 768),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ColorJitter(
                                                      brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                                  np.asarray,
                                                  normalize,
                                                  imutils.RandomCrop(
                                                      args.crop_size),
                                                  imutils.HWC_to_CHW,
                                                  torch.from_numpy,
                                              ])
                                                     )

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = dywsss.tool.data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
                                                   transform=transforms.Compose([
                                                np.asarray,
                                                normalize,
                                                imutils.CenterCrop(args.crop_size),
                                                imutils.HWC_to_CHW,
                                                torch.from_numpy,
                                            ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model = timm.create_model(args.network, pretrained=True, num_classes=20)
    if args.pretrain is not None:
        # path_pretrain = os.path.join('pretrain/', args.pretrain)
        print(f'Loading Pretrain Model from {args.pretrain}')
        load_checkpoint(model, args.pretrain, strict=False)

    # model = vit_base_patch16_224(num_classes=20)
    if args.finetune is True:
        print('freeze all layers but the last head')
        # freeze all layers but the last fc
        linear_keyword = 'head'
        for name, param in model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    writer = SummaryWriter(args.tensorboard_dir)
    optimizer = build_optimizer(cfg=args, model=model)
    scheduler = build_scheduler(args, optimizer)
    global_step = 0
    best_valid_loss = 99999999

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):

            img = pack[1]
            label = pack[2].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if (global_step-1) % 50 == 0:
                timer.update_progress(global_step / max_step)

                print('Iter:%5d/%5d' % (global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter+1) * args.batch_size /
                                     timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            writer.add_scalar('Train/loss', loss, global_step)
            writer.add_scalar('Train/learning_rate', learning_rate, global_step)

        valid_loss = validate(model, val_data_loader)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.module.state_dict(), os.path.join(args.model_dir, 'best.pth'))

        torch.save(model.module.state_dict(), os.path.join(args.model_dir, f'{ep}_{valid_loss}.pth'))
        writer.add_scalar('Evaluation/valid_loss', valid_loss, global_step)
        writer.add_scalar('Evaluation/best_valid_loss', best_valid_loss, global_step)
        timer.reset_stage()
        scheduler.step()

    torch.save(model.module.state_dict(), os.path.join(args.model_dir, 'last.pth'))
