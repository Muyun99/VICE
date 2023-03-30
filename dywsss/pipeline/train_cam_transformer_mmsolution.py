
import torch.nn.functional as F
from dywsss.tool import pyutils, imutils
import dywsss.tool.data
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.backends import cudnn

cudnn.enabled = True
import timm
import os
from dywsss.tool.load_pretrain import load_checkpoint
import nni
from nni.utils import merge_parameter
import logging

logger = logging.getLogger('mnist_AutoML')

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

    print('loss:', val_loss_meter.pop('loss'))

    return


def train(args):
    pyutils.Logger(os.path.join(args.log_dir, args.session_name) + '.log')
    normalize = Normalize()
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
        path_pretrain = os.path.join('/pretrain/', args.pretrain)
        print(f'Loading Pretrain Model from {path_pretrain}')
        load_checkpoint(model, path_pretrain, strict=False)
    # model = vit_base_patch16_224(num_classes=20)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    init_lr = args.lr * args.batch_size / 256
    init_lr = 0.1
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=0.9,
                                weight_decay=args.wt_dec)
    global_step = 0

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

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.session_name + '.pth'))


def train_nni():
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(get_params(), tuner_params))
    print(params)

if __name__ == '__main__':
    train_nni()