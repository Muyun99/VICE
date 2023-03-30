import os.path

import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dywsss.tool.data
from dywsss.network.build_network import build_model
from dywsss.network.dlinknet import UNet_ResNet50
from dywsss.tool import imutils
from dywsss.tool import pyutils


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


def show_reconstruction(args):
    normalize = Normalize()
    model_resnet = build_model(args)
    model = UNet_ResNet50(resnet_model=model_resnet)
    model.load_state_dict(torch.load(args.weights))
    print(f'Loading weights from {args.weights}')
    model.eval()
    model.cuda()

    dywsss.tool.data.NUM_CLS = args.num_cls
    dywsss.tool.data.IMG_FOLDER_NAME = args.img_dir
    dywsss.tool.data.ANNOT_FOLDER_NAME = args.gt_dir
    dywsss.tool.data.CLS_LABEL = args.cls_label
    infer_dataset = dywsss.tool.data.VOC12ClsDataset(args.infer_list, voc12_root=args.voc12_root,
                                                     transform=transforms.Compose([
                                                         np.asarray,
                                                         normalize,
                                                         imutils.CenterCrop(224),
                                                         imutils.HWC_to_CHW,
                                                         torch.from_numpy,
                                                     ]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    crf_alpha = [int(i) for i in args.crf_threshs.split('_')]
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    print("gpu num: " + str(n_gpus), flush=True)

    for iter, pack in enumerate(infer_data_loader):
        img = pack[1].cuda(non_blocking=True)
        label = pack[2].cuda(non_blocking=True)
        cls_result, restruction_result = model(img)

        # img = restruction_result[0].permute(2,1,0).cpu().detach().numpy()
        # print(restruction_result.shape)
        img_original = img[0].permute(1, 2, 0).cpu().detach().numpy()
        img_reconstruction = restruction_result[0].permute(1, 2, 0).cpu().detach().numpy()

        pyutils.show_two_figs(
            img1=img_original,
            img2=img_reconstruction,
            tag1='original',
            tag2='reconstruction',
            dir_save=os.path.join(args.reconstrcution_dir, f'{iter}.png'))

        # mmcv.imsave()
        # mmcv.imshow(img_original)
        # mmcv.imshow(img_reconstruction)
