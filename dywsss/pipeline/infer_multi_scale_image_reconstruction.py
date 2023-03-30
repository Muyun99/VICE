import numpy as np
import torch
import os
import dywsss.tool.data
from torch.utils.data import DataLoader
import torchvision
from dywsss.tool import pyutils, imutils
from dywsss.network.build_network import build_model
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from dywsss.network.dlinknet import UNet_ResNet50


def forward_cam(model, img):
    # model.fc.weight (20,2048)
    # feature (1,2048,6,8)

    feature = model.resnet_model.forward_features(img)
    weight = model.resnet_model.fc.weight
    weight = torch.unsqueeze(weight, 2)
    weight = torch.unsqueeze(weight, 2)

    cam = F.conv2d(feature, weight)
    cam = F.relu(cam)
    return cam

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

def infer_multi_scale(args):
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
    infer_dataset = dywsss.tool.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                        # pseudo_gt=args.pseudo_list,
                                                        scales=[0.5, 1.0, 1.5, 2.0],
                                                        inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                     normalize,
                                                     imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    crf_alpha = [int(i) for i in args.crf_threshs.split('_')]
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    print("gpu num: " + str(n_gpus), flush=True)

    for iter, (img_name, img_list, label) in tqdm(enumerate(infer_data_loader)):
        img_name = img_name[0]; label = label[0]

        img_path = dywsss.tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path).convert('RGB'))
        orig_img_size = orig_img.shape[:2]

        # img = restruction_result[0].permute(2,1,0).cpu().detach().numpy()
        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam = forward_cam(model_replicas[i % n_gpus], img.cuda())
                    # cam = model_replicas[i % n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(
            _work,
            list(enumerate(img_list)),
            batch_size=12,
            prefetch_size=0,
            processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(args.num_cls - 1):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        def _crf_with_alpha(cam_dict, alpha, num_cls):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()
            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t, args.num_cls)
                folder = args.out_crf
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)

        if iter % 500 == 0:
            print('iter: ' + str(iter), flush=True)
