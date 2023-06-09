import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
from network.scalenet import ScaleNet, SABlock

class Net(ScaleNet):
    def __init__(self, block, layers, structure, dilations=(1,1,1,1), num_cls=21):
        super(Net, self).__init__(block, layers, structure, dilations)
        self.fc8 = nn.Conv2d(2048, num_cls, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]
        self.not_training = [self.conv1, self.bn1, self.layer1]

        print('Total params: %.2fM' % (sum(p.numel() for p in self.parameters())/1000000.0), flush=True)

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward(x)
        cam = self.fc8(d['x4'])
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['x2'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['x3'].detach()), inplace=True)
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n,c,h,w = f.size()

        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return cam, cam_rv

    def PCM(self, cam, f):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


def ScaleNet50_SEAM(structure_path, ckpt=None, dilations=(1,1,1,1), num_cls=21, **kwargs):
    layer = [3, 4, 6, 3]
    structure = json.loads(open(structure_path).read())
    model = Net(SABlock, layer, structure, dilations, num_cls, **kwargs)

    # pretrained
    if ckpt != None:
        state_dict = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    return model


def ScaleNet101_SEAM(structure_path, ckpt=None, dilations=(1,1,1,1), num_cls=21, **kwargs):
    layer = [3, 4, 23, 3]
    structure = json.loads(open(structure_path).read())
    model = Net(SABlock, layer, structure, dilations, num_cls, **kwargs)

    # pretrained
    if ckpt != None:
        state_dict = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    return model

