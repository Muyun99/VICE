import timm
import torch
from thop import profile
model = timm.create_model('efficientnet_b4')
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))

from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")

print(macs, params)