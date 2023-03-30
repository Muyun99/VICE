
import timm
from dywsss.network.vit_LRP import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
# from dywsss.network.vit_new import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
def build_model(model_name, num_classes=20, pretrained=False):
    model = None

    if model_name == 'vit_base_patch16_224':
        # model = timm.create_model(model_name='vit_base_patch16_224', num_classes=20, pretrained=True).cuda()
        model = vit_base_patch16_224(num_classes=20, pretrained=False).cuda()
    elif model_name == 'vit_large_patch16_224':
        model = vit_large_patch16_224(num_classes=20, pretrained=False).cuda()
    elif model_name == 'vit_small_patch16_224':
        model = vit_small_patch16_224(num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_base_patch4_window7_224':
        model = timm.create_model('swin_base_patch4_window7_224', num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_base_patch4_window12_384':
        model = timm.create_model('swin_base_patch4_window12_384', num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_large_patch4_window7_224':
        model = timm.create_model('swin_large_patch4_window7_224', num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_large_patch4_window12_384':
        model = timm.create_model('swin_large_patch4_window12_384', num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_base_patch4_window7_224_in22k':
        model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=20, pretrained=False).cuda()
    elif model_name == 'swin_base_patch4_window12_384_in22k':
        model = timm.create_model('swin_base_patch4_window12_384_in22k', num_classes=20, pretrained=False).cuda()
    if model is None:
        raise Exception('Create Model Error!')
    return model
