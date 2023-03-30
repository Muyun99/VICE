import os
from collections import OrderedDict
import torch
import logging

from dywsss.network.vit_LRP import vit_base_patch16_224

_logger = logging.getLogger(__name__)


def load_state_dict_MoCov3(checkpoint_path, use_ema=False):
    linear_keyword = 'head'
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(
            state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_state_dict_DINO(checkpoint_path, use_ema=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    #     print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    #     state_dict = state_dict[args.checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_state_dict_DINO_VOC(checkpoint_path, use_ema=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")['teacher']
    # if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    #     print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    #     state_dict = state_dict[args.checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_state_dict_MAE(checkpoint_path, use_ema=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    # state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_state_dict_BEiT(checkpoint_path, use_ema=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    print("Expand the shared relative position embedding to each transformer block. ")

    # rel_pos_bias = state_dict["rel_pos_bias.relative_position_bias_table"]
    # num_layers = model.get_num_layers()
    # for i in range(num_layers):
    #     state_dict["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

    # state_dict.pop("rel_pos_bias.relative_position_bias_table")

    return state_dict

def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    linear_keyword = 'head'
    state_dict = None
    if 'MoCo' in checkpoint_path:
        state_dict = load_state_dict_MoCov3(checkpoint_path)
    elif 'DINO_VOC' in checkpoint_path:
        state_dict = load_state_dict_DINO_VOC(checkpoint_path)
    elif 'DINO'in checkpoint_path:
        state_dict = load_state_dict_DINO(checkpoint_path)
    elif 'MAE'in checkpoint_path:
        state_dict = load_state_dict_MAE(checkpoint_path)
    elif 'BEiT'in checkpoint_path:
        state_dict = load_state_dict_BEiT(checkpoint_path)

    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)
    # assert set(msg.missing_keys) == {"%s.weight" %linear_keyword, "%s.bias" % linear_keyword}



def test():
    # # ViT-small ImageNet pretrain
    # model_vit_small_ImageNet_pretrain = timm.create_model(
    #     'vit_small_patch16_224', pretrained=True)
    #
    # # ViT-small MoCov3 pretrain
    # model_vit_small_MoCov3_pretrain = vit_small()
    # checkpoint_path_small_MoCov3 = '/home/muyun99/github/CVPR/pipeline/reference/train_cls/self-supervised/mocov3/vit-s-300ep.pth.tar'
    # load_checkpoint(model_vit_small_MoCov3_pretrain,
    #                 checkpoint_path_small_MoCov3, strict=False)
    #
    # # ViT-base ImageNet pretrain
    # model_vit_base_ImageNet_pretrain = timm.create_model(
    #     'vit_base_patch16_224', pretrained=True)

    # ViT-base MoCov3 pretrain
    model_vit_base= vit_base_patch16_224(num_classes=20, pretrained=True).cuda()
    # model_vit_base_MoCov3_pretrain = vit_base()
    checkpoint_path_base_MoCov3 = 'pretrain/MoCov3/vit-b-300ep.pth.tar'
    state_dict_MoCov3 = load_state_dict_MoCov3(checkpoint_path_base_MoCov3)

    checkpoint_path_base_DINO2 = 'pretrain/DINO/dino_vitbase16_pretrain.pth'
    state_dict_DINO2 = load_state_dict_DINO(checkpoint_path_base_DINO2)

    checkpoint_path_base_MAE = 'pretrain/MAE/pretrain_mae_vit_base_mask_0.75_400e.pth'
    state_dict_MAE = load_state_dict_MAE(checkpoint_path_base_MAE)

    checkpoint_path_base_BEiT1 = 'pretrain/BEiT/beit_base_patch16_224_pt22k.pth'
    state_dict_BEiT1 = load_state_dict_BEiT(checkpoint_path_base_BEiT1)


    checkpoint_path_base_BEiT2 = 'pretrain/BEiT/beit_base_patch16_224_pt22k_ft22k.pth'
    state_dict_BEiT2 = load_state_dict_BEiT(checkpoint_path_base_BEiT2)

    # load_checkpoint(model_vit_base, checkpoint_path_base_MoCov3, strict=False)
    # print('Done')
    # load_checkpoint(model_vit_base, checkpoint_path_base_DINO2, strict=False)
    # print('Done')
    # load_checkpoint(model_vit_base, checkpoint_path_base_MAE, strict=False)
    # print('Done')
    load_checkpoint(model_vit_base, checkpoint_path_base_BEiT1, strict=False)
    print('Done')
    # load_checkpoint(model_vit_base, checkpoint_path_base_BEiT2, strict=False)
    # print('Done')




    # # freeze all layers but the last fc
    # for name, param in model_vit_small.named_parameters():
    #     if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
    #         param.requires_grad = False
if __name__ == '__main__':
    test()