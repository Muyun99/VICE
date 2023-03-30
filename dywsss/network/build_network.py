import timm
import torch

def build_model(cfg):
    if 'resnet38' in cfg.network:
        model = ''
    else:
        model = timm.create_model(cfg.network, num_classes=20, pretrained=True)
    if cfg.pretrain is None:
        print(f'Using Pytorch weight')
    else:
        print(f'Using {cfg.pretrain} weight')
        if 'pretrain' in cfg.pretrain:
            linear_keyword = 'fc'
            checkpoint = torch.load(cfg.pretrain, map_location="cpu")

            state_dict = checkpoint['state_dict']
            # for k in list(state_dict.keys()):
            #     # retain only base_encoder up to before the embedding layer
            #     if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            #         # remove prefix
            #         state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            #     # delete renamed or unused k
            #     del state_dict[k]

            # for DINO_VOC
            # state_dict = checkpoint['teacher']
            # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            # msg = model.load_state_dict(torch.load(cfg.pretrain)['state_dict'], strict=False)
        else:
            msg = model.load_state_dict(torch.load(cfg.pretrain), strict=False)
        # msg = model.load_state_dict(torch.load(cfg.pretrain), strict=False)
        print(msg)

    return model
