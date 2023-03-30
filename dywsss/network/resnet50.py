import timm


def get_model():

    model = timm.create_model('resnet50', pretrained=False, num_classes=20)
    model.forward_features()



if __name__ == '__main__':
    get_model()