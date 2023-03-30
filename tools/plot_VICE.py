import matplotlib.pyplot as plt

def plot_resnest():
    group_resnest = ['resnest14', 'resnest26', 'resnest50', 'resnest101']
    width = [14, 26, 50, 101]
    mAP_resnest = [85.382, 87.395, 88.031, 88.856]
    mAP_resnest = [item * 0.01 for item in mAP_resnest]
    JS_resnest = []
    Acc_resnest = [75.329, 79.170, 79.117, 80.415]
    ML_ECE_resnest = [0.1398947834968567, 0.13247719407081604, 0.1297757476568222, 0.127345010638237]
    VICE_resnest = [0.8496604746867761, 0.841529324137019, 0.9279237191859793, 0.9088535453684295]



    plt.scatter(width, mAP_resnest)
    plt.plot(width, mAP_resnest, label='mAP')

    plt.scatter(width, ML_ECE_resnest)
    plt.plot(width, ML_ECE_resnest, label='ML-ECE')

    plt.xlabel('Width of ResNeSt')
    plt.legend()
    plt.savefig(f'Fig/ResNeSt_ML_ECE.png')
    plt.clf()

    plt.scatter(width, mAP_resnest)
    plt.plot(width, mAP_resnest, label='mAP↑')

    plt.scatter(width, VICE_resnest)
    plt.plot(width, VICE_resnest, label='VICE↓')

    plt.xlabel('Width of ResNeSt')
    plt.legend()
    plt.savefig(f'Fig/ResNeSt_VICE.png')





group_resnet = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
mAP_resnet = [83.917, 86.296, 87.470, 24.984]
JS_resnet = []
Acc_resnet = [72.188, 76.945, 78.617, 6.250]
ML_ECE_resnet = [0.1489696055650711, 0.1425703465938568, 0.13401366770267487, 0.3402189314365387]
VICE_resnet = [0.6737683786971803, 0.9142357403488395, 0.8586332082449737, 0.22972092145482126]

group_vit = ['vit_small', 'vit_base']
mAP_vit = []
JS_vit = []
Acc_vit = []
ML_ECE_vit = []
VICE_vit = [0.8784828458164109, 0.9775964082200969]

def plot_diff_arch():
    group_diff_arch = ['ResNet50', 'Res2Net50', 'ResNeSt50', 'EfficientNet_b4', 'ViT-Small']
    mAP_diff_arch = [87.470, 86.759, 88.031, 83.69, 90.253]
    JS_diff_arch = [47.493, 44.229, 49.395, 43.801, 47.907]
    Acc_diff_arch = [78.617, 77.180, 79.117, 71.180, 81.172]
    ML_ECE_diff_arch = [0.13401366770267487, 0.13296562433242798, 0.1297757476568222, 0.11765245348215103, 0.10360883921384811]
    VICE_diff_arch = [0.8586332082449737, 0.7748880252360156, 0.9279237191859793, 0.8233741176056155, 0.9775964082200969]

    plt.scatter(mAP_diff_arch, ML_ECE_diff_arch)
    plt.scatter(mAP_diff_arch, VICE_diff_arch)
    plt.show()

if __name__ == '__main__':
    # plot_diff_arch()
    plot_resnest()