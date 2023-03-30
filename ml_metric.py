import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
def Accuracy(target, pred, threshold=0.5):
    # 首先把 y_pred 按照阈值给阈值化

    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    count = 0
    for i in range(target.shape[0]):
        p = sum(np.logical_and(target[i], pred[i]))
        q = sum(np.logical_or(target[i], pred[i]))
        count += p / q
    return count / target.shape[0]

def F1Measure(target, pred, threshold=0.5):
    # 首先把 pred 按照阈值给阈值化
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    count = 0
    for i in range(target.shape[0]):
        if (sum(target[i]) == 0) and (sum(pred[i]) == 0):
            continue
        p = sum(np.logical_and(target[i], pred[i]))
        q = sum(target[i]) + sum(pred[i])
        count += (2 * p) / q
    return count / target.shape[0]

def F1Measure_sklearn(target, pred, threshold=0.5):
    # 首先把 pred 按照阈值给阈值化
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    from sklearn.metrics import precision_score, recall_score, f1_score
    # precision = precision_score(y_true=target, y_pred=pred, average='samples')
    # recall = recall_score(y_true=target, y_pred=pred, average='samples')
    f1measure = f1_score(y_true=target, y_pred=pred, average='samples')
    return f1measure

def ECE_loss(target, pred, num_bin, threshold=0.5, network="", save_path="case_show"):
    # target: (N, 20)
    # pred: (N, 20)
    # pred 是概率分布


    bin_boundaries = torch.linspace(0, 1, num_bin + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 计算 confidence
    # 如果最大的 confidence > 0.5，Confidence 就是所有大于 0.5 的 confidence 做平均
    # 如果最大的 confidence < 0.5，Confidence 就是最大的 confidence
    confidences = []
    for i in range(pred.shape[0]):
        if pred[i].max().item() >= 0.5:
            confidences.append(pred[i][pred[i] >= 0.5].mean().item())
        else:
            confidences.append(pred[i].max().item())
    confidences = torch.from_numpy(np.array(confidences))


    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    count = 0
    for i in range(target.shape[0]):
        p = sum(np.logical_and(target[i], pred[i]))
        q = sum(np.logical_or(target[i], pred[i]))
        count += p / q
    accuracy_overall =  count / target.shape[0]

    accuracies = []
    for i in range(pred.shape[0]):
        p = sum(torch.logical_and(target[i], pred[i]))
        q = sum(torch.logical_or(target[i], pred[i]))
        accuracies.append(p / q)
    accuracies = torch.from_numpy(np.array(accuracies))


    ece = torch.zeros(1, device=pred.device)
    accuracy_bin = []
    confidence_bin = []
    count_bin = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

        # prop_in_bin 是当前 bin 中样本数量占总样本数量的比例
        prop_in_bin = in_bin.float().mean()
        count_in_bin = in_bin.float().sum().item()
        count_bin.append(count_in_bin)

        # 如果 bin 中有样本则计算 ML-ECE
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            accuracy_bin.append(accuracy_in_bin)
            confidence_bin.append(confidence_bin)
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_bin.append(0)
            confidence_bin.append(0)
    import matplotlib.font_manager
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        'font.style': 'italic',
        'font.weight': 'heavy',  # or 'bold'
        # 'font.size': 'medium',#or large,small
    })

    fig = plt.figure(figsize=(6, 9))
    gs = GridSpec(3, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(r'\textbf{%s}' % network, fontsize=20)
    ax1.set_ylabel(r'\textbf{Sample frac}.', fontsize=20)
    ax1.bar(bin_uppers, count_bin, width=1 / num_bin, color='#2776B7', edgecolor='white')

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.set_xlabel(r'\textbf{ML-Confidence (\%)}', fontsize=20)
    ax2.set_ylabel(r'\textbf{ML-Accuracy (\%)}', fontsize=20)
    ax2.plot([0.55, 1], [0.55, 1], color='gray', linewidth=4, linestyle="--")
    ax2.bar(bin_uppers[10:], bin_uppers[10:], width=1 / num_bin, color='#2776B7', edgecolor='white',
            label=r'\textbf{Gap}')
    ax2.bar(bin_uppers, accuracy_bin, width=1 / num_bin, color='#7EB8DA', edgecolor='white', label=r'\textbf{Output}')

    ax2.legend()

    import matplotlib.patches as patches
    ax2.add_patch(
        patches.Rectangle(
            (0.0, 0.13),  # (x,y)
            0.52, 0.2,  # width and height
            # You can add rotation as well with 'angle'
            alpha=0.7, facecolor="white", edgecolor="lightgray", linewidth=3, linestyle='solid')
    )
    ax2.text(0.07, 0.15, r"$\mathbf {ECE_{ML}}$" + r"\textbf{ = %.3f}" % round(ece.item() * 100, 3) ,
             color="black",
             ha="left", va="bottom", transform=ax2.transAxes, fontsize=17)
    ax2.text(0.07, 0.23, r"$\mathbf {ACC_{ML}}$" + r"\textbf{ = %.3f}" % round(accuracy_overall.item() * 100, 3),
             color="black",
             ha="left", va="bottom", transform=ax2.transAxes, fontsize=17)

    plt.savefig(os.path.join(save_path, f'{network}_ML_ECE.png'), dpi=1500, format='png')
    # plt.show()
    plt.clf()

    # fig, ax2 = plt.figure(figsize=(6, 6))
    # ax2 = fig.add_subplot(gs[1:, 0])
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 1)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.set_title(r'\textbf{%s}' % network, fontsize=20)
    ax2.set_xlabel(r'\textbf{ML-Confidence (\%)}', fontsize=20)
    ax2.set_ylabel(r'\textbf{ML-Accuracy (\%)}', fontsize=20)

    ax2.plot([0.55, 1], [0.55, 1], color='gray', linewidth=4, linestyle="--")
    ax2.bar(bin_uppers[10:], bin_uppers[10:], width=1 / num_bin, color='#2776B7', edgecolor='white',
            label=r'\textbf{Gap}')
    ax2.bar(bin_uppers, accuracy_bin, width=1 / num_bin, color='#7EB8DA', edgecolor='white', label=r'\textbf{Output}')
    ax2.legend()
    import matplotlib.patches as patches
    ax2.add_patch(
        patches.Rectangle(
            (0.0, 0.13),  # (x,y)
            0.52, 0.2,  # width and height
            # You can add rotation as well with 'angle'
            alpha=0.7, facecolor="white", edgecolor="lightgray", linewidth=3, linestyle='solid')
    )
    # ax2.text(0.07, 0.21, r"{$\mathbf {ECE_{ML}}$}" + r"\textbf{ = %.3f}" % round(CPvVI, 3), color="black",
    #          ha="left", va="bottom", transform=ax2.transAxes, fontsize=18)

    ax2.text(0.07, 0.15, r"$\mathbf {ECE_{ML}}$" + r"\textbf{ = %.3f }" % round(ece.item() * 100, 3) ,
             color="black",
             ha="left", va="bottom", transform=ax2.transAxes, fontsize=17)
    ax2.text(0.07, 0.23, r"$\mathbf {ACC_{ML}}$" + r"\textbf{ = %.3f }" % round(accuracy_overall.item() * 100, 3),
             color="black",
             ha="left", va="bottom", transform=ax2.transAxes, fontsize=17)
    plt.savefig(os.path.join(save_path, f'{network}_ML_ECE_lower_part.png'), dpi=1500, format='png')
    plt.clf()
    
    plt.title(f'{network}', fontsize=20)
    plt.bar(bin_uppers, accuracy_bin, width=1 / num_bin, color='#7EB8DA', edgecolor='white', label='Output')
    plt.xlabel('ML-Confidence', fontsize=20)
    plt.ylabel('ML-Accuracy', fontsize=20)
    plt.savefig(os.path.join(save_path, f'{network}_ML_ECE_true_calibrated.png'), dpi=1500, format='png')

    return ece.item()

