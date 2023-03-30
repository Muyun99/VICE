import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ECE_plot(df, n_bins=10):
    """
    n_bins (int): number of confidence interval bins
    """
    # bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_boundaries = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    loss = np.array(df['loss'].tolist())
    miou = np.array(df['miou'].tolist())
    # plt.scatter(loss, miou, alpha=0.5, c='darkcyan', marker='+')
    fig, axes = plt.subplots(1, len(bin_lowers))

    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (bin_lower < loss) & (loss < bin_upper)
        loss_in_bin = loss[in_bin]
        miou_in_bin = miou[in_bin]

        color = np.random.rand(len(loss_in_bin))
        axes[idx].set_yscale("log")
        # axes[idx].set_xlabel("Loss")
        # axes[idx].set_ylabel("CAM Quality")
        # axes[idx].set_title("Logarithmic scale (y)")
        try:
            axes[idx].scatter(loss_in_bin, miou_in_bin,
                              alpha=0.5, c=color, marker='+')
        except:
            continue

        # plt.scatter(loss_in_bin, miou_in_bin, alpha=0.5, c=color, marker='+')
    plt.show()
    # plt.xlabel('Loss')
    # plt.ylabel('CAM Quality')
    plt.show()


def ECE_plot2(df, n_bins=10):
    """
    n_bins (int): number of confidence interval bins
    """
    # bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_boundaries = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    loss = np.array(df['loss'].tolist())
    miou = np.array(df['miou'].tolist())
    # plt.scatter(loss, miou, alpha=0.5, c='darkcyan', marker='+')
    fig, axes = plt.subplots(1, len(bin_lowers))

    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (bin_lower < loss) & (loss < bin_upper)
        loss_in_bin = loss[in_bin]
        miou_in_bin = miou[in_bin]

        color = np.random.rand(len(loss_in_bin))
        axes[idx].set_yscale("log")
        # axes[idx].set_xlabel("Loss")
        # axes[idx].set_ylabel("CAM Quality")
        # axes[idx].set_title("Logarithmic scale (y)")
        try:
            axes[idx].scatter(loss_in_bin, miou_in_bin,
                              alpha=0.5, c=color, marker='+')
        except:
            continue

        # plt.scatter(loss_in_bin, miou_in_bin, alpha=0.5, c=color, marker='+')
    plt.show()
    # plt.xlabel('Loss')
    # plt.ylabel('CAM Quality')
    plt.show()


if __name__ == '__main__':
    df_miou_loss = pd.read_csv(
        'miou_loss_csv/miou_loss_resnest101e_PyTorch_pretrain_multi_epoch_lr_0.01.csv')
    ECE_plot(df=df_miou_loss)
