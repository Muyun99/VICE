import pandas as pd
import numpy as np
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import os

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def plot_VICE(args, num_bin, VICE):
    df = pd.read_csv(args.dir_csv)
    df.sort_values(by=['loss'], inplace=True)

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    df_loss = np.array(df['loss'].tolist())
    df_miou = np.array(df['miou'].tolist())

    split_loss = list(split(df_loss, num_bin))
    split_miou = list(split(df_miou, num_bin))

    list_loss_bin = []
    list_miou_bin = []
    for idx in range(num_bin):
        list_loss_bin.append(split_loss[idx].mean())
        list_miou_bin.append(split_miou[idx].mean())

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        # 'font.style': 'italic',
        'font.weight': 'heavy',  # or 'bold'
        # 'font.size': 'medium',#or large,small
    })
    # matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 1)
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.set_xlabel(r'\textbf{The index of loss bin}', fontsize=20)
    ax2.set_ylabel(r'\textbf{JI score (\%)/ Loss}', fontsize=20)
    ax2.set_title(r'\textbf{%s}' % args.plot_title, fontsize=20)
    ax2.set_ylim((0,0.8))

    plt.plot(list(range(num_bin + 1))[1:], list_miou_bin, color='#2776B7', label=r'\textbf{JI Score}')
    plt.bar(list(range(num_bin + 1))[1:], list_loss_bin, width=1 / num_bin, color='#F8AC8C', edgecolor='#F8AC8C',
            label=r'\textbf{Loss}')

    ax2.legend(loc=1)

    import matplotlib.patches as patches
    ax2.add_patch(
        patches.Rectangle(
            (0.0, 0.15),  # (x,y)
            135, 0.08,  # width and height
            # You can add rotation as well with 'angle'
            alpha=0.7, facecolor="white", edgecolor="lightgray", linewidth=3, linestyle='solid')
    )
    ax2.text(0.07, 0.21, r"{$\mathbf {VICE}$}" + r"\textbf{ = %.3f}" % round(VICE, 3), color="black",
             ha="left", va="bottom", transform=ax2.transAxes, fontsize=17)

    # ax2.text(0.07, 0.21, r"{$\mathbf {ECE_{ML}}$}" + r"\textbf{ = %.3f}" % round(VICE, 3), color="black",
    #          ha="left", va="bottom", transform=ax2.transAxes, fontsize=18)



    # plt.show()
    if os.path.exists('Fig/fig_VICE') == False:
        os.makedirs('Fig/fig_VICE')
    plt.savefig(f'Fig/fig_VICE/VICE_{args.plot_title}.png', dpi=1500, format='png')

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

def compute_pearson(dir_csv):
    df = pd.read_csv(dir_csv)
    df_loss = np.array(df['loss'].tolist())
    df_miou = np.array(df['miou'].tolist())

    pccs = np.corrcoef(-1 * df_loss, df_miou)
    VICE = round(pccs[0][1], 3)
    print(f'{dir_csv} Pearson 相关系数为 {round(pccs[0][1], 3)}')

    KL = scipy.stats.entropy(df_loss, df_miou)
    print(f'{dir_csv} miou 相对于 loss 的 KL 散度为 {round(KL, 3)}')

    # KL = scipy.stats.entropy(df_miou, df_loss)
    # print(f'{dir_csv} loss 相对于 miou 的 KL 散度为 {KL}')

    JS = JS_divergence(df_loss, df_miou)
    print(f'{dir_csv} JS 散度为 {round(JS, 3)}')

    writelog('VICE_log.txt', {'KL_散度': round(KL, 3), 'JS 散度': round(JS, 3), 'Pearson 系数': round(pccs[0][1], 3)}, args.plot_title)

    return VICE



def parse_args():
    parser = argparse.ArgumentParser(description='Compute corr')
    parser.add_argument('--dir_csv', required=True)
    parser.add_argument('--plot_title', type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    VICE = compute_pearson(args.dir_csv)
    plot_VICE(args, 300, VICE)