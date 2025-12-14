import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

def plot_comparison():
    # 1. 读取数据
    df1 = pd.read_csv(r'/data/nas-shared/lijy/Embodied AI/training_metrics_1.csv')
    df2 = pd.read_csv(r'/home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_cosine_2/training_metrics.csv')

    # 2. 论文风全局参数（不影响你手动指定的线条颜色）
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # 字体与字号
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,

        # 统一文字颜色（坐标轴标题、刻度、标题、图例等）
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        # 线宽/边框更适合论文
        "axes.linewidth": 1.1,

        # 导出
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # 矢量字体（避免 PDF 里文字变成 Type3）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- 图 1: 全程训练概览 (Overview) ---
    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss (MSE)')  # label 强制黑色（由 rcParams 保证）

    # Exp 1 曲线（颜色不动）
    l1, = ax1.plot(df1['Epoch'], df1['Val Loss'],
                   color='salmon', alpha=0.45, linewidth=1.2,
                   label='Exp 1: Aggressive Decay')
    l1_smooth, = ax1.plot(df1['Epoch'], df1['Val Loss'].rolling(window=10, min_periods=1).mean(),
                          color='red', linewidth=2.2,
                          label='Exp 1 (Smoothed)')

    # Exp 2 曲线（颜色不动）
    l2, = ax1.plot(df2['Epoch'], df2['Val Loss'],
                   color='skyblue', alpha=0.45, linewidth=1.2,
                   label='Exp 2: Sustained LR')
    l2_smooth, = ax1.plot(df2['Epoch'], df2['Val Loss'].rolling(window=10, min_periods=1).mean(),
                          color='blue', linewidth=2.2,
                          label='Exp 2 (Smoothed)')

    # log scale + 更清晰的主/次刻度
    ax1.set_yscale('log')
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax1.yaxis.set_minor_formatter(NullFormatter())

    # 网格更克制（不抢线条）
    ax1.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.30)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.18)

    # 双轴：Learning Rate（线条颜色不动，但文字统一黑色）
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')  # 黑色
    ax2.plot(df1['Epoch'], df1['Learning Rate'],
             color='green', linestyle='--', alpha=0.55, linewidth=1.4,
             label='LR Exp 1')
    ax2.plot(df2['Epoch'], df2['Learning Rate'],
             color='lime', linestyle=':', alpha=0.85, linewidth=1.6,
             label='LR Exp 2')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax2.yaxis.set_minor_formatter(NullFormatter())

    # 强制两个 y 轴刻度颜色为黑色（避免你之前 tick_params(labelcolor=...) 的彩色）
    ax1.tick_params(axis='both', colors='black')
    ax2.tick_params(axis='y', colors='black')

    # 合并图例（包含双轴线条），并把 legend 字体/边框做成论文风
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(handles1 + handles2, labels1 + labels2,
                     loc='upper right', frameon=True, fancybox=False,
                     framealpha=0.95, edgecolor='black')
    for t in leg.get_texts():
        t.set_color('black')

    ax1.set_title('Training Dynamics Comparison: Loss & Learning Rate')

    ax1.set_xlim(0,500)
    ax2.set_xlim(0,500)

    # 导出 PNG + PDF（论文更推荐 PDF/SVG）
    fig.savefig('fig_training_overview.png', dpi=300)
    # fig.savefig('fig_training_overview.pdf')
    print("Saved fig_training_overview.png / .pdf")

    # --- 图 2: 后期细节放大 (Zoomed-in) ---
    start_epoch = 200
    df1_zoom = df1[df1['Epoch'] >= start_epoch].copy()
    df2_zoom = df2[df2['Epoch'] >= start_epoch].copy()

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)

    ax.plot(df1_zoom['Epoch'], df1_zoom['Val Loss'], color='red', alpha=0.22, linewidth=1.0)
    ax.plot(df1_zoom['Epoch'], df1_zoom['Val Loss'].rolling(window=5, min_periods=1).mean(),
            color='red', linewidth=2.2, label='Exp 1 (Aggressive Decay)')

    ax.plot(df2_zoom['Epoch'], df2_zoom['Val Loss'], color='blue', alpha=0.22, linewidth=1.0)
    ax.plot(df2_zoom['Epoch'], df2_zoom['Val Loss'].rolling(window=5, min_periods=1).mean(),
            color='blue', linewidth=2.2, label='Exp 2 (Sustained LR)')

    # 标注最低点（颜色不动）
    min_loss_1 = df1_zoom['Val Loss'].min()
    min_epoch_1 = df1_zoom.loc[df1_zoom['Val Loss'].idxmin(), 'Epoch']
    ax.scatter(min_epoch_1, min_loss_1, color='darkred', marker='o', s=40, zorder=5)
    ax.annotate(f'Min: {min_loss_1:.1f}',
                xy=(min_epoch_1, min_loss_1),
                xytext=(6, 8), textcoords='offset points',
                fontsize=10, color='black',
                arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

    min_loss_2 = df2_zoom['Val Loss'].min()
    min_epoch_2 = df2_zoom.loc[df2_zoom['Val Loss'].idxmin(), 'Epoch']
    ax.scatter(min_epoch_2, min_loss_2, color='darkblue', marker='x', s=55, zorder=5)
    ax.annotate(f'Min: {min_loss_2:.1f}',
                xy=(min_epoch_2, min_loss_2),
                xytext=(6, -14), textcoords='offset points',
                fontsize=10, color='black',
                arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.set_title(f'Late-Stage Convergence Analysis (Epoch {start_epoch}+)')

    leg = ax.legend(frameon=True, fancybox=False, framealpha=0.95, edgecolor='black')
    for t in leg.get_texts():
        t.set_color('black')

    ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.30)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.6, alpha=0.18)
    ax.tick_params(axis='both', colors='black')

    fig.savefig('fig_training_zoom.png', dpi=300)
    # fig.savefig('fig_training_zoom.pdf')
    print("Saved fig_training_zoom.png / .pdf")

if __name__ == "__main__":
    plot_comparison()
