import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
def visualize_training_log(csv_file_path, outpath='./'):
    # ==========================================
    # 1. 参数配置 (Configuration) - 可在此处调整
    # ==========================================
    X_AXIS_COL = 'Step'       # X轴显示的数据列: 'Step' 或 'Epoch'
    
    # Y轴显示模式: 'linear' (线性) 或 'log' (对数)
    # 当Loss变化非常大(如从30000降到10)时，建议设为 'log'
    LOSS_Y_SCALE = 'log'   
    
    # X轴刻度间隔 (设置 None 为自动，设置数字则强制指定间隔)
    # 例如：设为 140 则每 140 个 step 显示一个刻度
    X_TICK_SPACING = None     
    
    # 图片大小 (宽, 高)
    FIG_SIZE = (10, 8)

    # ==========================================
    # 2. 读取与处理数据
    # ==========================================
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取数据，共 {len(df)} 行。")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # ==========================================
    # 3. 绘图逻辑
    # ==========================================
    # 创建 2 行 1 列的子图，共享 X 轴
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)

    # --- 绘制 Loss (子图 1) ---
    ax_loss.plot(df[X_AXIS_COL], df['Train Loss'], label='Train Loss', 
                 color='#1f77b4', linewidth=1, marker='o', markersize=2, alpha=0.8)
    ax_loss.plot(df[X_AXIS_COL], df['Val Loss'], label='Val Loss', 
                 color='#ff7f0e', linewidth=1, marker='s', markersize=2, alpha=0.8)
    
    ax_loss.set_ylabel('Loss Value')
    ax_loss.set_title(f'Training & Validation Loss (Scale: {LOSS_Y_SCALE})')
    ax_loss.legend(loc='upper right')
    ax_loss.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 设置 Loss 轴的 Scale (线性或对数)
    if LOSS_Y_SCALE == 'log':
        ax_loss.set_yscale('log')
    
    # --- 绘制 Learning Rate (子图 2) ---
    ax_lr.plot(df[X_AXIS_COL], df['Learning Rate'], label='Learning Rate', 
               color='#2ca02c', linestyle='--', linewidth=2)
    
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_xlabel(X_AXIS_COL)
    ax_lr.grid(True, linestyle='--', alpha=0.5)
    ax_lr.legend(loc='upper right')

    # --- X轴 刻度调整 ---
    if X_TICK_SPACING:
        ax_lr.xaxis.set_major_locator(ticker.MultipleLocator(X_TICK_SPACING))
    
    # 防止科学计数法导致 X 轴标签重叠或不直观（如果Step数值很大）
    ax_lr.ticklabel_format(style='plain', axis='x', useOffset=False)

    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'training_log_visualization.png'))


# 运行函数

csv_file = r"/home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear_joint/training_metrics.csv"
outpath = r"/home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear_joint/"
visualize_training_log(csv_file, outpath=outpath)