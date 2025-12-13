import torch
from torchinfo import summary
import argparse

# 导入你的项目模块
from config import Config
from model.unet import UNet_DDPM

def print_model_table():
    # 1. 初始化配置和模型
    config = Config()
    device = torch.device('cpu') # 打印表格用 CPU 即可
    config.device = device
    
    print("正在初始化模型...")
    model = UNet_DDPM(config).to(device)
    
    # 2. 准备假输入 (Dummy Inputs)
    # UNet_DDPM 的 forward 函数签名是: forward(self, x, y, cond=None, ...)
    # 我们必须严格按照这个顺序提供输入列表
    
    B = 1
    C = config.data.channels
    H = config.data.image_size
    W = config.data.image_size
    
    n_pred = config.data.num_frames          # 10
    n_cond = config.data.num_frames_cond     # 5
    
    # 输入 1: x (Noisy Future Frames)
    dummy_x = torch.randn(B, n_pred * C, H, W, device=device)
    
    # 输入 2: y (Timestep labels) - 必须是 Long 类型
    dummy_y = torch.randint(0, 1000, (B,), device=device)
    
    # 输入 3: cond (Condition Frames)
    dummy_cond = torch.randn(B, n_cond * C, H, W, device=device)

    # 3. 使用 torchinfo 生成表格
    # input_data 接受一个列表，对应 forward 函数的参数位置
    print("\n" + "="*80)
    print(" MCVD Model Summary")
    print("="*80)
    
    model_stats = summary(
        model,
        input_data=[dummy_x, dummy_y, dummy_cond], 
        col_names=["input_size", "output_size", "num_params", "mult_adds"], # 显示输入/输出尺寸、参数量、乘加运算量
        col_width=20,       # 调整列宽
        depth=3,            # 深度：3 通常能看到 ResBlock 级别，最适合展示
        device="cpu"
    )
    
    # 4. 如果你想把结果保存到文件里（方便复制到论文）
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model_stats))
    
    print(f"\n表格已保存到 model_summary.txt")
    print(f"总参数量 (Total Params): {model_stats.total_params:,}")
    print(f"总计算量 (Total Mult-Adds/FLOPs): {model_stats.total_mult_adds:,}")

if __name__ == "__main__":
    print_model_table()