import torch
import torch.onnx
import argparse
import os
import logging

# 导入你项目中的模块
from config import Config
from model.unet import UNet_DDPM

def export_to_onnx(ckpt_path=None, output_path="mcvd_model.onnx"):
    # 1. 初始化配置
    config = Config()
    device = torch.device('cpu') # 导出时建议使用 CPU，避免显存问题
    config.device = device
    
    # 2. 初始化模型
    print("正在初始化模型...")
    model = UNet_DDPM(config).to(device)
    
    # 3. (可选) 加载权重
    # 虽然用于可视化网络结构不需要真实权重，但如果以后想用ONNX推理，则需要加载
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"正在加载权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # 处理 DataParallel 的 module 前缀
        state_dict = checkpoint['model_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()

    # 4. 准备 Dummy Input (假输入)
    # ONNX 需要运行一次模型来追踪计算图，所以需要构造符合形状的随机输入
    B = 1 # Batch size 固定为 1 方便可视化
    C = config.data.channels
    H = config.data.image_size
    W = config.data.image_size
    
    n_pred = config.data.num_frames          # 预测帧数 (e.g., 10)
    n_cond = config.data.num_frames_cond     # 条件帧数 (e.g., 5)
    
    # 输入 1: Noisy Future Frames (x)
    # Shape: [B, 10*1, 64, 64]
    dummy_x = torch.randn(B, n_pred * C, H, W, device=device)
    
    # 输入 2: Timestep Labels (y)
    # Shape: [B]
    # 模拟随机的一个时间步
    dummy_y = torch.tensor([10], dtype=torch.long, device=device)
    
    # 输入 3: Condition Frames (cond)
    # Shape: [B, 5*1, 64, 64]
    dummy_cond = torch.randn(B, n_cond * C, H, W, device=device)

    print(f"输入形状:\n - x: {dummy_x.shape}\n - y: {dummy_y.shape}\n - cond: {dummy_cond.shape}")

    # 5. 执行导出
    print(f"正在导出 ONNX 模型到: {output_path} ...")
    
    # input_names 和 output_names 让你在 Netron 里能看懂每个节点是什么
    input_names = ["noisy_future_frames", "timestep_label", "past_condition_frames"]
    output_names = ["predicted_noise"]
    
    torch.onnx.export(
        model,
        (dummy_x, dummy_y, dummy_cond), # 模型的输入参数，必须是 Tuple
        output_path,
        export_params=True,        # 导出训练好的参数权重
        opset_version=12,          # Opset 版本，11 或 12 通常兼容性最好
        do_constant_folding=True,  # 优化常量折叠
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={             # 允许 Batch Size 动态变化
            'noisy_future_frames': {0: 'batch_size'},
            'timestep_label': {0: 'batch_size'},
            'past_condition_frames': {0: 'batch_size'},
            'predicted_noise': {0: 'batch_size'}
        }
    )
    
    print("✅ 导出成功！")
    print(f"请访问 https://netron.app 并打开 {output_path} 进行可视化。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='(Optional) Path to checkpoint file')
    parser.add_argument('--output', type=str, default='mcvd_structure.onnx', help='Output ONNX filename')
    args = parser.parse_args()
    
    export_to_onnx(args.ckpt, args.output)