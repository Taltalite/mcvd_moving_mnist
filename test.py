import os
import argparse
import torch
import numpy as np
import imageio
from torchvision.utils import make_grid, save_image
import logging
from tqdm import tqdm
from functools import partial

# Imports from your project
from config import Config
from datasets.moving_mnist import MovingMNIST, data_transform, inverse_data_transform
from model.unet import UNet_DDPM
from model.ema import EMAHelper
from model import ddpm_sampler
from losses.dsm import anneal_dsm_score_estimation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Test MCVD on Moving MNIST")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Where to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of video samples to generate')
    parser.add_argument('--data_path', type=str, help='Path to Moving MNIST data')
    return parser.parse_args()

def save_gif(frames_tensor, path, fps=4):
    """
    frames_tensor: [T, C, H, W] tensor in [0, 1]
    """
    frames = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    imageio.mimsave(path, frames, fps=fps)
    logging.info(f"Saved GIF: {path}")

def calc_test_loss(scorenet, config, x_target, cond):
    """
    计算测试样本的 Loss，用于通过数据对比排查问题
    """
    scorenet.eval()
    # 构造一个 batch (N=1)
    # 调用 dsm loss 计算
    with torch.no_grad():
        # cond_mask 全 1 (不 mask)
        cond_mask = torch.ones(x_target.shape[0], 1, 1, 1).to(x_target.device)
        
        # 为了获得稳定的评估，我们对同一个样本多次采样不同的 t 并求平均
        # 或者覆盖所有 timestep (代价较大)，这里随机采 10 次取平均
        total_loss = 0
        for _ in range(10):
            loss = anneal_dsm_score_estimation(
                scorenet, x_target, labels=None, cond=cond, cond_mask=cond_mask,
                loss_type='a', all_frames=config.model.output_all_frames
            )
            total_loss += loss.item()
        
    return total_loss / 10.0

@torch.no_grad()
def main():
    args = parse_args()
    config = Config()
    device = config.device
    
    # 强制覆盖 Config 以确保一致性 (Linear Schedule)
    config.model.sigma_dist = 'linear'
    config.model.sigma_begin = 1e-4
    config.model.sigma_end = 0.02
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data (Test Set)
    test_dataset = MovingMNIST(root=args.data_path, train=False)
    indices = list(range(args.num_samples))
    subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    logging.info(f"Loading model from {args.ckpt_path}...")
    scorenet = UNet_DDPM(config).to(device)
    scorenet = torch.nn.DataParallel(scorenet)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # --- 重要修改：本次测试暂时禁用 EMA ---
    # 原因：如果训练步数较少(12k)，EMA 可能还包含初始化时的偏差。
    # 我们先看 Base Model 的表现。如果 Base Model 好但 EMA 坏，说明 EMA 衰减率需要调整。
    logging.info("Loading base model state (Skipping EMA for debug)...")
    scorenet.load_state_dict(checkpoint['model_state'])
    scorenet.eval()
    
    # 3. Inference Loop
    for i, (X, _) in enumerate(tqdm(loader, desc="Generating videos")):
        X = X.to(device)
        X = data_transform(config, X) # -> [-1, 1]
        
        n_cond = config.data.num_frames_cond
        n_pred = config.data.num_frames
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        
        # Prepare Data
        cond_frames = X[:, :n_cond] # [1, 5, 1, 64, 64]
        gt_frames = X[:, n_cond : n_cond+n_pred] # [1, 10, 1, 64, 64]
        
        cond_tensor = cond_frames.reshape(1, n_cond*C, H, W)
        x_target = gt_frames.reshape(1, n_pred*C, H, W)
        
        # --- DEBUG 1: 计算 Test Loss ---
        # 这能告诉我们模型是否对当前样本过拟合，或者是否根本没见过这种数据
        test_loss = calc_test_loss(scorenet, config, x_target, cond_tensor)
        print(f"\n[Sample {i}] Test Loss (MSE): {test_loss:.5f}")
        
        if test_loss > 0.05:
            print("WARNING: Test Loss is high! Model generalization issue.")
        elif test_loss < 0.005:
            print("WARNING: Test Loss is suspiciously low! Potential data leakage or background overfitting.")
            
        # --- Sampling ---
        x_init = torch.randn(1, n_pred*C, H, W).to(device)
        
        logging.info(f"Sampling video {i}...")
        
        # 使用 Robust Linear Sampler
        generated_flat = ddpm_sampler(
            x_init, 
            scorenet, 
            cond=cond_tensor, 
            subsample_steps=None, # default 1000
            temperature=1.0,     # 尝试 0.7 或 0.5
            final_only=True, 
            denoise=False,      # 必须 False
            clip_before=True,   # 必须 True (x0 clipping)
            verbose=True        # 打开 verbose 观察 step
        )
        
        # 4. Visualization
        gen_frames = generated_flat.reshape(n_pred, C, H, W)
        cond_frames_sq = cond_frames.squeeze(0)
        gt_frames_sq = gt_frames.squeeze(0)
        
        # Denormalize
        gen_frames = inverse_data_transform(config, gen_frames)
        cond_frames_sq = inverse_data_transform(config, cond_frames_sq)
        gt_frames_sq = inverse_data_transform(config, gt_frames_sq)
        
        gen_frames = torch.clamp(gen_frames, 0, 1)
        
        # Seq 1: GT
        seq_gt = torch.cat([cond_frames_sq, gt_frames_sq], dim=0)
        # Seq 2: Pred
        seq_pred = torch.cat([cond_frames_sq, gen_frames], dim=0)
        
        # Compare GIF
        combined_gif_frames = torch.cat([seq_gt, seq_pred], dim=3)
        gif_path = os.path.join(args.output_dir, f'sample_{i}_loss_{test_loss:.4f}.gif')
        save_gif(combined_gif_frames, gif_path)
        
        # Grid
        grid_tensor = torch.cat([seq_gt, seq_pred], dim=0) 
        grid_img = make_grid(grid_tensor, nrow=15, padding=2, pad_value=1.0)
        save_image(grid_img, os.path.join(args.output_dir, f'sample_{i}_grid.png'))

    logging.info("Testing complete.")

if __name__ == "__main__":
    main()