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
    parser.add_argument('--temperature', type=float,default=1.0, help='Sampler temperature')
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
    
    # [修改后代码] 请完全替换为以下内容：
    logging.info(f"Loading EMA model state from {args.ckpt_path}...")
    # 1. 先加载原始权重以防万一
    scorenet.load_state_dict(checkpoint['model_state'])

    # 2. 尝试加载 EMA 权重
    if 'ema_state' in checkpoint:
        logging.info("Found EMA state, loading into model...")
        ema_shadow = checkpoint['ema_state']
        #以此将 EMA 权重复制到当前模型中
        #注意：你的 EMAHelper 存储的是 shadow 参数，我们需要手动赋值给 scorenet
        #这里必须处理 DataParallel 的 module 前缀问题
        
        net = scorenet.module if hasattr(scorenet, 'module') else scorenet
        
        # 遍历模型参数并赋值
        missed_keys = []
        for name, param in net.named_parameters():
            if param.requires_grad:
                if name in ema_shadow:
                    # 直接将 EMA 的数据拷贝到模型参数中
                    param.data.copy_(ema_shadow[name].data)
                else:
                    missed_keys.append(name)
        
        if missed_keys:
            logging.warning(f"EMA keys missed: {missed_keys}")
        else:
            logging.info("EMA weights loaded successfully!")
    else:
        logging.warning("No EMA state found in checkpoint! Using base model (Results might be poor).")

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

            
        # --- Sampling ---
        x_init = torch.randn(1, n_pred*C, H, W).to(device)
        
        logging.info(f"Sampling video {i}...")
        
        # 使用 Robust Linear Sampler
        generated_flat = ddpm_sampler(
            x_init, 
            scorenet, 
            cond=cond_tensor, 
            subsample_steps=None, # default 1000
            temperature=args.temperature,     # 尝试 0.7 或 0.5
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
        gif_path = os.path.join(args.output_dir, f'sample_{i}_loss_{test_loss:.4f}_t{args.temperature:.2f}.gif')
        save_gif(combined_gif_frames, gif_path)
        
        # Grid
        grid_tensor = torch.cat([seq_gt, seq_pred], dim=0) 
        grid_img = make_grid(grid_tensor, nrow=15, padding=2, pad_value=1.0)
        save_image(grid_img, os.path.join(args.output_dir, f'sample_{i}_grid_t{args.temperature:.2f}.png'))

    logging.info("Testing complete.")

if __name__ == "__main__":
    main()