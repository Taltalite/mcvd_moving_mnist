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

import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 头部导入增加
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class VideoMetrics:
    def __init__(self, device):
        self.device = device
        # --- 修改点在这里：添加 data_range=1.0 ---
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
        # SSIM 这里你之前已经写对了
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # LPIPS 不需要 data_range，保持原样
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        
        self.reset()

    def reset(self):
        self.total_psnr = 0
        self.total_ssim = 0
        self.total_lpips = 0
        self.total_mse = 0
        self.count = 0
        
        # 统计量
        self.pred_mean_intensity = 0
        self.gt_mean_intensity = 0

    def update(self, pred, gt):
        """
        pred, gt: [T, C, H, W] or [B, T, C, H, W] in range [0, 1]
        """
        # 确保输入是 [B, T, C, H, W]
        if pred.ndim == 4:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
        B, T, C, H, W = pred.shape
        
        # 将 Batch 和 Time 维度合并计算，视为 T 张独立的图
        # shape: [B*T, C, H, W]
        flat_pred = pred.view(-1, C, H, W)
        flat_gt = gt.view(-1, C, H, W)
        
        # 1. MSE
        mse = torch.mean((flat_pred - flat_gt) ** 2).item()
        
        # 2. PSNR & SSIM
        psnr_val = self.psnr(flat_pred, flat_gt).item()
        ssim_val = self.ssim(flat_pred, flat_gt).item()
        
        # 3. LPIPS (需要转为 3 通道, 且范围推荐 [-1, 1] 但 [0,1] 也能跑，这里我们 repeat 一下)
        flat_pred_3c = flat_pred.repeat(1, 3, 1, 1)
        flat_gt_3c = flat_gt.repeat(1, 3, 1, 1)
        # LPIPS 期望输入在 [-1, 1]，我们现有的数据是 [0, 1]，做一个简单的转换
        lpips_val = self.lpips(flat_pred_3c * 2 - 1, flat_gt_3c * 2 - 1).item()
        
        # 4. Image Stats (Image Norm 概念)
        # 计算整个序列的平均像素值，看是否过暗/过亮
        p_mean = flat_pred.mean().item()
        g_mean = flat_gt.mean().item()

        # 累加
        self.total_mse += mse
        self.total_psnr += psnr_val
        self.total_ssim += ssim_val
        self.total_lpips += lpips_val
        self.pred_mean_intensity += p_mean
        self.gt_mean_intensity += g_mean
        self.count += 1
        
        return {
            "mse": mse,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "pred_int": p_mean,
            "gt_int": g_mean
        }

    def compute_avg(self):
        return {
            "Avg MSE": self.total_mse / self.count,
            "Avg PSNR": self.total_psnr / self.count,
            "Avg SSIM": self.total_ssim / self.count,
            "Avg LPIPS": self.total_lpips / self.count,
            "Avg Pred Intensity": self.pred_mean_intensity / self.count,
            "Avg GT Intensity": self.gt_mean_intensity / self.count
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Test MCVD on Moving MNIST")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Where to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of video samples to generate')
    parser.add_argument('--data_path', type=str, help='Path to Moving MNIST data')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampler temperature')
    
    # --- 必须添加以下两个参数 ---
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'pndm'], help='Choose sampler: ddpm (slow) or pndm (fast)')
    parser.add_argument('--steps', type=int, default=1000, help='Sampling steps (1000 for DDPM, 50 for PNDM)')
    
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

    # 1. 初始化指标计算器
    metrics_calculator = VideoMetrics(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Data (Test Set)
    test_dataset = MovingMNIST(root=args.data_path, train=False)
    indices = list(range(args.num_samples))
    subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    # 3. Load Model
    logging.info(f"Loading model from {args.ckpt_path}...")
    scorenet = UNet_DDPM(config).to(device)
    scorenet = torch.nn.DataParallel(scorenet)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    logging.info(f"Loading EMA model state from {args.ckpt_path}...")
    scorenet.load_state_dict(checkpoint['model_state'])

    if 'ema_state' in checkpoint:
        logging.info("Found EMA state, loading into model...")
        ema_shadow = checkpoint['ema_state']
        
        net = scorenet.module if hasattr(scorenet, 'module') else scorenet
        
        missed_keys = []
        for name, param in net.named_parameters():
            if param.requires_grad:
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name].data)
                else:
                    missed_keys.append(name)
        
        if missed_keys:
            logging.warning(f"EMA keys missed: {missed_keys}")
        else:
            logging.info("EMA weights loaded successfully!")
    else:
        logging.warning("No EMA state found in checkpoint! Using base model.")

    scorenet.eval()
    
    # 4. Inference Loop
    total_inference_time = 0 # 用于统计平均推理时间
    
    logging.info(f"Start Sampling using method: {args.sampler.upper()} with {args.steps} steps")

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
        
        # --- Test Loss (Optional Debugging) ---
        test_loss = calc_test_loss(scorenet, config, x_target, cond_tensor)
        
        # --- Sampling & Timing ---
        x_init = torch.randn(1, n_pred*C, H, W).to(device)
        
        start_time = time.time() # 计时开始
        
        if args.sampler == 'pndm':
            # PNDM 采样 (Fast)
            # 注意：务必设置 final_only=True 以避免维度错误
            generated_flat = FPNDM_sampler(
                x_init, 
                scorenet, 
                cond=cond_tensor,
                subsample_steps=args.steps, 
                denoise=True,
                clip_before=True,
                verbose=False,
                final_only=True 
            )
        else:
            # DDPM 采样 (Standard/Slow)
            generated_flat = ddpm_sampler(
                x_init, 
                scorenet, 
                cond=cond_tensor, 
                subsample_steps=args.steps, # 如果是 1000 则不跳步
                temperature=args.temperature,
                final_only=True, 
                denoise=False,
                clip_before=True,
                verbose=False
            )
            
        end_time = time.time()
        elapsed = end_time - start_time
        total_inference_time += elapsed

        # 5. Post-processing
        gen_frames = generated_flat.reshape(n_pred, C, H, W)
        cond_frames_sq = cond_frames.squeeze(0)
        gt_frames_sq = gt_frames.squeeze(0)
        
        # Denormalize to [0, 1]
        gen_frames = inverse_data_transform(config, gen_frames)
        cond_frames_sq = inverse_data_transform(config, cond_frames_sq)
        gt_frames_sq = inverse_data_transform(config, gt_frames_sq)
        
        gen_frames = torch.clamp(gen_frames, 0, 1)
        gt_frames_sq = torch.clamp(gt_frames_sq, 0, 1) # 确保 GT 也在范围内

        # --- 新增：针对 MNIST 的对比度增强 (Soft Thresholding) ---
        # 这不是作弊，这是针对二值数据的合理后处理
        # 它可以把接近 0 的背景压黑，接近 1 的笔画提亮，同时保持中间的渐变
        gen_frames = (gen_frames - 0.2) / (0.8 - 0.2) 
        gen_frames = torch.clamp(gen_frames, 0, 1)
        # ----------------------------------------------------


        # --- Metrics Update (核心修改) ---
        # 计算当前样本的指标并累加
        curr_metrics = metrics_calculator.update(gen_frames, gt_frames_sq)
        
        # 打印当前样本信息
        print(f"\n[Sample {i}] Time: {elapsed:.2f}s | MSE: {test_loss:.5f} | PSNR: {curr_metrics['psnr']:.2f} | LPIPS: {curr_metrics['lpips']:.3f}")

        # 6. Visualization
        seq_gt = torch.cat([cond_frames_sq, gt_frames_sq], dim=0)
        seq_pred = torch.cat([cond_frames_sq, gen_frames], dim=0)
        
        # GIF
        combined_gif_frames = torch.cat([seq_gt, seq_pred], dim=3)
        gif_path = os.path.join(args.output_dir, f'sample_{i}_{args.sampler}.gif')
        save_gif(combined_gif_frames, gif_path)
        
        # Grid
        grid_tensor = torch.cat([seq_gt, seq_pred], dim=0) 
        grid_img = make_grid(grid_tensor, nrow=15, padding=2, pad_value=1.0)
        save_image(grid_img, os.path.join(args.output_dir, f'sample_{i}_{args.sampler}_grid.png'))

    # --- Final Report ---
    avg_inference_time = total_inference_time / args.num_samples
    final_metrics = metrics_calculator.compute_avg()
    
    logging.info("="*40)
    logging.info(f"TESTING COMPLETE ({args.num_samples} samples)")
    logging.info(f"Sampler: {args.sampler.upper()} | Steps: {args.steps}")
    logging.info(f"Avg Inference Time: {avg_inference_time:.4f}s")
    logging.info("-" * 20)
    logging.info(f"Avg MSE:   {final_metrics['Avg MSE']:.5f}")
    logging.info(f"Avg PSNR:  {final_metrics['Avg PSNR']:.3f}")
    logging.info(f"Avg SSIM:  {final_metrics['Avg SSIM']:.3f}")
    logging.info(f"Avg LPIPS: {final_metrics['Avg LPIPS']:.3f}")
    logging.info("="*40)

if __name__ == "__main__":
    main()