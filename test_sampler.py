import os
import argparse
import torch
import numpy as np
import imageio
from torchvision.utils import make_grid, save_image
import logging
from tqdm import tqdm
import time

# Imports from your project
from config import Config
from datasets.moving_mnist import MovingMNIST, data_transform, inverse_data_transform
from model.unet import UNet_DDPM
# 注意：这里我们导入 ddim_sampler 来替代依赖外部文件的 FPNDM
from model import ddpm_sampler, ddim_sampler 
from losses.dsm import anneal_dsm_score_estimation

# Metrics
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class VideoMetrics:
    def __init__(self, device):
        self.device = device
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.reset()

    def reset(self):
        self.total_psnr = 0
        self.total_ssim = 0
        self.total_lpips = 0
        self.total_mse = 0
        self.count = 0

    def update(self, pred, gt):
        """
        pred, gt: [B, T, C, H, W] in range [0, 1]
        """
        B, T, C, H, W = pred.shape
        flat_pred = pred.view(-1, C, H, W)
        flat_gt = gt.view(-1, C, H, W)
        
        mse = torch.mean((flat_pred - flat_gt) ** 2).item()
        psnr_val = self.psnr(flat_pred, flat_gt).item()
        ssim_val = self.ssim(flat_pred, flat_gt).item()
        
        # --- 修复点：LPIPS 需要 3 通道输入 ---
        # Moving MNIST 是单通道 (N, 1, H, W)，我们需要将其复制为 (N, 3, H, W)
        flat_pred_3c = flat_pred.repeat(1, 3, 1, 1)
        flat_gt_3c = flat_gt.repeat(1, 3, 1, 1)
        
        # LPIPS 期望输入在 [-1, 1] 之间
        lpips_val = self.lpips(flat_pred_3c * 2 - 1, flat_gt_3c * 2 - 1).item()
        # -----------------------------------

        self.total_mse += mse
        self.total_psnr += psnr_val
        self.total_ssim += ssim_val
        self.total_lpips += lpips_val
        self.count += 1
        
        return {"mse": mse, "psnr": psnr_val, "lpips": lpips_val}

    def compute_avg(self):
        return {
            "Avg MSE": self.total_mse / self.count,
            "Avg PSNR": self.total_psnr / self.count,
            "Avg SSIM": self.total_ssim / self.count,
            "Avg LPIPS": self.total_lpips / self.count
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Test MCVD on Moving MNIST")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Where to save visualizations')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of video samples to generate')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Moving MNIST data')
    
    # Sampler options
    # Default to DDPM (slow, high quality). Use 'ddim' or 'pndm' for fast sampling.
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim', 'pndm'], help='Sampler type')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps. Default: 1000 for DDPM, 50 for DDIM/PNDM')
    
    return parser.parse_args()

def save_gif(frames_tensor, path, fps=4):
    frames = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    imageio.mimsave(path, frames, fps=fps)

@torch.no_grad()
def main():
    args = parse_args()
    config = Config()
    device = config.device

    # Set default steps if not provided
    if args.steps is None:
        if args.sampler == 'ddpm':
            args.steps = 1000
        else:
            args.steps = 50 # Standard for fast sampling

    logging.info(f"Initializing Testing: Sampler={args.sampler.upper()}, Steps={args.steps}")

    metrics_calculator = VideoMetrics(device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    test_dataset = MovingMNIST(root=args.data_path, train=False)
    # Ensure we don't go out of bounds
    num_samples = min(args.num_samples, len(test_dataset))
    indices = list(range(num_samples))
    subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    # Load Model
    scorenet = UNet_DDPM(config).to(device)
    scorenet = torch.nn.DataParallel(scorenet)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    scorenet.load_state_dict(checkpoint['model_state'])
    if 'ema_state' in checkpoint:
        logging.info("Loading EMA state...")
        # A simple way to load EMA state for DataParallel wrapper
        try:
            scorenet.module.load_state_dict(checkpoint['ema_state'], strict=False)
        except:
            logging.warning("Direct EMA load failed, using standard model weights.")
            
    scorenet.eval()
    
    total_inference_time = 0
    
    for i, (X, _) in enumerate(tqdm(loader, desc="Sampling")):
        X = X.to(device)
        X = data_transform(config, X)
        
        n_cond = config.data.num_frames_cond
        n_pred = config.data.num_frames
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        
        cond_frames = X[:, :n_cond]
        gt_frames = X[:, n_cond : n_cond+n_pred]
        
        cond_tensor = cond_frames.reshape(1, n_cond*C, H, W)
        x_init = torch.randn(1, n_pred*C, H, W).to(device)
        
        start_time = time.time()
        
        if args.sampler == 'ddpm':
            # Standard DDPM
            generated_flat = ddpm_sampler(
                x_init, 
                scorenet, 
                cond=cond_tensor, 
                subsample_steps=args.steps, # Usually None (1000 steps)
                final_only=True, 
                denoise=False,
                clip_before=True,
                verbose=False
            )
        else:
            # Fast Sampling (DDIM/PNDM)
            generated_flat = ddim_sampler(
                x_init, 
                scorenet, 
                cond=cond_tensor,
                subsample_steps=args.steps, # Usually 50
                final_only=True,
                denoise=True, # DDIM often benefits from a final denoise step
                clip_before=True,
                verbose=False
            )
            
        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # Post-processing
        gen_frames = generated_flat.reshape(1, n_pred, C, H, W)
        
        gen_frames = inverse_data_transform(config, gen_frames)
        gt_frames = inverse_data_transform(config, gt_frames)
        cond_frames = inverse_data_transform(config, cond_frames)
        
        gen_frames = torch.clamp(gen_frames, 0, 1)
        gt_frames = torch.clamp(gt_frames, 0, 1)
        cond_frames = torch.clamp(cond_frames, 0, 1)

        # Contrast enhancement for MNIST
        gen_frames = (gen_frames - 0.2) / (0.8 - 0.2)
        gen_frames = torch.clamp(gen_frames, 0, 1)

        metrics_calculator.update(gen_frames, gt_frames)
        
        # Save visualization for the first sample only to keep it clean
        seq_gt = torch.cat([cond_frames[0], gt_frames[0]], dim=0)
        seq_pred = torch.cat([cond_frames[0], gen_frames[0]], dim=0)
        combined = torch.cat([seq_gt, seq_pred], dim=0)
        
        # 文件名加上 i 以区分
        save_image(make_grid(combined, nrow=15, padding=2, pad_value=1), 
                   os.path.join(args.output_dir, f'sample_{i}_{args.sampler}_grid.png'))

    avg_time = total_inference_time / num_samples
    final_metrics = metrics_calculator.compute_avg()
    
    logging.info("="*40)
    logging.info(f"TESTING COMPLETE ({num_samples} samples)")
    logging.info(f"Sampler: {args.sampler.upper()} | Steps: {args.steps}")
    logging.info(f"Avg Inference Time: {avg_time:.4f}s")
    logging.info("-" * 20)
    logging.info(f"Avg MSE:   {final_metrics['Avg MSE']:.5f}")
    logging.info(f"Avg PSNR:  {final_metrics['Avg PSNR']:.3f}")
    logging.info(f"Avg SSIM:  {final_metrics['Avg SSIM']:.3f}")
    logging.info(f"Avg LPIPS: {final_metrics['Avg LPIPS']:.3f}")
    logging.info("="*40)

if __name__ == "__main__":
    main()