import os
import argparse
import torch
import numpy as np
import imageio
from torchvision.utils import make_grid, save_image
import logging
from tqdm import tqdm

# Imports from your project
from config import Config
from datasets.moving_mnist import MovingMNIST, data_transform, inverse_data_transform
from model.unet import UNet_DDPM
from model.ema import EMAHelper
from model import ddpm_sampler  # 确保你已经创建了 model_init.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Test MCVD on Moving MNIST")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file (e.g., logs/mnist_experiment/latest.pt)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Where to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of video samples to generate')
    parser.add_argument('--data_path', type=str, help='Path to Moving MNIST data')
    return parser.parse_args()

def save_gif(frames_tensor, path, fps=4):
    """
    frames_tensor: [T, C, H, W] tensor in [0, 1]
    """
    # [T, C, H, W] -> [T, H, W, C] -> numpy uint8
    frames = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    
    # 如果是单通道灰度图，imageio 需要 [H, W] 或者 [H, W, 3] 才能正常显示颜色，这里复制成3通道看起来更直观
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
        
    imageio.mimsave(path, frames, fps=fps)
    logging.info(f"Saved GIF: {path}")

@torch.no_grad()
def main():
    args = parse_args()
    config = Config()
    device = config.device
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data (Test Set)
    test_dataset = MovingMNIST(root=args.data_path, train=False)
    # 取前N个样本进行可视化
    indices = list(range(args.num_samples))
    subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    logging.info(f"Loading model from {args.ckpt_path}...")
    scorenet = UNet_DDPM(config).to(device)
    scorenet = torch.nn.DataParallel(scorenet)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # 1. 必须先加载基础模型权重！
    # 这步至关重要，它会恢复 GaussianFourierProjection.W (不可训练参数) 
    # 以及 alphas/betas 等 registered buffers。
    logging.info("Loading base model state...")
    scorenet.load_state_dict(checkpoint['model_state'])
    
    # 2. 如果有 EMA，则用 EMA 的平滑权重覆盖可训练参数
    if 'ema_state' in checkpoint:
        logging.info("Loading and applying EMA weights...")
        ema = EMAHelper(mu=config.model.ema_rate)
        ema.register(scorenet)
        ema.load_state_dict(checkpoint['ema_state'])
        # 这会将 EMA 影子权重复制到 scorenet 的可训练参数中
        ema.ema(scorenet)
    else:
        logging.warning("No EMA state found! Using base weights.")
        
    scorenet.eval()
    
    # 3. Inference Loop
    for i, (X, _) in enumerate(tqdm(loader, desc="Generating videos")):
        X = X.to(device) # [1, 20, 1, 64, 64]
        X = data_transform(config, X) # -> [-1, 1]
        
        # Split Data
        # Moving MNIST has 20 frames total.
        # We use 5 as condition, predict next 10 (as trained).
        # Ground Truth for comparison will be the full 15 frames.
        
        n_cond = config.data.num_frames_cond # 5
        n_pred = config.data.num_frames      # 10
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        
        # Prepare Condition
        cond_frames = X[:, :n_cond] # [1, 5, 1, 64, 64]
        
        # Prepare Ground Truth (Target)
        gt_frames = X[:, n_cond : n_cond+n_pred] # [1, 10, 1, 64, 64]
        
        # Flatten Cond for Model Input
        cond_tensor = cond_frames.reshape(1, n_cond*C, H, W)
        
        # Initialize random noise for the frames we want to predict
        # Shape matches what the model expects for output: [B, n_pred*C, H, W]
        x_init = torch.randn(1, n_pred*C, H, W).to(device)
        
        # Run Sampling (Reverse Diffusion)
        # Using DDPM sampler from mcvd_model_init
        logging.info(f"Sampling video {i}...")
        
        # sampler args: x_mod (noise), scorenet, cond (condition), ...
        # 注意: ddpm_sampler 的 cond_mask 参数默认为 None，表示全部 conditioned，这正是我们在推理时想要的
        # 必须设置 denoise=False！
        # 因为 DDPM 采样循环已经包含了去噪到 t=0 的过程。
        # 额外的 denoise 步骤会使用 t=999 的参数破坏生成的图像。
        generated_flat = ddpm_sampler(
            x_init, 
            scorenet, 
            cond=cond_tensor, 
            final_only=True, 
            denoise=False,  # <--- 修改这里为 False
            clip_before=True,
            subsample_steps=None, 
            verbose=False
        )
        # generated_flat shape: [1, 10*C, H, W]
        
        # 4. Post-processing & Visualization
        
        # Reshape generated to [T, C, H, W]
        gen_frames = generated_flat.reshape(n_pred, C, H, W)
        cond_frames_sq = cond_frames.squeeze(0) # [5, C, H, W]
        gt_frames_sq = gt_frames.squeeze(0)     # [10, C, H, W]
        
        # Denormalize [-1, 1] -> [0, 1]
        gen_frames = inverse_data_transform(config, gen_frames)
        cond_frames_sq = inverse_data_transform(config, cond_frames_sq)
        gt_frames_sq = inverse_data_transform(config, gt_frames_sq)
        
        # Clip to be safe
        gen_frames = torch.clamp(gen_frames, 0, 1)
        
        # Concatenate sequences: Past (5) + Future (10)
        # Sequence 1: Ground Truth (Past + GT Future)
        seq_gt = torch.cat([cond_frames_sq, gt_frames_sq], dim=0)
        
        # Sequence 2: Prediction (Past + Generated Future)
        seq_pred = torch.cat([cond_frames_sq, gen_frames], dim=0)
        
        # --- Visualization 1: Comparison GIF ---
        # Top: Ground Truth, Bottom: Prediction
        # We stack them side-by-side or top-down. Let's do Side-by-Side.
        combined_gif_frames = torch.cat([seq_gt, seq_pred], dim=3) # Concatenate along Width
        
        gif_path = os.path.join(args.output_dir, f'sample_{i}_compare.gif')
        save_gif(combined_gif_frames, gif_path)
        
        # --- Visualization 2: Static Grid Image ---
        # Row 1: GT Frames
        # Row 2: Pred Frames
        # Only showing the predicted part to save space, or full sequence? Let's show full.
        
        # seq_gt: [15, 1, 64, 64], seq_pred: [15, 1, 64, 64]
        # Make a grid where top row is GT, bottom is Pred
        # We need to reshape to [N_images, C, H, W] -> [30, 1, 64, 64]
        
        grid_tensor = torch.cat([seq_gt, seq_pred], dim=0) 
        grid_img = make_grid(grid_tensor, nrow=15, padding=2, pad_value=1.0) # nrow=15 means one row per sequence
        
        print(f"Gen min: {gen_frames.min()}, max: {gen_frames.max()}")
        
        img_path = os.path.join(args.output_dir, f'sample_{i}_grid.png')
        save_image(grid_img, img_path)
        logging.info(f"Saved Grid: {img_path}")

    logging.info("Testing complete.")

if __name__ == "__main__":
    main()