import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
import csv
# 设置 matplotlib 后端为 Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Imports from your modules
# 确保这些文件路径存在
from config import Config
from datasets.moving_mnist import MovingMNIST, data_transform
from model.unet import UNet_DDPM
from model.ema import EMAHelper
from losses.dsm import anneal_dsm_score_estimation

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Train MCVD on Moving MNIST")
    parser.add_argument('--log_dir', type=str, default='./logs/mnist_experiment', help='Directory to save logs, checkpoints and plots')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    # 你的命令中使用了 --data_path，这里必须对应
    parser.add_argument('--data_path', type=str, required=True, help='Path to Moving MNIST data')
    return parser.parse_args()

def get_optimizer(config, parameters):
    if config.optim.weight_decay > 0:
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                           betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    else:
        return optim.Adam(parameters, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                          amsgrad=config.optim.amsgrad, eps=config.optim.eps)

def warmup_lr(optimizer, step, warmup, max_lr):
    if step < warmup:
        lr = max_lr * float(step) / warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer.param_groups[0]['lr']

def conditioning_fn(config, X, conditional=True, training=True):
    if not conditional:
        return X, None, None
    
    B, T, C, H, W = X.shape
    n_cond = config.data.num_frames_cond
    n_pred = config.data.num_frames
    
    if T < n_cond + n_pred:
        raise ValueError(f"Dataset frames {T} < needed {n_cond + n_pred}")

    cond_frames = X[:, :n_cond, ...] 
    target_frames = X[:, n_cond:n_cond+n_pred, ...] 
    
    cond_tensor = cond_frames.reshape(B, n_cond*C, H, W)
    X_target = target_frames.reshape(B, n_pred*C, H, W)
    
    mask_cond = False
    # 只有训练时才随机 Mask
    if training and config.data.prob_mask_cond > 0.0:
        if torch.rand(1).item() < config.data.prob_mask_cond:
            mask_cond = True
            
    if mask_cond:
        cond_tensor = torch.zeros_like(cond_tensor)
        cond_mask = torch.zeros(B, 1, 1, 1).to(X.device) 
    else:
        cond_mask = torch.ones(B, 1, 1, 1).to(X.device) 
        
    return X_target, cond_tensor, cond_mask

def log_to_csv(log_dir, epoch, step, train_loss, val_loss, lr):
    csv_path = os.path.join(log_dir, 'training_metrics.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Step', 'Train Loss', 'Val Loss', 'Learning Rate'])
        writer.writerow([epoch, step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}"])

def plot_training_stats(log_dir, train_losses, val_losses, lrs):
    try:
        epochs = range(len(train_losses))
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, train_losses, color=color, label='Train Loss')
        if val_losses:
            # 确保 val_losses 长度和 epochs 一致，防止绘图报错
            if len(val_losses) == len(epochs):
                ax1.plot(epochs, val_losses, color='tab:orange', linestyle='--', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)
        if len(lrs) == len(epochs):
            ax2.plot(epochs, lrs, color=color, linestyle=':', label='LR')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Training & Validation Metrics')
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting stats: {e}")

def validate(scorenet, val_loader, config):
    scorenet.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(config.device)
            X_batch = data_transform(config, X_batch)
            
            X_target, cond, cond_mask = conditioning_fn(config, X_batch, training=False)
            
            loss = anneal_dsm_score_estimation(
                scorenet, X_target, labels=None, cond=cond, cond_mask=cond_mask,
                loss_type='a', all_frames=config.model.output_all_frames
            )
            total_loss += loss.item()
            count += 1
    # 避免除以零
    return total_loss / max(count, 1)

def main():
    args = parse_args()
    config = Config()
    # 覆盖 Config 参数
    config.training.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 1. Dataset
    train_dataset = MovingMNIST(root=args.data_path, train=True)
    val_dataset = MovingMNIST(root=args.data_path, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, 
                              num_workers=config.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, 
                            num_workers=config.data.num_workers, drop_last=True)
    
    # 2. Model
    logging.info(f"Initializing model on {config.device}...")
    scorenet = UNet_DDPM(config).to(config.device)
    scorenet = torch.nn.DataParallel(scorenet)
    
    optimizer = get_optimizer(config, scorenet.parameters())
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(scorenet)
    
    start_epoch = 0
    step = 0
    latest_ckpt_path = os.path.join(args.log_dir, 'latest.pt')
    
    train_loss_hist, val_loss_hist, lr_hist = [], [], []
    
    if os.path.exists(latest_ckpt_path):
        logging.info(f"Resuming from {latest_ckpt_path}")
        try:
            ckpt = torch.load(latest_ckpt_path)
            scorenet.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            ema_helper.load_state_dict(ckpt['ema_state'])
            start_epoch = ckpt['epoch'] + 1
            step = ckpt['step']
            train_loss_hist = ckpt.get('train_loss_hist', [])
            val_loss_hist = ckpt.get('val_loss_hist', [])
            lr_hist = ckpt.get('lr_hist', [])
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")

    logging.info(f"Training start: Train size {len(train_dataset)}, Val size {len(val_dataset)}")

    for epoch in range(start_epoch, config.training.n_epochs):
        # --- Training ---
        scorenet.train()
        epoch_loss = 0.0
        batches = 0
        current_lr = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for X_batch, _ in pbar:
                optimizer.zero_grad()
                step += 1
                current_lr = warmup_lr(optimizer, step, config.optim.warmup, config.optim.lr)
                
                X_batch = X_batch.to(config.device)
                X_batch = data_transform(config, X_batch)
                
                # Conditioning function
                X_target, cond, cond_mask = conditioning_fn(config, X_batch, training=True)
                
                # DSM Loss
                loss = anneal_dsm_score_estimation(
                    scorenet, X_target, labels=None, cond=cond, cond_mask=cond_mask,
                    loss_type='a', all_frames=config.model.output_all_frames
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scorenet.parameters(), config.optim.grad_clip)
                optimizer.step()
                ema_helper.update(scorenet)
                
                loss_val = loss.item()
                epoch_loss += loss_val
                batches += 1
                pbar.set_postfix({'loss': loss_val, 'lr': f"{current_lr:.1e}"})
        
        avg_train_loss = epoch_loss / batches
        train_loss_hist.append(avg_train_loss)
        lr_hist.append(current_lr)
        
        # --- Validation ---
        logging.info("Running Validation...")
        avg_val_loss = validate(scorenet, val_loader, config)
        val_loss_hist.append(avg_val_loss)
        
        logging.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # --- Logging ---
        log_to_csv(args.log_dir, epoch, step, avg_train_loss, avg_val_loss, current_lr)
        plot_training_stats(args.log_dir, train_loss_hist, val_loss_hist, lr_hist)
        
        # --- Checkpointing ---
        states = {
            'model_state': scorenet.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'ema_state': ema_helper.state_dict(),
            'epoch': epoch,
            'step': step,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist,
            'lr_hist': lr_hist
        }
        torch.save(states, latest_ckpt_path)
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save(states, os.path.join(args.log_dir, f'ckpt_epoch_{epoch+1}.pt'))

if __name__ == "__main__":
    main()