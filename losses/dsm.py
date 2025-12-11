import torch
from functools import partial
from torch.distributions.gamma import Gamma

def anneal_dsm_score_estimation(scorenet, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):
    # x 是 Target (预测目标), cond 是 Condition (历史帧)
    
    if labels is None:
        # 随机采样时间步 t
        labels = torch.randint(0, len(scorenet.module.alphas), (x.shape[0],), device=x.device)
    
    alphas = scorenet.module.alphas
    # 正确获取当前 batch 每个样本对应的 alpha_bar
    used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
    
    # 1. 生成高斯噪声 epsilon
    z = torch.randn_like(x)
    
    # 2. 加噪过程 (Forward Diffusion)
    perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
    
    # 3. 模型预测
    # scorenet 内部会自动拼接 perturbed_x 和 cond
    scorenet_fn = partial(scorenet, cond=cond)
    pred = scorenet_fn(perturbed_x, labels, cond_mask=cond_mask)

    # --- DEBUG INSERT START ---
    if torch.rand(1).item() < 0.001: # 随机抽取约 0.1% 的步数打印
        print(f"\n[DEBUG TRAIN DSM]")
        print(f"  Labels (t): {labels[:5].tolist()}") # 看看训练时 sampled 到的 t 是什么
        print(f"  Alpha_bar[t]: {used_alphas.squeeze()[:5].tolist()}")
        print(f"  Z (Target) std: {z.std().item():.4f}, mean: {z.mean().item():.4f}")
        print(f"  Pred (Eps) std: {pred.std().item():.4f}, mean: {pred.mean().item():.4f}")
        print(f"  Diff (Z-Pred) mean abs: {(z-pred).abs().mean().item():.4f}")
    # --- DEBUG INSERT END ---
    
    # 4. 维度对齐
    # 如果模型输出了所有帧 (Pred + Cond)，截取 Pred 部分
    if pred.shape[1] != z.shape[1]:
        pred = pred[:, :x.shape[1], ...]
        
    # 5. 计算 Loss
    if L1:
        loss = (z - pred).abs()
    else:
        loss = 0.5 * (z - pred).square()
    
    # --- CRITICAL FIX START ---
    # 原代码使用 sum(dim=-1) 会导致 Loss 值过大 (数万)，结合 grad_clip=1.0 会导致梯度几乎被抹零。
    # 改为全维度 mean()，保持 Loss 在 1.0 附近，适配 standard diffusion 训练参数。
    return loss.mean()
    # --- CRITICAL FIX END ---