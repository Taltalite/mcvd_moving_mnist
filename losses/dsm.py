import torch
from functools import partial
from torch.distributions.gamma import Gamma

def anneal_dsm_score_estimation(scorenet, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):
    # x: Target [B, 10, H, W]
    # cond: Condition [B, 5, H, W]
    
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    
    # --- 1. 联合训练核心逻辑 ---
    # 如果 output_all_frames=True，我们将 cond 和 x 拼在一起
    # 把它看作一个整体视频序列 [B, 15, H, W]
    if all_frames and cond is not None:
        x = torch.cat([x, cond], dim=1)
        # 既然已经拼进去了，conditional input 设为 None
        cond = None 
    
    # 获取 alphas
    alphas = net.alphas
    if labels is None:
        labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
    used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
    
    # --- 2. 整体加噪 ---
    # 此时 x 是 15 帧，我们对这 15 帧全部加噪！
    z = torch.randn_like(x)
    perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
    
    # --- 3. 预测 ---
    # scorenet 接收 15 帧的 noisy input，cond=None
    scorenet_fn = partial(scorenet, cond=cond)
    pred = scorenet_fn(perturbed_x, labels, cond_mask=cond_mask)
    
    # --- 4. Loss 计算 (使用 Sum) ---
    if L1:
        loss_val = (z - pred).abs()
    else:
        loss_val = 0.5 * (z - pred).square()
        
    # 原作者逻辑：先 reshape 成 [B, -1] 然后 sum(dim=-1)
    # 这意味着每个样本贡献的 loss 是其所有像素误差之和
    loss = loss_val.reshape(len(x), -1).sum(dim=-1)

    return loss.mean(dim=0)