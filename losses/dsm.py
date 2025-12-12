import torch
from functools import partial
from torch.distributions.gamma import Gamma

def anneal_dsm_score_estimation(scorenet, x, labels=None, cond=None, cond_mask=None, loss_type='a', L1=False, all_frames=False, **kwargs):    
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet

    n_cond = cond.shape[1] if cond is not None else 0
    n_target = x.shape[1]
    

    if all_frames and cond is not None:
        x_joint = torch.cat([cond, x], dim=1)
        cond_input = None 
    else:
        x_joint = x
        cond_input = cond

    # 获取 alphas
    alphas = net.alphas
    if labels is None:
        labels = torch.randint(0, len(alphas), (x_joint.shape[0],), device=x_joint.device)
    used_alphas = alphas[labels].reshape(x_joint.shape[0], *([1] * len(x_joint.shape[1:])))
    
    # 加噪
    z = torch.randn_like(x_joint)
    perturbed_x = used_alphas.sqrt() * x_joint + (1 - used_alphas).sqrt() * z
    
    # 预测
    scorenet_fn = partial(scorenet, cond=cond_input)
    pred = scorenet_fn(perturbed_x, labels, cond_mask=cond_mask)
    
    # 计算 Loss
    if L1:
        loss_val = (z - pred).abs()
    else:
        loss_val = 0.5 * (z - pred).square()
    if all_frames and cond is not None:
        mask = torch.ones_like(loss_val)
        mask[:, :n_cond, ...] = 0.0
        loss_val = loss_val * mask
        loss = loss_val.reshape(len(x), -1).sum(dim=-1)
    else:
        loss = loss_val.reshape(len(x), -1).sum(dim=-1)

    return loss.mean(dim=0)