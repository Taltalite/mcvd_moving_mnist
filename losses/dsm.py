import torch
from functools import partial
from torch.distributions.gamma import Gamma

def anneal_dsm_score_estimation(scorenet, x, labels=None, cond=None, cond_mask=None, loss_type='a', L1=False, all_frames=False, **kwargs):
    # x: Target [B, 10, H, W] (Future)
    # cond: Condition [B, 5, H, W] (Past)
    
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    
    x_target = x 
    
    alphas = net.alphas
    if labels is None:
        labels = torch.randint(0, len(alphas), (x_target.shape[0],), device=x_target.device)
    used_alphas = alphas[labels].reshape(x_target.shape[0], *([1] * len(x_target.shape[1:])))
    
    z = torch.randn_like(x_target)
    perturbed_x = used_alphas.sqrt() * x_target + (1 - used_alphas).sqrt() * z
    
    scorenet_fn = partial(scorenet, cond=cond)
    
    pred = scorenet_fn(perturbed_x, labels, cond_mask=cond_mask)
    
    
    if L1:
        loss_val = (z - pred).abs()
    else:
        loss_val = 0.5 * (z - pred).square()
        
    loss = loss_val.reshape(len(x), -1).sum(dim=-1)

    return loss.mean(dim=0)