# # SMLD: s = -1/sigma * z
# # DDPM: s = -1/sqrt(1 - alpha) * z
# # All `scorenet` models return z, not s!

# import torch
# import logging
# import numpy as np

# from functools import partial
# from scipy.stats import hmean
# from torch.distributions.gamma import Gamma
# from tqdm import tqdm
# from . import pndm


# def get_sigmas(config):
#     T = getattr(config.model, 'num_classes')
#     if config.model.sigma_dist == 'geometric':
#         return torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end), T).to(config.device)
#     elif config.model.sigma_dist == 'linear':
#         return torch.linspace(config.model.sigma_begin, config.model.sigma_end, T).to(config.device)
#     elif config.model.sigma_dist == 'cosine':
#         # Cosine Schedule 通常是直接计算 alphas_cumprod，然后反推 betas
#         t = torch.arange(T + 1, dtype=torch.float64, device=config.device) / T
#         s = 0.008
#         f = torch.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
#         alphas_cumprod = f / f[0]
#         betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         return torch.clip(betas, 0, 0.999).float() # 必须返回 betas!
#     else:
#         raise NotImplementedError('sigma distribution not supported')


# @torch.no_grad()
# def FPNDM_sampler(x_mod, scorenet, cond=None, final_only=False, denoise=True, subsample_steps=None,
#                  verbose=False, log=True, clip_before=True, t_min=-1, gamma=False, **kwargs):

#     net = scorenet.module if hasattr(scorenet, 'module') else scorenet

#     alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
#     steps = np.arange(len(betas))

#     net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

#     alphas_old = alphas.flip(0)
#     skip = len(alphas) // subsample_steps
#     steps = range(0, len(alphas), skip)
#     steps_next = [-1] + list(steps[:-1])


#     steps = torch.tensor(steps, device=alphas.device)
#     steps_next = torch.tensor(steps_next, device=alphas.device)
#     #alphas = alphas.index_select(0, steps)
#     alphas_next = alphas.index_select(0, steps_next + 1)
#     alphas = alphas.index_select(0, steps + 1)
#     #print(alphas_next)
#     #print(alphas)

#     images = []
#     scorenet = partial(scorenet, cond=cond)

#     L = len(steps)
#     ets = []
#     for i, step in enumerate(steps):

#         t_ = (steps[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
#         t_next = (steps_next[i] * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
#         #print(alphas_next[i])
#         #print(alphas[i])
#         #print(alphas_old[t_next.long() + 1][0])
#         #print(alphas_old[t_.long() + 1][0])
#         x_mod, ets = pndm.gen_order_4(x_mod, t_, t_next, model=scorenet, alphas_cump=alphas_old, ets=ets, clip_before=clip_before)

#         if not final_only:
#             images.append(x_mod.to('cpu'))

#     if final_only:
#         return x_mod.unsqueeze(0)
#     else:
#         return torch.stack(images)


# @torch.no_grad()
# def ddim_sampler(x_mod, scorenet, cond=None, final_only=False, denoise=True, subsample_steps=None,
#                  verbose=False, log=True, clip_before=True, t_min=-1, gamma=False, **kwargs):

#     net = scorenet.module if hasattr(scorenet, 'module') else scorenet
#     alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
#     if gamma:
#         ks_cum, thetas = net.k_cum, net.theta_t
#     steps = np.arange(len(betas))

#     net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'
    
#     if subsample_steps is not None:
#         if subsample_steps < len(alphas):
#             skip = len(alphas) // subsample_steps
#             steps = range(0, len(alphas), skip)
#             steps = torch.tensor(steps, device=alphas.device)
#             # new alpha, beta, alpha_prev
#             alphas = alphas.index_select(0, steps)
#             alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
#             betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
#             if gamma:
#                 ks_cum = ks_cum.index_select(0, steps)
#                 thetas = thetas.index_select(0, steps)

#     images = []
#     scorenet = partial(scorenet, cond=cond)
#     x_transf = False

#     L = len(steps)
#     for i, step in enumerate(steps):

#         if step < t_min*len(alphas): # otherwise, wait until it happens
#             continue

#         if not x_transf and t_min > 0: # we must add noise to the previous frame
#             if gamma:
#                 z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
#                           torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
#                 z = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
#             else:
#                 z = torch.randn_like(x_mod)
#             x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
#         x_transf = True

#         c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
#         labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
#         grad = scorenet(x_mod, labels)

#         #print(alphas_prev[i])
#         #print(alphas[i])

#         x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * grad)
#         if clip_before:
#             x0 = x0.clip_(-1, 1)
#         x_mod = c_alpha_prev.sqrt() * x0 + (1 - c_alpha_prev).sqrt() * grad

#         if not final_only:
#             images.append(x_mod.to('cpu'))

#         if i == 0 or (i+1) % max(L//10, 1) == 0:

#             if verbose or log:
#                 grad = -1/(1 - c_alpha).sqrt() * grad
#                 grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
#                 image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
#                 grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * (1 - c_alpha)

#             if verbose:
#                 print("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
#                     "DDIM gamma" if gamma else "DDIM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))
#             if log:
#                 logging.info("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
#                     "DDIM gamma" if gamma else "DDIM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))

#         # # If last step, don't add noise
#         # last_step = i + 1 == L
#         # if last_step:
#         #     continue

#     # Denoise
#     if denoise: # x + batch_mul(std ** 2, score_fn(x, eps_t))
#         last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
#         x_mod = x_mod - (1 - alphas[-1]).sqrt() * scorenet(x_mod, last_noise)
#         if not final_only:
#             images.append(x_mod.to('cpu'))

#     if final_only:
#         return x_mod.unsqueeze(0)
#     else:
#         return torch.stack(images)


# # @torch.no_grad()
# # def ddpm_sampler(x_mod, scorenet, cond=None, just_beta=False, final_only=False, denoise=True, subsample_steps=None,
# #                  same_noise=False, noise_val=None, frac_steps=None, verbose=False, log=False, clip_before=True, 
# #                  t_min=-1, gamma=False, temperature=1.0, **kwargs):

# #     net = scorenet.module if hasattr(scorenet, 'module') else scorenet
# #     betas = net.betas
# #     alphas_cumprod = net.alphas
# #     alphas_cumprod_prev = net.alphas_prev
# #     num_timesteps = len(betas)
# #     steps = list(range(num_timesteps))[::-1]
    
# #     if subsample_steps is not None:
# #         skip = num_timesteps // subsample_steps
# #         steps = steps[::skip]

# #     images = []
# #     scorenet_fn = partial(scorenet, cond=cond)

# #     for i, step_idx in enumerate(steps):
# #         t = torch.full((x_mod.shape[0],), step_idx, device=x_mod.device, dtype=torch.long)
        
# #         beta_t = betas[step_idx]
# #         alpha_cumprod_t = alphas_cumprod[step_idx]
# #         alpha_cumprod_prev_t = alphas_cumprod_prev[step_idx]
# #         alpha_t = alpha_cumprod_t / alpha_cumprod_prev_t

# #         grad = scorenet_fn(x_mod, t) 

# #         sqrt_recip_alpha_bar = torch.sqrt(1. / alpha_cumprod_t).view(-1, 1, 1, 1)
# #         sqrt_recip_m1_alpha_bar = torch.sqrt(1. / alpha_cumprod_t - 1.).view(-1, 1, 1, 1)
# #         pred_x0 = sqrt_recip_alpha_bar * x_mod - sqrt_recip_m1_alpha_bar * grad
        
# #         if clip_before:
# #             pred_x0 = pred_x0.clamp(-1., 1.)

# #         coeff_x0 = ((beta_t * torch.sqrt(alpha_cumprod_prev_t)) / (1 - alpha_cumprod_t)).view(-1, 1, 1, 1)
# #         coeff_xt = (((1 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t)) / (1 - alpha_cumprod_t)).view(-1, 1, 1, 1)
# #         mean = coeff_x0 * pred_x0 + coeff_xt * x_mod

# #         if step_idx > 0:
# #             noise = torch.randn_like(x_mod) * temperature
# #             sigma_t_sq = ((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * beta_t
# #             sigma_t = torch.sqrt(sigma_t_sq + 1e-10).view(-1, 1, 1, 1)
# #             x_mod = mean + sigma_t * noise
# #         else:
# #             x_mod = mean
# #     if final_only:
# #         return x_mod.unsqueeze(0)
# #     else:
# #         return torch.stack(images) if len(images) > 0 else x_mod.unsqueeze(0)

# @torch.no_grad()
# def ddpm_sampler(x_mod, scorenet, cond=None, just_beta=False, final_only=False, denoise=True, subsample_steps=None,
#                  same_noise=False, noise_val=None, frac_steps=None, verbose=False, log=False, clip_before=True, 
#                  t_min=-1, gamma=False, temperature=1.0, **kwargs):
#     """
#     Standard DDPM Sampler with Hard Clipping.
#     Optimized for MNIST-like datasets where strict range [-1, 1] is required.
#     """
#     net = scorenet.module if hasattr(scorenet, 'module') else scorenet
#     betas = net.betas
#     alphas_cumprod = net.alphas
#     alphas_cumprod_prev = net.alphas_prev
#     num_timesteps = len(betas)
    
#     # 倒序采样：从 T-1 到 0
#     steps = list(range(num_timesteps))[::-1]
    
#     if subsample_steps is not None:
#         # 如果需要跳步（加速），均匀切分
#         skip = num_timesteps // subsample_steps
#         steps = steps[::skip]

#     images = []
#     scorenet_fn = partial(scorenet, cond=cond)

#     for i, step_idx in enumerate(steps):
#         # 1. 构造时间步 t
#         t = torch.full((x_mod.shape[0],), step_idx, device=x_mod.device, dtype=torch.long)
        
#         # 2. 获取参数
#         beta_t = betas[step_idx]
#         alpha_cumprod_t = alphas_cumprod[step_idx]
#         alpha_cumprod_prev_t = alphas_cumprod_prev[step_idx]
#         alpha_t = alpha_cumprod_t / alpha_cumprod_prev_t

#         # 3. 预测噪声 epsilon
#         grad = scorenet_fn(x_mod, t) 

#         # 4. 预测 x0 (Reconstruction)
#         # Formula: x0 = (xt - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
#         sqrt_recip_alpha_bar = torch.rsqrt(alpha_cumprod_t)
#         sqrt_recip_m1_alpha_bar = torch.sqrt(1. / alpha_cumprod_t - 1.)
#         pred_x0 = sqrt_recip_alpha_bar * x_mod - sqrt_recip_m1_alpha_bar * grad
        
#         # --- 关键修正：必须使用硬截断 (Hard Clipping) ---
#         # 对于 MNIST，必须强制把数值拉回 [-1, 1]，否则背景会发灰
#         if clip_before:
#             pred_x0 = pred_x0.clamp(-1., 1.)

#         # 5. 计算后验均值 (Posterior Mean)
#         # mu_t = (beta_t * sqrt(alpha_bar_prev) / (1-alpha_bar)) * x0 + 
#         #        ((1-alpha_bar_prev) * sqrt(alpha_t) / (1-alpha_bar)) * xt
#         coeff_x0 = (beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t))
#         coeff_xt = ((1.0 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t))
#         mean = coeff_x0 * pred_x0 + coeff_xt * x_mod

#         # 6. 加入噪声 (Langevin Step)
#         if step_idx > 0:
#             # 计算后验方差 (Posterior Variance)
#             # sigma^2 = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
#             posterior_variance = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
            
#             # log var trick for stability: 0.5 * log(var)
#             # 加上 1e-20 防止 log(0)
#             posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
            
#             noise = torch.randn_like(x_mod) * temperature
            
#             # 使用对数方差来乘，数值更稳定
#             sigma = torch.exp(0.5 * posterior_log_variance_clipped)
#             x_mod = mean + sigma * noise
#         else:
#             # 最后一步不加噪声
#             x_mod = mean

#     if final_only:
#         return x_mod.unsqueeze(0)
#     else:
#         return torch.stack(images) if len(images) > 0 else x_mod.unsqueeze(0)

import torch
import logging
import numpy as np
from functools import partial
from torch.distributions.gamma import Gamma

def get_sigmas(config):
    T = getattr(config.model, 'num_classes')
    if config.model.sigma_dist == 'geometric':
        return torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end), T).to(config.device)
    elif config.model.sigma_dist == 'linear':
        return torch.linspace(config.model.sigma_begin, config.model.sigma_end, T).to(config.device)
    elif config.model.sigma_dist == 'cosine':
        t = torch.arange(T + 1, dtype=torch.float64, device=config.device) / T
        s = 0.008
        f = torch.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()
    else:
        raise NotImplementedError('sigma distribution not supported')

@torch.no_grad()
def ddim_sampler(x_mod, scorenet, cond=None, final_only=False, denoise=True, subsample_steps=None,
                 verbose=False, log=True, clip_before=True, t_min=-1, gamma=False, **kwargs):
    """
    Corrected DDIM Sampler.
    Ensures reverse sampling from T -> 0.
    """
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    # Get full schedule parameters
    alphas_cumprod = net.alphas # alpha_bar
    
    num_train_timesteps = len(alphas_cumprod)
    
    # 1. Construct the time steps sequence
    if subsample_steps is None:
        # Full DDPM steps: [999, 998, ..., 0]
        timesteps = np.arange(num_train_timesteps)[::-1]
    else:
        # DDIM strided steps: e.g. [980, 960, ..., 0]
        # We define a linear sequence and flip it
        skip = num_train_timesteps // subsample_steps
        timesteps = np.arange(0, num_train_timesteps, skip)[::-1]
        
    # We need pairs of (t, prev_t). 
    # For the last step (t=0), prev_t is -1.
    timesteps_next = np.append(timesteps[1:], -1)
    
    images = []
    scorenet = partial(scorenet, cond=cond)

    L = len(timesteps)
    
    for i, (t, t_next) in enumerate(zip(timesteps, timesteps_next)):
        
        # 1. Construct tensor labels for the model
        # The model expects labels in range [0, T-1]
        labels = torch.full((x_mod.shape[0],), t, device=x_mod.device, dtype=torch.long)
        
        # 2. Get model prediction (noise / epsilon)
        # Note: Depending on config, this might be epsilon or score. 
        # Assuming standard DDPM epsilon prediction here.
        et = scorenet(x_mod, labels)
        
        # 3. Get alpha_bar at current step t and next step t_next
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(x_mod.device)
        
        # 4. Predict x0 (Denoised observation)
        # x0 = (xt - sqrt(1-at) * et) / sqrt(at)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
        
        pred_x0 = (x_mod - sqrt_one_minus_alpha_bar_t * et) / sqrt_alpha_bar_t
        
        if clip_before:
            pred_x0 = pred_x0.clamp(-1., 1.)
            
        # 5. DDIM Update Step (Deterministic)
        # xt-1 = sqrt(at-1) * x0 + sqrt(1 - at-1) * et
        # Note: This is the "sigma=0" version (fully deterministic ODE)
        
        sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()
        sqrt_one_minus_alpha_bar_t_next = (1 - alpha_bar_t_next).sqrt()
        
        # Direction pointing to xt
        dir_xt = sqrt_one_minus_alpha_bar_t_next * et
        
        x_mod = sqrt_alpha_bar_t_next * pred_x0 + dir_xt

        if not final_only:
            images.append(x_mod.to('cpu'))
            
        if verbose and i % (L // 5) == 0:
             print(f"DDIM Step {i}/{L} (t={t} -> {t_next})")

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

@torch.no_grad()
def ddpm_sampler(x_mod, scorenet, cond=None, just_beta=False, final_only=False, denoise=True, subsample_steps=None,
                 same_noise=False, noise_val=None, frac_steps=None, verbose=False, log=False, clip_before=True, 
                 t_min=-1, gamma=False, temperature=1.0, **kwargs):
    """
    Standard DDPM Sampler.
    """
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    betas = net.betas
    alphas_cumprod = net.alphas
    alphas_cumprod_prev = net.alphas_prev
    num_timesteps = len(betas)
    
    # Reverse order: T-1 to 0
    steps = list(range(num_timesteps))[::-1]
    
    if subsample_steps is not None:
        skip = num_timesteps // subsample_steps
        steps = steps[::skip]

    images = []
    scorenet_fn = partial(scorenet, cond=cond)

    for i, step_idx in enumerate(steps):
        t = torch.full((x_mod.shape[0],), step_idx, device=x_mod.device, dtype=torch.long)
        
        beta_t = betas[step_idx]
        alpha_cumprod_t = alphas_cumprod[step_idx]
        alpha_cumprod_prev_t = alphas_cumprod_prev[step_idx]
        alpha_t = alpha_cumprod_t / alpha_cumprod_prev_t

        grad = scorenet_fn(x_mod, t) 

        sqrt_recip_alpha_bar = torch.rsqrt(alpha_cumprod_t)
        sqrt_recip_m1_alpha_bar = torch.sqrt(1. / alpha_cumprod_t - 1.)
        pred_x0 = sqrt_recip_alpha_bar * x_mod - sqrt_recip_m1_alpha_bar * grad
        
        if clip_before:
            pred_x0 = pred_x0.clamp(-1., 1.)

        coeff_x0 = (beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t))
        coeff_xt = ((1.0 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t))
        mean = coeff_x0 * pred_x0 + coeff_xt * x_mod

        if step_idx > 0:
            posterior_variance = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
            noise = torch.randn_like(x_mod) * temperature
            sigma = torch.exp(0.5 * posterior_log_variance_clipped)
            x_mod = mean + sigma * noise
        else:
            x_mod = mean

    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images) if len(images) > 0 else x_mod.unsqueeze(0)