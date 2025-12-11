import torch
import argparse

class Config:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Data Config
        self.data = type('DataConfig', (), {})()
        self.data.dataset = 'MOVINGMNIST'
        self.data.image_size = 64
        self.data.channels = 1
        self.data.logit_transform = False
        self.data.rescaled = False 
        self.data.num_workers = 4
        
        # Video Config
        self.data.num_frames = 10     
        self.data.num_frames_cond = 5 
        self.data.num_frames_future = 0 
        self.data.prob_mask_cond = 0.0  # 保持 0，先确保能跑通
        self.data.prob_mask_future = 0.0
        self.data.prob_mask_sync = False

        # Model Config
        self.model = type('ModelConfig', (), {})()
        self.model.type = 'simple'
        self.model.version = 'DDPM'
        self.model.arch = 'unetmore'
        self.model.ngf = 128
        self.model.depth = 'deep'
        self.model.num_classes = 1000 
        
        # --- CRITICAL CHANGE: LINEAR SCHEDULE ---
        self.model.sigma_dist = 'linear'  # 改为 linear
        self.model.sigma_begin = 1e-4     # 标准 DDPM 参数
        self.model.sigma_end = 0.02       # 标准 DDPM 参数
        # ----------------------------------------
        
        self.model.ema = True
        self.model.ema_rate = 0.999
        self.model.time_conditional = True
        self.model.gamma = False 
        self.model.dropout = 0.1
        self.model.output_all_frames = True 

        # Training Config
        self.training = type('TrainingConfig', (), {})()
        self.training.batch_size = 64
        self.training.n_epochs = 200
        self.training.n_iters = 200000
        self.training.snapshot_freq = 5000
        self.training.val_freq = 1000
        self.training.log_freq = 100
        self.training.loss_type = 'mse' 
        self.training.L1 = False
        self.training.log_all_sigmas = False

        # Optim Config
        self.optim = type('OptimConfig', (), {})()
        self.optim.lr = 2e-4
        self.optim.weight_decay = 0.000
        self.optim.beta1 = 0.9
        self.optim.amsgrad = False
        self.optim.eps = 1e-8
        self.optim.grad_clip = 1.0
        self.optim.warmup = 5000

        # Sampling Config
        self.sampling = type('SamplingConfig', (), {})()
        self.sampling.batch_size = 16
        self.sampling.data_init = False
        self.sampling.step_lr = 0.0000062
        self.sampling.n_steps_each = 1
        self.sampling.ckp_id = None
        self.sampling.final_only = True
        self.sampling.denoise = False   # 保持 False
        self.sampling.clip_before = True # 保持 True (现在是 x0 clipping)

def dict2namespace(config):
    return config