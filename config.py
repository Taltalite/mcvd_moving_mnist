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
        
        # --- CRITICAL CHANGE ---
        # 暂时将其设为 0，确保模型先学会基于条件生成，排除 mask 造成的干扰
        self.data.prob_mask_cond = 0.0  
        # -----------------------
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
        self.model.sigma_dist = 'cosine'
        self.model.sigma_begin = 1e-4
        self.model.sigma_end = 0.02
        self.model.ema = True
        self.model.ema_rate = 0.999
        self.model.time_conditional = True
        self.model.gamma = False 
        self.model.dropout = 0.1
        self.model.output_all_frames = True 

        # Training Config
        self.training = type('TrainingConfig', (), {})()
        self.training.batch_size = 64 # 建议调小一点，确保显存充裕，128可能在某些24G环境勉强
        self.training.n_epochs = 50
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
        
        # --- CRITICAL NOTE ---
        # 由于我们修改了 Loss 为 mean，现在的 grad_clip=1.0 是合理的。
        # 如果你坚持用 sum loss，这里需要改为 1000.0 或更高。
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
        self.sampling.denoise = True
        self.sampling.clip_before = True

def dict2namespace(config):
    return config