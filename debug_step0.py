import torch
import matplotlib.pyplot as plt
from config import Config
from model.unet import UNet_DDPM
from datasets.moving_mnist import data_transform, inverse_data_transform

# 加载配置和模型
config = Config()
device = config.device
scorenet = UNet_DDPM(config).to(device)
scorenet = torch.nn.DataParallel(scorenet)
ckpt = torch.load('/home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment/ckpt_epoch_40.pt', map_location=device)
scorenet.load_state_dict(ckpt['model_state']) # 先试主权重
scorenet.eval()

# 构造一个全噪输入 (模拟 t=0, 即最高噪声)
# 注意：根据你的代码，index 0 是高噪声
x_noise = torch.randn(1, 10, 64, 64).to(device) # 10个预测帧通道
cond = torch.zeros(1, 5, 64, 64).to(device) # 模拟条件

# 构造 Label 0 (对应 Alpha 最小，噪声最大)
labels = torch.zeros(1, dtype=torch.long).to(device)

# 预测噪声
with torch.no_grad():
    # scorenet 输出的是 predicted_noise (z)
    pred_noise = scorenet(x_noise, labels, cond=cond)
    
    # 尝试直接用 DDPM 公式一步还原 x0
    # x0 = (x_t - sqrt(1-alpha)*z) / sqrt(alpha)
    # alpha[0] 应该很小，例如 0.0001
    alpha_0 = scorenet.module.alphas[0]
    print(f"Alpha at step 0: {alpha_0.item()}")
    
    x0_pred = (x_noise - (1 - alpha_0).sqrt() * pred_noise) / alpha_0.sqrt()
    
    # 归一化并显示
    x0_img = inverse_data_transform(config, x0_pred[0, 0]) # 看第1帧
    
plt.imshow(x0_img.cpu().numpy(), cmap='gray')
plt.title("Step 0 Prediction of x0")
plt.savefig("debug_x0.png")