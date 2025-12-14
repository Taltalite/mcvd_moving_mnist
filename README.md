# MCVD Reproduction on Moving MNIST

This repository contains a PyTorch reproduction and analysis of **Masked Conditional Video Diffusion (MCVD)** for stochastic video prediction.

We focus on the **Moving MNIST** dataset, predicting the next **10 frames** given the past **5 frames**. This project also investigates the impact of Learning Rate scheduling and analyzes the trade-off between inference speed and generation quality using **DDPM** vs. **DDIM** samplers.

## Features

*   **Conditional U-Net**: Implementation of a 2D U-Net with channel concatenation for spatiotemporal modeling.
*   **Training Dynamics**: Implemented an aggressive learning rate decay strategy for stable convergence.
*   **Flexible Sampling**: Supports both:
    *   **DDPM** (1000 steps): High fidelity, slower inference.
    *   **DDIM** (50 steps): Fast inference, lower fidelity.
*   **Metrics**: Automated calculation of MSE, PSNR, SSIM, and LPIPS.

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Taltalite/mcvd_moving_mnist.git
    cd mcvd_moving_mnist
    ```

2.  **Install dependencies**
    ```bash
    pip install torch torchvision numpy matplotlib tqdm imageio torchmetrics lpips
    ```

## Usage

### 1. Data Preparation
The code automatically downloads the Moving MNIST dataset (`mnist_test_seq.npy`) to the specified `data_path` on the first run.

### 2. Training
To train the model from scratch:

```bash
python train.py --data_path ./data --log_dir ./logs/mnist_experiment --batch_size 64
```

*   **Note**: The training script includes a "Warmup + ReduceLROnPlateau" scheduler strategy which is crucial for convergence.

### 3. Testing & Inference
You can choose between **DDPM** (High Quality) and **DDIM** (Fast Speed) samplers.

**High Quality (Standard DDPM, 1000 steps):**
```bash
python test.py \
  --ckpt_path ./logs/mnist_experiment/val_best_perf.pt \
  --data_path ./data \
  --output_dir ./results/ddpm \
  --sampler ddpm \
  --num_samples 30
```

**Fast Inference (DDIM, 50 steps):**
```bash
python test.py \
  --ckpt_path ./logs/mnist_experiment/val_best_perf.pt \
  --data_path ./data \
  --output_dir ./results/ddim \
  --sampler ddim \
  --steps 50 \
  --num_samples 30
```

## Experimental Results

We evaluated the trade-off between speed and quality on an Nvidia RTX 3090:

| Sampler | Steps | Inference Time (s) | SSIM $\uparrow$ | LPIPS $\downarrow$ | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDPM** | 1000 | ~10.37s | **0.804** | **0.134** | Best Quality |
| **DDIM** | 50 | **~0.52s** | 0.299 | 0.343 | 20x Faster (Blurry) |

## References

1.  **MCVD**: Voleti et al., "MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation", NeurIPS 2022.
2.  **Moving MNIST**: Srivastava et al., "Unsupervised Learning of Video Representations using LSTMs", ICML 2015.
3.  **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
