import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets.utils import download_url

class MovingMNIST(data.Dataset):
    """
    Moving MNIST Dataset.
    Format: [Seq_Len, Batch, Channels, H, W] in the original file.
    We transform it to [Batch, Seq_Len, Channels, H, W].
    """
    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"

    def __init__(self, root, train=True, transform=None, seed=42):
        self.root = root
        self.transform = transform
        self.file_path = os.path.join(root, 'mnist_test_seq.npy')
        
        if not os.path.exists(self.file_path):
            os.makedirs(root, exist_ok=True)
            print("Downloading Moving MNIST...")
            download_url(self.url, root, "mnist_test_seq.npy", md5=None)

        # Shape: [20, 10000, 64, 64]
        self.data = np.load(self.file_path)
        
        # Split train/test
        # Usually standard split is first 9000 train, 1000 test if not specified differently
        # But this file is technically the "test set" of the original paper. 
        # For MCVD reproduction purposes, we just split this file.
        total = self.data.shape[1]
        train_size = int(total * 0.9)
        
        if train:
            self.data = self.data[:, :train_size, ...]
        else:
            self.data = self.data[:, train_size:, ...]

        # Transpose to [N, T, H, W] -> Add channel -> [N, T, C, H, W]
        self.data = self.data.transpose(1, 0, 2, 3)
        self.data = np.expand_dims(self.data, axis=2) 
        
        # Normalize to [0, 1] float32
        self.data = self.data.astype(np.float32) / 255.0

    def __getitem__(self, index):
        # Return x, y (y is dummy label 0)
        x = torch.from_numpy(self.data[index])
        if self.transform:
            x = self.transform(x)
        return x, 0

    def __len__(self):
        return self.data.shape[0]

def data_transform(config, x):
    # MCVD expects [-1, 1] usually if not specified otherwise in config
    if not config.data.rescaled:
        return 2. * x - 1.
    return x

def inverse_data_transform(config, x):
    if not config.data.rescaled:
        return (x + 1.) / 2.
    return x