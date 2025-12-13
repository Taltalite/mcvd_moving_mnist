uv run python train.py --log_dir /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment \
 --save_interval 10 --batch_size 64 --num_workers 4 \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST

uv run python train.py --log_dir /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear \
 --save_interval 10 --batch_size 64 --num_workers 8 \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST


uv run python train.py --log_dir /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear_joint \
 --save_interval 10 --batch_size 64 --num_workers 4 \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST


uv run python train.py --log_dir /data/biolab-nvme-pcie2/lijy/video-diff-ckpt/mnist_experiment_debug_cosine \
 --save_interval 10 --batch_size 64 --num_workers 4 \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST


uv run python train.py --log_dir /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_cosine_2 \
 --save_interval 20 --batch_size 64 --num_workers 4 \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST

# ================== TESTING SCRIPTS ==================

uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment/ckpt_epoch_40.pt \
 --output_dir ./outputs/mnist_experiment_vis \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --num_samples 1


uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_2/ckpt_epoch_90.pt \
 --output_dir ./outputs/mnist_experiment_vis_2/ \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --num_samples 1


uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_3/latest.pt \
 --output_dir ./outputs/mnist_experiment_vis_3/ \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --num_samples 1

uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear/latest.pt \
 --output_dir ./outputs/mnist_experiment_vis_debug_linear/ \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --num_samples 1

uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_linear_joint/latest.pt \
 --output_dir ./outputs/mnist_experiment_vis_debug_linear_joint/ \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --temperature 0.9 \
 --num_samples 5

uv run python test.py \
  --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_cosine_2/ckpt_epoch_400.pt \
  --output_dir ./outputs/mnist_experiment_debug_cosine_2 \
  --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
  --temperature 0.9 \
  --num_samples 20


uv run python test.py \
  --ckpt_path /data/biolab-nvme-pcie2/lijy/video-diff-ckpt/mnist_experiment_debug_cosine_2/ckpt_epoch_500.pt \
  --output_dir ./outputs/mnist_experiment_debug_cosine_2_1e-6 \
  --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
  --temperature 0.9 \
  --num_samples 5

uv run python test.py \
  --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_cosine_2/val_best_perf.pt \
  --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
  --output_dir ./outputs/mnist_experiment_debug_cosine_2_smooth \
  --temperature 1.0 \
  --num_samples 5


uv run python test.py \
  --ckpt_path /data/biolab-nvme-pcie2/lijy/video-diff-ckpt/mnist_experiment_debug_cosine_2/ckpt_epoch_500.pt \
  --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
  --output_dir ./outputs/mnist_experiment_debug_cosine_2_aggr \
  --temperature 1.0 \
  --num_samples 3


uv run python test.py \
  --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment_debug_cosine_2/latest.pt \
  --output_dir ./outputs/mnist_experiment_debug_cosine_2_smooth \
  --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
  --temperature 0.9 \
  --num_samples 30
