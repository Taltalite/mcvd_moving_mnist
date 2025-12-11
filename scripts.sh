uv run python train.py --log_dir /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment \
 --save_interval 10 --batch_size 64 --num_workers 4 \
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

uv run python test.py --ckpt_path /home/lijy/workspace/video-diff-worldmodel/logs/mnist_experiment/latest.pt \
 --output_dir ./outputs/mnist_experiment_vis/ \
 --data_path /data/biolab-nvme-pcie2/lijy/MCVD/Datasets/MNIST \
 --num_samples 2