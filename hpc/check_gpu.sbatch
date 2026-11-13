#!/bin/bash
# A tiny GPU job: about one minute on one GPU.
# Submit from the repository root:   sbatch hpc/check_gpu.sbatch
# The output appears in this folder:  gpu-check-<job id>.out

#SBATCH --job-name=gpu-check
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gpu-check-%j.out

source .venv/bin/activate

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== check_gpu.py ====="
python hpc/check_gpu.py
