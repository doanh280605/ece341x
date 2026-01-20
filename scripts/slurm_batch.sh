#!/usr/bin/env bash
#SBATCH --job-name=ece341x_lab1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:50:00
#SBATCH --output=results/slurm-%j.out

set -euo pipefail

echo "Starting batch job on $(hostname) at $(date)"
nvidia-smi || true

# If your cluster uses modules:
module load cuda || true
module load gcc || true

# Activate conda env (adjust if your cluster uses a different conda setup)
source ~/.bashrc || true
conda activate ece341x-lab1

python python/bench_matmul.py --sizes 256 512 1024 2048 --dtype fp32 --trials 20 --warmup 5
python python/bench_images.py --repeat 4096 --dtype fp32 --trials 50 --warmup 10

echo "Done at $(date)"
