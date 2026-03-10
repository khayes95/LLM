#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=mseed_v3
#SBATCH --output=logs/multi_seed_v3_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=0 python scripts/multi_seed_training.py \
#   --output_dir data/ablations/multi_seed_v3 --smoke_test --seeds 42

source activate uq_eval

echo "=== Multi-Seed Training (v3 question-level split, 5 seeds) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/multi_seed_training.py \
    --output_dir data/ablations/multi_seed_v3 \
    --seeds 123 456 789 314 \
    --epochs 3 \
    --lora_r 32

echo "Done: $(date)"
