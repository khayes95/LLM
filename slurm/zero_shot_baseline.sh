#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=zeroshot
#SBATCH --output=logs/zero_shot_baseline_%j.out

# Zero-shot baseline: Run base Qwen3-VL-8B WITHOUT LoRA on scored data
# Smoke test: CUDA_VISIBLE_DEVICES=0 python scripts/zero_shot_baseline.py --smoke_test

source ~/.bashrc
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,2

echo "Starting zero-shot baseline at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python scripts/zero_shot_baseline.py \
    --scored_dir data/use_cases/scored_unified \
    --batch_size 1

echo "Finished at $(date)"
