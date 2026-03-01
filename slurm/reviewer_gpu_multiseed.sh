#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=multi_seed
#SBATCH --output=logs/multi_seed_%j.out

# Multi-seed training: 3 seeds for error bars
# Estimated: 7-9 hours (3 seeds × ~2.5 hours each)
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/multi_seed_training.py --smoke_test

source ~/.bashrc
conda activate uq_eval
cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,4,5,7

echo "============================================"
echo "MULTI-SEED TRAINING — $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python scripts/multi_seed_training.py \
    --output_dir data/ablations/multi_seed \
    --seeds 42 123 456 \
    --epochs 3 \
    --lora_r 32

echo ""
echo "============================================"
echo "MULTI-SEED COMPLETE — $(date)"
echo "============================================"
