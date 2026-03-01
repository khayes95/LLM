#!/bin/bash
#SBATCH --job-name=best_uq
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/best_uq_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/best_uq_%j.log

# Train the BEST unified UQ model on ALL data (text + VLM with real images)
# Uses 4 GPUs, ~2-3 hours estimated

# Smoke test (uncomment):
# CUDA_VISIBLE_DEVICES=0 python scripts/train_best_uq.py --output_dir uq_models/best_unified_smoke --smoke_test

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Training Best Unified UQ Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_best_uq.py \
    --output_dir uq_models/best_unified \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --test_fraction 0.15

echo ""
echo "Done: $(date)"
echo "=========================================="
