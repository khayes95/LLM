#!/bin/bash
#SBATCH --job-name=uq_smoke
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/best_uq_smoke_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/best_uq_smoke_%j.log

# Smoke test for best unified UQ model training

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Best UQ Model - Smoke Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python scripts/train_best_uq.py \
    --output_dir uq_models/best_unified_smoke \
    --smoke_test \
    --epochs 1 \
    --grad_accum 4

echo ""
echo "Done: $(date)"
echo "=========================================="
