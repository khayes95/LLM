#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --job-name=fg_uq_new
#SBATCH --output=logs/fg_uq_newmodels_%j.out

# Smoke test (uncomment to run quick test):
# CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py --smoke_test

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=========================================="
echo "FineGRAIN UQ: New Model Evaluation"
echo "Started: $(date)"
echo "=========================================="

# Score all 15 models with LLM judge labels
# Use best prompt variant from ablation (default: combined; update after Part 1)
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py \
    --all \
    --prompt_variant combined \
    --output_dir data/finegrain_uq/newmodels

echo ""
echo "=========================================="
echo "All models scored: $(date)"
echo "=========================================="
