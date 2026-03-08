#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_prbench_legal_%j.out
#SBATCH --job-name=score_prb

# Smoke test: CUDA_VISIBLE_DEVICES=1 python scripts/score_prbench_legal.py --smoke_test

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate uq_eval

echo "=========================================="
echo "Score PRBench Legal with Calibrator"
echo "Start: $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python scripts/score_prbench_legal.py

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
