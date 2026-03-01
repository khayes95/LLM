#!/bin/bash
#SBATCH --job-name=score_smoke
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/khayes/LLM/logs/score_unified_smoke_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/score_unified_smoke_%j.log

# Smoke test for unified scoring

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Scoring Smoke Test - Unified UQ Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py \
    --target gpt5mini \
    --checkpoint uq_models/best_unified \
    --output_dir data/use_cases/scored_unified_smoke \
    --smoke_test

echo ""
echo "Done: $(date)"
echo "=========================================="
