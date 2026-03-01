#!/bin/bash
#SBATCH --job-name=score_uq
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/score_unified_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/score_unified_%j.log

# Score all 3 targets with the unified UQ model
# ~30-45 min per target on 1 GPU, ~1.5-2 hours total

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Scoring All Targets with Unified UQ Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py \
    --target all \
    --checkpoint uq_models/best_unified \
    --output_dir data/use_cases/scored_unified

echo ""
echo "Done: $(date)"
echo "=========================================="
