#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/elicitation_multi_sample_%j.out
#SBATCH --job-name=elic_mc

# MC Dropout multi-sample consistency (inference-only, no training)
# Runs N=5 forward passes with dropout active on existing best_unified checkpoint

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "MC Dropout Multi-Sample (N=5, temp=1.0)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
    --strategy multi_sample \
    --n_samples 5 --temperature 1.0 \
    --checkpoint uq_models/best_unified \
    --output_dir data/ablations/elicitation/multi_sample_5

echo "Done: $(date)"
