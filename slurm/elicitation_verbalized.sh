#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/elicitation_verbalized_%j.out
#SBATCH --job-name=elic_vrb

# Verbalized judge: output calibrated probability text (e.g. "0.73")
# instead of binary (i)/(ii). Requires retraining.

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Verbalized Judge Confidence"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=5 python scripts/elicitation_ablations.py \
    --strategy verbalized \
    --epochs 3 --learning_rate 1e-4 \
    --output_dir data/ablations/elicitation/verbalized

echo "Done: $(date)"
