#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/elicitation_contrastive_%j.out
#SBATCH --job-name=elic_ctr

# Contrastive prompting: include gold/reference answer in the prompt
# Requires retraining from scratch with new prompt template

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Contrastive Prompting (with reference answers)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=4 python scripts/elicitation_ablations.py \
    --strategy contrastive \
    --epochs 3 --learning_rate 1e-4 \
    --output_dir data/ablations/elicitation/contrastive

echo "Done: $(date)"
