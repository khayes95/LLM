#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/elicitation_contrastive_resume_%j.out
#SBATCH --job-name=elic_ctr

# Resume contrastive training from last checkpoint

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Contrastive (resume) — $(date)"
CUDA_VISIBLE_DEVICES=4 python scripts/elicitation_ablations.py \
    --strategy contrastive \
    --epochs 3 --learning_rate 1e-4 \
    --output_dir data/ablations/elicitation/contrastive \
    --resume_from_checkpoint
echo "Done: $(date)"
