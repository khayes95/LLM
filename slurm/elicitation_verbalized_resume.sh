#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/elicitation_verbalized_resume_%j.out
#SBATCH --job-name=elic_vrb

# Resume verbalized training from last checkpoint

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Verbalized (resume) — $(date)"
CUDA_VISIBLE_DEVICES=5 python scripts/elicitation_ablations.py \
    --strategy verbalized \
    --epochs 3 --learning_rate 1e-4 \
    --output_dir data/ablations/elicitation/verbalized \
    --resume_from_checkpoint
echo "Done: $(date)"
