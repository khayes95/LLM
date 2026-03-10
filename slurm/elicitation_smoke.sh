#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/elicitation_smoke_%j.out
#SBATCH --job-name=elic_smk

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Elicitation smoke test — $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
    --strategy multi_sample --n_samples 3 --smoke_test \
    --output_dir data/ablations/elicitation/smoke
echo "Done — $(date)"
