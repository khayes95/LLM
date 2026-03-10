#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=trunc_scr
#SBATCH --output=logs/ablation_trunc_scramble_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py --smoke_test

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting truncation + scramble ablation: $(date)"

CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/truncation_scramble

echo "Finished: $(date)"
