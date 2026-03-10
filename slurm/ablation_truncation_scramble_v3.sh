#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=trunc_v3
#SBATCH --output=logs/truncation_scramble_v3_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=4 python scripts/ablation_truncation_scramble.py \
#   --checkpoint uq_models/best_v3_qsplit --output_dir data/ablations/truncation_scramble_v3 --smoke_test

source activate uq_eval

echo "=== Truncation & Scramble Ablation (v3 checkpoint) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=4 python scripts/ablation_truncation_scramble.py \
    --checkpoint uq_models/best_v3_qsplit \
    --output_dir data/ablations/truncation_scramble_v3

echo "Done: $(date)"
