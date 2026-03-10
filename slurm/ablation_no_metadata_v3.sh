#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=no_meta_v3
#SBATCH --output=logs/no_metadata_v3_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py \
#   --checkpoint uq_models/best_v3_qsplit --output_dir data/ablations/no_metadata_v3 --smoke_test

source activate uq_eval

echo "=== No-Metadata Ablation (v3 checkpoint) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py \
    --checkpoint uq_models/best_v3_qsplit \
    --output_dir data/ablations/no_metadata_v3

echo "Done: $(date)"
