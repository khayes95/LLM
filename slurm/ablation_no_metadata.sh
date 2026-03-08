#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#No --mem for GPU partition
#SBATCH --time=04:00:00
#SBATCH --job-name=no_meta_abl
#SBATCH --output=logs/ablation_no_metadata_%j.out

# Smoke test (uncomment to test):
# CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py --smoke_test

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting no-metadata ablation: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/no_metadata

echo "Finished: $(date)"
