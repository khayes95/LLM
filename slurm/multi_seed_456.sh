#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --job-name=seed456
#SBATCH --output=logs/multi_seed_456_%j.out

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting multi-seed training (seed=456): $(date)"

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/retrain_best_v2.py \
    --output_dir data/ablations/multi_seed/seed_456 \
    --seed 456 \
    --epochs 3 \
    --lora_r 32

echo "Finished: $(date)"
