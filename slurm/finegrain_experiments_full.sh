#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=fg_expts
#SBATCH --output=logs/fg_experiments_full_%j.out

cd /scratch/khayes/LLM
export CUDA_VISIBLE_DEVICES=0,1

echo "=== Starting FineGRAIN experiments at $(date) ==="

# Run all experiments sequentially (each uses 1 GPU, models are loaded/unloaded)
/scratch/khayes/.conda/envs/uq_eval/bin/python scripts/finegrain_all_experiments.py \
    --experiment all \
    --device cuda:0 \
    --output_dir data/finegrain_uq/experiments

echo "=== Finished at $(date) ==="
