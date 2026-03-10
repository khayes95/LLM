#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=fg_smoke
#SBATCH --output=logs/fg_experiments_smoke_%j.out

cd /scratch/khayes/LLM
export CUDA_VISIBLE_DEVICES=0

/scratch/khayes/.conda/envs/uq_eval/bin/python scripts/finegrain_all_experiments.py \
    --experiment all \
    --smoke_test \
    --device cuda:0 \
    --output_dir data/finegrain_uq/experiments_smoke
