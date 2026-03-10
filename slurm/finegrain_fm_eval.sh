#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/finegrain_fm_eval_%j.out

# Smoke test:
# CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_failure_mode_eval.py --smoke_test

source activate uq_eval

CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_failure_mode_eval.py \
    --output_dir data/finegrain_uq/failure_mode_eval
