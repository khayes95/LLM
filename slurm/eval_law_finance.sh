#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_law_finance_%j.out
#SBATCH --job-name=law_finance_eval

# Smoke test (uncomment for testing):
# CUDA_VISIBLE_DEVICES=0 python scripts/eval_law_finance.py --smoke_test

# Full run: generate with Qwen3.5-0.8B + score with calibrator
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_law_finance.py
