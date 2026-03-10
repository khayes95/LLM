#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --job-name=unseen_ev
#SBATCH --output=logs/unseen_model_eval_%j.out

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting unseen model eval (LLaMA-3.1-8B): $(date)"

# GPU 2: LLaMA generation, GPU 3: calibrator scoring
# 100 examples per benchmark x 8 benchmarks = ~800 samples
CUDA_VISIBLE_DEVICES=2,3 python scripts/unseen_model_eval.py \
    --llama_gpu 0 \
    --calibrator_gpu 1 \
    --max_per_benchmark 100 \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/unseen_model

echo "Finished: $(date)"
