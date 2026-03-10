#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=unseen_smk
#SBATCH --output=logs/unseen_model_smoke_%j.out

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting unseen model eval SMOKE TEST: $(date)"

# GPU 0: LLaMA generation, GPU 1: calibrator scoring
CUDA_VISIBLE_DEVICES=2,3 python scripts/unseen_model_eval.py \
    --smoke_test \
    --llama_gpu 0 \
    --calibrator_gpu 1 \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/unseen_model_smoke

echo "Finished: $(date)"
