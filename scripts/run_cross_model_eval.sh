#!/usr/bin/env bash
#SBATCH -J cross_model
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/cross_model_%j.out
#SBATCH -e logs/cross_model_%j.err

# Cross-model evaluation: test UQ model on Qwen answers
# Usage: sbatch scripts/run_cross_model_eval.sh

set -e

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p logs results figures

echo "=========================================="
echo "CROSS-MODEL EVALUATION"
echo "UQ model trained on Llama, tested on Qwen"
echo "Started at: $(date)"
echo "=========================================="

python cross_model_eval.py \
    --test_path data/finetune/test_v2.jsonl \
    --uq_model_path uq_models/llama-8b-uq-lora-v2 \
    --qwen_model Qwen/Qwen2.5-7B-Instruct \
    --base_model meta-llama/Llama-3.1-8B-Instruct

echo ""
echo "=========================================="
echo "COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
