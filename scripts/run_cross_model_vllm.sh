#!/usr/bin/env bash
#SBATCH -J cross_vllm
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/cross_model_vllm_%j.out
#SBATCH -e logs/cross_model_vllm_%j.err

# Cross-model evaluation using vLLM
# Usage: sbatch scripts/run_cross_model_vllm.sh

set -e

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Use free GPUs (3-7 are free, 0-2 have jobs)
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

mkdir -p logs results figures

echo "=========================================="
echo "CROSS-MODEL EVALUATION (vLLM)"
echo "Started at: $(date)"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python cross_model_eval_vllm.py \
    --test_path data/finetune/test_v2.jsonl \
    --uq_model_path uq_models/llama-8b-uq-lora-v2 \
    --qwen_model Qwen/Qwen2.5-7B-Instruct \
    --base_model meta-llama/Llama-3.1-8B-Instruct

echo ""
echo "=========================================="
echo "COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
