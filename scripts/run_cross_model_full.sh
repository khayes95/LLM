#!/usr/bin/env bash
#SBATCH -J cross_full
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/cross_model_full_%j.out
#SBATCH -e logs/cross_model_full_%j.err

# Full cross-model evaluation using existing runs
# Usage: sbatch scripts/run_cross_model_full.sh

set -e

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Use free GPUs
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

mkdir -p logs results figures

echo "=========================================="
echo "CROSS-MODEL EVALUATION (Full)"
echo "Using existing Qwen runs"
echo "Started at: $(date)"
echo "=========================================="

python cross_model_full.py \
    --runs_dir runs \
    --uq_model_path uq_models/llama-8b-uq-lora-v2 \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --max_per_benchmark 200

echo ""
echo "=========================================="
echo "COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
