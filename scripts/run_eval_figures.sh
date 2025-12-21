#!/usr/bin/env bash
#SBATCH -J uq_eval_fig
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/uq_eval_figures_%j.out
#SBATCH -e logs/uq_eval_figures_%j.err

# Evaluate UQ methods and generate figures for paper
# Usage: sbatch scripts/run_eval_figures.sh [model_path] [test_path]

set -e

MODEL_PATH="${1:-uq_models/llama-8b-uq-lora}"
TEST_PATH="${2:-data/finetune/val.jsonl}"

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p logs results figures

echo "=========================================="
echo "UQ EVALUATION AND FIGURES"
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_PATH"
echo "=========================================="

python evaluate_and_plot.py \
    --model_path "$MODEL_PATH" \
    --test_path "$TEST_PATH" \
    --base_model meta-llama/Llama-3.1-8B-Instruct

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "Results in: results/ and figures/"
echo "=========================================="
