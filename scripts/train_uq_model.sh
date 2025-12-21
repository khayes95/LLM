#!/usr/bin/env bash
#SBATCH -J uq_train
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_train_%j.out
#SBATCH -e logs/uq_train_%j.err

# Train UQ classifier model
# Usage: sbatch scripts/train_uq_model.sh [base_model] [runs_dir]

set -e

BASE_MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
RUNS_DIR="${2:-runs}"
OUTPUT_DIR="${3:-uq_models/uq_classifier_$(date +%Y%m%d_%H%M%S)}"

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs uq_models

echo "=========================================="
echo "UQ MODEL TRAINING"
echo "Started at: $(date)"
echo "Base model: $BASE_MODEL"
echo "Runs dir: $RUNS_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python -m uq_eval.uq_finetune \
    --base_model "$BASE_MODEL" \
    --runs_dir "$RUNS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 16

echo "=========================================="
echo "TRAINING COMPLETE"
echo "Finished at: $(date)"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
