#!/usr/bin/env bash
#SBATCH -J uq_finetune
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/uq_finetune_%j.out
#SBATCH -e logs/uq_finetune_%j.err

# LoRA finetuning for UQ classifier
# Following Kapoor et al. (2024) "LLMs Must Be Taught to Know What They Don't Know"
# Usage: sbatch scripts/run_finetune.sh

set -e

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Use GPUs 4,5,6,7 (free)
export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p logs uq_models

echo "=========================================="
echo "UQ MODEL FINETUNING (LoRA + Prompt)"
echo "Following Kapoor et al. (2024)"
echo "Started at: $(date)"
echo "=========================================="

# Show data stats
echo "Training data:"
wc -l data/finetune/train.jsonl data/finetune/val.jsonl

echo ""
echo "Hyperparameters (from paper):"
echo "  - LoRA rank: 8"
echo "  - Learning rate: 1e-4"
echo "  - Batch size: 4 x 8 = 32 effective"
echo "  - Scheduler: cosine"
echo ""

echo "Starting finetuning..."

python -m uq_eval.uq_finetune \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --train_file data/finetune/train.jsonl \
    --val_file data/finetune/val.jsonl \
    --output_dir uq_models/llama-8b-uq-lora \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-4 \
    --lora_r 8 \
    --calibrate

echo ""
echo "=========================================="
echo "FINETUNING COMPLETE"
echo "Finished at: $(date)"
echo "Model saved to: uq_models/llama-8b-uq-lora"
echo "=========================================="
