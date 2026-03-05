#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen35_full_ablation_%j.out
#SBATCH --job-name=q35_full

# Qwen3.5 Model Size Ablation - Full Training
# Models: 0.8B, 2B, 4B, 9B (sequential, ~6-7 hours total)
# Uses split_info to prevent data leakage

set -e

cd /scratch/khayes/LLM

# Activate conda
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=2,3,4

echo "Start time: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

python scripts/qwen35_size_ablation.py \
    --models 0.8b,2b,4b,9b \
    --epochs 3

echo ""
echo "End time: $(date)"
