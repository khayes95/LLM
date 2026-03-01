#!/bin/bash
#SBATCH --job-name=train_qwen35_uq
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/train_qwen35_uq_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/train_qwen35_uq_%j.log

# Train a Qwen3.5-specific UQ calibrator
# Uses Qwen2.5-7B-Instruct + LoRA, trained on Qwen3.5-397B prediction data
# Expected: ~3,500 training samples after exclusions, ~15 min training on 3 GPUs

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

# Use GPUs 1-3 (GPU 0 may be in use by other users during daytime)
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training Qwen3.5-Specific UQ Calibrator"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

python scripts/train_uq_unified.py \
    --data_dir runs/qwen35_combined \
    --output_dir uq_models/text_calibrator_qwen35 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 \
    --seed 42

echo ""
echo "Done: $(date)"
