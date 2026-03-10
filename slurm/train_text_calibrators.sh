#!/bin/bash
#SBATCH --job-name=train_uq
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_uq_%j.log
#SBATCH --error=logs/train_uq_%j.log

# Train two text UQ calibrators:
#   A) GPT-5.2 only (in-distribution baseline)
#   B) Combined GPT-5-mini + GPT-5.2 (universal calibrator)
#
# Uses GPUs 0-3 (daytime safe)
# Each training run ~15 min -> total ~30 min
#
# Smoke test: sbatch --export=SMOKE_TEST=1 slurm/train_text_calibrators.sh

echo "=========================================="
echo "UQ Calibrator Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Smoke test: ${SMOKE_TEST:-0}"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3

source ~/.bashrc
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

# Smoke test uses 1 epoch, 5 samples
EXTRA_ARGS=""
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    EXTRA_ARGS="--epochs 1"
fi

echo ""
echo "=========================================="
echo "Training A: GPT-5.2 only calibrator"
echo "Start: $(date)"
echo "=========================================="

python scripts/train_uq_unified.py \
    --data_dir runs/gpt52_high_combined \
    --output_dir uq_models/text_calibrator_gpt52 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 \
    $EXTRA_ARGS

RC_A=$?
echo "Training A exit code: $RC_A"
echo "Training A done: $(date)"

echo ""
echo "=========================================="
echo "Training B: Combined GPT-5-mini + GPT-5.2 calibrator"
echo "Start: $(date)"
echo "=========================================="

python scripts/train_uq_unified.py \
    --data_dir runs/all_models_combined \
    --output_dir uq_models/text_calibrator_combined \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 \
    $EXTRA_ARGS

RC_B=$?
echo "Training B exit code: $RC_B"
echo "Training B done: $(date)"

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Training A (GPT-5.2 only): exit=$RC_A"
echo "Training B (Combined): exit=$RC_B"
echo ""
echo "Checkpoints:"
ls -la uq_models/text_calibrator_gpt52/results.json 2>/dev/null
ls -la uq_models/text_calibrator_combined/results.json 2>/dev/null
echo ""
echo "Done: $(date)"
