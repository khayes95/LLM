#!/bin/bash
#SBATCH --job-name=uq_size_ablation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/size_ablation_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/size_ablation_%j.log

# Model size ablation: train UQ calibrators at 0.5B, 1.5B, 3B, 7B
# All on GPT-5-mini data for a clean comparison (same data, different model size)
# Night run: using GPUs 0-7

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "UQ Calibrator Size Ablation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

DATA_DIR="runs/gpt5_mini_combined"
BASE_OUTPUT="uq_models/size_ablation"
mkdir -p "$BASE_OUTPUT"

# Train 4 models in parallel, 2 GPUs each
# 0.5B on GPUs 0-1
echo ""
echo "--- Starting 0.5B training on GPUs 0-1 ---"
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_uq_unified.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT/qwen25_0.5b" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --learning_rate 2e-5 &
PID1=$!

# 1.5B on GPUs 2-3
echo "--- Starting 1.5B training on GPUs 2-3 ---"
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_uq_unified.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT/qwen25_1.5b" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 &
PID2=$!

# 3B on GPUs 4-5
echo "--- Starting 3B training on GPUs 4-5 ---"
CUDA_VISIBLE_DEVICES=4,5 python scripts/train_uq_unified.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT/qwen25_3b" \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 &
PID3=$!

# 7B on GPUs 6-7 (re-train with same seed for consistent comparison)
echo "--- Starting 7B training on GPUs 6-7 ---"
CUDA_VISIBLE_DEVICES=6,7 python scripts/train_uq_unified.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT/qwen25_7b" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-5 &
PID4=$!

echo ""
echo "PIDs: $PID1 (0.5B) $PID2 (1.5B) $PID3 (3B) $PID4 (7B)"
echo "Waiting for all to complete..."

# Wait for each and track exit codes
wait $PID1 && RC1=0 || RC1=$?
echo "0.5B done: exit=$RC1 $(date)"

wait $PID2 && RC2=0 || RC2=$?
echo "1.5B done: exit=$RC2 $(date)"

wait $PID3 && RC3=0 || RC3=$?
echo "3B done: exit=$RC3 $(date)"

wait $PID4 && RC4=0 || RC4=$?
echo "7B done: exit=$RC4 $(date)"

echo ""
echo "=========================================="
echo "Exit codes: 0.5B=$RC1 1.5B=$RC2 3B=$RC3 7B=$RC4"
echo ""

# Print results summary
for size in "qwen25_0.5b" "qwen25_1.5b" "qwen25_3b" "qwen25_7b"; do
    results="$BASE_OUTPUT/$size/results.json"
    if [ -f "$results" ]; then
        echo "--- $size ---"
        python -c "
import json
with open('$results') as f:
    r = json.load(f)
print(f'  AUROC: {r[\"auroc\"]:.4f}')
print(f'  AUPRC: {r[\"auprc\"]:.4f}')
print(f'  ECE:   {r[\"ece\"]:.4f}')
print(f'  Brier: {r[\"brier\"]:.4f}')
print(f'  N:     {r[\"n_samples\"]}')
"
    else
        echo "--- $size: NO RESULTS (training may have failed) ---"
    fi
done

echo ""
echo "Done: $(date)"
echo "=========================================="
