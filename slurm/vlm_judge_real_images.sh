#!/bin/bash
#SBATCH --job-name=vlm_judge_images
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/vlm_judge_images_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/vlm_judge_images_%j.log

# VLM judge cross-model evaluation WITH real images
# Runs 3 targets in parallel on separate GPUs: gpt5mini (GPU 0), gpt52 (GPU 1), qwen35 (GPU 2)

# Smoke test (uncomment to test):
# CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py --target gpt52 --smoke_test

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "VLM Judge Cross-Model Eval WITH Real Images"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

OUTPUT_DIR="data/cross_model"
mkdir -p "$OUTPUT_DIR"

# Run all 3 targets in parallel on separate GPUs
echo ""
echo "--- Starting GPT-5-mini eval on GPU 0 ---"
CUDA_VISIBLE_DEVICES=0 python scripts/vlm_judge_cross_model_eval.py \
    --target gpt5mini \
    --output "$OUTPUT_DIR/vlm_judge_images_on_gpt5mini.json" &
PID1=$!

echo "--- Starting GPT-5.2 eval on GPU 1 ---"
CUDA_VISIBLE_DEVICES=1 python scripts/vlm_judge_cross_model_eval.py \
    --target gpt52 \
    --output "$OUTPUT_DIR/vlm_judge_images_on_gpt52.json" &
PID2=$!

echo "--- Starting Qwen3.5 eval on GPU 2 ---"
CUDA_VISIBLE_DEVICES=2 python scripts/vlm_judge_cross_model_eval.py \
    --target qwen35 \
    --output "$OUTPUT_DIR/vlm_judge_images_on_qwen35.json" &
PID3=$!

echo ""
echo "PIDs: $PID1 (gpt5mini) $PID2 (gpt52) $PID3 (qwen35)"
echo "Waiting for all to complete..."

# Wait and track exit codes
wait $PID1 && RC1=0 || RC1=$?
echo "GPT-5-mini done: exit=$RC1 $(date)"

wait $PID2 && RC2=0 || RC2=$?
echo "GPT-5.2 done: exit=$RC2 $(date)"

wait $PID3 && RC3=0 || RC3=$?
echo "Qwen3.5 done: exit=$RC3 $(date)"

echo ""
echo "=========================================="
echo "Exit codes: gpt5mini=$RC1 gpt52=$RC2 qwen35=$RC3"
echo ""

# Print results summary
for target in "gpt5mini" "gpt52" "qwen35"; do
    results="$OUTPUT_DIR/vlm_judge_images_on_${target}.json"
    if [ -f "$results" ]; then
        echo "--- $target ---"
        python -c "
import json
with open('$results') as f:
    r = json.load(f)
print(f'  Overall AUROC: {r[\"auroc\"]:.4f}')
print(f'  VLM AUROC:     {r.get(\"vlm_auroc\", \"N/A\")}')
print(f'  Text AUROC:    {r.get(\"text_auroc\", \"N/A\")}')
print(f'  Real images:   {r.get(\"n_real_images\", 0)}')
print(f'  Gray images:   {r.get(\"n_gray_images\", 0)}')
print(f'  N samples:     {r[\"n_samples\"]}')
"
    else
        echo "--- $target: NO RESULTS (may have failed) ---"
    fi
done

echo ""
echo "Done: $(date)"
echo "=========================================="
