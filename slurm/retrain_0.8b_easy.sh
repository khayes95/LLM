#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/retrain_0.8b_easy_%j.out
#SBATCH --job-name=q35_easy

# Retrain Qwen3.5-0.8B with easy + impossible questions mixed in
# Smoke test: sbatch --export=ALL,SMOKE=1 slurm/retrain_0.8b_easy.sh

set -e
cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Start: $(date)"
echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

SPLIT_INFO="uq_models/best_v3_qsplit/split_info.json"
OUTPUT_DIR="uq_models/best_0.8b_easy_v3"

EXTRA_ARGS=""
if [ "${SMOKE}" = "1" ]; then
    echo "=== SMOKE TEST MODE ==="
    EXTRA_ARGS="--smoke_test"
    OUTPUT_DIR="uq_models/best_0.8b_easy_v3_smoke"
fi

python scripts/train_best_uq.py \
    --output_dir "$OUTPUT_DIR" \
    --base_model "Qwen/Qwen3.5-0.8B" \
    --split_info "$SPLIT_INFO" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 32 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --prompt_variant combined \
    --extra_data data/finetune/easy_questions/all_easy.jsonl \
                 data/finetune/easy_questions/impossible_questions.jsonl \
                 data/finetune/easy_questions/trivial_correct.jsonl \
                 data/finetune/easy_questions/adversarial_incorrect.jsonl \
    --extra_max_samples 2000 \
    --randomize_extra_metadata \
    $EXTRA_ARGS

echo ""
echo "End: $(date)"

if [ -f "$OUTPUT_DIR/results.json" ]; then
    echo "Results:"
    cat "$OUTPUT_DIR/results.json"
fi
