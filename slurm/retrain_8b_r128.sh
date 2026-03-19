#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=18:00:00
#SBATCH --output=logs/retrain_8b_r128_%j.out
#SBATCH --job-name=q3vl_r128

# Retrain Qwen3-VL-8B with r=128 LoRA (up from r=32)
# Same v3 data + easy/impossible/adversarial augmentation
# Tests whether larger LoRA improves headline AUROC

set -e
cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "Start: $(date)"

SPLIT_INFO="uq_models/best_v3_qsplit/split_info.json"
OUTPUT_DIR="uq_models/best_8b_r128"

python scripts/train_best_uq.py \
    --output_dir "$OUTPUT_DIR" \
    --split_info "$SPLIT_INFO" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 5e-5 \
    --lora_r 128 \
    --lora_alpha 256 \
    --prompt_variant combined \
    --extra_data data/finetune/easy_questions/all_easy.jsonl \
                 data/finetune/easy_questions/impossible_questions.jsonl \
                 data/finetune/easy_questions/trivial_correct.jsonl \
                 data/finetune/easy_questions/adversarial_incorrect.jsonl \
    --extra_max_samples 2000

echo ""
echo "End: $(date)"

if [ -f "$OUTPUT_DIR/results.json" ]; then
    echo "Results:"
    cat "$OUTPUT_DIR/results.json"
fi
