#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/retrain_v3_qsplit_%j.out
#SBATCH --job-name=v3_qsplit

# Full retraining with question-level split fix (v3)
# Same config as best_v2_r32_combined but with leakage-free split
cd /scratch/khayes/LLM
source activate uq_eval

# Smoke test line (uncomment to test):
# CUDA_VISIBLE_DEVICES=0,1 python scripts/train_best_uq.py --output_dir uq_models/best_v3_qsplit_smoke --smoke_test --epochs 1 --lora_r 32 --lora_alpha 64 --learning_rate 1e-4 --prompt_variant combined

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_best_uq.py \
    --output_dir uq_models/best_v3_qsplit \
    --epochs 3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --prompt_variant combined

echo "=== Split info ==="
python -c "
import json
with open('uq_models/best_v3_qsplit/split_info.json') as f:
    info = json.load(f)
print(f'Train: {info[\"n_train\"]} samples, {info[\"n_train_questions\"]} questions')
print(f'Test:  {info[\"n_test\"]} samples, {info[\"n_test_questions\"]} questions')
print(f'Question overlap: {info[\"question_overlap\"]}')
print(f'Split method: {info[\"split_method\"]}')
"

echo "=== Results ==="
cat uq_models/best_v3_qsplit/results.json | python -m json.tool | head -30

echo "=== Done ==="
