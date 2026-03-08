#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_retrain_qsplit_%j.out
#SBATCH --job-name=smoke_qsplit

# Smoke test: retrain with question-level split fix
cd /scratch/khayes/LLM
source activate uq_eval

CUDA_VISIBLE_DEVICES=0,1 python scripts/train_best_uq.py \
    --output_dir uq_models/best_v3_qsplit_smoke \
    --smoke_test \
    --epochs 1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --prompt_variant combined

echo "=== Split info ==="
cat uq_models/best_v3_qsplit_smoke/split_info.json | python -m json.tool | head -20

echo "=== Done ==="
