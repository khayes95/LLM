#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen35_smoke_test_%j.out
#SBATCH --job-name=q35_smoke

# Qwen3.5 Size Ablation - Smoke Test
# Phase 1: Regression test with Qwen3-VL-8B (verify code changes)
# Phase 2: Smoke test all 4 Qwen3.5 sizes

set -e

cd /scratch/khayes/LLM

# Activate conda
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=2,3,4

echo "=============================================="
echo "PHASE 1: Regression test (Qwen3-VL-8B)"
echo "=============================================="
python scripts/train_best_uq.py \
    --output_dir data/ablations/qwen35_model_size/_regression_test \
    --split_info uq_models/best_v2_r32_combined/split_info.json \
    --prompt_variant combined \
    --lora_r 16 \
    --smoke_test

echo ""
echo "=============================================="
echo "PHASE 2: Qwen3.5 smoke test (all 4 sizes)"
echo "=============================================="
python scripts/qwen35_size_ablation.py --smoke_test

echo ""
echo "SMOKE TESTS COMPLETE"
