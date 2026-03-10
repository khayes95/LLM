#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=retrain_v2
#SBATCH --output=logs/retrain_v2_%j.out

# Retrain best model with r=32 + combined prompt, then score + filter
# Estimated: 4-5 hours total
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/retrain_best_v2.py --smoke_test

source ~/.bashrc
conda activate uq_eval
cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,4,5,7

echo "============================================"
echo "RETRAIN V2 + SCORE — $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Step 1: Retrain with r=32 + combined prompt
echo ">>> Step 1: Retraining with r=32 + combined prompt ($(date))"
python scripts/retrain_best_v2.py \
    --output_dir uq_models/best_unified_v2 \
    --epochs 3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4

echo "--- Retrain done ($(date)) ---"

# Step 2: Score all targets with v2
echo ""
echo ">>> Step 2: Score all targets with v2 ($(date))"
python scripts/score_all_unified.py \
    --target all \
    --checkpoint uq_models/best_unified_v2 \
    --output_dir data/use_cases/scored_unified_v2 \
    --prompt_variant combined

echo "--- Scoring done ($(date)) ---"

# Step 3: Filter v2 to test-only + re-run use cases
echo ""
echo ">>> Step 3: Filter v2 to test-only + use cases ($(date))"
python scripts/filter_test_only.py \
    --split_info uq_models/best_unified_v2/split_info.json \
    --scored_dir data/use_cases/scored_unified_v2 \
    --output_scored data/use_cases/scored_test_only_v2 \
    --output_dir data/use_cases/results_test_only_v2 \
    --fig_dir figures/use_cases_test_only_v2

echo "--- Filter + use cases done ($(date)) ---"

# Step 4: Baselines + bootstrap on v2 test-only
echo ""
echo ">>> Step 4: Baselines + bootstrap on v2 ($(date))"
python scripts/compute_baselines.py --scored_dir data/use_cases/scored_test_only_v2
python scripts/bootstrap_ci.py \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --output data/use_cases/results_test_only_v2/bootstrap_ci.json \
    --n_bootstrap 2000

echo ""
echo "============================================"
echo "RETRAIN V2 PIPELINE COMPLETE — $(date)"
echo "============================================"
