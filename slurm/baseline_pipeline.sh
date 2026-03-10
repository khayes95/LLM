#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=baselines
#SBATCH --output=logs/baseline_pipeline_%j.out

# Full pipeline: re-score → CPU baselines → zero-shot → bootstrap CIs → UC5 fix
# Estimated: 2-3 hours total
#
# Smoke test:
# CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py --target gpt5mini --smoke_test

source ~/.bashrc
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,2

echo "============================================"
echo "BASELINE PIPELINE — $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Step 1: Re-score all targets with unified model (~90 min)
echo ""
echo ">>> Step 1: Re-scoring with unified model"
echo "============================================"
for target in gpt5mini gpt52 qwen35; do
    echo "--- Scoring $target at $(date) ---"
    python scripts/score_all_unified.py --target $target
    echo "--- Done $target at $(date) ---"
done

echo ""
echo ">>> Step 1 complete — verifying file sizes"
wc -l data/use_cases/scored_test_only_v2/*.jsonl

# Step 2: Compute CPU baselines (Platt, length, combined) — ~1 min
echo ""
echo ">>> Step 2: Computing CPU baselines"
echo "============================================"
python scripts/compute_baselines.py --scored_dir data/use_cases/scored_test_only_v2

# Step 3: Zero-shot base model baseline (~60 min)
echo ""
echo ">>> Step 3: Zero-shot base model baseline"
echo "============================================"
python scripts/zero_shot_baseline.py \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --batch_size 1

# Step 4: Bootstrap confidence intervals — ~5 min
echo ""
echo ">>> Step 4: Bootstrap confidence intervals"
echo "============================================"
python scripts/bootstrap_ci.py \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --n_bootstrap 2000

# Step 5: Re-run UC5 with unified scores
echo ""
echo ">>> Step 5: UC5 with unified calibrator"
echo "============================================"
python scripts/uc5_uq_reward_model.py \
    --scored_dir data/use_cases/scored_test_only_v2 \
    --output_dir data/use_cases/results_test_only_v2 \
    --fig_dir figures/use_cases_unified

echo ""
echo "============================================"
echo "PIPELINE COMPLETE — $(date)"
echo "============================================"
echo "Outputs:"
echo "  Scored data: data/use_cases/scored_test_only_v2/"
echo "  Bootstrap CIs: data/use_cases/results_test_only_v2/bootstrap_ci.json"
echo "  UC5 results: data/use_cases/results_test_only_v2/"
