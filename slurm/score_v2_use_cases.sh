#!/bin/bash
#SBATCH --job-name=score_v2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/score_v2_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/score_v2_%j.log

# Score all predictions with v2 checkpoint (r=32 + combined prompt, AUROC 0.890)
# Then filter to test-only, re-run all use cases, and compute bootstrap CIs
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py \
#       --checkpoint uq_models/best_v2_r32_combined --target gpt5mini \
#       --prompt_variant combined --max_per_benchmark 5

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

# Use whichever GPU is free
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT="uq_models/best_v2_r32_combined"
SCORED_DIR="data/use_cases/scored_test_only_v2"
TEST_SCORED="data/use_cases/scored_test_only_v2"
RESULTS_DIR="data/use_cases/results_test_only_v2"
FIGURES_DIR="figures/use_cases_v2"

echo "============================================"
echo "SCORE V2 + USE CASES PIPELINE"
echo "Job: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Start: $(date)"
echo "============================================"

# ===========================================================================
# STEP 1: Score all targets with v2 checkpoint
# ===========================================================================
echo ""
echo ">>> Step 1: Scoring all targets with v2 ($(date))"

python scripts/score_all_unified.py \
    --checkpoint "$CHECKPOINT" \
    --target all \
    --output_dir "$SCORED_DIR" \
    --prompt_variant combined \
    2>&1

echo "--- Scoring done ($(date)) ---"

# ===========================================================================
# STEP 2: Filter to test-only + re-run all use cases
# ===========================================================================
echo ""
echo ">>> Step 2: Filter to test-only + use cases ($(date))"

python scripts/filter_test_only.py \
    --split_info "$CHECKPOINT/split_info.json" \
    --scored_dir "$SCORED_DIR" \
    --output_scored "$TEST_SCORED" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" \
    2>&1

echo "--- Filter + use cases done ($(date)) ---"

# ===========================================================================
# STEP 3: Bootstrap CIs on v2 test-only
# ===========================================================================
echo ""
echo ">>> Step 3: Bootstrap CIs ($(date))"

python scripts/bootstrap_ci.py \
    --scored_dir "$TEST_SCORED" \
    --output "$RESULTS_DIR/bootstrap_ci_v2.json" \
    --n_bootstrap 2000 \
    2>&1

echo "--- Bootstrap done ($(date)) ---"

# ===========================================================================
# STEP 4: Per-benchmark analysis on v2
# ===========================================================================
echo ""
echo ">>> Step 4: Per-benchmark analysis ($(date))"

python scripts/per_benchmark_analysis.py \
    --scored_dir "$TEST_SCORED" \
    --split_info "$CHECKPOINT/split_info.json" \
    --output "$RESULTS_DIR/per_benchmark_v2.json" \
    2>&1 || echo "WARNING: per_benchmark_analysis failed (may need --scored_dir fix)"

echo "--- Per-benchmark done ($(date)) ---"

# ===========================================================================
# STEP 5: Held-out benchmark eval on v2
# ===========================================================================
echo ""
echo ">>> Step 5: Held-out benchmark eval ($(date))"

python scripts/held_out_benchmark_eval.py \
    --scored_dir "$SCORED_DIR" \
    --split_info "$CHECKPOINT/split_info.json" \
    --output_dir "$RESULTS_DIR" \
    2>&1 || echo "WARNING: held_out_benchmark_eval failed"

echo ""
echo "============================================"
echo "SCORE V2 PIPELINE COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Scored:    $SCORED_DIR"
echo "  Test-only: $TEST_SCORED"
echo "  Results:   $RESULTS_DIR"
echo "  Figures:   $FIGURES_DIR"
