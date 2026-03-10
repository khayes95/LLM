#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=reviewer_pipeline
#SBATCH --output=logs/reviewer_pipeline_%j.out

# Full reviewer experiment pipeline (Feb 28, 2026)
# Estimated: 12-18 hours total
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/retrain_best_v2.py --smoke_test
#
# Steps:
#   1. CPU: Filter scored data to test-only, re-run all use cases (~15 min)
#   2. CPU: Baselines on test-only data (~5 min)
#   3. CPU: Per-benchmark breakdown (~1 min)
#   4. CPU: Paired significance tests (~5 min)
#   5. CPU: Literature comparison table (~1 min)
#   6. GPU: Retrain best model with r=32 + longer prompt (~2-3 hrs)
#   7. GPU: Score all targets with new model (~2 hrs)
#   8. GPU: Held-out benchmark evaluation (5-fold, ~10 hrs)
#   9. GPU: Multi-seed training (3 seeds × ~2.5 hrs = ~7.5 hrs)
#  10. CPU: Bootstrap CIs on test-only data (~5 min)

source ~/.bashrc
conda activate uq_eval

cd /scratch/khayes/LLM

# Use GPUs 0, 4, 5, 7 (free as of 2am Feb 28)
export CUDA_VISIBLE_DEVICES=0,4,5,7

echo "============================================"
echo "REVIEWER EXPERIMENT PIPELINE — $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# ============================================================
# PHASE 1: CPU-ONLY (parallel-safe, no GPU needed)
# ============================================================

echo ""
echo ">>> PHASE 1: CPU-only analyses (filter, baselines, breakdown, significance)"
echo "============================================"

# Step 1: Filter scored data to test-only + re-run all use cases
echo "--- Step 1: Filter to test-only + re-run use cases ($(date)) ---"
python scripts/filter_test_only.py \
    --scored_dir data/use_cases/CONTAMINATED_scored_unified \
    --output_scored data/use_cases/scored_test_only \
    --output_dir data/use_cases/results_test_only \
    --fig_dir figures/use_cases_test_only
echo "--- Step 1 done ($(date)) ---"

# Step 2: Compute baselines on test-only data
echo "--- Step 2: Baselines on test-only data ($(date)) ---"
python scripts/compute_baselines.py --scored_dir data/use_cases/scored_test_only
echo "--- Step 2 done ($(date)) ---"

# Step 3: Bootstrap CIs on test-only data
echo "--- Step 3: Bootstrap CIs on test-only ($(date)) ---"
python scripts/bootstrap_ci.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/bootstrap_ci.json \
    --n_bootstrap 2000
echo "--- Step 3 done ($(date)) ---"

# Step 4: Per-benchmark breakdown
echo "--- Step 4: Per-benchmark breakdown ($(date)) ---"
python scripts/per_benchmark_breakdown.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/per_benchmark_breakdown.json
echo "--- Step 4 done ($(date)) ---"

# Step 5: Paired significance tests
echo "--- Step 5: Significance tests ($(date)) ---"
python scripts/paired_significance_tests.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/significance_tests.json \
    --n_permutations 10000
echo "--- Step 5 done ($(date)) ---"

# Step 6: Literature comparison
echo "--- Step 6: Literature comparison ($(date)) ---"
python scripts/literature_comparison.py
echo "--- Step 6 done ($(date)) ---"

echo ""
echo "============================================"
echo "PHASE 1 COMPLETE — $(date)"
echo "============================================"

# ============================================================
# PHASE 2: GPU - Retrain best model (r=32 + combined prompt)
# ============================================================

echo ""
echo ">>> PHASE 2: Retrain best model (r=32 + combined prompt)"
echo "============================================"

python scripts/retrain_best_v2.py \
    --output_dir uq_models/best_unified_v2 \
    --epochs 3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4

echo "--- Phase 2 done ($(date)) ---"

# Score all targets with new model
echo ""
echo ">>> Scoring all targets with best_unified_v2..."
echo "============================================"

# Need to update score_all_unified.py to use v2 checkpoint
python scripts/score_all_unified.py \
    --target all \
    --checkpoint uq_models/best_unified_v2 \
    --output_dir data/use_cases/scored_unified_v2 \
    --prompt_variant combined

echo "--- Scoring done ($(date)) ---"

# Filter v2 scored data to test-only
echo ">>> Filtering v2 scored data to test-only..."
python scripts/filter_test_only.py \
    --split_info uq_models/best_unified_v2/split_info.json \
    --scored_dir data/use_cases/scored_unified_v2 \
    --output_scored data/use_cases/scored_test_only_v2 \
    --output_dir data/use_cases/results_test_only_v2 \
    --fig_dir figures/use_cases_test_only_v2

echo "--- v2 test-only done ($(date)) ---"

# ============================================================
# PHASE 3: GPU - Held-out benchmark evaluation (5-fold)
# ============================================================

echo ""
echo ">>> PHASE 3: Held-out benchmark evaluation (5-fold)"
echo "============================================"

python scripts/held_out_benchmark_eval.py \
    --output_dir data/ablations/held_out_benchmark \
    --n_folds 5 \
    --epochs 2 \
    --lora_r 32

echo "--- Phase 3 done ($(date)) ---"

# ============================================================
# PHASE 4: GPU - Multi-seed training (3 seeds)
# ============================================================

echo ""
echo ">>> PHASE 4: Multi-seed training (3 seeds)"
echo "============================================"

python scripts/multi_seed_training.py \
    --output_dir data/ablations/multi_seed \
    --seeds 42 123 456 \
    --epochs 3 \
    --lora_r 32

echo "--- Phase 4 done ($(date)) ---"

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "============================================"
echo "REVIEWER PIPELINE COMPLETE — $(date)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Test-only (v1): data/use_cases/scored_test_only/"
echo "  Test-only results: data/use_cases/results_test_only/"
echo "  Best model v2: uq_models/best_unified_v2/"
echo "  Scored v2: data/use_cases/scored_unified_v2/"
echo "  Test-only v2: data/use_cases/scored_test_only_v2/"
echo "  Held-out benchmark: data/ablations/held_out_benchmark/"
echo "  Multi-seed: data/ablations/multi_seed/"
echo "  Significance tests: data/use_cases/results_test_only/significance_tests.json"
echo "  Per-benchmark: data/use_cases/results_test_only/per_benchmark_breakdown.json"
echo "  Literature: data/use_cases/results_test_only_v2/literature_comparison.json"
