#!/bin/bash
# Run all use cases with unified model scores
# CPU-only, ~5 min total

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

SCORED_DIR="data/use_cases/scored_test_only_v2"
RESULTS_DIR="data/use_cases/results_test_only_v2"
FIGURES_DIR="figures/use_cases_unified"

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

echo "=========================================="
echo "Running All Use Cases with Unified Model"
echo "Scored data: $SCORED_DIR"
echo "Start: $(date)"
echo "=========================================="

echo ""
echo "=== UC1: Selective Prediction ==="
python scripts/uc1_selective_prediction.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=== UC2: Model Routing ==="
python scripts/uc2_model_routing.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=== UC3: Error Flagging ==="
python scripts/uc3_hallucination_detection.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=== UC4: Difficulty Estimation ==="
python scripts/uc4_difficulty_estimation.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=== UC5: Reward Model ==="
python scripts/uc5_uq_reward_model.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=== UC7: Self-Improvement ==="
python scripts/uc7_self_improvement.py \
    --scored_dir "$SCORED_DIR" \
    --output_dir "$RESULTS_DIR" \
    --fig_dir "$FIGURES_DIR" 2>&1

echo ""
echo "=========================================="
echo "All use cases complete: $(date)"
echo "Results: $RESULTS_DIR/"
echo "Figures: $FIGURES_DIR/"
echo "=========================================="
