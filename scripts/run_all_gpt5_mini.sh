#!/bin/bash
# Run GPT-5-mini on all verified benchmarks using existing uq_eval CLI.
#
# Usage:
#   ./scripts/run_all_gpt5_mini.sh [--dry-run] [--max-samples N] [--seed N] [--train-after]
#
# Cost estimate: ~$5-10 for validation run (100 samples each)
# Output: runs/<timestamp>_<bench>_gpt-5-mini/
#
# Random sampling: Samples are randomly selected with --seed for reproducibility.
# Sampled IDs are saved to sampled_ids.json for future exclusion.
# Use --exclude_ids to skip previously evaluated samples in larger runs.

set -e

# Ensure API key is available
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi
export OPENAI_API_KEY

MODEL="gpt-5-mini"
MODEL_BACKEND="openai"
MAX_SAMPLES="${MAX_SAMPLES:-100}"  # Default 100 samples per benchmark for validation
SEED="${SEED:-42}"  # Default seed for reproducibility
DRY_RUN=false
TRAIN_AFTER=false
OUTPUT_BASE="runs"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --train-after) TRAIN_AFTER=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "GPT-5-MINI EVALUATION ON ALL BENCHMARKS"
echo "============================================================"
echo "Model: $MODEL"
echo "Backend: $MODEL_BACKEND"
echo "Max samples: $MAX_SAMPLES"
echo "Seed: $SEED"
echo "Timeout: 300s (set after config)"
echo "Dry run: $DRY_RUN"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Text benchmarks - organized by tier (ONLY useful accuracy ranges for UQ training)
#
# TIER 1: Ideal accuracy (40-60%) - best for balanced training data
TEXT_TIER1="bbeh simpleqa prbench tutorbench healthbench"
# TIER 2: Hard (<40%) - more incorrect samples, still useful
TEXT_TIER2="hle arc_agi chembench"
# TIER 3: Moderate-high accuracy (70-80%) - usable with subsampling
TEXT_TIER3="gpqa livebench omnimath"
# Long context (varies by task)
TEXT_LC="oolong babilong"

# EXCLUDED - Too easy (>85% accuracy = not enough incorrect samples for UQ training):
# math, mmlu_pro, gsm8k, arc, drop, triviaqa, hellaswag, winogrande

ALL_TEXT="$TEXT_TIER1 $TEXT_TIER2 $TEXT_TIER3 $TEXT_LC"

# VLM benchmarks - useful accuracy ranges only (<75%)
# HARD (<40%): More incorrect samples
VLM_HARD="mathvision mathverse hallusionbench"
# MED (40-70%): Good balance
VLM_MED="mathvista mmmu mmstar charxiv realworldqa aokvqa vizwiz mmvet vsr"

# EXCLUDED VLM - Too easy (>75%): ai2d, chartqa, docvqa, nlvr2, pope, ocrbench, gqa, infographicvqa

ALL_VLM="$VLM_HARD $VLM_MED"

# Parallelism settings (OpenAI API, no GPU needed)
# Default 8 jobs - conservative for most API tiers. Increase to 16+ if you have higher limits.
# Override with: PARALLEL_JOBS=16 ./scripts/run_all_gpt5_mini.sh
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"

# Timeout settings - reasoning models (gpt-5-mini, o1, etc.) need longer timeouts
# They can take 30-120+ seconds per request due to internal reasoning
TIMEOUT_S="${TIMEOUT_S:-300}"  # 5 minutes default for reasoning models

echo "Parallel jobs: $PARALLEL_JOBS"

run_benchmark() {
    local bench=$1
    local extra_args=$2

    echo ""
    echo ">>> Running $bench..."

    cmd="python -m uq_eval.cli --bench $bench --model_backend $MODEL_BACKEND --model_name $MODEL --timeout_s $TIMEOUT_S"

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_examples $MAX_SAMPLES"
    fi

    if [[ -n "$SEED" ]]; then
        cmd="$cmd --seed $SEED"
    fi

    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    echo "    $cmd"

    if [[ "$DRY_RUN" == "false" ]]; then
        $cmd || echo "    WARNING: $bench failed"
    fi
}

run_benchmark_bg() {
    # Run benchmark in background, return immediately
    local bench=$1
    local extra_args=$2
    local logfile="logs/bench_${bench}.log"

    cmd="python -m uq_eval.cli --bench $bench --model_backend $MODEL_BACKEND --model_name $MODEL --timeout_s $TIMEOUT_S"

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max_examples $MAX_SAMPLES"
    fi

    if [[ -n "$SEED" ]]; then
        cmd="$cmd --seed $SEED"
    fi

    if [[ -n "$extra_args" ]]; then
        cmd="$cmd $extra_args"
    fi

    echo "  [BG] $bench -> $logfile"

    if [[ "$DRY_RUN" == "false" ]]; then
        $cmd > "$logfile" 2>&1 &
    fi
}

wait_for_jobs() {
    # Wait until fewer than N jobs are running
    local max_jobs=$1
    while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
        sleep 5
    done
}

echo ""
echo "============================================================"
echo "TEXT BENCHMARKS (parallel: $PARALLEL_JOBS jobs)"
echo "============================================================"

for bench in $ALL_TEXT; do
    wait_for_jobs $PARALLEL_JOBS
    case $bench in
        gpqa)
            run_benchmark_bg "$bench" "--subset gpqa_diamond"
            ;;
        hle)
            run_benchmark_bg "$bench" "--hle_answer_type exact_match"
            ;;
        *)
            run_benchmark_bg "$bench"
            ;;
    esac
done

echo ""
echo "============================================================"
echo "VLM BENCHMARKS (parallel: $PARALLEL_JOBS jobs)"
echo "============================================================"

for bench in $ALL_VLM; do
    wait_for_jobs $PARALLEL_JOBS
    run_benchmark_bg "$bench"
done

echo ""
echo "Waiting for all benchmarks to complete..."
wait
echo "All benchmarks completed!"

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo "Runs saved to: $OUTPUT_BASE/"
echo "Logs saved to: logs/bench_*.log"

# Start training if requested
if [[ "$TRAIN_AFTER" == "true" && "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "STARTING QWEN3-VL UQ TRAINING"
    echo "============================================================"
    echo ""
    CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/train_qwen3_vlm_uq.py --data_dir runs/ --wait_for_gpus
else
    echo ""
    echo "Next steps:"
    echo "1. Grade open-ended benchmarks:"
    echo "   python -m uq_eval.grader --predictions runs/<run>/predictions.jsonl"
    echo ""
    echo "2. Train Qwen3-VL UQ classifier:"
    echo "   CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/train_qwen3_vlm_uq.py --data_dir runs/"
    echo ""
fi
