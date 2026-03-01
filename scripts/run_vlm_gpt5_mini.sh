#!/bin/bash
# Run GPT-5-mini on VLM benchmarks only.
#
# Usage:
#   ./scripts/run_vlm_gpt5_mini.sh [--dry-run] [--max-samples N] [--seed N]
#
# Output: runs/<timestamp>_<bench>_gpt-5-mini/

set -e

# Ensure API key is available
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi
export OPENAI_API_KEY

MODEL="gpt-5-mini"
MODEL_BACKEND="openai"
MAX_SAMPLES="${MAX_SAMPLES:-100}"  # Default 100 samples per benchmark
SEED="${SEED:-42}"  # Default seed for reproducibility
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "GPT-5-MINI VLM BENCHMARK EVALUATION"
echo "============================================================"
echo "Model: $MODEL"
echo "Backend: $MODEL_BACKEND"
echo "Max samples: $MAX_SAMPLES"
echo "Seed: $SEED"
echo "Timeout: 300s"
echo "Dry run: $DRY_RUN"
echo ""

# Ensure logs directory exists
mkdir -p logs

# VLM benchmarks - all registered in uq_eval.registry
# HARD (<40%): More incorrect samples for UQ training
VLM_HARD="mathvision mathverse hallusionbench"
# MED (40-70%): Good balance
VLM_MED="mathvista mmmu mmstar charxiv realworldqa aokvqa vizwiz mmvet vsr erqa"

ALL_VLM="$VLM_HARD $VLM_MED"

# Parallelism settings - API calls only, no GPU needed
# Run all benchmarks in parallel (13 total)
PARALLEL_JOBS="${PARALLEL_JOBS:-20}"

# Timeout for reasoning models
TIMEOUT_S="${TIMEOUT_S:-300}"

echo "VLM benchmarks: $ALL_VLM"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

run_benchmark_bg() {
    local bench=$1
    local extra_args=$2
    local logfile="logs/vlm_${bench}.log"

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
    local max_jobs=$1
    while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
        sleep 5
    done
}

echo "============================================================"
echo "STARTING VLM BENCHMARKS"
echo "============================================================"

for bench in $ALL_VLM; do
    wait_for_jobs $PARALLEL_JOBS
    run_benchmark_bg "$bench"
done

echo ""
echo "Waiting for all benchmarks to complete..."
wait
echo "All VLM benchmarks completed!"

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"

# Print results
echo ""
echo "Results:"
for bench in $ALL_VLM; do
    latest=$(ls -td runs/*_${bench}_gpt-5-mini 2>/dev/null | head -1)
    if [[ -n "$latest" && -f "$latest/metrics.json" ]]; then
        acc=$(python -c "import json; m=json.load(open('$latest/metrics.json')); print(f'{m.get(\"accuracy\", -1)*100:.1f}%')" 2>/dev/null || echo "N/A")
        echo "  $bench: $acc"
    else
        echo "  $bench: No results"
    fi
done
