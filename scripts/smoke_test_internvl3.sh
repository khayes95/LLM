#!/bin/bash
# Smoke test InternVL3-78B on VLM benchmarks
# Runs 2 samples per benchmark to catch errors quickly
#
# Usage:
#   ./scripts/smoke_test_internvl3.sh

set -e

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=========================================="
echo "InternVL3-78B Smoke Test"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Check GPUs
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# VLM benchmarks to test
VLM_BENCHMARKS="mmmu mathvista charxiv realworldqa vizwiz erqa mathverse mathvision mmstar"

echo "Testing benchmarks: $VLM_BENCHMARKS"
echo ""

# Track results
PASSED=""
FAILED=""

for bench in $VLM_BENCHMARKS; do
    echo "=========================================="
    echo "Testing: $bench (2 samples)"
    echo "=========================================="

    # Run with just 2 samples, no resume (fresh each time)
    if python -m uq_eval.cli \
        --bench "$bench" \
        --model_backend internvl \
        --model_name "OpenGVLab/InternVL3-78B-Instruct" \
        --max_examples 2 \
        --seed 99 \
        --timeout_s 300 \
        --max_output_tokens 1024 \
        --no-resume \
        --out_dir "runs/smoke_test_internvl3_${bench}" 2>&1; then

        echo "PASS: $bench"
        PASSED="$PASSED $bench"
    else
        echo "FAIL: $bench"
        FAILED="$FAILED $bench"
    fi
    echo ""
done

echo "=========================================="
echo "Smoke Test Results"
echo "=========================================="
echo "PASSED:$PASSED"
echo "FAILED:$FAILED"
echo ""
echo "End time: $(date)"

if [[ -n "$FAILED" ]]; then
    echo "Some benchmarks failed! Check logs above."
    exit 1
else
    echo "All benchmarks passed! Ready for full run."
    exit 0
fi
