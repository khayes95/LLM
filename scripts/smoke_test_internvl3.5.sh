#!/bin/bash
# Smoke test InternVL3.5-38B on VLM benchmarks
# Runs 2 samples per benchmark to catch errors quickly
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/smoke_test_internvl3.5.sh
#
# Requirements:
#   - 2x A100-80GB for InternVL3.5-38B (~80GB VRAM)

set -e

cd /scratch/khayes/LLM

# Use 2 GPUs by default if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MODEL_NAME="OpenGVLab/InternVL3_5-38B"

echo "=========================================="
echo "InternVL3.5-38B Smoke Test"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
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
        --model_name "$MODEL_NAME" \
        --max_examples 2 \
        --seed 99 \
        --timeout_s 300 \
        --max_output_tokens 1024 \
        --no-resume \
        --out_dir "runs/smoke_test_internvl3.5_${bench}" 2>&1; then

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
echo "Model: $MODEL_NAME"
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
