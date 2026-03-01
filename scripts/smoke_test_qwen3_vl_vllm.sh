#!/bin/bash
# Smoke test Qwen3-VL with vLLM on VLM benchmarks
# Runs 2 samples per benchmark to catch errors quickly
# Uses finegrain_vlm conda environment with vLLM 0.13+
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/smoke_test_qwen3_vl_vllm.sh
#
# Requirements:
#   - finegrain_vlm conda environment with vLLM >= 0.13.0
#   - 1x A100-80GB for Qwen3-VL-30B-A3B-Instruct (~60GB VRAM)
#   - For 235B MoE: 4x A100-80GB (~120GB VRAM)
#
# Expected speedup: 2-4x over standard HuggingFace transformers

set -e

cd /scratch/khayes/LLM

# Use 1 GPU by default if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Default to smaller model if not specified
MODEL_NAME="${QWEN_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"

echo "=========================================="
echo "Qwen3-VL + vLLM Smoke Test"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Backend: qwen3_vl_vllm (vLLM accelerated)"
echo "Start time: $(date)"
echo ""

# Check GPUs
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Activate finegrain_vlm environment
# Note: This script should be run from a shell where conda is initialized
# or the environment should be activated before running
PYTHON_BIN="/scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: finegrain_vlm environment not found at $PYTHON_BIN"
    echo "Please create it with: conda create -n finegrain_vlm python=3.10"
    echo "Then install vLLM: pip install vllm>=0.13.0"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
echo "vLLM version: $($PYTHON_BIN -c 'import vllm; print(vllm.__version__)')"
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
    if $PYTHON_BIN -m uq_eval.cli \
        --bench "$bench" \
        --model_backend qwen3_vl_vllm \
        --model_name "$MODEL_NAME" \
        --max_examples 2 \
        --seed 99 \
        --timeout_s 300 \
        --max_output_tokens 1024 \
        --no-resume \
        --out_dir "runs/smoke_test_qwen3vl_vllm_${bench}" 2>&1; then

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
echo "Backend: qwen3_vl_vllm (vLLM)"
echo "PASSED:$PASSED"
echo "FAILED:$FAILED"
echo ""
echo "End time: $(date)"

if [[ -n "$FAILED" ]]; then
    echo "Some benchmarks failed! Check logs above."
    exit 1
else
    echo "All benchmarks passed! Ready for full run with vLLM."
    exit 0
fi
