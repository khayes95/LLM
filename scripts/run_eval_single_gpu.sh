#!/usr/bin/env bash
#SBATCH -J uq_eval_1gpu
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_eval_%j.out
#SBATCH -e logs/uq_eval_%j.err

# UQ Eval Harness with vLLM Server (Single GPU)
# For smaller models that fit on one GPU (7B-9B)
# Usage: sbatch scripts/run_eval_single_gpu.sh [model_name] [benchmark]

set -e

# Configuration - use a model that fits on single A100 80GB
MODEL_NAME="${1:-google/gemma-2-9b-it}"
BENCHMARK="${2:-sanity_mcq}"
MAX_EXAMPLES="${3:-}"
VLLM_PORT=8000

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Use only one GPU (pick an available one)
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs

echo "=========================================="
echo "UQ EVAL HARNESS (Single GPU)"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "=========================================="

# Start vLLM in background (single GPU, no tensor parallelism)
echo "Starting vLLM server..."
vllm serve "$MODEL_NAME" \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --disable-log-requests \
    --enforce-eager \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready - check /v1/models to ensure model is actually loaded
echo "Waiting for vLLM server to initialize..."
MAX_WAIT=600
WAITED=0
while true; do
    # Check if model is loaded by querying /v1/models
    MODELS_RESP=$(curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null || echo "")
    if echo "$MODELS_RESP" | grep -q '"id"'; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waited ${WAITED}s..."
done

echo "vLLM server is ready!"
echo "Model loaded: $(curl -s http://localhost:$VLLM_PORT/v1/models | head -c 200)"

# Run evaluation
echo ""
echo "=========================================="
echo "Running evaluation: $BENCHMARK"
echo "=========================================="

EVAL_ARGS=(
    --bench "$BENCHMARK"
    --model_backend chat_http
    --model_name "$MODEL_NAME"
    --base_url "http://localhost:$VLLM_PORT/v1"
    --temperature 0.0
    --max_output_tokens 2048
)

if [ -n "$MAX_EXAMPLES" ]; then
    EVAL_ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

python -m uq_eval.cli "${EVAL_ARGS[@]}"

# Cleanup
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
