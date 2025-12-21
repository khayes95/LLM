#!/usr/bin/env bash
#SBATCH -J uq_fixed
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_fixed_%j.out
#SBATCH -e logs/uq_fixed_%j.err

# Run the 3 fixed benchmarks: ARC, Winogrande, TriviaQA

set -e

MODEL_NAME="${1:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_EXAMPLES="${2:-50}"
VLLM_PORT=8000

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs

echo "=========================================="
echo "FIXED BENCHMARK RUN"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

# Start vLLM server
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

# Wait for server
MAX_WAIT=600
WAITED=0
while true; do
    MODELS_RESP=$(curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null || echo "")
    if echo "$MODELS_RESP" | grep -q '"id"'; then
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waited ${WAITED}s..."
done

echo "vLLM server is ready!"

# Run the 3 fixed benchmarks
for BENCH in arc winogrande triviaqa; do
    echo ""
    echo "=========================================="
    echo "Running benchmark: $BENCH"
    echo "=========================================="

    python -m uq_eval.cli \
        --bench "$BENCH" \
        --model_backend chat_http \
        --model_name "$MODEL_NAME" \
        --base_url "http://localhost:$VLLM_PORT/v1" \
        --temperature 0.0 \
        --max_output_tokens 2048 \
        --max_examples "$MAX_EXAMPLES" \
        || echo "WARNING: $BENCH failed"

    echo "Completed: $BENCH"
done

# Cleanup
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "DONE at: $(date)"
echo "=========================================="
