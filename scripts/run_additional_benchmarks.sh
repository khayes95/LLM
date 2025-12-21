#!/usr/bin/env bash
#SBATCH -J uq_additional
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_additional_%j.out
#SBATCH -e logs/uq_additional_%j.err

# Run additional benchmarks not in the main eval
# Using GPU 2 to run in parallel with main job

set -e

MODEL_NAME="${1:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT=8002  # Different port to avoid conflict

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=2

mkdir -p logs

echo "=========================================="
echo "ADDITIONAL BENCHMARKS"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "GPU: 2, Port: $VLLM_PORT"
echo "=========================================="

# Start vLLM server on different port
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

# Additional benchmarks (not in main run)
BENCHMARKS=(
    "humaneval"
    "mbpp"
    "longbench"
    "bigcodebench"
)

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running benchmark: $BENCH (FULL DATASET)"
    echo "Time: $(date)"
    echo "=========================================="

    python -m uq_eval.cli \
        --bench "$BENCH" \
        --model_backend chat_http \
        --model_name "$MODEL_NAME" \
        --base_url "http://localhost:$VLLM_PORT/v1" \
        --temperature 0.0 \
        --max_output_tokens 2048 \
        || echo "WARNING: $BENCH failed, continuing..."

    echo "Completed: $BENCH at $(date)"
done

# Cleanup
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "ADDITIONAL BENCHMARKS COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
