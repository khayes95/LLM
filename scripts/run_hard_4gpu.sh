#!/usr/bin/env bash
#SBATCH -J uq_4gpu
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o logs/uq_4gpu_%j.out
#SBATCH -e logs/uq_4gpu_%j.err

# 4-GPU tensor parallel evaluation - faster inference per example
# Usage: sbatch scripts/run_hard_4gpu.sh [model_name]

set -e

MODEL_NAME="${1:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_PORT=8000

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs

echo "=========================================="
echo "4-GPU TENSOR PARALLEL EVALUATION"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "GPUs: 0,1,2,3 (tensor parallel)"
echo "=========================================="

# Start vLLM with tensor parallelism across 4 GPUs
echo "Starting vLLM server with tensor_parallel_size=4..."
vllm serve "$MODEL_NAME" \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --tensor-parallel-size 4 \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to initialize..."
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
        echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waited ${WAITED}s..."
done

echo "vLLM server is ready!"

# Hard benchmarks targeting ~40-60% accuracy
BENCHMARKS=(
    "bbeh"
    "simpleqa"
    "tutorbench"
    "multichallenge"
    "healthbench"
    "bigcodebench"
    "multinrc"
    "hle"
    "ether0"
    "gpqa"
    "omnimath"
    "livecodebench"
    "swebench"
    "oolong"
)

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running benchmark: $BENCH"
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
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "4-GPU EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="

# Show summary
echo ""
echo "Results summary:"
for BENCH in "${BENCHMARKS[@]}"; do
    LATEST=$(ls -td runs/*_${BENCH}_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/metrics.json" ]; then
        echo "$BENCH: $(cat $LATEST/metrics.json)"
    fi
done
