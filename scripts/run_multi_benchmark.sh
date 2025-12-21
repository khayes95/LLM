#!/usr/bin/env bash
#SBATCH -J uq_multi_bench
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_multi_%j.out
#SBATCH -e logs/uq_multi_%j.err

# Multi-benchmark evaluation run
# Usage: sbatch scripts/run_multi_benchmark.sh [model_name] [max_examples]

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
echo "MULTI-BENCHMARK EVALUATION"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "Max examples per benchmark: $MAX_EXAMPLES"
echo "=========================================="

# Start vLLM server
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

# Benchmarks to run (text-based, known working)
BENCHMARKS=(
    "gsm8k"
    "mmlu"
    "arc"
    "hellaswag"
    "winogrande"
    "boolq"
    "triviaqa"
    "drop"
    "gpqa"
    "simpleqa"
    "bbeh"
    "hle"
    "aime"
    "mgsm"
)

# Run each benchmark
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
        --max_examples "$MAX_EXAMPLES" \
        || echo "WARNING: $BENCH failed, continuing..."

    echo "Completed: $BENCH"
done

# Cleanup
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="

# Show summary of results
echo ""
echo "Results summary:"
for BENCH in "${BENCHMARKS[@]}"; do
    LATEST=$(ls -td runs/*_${BENCH}_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/metrics.json" ]; then
        echo "$BENCH: $(cat $LATEST/metrics.json)"
    fi
done
