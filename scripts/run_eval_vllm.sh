#!/usr/bin/env bash
#SBATCH -J uq_eval
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=8
# GPU allocation handled by partition/node selection
#SBATCH -o logs/uq_eval_%j.out
#SBATCH -e logs/uq_eval_%j.err

# UQ Eval Harness with vLLM Server
# Usage: sbatch scripts/run_eval_vllm.sh [model_name] [benchmark]
# Example: sbatch scripts/run_eval_vllm.sh Qwen/Qwen2.5-72B-Instruct sanity_mcq

set -e

# Configuration
MODEL_NAME="${1:-Qwen/Qwen2.5-72B-Instruct}"
BENCHMARK="${2:-sanity_mcq}"
MAX_EXAMPLES="${3:-}"  # Leave empty for all examples
VLLM_PORT=8000
TP_SIZE=4  # tensor parallel size (adjust based on model size)

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,1,2,3

mkdir -p logs

echo "=========================================="
echo "UQ EVAL HARNESS"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "=========================================="

# ============================================
# STEP 1: START VLLM SERVER
# ============================================
echo ""
echo "Starting vLLM server..."

# Start vLLM in background
# Use distributed-executor-backend=ray for stable multi-GPU tensor parallelism
vllm serve "$MODEL_NAME" \
    --tensor-parallel-size $TP_SIZE \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --disable-log-requests \
    --distributed-executor-backend ray \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to initialize..."
MAX_WAIT=300  # 5 minutes max
WAITED=0
while ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waited ${WAITED}s..."
done

echo "vLLM server is ready!"

# ============================================
# STEP 2: RUN EVALUATION
# ============================================
echo ""
echo "=========================================="
echo "Running evaluation: $BENCHMARK"
echo "=========================================="

# Build CLI args
EVAL_ARGS=(
    --bench "$BENCHMARK"
    --model_backend chat_http
    --model_name "$MODEL_NAME"
    --base_url "http://localhost:$VLLM_PORT/v1"
    --temperature 0.0
    --max_output_tokens 256
)

if [ -n "$MAX_EXAMPLES" ]; then
    EVAL_ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

python -m uq_eval.cli "${EVAL_ARGS[@]}"

# ============================================
# CLEANUP
# ============================================
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
