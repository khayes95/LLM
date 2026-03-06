#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=bsbench
#SBATCH --output=logs/bullshitbench_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=4,5 bash slurm/bullshitbench_eval.sh --smoke
# Full run submitted via sbatch

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
PORT=8199
GPUS="4,5"
SMOKE=0

if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
fi

export CUDA_VISIBLE_DEVICES=$GPUS

echo "=== Starting vLLM server on GPUs $GPUS, port $PORT ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --port $PORT \
    --max-model-len 4096 \
    --disable-log-requests \
    --gpu-memory-utilization 0.85 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "vLLM failed to start after 120s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run the benchmark
MAX_EXAMPLES_FLAG=""
if [[ $SMOKE -eq 1 ]]; then
    MAX_EXAMPLES_FLAG="--max_examples 5"
fi

echo "=== Running BullshitBench eval ==="
python -m uq_eval.cli \
    --bench bullshitbench \
    --model_backend chat_http \
    --model_name "$MODEL" \
    --base_url "http://localhost:$PORT/v1" \
    --max_output_tokens 512 \
    --temperature 0.0 \
    --disable_thinking \
    $MAX_EXAMPLES_FLAG

echo "=== Done ==="

# Cleanup
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
