#!/usr/bin/env bash
#SBATCH -J uq_parallel
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o logs/uq_parallel_%j.out
#SBATCH -e logs/uq_parallel_%j.err

# Parallel evaluation across 4 GPUs
# Each GPU runs a subset of benchmarks with its own vLLM server

set -e

MODEL_NAME="${1:-meta-llama/Llama-3.1-8B-Instruct}"

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

mkdir -p logs

echo "=========================================="
echo "PARALLEL HARD BENCHMARKS (4 GPUs)"
echo "Started at: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

# Function to run benchmarks on a specific GPU
run_gpu_worker() {
    local GPU_ID=$1
    local PORT=$2
    shift 2
    local BENCHMARKS=("$@")

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "[GPU $GPU_ID] Starting vLLM server on port $PORT..."

    vllm serve "$MODEL_NAME" \
        --port $PORT \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.80 \
        --disable-log-requests \
        --enforce-eager \
        &

    local VLLM_PID=$!

    # Wait for server
    local MAX_WAIT=600
    local WAITED=0
    while true; do
        MODELS_RESP=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null || echo "")
        if echo "$MODELS_RESP" | grep -q '"id"'; then
            break
        fi
        sleep 10
        WAITED=$((WAITED + 10))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[GPU $GPU_ID] ERROR: vLLM server failed to start"
            kill $VLLM_PID 2>/dev/null || true
            return 1
        fi
    done

    echo "[GPU $GPU_ID] Server ready, running benchmarks: ${BENCHMARKS[*]}"

    for BENCH in "${BENCHMARKS[@]}"; do
        echo "[GPU $GPU_ID] Running: $BENCH at $(date)"
        python -m uq_eval.cli \
            --bench "$BENCH" \
            --model_backend chat_http \
            --model_name "$MODEL_NAME" \
            --base_url "http://localhost:$PORT/v1" \
            --temperature 0.0 \
            --max_output_tokens 2048 \
            || echo "[GPU $GPU_ID] WARNING: $BENCH failed"
        echo "[GPU $GPU_ID] Completed: $BENCH at $(date)"
    done

    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "[GPU $GPU_ID] Done with all benchmarks"
}

# Split 14 benchmarks across 4 GPUs (~3-4 each)
# Trying to balance by expected runtime

# GPU 0: Tier 1 benchmarks (medium size)
GPU0_BENCHMARKS=("bbeh" "simpleqa" "tutorbench" "multichallenge")

# GPU 1: Tier 1 + Tier 2 (mixed)
GPU1_BENCHMARKS=("healthbench" "bigcodebench" "multinrc" "hle")

# GPU 2: Tier 2 + Tier 3 (harder)
GPU2_BENCHMARKS=("ether0" "gpqa" "omnimath")

# GPU 3: Coding + Long context
GPU3_BENCHMARKS=("livecodebench" "swebench" "oolong")

# Launch all 4 workers in parallel
run_gpu_worker 0 8000 "${GPU0_BENCHMARKS[@]}" &
PID0=$!

run_gpu_worker 1 8001 "${GPU1_BENCHMARKS[@]}" &
PID1=$!

run_gpu_worker 2 8002 "${GPU2_BENCHMARKS[@]}" &
PID2=$!

run_gpu_worker 3 8003 "${GPU3_BENCHMARKS[@]}" &
PID3=$!

# Wait for all workers to complete
echo "Waiting for all GPU workers to complete..."
wait $PID0
echo "GPU 0 finished"
wait $PID1
echo "GPU 1 finished"
wait $PID2
echo "GPU 2 finished"
wait $PID3
echo "GPU 3 finished"

echo "=========================================="
echo "PARALLEL EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="

# Show summary
echo ""
echo "Results summary:"
ALL_BENCHMARKS=("bbeh" "simpleqa" "tutorbench" "multichallenge" "healthbench" "bigcodebench" "multinrc" "hle" "ether0" "gpqa" "omnimath" "livecodebench" "swebench" "oolong")
for BENCH in "${ALL_BENCHMARKS[@]}"; do
    LATEST=$(ls -td runs/*_${BENCH}_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/metrics.json" ]; then
        echo "$BENCH: $(cat $LATEST/metrics.json)"
    fi
done
