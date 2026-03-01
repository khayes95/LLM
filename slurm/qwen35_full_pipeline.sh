#!/bin/bash
#SBATCH --job-name=q35_pipe
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen35_pipeline_%j.log
#SBATCH --error=logs/qwen35_pipeline_%j.log

# Combined pipeline: serve Qwen3.5-397B-A17B-FP8 + run all benchmarks
# Sunday night run — 12 hour wall time
# Smoke test: sbatch --export=SMOKE_TEST=1 slurm/qwen35_full_pipeline.sh

echo "=========================================="
echo "Qwen3.5-397B Full Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Smoke test: ${SMOKE_TEST:-0}"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

source ~/.bashrc
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

echo ""
echo "Step 1: Starting vLLM server on port 8100..."
echo ""

# Start server in background
python -m vllm.entrypoints.openai.api_server \
    --model /scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8 \
    --port 8100 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --dtype auto &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready (check every 10s, up to 10 min)
echo "Waiting for server to be ready..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${WAITED}s"
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  ...waiting (${WAITED}s / ${MAX_WAIT}s)"
done

if ! curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
    echo "ERROR: Server failed to start after ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Quick smoke test
echo ""
echo "Step 2: Quick smoke test..."
python scripts/smoke_test_qwen3_5.py --base_url http://localhost:8100
if [ $? -ne 0 ]; then
    echo "ERROR: Smoke test failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "Smoke test passed!"

# Determine flags
SMOKE_FLAG=""
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke_test"
fi

echo ""
echo "=========================================="
echo "Step 3: Running text benchmarks..."
echo "Start: $(date)"
echo "=========================================="

python scripts/run_all_qwen35_397b.py \
    --base_url http://localhost:8100/v1 \
    --parallel 4 \
    $SMOKE_FLAG

TEXT_RC=$?
echo "Text benchmarks exit code: $TEXT_RC"
echo "Text benchmarks done: $(date)"

echo ""
echo "=========================================="
echo "Step 4: Running VLM benchmarks (text-only mode)..."
echo "Start: $(date)"
echo "=========================================="

python scripts/run_all_qwen35_397b_vlm.py \
    --base_url http://localhost:8100/v1 \
    --parallel 2 \
    $SMOKE_FLAG

VLM_RC=$?
echo "VLM benchmarks exit code: $VLM_RC"
echo "VLM benchmarks done: $(date)"

# Print final summary
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Text exit code: $TEXT_RC"
echo "VLM exit code: $VLM_RC"
echo ""
echo "Predictions per benchmark:"
for d in runs/qwen35_397b_*/; do
    bench=$(basename "$d" | sed 's/qwen35_397b_//')
    count=$(wc -l < "$d/predictions.jsonl" 2>/dev/null || echo 0)
    echo "  $bench: $count"
done

# Shutdown
echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Pipeline complete: $(date)"
