#!/bin/bash
#SBATCH --job-name=qwen35_miss
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=logs/missing_qwen35_%j.log
#SBATCH --error=logs/missing_qwen35_%j.log

# Complete incomplete Qwen3.5-397B benchmarks via local vLLM
# Needs all 8 GPUs for TP=8
#
# Benchmarks to complete (current -> target 250):
#   hle_multimodal: 0 -> 250 (new)
#   mathvision: 71 -> 250
#   bbeh: 129 -> 250
#   hle: 149 -> 250
#   livebench: 150 -> 250
#   mathverse: 157 -> 250
#   prbench: 169 -> 250
#   omnimath: 121 -> 250
#
# Smoke test: SMOKE=1 sbatch slurm/complete_missing_qwen35.sh

echo "=========================================="
echo "Complete Missing Qwen3.5-397B Benchmarks"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

MAX_SAMPLES=250
SEED=42
PORT=8100

if [[ -n "$SMOKE" ]]; then
    MAX_SAMPLES=5
    echo "*** SMOKE TEST MODE (5 samples) ***"
fi

# Start vLLM server for Qwen3.5-397B
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model /scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 \
    --port $PORT \
    --max-model-len 32768 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    > logs/vllm_qwen35_missing.log 2>&1 &
VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"
echo "Waiting for server to be ready..."

# Wait for server
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died! Check logs/vllm_qwen35_missing.log"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "vLLM server failed to start after 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

MODEL_NAME=$(curl -s http://localhost:$PORT/v1/models | python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: $MODEL_NAME"

run_bench() {
    local bench=$1
    local out_dir=$2
    local extra=$3
    local logfile="logs/missing_qwen35_${bench}.log"

    cmd="python -m uq_eval.cli --bench $bench"
    cmd="$cmd --model_backend chat_http --model_name $MODEL_NAME"
    cmd="$cmd --base_url http://localhost:$PORT/v1"
    cmd="$cmd --timeout_s 900 --max_output_tokens 2048 --disable_thinking"
    cmd="$cmd --out_dir $out_dir --max_examples $MAX_SAMPLES --seed $SEED --resume"

    if [[ -n "$extra" ]]; then
        cmd="$cmd $extra"
    fi

    echo ""
    echo "=== $bench ==="
    echo "  CMD: $cmd"
    $cmd > "$logfile" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "  DONE: $bench"
        tail -3 "$logfile"
    else
        echo "  FAILED: $bench (exit $rc)"
        tail -10 "$logfile"
    fi
}

echo ""
echo "============================================================"
echo "Running benchmarks..."
echo "============================================================"

# Fast text benchmarks first (most important to complete)
run_bench "bbeh" "runs/qwen35_397b_bbeh" ""
run_bench "livebench" "runs/qwen35_397b_livebench" ""
run_bench "omnimath" "runs/qwen35_397b_omnimath" ""
run_bench "prbench" "runs/qwen35_397b_prbench" ""
run_bench "hle" "runs/qwen35_397b_hle" "--hle_answer_type exact_match"

# VLM benchmarks (slower due to image processing)
run_bench "mathvision" "runs/qwen35_397b_mathvision" ""
run_bench "mathverse" "runs/qwen35_397b_mathverse" ""

# hle_multimodal last (slowest - hard questions + images)
run_bench "hle" "runs/qwen35_397b_hle_multimodal" "--hle_with_images"

echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "All Qwen3.5 benchmarks finished: $(date)"
echo "=========================================="
