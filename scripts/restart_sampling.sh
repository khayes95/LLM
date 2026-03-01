#!/bin/bash
# Qwen3.5-397B sampling with thinking enabled + auto-restart on vLLM crash
# Skipping: bbeh, hle, healthbench, tutorbench, prbench, mmvet (too slow / LLM-graded)
# vLLM CUDA graphs can crash after ~1-2h; this script detects and restarts automatically.

CONDA_ENV="/scratch/khayes/.conda/envs/uq_eval/bin/python"
BASE_URL="http://localhost:8100/v1"
LOGDIR="logs"
mkdir -p "$LOGDIR"

# Text benchmarks (skipping bbeh, hle, and LLM-graded)
TEXT_BENCHES="simpleqa,gpqa,chembench,babilong,arc_agi,oolong,livebench,omnimath"
# VLM benchmarks (skipping mmvet which is LLM-graded)
VLM_BENCHES="mmmu,charxiv,mathvista,mathverse,mathvision,mmstar,realworldqa,hallusionbench,vizwiz,aokvqa,vsr"

echo "=============================================="
echo "Qwen3.5-397B Sampling (THINKING + AUTO-RESTART)"
echo "$(date)"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

start_vllm() {
    echo "[$(date +%H:%M:%S)] Starting vLLM server..."
    $CONDA_ENV -m vllm.entrypoints.openai.api_server \
        --model /scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8 \
        --port 8100 --tensor-parallel-size 8 --max-model-len 32768 \
        --trust-remote-code --gpu-memory-utilization 0.90 --dtype auto \
        --enable-prefix-caching \
        --max-num-seqs 32 \
        > "$LOGDIR/vllm_restart.log" 2>&1 &
    VLLM_PID=$!
    echo "[$(date +%H:%M:%S)] vLLM PID=$VLLM_PID"

    for i in $(seq 1 120); do
        if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] vLLM server ready after ${i}0 seconds"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] ERROR: vLLM process died during startup"
            return 1
        fi
        sleep 10
    done
    echo "[$(date +%H:%M:%S)] ERROR: vLLM not ready after 20 minutes"
    kill $VLLM_PID 2>/dev/null
    return 1
}

kill_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "[$(date +%H:%M:%S)] Killing vLLM (PID=$VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        sleep 5
        # Kill any orphaned vLLM workers
        pkill -f "vllm.entrypoints" 2>/dev/null
        sleep 3
    fi
}

purge_api_failed() {
    echo "[$(date +%H:%M:%S)] Purging API_FAILED entries..."
    $CONDA_ENV -c "
import json
from pathlib import Path
purged = 0
for d in Path('runs').glob('qwen35_397b_*'):
    if 'nothinking' in d.name: continue
    pred = d / 'predictions.jsonl'
    if not pred.exists(): continue
    rows = [json.loads(l) for l in open(pred)]
    good = [r for r in rows if 'API_FAILED' not in r.get('response_text','')]
    bad = len(rows) - len(good)
    if bad:
        purged += bad
        with open(pred, 'w') as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Purged {purged} API_FAILED entries')
"
}

print_progress() {
    $CONDA_ENV -c "
import json
from pathlib import Path
total = 0
for d in sorted(Path('runs').glob('qwen35_397b_*')):
    if 'nothinking' in d.name: continue
    pred = d / 'predictions.jsonl'
    if not pred.exists(): continue
    rows = [json.loads(l) for l in open(pred)]
    real = sum(1 for r in rows if 'API_FAILED' not in r.get('response_text',''))
    if real == 0: continue
    bench = d.name.replace('qwen35_397b_','')
    target = 150 if bench == 'mmmu' else (100 if bench in ('babilong','aokvqa') else 250)
    total += real
    print(f'  {bench:20s} {real:4d}/{target}')
print(f'Total: {total} real predictions')
"
}

run_benchmarks() {
    local MODEL_NAME
    MODEL_NAME=$(curl -s "$BASE_URL/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
    echo "[$(date +%H:%M:%S)] Model: $MODEL_NAME"

    echo "[$(date +%H:%M:%S)] Launching text (parallel=4) + VLM (parallel=4)..."
    $CONDA_ENV scripts/run_all_qwen35_397b.py \
        --base_url "$BASE_URL" \
        --model_name "$MODEL_NAME" \
        --benchmarks "$TEXT_BENCHES" \
        --parallel 4 \
        2>&1 | tee "$LOGDIR/restart_text.log" &
    local TEXT_PID=$!

    $CONDA_ENV scripts/run_all_qwen35_397b_vlm.py \
        --base_url "$BASE_URL" \
        --model_name "$MODEL_NAME" \
        --benchmarks "$VLM_BENCHES" \
        --parallel 4 \
        2>&1 | tee "$LOGDIR/restart_vlm.log" &
    local VLM_PID=$!

    echo "[$(date +%H:%M:%S)] Text PID=$TEXT_PID, VLM PID=$VLM_PID (8 concurrent workers)"

    # Wait for both — they may exit early if vLLM crashes
    wait $TEXT_PID 2>/dev/null
    wait $VLM_PID 2>/dev/null
    echo "[$(date +%H:%M:%S)] Benchmark workers exited."
}

check_all_done() {
    # Returns 0 if all benchmarks are complete
    $CONDA_ENV -c "
import json, sys
from pathlib import Path

targets = {
    'simpleqa':250,'gpqa':198,'chembench':250,'babilong':100,
    'arc_agi':250,'oolong':250,'livebench':250,'omnimath':250,
    'mmmu':150,'charxiv':250,'mathvista':250,'mathverse':250,
    'mathvision':250,'mmstar':250,'realworldqa':250,
    'hallusionbench':250,'vizwiz':250,'aokvqa':100,'vsr':250
}
for bench, target in targets.items():
    pred = Path(f'runs/qwen35_397b_{bench}/predictions.jsonl')
    if not pred.exists():
        sys.exit(1)
    rows = [json.loads(l) for l in open(pred)]
    real = sum(1 for r in rows if 'API_FAILED' not in r.get('response_text',''))
    if real < target:
        sys.exit(1)
sys.exit(0)
"
}

# ============================================
# Main loop: start vLLM, run benchmarks, restart on crash
# ============================================
MAX_RESTARTS=10
for attempt in $(seq 1 $MAX_RESTARTS); do
    echo ""
    echo "======== ATTEMPT $attempt/$MAX_RESTARTS ($(date)) ========"

    # Check if we're already done
    if check_all_done; then
        echo "[$(date +%H:%M:%S)] All benchmarks complete!"
        break
    fi

    # Purge failed entries from previous crash
    purge_api_failed

    # Start vLLM
    if ! start_vllm; then
        echo "[$(date +%H:%M:%S)] Failed to start vLLM, retrying in 30s..."
        kill_vllm
        sleep 30
        continue
    fi

    # Run benchmarks (blocks until workers exit)
    run_benchmarks

    # Check if vLLM is still alive
    if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] vLLM still alive — benchmarks completed normally."
        kill_vllm
        break
    else
        echo "[$(date +%H:%M:%S)] vLLM crashed! Restarting..."
        kill_vllm
        print_progress
    fi
done

# --- Final Summary ---
echo ""
echo "=============================================="
echo "FINAL RESULTS - $(date)"
echo "=============================================="
purge_api_failed
print_progress
