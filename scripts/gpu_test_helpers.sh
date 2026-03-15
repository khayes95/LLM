#!/bin/bash
# Shared helper functions for GPU orphan/brick test scripts.
# Source this from SLURM scripts: source scripts/gpu_test_helpers.sh

gpu_health_check() {
    # Check if the assigned GPU is healthy before running.
    # Returns 1 if GPU is in ERR state so the script can abort.
    local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
    local status
    status=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --id="$gpu_id" 2>&1)
    if echo "$status" | grep -qiE "Unknown Error|ERR|error"; then
        echo "ERROR: GPU $gpu_id is in ERR state. Needs manual reset before this test can run."
        echo "  Run: nvidia-smi --gpu-reset --id=$gpu_id"
        # Uncomment the next line to auto-reset (requires root):
        # gpu_reset
        return 1
    fi
    echo "GPU $gpu_id healthy (temp: ${status}C)"
    return 0
}

gpu_reset() {
    # Reset the assigned GPU. Requires root.
    local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
    echo "Resetting GPU $gpu_id..."
    nvidia-smi --gpu-reset --id="$gpu_id" 2>&1 || echo "WARNING: GPU reset failed (may need node reboot)"
}

kill_orphans() {
    # Find and kill any orphaned vLLM processes belonging to khayes.
    local orphans
    orphans=$(ps aux | grep -E "[V]LLM|[v]llm" | grep khayes | awk '{print $2}')
    if [ -n "$orphans" ]; then
        echo "Found orphaned vLLM processes: $orphans"
        echo "Killing..."
        echo "$orphans" | xargs kill -9 2>/dev/null || true
        sleep 2
        local remaining
        remaining=$(ps aux | grep -E "[V]LLM|[v]llm" | grep khayes | awk '{print $2}')
        if [ -n "$remaining" ]; then
            echo "WARNING: These processes survived kill -9 (zombies): $remaining"
        else
            echo "All orphans killed."
        fi
    else
        echo "No orphaned vLLM processes found."
    fi
}

post_test_report() {
    # Run after each test to report GPU state and orphan status.
    echo ""
    echo "=== POST-TEST REPORT ==="
    echo "Time: $(date)"
    echo ""
    echo "Orphan check:"
    kill_orphans
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu,power.draw --format=csv 2>&1
    echo ""
    local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
    local status
    status=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader --id="$gpu_id" 2>&1)
    if echo "$status" | grep -qiE "Unknown Error|ERR|error"; then
        echo "RESULT: GPU $gpu_id is FAULTED. Reset needed before next test."
        echo "  Run: nvidia-smi --gpu-reset --id=$gpu_id"
        # Uncomment the next line to auto-reset (requires root):
        # gpu_reset
    else
        echo "RESULT: GPU $gpu_id is healthy."
    fi
    echo "=== END REPORT ==="
}
