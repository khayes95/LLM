#!/bin/bash
# Wait for oolong to complete and GPUs to be free, then start UQ training

cd /scratch/khayes/LLM

echo "=== Wait-and-Train Script ==="
echo "Started at: $(date)"
echo ""

# Configuration
TARGET_SAMPLES=250
OOLONG_PREDICTIONS="runs/gpt5_mini_combined/oolong/predictions.jsonl"
OOLONG_SAMPLED_IDS="runs/gpt5_mini_combined/oolong/sampled_ids.json"
TRAIN_GPUS="0,1,2,3"
GPU_MEMORY_THRESHOLD=5000  # MB - GPUs with less than this are considered "free"
CHECK_INTERVAL=60  # seconds between checks

# Function to count unique oolong predictions
count_oolong() {
    python3 -c "
import json
preds = [json.loads(l) for l in open('$OOLONG_PREDICTIONS')]
ids = set(p.get('id') for p in preds)
print(len(ids))
" 2>/dev/null || echo "0"
}

# Function to check if GPUs are free
gpus_are_free() {
    # Check if GPUs 0-3 have less than threshold memory used
    local free_count=0
    for gpu in 0 1 2 3; do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        if [ -n "$mem_used" ] && [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD" ]; then
            ((free_count++))
        fi
    done

    # All 4 GPUs must be free
    [ "$free_count" -eq 4 ]
}

# Wait for oolong to complete
echo "Step 1: Waiting for oolong to complete ($TARGET_SAMPLES samples)..."
while true; do
    current=$(count_oolong)
    echo "  $(date +%H:%M:%S) - Oolong progress: $current/$TARGET_SAMPLES"

    if [ "$current" -ge "$TARGET_SAMPLES" ]; then
        echo "  Oolong complete!"
        break
    fi

    # Also check if progress has stalled (file not modified in 10 minutes)
    if [ -f "$OOLONG_PREDICTIONS" ]; then
        last_mod=$(stat -c %Y "$OOLONG_PREDICTIONS" 2>/dev/null || echo 0)
        now=$(date +%s)
        age=$((now - last_mod))

        if [ "$age" -gt 600 ] && [ "$current" -gt 200 ]; then
            echo "  Warning: No progress in 10 minutes. Proceeding with $current samples."
            break
        fi
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "Step 2: Waiting for GPUs 0-3 to be free..."
while true; do
    echo -n "  $(date +%H:%M:%S) - GPU memory (MB): "
    for gpu in 0 1 2 3; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        echo -n "GPU$gpu=$mem "
    done
    echo ""

    if gpus_are_free; then
        echo "  All GPUs free!"
        break
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "Step 3: Starting UQ training..."
echo "Command: CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python scripts/train_uq_unified.py ..."
echo "Log: logs/train_uq_gpt5.log"
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS /scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python scripts/train_uq_unified.py \
    --data_dir runs/gpt5_mini_combined \
    --output_dir uq_models/gpt5_mini_uq \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 2>&1 | tee logs/train_uq_gpt5.log

echo ""
echo "=== Training complete ==="
echo "Finished at: $(date)"

# After training, run Qwen3 validation
echo ""
echo "Step 4: Running Qwen3-VL-30B validation..."
bash scripts/validate_and_train_qwen3.sh
