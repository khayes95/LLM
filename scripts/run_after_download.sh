#!/bin/bash
# Wait for download to complete, then run evaluation

MODEL_DIR="/scratch/khayes/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking"
MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
EXPECTED_FILES=13

echo "Waiting for Qwen3-VL-30B-A3B-Thinking download to complete..."
echo "Expected: $EXPECTED_FILES safetensor files"

while true; do
    # Count completed safetensor files
    if [ -d "$MODEL_DIR/snapshots" ]; then
        COMPLETED=$(find "$MODEL_DIR/snapshots" -name "model-*.safetensors" 2>/dev/null | wc -l)
    else
        COMPLETED=0
    fi

    # Check for incomplete files
    INCOMPLETE=$(find "$MODEL_DIR" -name "*.incomplete" 2>/dev/null | wc -l)

    # Get current size
    SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)

    echo "[$(date '+%H:%M:%S')] Downloaded: $SIZE, Files: $COMPLETED/$EXPECTED_FILES, Incomplete: $INCOMPLETE"

    if [ "$COMPLETED" -ge "$EXPECTED_FILES" ] && [ "$INCOMPLETE" -eq 0 ]; then
        echo ""
        echo "Download complete! Starting evaluation..."
        break
    fi

    sleep 30
done

# Run the evaluation
echo "=============================================="
echo "Starting Qwen3-VL-30B-Thinking Full Evaluation"
echo "=============================================="

cd /scratch/khayes/LLM

CUDA_VISIBLE_DEVICES=0 /scratch/khayes/anaconda3/envs/finegrain_vlm/bin/python scripts/run_qwen3_vl_30b_full.py \
    --model "$MODEL" \
    --tensor_parallel_size 1 \
    --max_samples 1000 \
    2>&1 | tee logs/qwen3_vl_30b_thinking_full.log
