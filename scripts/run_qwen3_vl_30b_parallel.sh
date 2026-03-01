#!/bin/bash
# Run Qwen3-VL-30B-A3B-Thinking on multiple GPUs in parallel
# Each GPU runs a different subset of benchmarks
#
# Usage: bash scripts/run_qwen3_vl_30b_parallel.sh

cd /scratch/khayes/LLM
mkdir -p logs

MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
MAX_SAMPLES=1000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Parallel Qwen3-VL-30B-A3B-Thinking Evaluation"
echo "============================================================"
echo "Model: $MODEL"
echo "Max samples: $MAX_SAMPLES"
echo "Timestamp: $TIMESTAMP"
echo ""

# Split benchmarks across 4 GPUs (each 30B model needs ~76GB, fits on 1 A100)
# GPU 0: hle_multimodal (the big one currently running - skip if resuming)
# GPU 1: mathverse, mathvision, mathvista
# GPU 2: mmstar, realworldqa, vizwiz, erqa
# GPU 3: charxiv, hle, mmmu

# Check which benchmarks are already done
DONE_DIR="/scratch/khayes/LLM/runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking"

echo "Checking completed benchmarks in $DONE_DIR..."
for b in charxiv erqa mathverse mathvision mathvista mmstar realworldqa vizwiz hle_multimodal mmmu hle; do
    if [ -f "$DONE_DIR/$b/predictions.jsonl" ]; then
        count=$(wc -l < "$DONE_DIR/$b/predictions.jsonl")
        echo "  $b: DONE ($count samples)"
    else
        echo "  $b: PENDING"
    fi
done
echo ""

# GPU 0: hle_multimodal (if not done)
if [ ! -f "$DONE_DIR/hle_multimodal/predictions.jsonl" ]; then
    echo "Starting GPU 0: hle_multimodal"
    CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen3_vl_30b_full.py \
        --model "$MODEL" \
        --max_samples $MAX_SAMPLES \
        --benchmarks hle_multimodal \
        > logs/qwen3_gpu0_${TIMESTAMP}.log 2>&1 &
    PID0=$!
    echo "  PID: $PID0"
else
    echo "GPU 0: hle_multimodal already done, skipping"
    PID0=""
fi

# GPU 1: mathverse, mathvision, mathvista (skip if done)
BENCH1=""
for b in mathverse mathvision mathvista; do
    if [ ! -f "$DONE_DIR/$b/predictions.jsonl" ]; then
        BENCH1="$BENCH1 $b"
    fi
done
BENCH1=$(echo $BENCH1 | xargs)  # trim whitespace

if [ -n "$BENCH1" ]; then
    echo "Starting GPU 1: $BENCH1"
    CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen3_vl_30b_full.py \
        --model "$MODEL" \
        --max_samples $MAX_SAMPLES \
        --benchmarks $BENCH1 \
        > logs/qwen3_gpu1_${TIMESTAMP}.log 2>&1 &
    PID1=$!
    echo "  PID: $PID1"
else
    echo "GPU 1: all benchmarks done, skipping"
    PID1=""
fi

# GPU 2: mmstar, realworldqa, vizwiz, erqa (skip if done)
BENCH2=""
for b in mmstar realworldqa vizwiz erqa; do
    if [ ! -f "$DONE_DIR/$b/predictions.jsonl" ]; then
        BENCH2="$BENCH2 $b"
    fi
done
BENCH2=$(echo $BENCH2 | xargs)

if [ -n "$BENCH2" ]; then
    echo "Starting GPU 2: $BENCH2"
    CUDA_VISIBLE_DEVICES=2 python scripts/run_qwen3_vl_30b_full.py \
        --model "$MODEL" \
        --max_samples $MAX_SAMPLES \
        --benchmarks $BENCH2 \
        > logs/qwen3_gpu2_${TIMESTAMP}.log 2>&1 &
    PID2=$!
    echo "  PID: $PID2"
else
    echo "GPU 2: all benchmarks done, skipping"
    PID2=""
fi

# GPU 3: charxiv, hle, mmmu (skip if done)
BENCH3=""
for b in charxiv hle mmmu; do
    if [ ! -f "$DONE_DIR/$b/predictions.jsonl" ]; then
        BENCH3="$BENCH3 $b"
    fi
done
BENCH3=$(echo $BENCH3 | xargs)

if [ -n "$BENCH3" ]; then
    echo "Starting GPU 3: $BENCH3"
    CUDA_VISIBLE_DEVICES=3 python scripts/run_qwen3_vl_30b_full.py \
        --model "$MODEL" \
        --max_samples $MAX_SAMPLES \
        --benchmarks $BENCH3 \
        > logs/qwen3_gpu3_${TIMESTAMP}.log 2>&1 &
    PID3=$!
    echo "  PID: $PID3"
else
    echo "GPU 3: all benchmarks done, skipping"
    PID3=""
fi

echo ""
echo "============================================================"
echo "All jobs launched!"
echo "============================================================"
echo ""
echo "Monitor with:"
echo "  tail -f logs/qwen3_gpu*_${TIMESTAMP}.log"
echo "  nvidia-smi -l 5"
echo ""
echo "PIDs: $PID0 $PID1 $PID2 $PID3"
echo ""

# Wait for all
echo "Waiting for all jobs to complete..."
[ -n "$PID0" ] && wait $PID0
[ -n "$PID1" ] && wait $PID1
[ -n "$PID2" ] && wait $PID2
[ -n "$PID3" ] && wait $PID3

echo ""
echo "============================================================"
echo "All jobs complete!"
echo "============================================================"
