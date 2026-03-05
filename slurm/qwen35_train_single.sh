#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/qwen35_%x_%j.out
#SBATCH --job-name=q35

# Qwen3.5 single model training
# Usage: sbatch --export=ALL,SIZE=0.8b,GPU=2 --job-name=q35_0.8b slurm/qwen35_train_single.sh

set -e
cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

# Validate required env vars
if [ -z "$SIZE" ] || [ -z "$GPU" ]; then
    echo "ERROR: SIZE and GPU must be set"
    echo "Usage: sbatch --export=ALL,SIZE=0.8b,GPU=2 slurm/qwen35_train_single.sh"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

echo "Start: $(date)"
echo "Model size: $SIZE"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"

# Model configs
case $SIZE in
    0.8b)
        MODEL="Qwen/Qwen3.5-0.8B"
        LORA_R=16; LORA_ALPHA=32; LR=2e-4; BATCH=1; GRAD_ACCUM=32
        ;;
    2b)
        MODEL="Qwen/Qwen3.5-2B"
        LORA_R=16; LORA_ALPHA=32; LR=1e-4; BATCH=1; GRAD_ACCUM=16
        ;;
    4b)
        MODEL="Qwen/Qwen3.5-4B"
        LORA_R=16; LORA_ALPHA=32; LR=1e-4; BATCH=1; GRAD_ACCUM=16
        ;;
    9b)
        MODEL="Qwen/Qwen3.5-9B"
        LORA_R=32; LORA_ALPHA=64; LR=1e-4; BATCH=1; GRAD_ACCUM=16
        ;;
    *)
        echo "ERROR: Unknown size $SIZE"
        exit 1
        ;;
esac

OUTPUT_DIR="data/ablations/qwen35_model_size/$SIZE"
SPLIT_INFO="uq_models/best_v2_r32_combined/split_info.json"

echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "LR=$LR, batch=$BATCH, grad_accum=$GRAD_ACCUM"
echo ""

python scripts/train_best_uq.py \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$MODEL" \
    --split_info "$SPLIT_INFO" \
    --epochs 3 \
    --batch_size $BATCH \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --prompt_variant combined

echo ""
echo "End: $(date)"

# Print results if available
if [ -f "$OUTPUT_DIR/results.json" ]; then
    echo "Results:"
    cat "$OUTPUT_DIR/results.json"
fi
