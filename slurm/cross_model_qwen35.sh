#!/bin/bash
#SBATCH --job-name=xm_q35
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/cross_model_qwen35_%j.log
#SBATCH --error=logs/cross_model_qwen35_%j.log

# Cross-model evaluation on fresh Qwen3.5-397B responses
# Runs all 3 text calibrators on Qwen3.5 target (updated responses)
# Also runs VLM judge on Qwen3.5 VLM responses
#
# Submit as dependency: sbatch --dependency=afterany:7714 slurm/cross_model_qwen35.sh

echo "=========================================="
echo "Cross-Model Eval: Qwen3.5-397B (updated)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

source ~/.bashrc
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

mkdir -p data/cross_model

# Check how many Qwen3.5 predictions we have now
echo ""
echo "Qwen3.5 prediction counts:"
TOTAL=0
for d in runs/qwen35_397b_*/; do
    bench=$(basename "$d" | sed 's/qwen35_397b_//')
    if [ -f "$d/predictions.jsonl" ]; then
        count=$(wc -l < "$d/predictions.jsonl" 2>/dev/null || echo 0)
        echo "  $bench: $count"
        TOTAL=$((TOTAL + count))
    fi
done
echo "Total: $TOTAL"

if [ "$TOTAL" -lt 100 ]; then
    echo "ERROR: Too few predictions ($TOTAL). Qwen3.5 generation may have failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 1: Text calibrators on Qwen3.5 (3 parallel GPUs)"
echo "Start: $(date)"
echo "=========================================="

# v3 calibrator
CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_v3 \
    --target qwen35 \
    --output data/cross_model/text_v3_on_qwen35_full.json &
PID1=$!

# Combined calibrator
CUDA_VISIBLE_DEVICES=1 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_combined \
    --target qwen35 \
    --output data/cross_model/text_combined_on_qwen35_full.json &
PID2=$!

# GPT-5.2 calibrator
CUDA_VISIBLE_DEVICES=2 python scripts/cross_model_eval_v3.py \
    --calibrator uq_models/text_calibrator_gpt52 \
    --target qwen35 \
    --output data/cross_model/text_gpt52cal_on_qwen35_full.json &
PID3=$!

echo "Waiting for text evals: PIDs $PID1 $PID2 $PID3"
wait $PID1; RC1=$?
wait $PID2; RC2=$?
wait $PID3; RC3=$?

echo "Text evals done: $(date)"
echo "  v3→qwen35: exit=$RC1"
echo "  combined→qwen35: exit=$RC2"
echo "  gpt52cal→qwen35: exit=$RC3"

echo ""
echo "=========================================="
echo "STEP 2: VLM judge on Qwen3.5 (if VLM responses available)"
echo "Start: $(date)"
echo "=========================================="

# Check if there are VLM predictions for Qwen3.5
VLM_COUNT=0
for bench in vsr mmmu charxiv hallusionbench mathvista realworldqa mathvision vizwiz mmstar mmvet; do
    d="runs/qwen35_397b_${bench}"
    if [ -f "$d/predictions.jsonl" ]; then
        c=$(wc -l < "$d/predictions.jsonl" 2>/dev/null || echo 0)
        VLM_COUNT=$((VLM_COUNT + c))
    fi
done
echo "VLM predictions available: $VLM_COUNT"

if [ "$VLM_COUNT" -ge 50 ]; then
    echo "Running VLM judge cross-model eval..."
    CUDA_VISIBLE_DEVICES=3 python scripts/vlm_judge_cross_model_eval.py \
        --target qwen35 \
        --output data/cross_model/vlm_judge_vsr_fixed_on_qwen35_full.json
    RC4=$?
    echo "VLM eval exit code: $RC4"
else
    echo "Skipping VLM eval — insufficient predictions"
    RC4="skipped"
fi

# Print results
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
for f in data/cross_model/text_*_on_qwen35_full.json data/cross_model/vlm_*_on_qwen35_full.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        auroc=$(python3 -c "import json; print(f'{json.load(open(\"$f\"))[\"auroc\"]:.4f}')" 2>/dev/null || echo "error")
        n=$(python3 -c "import json; print(json.load(open('$f')).get('n_samples', '?'))" 2>/dev/null || echo "?")
        echo "  $name: AUROC=$auroc (N=$n)"
    fi
done

echo ""
echo "Exit codes: text_v3=$RC1 text_combined=$RC2 text_gpt52=$RC3 vlm=$RC4"
echo "Done: $(date)"
