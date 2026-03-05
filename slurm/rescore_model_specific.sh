#!/bin/bash
#SBATCH --job-name=rescore_uq
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/rescore_model_specific_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/rescore_model_specific_%j.log

# Re-score all predictions with model-specific calibrators
# GPT-5-mini → text_calibrator_v3 (unchanged)
# GPT-5.2 → text_calibrator_gpt52
# Qwen3.5 → text_calibrator_qwen35

set -e

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=========================================="
echo "Re-scoring with model-specific calibrators"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Smoke mode: ${SMOKE:-0}"
echo "=========================================="

SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke_test"
fi

OUTPUT_DIR="data/use_cases/scored_test_only_v2"
mkdir -p "$OUTPUT_DIR"

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Score all 3 targets in parallel on separate GPUs, each with its own calibrator
CUDA_VISIBLE_DEVICES=1 python scripts/score_all_samples.py \
    --target gpt5mini \
    --calibrator uq_models/text_calibrator_v3 \
    --output_dir "$OUTPUT_DIR" \
    $SMOKE_FLAG &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python scripts/score_all_samples.py \
    --target gpt52 \
    --calibrator uq_models/text_calibrator_gpt52 \
    --output_dir "$OUTPUT_DIR" \
    $SMOKE_FLAG &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python scripts/score_all_samples.py \
    --target qwen35 \
    --calibrator uq_models/text_calibrator_qwen35 \
    --output_dir "$OUTPUT_DIR" \
    $SMOKE_FLAG &
PID3=$!

echo "PIDs: $PID1 (gpt5mini) $PID2 (gpt52) $PID3 (qwen35)"

wait $PID1 && RC1=0 || RC1=$?
echo "gpt5mini done: exit=$RC1 $(date)"
wait $PID2 && RC2=0 || RC2=$?
echo "gpt52 done: exit=$RC2 $(date)"
wait $PID3 && RC3=0 || RC3=$?
echo "qwen35 done: exit=$RC3 $(date)"

echo ""
echo "Exit codes: gpt5mini=$RC1 gpt52=$RC2 qwen35=$RC3"

if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ] || [ "$RC3" -ne 0 ]; then
    echo "WARNING: One or more scoring jobs failed"
fi

echo ""
echo "Scored files:"
wc -l "$OUTPUT_DIR"/*.jsonl 2>/dev/null
echo ""
echo "Done: $(date)"
