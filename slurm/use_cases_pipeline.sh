#!/bin/bash
#SBATCH --job-name=uq_usecases
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/use_cases_pipeline_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/use_cases_pipeline_%j.log

# UQ Use Case Pipeline
# Phase 1: Score all samples with calibrator (GPU, ~45 min per target)
# Phase 2: Run all 7 use case analyses (CPU, ~10 min)
#
# Smoke test: sbatch --export=SMOKE=1 slurm/use_cases_pipeline.sh

set -e

echo "=========================================="
echo "UQ Use Case Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Smoke mode: ${SMOKE:-0}"
echo "=========================================="

cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke_test"
fi

mkdir -p data/use_cases/scored data/use_cases/results figures/use_cases logs

echo ""
echo "=========================================="
echo "PHASE 1: Scoring all samples (GPU)"
echo "Start: $(date)"
echo "=========================================="

# Score all 3 targets in parallel on separate GPUs
CUDA_VISIBLE_DEVICES=0 python scripts/score_all_samples.py --target gpt5mini $SMOKE_FLAG &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python scripts/score_all_samples.py --target gpt52 $SMOKE_FLAG &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python scripts/score_all_samples.py --target qwen35 $SMOKE_FLAG &
PID3=$!

echo "Scoring PIDs: $PID1 (gpt5mini) $PID2 (gpt52) $PID3 (qwen35)"
# Use || true to prevent set -e from aborting if a scoring job fails
wait $PID1 && RC1=0 || RC1=$?
echo "gpt5mini done: exit=$RC1 $(date)"
wait $PID2 && RC2=0 || RC2=$?
echo "gpt52 done: exit=$RC2 $(date)"
wait $PID3 && RC3=0 || RC3=$?
echo "qwen35 done: exit=$RC3 $(date)"

echo ""
echo "Scoring exit codes: gpt5mini=$RC1 gpt52=$RC2 qwen35=$RC3"

# Check that at least gpt5mini and gpt52 scored (required for routing/cascade)
if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
    echo "CRITICAL: gpt5mini or gpt52 scoring failed. Aborting."
    exit 1
fi

echo ""
echo "Scored files:"
ls -la data/use_cases/scored/*.jsonl 2>/dev/null
wc -l data/use_cases/scored/*.jsonl 2>/dev/null

echo ""
echo "=========================================="
echo "PHASE 2: Use Case Analysis (CPU)"
echo "Start: $(date)"
echo "=========================================="

# Run each use case sequentially
for uc in 1 2 3 4 5 6 7; do
    echo ""
    echo "--- UC${uc} ---"
    python scripts/uc${uc}_*.py 2>&1 || echo "UC${uc} failed with exit code $?"
done

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="

echo "Scored data:"
wc -l data/use_cases/scored/*.jsonl 2>/dev/null

echo ""
echo "Result files:"
ls -la data/use_cases/results/*.json 2>/dev/null

echo ""
echo "Figures:"
ls -la figures/use_cases/*.pdf 2>/dev/null

echo ""
echo "Done: $(date)"
