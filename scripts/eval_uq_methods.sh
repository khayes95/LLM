#!/usr/bin/env bash
#SBATCH -J uq_eval_methods
#SBATCH -p GPU
#SBATCH -w gpunode00
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o logs/uq_eval_methods_%j.out
#SBATCH -e logs/uq_eval_methods_%j.err

# Evaluate UQ methods (verbalized, zero-shot, fine-tuned)
# Usage: sbatch scripts/eval_uq_methods.sh [runs_dir] [base_model] [uq_model]

set -e

RUNS_DIR="${1:-runs}"
BASE_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
UQ_MODEL="${3:-}"

cd /scratch/$USER/LLM

source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="results/uq_eval_${TIMESTAMP}.json"

echo "=========================================="
echo "UQ METHODS EVALUATION"
echo "Started at: $(date)"
echo "Runs dir: $RUNS_DIR"
echo "Base model: $BASE_MODEL"
echo "UQ model: $UQ_MODEL"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

UQ_ARGS="--runs_dir $RUNS_DIR --base_model $BASE_MODEL --output $OUTPUT_FILE"

if [ -n "$UQ_MODEL" ]; then
    UQ_ARGS="$UQ_ARGS --uq_model $UQ_MODEL"
fi

python -m uq_eval.uq_evaluate $UQ_ARGS

echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

# Print results
echo ""
echo "Results:"
cat "$OUTPUT_FILE"
