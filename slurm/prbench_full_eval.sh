#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prbench_full_%j.out
#SBATCH --job-name=prbench_full

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate uq_eval

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY env var}"

echo "=========================================="
echo "PRBench Full Evaluation - Legal Splits"
echo "Start: $(date)"
echo "=========================================="

# Run legal splits across both models
python scripts/eval_prbench_full.py --splits legal,legal_hard

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
