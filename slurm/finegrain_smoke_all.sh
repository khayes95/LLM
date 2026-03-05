#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=fg_smoke
#SBATCH --output=logs/fg_smoke_all_%j.out

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=========================================="
echo "FineGRAIN UQ: Smoke Tests (all scripts)"
echo "Started: $(date)"
echo "=========================================="

# Smoke test 1: Existing eval script with baseline variant
echo ""
echo ">>> Smoke test 1: finegrain_uq_eval.py --prompt_variant baseline"
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --smoke_test \
    --prompt_variant baseline \
    --output_dir data/finegrain_uq/smoke_baseline
echo ">>> PASS"

# Smoke test 2: New models script
echo ""
echo ">>> Smoke test 2: finegrain_uq_newmodels.py --smoke_test"
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_newmodels.py \
    --smoke_test \
    --models flux2_dev gpt_image15 \
    --output_dir data/finegrain_uq/smoke_newmodels
echo ">>> PASS"

# Smoke test 3: Analysis script (CPU only, uses existing data)
echo ""
echo ">>> Smoke test 3: finegrain_uq_analysis.py"
python scripts/finegrain_uq_analysis.py \
    --output_dir data/finegrain_uq/smoke_analysis
echo ">>> PASS"

echo ""
echo "=========================================="
echo "All smoke tests passed: $(date)"
echo "=========================================="
