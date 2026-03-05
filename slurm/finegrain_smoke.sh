#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/finegrain_smoke_%j.out
#SBATCH --job-name=fg_smoke

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=== FineGRAIN Smoke Test ==="
date

CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --checkpoint uq_models/best_v2_r32_combined \
    --prompt_variant combined \
    --output_dir data/finegrain_uq/smoke \
    --smoke_test

echo "=== Done ==="
date
