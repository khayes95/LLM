#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/finegrain_uq_%j.out
#SBATCH --job-name=finegrain_uq

# FineGRAIN x UQ Evaluation
# Scores FineGRAIN T2I failure mode detection with our UQ calibrator
# 3800 samples, ~30 min on 1 GPU

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH

echo "=== FineGRAIN UQ Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
mkdir -p logs data/finegrain_uq

# Run full evaluation with v2 checkpoint (combined prompt)
CUDA_VISIBLE_DEVICES=3 python scripts/finegrain_uq_eval.py \
    --checkpoint uq_models/best_v2_r32_combined \
    --prompt_variant combined \
    --output_dir data/finegrain_uq

echo ""
echo "=== Done ==="
echo "Results: data/finegrain_uq/results.json"
echo "Scored:  data/finegrain_uq/scored_samples.jsonl"
date
