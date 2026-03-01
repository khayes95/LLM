#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/phase5_analysis_%j.out
#SBATCH --job-name=phase5

# Phase 5 analysis: reliability diagrams, confidence histograms,
# bootstrap CIs, selective prediction curves, use case summary.
# CPU-only (no GPU), runs on GPU partition for conda/filesystem access.

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "============================================"
echo "Phase 5 Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

python scripts/phase5_analysis.py \
    --scored_dir data/use_cases/scored_unified \
    --output_dir data/use_cases/results_unified \
    --fig_dir figures/paper \
    --n_bootstrap 2000

echo "Done: $(date)"
