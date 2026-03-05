#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/features_%j.out

cd /scratch/khayes/LLM

PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"

echo "=== Feature Extraction & Error Analysis ==="
echo "Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"

$PYTHON scripts/cpu_feature_analysis.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/feature_analysis.json \
    "$@"

echo "End: $(date)"
