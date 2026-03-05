#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=bootstrap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/bootstrap_%j.out

cd /scratch/khayes/LLM

PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"

echo "=== Exhaustive Bootstrap (100K iterations, BCa) ==="
echo "Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"

$PYTHON scripts/cpu_exhaustive_bootstrap.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/exhaustive_bootstrap.json \
    "$@"

echo "End: $(date)"
