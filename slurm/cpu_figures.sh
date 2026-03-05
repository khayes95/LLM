#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=figures
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/figures_%j.out

cd /scratch/khayes/LLM

PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"

echo "=== Paper Figure Generation ==="
echo "Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"

$PYTHON scripts/cpu_generate_all_figures.py \
    --fig_dir figures/paper \
    "$@"

echo "End: $(date)"
