#!/bin/bash
#SBATCH --job-name=qwen35_sampling
#SBATCH --partition=GPU
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/restart_sampling_%j.log
#SBATCH --error=logs/restart_sampling_%j.log

# Smoke test: sbatch slurm/restart_sampling.sh
# Or with --smoke_test flag: sbatch slurm/restart_sampling.sh --smoke_test

source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

cd /scratch/khayes/LLM
bash scripts/restart_sampling.sh "$@"
