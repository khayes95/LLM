#!/bin/bash
#SBATCH --job-name=uq_demos
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1M
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/khayes/LLM/logs/use_case_demos_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/use_case_demos_%j.log

# CPU-only: run use case demos on existing prediction data

set -e

cd /scratch/khayes/LLM

eval "$(conda shell.bash hook)"
conda activate uq_eval

python scripts/use_case_demos.py
