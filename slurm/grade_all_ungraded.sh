#!/bin/bash
#SBATCH --job-name=grade_uq
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/grade_all_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/grade_all_%j.log

# Grade all ungraded predictions using GPT-5-mini as LLM judge
# CPU-only (API calls), ~933 samples, ~$0.50-1.00

set -e
cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

bash scripts/grade_all_ungraded.sh
