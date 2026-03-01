#!/bin/bash
#SBATCH --job-name=gpt52-high
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpt52_high_%j.log
#SBATCH --error=logs/gpt52_high_%j.log

# Smoke test (uncomment to verify):
# bash scripts/run_all_gpt5_2_high.sh --max-samples 3

echo "=========================================="
echo "GPT-5.2 High Reasoning Eval"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

# Source conda from anaconda3 (matches working SLURM scripts)
source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

# Source API key
source /home/khayes/.bashrc

echo "OPENAI_API_KEY set: ${OPENAI_API_KEY:0:8}..."
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Run all benchmarks with 100 samples each
bash scripts/run_all_gpt5_2_high.sh --max-samples 100 --seed 42

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
