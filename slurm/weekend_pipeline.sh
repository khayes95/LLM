#!/bin/bash
#SBATCH --job-name=weekend
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/weekend_pipeline_%j.log
#SBATCH --error=logs/weekend_pipeline_%j.log

# Weekend Pipeline SLURM Wrapper
# Smoke test (uncomment to verify):
# bash scripts/weekend_pipeline.sh --smoke_test

echo "=========================================="
echo "Weekend Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

# Source API key first, then conda (bashrc resets PATH so conda must come after)
source /home/khayes/.bashrc
eval "$(/scratch/khayes/anaconda3/bin/conda shell.bash hook)"
conda activate uq_eval

echo "OPENAI_API_KEY set: ${OPENAI_API_KEY:0:8}..."
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
echo ""

# Run the pipeline, passing through all args
bash scripts/weekend_pipeline.sh "$@"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
