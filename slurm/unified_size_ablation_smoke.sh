#!/bin/bash
#SBATCH --job-name=uq_sa_smk
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/khayes/LLM/logs/size_ablation_smoke_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/size_ablation_smoke_%j.log

set -e
cd /scratch/khayes/LLM

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Smoke test: Qwen3-VL-2B unified size ablation"
echo "Start: $(date)"

python scripts/unified_size_ablation.py --smoke_test --models 2b --epochs 1

echo "Done: $(date)"
