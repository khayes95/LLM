#!/bin/bash
#SBATCH --job-name=nccl_v2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_nccl_v2_%j.log
#SBATCH --error=logs/test_nccl_v2_%j.log

cd /scratch/khayes/LLM
source ~/.bashrc
conda activate uq_eval

python scripts/test_nccl_v2.py 2>&1
echo "Exit code: $?"
