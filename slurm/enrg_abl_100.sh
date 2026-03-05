#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=enrg_abl_100
#SBATCH --output=logs/enrg_abl_100_%j.out

# Smoke test: CUDA_VISIBLE_DEVICES=0 python uq_models/energetics_ablation/frac_100/run_train.py

cd /scratch/khayes/LLM
source /scratch/khayes/anaconda3/etc/profile.d/conda.sh
conda activate uq_eval

CUDA_VISIBLE_DEVICES=3,7 python uq_models/energetics_ablation/frac_100/run_train.py
