#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#No --mem for GPU partition
#SBATCH --time=00:30:00
#SBATCH --job-name=no_meta_smk
#SBATCH --output=logs/ablation_no_metadata_smoke_%j.out

cd /scratch/khayes/LLM
source activate uq_eval

echo "Starting no-metadata ablation SMOKE TEST: $(date)"

CUDA_VISIBLE_DEVICES=2 python scripts/ablation_no_metadata.py \
    --smoke_test \
    --checkpoint uq_models/best_v2_r32_combined \
    --output_dir data/ablations/no_metadata_smoke

echo "Finished: $(date)"
