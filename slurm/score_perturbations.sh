#!/bin/bash
#SBATCH --job-name=score_perturb
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/score_perturbations_%j.log
#SBATCH --error=/scratch/khayes/LLM/logs/score_perturbations_%j.log

# Score all perturbations with the unified UQ model (4 GPUs)
# ~58k perturbations, ~4-6 hours with 4 GPUs

# Smoke test (uncomment to test with 100 samples first):
# /scratch/khayes/.conda/envs/uq_eval/bin/python scripts/score_perturbations.py --smoke_test --num_gpus 1

set -e

cd /scratch/khayes/LLM

echo "=========================================="
echo "Score Perturbations with Unified UQ Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

export CUDA_VISIBLE_DEVICES=0,1,2,3

/scratch/khayes/.conda/envs/uq_eval/bin/python scripts/score_perturbations.py \
    --input data/use_cases/perturbations/all_perturbations.jsonl \
    --output data/use_cases/perturbations/scored_perturbations.jsonl \
    --checkpoint uq_models/best_unified \
    --num_gpus 4

echo ""
echo "Done: $(date)"
echo "=========================================="
