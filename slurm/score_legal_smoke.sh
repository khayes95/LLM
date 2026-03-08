#!/bin/bash
#SBATCH --job-name=legal_smoke
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/score_legal_smoke_%j.out

CUDA_VISIBLE_DEVICES=0 python scripts/score_legal_hallucinations.py --smoke_test

echo "Done! Exit code: $?"
