#!/bin/bash
#SBATCH --job-name=score_legal
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_legal_%j.out

# Smoke test (uncomment to test first):
# CUDA_VISIBLE_DEVICES=0 python scripts/score_legal_hallucinations.py --smoke_test

# Full run: ~2000 samples, ~1-2h on 1 GPU
CUDA_VISIBLE_DEVICES=0 python scripts/score_legal_hallucinations.py

echo "Done! Exit code: $?"
