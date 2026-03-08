#!/bin/bash
#SBATCH --job-name=score_unseen
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/score_unseen_%j.out

# Smoke test (uncomment to test first):
# CUDA_VISIBLE_DEVICES=0 python scripts/score_unseen_benchmarks.py --smoke_test

# Full run: Score unseen benchmarks (healthbench x3, triviaqa x1)
# ~873 samples total, ~30-45 min on 1 GPU
CUDA_VISIBLE_DEVICES=0 python scripts/score_unseen_benchmarks.py

echo "Done! Exit code: $?"
