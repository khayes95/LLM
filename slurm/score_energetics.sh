#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/score_energetics_%j.out
#SBATCH --job-name=score_energetics

# Smoke test (uncomment to test):
# CUDA_VISIBLE_DEVICES=0 python scripts/score_external.py \
#     --input /scratch/khayes/energetics_bench/scoring/results/uq_input.jsonl \
#     --output data/external/energetics_scored.jsonl \
#     --smoke_test

# Full run:
CUDA_VISIBLE_DEVICES=0 python scripts/score_external.py \
    --input /scratch/khayes/energetics_bench/scoring/results/uq_input.jsonl \
    --output data/external/energetics_scored.jsonl
