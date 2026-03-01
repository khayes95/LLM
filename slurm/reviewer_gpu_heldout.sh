#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --job-name=heldout_eval
#SBATCH --output=logs/heldout_eval_%j.out

# Held-out benchmark evaluation: 5-fold cross-validation over benchmarks
# Estimated: 8-12 hours (5 folds × ~2 hours each)
#
# Smoke test:
#   CUDA_VISIBLE_DEVICES=0 python scripts/held_out_benchmark_eval.py --smoke_test

source ~/.bashrc
conda activate uq_eval
cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,4,5,7

echo "============================================"
echo "HELD-OUT BENCHMARK EVAL — $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python scripts/held_out_benchmark_eval.py \
    --output_dir data/ablations/held_out_benchmark \
    --n_folds 5 \
    --epochs 2 \
    --lora_r 32

echo ""
echo "============================================"
echo "HELD-OUT EVAL COMPLETE — $(date)"
echo "============================================"
