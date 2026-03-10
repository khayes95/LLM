#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=scram_v3
#SBATCH --output=logs/scramble_v3_%j.out

source activate uq_eval

echo "=== Scramble + trunc_r3000 Ablation (v3 checkpoint, remaining conditions) ==="
echo "Start: $(date)"

# Run only the missing conditions: trunc_r3000 and both scrambles
CUDA_VISIBLE_DEVICES=0 python scripts/ablation_truncation_scramble.py \
    --checkpoint uq_models/best_v3_qsplit \
    --output_dir data/ablations/truncation_scramble_v3 \
    --mode scramble

echo "Done: $(date)"
