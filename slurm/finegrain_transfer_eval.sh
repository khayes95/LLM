#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=fg_transfer
#SBATCH --output=logs/fg_transfer_eval_%j.out

cd /scratch/khayes/LLM
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

PYTHON=/scratch/khayes/.conda/envs/uq_eval/bin/python

echo "=== FineGRAIN Transfer Eval at $(date) ==="

# 200 samples per model × 12 models = ~2400 samples
# Compare zero-shot vs best fold (flux, which had highest CV AUROC 0.973)
$PYTHON scripts/finegrain_transfer_eval.py \
    --checkpoint data/finegrain_uq/exp1_human_cv/fold_flux/checkpoint-best \
    --max_per_model 200 \
    --device cuda:0 \
    --output_dir data/finegrain_uq/transfer_eval

echo "=== Finished at $(date) ==="
