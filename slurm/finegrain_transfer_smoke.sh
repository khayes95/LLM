#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=fg_xfer_smk
#SBATCH --output=logs/fg_transfer_smoke_%j.out

cd /scratch/khayes/LLM
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

PYTHON=/scratch/khayes/.conda/envs/uq_eval/bin/python

echo "=== FineGRAIN Transfer Eval SMOKE TEST at $(date) ==="

# Only evaluate one fold with 5 samples per model
$PYTHON scripts/finegrain_transfer_eval.py \
    --checkpoint data/finegrain_uq/exp1_human_cv/fold_flux/checkpoint-best \
    --smoke_test \
    --device cuda:0 \
    --output_dir data/finegrain_uq/transfer_eval_smoke

echo "=== Finished at $(date) ==="
