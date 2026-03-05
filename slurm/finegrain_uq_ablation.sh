#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=fg_uq_abl
#SBATCH --output=logs/fg_uq_ablation_%j.out

# Smoke test (uncomment to run quick test):
# CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py --smoke_test --prompt_variant baseline --output_dir data/finegrain_uq/smoke_baseline

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=========================================="
echo "FineGRAIN UQ: Prompt Variant Ablation"
echo "Started: $(date)"
echo "=========================================="

# Run 3 variants (combined already exists)
for variant in baseline finegrain_direct finegrain_qa; do
    echo ""
    echo ">>> Running variant: $variant ($(date))"
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
        --prompt_variant $variant \
        --output_dir data/finegrain_uq/ablation_${variant}
    echo ">>> Done with $variant ($(date))"
done

echo ""
echo "=========================================="
echo "All ablations complete: $(date)"
echo "=========================================="
