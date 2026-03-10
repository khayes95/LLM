#!/bin/bash
#SBATCH --job-name=filter_v3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/filter_v3_%j.out

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== Filter v3 scored data to test-only + re-run all use cases ==="
echo "Start: $(date)"

python scripts/filter_test_only.py \
    --split_info uq_models/best_v3_qsplit/split_info.json \
    --scored_dir data/use_cases/scored_v3_all \
    --output_scored data/use_cases/scored_test_only_v3 \
    --output_dir data/use_cases/results_test_only_v3 \
    --fig_dir figures/use_cases_v3 \
    --old_results_dir data/use_cases/results_test_only_v2

echo "=== Filter + use cases complete ==="
echo "End: $(date)"
