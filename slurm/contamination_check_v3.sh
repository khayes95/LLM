#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=contam_v3
#SBATCH --output=logs/contamination_v3_%j.out

source activate uq_eval

echo "=== Near-Duplicate Contamination Check (v3 split) ==="
echo "Start: $(date)"

python scripts/cpu_contamination_check.py \
    --train_file data/finetune/train_v3.jsonl \
    --test_file data/finetune/test_v3.jsonl \
    --scored_dir data/use_cases/scored_test_only_v3 \
    --output data/use_cases/results_test_only_v3/contamination_report_v3.json

echo "Done: $(date)"
