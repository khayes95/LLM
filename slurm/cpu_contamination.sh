#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=contam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/contamination_%j.out

cd /scratch/khayes/LLM

PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"

echo "=== Contamination & Deduplication Check ==="
echo "Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"

$PYTHON scripts/cpu_contamination_check.py \
    --train_file data/finetune/train_v2.jsonl \
    --test_file data/finetune/test_v2.jsonl \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/results_test_only/contamination_report.json \
    "$@"

echo "End: $(date)"
