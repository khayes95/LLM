#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=perturb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/khayes/LLM/logs/perturbation_%j.out

cd /scratch/khayes/LLM

PYTHON="/scratch/khayes/.conda/envs/uq_eval/bin/python"

echo "=== Prompt Perturbation Generation ==="
echo "Start: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"

$PYTHON scripts/cpu_prompt_perturbation.py \
    --scored_dir data/use_cases/scored_test_only \
    --output data/use_cases/perturbations/all_perturbations.jsonl \
    "$@"

echo "End: $(date)"
