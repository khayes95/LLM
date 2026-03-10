#!/bin/bash
#SBATCH --job-name=score_v3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_v3_%j.out

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== Scoring ALL targets with v3 checkpoint ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py \
    --target all \
    --checkpoint uq_models/best_v3_qsplit \
    --output_dir data/use_cases/scored_v3_all \
    --prompt_variant combined

echo "=== Scoring complete ==="
echo "End: $(date)"
wc -l data/use_cases/scored_v3_all/*.jsonl 2>/dev/null
