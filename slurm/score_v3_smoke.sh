#!/bin/bash
#SBATCH --job-name=v3_smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/v3_smoke_%j.out

cd /scratch/khayes/LLM
eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== V3 Smoke Test: score 5 samples per benchmark ==="
CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py \
    --target gpt5mini \
    --checkpoint uq_models/best_v3_qsplit \
    --output_dir data/use_cases/scored_v3_smoke \
    --prompt_variant combined \
    --smoke_test

echo "=== Smoke test complete ==="
echo "Sample output:"
tail -1 data/use_cases/scored_v3_smoke/gpt5mini_scored.jsonl 2>/dev/null | python3 -m json.tool
wc -l data/use_cases/scored_v3_smoke/*.jsonl 2>/dev/null
