#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=fg_ft_smk
#SBATCH --output=logs/fg_ft_smoke_%j.out

set -euo pipefail
cd /scratch/khayes/LLM
export PATH=/scratch/khayes/anaconda3/bin:$PATH
mkdir -p logs data/finegrain_uq

echo "=========================================="
echo "FineGRAIN UQ Fine-Tune: Smoke Test"
echo "Started: $(date)"
echo "=========================================="

# Test the human_cv experiment mode with 10 samples
echo ""
echo ">>> Smoke: human_cv mode"
# Find a GPU with enough free memory (need ~18 GiB for model loading)
FREE_GPU=$(python3 -c "
import subprocess, re
result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
for line in result.stdout.strip().split('\n'):
    idx, free = line.strip().split(',')
    if int(free.strip()) > 18000:
        print(idx.strip())
        break
else:
    print('4')  # fallback to GPU 4
" 2>/dev/null || echo "4")
echo "Selected GPU: $FREE_GPU"

CUDA_VISIBLE_DEVICES=$FREE_GPU python scripts/finegrain_finetune.py \
    --experiment human_cv \
    --output_dir data/finegrain_uq/smoke_ft \
    --held_out_model sd3_xl \
    --smoke_test
echo ">>> PASS"

echo ""
echo "=========================================="
echo "Smoke test passed: $(date)"
echo "=========================================="
