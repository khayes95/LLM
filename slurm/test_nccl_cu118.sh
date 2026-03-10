#!/bin/bash
#SBATCH --job-name=nccl118
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_nccl_cu118_%j.log
#SBATCH --error=logs/test_nccl_cu118_%j.log

cd /scratch/khayes/LLM
source ~/.bashrc
conda activate qwen35_serve

python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, NCCL={torch.cuda.nccl.version()}')"

python scripts/test_nccl_v2.py 2>&1
echo "Exit code: $?"
