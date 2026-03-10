#!/bin/bash
#SBATCH --job-name=nccl_cust
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_nccl_custom_%j.log
#SBATCH --error=logs/test_nccl_custom_%j.log

echo "=========================================="
echo "NCCL Custom Build Test (uq_eval + LD_PRELOAD)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# LD_PRELOAD our custom-built NCCL 2.27.5 (built with CUDA 11.8 / GCC 11)
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

source ~/.bashrc
conda activate uq_eval

echo "--- Environment ---"
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, NCCL={torch.cuda.nccl.version()}')"
echo "LD_PRELOAD=$LD_PRELOAD"

echo ""
echo "--- NCCL Test ---"
python scripts/test_nccl_v2.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
