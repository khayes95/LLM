#!/bin/bash
#SBATCH --job-name=q35_cu118
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_qwen3_5_cu118_%j.log
#SBATCH --error=logs/test_qwen3_5_cu118_%j.log

echo "=========================================="
echo "Qwen3.5-397B-A17B-FP8 Load Test (cu118)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source ~/.bashrc
conda activate qwen35_serve

python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, NCCL={torch.cuda.nccl.version()}')"

python scripts/test_qwen3_5_load.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
