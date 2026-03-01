#!/bin/bash
#SBATCH --job-name=q35_load
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_qwen3_5_load_custom_nccl_%j.log
#SBATCH --error=logs/test_qwen3_5_load_custom_nccl_%j.log

echo "=========================================="
echo "Qwen3.5-397B-A17B-FP8 Load Test"
echo "  vLLM nightly + custom NCCL 2.27.5 (cu11.8)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use our custom-built NCCL 2.27.5 (compiled with CUDA 11.8 / GCC 11)
# This replaces the cu128 NCCL that segfaults on the CUDA 12.0 driver
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

source ~/.bashrc
conda activate uq_eval

echo "--- Environment ---"
python -c "
import torch
print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, NCCL={torch.cuda.nccl.version()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

echo ""
echo "--- Loading Qwen3.5-397B-A17B-FP8 with vLLM TP=8 ---"
python scripts/test_qwen3_5_load.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
