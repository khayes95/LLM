#!/bin/bash
#SBATCH --job-name=q35_serve
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/serve_qwen3_5_397b_%j.log
#SBATCH --error=logs/serve_qwen3_5_397b_%j.log

# Serve Qwen3.5-397B-A17B-FP8 with vLLM on 8x A100-80GB
# Smoke test: python scripts/smoke_test_qwen3_5.py

echo "=========================================="
echo "Qwen3.5-397B-A17B-FP8 vLLM Server"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Custom NCCL 2.27.5 built with CUDA 11.8 / GCC 11
# Required because system CUDA driver is 525.105.17 (CUDA 12.0)
# and the bundled NCCL cu128 segfaults on this driver.
export LD_PRELOAD=/scratch/khayes/nccl-src/build/lib/libnccl.so.2.27.5

source ~/.bashrc
conda activate uq_eval

nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader

echo ""
echo "Starting vLLM server on port 8100..."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model /scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8 \
    --port 8100 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --dtype auto

echo ""
echo "=========================================="
echo "Server stopped: $(date)"
echo "=========================================="
