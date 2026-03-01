#!/bin/bash
#SBATCH --job-name=q35_test
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_qwen3_5_load_%j.log
#SBATCH --error=logs/test_qwen3_5_load_%j.log

echo "=========================================="
echo "Qwen3.5-397B-A17B-FP8 Load Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/khayes/LLM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG

source ~/.bashrc
conda activate uq_eval

nvidia-smi

echo ""
echo "Running load test..."
echo ""

python scripts/test_qwen3_5_load.py 2>&1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
