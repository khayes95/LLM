#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu_test_%j.log

# Simple test to show GPU partition is blocked.
# This job will stay PENDING forever while job 8240 holds gpunode00,
# because OverSubscribe=NO on the GPU partition.
#
# To submit:   sbatch slurm/gpu_test.sh
# To check:    squeue -u khayes
# Expected:    Job stays in PD (Pending) state with reason "Resources"
#
# After fixing (either move job 8240 or set OverSubscribe=YES),
# resubmit and it should run in seconds.

echo "=== GPU Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate uq_eval

nvidia-smi
python /scratch/khayes/LLM/slurm/gpu_test.py

echo "=== Test complete ==="
