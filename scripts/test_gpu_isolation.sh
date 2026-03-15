#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --output=/scratch/khayes/LLM/logs/test_gpu_isolation_%j.out
#SBATCH --job-name=gpu_isol

# =============================================================================
# GPU Isolation Test Script
# =============================================================================
# Purpose: Test whether SLURM enforces GPU isolation via cgroups.
#
# This job requests 1 GPU. It then:
#   1. Checks what SLURM allocated (CUDA_VISIBLE_DEVICES)
#   2. Confirms PyTorch sees only 1 GPU (respecting SLURM env var)
#   3. Overrides CUDA_VISIBLE_DEVICES to all 8 GPUs and checks if
#      PyTorch can actually access them
#
# If isolation is ENFORCED (cgroup device restrictions):
#   - Test 2 should show only 1 accessible GPU (others BLOCKED)
#
# If isolation is NOT enforced (env-var only):
#   - Test 2 will show all 8 GPUs accessible
#
# Known issue (2026-03-10): SLURM 21.08 + Ubuntu 22.04 (cgroup v2 only)
#   cannot enforce GPU device isolation. Requires SLURM >= 22.05 for
#   native cgroup v2 support, or hybrid cgroup v1/v2 kernel boot mode.
# =============================================================================

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== System Info ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "Python: $(which python)"

echo ""
echo "=== cgroup info ==="
cat /proc/self/cgroup
echo ""
ls /sys/fs/cgroup/devices/ 2>/dev/null && echo "cgroup v1 devices: AVAILABLE" || echo "cgroup v1 devices: NOT AVAILABLE (v2 only)"
cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null

echo ""
echo "=== Test 1: Respecting SLURM CUDA_VISIBLE_DEVICES ==="
python -c "
import torch
count = torch.cuda.device_count()
print(f'Visible GPUs: {count}')
for i in range(count):
    print(f'  cuda:{i} = {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=== Test 2: Override CUDA_VISIBLE_DEVICES to all 8 GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -c "
import torch
count = torch.cuda.device_count()
print(f'Visible GPUs after override: {count}')
accessible = 0
blocked = 0
for i in range(count):
    try:
        t = torch.zeros(1, device=f'cuda:{i}')
        print(f'  cuda:{i}: {torch.cuda.get_device_name(i)} -- ACCESSIBLE')
        accessible += 1
        del t; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  cuda:{i}: BLOCKED -- {e}')
        blocked += 1

print()
if accessible > 1:
    print(f'RESULT: GPU isolation NOT enforced ({accessible} GPUs accessible, expected 1)')
    print('Any job can override CUDA_VISIBLE_DEVICES and use all GPUs.')
elif accessible == 1:
    print(f'RESULT: GPU isolation IS enforced (only 1 GPU accessible, {blocked} blocked)')
else:
    print(f'RESULT: ERROR - no GPUs accessible at all')
"

echo ""
echo "=== nvidia-smi (driver level, shows all GPUs regardless of allocation) ==="
nvidia-smi -L

echo ""
echo "Done: $(date)"
