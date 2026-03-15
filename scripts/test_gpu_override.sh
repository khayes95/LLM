#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/khayes/LLM/logs/gpu_override_%j.out
#SBATCH --job-name=gpu_ovrd

# =============================================================================
# GPU Override Brute Force Test
# =============================================================================
# Request 1 GPU from SLURM, then try to override CUDA_VISIBLE_DEVICES
# and allocate 60GB on EVERY GPU (0-7). If cgroup isolation works,
# only the allocated GPU should be accessible.
# =============================================================================

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== SLURM Allocation ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

echo "=== Attempting CUDA_VISIBLE_DEVICES override to 0,1,2,3,4,5,6,7 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -c "
import torch
import os

print(f'CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"not set\")}')
count = torch.cuda.device_count()
print(f'PyTorch sees {count} GPU(s)')
print()

for i in range(8):
    try:
        # Try to allocate 60GB on each GPU
        print(f'GPU {i}: Attempting 60GB allocation...', end=' ', flush=True)
        t = torch.zeros(15_000_000_000, dtype=torch.float32, device=f'cuda:{i}')
        mem = torch.cuda.memory_allocated(i) / 1e9
        name = torch.cuda.get_device_name(i)
        print(f'SUCCESS — {mem:.1f} GB on {name}')
        del t
        torch.cuda.empty_cache()
    except RuntimeError as e:
        err = str(e)
        if 'invalid device ordinal' in err:
            print(f'BLOCKED (invalid device ordinal — cgroup hiding it)')
        elif 'out of memory' in err:
            print(f'OOM (GPU exists but full)')
        else:
            print(f'ERROR: {err[:100]}')
    except Exception as e:
        print(f'ERROR: {e}')

print()
accessible = torch.cuda.device_count()
if accessible == 1:
    print('VERDICT: cgroup isolation ENFORCED — only 1 GPU visible despite override')
elif accessible == 8:
    print('VERDICT: cgroup isolation BROKEN — all 8 GPUs accessible, system can be bricked')
else:
    print(f'VERDICT: partial isolation — {accessible} GPUs visible')
"

echo ""
echo "=== nvidia-smi from inside the job ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Done: $(date)"
