#!/bin/bash
# =============================================================================
# GPU Collision Test
# =============================================================================
# Demonstrates that SLURM GPU isolation is NOT enforced.
#
# Submits 2 jobs, each requesting 1 GPU via --gres.
# Both jobs override CUDA_VISIBLE_DEVICES and allocate a large tensor
# on GPU 0, proving they can collide on the same physical GPU.
#
# Expected behavior if isolation WORKS:
#   - Each job can only see its allocated GPU, collision impossible
#
# Expected behavior if isolation is BROKEN (current state):
#   - Both jobs grab GPU 0, filling VRAM
#   - Second job may OOM or both degrade performance
#
# Usage: bash scripts/test_gpu_collision.sh
# =============================================================================

LOGDIR=/scratch/khayes/LLM/logs
mkdir -p $LOGDIR

# Job A: allocate ~40GB on GPU 0 and hold it
JOBA=$(sbatch --parsable << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/khayes/LLM/logs/collision_jobA_%j.out
#SBATCH --job-name=collideA

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== Job A (allocated GPU: $CUDA_VISIBLE_DEVICES) ==="
echo "Overriding CUDA_VISIBLE_DEVICES to force GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -c "
import torch, time, os

print(f'Job A (SLURM job {os.environ.get(\"SLURM_JOB_ID\", \"?\")})')
print(f'Allocating 40GB tensor on GPU 0...')

try:
    # Allocate ~40GB on GPU 0
    t = torch.zeros(10_000_000_000, dtype=torch.float32, device='cuda:0')
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f'SUCCESS: Allocated {mem:.1f} GB on GPU 0')
    print(f'Holding for 120 seconds so Job B can try the same GPU...')
    time.sleep(120)
    del t
    print('Job A released GPU memory.')
except torch.cuda.OutOfMemoryError as e:
    print(f'OOM on GPU 0: {e}')
except Exception as e:
    print(f'ERROR: {e}')
"
SBATCH_EOF
)

echo "Submitted Job A: $JOBA"

# Wait a few seconds for Job A to start allocating
sleep 10

# Job B: also try to allocate ~40GB on GPU 0
JOBB=$(sbatch --parsable << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/khayes/LLM/logs/collision_jobB_%j.out
#SBATCH --job-name=collideB

eval "$(conda shell.bash hook)"
conda activate uq_eval

echo "=== Job B (allocated GPU: $CUDA_VISIBLE_DEVICES) ==="
echo "Overriding CUDA_VISIBLE_DEVICES to force GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -c "
import torch, time, os

print(f'Job B (SLURM job {os.environ.get(\"SLURM_JOB_ID\", \"?\")})')
print(f'Trying to allocate 40GB tensor on GPU 0...')

try:
    t = torch.zeros(10_000_000_000, dtype=torch.float32, device='cuda:0')
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f'SUCCESS: Allocated {mem:.1f} GB on GPU 0')
    print(f'Both jobs sharing GPU 0 — isolation is NOT enforced!')
    time.sleep(30)
    del t
except torch.cuda.OutOfMemoryError as e:
    print(f'OOM: GPU 0 already full from Job A — expected if isolation broken')
    print(f'(Two jobs fighting over same GPU = bad)')
except Exception as e:
    print(f'ERROR: {e}')
"
SBATCH_EOF
)

echo "Submitted Job B: $JOBB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER | grep collide"
echo "  cat $LOGDIR/collision_jobA_${JOBA}.out"
echo "  cat $LOGDIR/collision_jobB_${JOBB}.out"
echo ""
echo "Expected result (isolation broken):"
echo "  Job A: allocates 40GB on GPU 0"
echo "  Job B: OOM on GPU 0 because Job A already filled it"
echo "  Both jobs were allocated DIFFERENT GPUs by SLURM, but both accessed GPU 0"
echo ""
echo "Expected result (isolation working):"
echo "  Job B cannot access GPU 0 at all — cgroup blocks it"
