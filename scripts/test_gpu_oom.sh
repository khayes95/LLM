#!/bin/bash
# =============================================================================
# GPU OOM Collision Test
# =============================================================================
# Submit 2 jobs each requesting 1 GPU. Each allocates 60GB.
# If SLURM assigns both to the same GPU → 120GB > 80GB → OOM crash.
# =============================================================================

LOGDIR=/scratch/khayes/LLM/logs
mkdir -p $LOGDIR

# Job A: grab 60GB and hold it
JOBA=$(sbatch --parsable << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/khayes/LLM/logs/oom_jobA_%j.out
#SBATCH --job-name=oomA

eval "$(conda shell.bash hook)"
conda activate uq_eval

python -c "
import torch, time, os

gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
jid = os.environ.get('SLURM_JOB_ID', '?')
print(f'Job A | SLURM job {jid} | CUDA_VISIBLE_DEVICES={gpu}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
free = torch.cuda.mem_get_info(0)
print(f'VRAM: {free[0]/1e9:.1f} GB free / {free[1]/1e9:.1f} GB total')
print()

print('Allocating 60GB on assigned GPU...', flush=True)
try:
    t = torch.zeros(15_000_000_000, dtype=torch.float32, device='cuda:0')
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f'SUCCESS: {mem:.1f} GB allocated')
    print(f'Holding for 120 seconds...', flush=True)
    time.sleep(120)
    del t
    print('Released.')
except torch.cuda.OutOfMemoryError as e:
    print(f'OOM: {e}')
except Exception as e:
    print(f'ERROR: {e}')
"
SBATCH_EOF
)

echo "Submitted Job A: $JOBA"

# Submit Job B immediately — don't wait
JOBB=$(sbatch --parsable << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/khayes/LLM/logs/oom_jobB_%j.out
#SBATCH --job-name=oomB

eval "$(conda shell.bash hook)"
conda activate uq_eval

python -c "
import torch, time, os

gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
jid = os.environ.get('SLURM_JOB_ID', '?')
print(f'Job B | SLURM job {jid} | CUDA_VISIBLE_DEVICES={gpu}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
free = torch.cuda.mem_get_info(0)
print(f'VRAM: {free[0]/1e9:.1f} GB free / {free[1]/1e9:.1f} GB total')
print()

print('Allocating 60GB on assigned GPU...', flush=True)
try:
    t = torch.zeros(15_000_000_000, dtype=torch.float32, device='cuda:0')
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f'SUCCESS: {mem:.1f} GB allocated')
    print('No collision — SLURM gave us different GPUs.')
    time.sleep(30)
    del t
except torch.cuda.OutOfMemoryError as e:
    print(f'OOM! SLURM put both jobs on the same GPU.')
    print(f'Error: {e}')
except Exception as e:
    print(f'ERROR: {e}')
"
SBATCH_EOF
)

echo "Submitted Job B: $JOBB"
echo ""
echo "If SLURM assigns both to the same GPU:"
echo "  Job A grabs 60GB, Job B tries 60GB → OOM (only 80GB total)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep oom"
echo "  cat $LOGDIR/oom_jobA_${JOBA}.out"
echo "  cat $LOGDIR/oom_jobB_${JOBB}.out"
