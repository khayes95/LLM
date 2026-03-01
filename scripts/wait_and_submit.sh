#!/bin/bash
# Wait for GPUs 6-7 to be free, then submit the overnight pipeline.
# Checks every 5 minutes.
# Usage: bash scripts/wait_and_submit.sh &

cd /scratch/khayes/LLM

echo "Waiting for GPUs 6+7 to free up..."
echo "Start: $(date)"

while true; do
    # Check if GPUs 6 and 7 have < 1GB used memory
    gpu6_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    gpu7_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)

    if [ -z "$gpu6_mem" ] || [ -z "$gpu7_mem" ]; then
        echo "$(date): Cannot query GPUs, retrying..."
        sleep 300
        continue
    fi

    # Convert to integers
    gpu6_mem=${gpu6_mem// /}
    gpu7_mem=${gpu7_mem// /}

    if [ "$gpu6_mem" -lt 1000 ] && [ "$gpu7_mem" -lt 1000 ]; then
        echo "$(date): GPUs free! GPU6=${gpu6_mem}MiB GPU7=${gpu7_mem}MiB"
        echo "Submitting overnight pipeline..."

        # Check no other khayes GPU jobs running
        my_gpu_jobs=$(squeue -u $USER -p GPU -h | wc -l)
        if [ "$my_gpu_jobs" -gt 0 ]; then
            echo "WARNING: Already have $my_gpu_jobs GPU jobs. Skipping submission."
            sleep 300
            continue
        fi

        JOB_ID=$(sbatch --parsable slurm/overnight_retrain_pipeline.sh)
        echo "Submitted job $JOB_ID at $(date)"
        echo "Monitor with: tail -f logs/overnight_pipeline_${JOB_ID}.log"
        exit 0
    else
        echo "$(date): GPU6=${gpu6_mem}MiB GPU7=${gpu7_mem}MiB — waiting..."
        sleep 300
    fi
done
