"""Stress test: repeatedly load, infer, SIGKILL mid-kernel in a loop.

Runs N iterations of: load model, start inference, SIGKILL mid-kernel.
Each iteration spawns a subprocess so the parent can monitor and continue.
If any iteration faults the GPU, it logs which iteration and stops.

WARNING: This is intentionally trying to break a GPU. Run on a dedicated
GPU during off-hours. Expect this to take 30-60 minutes for 50 iterations.
"""
import os
import subprocess
import sys
import time
import argparse

def check_gpu_health(gpu_index):
    """Check if a GPU is responsive by querying nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
             "--format=csv,noheader", f"--id={gpu_index}"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        if "Unknown Error" in output or "ERR" in output or result.returncode != 0:
            return False, output
        return True, output
    except Exception as e:
        return False, str(e)

def run_single_kill_iteration(gpu_index, iteration, kill_delay=30):
    """Spawn a subprocess that loads vLLM and SIGKILLs itself mid-inference."""
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}: spawning inference subprocess (kill after {kill_delay}s)")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "test_gpu_brick_repro.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print(f"  Subprocess PID: {proc.pid}")

    # Wait for it to self-SIGKILL (the script does this internally after KILL_DELAY seconds)
    # Give it extra time for model loading
    try:
        proc.wait(timeout=kill_delay + 120)
    except subprocess.TimeoutExpired:
        print(f"  Subprocess didn't die in time, sending SIGKILL")
        proc.kill()
        proc.wait()

    print(f"  Subprocess exited with code: {proc.returncode}")

    # Brief pause for GPU state to settle
    time.sleep(5)

    # Check GPU health
    healthy, status = check_gpu_health(gpu_index)
    if healthy:
        print(f"  GPU {gpu_index} healthy: {status}")
    else:
        print(f"  *** GPU {gpu_index} FAULTED: {status} ***")

    return healthy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to test")
    parser.add_argument("--iterations", type=int, default=50, help="Number of kill iterations")
    parser.add_argument("--kill_delay", type=int, default=30, help="Seconds of inference before SIGKILL")
    args = parser.parse_args()

    print(f"=== GPU Brick Stress Test ===")
    print(f"GPU: {args.gpu}")
    print(f"Iterations: {args.iterations}")
    print(f"Kill delay: {args.kill_delay}s per iteration")
    print(f"Estimated time: {args.iterations * (args.kill_delay + 40) / 60:.0f} minutes")
    print()

    # Initial health check
    healthy, status = check_gpu_health(args.gpu)
    if not healthy:
        print(f"GPU {args.gpu} already unhealthy: {status}")
        print("Aborting.")
        sys.exit(1)
    print(f"GPU {args.gpu} initial health: {status}")

    for i in range(1, args.iterations + 1):
        healthy = run_single_kill_iteration(args.gpu, i, args.kill_delay)
        if not healthy:
            print(f"\n*** GPU FAULTED ON ITERATION {i}/{args.iterations} ***")
            print("Stopping stress test.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"All {args.iterations} iterations completed. GPU {args.gpu} survived.")
    print("Could not reproduce the GPU fault.")

if __name__ == "__main__":
    main()
