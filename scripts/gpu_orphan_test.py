"""GPU orphan and brick test suite.

Usage:
    python gpu_orphan_test.py --mode <mode>

Modes:
    repro           Bare infinite inference loop. No signal handling.
                    Used by the repro and srun/signal shell tests.
    fixed           Same loop but with Python signal handlers that
                    cleanly shut down vLLM on SIGTERM/SIGUSR1.
    normal          Runs 5 batches then exits cleanly. Sanity check
                    that fixes don't break normal jobs.
    brick_kernel    SIGKILLs itself mid-CUDA-kernel after --kill_delay seconds.
    brick_teardown  On SIGTERM, starts CUDA cleanup then SIGKILLs mid-teardown.
                    Send "kill -TERM <pid>" manually once inference is running.
    brick_loop      Repeatedly spawns brick_kernel subprocesses and checks
                    GPU health after each. Stops if the GPU faults.
"""
import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time


# ---------------------------------------------------------------------------
# Shared: model loading and inference
# ---------------------------------------------------------------------------

def load_model():
    from vllm import LLM
    return LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        disable_log_stats=True,
    )


def run_inference_loop(llm, batches=None, large=False):
    """Run inference. If batches is None, run forever."""
    from vllm import SamplingParams
    if large:
        params = SamplingParams(max_tokens=512, temperature=0.7)
        prompts = ["Explain the entire history of mathematics from ancient "
                    "Babylon to modern day. " * 50] * 8
    else:
        params = SamplingParams(max_tokens=200, temperature=0.7)
        prompts = ["Explain quantum computing in detail. " * 20] * 4

    i = 0
    while batches is None or i < batches:
        i += 1
        outputs = llm.generate(prompts, params)
        print(f"  Batch {i} done ({len(outputs)} outputs)", flush=True)
        if batches is None:
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Mode: repro
# ---------------------------------------------------------------------------

def mode_repro():
    print("=== MODE: repro (no signal handling, infinite loop) ===")
    print(f"PID: {os.getpid()}")
    llm = load_model()
    print("Model loaded. Starting infinite inference loop...")
    run_inference_loop(llm)


# ---------------------------------------------------------------------------
# Mode: fixed (Python signal handlers)
# ---------------------------------------------------------------------------

def mode_fixed():
    print("=== MODE: fixed (Python signal handlers) ===")
    print(f"PID: {os.getpid()}, PGID: {os.getpgid(os.getpid())}")

    state = {"llm": None}

    def shutdown(signum=None, frame=None):
        sig_name = signal.Signals(signum).name if signum else "atexit"
        print(f"\n=== Received {sig_name}, shutting down vLLM ===", flush=True)
        if state["llm"] is not None:
            try:
                del state["llm"]
            except Exception as e:
                print(f"  Warning during cleanup: {e}")
            state["llm"] = None
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGUSR1, shutdown)
    atexit.register(lambda: shutdown() if state["llm"] is not None else None)

    state["llm"] = load_model()
    print("Model loaded. Starting infinite inference loop...")
    run_inference_loop(state["llm"])


# ---------------------------------------------------------------------------
# Mode: normal (runs 5 batches, exits cleanly)
# ---------------------------------------------------------------------------

def mode_normal():
    print("=== MODE: normal (5 batches, clean exit) ===")
    print(f"PID: {os.getpid()}, PGID: {os.getpgid(os.getpid())}")

    state = {"llm": None}

    def shutdown(signum=None, frame=None):
        sig_name = signal.Signals(signum).name if signum else "atexit"
        print(f"\n=== Received {sig_name}, shutting down vLLM ===", flush=True)
        if state["llm"] is not None:
            try:
                del state["llm"]
            except Exception:
                pass
            state["llm"] = None
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGUSR1, shutdown)

    state["llm"] = load_model()
    print("Model loaded. Running 5 batches then exiting cleanly...")
    run_inference_loop(state["llm"], batches=5)

    print("\n=== All batches complete. Cleaning up normally. ===")
    del state["llm"]
    state["llm"] = None
    print("Done.")


# ---------------------------------------------------------------------------
# Mode: brick_kernel (SIGKILL self mid-CUDA-kernel)
# ---------------------------------------------------------------------------

def mode_brick_kernel(kill_delay):
    print(f"=== MODE: brick_kernel (SIGKILL self after {kill_delay}s) ===")
    print(f"PID: {os.getpid()}")

    def delayed_kill(seconds):
        time.sleep(seconds)
        print(f"\n=== Sending SIGKILL to self (PID {os.getpid()}) mid-inference ===",
              flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    llm = load_model()
    print(f"Model loaded. Starting inference, will SIGKILL in {kill_delay}s...")

    threading.Thread(target=delayed_kill, args=(kill_delay,), daemon=True).start()
    run_inference_loop(llm, large=True)


# ---------------------------------------------------------------------------
# Mode: brick_teardown (SIGKILL during CUDA context teardown)
# ---------------------------------------------------------------------------

def mode_brick_teardown():
    print("=== MODE: brick_teardown (SIGKILL mid-CUDA-cleanup) ===")
    print(f"PID: {os.getpid()}")
    print(f"Send SIGTERM to trigger: kill -TERM {os.getpid()}")
    print()

    state = {"llm": None}

    def teardown_and_die(signum, frame):
        print(f"\n=== SIGTERM received, starting teardown then SIGKILL ===",
              flush=True)
        if state["llm"] is not None:
            try:
                del state["llm"]
            except Exception:
                pass
        print("=== Sending SIGKILL mid-teardown ===", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, teardown_and_die)

    state["llm"] = load_model()
    print("Model loaded. Running inference until SIGTERM is received...")
    run_inference_loop(state["llm"], large=True)


# ---------------------------------------------------------------------------
# Mode: brick_loop (repeated SIGKILL stress test)
# ---------------------------------------------------------------------------

def check_gpu_health(gpu_index):
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


def mode_brick_loop(gpu, iterations, kill_delay):
    print(f"=== MODE: brick_loop ({iterations} iterations on GPU {gpu}) ===")
    print(f"Kill delay: {kill_delay}s per iteration")
    print(f"Estimated time: {iterations * (kill_delay + 40) / 60:.0f} minutes")
    print()

    healthy, status = check_gpu_health(gpu)
    if not healthy:
        print(f"GPU {gpu} already unhealthy: {status}")
        sys.exit(1)
    print(f"GPU {gpu} initial health: {status}")

    for i in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {i}/{iterations}: spawning inference subprocess")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        proc = subprocess.Popen(
            [sys.executable, __file__, "--mode", "brick_kernel",
             "--kill_delay", str(kill_delay)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print(f"  Subprocess PID: {proc.pid}")

        try:
            proc.wait(timeout=kill_delay + 120)
        except subprocess.TimeoutExpired:
            print(f"  Subprocess didn't die in time, sending SIGKILL")
            proc.kill()
            proc.wait()

        print(f"  Exit code: {proc.returncode}")
        time.sleep(5)

        healthy, status = check_gpu_health(gpu)
        if healthy:
            print(f"  GPU {gpu} healthy: {status}")
        else:
            print(f"\n*** GPU {gpu} FAULTED ON ITERATION {i}/{iterations}: {status} ***")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"All {iterations} iterations completed. GPU {gpu} survived.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU orphan and brick test suite")
    parser.add_argument("--mode", required=True,
                        choices=["repro", "fixed", "normal",
                                 "brick_kernel", "brick_teardown", "brick_loop"],
                        help="Test mode to run")
    parser.add_argument("--kill_delay", type=int, default=30,
                        help="Seconds before SIGKILL in brick modes (default: 30)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Physical GPU index for brick_loop (default: 0)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations for brick_loop (default: 50)")
    args = parser.parse_args()

    if args.mode == "repro":
        mode_repro()
    elif args.mode == "fixed":
        mode_fixed()
    elif args.mode == "normal":
        mode_normal()
    elif args.mode == "brick_kernel":
        mode_brick_kernel(args.kill_delay)
    elif args.mode == "brick_teardown":
        mode_brick_teardown()
    elif args.mode == "brick_loop":
        mode_brick_loop(args.gpu, args.iterations, args.kill_delay)


if __name__ == "__main__":
    main()
