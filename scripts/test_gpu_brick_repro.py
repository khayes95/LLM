"""Aggressive repro: SIGKILL self mid-CUDA-kernel to try to fault the GPU.

This sends SIGKILL to its own process while a large inference batch is
actively running on the GPU. SIGKILL cannot be caught, so the CUDA driver
has no chance to run cleanup handlers. This is the most likely way to
leave a GPU in an unrecoverable error state.

WARNING: This is intentionally trying to break a GPU. Run on a dedicated
GPU during off-hours.
"""
import os
import signal
import threading
import time
from vllm import LLM, SamplingParams

KILL_DELAY = 30  # seconds after inference starts to send SIGKILL

def delayed_kill(seconds):
    """Kill ourselves mid-inference with SIGKILL (uncatchable)."""
    time.sleep(seconds)
    print(f"\n=== Sending SIGKILL to self (PID {os.getpid()}) mid-inference ===", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)

print("=== GPU Brick Repro: SIGKILL mid-CUDA-kernel ===")
print(f"PID: {os.getpid()}")
print(f"Will SIGKILL self {KILL_DELAY}s after inference starts")
print()

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    disable_log_stats=True,
)
print("Model loaded. Starting inference and arming kill timer...")

# Use large batches to maximize chance of being mid-kernel when killed
params = SamplingParams(max_tokens=512, temperature=0.7)
prompts = ["Explain the entire history of mathematics from ancient Babylon to modern day. " * 50] * 8

# Arm the kill timer
threading.Thread(target=delayed_kill, args=(KILL_DELAY,), daemon=True).start()

i = 0
while True:
    i += 1
    outputs = llm.generate(prompts, params)
    print(f"  Batch {i} done ({len(outputs)} outputs)", flush=True)
