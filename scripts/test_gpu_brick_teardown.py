"""Aggressive repro: SIGKILL during CUDA context teardown.

Catches SIGTERM, starts cleaning up vLLM (which triggers CUDA context
teardown), then immediately SIGKILLs itself mid-cleanup. This targets
the specific window where the driver is deallocating GPU resources.

WARNING: This is intentionally trying to break a GPU. Run on a dedicated
GPU during off-hours.
"""
import os
import signal
import sys
import time
from vllm import LLM, SamplingParams

llm = None

def teardown_and_die(signum, frame):
    """Start CUDA cleanup, then SIGKILL mid-teardown."""
    global llm
    print(f"\n=== SIGTERM received, starting teardown then SIGKILL mid-cleanup ===", flush=True)
    if llm is not None:
        # This triggers CUDA context cleanup internally
        try:
            del llm
        except:
            pass
    # Kill mid-cleanup before driver finishes deallocating
    print("=== Sending SIGKILL mid-teardown ===", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)

signal.signal(signal.SIGTERM, teardown_and_die)

print("=== GPU Brick Repro: SIGKILL mid-teardown ===")
print(f"PID: {os.getpid()}")
print("Send SIGTERM to trigger: kill -TERM {pid}")
print()

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    disable_log_stats=True,
)
print("Model loaded. Running inference until SIGTERM is received...")

params = SamplingParams(max_tokens=512, temperature=0.7)
prompts = ["Explain the entire history of mathematics from ancient Babylon to modern day. " * 50] * 8

i = 0
while True:
    i += 1
    outputs = llm.generate(prompts, params)
    print(f"  Batch {i} done ({len(outputs)} outputs)", flush=True)
