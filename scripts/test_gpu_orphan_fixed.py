"""Fixed version: proper signal handling to shut down vLLM cleanly.

Same workload as the repro, but with signal handlers that ensure
vLLM's subprocess is cleaned up on SIGTERM/SIGUSR1.
"""
import atexit
import os
import signal
import sys
import time

llm = None

def shutdown(signum=None, frame=None):
    global llm
    sig_name = signal.Signals(signum).name if signum else "atexit"
    print(f"\n=== Received {sig_name}, shutting down vLLM ===")
    if llm is not None:
        try:
            del llm
        except Exception as e:
            print(f"  Warning during cleanup: {e}")
        llm = None
    # Kill our entire process group to catch any stragglers
    try:
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGUSR1, shutdown)
atexit.register(lambda: shutdown() if llm is not None else None)

print("=== GPU Orphan FIXED: Loading vLLM model ===")
print(f"PID: {os.getpid()}, PGID: {os.getpgid(os.getpid())}")

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    disable_log_stats=True,
)
print("Model loaded. Starting infinite inference loop...")

params = SamplingParams(max_tokens=200, temperature=0.7)
prompts = ["Explain quantum computing in detail. " * 20] * 4

i = 0
while True:
    i += 1
    outputs = llm.generate(prompts, params)
    print(f"  Batch {i} done ({len(outputs)} outputs)")
    time.sleep(0.5)
