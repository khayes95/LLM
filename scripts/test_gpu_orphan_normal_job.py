"""Normal job: loads vLLM, runs 5 batches, exits cleanly.

Verifies that signal handling + srun don't interfere with
a job that finishes before the time limit.
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
    try:
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGUSR1, shutdown)
atexit.register(lambda: shutdown() if llm is not None else None)

print("=== Normal Job: Loading vLLM model ===")
print(f"PID: {os.getpid()}, PGID: {os.getpgid(os.getpid())}")

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    disable_log_stats=True,
)
print("Model loaded. Running 5 batches then exiting cleanly...")

params = SamplingParams(max_tokens=100, temperature=0.7)
prompts = ["What is 2+2? Explain briefly."] * 4

for i in range(1, 6):
    outputs = llm.generate(prompts, params)
    print(f"  Batch {i}/5 done ({len(outputs)} outputs)")

print("\n=== All batches complete. Cleaning up normally. ===")
del llm
llm = None
print("Done.")
