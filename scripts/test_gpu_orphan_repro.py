"""Repro: vLLM subprocess orphaned by SLURM timeout.

Loads a model via vLLM, starts a long-running inference loop,
and expects to be killed by SLURM's time limit. The vLLM EngineCore
subprocess should become orphaned and hold GPU memory.
"""
import time
import os
from vllm import LLM, SamplingParams

print("=== GPU Orphan Repro: Loading vLLM model ===")
print(f"PID: {os.getpid()}")

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
