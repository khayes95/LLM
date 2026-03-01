#!/usr/bin/env python3
"""
Test 235B using torchrun for distributed initialization.
This runs a single process with vLLM handling the internal parallelism.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from vllm import LLM, SamplingParams


def main():
    print("=" * 60)
    print("Qwen3-VL-235B Test with vLLM")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    model_path = "Qwen/Qwen3-VL-235B-A22B-Thinking"
    tp_size = torch.cuda.device_count()

    print(f"\nLoading {model_path} with TP={tp_size}...")

    try:
        llm = LLM(
            model=model_path,
            dtype='bfloat16',
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.80,
            max_model_len=4096,
            trust_remote_code=True,
            # MoE optimization flags from official docs
            mm_encoder_tp_mode="data",
            enable_expert_parallel=True,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 0, "image": 1},
        )

        print("\n" + "=" * 60)
        print("MODEL LOADED SUCCESSFULLY!")
        print("=" * 60)

        # Test inference
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=64,
        )

        prompts = ["What is 2 + 2?", "What is the capital of France?"]
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Response: {output.outputs[0].text}")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
