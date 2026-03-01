"""Test Qwen3.5-397B-A17B-FP8 with Ray distributed backend.
Also tries to capture detailed worker errors.
"""

import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
# Force Ray executor
os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"

MODEL = "/scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8"

def main():
    print(f"Loading model: {MODEL}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Trying Ray distributed backend")

    try:
        from vllm import LLM, SamplingParams
        print(f"vLLM version: {__import__('vllm').__version__}")

        llm = LLM(
            model=MODEL,
            tensor_parallel_size=8,
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            dtype="auto",
            distributed_executor_backend="ray",
        )
        print("Model loaded successfully!")

        params = SamplingParams(temperature=0.7, max_tokens=128)
        outputs = llm.generate(["What is 2 + 2?"], params)

        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")

        print("\nSMOKE TEST PASSED")

    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
