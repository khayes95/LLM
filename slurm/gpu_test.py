"""Simple GPU test - confirms GPUs are accessible and working."""
import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Quick matrix multiply to prove GPU works
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device="cuda:0")
    y = torch.randn(1000, 1000, device="cuda:0")
    start = time.time()
    z = x @ y
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"\nGPU matrix multiply (1000x1000): {elapsed*1000:.1f}ms")
    print("GPU is working!")
else:
    print("\nERROR: No GPUs accessible!")
