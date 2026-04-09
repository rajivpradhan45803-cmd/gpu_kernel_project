import torch
import time

# Detect device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def run_matmul(size, dtype):
    # Create random matrices and move to device
    A = torch.randn(size, size, dtype=dtype).to(device)
    B = torch.randn(size, size, dtype=dtype).to(device)

    # Measure execution time
    start = time.perf_counter()
    C = torch.matmul(A, B)
    end = time.perf_counter()

    return end - start


def run_experiments():
    configs = [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ]

    sizes = [256, 512]  # Multiple experiment sizes

    results = []

    for size in sizes:
        print(f"\nMatrix Size: {size}")

        for name, dtype in configs:
            try:
                time_taken = run_matmul(size, dtype)
                results.append((name, size, time_taken))
                print(f"{name} -> {time_taken:.4f} seconds")
            except Exception as e:
                print(f"{name} failed: {e}")

    return results