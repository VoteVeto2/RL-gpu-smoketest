"""Stage 0: report what Slurm gave us and run a quick matmul throughput test."""

import argparse
import json
import os
import time

import torch


def matmul_tflops(dtype, n=8192, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return round(2 * n**3 * iters / (time.time() - t) / 1e12, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available(), "No GPU visible -- did you request one with --gpus-per-node?"
    props = torch.cuda.get_device_properties(0)
    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_gpus": torch.cuda.device_count(),
        "gpu": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "memory_gb": round(props.total_memory / 1e9, 1),
        "native_bf16": props.major >= 8,
        "tflops_fp32": matmul_tflops(torch.float32),
        "tflops_fp16": matmul_tflops(torch.float16),
    }
    if info["native_bf16"]:
        info["tflops_bf16"] = matmul_tflops(torch.bfloat16)
    print(json.dumps(info, indent=2), flush=True)
    with open(os.path.join(args.output_dir, "gpu_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
