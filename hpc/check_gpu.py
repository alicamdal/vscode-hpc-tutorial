"""Print what the HPC cluster gives you: machine, Python, PyTorch, and GPU.

Run it through Slurm (see hpc/check_gpu.sbatch), not on the login node:
    sbatch hpc/check_gpu.sbatch
"""

from __future__ import annotations

import os
import platform
import socket
import sys
import time

import torch


def line(label: str, value: object) -> None:
    print(f"  {label:<22} {value}")


def matmul_seconds(device: torch.device, n: int = 4096, repeats: int = 3) -> float:
    """Average time of one n x n matrix multiplication."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    in_job = "SLURM_JOB_ID" in os.environ

    print("\nWhere am I?")
    line("machine", socket.gethostname())
    line("Slurm job id", os.environ.get("SLURM_JOB_ID", "none (not inside a job)"))
    line("partition", os.environ.get("SLURM_JOB_PARTITION", "-"))
    line("CPU cores for me", os.environ.get("SLURM_CPUS_PER_TASK", "-"))

    print("\nSoftware")
    line("operating system", platform.platform())
    line("Python", sys.version.split()[0])
    line("PyTorch", torch.__version__)
    line("PyTorch built for CUDA", torch.version.cuda)

    print("\nGPU")
    if not torch.cuda.is_available():
        line("GPU available", "no")
        if not in_job:
            print("\n  This is expected on the login node: it has no GPU.")
            print("  Submit the job instead:  sbatch hpc/check_gpu.sbatch")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        line(f"GPU {i}", props.name)
        line("  memory (VRAM)", f"{total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")
        line("  compute capability", f"{props.major}.{props.minor}")
        line("  multiprocessors", props.multi_processor_count)

    print("\nSpeed test: multiply two 4096 x 4096 matrices")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    flops = 2 * 4096**3
    cpu = matmul_seconds(torch.device("cpu"), repeats=1)
    gpu = matmul_seconds(torch.device("cuda"))
    line("CPU", f"{cpu * 1000:8.1f} ms  ({flops / cpu / 1e12:6.2f} TFLOP/s)")
    line("GPU", f"{gpu * 1000:8.1f} ms  ({flops / gpu / 1e12:6.2f} TFLOP/s)")
    line("GPU is faster by", f"{cpu / gpu:.0f} x")
    print()


if __name__ == "__main__":
    main()
