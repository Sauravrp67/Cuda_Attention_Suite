"""
benchmarks/utils/bench_core.py

Shared benchmark primitives used by run_bench.py and run_compare.py.

Adding a new kernel
-------------------
1. Add an entry to KERNEL_REGISTRY:
       "my_kernel": KernelMeta(label="My Kernel (description)")
2. Add a branch to _make_fn().
That's it — both scripts pick it up automatically.

Adding a new config axis (e.g. causal masking)
-----------------------------------------------
Pass `causal: bool` through benchmark_shape() → _make_fn().
Each kernel branch decides how to honour it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.hardware_constants import RTX_4050_LAPTOP

HW = RTX_4050_LAPTOP

# Kernel registry
@dataclass(frozen=True)
class KernelMeta:
    label: str                       
    # Future fields: supports_causal, supports_fp8, min_sm, …


KERNEL_REGISTRY: dict[str, KernelMeta] = {
    "naive_v1": KernelMeta(label="Naive v1 (CUDA)"),
    "sdpa":     KernelMeta(label="PyTorch SDPA"),
    "fused_v2": KernelMeta(label="Fused v2 (Shared Mem)"),
    "flash_v3": KernelMeta(label="Flash v3 (Tiled)"),
    "paged_v4": KernelMeta(label="Paged v4 (KV Cache)"),
}

KERNEL_STYLE: dict[str, dict] = {
    "naive_v1": dict(color="#f78166", marker="o", lw=2.0, ms=7,  ls="-"),
    "sdpa":     dict(color="#58a6ff", marker="*", lw=2.5, ms=10, ls="--"),
    "fused_v2": dict(color="#7ee787", marker="s", lw=2.0, ms=7,  ls="-"),
    "flash_v3": dict(color="#ffa657", marker="^", lw=2.0, ms=7,  ls="-"),
    "paged_v4": dict(color="#d2a8ff", marker="D", lw=2.0, ms=7,  ls="-"),
}
BAR_COLORS: dict[str, str] = {
    "naive_v1": "#4878CF",   
    "sdpa":     "#D65F5F",   
    "fused_v2": "#6ACC65",   
    "flash_v3": "#B47CC7",   
    "paged_v4": "#C4AD66",   
}
DEFAULT_STYLE = dict(color="#8b949e", marker="x", lw=1.5, ms=6, ls="-")
DEFAULT_BAR_COLOR = "#999999"


def kernel_style(k: str) -> dict:
    s = KERNEL_STYLE.get(k, DEFAULT_STYLE).copy()
    s.setdefault("ls", "-")
    return s


def kernel_bar_color(k: str) -> str:
    return BAR_COLORS.get(k, DEFAULT_BAR_COLOR)


def kernel_label(k: str) -> str:
    return KERNEL_REGISTRY[k].label if k in KERNEL_REGISTRY else k

# Analytical accounting
def attention_flops(B: int, H: int, N: int, D: int) -> int:
    """QK^T + AV = 4·B·H·N²·D FLOPs."""
    return 4 * B * H * N * N * D


def attention_bytes(B: int, H: int, N: int, D: int, itemsize: int) -> int:
    """Algorithmic lower bound: read Q+K+V, write O — each touched once."""
    return 4 * B * H * N * D * itemsize

def _make_fn(
    kernel_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,           # stub: passed through, kernels may ignore
) -> Callable[[], torch.Tensor]:
    if kernel_name == "naive_v1":
        from PyCuAttention.kernels.loader import naive_attn
        return lambda: naive_attn(q, k, v)

    elif kernel_name == "sdpa":
        from torch.nn.functional import scaled_dot_product_attention
        scale = q.shape[-1] ** -0.5
        return lambda: scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=causal
        )

    elif kernel_name == "fused_v2":
        from PyCuAttention.kernels.loader import fused_attn
        scale = q.shape[-1] ** -0.5
        return lambda: fused_attn(q, k, v, scale)

    elif kernel_name == "flash_v3":
        # TODO: import flash_attn once implemented
        raise NotImplementedError("flash_v3 not yet implemented")

    elif kernel_name == "paged_v4":
        raise NotImplementedError("paged_v4 not yet implemented")

    else:
        raise ValueError(f"Unknown kernel: {kernel_name!r}")

# Public API: benchmark_shape
def benchmark_shape(
    kernel_name: str,
    B: int,
    H: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    cold_l2: bool,
    use_cuda_graph: bool,
    warmup_ms: float,
    timed_ms: float,
    causal: bool = False,
) -> dict:
    """
    Benchmark a single (kernel, shape) combination.

    Returns a dict with timing stats and derived metrics, or
    {"error": str} if the kernel fails or is not implemented.
    """
    from benchmarks.utils.timer import bench_gpu_time

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

    try:
        fn = _make_fn(kernel_name, q, k, v, causal=causal)
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (B, H, N, D), f"shape mismatch: {out.shape}"
    except NotImplementedError as e:
        return {"error": f"not_implemented: {e}"}
    except Exception as e:
        return {"error": str(e)}

    times = bench_gpu_time(
        fn,
        cold_l2_cache=cold_l2,
        use_cuda_graph=use_cuda_graph,
        warmup_time_ms=warmup_ms,
        timed_time_ms=timed_ms,
        warmup_iters=25,
        timed_iters=100
    )

    median_ms       = float(np.median(times))
    itemsize        = torch.finfo(dtype).bits // 8
    flops           = attention_flops(B, H, N, D)
    algo_bytes      = attention_bytes(B, H, N, D, itemsize)
    achieved_tflops = flops      / (median_ms * 1e-3) / 1e12
    achieved_bw     = (algo_bytes * 1e-9) / (median_ms * 1e-3)
    sol_compute_pct = achieved_tflops / HW.peak_flops_fp32_tflops * 100
    sol_bw_pct      = achieved_bw     / HW.peak_bandwidth_gbs     * 100
    arithmeticIntensity = (achieved_tflops * 1e12) / (achieved_bw * 1e9)

    return {
        # identity
        "kernel":           kernel_name,
        "B": B, "H": H, "N": N, "D": D,
        "dtype":            str(dtype).replace("torch.", ""),
        "causal":           causal,
        # timing
        "median_ms":        round(median_ms, 6),
        "mean_ms":          round(float(np.mean(times)), 6),
        "std_ms":           round(float(np.std(times)), 6),
        "min_ms":           round(float(np.min(times)), 6),
        "max_ms":           round(float(np.max(times)), 6),
        "cv_pct":           round(float(np.std(times) / np.mean(times) * 100), 3),
        "num_iters":        len(times),
        # analytical
        "flops":            flops,
        "algo_bytes":       algo_bytes,
        # derived metrics
        "achieved_tflops":  round(achieved_tflops, 6),
        "achieved_bw":      round(achieved_bw,      6),
        "sol_compute_pct":  round(sol_compute_pct,  6),
        "sol_bw_pct":       round(sol_bw_pct,       6),
        "arithmetic_intensity": round(arithmeticIntensity, 6)
    }