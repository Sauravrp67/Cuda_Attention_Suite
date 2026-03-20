#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.timer import bench_gpu_time
from benchmarks.utils.hardware_constants import RTX_4050_LAPTOP

HW = RTX_4050_LAPTOP

# ---------------------------------------------------------------------------
# Shape registry — mirrors conftest.py exactly
# ---------------------------------------------------------------------------
SHAPES_SMALL = [
    (1, 1,    1,  64),
    (1, 1,   64,  64),
    (1, 1,  128,  64),
    (1, 8,  128,  64),
    (2, 8,  128,  64),
    (1, 1,  512,  64),
    (1, 8,  512,  64),
    (2, 8,  512,  64),
    (1, 8, 1024,  64),
    (2, 8, 1024,  64),
]
SHAPES_LARGE = [
    # (1,  8, 2048,  64),
    # (1,  8, 4096,  64),
    # (2, 32, 2048, 128),
    (2,8,2048,64)
]
SHAPES_ALL = SHAPES_SMALL + SHAPES_LARGE

# ---------------------------------------------------------------------------
# Analytical accounting (algorithmic lower bound — FA paper convention)
# ---------------------------------------------------------------------------
def attention_flops(B, H, N, D) -> int:
    """QK^T + AV = 2 * 2 * B*H*N^2*D FLOPs."""
    return 4 * B * H * N * N * D

def attention_bytes(B, H, N, D, elem_bytes: int = 4) -> int:
    """Q+K+V reads + O write, each touched once. Lower bound."""
    return 4 * B * H * N * D * elem_bytes

ELEM_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------
def make_fn(kernel_name, q, k, v):
    if kernel_name == "naive_v1":
        from PyCuAttention.kernels.loader import naive_attn
        return lambda: naive_attn(q, k, v)
    elif kernel_name == "sdpa":
        from torch.nn.functional import scaled_dot_product_attention
        scale = q.shape[-1] ** -0.5
        return lambda: scaled_dot_product_attention(q, k, v, scale=scale)
    elif kernel_name == "fused_v2":
        from PyCuAttention.kernels.loader import fused_attn
        scale = q.shape[-1] ** -0.5
        return lambda: fused_attn(q, k, v, scale)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

# ---------------------------------------------------------------------------
# Single shape benchmark
# ---------------------------------------------------------------------------
def benchmark_shape(
    kernel_name, B, H, N, D, dtype_str,
    cold_l2, use_cuda_graph,
    warmup_time_ms, timed_time_ms,
):
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_str]

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

    try:
        fn = make_fn(kernel_name, q, k, v)
        # Validate before timing
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (B, H, N, D)
    except Exception as e:
        return {"error": str(e)}

    times = bench_gpu_time(
        fn,
        cold_l2_cache=cold_l2,
        use_cuda_graph=use_cuda_graph,
        warmup_time_ms=warmup_time_ms,
        timed_time_ms=timed_time_ms,
    )

    median_ms = float(np.median(times))
    elem      = ELEM_BYTES[dtype_str]
    flops     = attention_flops(B, H, N, D)
    # bytesmov  = attention_bytes(B, H, N, D, elem)

    # achieved_bw_gbs = bytesmov / (median_ms * 1e-3) / 1e9
    achieved_tflops = flops    / (median_ms * 1e-3) / 1e12
    # sol_bw_pct      = achieved_bw_gbs  / HW.peak_bandwidth_gbs       * 100
    sol_compute_pct = achieved_tflops  / HW.peak_flops_fp32_tflops   * 100

    # Classify: bound by whichever SOL% is higher
    # bound_by = "memory" if sol_bw_pct > sol_compute_pct else "compute"
    # sol_pct  = max(sol_bw_pct, sol_compute_pct)

    return {
        "kernel":           kernel_name,
        "B": B, "H": H, "N": N, "D": D,
        "dtype":            dtype_str,
        # timing
        "median_ms":        round(median_ms,              6),
        "mean_ms":          round(float(np.mean(times)),  6),
        "std_ms":           round(float(np.std(times)),   6),
        "min_ms":           round(float(np.min(times)),   6),
        "max_ms":           round(float(np.max(times)),   6),
        "cv_pct":           round(float(np.std(times)/np.mean(times)*100), 3),
        "num_iters":        len(times),
        # analytical accounting
        "flops":            flops,
        # "bytes_moved":      bytesmov,          # algorithmic lower bound
        # achieved metrics
        # "achieved_bw_gbs":  round(achieved_bw_gbs, 4),
        "achieved_tflops":  round(achieved_tflops, 6),
        # "sol_bw_pct":       round(sol_bw_pct, 4),
        "sol_compute_pct":  round(sol_compute_pct, 6),
        # "bound_by":         bound_by,
        "sol_pct":          round(sol_compute_pct,4),
        # config flags
        "cold_l2_cache":    cold_l2,
        "use_cuda_graph":   use_cuda_graph,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels",        nargs="+", default=["naive_v1"])
    parser.add_argument("--shapes",         choices=["small","large","all"], default="all")
    parser.add_argument("--dtype",          default="float32",
                        choices=["float32","float16","bfloat16"])
    parser.add_argument("--cuda-graph",     action="store_true",
                        help="Use CUDA Graphs (better for fast kernels <200µs)")
    parser.add_argument("--no-l2-flush",    action="store_true",
                        help="Skip L2 flush (faster runs, warm-cache numbers)")
    parser.add_argument("--warmup-ms",      type=float, default=25.0)
    parser.add_argument("--timed-ms",       type=float, default=100.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available."); sys.exit(1)

    shapes = {"small": SHAPES_SMALL, "large": SHAPES_LARGE, "all": SHAPES_ALL}[args.shapes]
    cold_l2     = not args.no_l2_flush
    cuda_graph  = args.cuda_graph

    report_dir = PROJECT_ROOT / "benchmarks" / "reports" / "timing"
    report_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print(f"  run_timing.py  |  {HW.name}  |  Peak {HW.peak_flops_fp32_tflops} TFLOPS  "
          f"| Peak BW {HW.peak_bandwidth_gbs} GB/s  |  Ridge {HW.ridge_point:.1f} FLOPs/B")
    print(f"  kernels={args.kernels}  dtype={args.dtype}  "
          f"cold_l2={cold_l2}  cuda_graph={cuda_graph}")
    print(f"  bytes_moved = algorithmic lower bound  "
          f"(4·B·H·N·D·{ELEM_BYTES[args.dtype]} bytes  —  FA paper convention)")
    print("=" * 100)

    # Column header
    HDR = (f"  {'Kernel':<12} {'B':>2} {'H':>2} {'N':>5} {'D':>4}  "
           f"{'median_ms':>10}  {'TFLOPS':>8}  {'BW GB/s':>9}  "
           f"{'SOL BW%':>8}  {'SOL TF%':>8}  {'Bound':>8}  {'%SOL':>6}  {'CV%':>5}")
    SEP = "  " + "-" * (len(HDR) - 2)
    print(); print(HDR); print(SEP)

    all_results = []

    for B, H, N, D in shapes:
        for kernel_name in args.kernels:
            rec = benchmark_shape(
                kernel_name, B, H, N, D, args.dtype,
                cold_l2, cuda_graph, args.warmup_ms, args.timed_ms
            )

            if "error" in rec:
                print(f"  {'ERROR':<12} B={B} H={H} N={N} D={D}  {rec['error']}")
                continue

            def fms(v): return f"{v:.4f}" if v is not None else "—"
            print(
                f"  {kernel_name:<12} {B:>2} {H:>2} {N:>5} {D:>4}"
                f"  {rec['median_ms']:>10.4f}"
                f"  {rec['achieved_tflops']:>8.5f}"
                # f"  {rec['achieved_bw_gbs']:>9.3f}"
                # f"  {rec['sol_bw_pct']:>8.3f}%"
                f"  {rec['sol_compute_pct']:>8.5f}%"
                # f"  {rec['bound_by']:>8}"
                # f"  {rec['sol_pct']:>6.3f}%"
                f"  {rec['cv_pct']:>5.2f}%"
            )

            stem     = f"{kernel_name}_B{B}_H{H}_N{N}_D{D}_{args.dtype}"
            per_path = report_dir / f"{stem}_timing.json"
            with open(per_path, "w") as f:
                json.dump(rec, f, indent=2)

            all_results.append(rec)

    combined = report_dir / "all_timings.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    print(SEP)
    print(f"\n  {len(all_results)} records saved → {combined}")
    print("  Next: python3 benchmarks/runners/plot_results.py")
    print("=" * 100)


if __name__ == "__main__":
    main()