#!/usr/bin/env python3
"""
benchmarks/runners/run_viz.py

Run attention kernel benchmarks fresh and produce a 4-panel comparison plot.

Fixed config:  B=1, H=8, D=64, dtype=float32
N sweep:       [64, 128, 256, 512, 1024, 2048, 4096]
Kernels:       naive_v1  +  sdpa (PyTorch baseline)
Output PNG:    benchmarks/reports/plots/attention_benchmark_<timestamp>.png

Usage:
    python3 benchmarks/runners/run_viz.py
    python3 benchmarks/runners/run_viz.py --skip-n 4096   # drop slowest point
    python3 benchmarks/runners/run_viz.py --warmup-ms 50 --timed-ms 200
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Path setup — works whether called from repo root or runners/

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.timer import bench_gpu_time
from benchmarks.utils.hardware_constants import RTX_4050_LAPTOP

HW = RTX_4050_LAPTOP

# Fixed benchmark configuration

B, H, D          = 64, 8, 64
DTYPE_STR        = "float32"
DTYPE            = torch.float32
N_VALUES         = [2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
KERNELS_ORDERED  = ["naive_v1", "sdpa"]   # sdpa is always the baseline


# Analytical accounting
def attention_flops(B, H, N, D) -> int:
    """QK^T + AV = 4·B·H·N²·D FLOPs  (FA paper convention)."""
    return 4 * B * H * N * N * D

def attention_bytes(B, H, N, D) -> int:
    """Algorithmic lower bound: read Q+K+V, write O — each touched once."""
    return 4 * B * H * N * D * DTYPE.itemsize

# Kernel factory
def make_fn(kernel_name: str, q, k, v):
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

# Single-shape benchmark → dict
def benchmark_shape(
    kernel_name: str,
    N: int,
    cold_l2: bool,
    use_cuda_graph: bool,
    warmup_ms: float,
    timed_ms: float,
) -> dict:
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, dtype=DTYPE, device="cuda")
    k = torch.randn(B, H, N, D, dtype=DTYPE, device="cuda")
    v = torch.randn(B, H, N, D, dtype=DTYPE, device="cuda")

    try:
        fn = make_fn(kernel_name, q, k, v)
        out = fn()
        torch.cuda.synchronize()
        assert out.shape == (B, H, N, D), f"shape mismatch: {out.shape}"
    except Exception as e:
        return {"error": str(e)}

    times = bench_gpu_time(
        fn,
        cold_l2_cache=cold_l2,
        use_cuda_graph=use_cuda_graph,
        warmup_time_ms=warmup_ms,
        timed_time_ms=timed_ms,
    )

    median_ms       = float(np.median(times))
    flops           = attention_flops(B, H, N, D)
    algo_bytes      = attention_bytes(B, H, N, D)
    achieved_tflops = flops      / (median_ms * 1e-3) / 1e12
    achieved_bw     = (algo_bytes * 1e-9) / (median_ms * 1e-3)   # GB/s
    sol_compute_pct = achieved_tflops / HW.peak_flops_fp32_tflops * 100
    sol_bw_pct      = achieved_bw     / HW.peak_bandwidth_gbs     * 100

    return {
        "kernel":           kernel_name,
        "N":                N,
        "median_ms":        round(median_ms, 6),
        "mean_ms":          round(float(np.mean(times)), 6),
        "std_ms":           round(float(np.std(times)), 6),
        "min_ms":           round(float(np.min(times)), 6),
        "max_ms":           round(float(np.max(times)), 6),
        "cv_pct":           round(float(np.std(times) / np.mean(times) * 100), 3),
        "num_iters":        len(times),
        "flops":            flops,
        "algo_bytes":       algo_bytes,
        "achieved_tflops":  round(achieved_tflops, 6),
        "achieved_bw":      round(achieved_bw,      6),
        "sol_compute_pct":  round(sol_compute_pct,  6),
        "sol_bw_pct":       round(sol_bw_pct,       6),
    }

# Run all benchmarks → nested dict  results[kernel][N] = rec
def run_all_benchmarks(
    kernels: list[str],
    n_values: list[int],
    cold_l2: bool,
    use_cuda_graph: bool,
    warmup_ms: float,
    timed_ms: float,
) -> dict:
    results: dict[str, dict[int, dict]] = {k: {} for k in kernels}

    total = len(kernels) * len(n_values)
    done  = 0

    print()
    print("=" * 90)
    print(f"  CUDA Attention Suite — Benchmark Run")
    print(f"  Hardware : {HW.name}  ({HW.compute_capability})")
    print(f"  Peak     : {HW.peak_flops_fp32_tflops} TFLOPS FP32  |  {HW.peak_bandwidth_gbs} GB/s")
    print(f"  Config   : B={B}  H={H}  D={D}  dtype={DTYPE_STR}  cold_l2={cold_l2}")
    print(f"  Kernels  : {kernels}")
    print(f"  N values : {n_values}")
    print("=" * 90)
    HDR = (f"  {'Kernel':<13}  {'N':>5}  {'median_ms':>10}  {'TFLOPS':>9}  "
           f"{'BW GB/s':>9}  {'SOL_TF%':>8}  {'SOL_BW%':>8}  {'CV%':>6}  {'iters':>6}")
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    for kernel_name in kernels:
        for N in n_values:
            done += 1
            # print(f"  [{done:>2}/{total}] running {kernel_name:<12} N={N:<5} ...", flush=True)
            t0  = time.perf_counter()
            rec = benchmark_shape(kernel_name, N, cold_l2, use_cuda_graph, warmup_ms, timed_ms)
            elapsed = time.perf_counter() - t0

            if "error" in rec:
                print(f"  {'ERROR':<13}  {N:>5}  {rec['error']}")
                continue

            results[kernel_name][N] = rec
            # Each field width matches HDR definition above exactly
            print(
                f"  {kernel_name:<13}  {N:>5}  "
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>9.5f}  "
                f"{rec['achieved_bw']:>9.3f}  "
                f"{rec['sol_compute_pct']:>8.3f}  "
                f"{rec['sol_bw_pct']:>8.3f}  "
                f"{rec['cv_pct']:>6.2f}  "
                f"{rec['num_iters']:>6}  "
                f"({elapsed:.1f}s)"
            )

    print("=" * 90)
    return results

def build_plot(results: dict, n_values: list[int], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # ── Palette — light theme ─────────
    BG_DARK   = "#ffffff"
    BG_PANEL  = "#f6f8fa"
    GRID_COL  = "#d0d7de"
    TEXT_COL  = "#1f2328"
    SPINE_COL = "#d0d7de"

    KERNEL_STYLE: dict[str, dict] = {
        "naive_v1": dict(color="#cf222e", marker="o",  lw=2.0, ms=7,  label="Naive v1  (CUDA)"),
        "fused_v2": dict(color="#1a7f37", marker="s",  lw=2.0, ms=7,  label="Fused v2  (Shared Mem)"),
        "flash_v3": dict(color="#b45309", marker="^",  lw=2.0, ms=7,  label="Flash v3  (Tiled)"),
        "paged_v4": dict(color="#6639ba", marker="D",  lw=2.0, ms=7,  label="Paged v4  (KV Cache)"),
        "sdpa":     dict(color="#0969da", marker="*",  lw=2.5, ms=10, ls="--", label="PyTorch SDPA (baseline)"),
    }
    DEFAULT_STYLE = dict(color="#57606a", marker="x", lw=1.5, ms=6)

    REF_TFLOPS_COL = "#b45309"
    REF_BW_COL     = "#1a7f37"
    REF_SDPA_COL   = "#0969da"

    def style(k):
        s = KERNEL_STYLE.get(k, DEFAULT_STYLE).copy()
        s.setdefault("ls", "-")
        return s

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)

    kernel_list = [k for k in results if results[k]]

    def _panel_setup(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG_PANEL)
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)  # plain text, no emoji
        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.grid(True, which="both",  color=GRID_COL, lw=0.5, alpha=0.9)
        ax.grid(True, which="minor", color=GRID_COL, lw=0.3, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)

    # ── Panel 0: Execution Time ──────────────────────────────────────────────
    ax0 = axes[0, 0]
    _panel_setup(ax0, "Execution Time vs Sequence Length",
                 "Sequence Length N", "Time (ms)")
    for k in kernel_list:
        ns   = sorted(results[k])
        vals = [results[k][n]["median_ms"] for n in ns]
        s    = style(k)
        ax0.plot(ns, vals, color=s["color"], marker=s["marker"],
                 lw=s["lw"], ms=s["ms"], ls=s["ls"])

    # ── Panel 1: Achieved TFLOPS ─────────────────────────────────────────────
    ax1 = axes[0, 1]
    _panel_setup(ax1, "Achieved TFLOPS vs Sequence Length",
                 "Sequence Length N", "Achieved TFLOPS")
    ax1.axhline(HW.peak_flops_fp32_tflops, color=REF_TFLOPS_COL, lw=2.0,   # thicker
                ls="--", alpha=0.8, label=f"Peak FP32 ({HW.peak_flops_fp32_tflops} TFLOPS)")
    for k in kernel_list:
        ns   = sorted(results[k])
        vals = [results[k][n]["achieved_tflops"] for n in ns]
        s    = style(k)
        ax1.plot(ns, vals, color=s["color"], marker=s["marker"],
                 lw=s["lw"], ms=s["ms"], ls=s["ls"])

    # ── Panel 2: Achieved Memory Bandwidth ──────────────────────────────────
    ax2 = axes[1, 0]
    _panel_setup(ax2, "Achieved Memory BW vs Sequence Length",
                 "Sequence Length N", "Bandwidth (GB/s)")
    ax2.axhline(HW.peak_bandwidth_gbs, color=REF_BW_COL, lw=2.0,           # thicker
                ls="--", alpha=0.8, label=f"Peak BW ({HW.peak_bandwidth_gbs} GB/s)")
    for k in kernel_list:
        ns   = sorted(results[k])
        vals = [results[k][n]["achieved_bw"] for n in ns]
        s    = style(k)
        ax2.plot(ns, vals, color=s["color"], marker=s["marker"],
                 lw=s["lw"], ms=s["ms"], ls=s["ls"])

    # ── Panel 3: Speedup vs sdpa ─────────────────────────────────────────────
    ax3 = axes[1, 1]
    _panel_setup(ax3, "Speedup vs PyTorch SDPA (baseline)",
                 "Sequence Length N", "Speedup (x)")                        # plain "x", not "×"
    ax3.set_yscale("linear")
    ax3.axhline(1.0, color=REF_SDPA_COL, lw=1.5, ls="--", alpha=0.9,
                label="PyTorch SDPA (1.0x)")

    if "sdpa" in results and results["sdpa"]:
        for k in kernel_list:
            if k == "sdpa":
                continue
            ns_common = sorted(set(results[k]) & set(results["sdpa"]))
            if not ns_common:
                continue
            speedups = [
                results["sdpa"][n]["median_ms"] / results[k][n]["median_ms"]
                for n in ns_common
            ]
            s = style(k)
            ax3.plot(ns_common, speedups, color=s["color"], marker=s["marker"],
                     lw=s["lw"], ms=s["ms"], ls="-")
    else:
        ax3.text(0.5, 0.5, "sdpa baseline not available",
                 transform=ax3.transAxes, ha="center", va="center",
                 color="#57606a", fontsize=10)

    # ── Unified legend below all panels ──────────────────────────────────────
    legend_handles = []
    legend_labels  = []
    for k in kernel_list:
        s = style(k)
        h = plt.Line2D([0], [0], color=s["color"], marker=s["marker"],
                       lw=s["lw"], ms=s["ms"], ls=s["ls"])
        legend_handles.append(h)
        legend_labels.append(s["label"])
    legend_handles.append(plt.Line2D([0], [0], color=REF_TFLOPS_COL, lw=2.0, ls="--"))
    legend_labels.append(f"Peak FP32 ({HW.peak_flops_fp32_tflops} TFLOPS)")
    legend_handles.append(plt.Line2D([0], [0], color=REF_BW_COL, lw=2.0, ls="--"))
    legend_labels.append(f"Peak BW ({HW.peak_bandwidth_gbs} GB/s)")
    legend_handles.append(plt.Line2D([0], [0], color=REF_SDPA_COL, lw=1.5, ls="--"))
    legend_labels.append("PyTorch SDPA (baseline)")

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        frameon=True,
        framealpha=0.9,
        facecolor=BG_PANEL,
        edgecolor=SPINE_COL,
        fontsize=9,
        labelcolor=TEXT_COL,
        bbox_to_anchor=(0.5, 0.01),
    )

    # ── Title & metadata ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Attention Kernel Benchmarks  —  {HW.name}  (sm_89 Ada Lovelace)\n"
        f"Batch={B}  Head={H}  Head_Dim={D}  dtype={DTYPE_STR}",
        color=TEXT_COL, fontsize=12, fontweight="bold", y=0.99,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")
    
# Main
def main():
    parser = argparse.ArgumentParser(
        description="Run attention benchmarks and produce 4-panel comparison plot."
    )
    parser.add_argument(
        "--kernels", nargs="+",
        default=["naive_v1","sdpa"],
        help="Kernels to benchmark. sdpa is always included as baseline.",
    )
    parser.add_argument(
        "--skip-n", nargs="*", type=int, default=[],
        metavar="N",
        help="N values to skip (e.g. --skip-n 4096 to drop the slowest point).",
    )
    parser.add_argument("--warmup-ms",    type=float, default=25.0)
    parser.add_argument("--timed-ms",     type=float, default=100.0)
    parser.add_argument("--cuda-graph",   action="store_true")
    parser.add_argument("--no-l2-flush",  action="store_true")
    parser.add_argument("--out",          type=str, default=None,
                        help="Override output PNG path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available."); sys.exit(1)

    # sdpa must always be present for the speedup panel
    kernels = list(dict.fromkeys(args.kernels))   # dedup, preserve order
    if "sdpa" not in kernels:
        kernels.append("sdpa")

    n_values = [n for n in N_VALUES if n not in args.skip_n]
    cold_l2  = not args.no_l2_flush

    # ── Run ──────────────────────────────────────────────────────────────────
    results = run_all_benchmarks(
        kernels      = kernels,
        n_values     = n_values,
        cold_l2      = cold_l2,
        use_cuda_graph = args.cuda_graph,
        warmup_ms    = args.warmup_ms,
        timed_ms     = args.timed_ms,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "benchmarks" / "reports" / "plots"
                         / f"attention_benchmark_{ts}.png"
    )

    build_plot(results, n_values, out_path)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("  SUMMARY TABLE")
    print(f"  {'Kernel':<12}  {'N':>5}  {'median_ms':>10}  {'TFLOPS':>8}  {'SOL%':>7}  {'speedup':>8}")
    print("  " + "-" * 62)
    sdpa_times = results.get("sdpa", {})
    for k in kernels:
        for N in n_values:
            rec = results[k].get(N)
            if rec is None:
                continue
            speedup = (
                f"{sdpa_times[N]['median_ms'] / rec['median_ms']:.3f}×"
                if N in sdpa_times and k != "sdpa" else "—"
            )
            print(
                f"  {k:<12}  {N:>5}  "
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>8.5f}  "
                f"{rec['sol_compute_pct']:>6.3f}%  "
                f"{speedup:>8}"
            )
    print()


if __name__ == "__main__":
    main()