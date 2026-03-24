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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.timer import bench_gpu_time
from benchmarks.utils.hardware_constants import RTX_4050_LAPTOP
from benchmarks.utils.bench_core import (
    benchmark_shape, HW, KERNEL_STYLE, kernel_label, attention_flops, attention_bytes
    )

H                = 8
D                = 64
DTYPE_STR        = "float32"
DTYPE            = torch.float32
N_VALUES         = [2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
B_VALUES         = [1, 2, 4, 8, 16, 32]   
B_DEFAULT        = 1                        
N_FIXED_DEFAULT  = 512                      
KERNELS_ORDERED  = ["naive_v1", "sdpa"]

def run_all_benchmarks(
    kernels:        list[str],
    sweep_axis:     str,        
    x_values:       list[int],  
    B_fixed:        int,        
    N_fixed:        int,        
    cold_l2:        bool,
    use_cuda_graph: bool,
    warmup_ms:      float,
    timed_ms:       float,
) -> dict:
    results: dict[str, dict[int, dict]] = {k: {} for k in kernels}

    total = len(kernels) * len(x_values)
    done  = 0

    if sweep_axis == 'N':
        sweep_desc = f'N sweep  B={B_fixed} H={H} D={D}'
    else:
        sweep_desc = f'B sweep  N={N_fixed} H={H} D={D}'

    print()
    print('=' * 90)
    print(f'  CUDA Attention Suite -- Benchmark Run')
    print(f'  Hardware : {HW.name}  ({HW.compute_capability})')
    print(f'  Peak     : {HW.peak_flops_fp32_tflops} TFLOPS FP32  |  {HW.peak_bandwidth_gbs} GB/s')
    print(f'  Sweep    : {sweep_desc}  dtype={DTYPE_STR}  cold_l2={cold_l2}')
    print(f'  Kernels  : {kernels}')
    print(f'  {sweep_axis} values : {x_values}')
    print('=' * 90)
    HDR = (f"  {'Kernel':<13}  {sweep_axis:>5}  {'median_ms':>10}  {'TFLOPS':>9}  "
           f"{'BW GB/s':>9}  {'AI FLOPs/B':>11}  {'SOL_TF%':>8}  {'SOL_BW%':>8}  {'CV%':>6}  {'iters':>6}")
    print(HDR)
    print('  ' + '-' * (len(HDR) - 2))

    for kernel_name in kernels:
        for x_val in x_values:
            done += 1
            B_run = B_fixed if sweep_axis == 'N' else x_val
            N_run = x_val   if sweep_axis == 'N' else N_fixed
            print(f'  [{done:>2}/{total}] running {kernel_name:<12} {sweep_axis}={x_val:<6} ...', flush=True)
            t0  = time.perf_counter()
            rec = benchmark_shape(kernel_name, B_run, H, N_run, D, DTYPE, cold_l2, use_cuda_graph, warmup_ms, timed_ms)
            elapsed = time.perf_counter() - t0

            if 'error' in rec:
                print(f"  {'SKIP':<13}  {x_val:>5}  {rec['error']}")
                continue

            results[kernel_name][x_val] = rec
            print(
                f'  {kernel_name:<13}  {x_val:>5}  '
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>9.5f}  "
                f"{rec['achieved_bw']:>9.3f}  "
                f"{rec['arithmetic_intensity']:>11.3f}  "
                f"{rec['sol_compute_pct']:>8.3f}  "
                f"{rec['sol_bw_pct']:>8.3f}  "
                f"{rec['cv_pct']:>6.2f}  "
                f"{rec['num_iters']:>6}  "
                f'({elapsed:.1f}s)'
            )

    print('=' * 90)
    return results

def build_plot(results: dict, sweep_axis: str, x_values: list[int], B_fixed: int, N_fixed: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    BG_DARK   = "#ffffff"
    BG_PANEL  = "#f6f8fa"
    GRID_COL  = "#d0d7de"
    TEXT_COL  = "#1f2328"
    SPINE_COL = "#d0d7de"

    DEFAULT_STYLE = dict(color="#57606a", marker="x", lw=1.5, ms=6)

    REF_TFLOPS_COL = "#b45309"
    REF_BW_COL     = "#1a7f37"
    REF_SDPA_COL   = "#0969da"

    def style(k):
        s = KERNEL_STYLE.get(k, DEFAULT_STYLE).copy()
        s.setdefault("ls", "-")
        return s

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)

    kernel_list = [k for k in results if results[k]]

    def _panel_setup(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG_PANEL)
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)
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

    if sweep_axis == 'N':
        xlabel     = 'Sequence Length N'
        fixed_desc = f'B={B_fixed}'
    else:
        xlabel     = 'Batch Size B'
        fixed_desc = f'N={N_fixed}'

    ax0 = axes[0, 0]
    _panel_setup(ax0, f'Execution Time vs {xlabel}', xlabel, 'Time (ms)')
    for k in kernel_list:
        xs   = sorted(results[k])
        vals = [results[k][x]['median_ms'] for x in xs]
        s    = style(k)
        ax0.plot(xs, vals, color=s['color'], marker=s['marker'],
                 lw=s['lw'], ms=s['ms'], ls=s['ls'])

    ax1 = axes[0, 1]
    _panel_setup(ax1, f'Achieved TFLOPS vs {xlabel}', xlabel, 'Achieved TFLOPS')
    ax1.axhline(HW.peak_flops_fp32_tflops, color=REF_TFLOPS_COL, lw=2.0,
                ls='--', alpha=0.8, label=f'Peak FP32 ({HW.peak_flops_fp32_tflops} TFLOPS)')
    for k in kernel_list:
        xs   = sorted(results[k])
        vals = [results[k][x]['achieved_tflops'] for x in xs]
        s    = style(k)
        ax1.plot(xs, vals, color=s['color'], marker=s['marker'],
                 lw=s['lw'], ms=s['ms'], ls=s['ls'])

    ax2 = axes[1, 0]
    _panel_setup(ax2, f'Achieved Memory BW vs {xlabel}', xlabel, 'Bandwidth (GB/s)')
    ax2.axhline(HW.peak_bandwidth_gbs, color=REF_BW_COL, lw=2.0,
                ls='--', alpha=0.8, label=f'Peak BW ({HW.peak_bandwidth_gbs} GB/s)')
    for k in kernel_list:
        xs   = sorted(results[k])
        vals = [results[k][x]['achieved_bw'] for x in xs]
        s    = style(k)
        ax2.plot(xs, vals, color=s['color'], marker=s['marker'],
                 lw=s['lw'], ms=s['ms'], ls=s['ls'])

    ax3 = axes[1, 1]
    _panel_setup(ax3, f'Speedup vs PyTorch SDPA ({xlabel})', xlabel, 'Speedup (x)')
    ax3.set_yscale("linear")
    ax3.axhline(1.0, color=REF_SDPA_COL, lw=1.5, ls="--", alpha=0.9,
                label="PyTorch SDPA (1.0x)")

    if 'sdpa' in results and results['sdpa']:
        for k in kernel_list:
            if k == 'sdpa':
                continue
            xs_common = sorted(set(results[k]) & set(results['sdpa']))
            if not xs_common:
                continue
            speedups = [
                results['sdpa'][xv]['median_ms'] / results[k][xv]['median_ms']
                for xv in xs_common
            ]
            s = style(k)
            ax3.plot(xs_common, speedups, color=s['color'], marker=s['marker'],
                     lw=s['lw'], ms=s['ms'], ls='-')
    else:
        ax3.text(0.5, 0.5, "sdpa baseline not available",
                 transform=ax3.transAxes, ha="center", va="center",
                 color="#57606a", fontsize=10)

    legend_handles = []
    legend_labels  = []
    for k in kernel_list:
        s = style(k)
        h = plt.Line2D([0], [0], color=s["color"], marker=s["marker"],
                       lw=s["lw"], ms=s["ms"], ls=s["ls"])
        legend_handles.append(h)
        legend_labels.append(kernel_label(k))  # label lives in KERNEL_REGISTRY
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

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(
        f'Attention Kernel Benchmarks  --  {HW.name}  (sm_89 Ada Lovelace)\n'
        f'{fixed_desc}  Head={H}  Head_Dim={D}  dtype={DTYPE_STR}',
        color=TEXT_COL, fontsize=12, fontweight='bold', y=0.99,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")

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
    parser.add_argument("--sweep-axis", choices=["N", "B"], default="N", dest="sweep_axis")
    parser.add_argument("--B", nargs="+", type=int, default=[1,2,4,8,16,32], dest="b_values")
    parser.add_argument("--B-fixed", type=int, default=1, dest="b_fixed")
    parser.add_argument("--N-fixed", type=int, default=512, dest="n_fixed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available."); sys.exit(1)

    kernels = list(dict.fromkeys(args.kernels))
    if 'sdpa' not in kernels:
        kernels.append('sdpa')

    if args.sweep_axis == 'N':
        x_values = [n for n in N_VALUES if n not in args.skip_n]
    else:
        x_values = args.b_values  
    cold_l2 = not args.no_l2_flush

    results = run_all_benchmarks(
        kernels        = kernels,
        sweep_axis     = args.sweep_axis,
        x_values       = x_values,
        B_fixed        = args.b_fixed,
        N_fixed        = args.n_fixed,
        cold_l2        = cold_l2,
        use_cuda_graph = args.cuda_graph,
        warmup_ms      = args.warmup_ms,
        timed_ms       = args.timed_ms,
    )

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_tag = f'sweep{args.sweep_axis}'
    out_path  = (
        Path(args.out) if args.out
        else PROJECT_ROOT / 'benchmarks' / 'reports' / 'plots'
                         / f'attention_benchmark_{sweep_tag}_{ts}.png'
    )

    build_plot(results, args.sweep_axis, x_values, args.b_fixed, args.n_fixed, out_path)

    print()
    print('  SUMMARY TABLE')
    ax_lbl = args.sweep_axis
    print(f"  {'Kernel':<12}  {ax_lbl:>5}  {'median_ms':>10}  {'TFLOPS':>8}  {'AI FLOPs/B':>11}  {'SOL%':>7}  {'speedup':>8}")
    print('  ' + '-' * 72)
    sdpa_times = results.get('sdpa', {})
    for k in kernels:
        for xv in x_values:
            rec = results[k].get(xv)
            if rec is None:
                continue
            speedup = (
                f"{sdpa_times[xv]['median_ms'] / rec['median_ms']:.3f}x"
                if xv in sdpa_times and k != 'sdpa' else '--'
            )
            print(
                f'  {k:<12}  {xv:>5}  '
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>8.5f}  "
                f"{rec['arithmetic_intensity']:>11.3f}  "
                f"{rec['sol_compute_pct']:>6.3f}%  "
                f'{speedup:>8}'
            )
    print()


if __name__ == "__main__":
    main()