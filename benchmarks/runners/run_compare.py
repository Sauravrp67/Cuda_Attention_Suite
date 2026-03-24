#!/usr/bin/env python3
"""
benchmarks/runners/run_compare.py

Head-to-head kernel comparison — bar chart style, FA paper aesthetic.

Sweep modes
-----------
--sweep-axis N  (default)
    N on x-axis, B fixed.  Shows scaling behavior with sequence length.

--sweep-axis B
    B on x-axis, N fixed.  Shows how arithmetic intensity scales with
    batch size for a fixed sequence length.

Usage
-----
# N sweep (default)
python3 benchmarks/runners/run_compare.py

# B sweep, N fixed at 1024
python3 benchmarks/runners/run_compare.py --sweep-axis B --N-fixed 1024

# Custom values
python3 benchmarks/runners/run_compare.py --sweep-axis N --N 64 256 1024 4096
python3 benchmarks/runners/run_compare.py --sweep-axis B --B 1 2 4 8 16 32

# Annotate bars with exact values
python3 benchmarks/runners/run_compare.py --annotate

# With causal mask (once kernels support it)
python3 benchmarks/runners/run_compare.py --causal

Extending
---------
- New kernel     -> KERNEL_REGISTRY + _make_fn() in bench_core.py only
- New metric     -> uncomment / add entry in METRICS dict below
- Causal mask    -> --causal flag; kernels read causal from benchmark_shape()
- New axis       -> add argparse arg, pass through run_all()
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.bench_core import (
    HW,
    KERNEL_REGISTRY,
    benchmark_shape,
    kernel_bar_color,
    kernel_label,
)

# Defaults
DEFAULT_N_VALUES    = [2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_B_VALUES    = [1, 2, 4, 8, 16, 32,64,128,256]
DEFAULT_KERNELS     = ["sdpa"]

H               = 8
D_DEFAULT       = 64
B_DEFAULT       = 1        
N_FIXED_DEFAULT = 512   

DTYPE     = torch.float32
DTYPE_STR = "float32"

# Metric definitions
METRICS = {
    "tflops": dict(
        key        = "achieved_tflops",
        ylabel     = "Speed (TFLOPs/s)",
        title_tmpl = "Attention forward speed ({hw})  --  {config}",
        ref_line   = None,
        log_y      = False,
    ),
    # "latency": dict(
    #     key        = "median_ms",
    #     ylabel     = "Latency (ms)",
    #     title_tmpl = "Attention latency ({hw})  --  {config}",
    #     ref_line   = None,
    #     log_y      = True,
    # ),
    # "bandwidth": dict(
    #     key        = "achieved_bw",
    #     ylabel     = "Achieved BW (GB/s)",
    #     title_tmpl = "Achieved memory bandwidth ({hw})  --  {config}",
    #     ref_line   = HW.peak_bandwidth_gbs,
    #     log_y      = False,
    # ),
}


# Run benchmarks
def run_all(
    kernels:        list[str],
    sweep_axis:     str,
    x_values:       list[int],
    B_fixed:        int,
    N_fixed:        int,
    D:              int,
    causal:         bool,
    cold_l2:        bool,
    use_cuda_graph: bool,
    warmup_ms:      float,
    timed_ms:       float,
) -> dict[str, dict[int, dict]]:
    results: dict[str, dict[int, dict]] = {k: {} for k in kernels}
    total = len(kernels) * len(x_values)
    done  = 0

    causal_str = "causal" if causal else "non-causal"
    if sweep_axis == "N":
        sweep_desc = f"N sweep  B={B_fixed} H={H} D={D}"
    else:
        sweep_desc = f"B sweep  N={N_fixed} H={H} D={D}"

    print()
    print("=" * 95)
    print(f"  run_compare.py  |  {HW.name}  ({HW.compute_capability})")
    print(f"  Peak  : {HW.peak_flops_fp32_tflops} TFLOPS FP32  |  {HW.peak_bandwidth_gbs} GB/s")
    print(f"  Sweep : {sweep_desc}  dtype={DTYPE_STR}  {causal_str}  cold_l2={cold_l2}")
    print(f"  Kernels  : {kernels}")
    print(f"  {sweep_axis} values : {x_values}")
    print("=" * 95)

    HDR = (
        f"  {'Kernel':<13}  {sweep_axis:>5}  {'median_ms':>10}  {'TFLOPS':>9}  "
        f"{'BW GB/s':>9}  {'AI FLOPs/B':>11}  {'SOL_TF%':>8}  {'CV%':>6}  {'iters':>6}"
    )
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    for kernel_name in kernels:
        for x_val in x_values:
            done += 1
            B = B_fixed if sweep_axis == "N" else x_val
            N = x_val   if sweep_axis == "N" else N_fixed

            print(
                f"  [{done:>2}/{total}] running {kernel_name:<12} {sweep_axis}={x_val:<6} ...",
                flush=True,
            )
            t0  = time.perf_counter()
            rec = benchmark_shape(
                kernel_name, B, H, N, D, DTYPE,
                cold_l2, use_cuda_graph, warmup_ms, timed_ms,
                causal=causal,
            )
            elapsed = time.perf_counter() - t0

            if "error" in rec:
                print(f"  {'SKIP':<13}  {x_val:>5}  {rec['error']}")
                continue

            results[kernel_name][x_val] = rec
            print(
                f"  {kernel_name:<13}  {x_val:>5}  "
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>9.5f}  "
                f"{rec['achieved_bw']:>9.3f}  "
                f"{rec['arithmetic_intensity']:>11.3f}  "
                f"{rec['sol_compute_pct']:>8.3f}  "
                f"{rec['cv_pct']:>6.2f}  "
                f"{rec['num_iters']:>6}  "
                f"({elapsed:.1f}s)"
            )

    print("=" * 95)
    return results

# Plotting
def build_figures(
    results:    dict[str, dict[int, dict]],
    kernels:    list[str],
    sweep_axis: str,
    x_values:   list[int],
    B_fixed:    int,
    N_fixed:    int,
    D:          int,
    causal:     bool,
    annotate:   bool,
    out_dir:    Path,
    ts:         str,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family":        "DejaVu Serif",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "axes.edgecolor":     "#333333",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         "#dddddd",
        "grid.linewidth":     0.6,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.color":        "#333333",
        "ytick.color":        "#333333",
        "text.color":         "#111111",
        "axes.titlesize":     11,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
    })

    active_kernels = [k for k in kernels if results.get(k)]
    n_kernels      = len(active_kernels)
    causal_str     = "causal" if causal else "non-causal"

    if sweep_axis == "N":
        config_str    = f"{causal_str}, B={B_fixed}, head dim {D}"
        footnote_str  = f"B={B_fixed}  H={H}  D={D}  dtype={DTYPE_STR}"
        xlabel        = "Sequence length N"
        x_tick_labels = [
            str(v) if v < 1024 else f"{v // 1024}k" for v in x_values
        ]
    else:
        config_str    = f"{causal_str}, N={N_fixed}, head dim {D}"
        footnote_str  = f"N={N_fixed}  H={H}  D={D}  dtype={DTYPE_STR}"
        xlabel        = "Batch size B"
        x_tick_labels = [str(v) for v in x_values]

    x         = np.arange(len(x_values))
    bar_width = 0.8 / max(n_kernels, 1)
    offsets   = (np.arange(n_kernels) - (n_kernels - 1) / 2) * bar_width

    written: list[Path] = []

    for metric_name, meta in METRICS.items():
        key     = meta["key"]
        ylabel  = meta["ylabel"]
        title   = meta["title_tmpl"].format(
            hw=f"{HW.name}  ({HW.compute_capability})",
            config=config_str,
        )

        fig, ax = plt.subplots(figsize=(max(9, len(x_values) * 0.9), 5))
        fig.subplots_adjust(bottom=0.18)

        for i, kernel_name in enumerate(active_kernels):
            vals, x_pos = [], []
            for j, x_val in enumerate(x_values):
                rec = results[kernel_name].get(x_val)
                if rec is not None:
                    vals.append(rec[key])
                    x_pos.append(x[j] + offsets[i])

            if not vals:
                continue

            bars = ax.bar(
                x_pos, vals,
                width     = bar_width * 0.92,
                color     = kernel_bar_color(kernel_name),
                label     = kernel_label(kernel_name),
                zorder    = 3,
                linewidth = 0.4,
                edgecolor = "white",
            )

            if annotate:
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:.1f}" if val >= 1 else f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=6, color="#333333",
                        rotation=90, clip_on=True,
                    )

        if meta["ref_line"] is not None:
            ax.axhline(meta["ref_line"], color="#cc4444", lw=1.0, ls="--",
                       zorder=2, label=f"HW peak ({meta['ref_line']})")

        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=0)
        if meta["log_y"]:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlim(-0.5, len(x_values) - 0.5)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", ncol=1)
        ax.text(
            0.99, -0.14,
            f"{footnote_str}  |  {HW.name}  |  {ts}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#666666",
        )

        causal_tag = "causal" if causal else "noncausal"
        sweep_tag  = f"sweep{sweep_axis}"
        dim_tag    = f"B{B_fixed}_D{D}" if sweep_axis == "N" else f"N{N_fixed}_D{D}"
        fname      = f"compare_{metric_name}_{sweep_tag}_{dim_tag}_{causal_tag}_{ts}.png"
        out_path   = out_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  Saved -> {out_path}")
        written.append(out_path)

    return written

# Summary table
def print_summary(
    results:    dict[str, dict[int, dict]],
    kernels:    list[str],
    sweep_axis: str,
    x_values:   list[int],
) -> None:
    print()
    print("  SUMMARY")
    HDR = (
        f"  {'Kernel':<13}  {sweep_axis:>5}  {'median_ms':>10}  "
        f"{'TFLOPS':>9}  {'BW GB/s':>9}  {'AI FLOPs/B':>11}  "
        f"{'SOL_TF%':>8}  {'SOL_BW%':>8}"
    )
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))
    for k in kernels:
        for x_val in x_values:
            rec = results[k].get(x_val)
            if rec is None:
                continue
            print(
                f"  {k:<13}  {x_val:>5}  "
                f"{rec['median_ms']:>10.4f}  "
                f"{rec['achieved_tflops']:>9.5f}  "
                f"{rec['achieved_bw']:>9.3f}  "
                f"{rec['arithmetic_intensity']:>11.3f}  "
                f"{rec['sol_compute_pct']:>8.3f}  "
                f"{rec['sol_bw_pct']:>8.3f}"
            )
    print()

# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head kernel comparison bar chart (FA paper style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernels", nargs="+", default=DEFAULT_KERNELS,
        help=f"Kernels to compare. Known: {list(KERNEL_REGISTRY)}",
    )
    parser.add_argument(
        "--sweep-axis", choices=["N", "B"], default="N",
        dest="sweep_axis",
        help="Dimension on the x-axis: N (sequence length) or B (batch size).",
    )
    # N-sweep args
    parser.add_argument(
        "--N", nargs="+", type=int, default=DEFAULT_N_VALUES,
        metavar="N", dest="n_values",
        help="Sequence length values (used when --sweep-axis N).",
    )
    parser.add_argument(
        "--B-fixed", type=int, default=B_DEFAULT,
        metavar="B", dest="b_fixed",
        help="Fixed batch size when --sweep-axis N (default: 1).",
    )
    # B-sweep args
    parser.add_argument(
        "--B", nargs="+", type=int, default=DEFAULT_B_VALUES,
        metavar="B", dest="b_values",
        help="Batch size values (used when --sweep-axis B).",
    )
    parser.add_argument(
        "--N-fixed", type=int, default=N_FIXED_DEFAULT,
        metavar="N", dest="n_fixed",
        help="Fixed sequence length when --sweep-axis B (default: 512).",
    )
    # Shared
    parser.add_argument("--D",           type=int,   default=D_DEFAULT)
    parser.add_argument("--causal",      action="store_true")
    parser.add_argument("--annotate",    action="store_true",
                        help="Print exact values above each bar.")
    parser.add_argument("--warmup-ms",   type=float, default=25.0)
    parser.add_argument("--timed-ms",    type=float, default=100.0)
    parser.add_argument("--cuda-graph",  action="store_true")
    parser.add_argument("--no-l2-flush", action="store_true")
    parser.add_argument("--out-dir",     type=str,   default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available."); sys.exit(1)

    kernels = list(dict.fromkeys(args.kernels))
    if "sdpa" not in kernels:
        kernels.append("sdpa")

    x_values = args.n_values if args.sweep_axis == "N" else args.b_values
    cold_l2  = not args.no_l2_flush
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = (
        Path(args.out_dir) if args.out_dir
        else PROJECT_ROOT / "benchmarks" / "reports" / "plots"
    )

    results = run_all(
        kernels        = kernels,
        sweep_axis     = args.sweep_axis,
        x_values       = x_values,
        B_fixed        = args.b_fixed,
        N_fixed        = args.n_fixed,
        D              = args.D,
        causal         = args.causal,
        cold_l2        = cold_l2,
        use_cuda_graph = args.cuda_graph,
        warmup_ms      = args.warmup_ms,
        timed_ms       = args.timed_ms,
    )

    build_figures(
        results    = results,
        kernels    = kernels,
        sweep_axis = args.sweep_axis,
        x_values   = x_values,
        B_fixed    = args.b_fixed,
        N_fixed    = args.n_fixed,
        D          = args.D,
        causal     = args.causal,
        annotate   = args.annotate,
        out_dir    = out_dir,
        ts         = ts,
    )

    print_summary(results, kernels, args.sweep_axis, x_values)


if __name__ == "__main__":
    main()