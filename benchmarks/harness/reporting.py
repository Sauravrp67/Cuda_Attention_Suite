"""Legacy-style reporting for the structured benchmark harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from benchmarks.harness.baselines import (
    default_baseline_kernel,
    kernel_bar_color,
    kernel_label,
    kernel_line_style,
    resolve_kernel_name,
)
from benchmarks.harness.cases import BenchmarkCaseSpec
from benchmarks.utils.hardware_constants import HardwareSpecs, RTX_4050_LAPTOP


COMPARE_METRICS: dict[str, dict[str, Any]] = {
    "latency": {
        "key": "median_ms",
        "ylabel": "Latency (ms)",
        "log_y": True,
        "filename": "latency",
        "title_prefix": "latency",
        "ref_line": None,
    },
    "tflops": {
        "key": "achieved_tflops",
        "ylabel": "Speed (TFLOPs/s)",
        "log_y": False,
        "filename": "tflops",
        "title_prefix": "forward speed",
        "ref_line": "peak_flops_fp32_tflops",
    },
    "bandwidth": {
        "key": "achieved_bw",
        "ylabel": "Achieved BW (GB/s)",
        "log_y": False,
        "filename": "bandwidth",
        "title_prefix": "memory bandwidth",
        "ref_line": "peak_bandwidth_gbs",
    },
}


@dataclass(frozen=True)
class ReportMetadata:
    """Serializable run metadata for reports and JSON artifacts."""

    operation: str
    operation_label: str
    report_style: str
    dtype: str
    sweep_axis: str
    x_values: list[int]
    kernels: list[str]
    baseline_kernel: str
    cold_l2: bool
    use_cuda_graph: bool
    fixed_params: dict[str, Any]
    created_at: str
    run_id: str
    hardware: dict[str, Any]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_report_dirs(
    project_root: Path,
    operation: str,
    run_id: str,
    out_dir: str | None = None,
) -> tuple[Path, Path]:
    report_root = (
        Path(out_dir)
        if out_dir is not None
        else project_root / "benchmarks" / "reports"
    )
    timing_dir = report_root / "timing" / operation / run_id
    plot_dir = report_root / "plots" / operation / run_id
    timing_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return timing_dir, plot_dir


def build_report_metadata(
    case_spec: BenchmarkCaseSpec,
    report_style: str,
    dtype: str,
    sweep_axis: str,
    x_values: Iterable[int],
    kernels: list[str],
    fixed_params: Mapping[str, Any],
    cold_l2: bool,
    use_cuda_graph: bool,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
) -> ReportMetadata:
    baseline = default_baseline_kernel(case_spec.operation)
    merged_params = dict(case_spec.default_params)
    merged_params.update(dict(fixed_params))
    run_id = timestamp()
    return ReportMetadata(
        operation=case_spec.operation,
        operation_label=case_spec.operation_label,
        report_style=report_style,
        dtype=dtype,
        sweep_axis=sweep_axis,
        x_values=list(x_values),
        kernels=[resolve_kernel_name(case_spec.operation, k) for k in kernels],
        baseline_kernel=baseline,
        cold_l2=cold_l2,
        use_cuda_graph=use_cuda_graph,
        fixed_params=merged_params,
        created_at=datetime.now().isoformat(),
        run_id=run_id,
        hardware={
            "name": hardware.name,
            "architecture": hardware.architecture,
            "compute_capability": hardware.compute_capability,
            "peak_flops_fp32_tflops": hardware.peak_flops_fp32_tflops,
            "peak_bandwidth_gbs": hardware.peak_bandwidth_gbs,
            "ridge_point": hardware.ridge_point,
            "num_sms": hardware.num_sms,
            "l2_cache_bytes": hardware.l2_cache_bytes,
            "vram_gb": hardware.vram_gb,
        },
    )


def save_results_json(
    metadata: ReportMetadata,
    records: list[dict],
    out_path: Path,
) -> Path:
    payload = {
        **asdict(metadata),
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _records_by_kernel(
    case_spec: BenchmarkCaseSpec, records: list[dict]
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        kernel = resolve_kernel_name(case_spec.operation, record["kernel"])
        grouped.setdefault(kernel, []).append(record)
    return grouped


def _record_lookup(
    case_spec: BenchmarkCaseSpec, records: list[dict], axis_key: str
) -> dict[str, dict[int, dict]]:
    lookup: dict[str, dict[int, dict]] = {}
    for record in records:
        if "error" in record:
            continue
        kernel = resolve_kernel_name(case_spec.operation, record["kernel"])
        lookup.setdefault(kernel, {})[int(record[axis_key])] = record
    return lookup


def _speedup_map(
    case_spec: BenchmarkCaseSpec, records: list[dict], sweep_axis: str, baseline_kernel: str
) -> dict[str, dict[int, float]]:
    axis_key = case_spec.get_axis(sweep_axis).parameter
    by_kernel = _record_lookup(case_spec, records, axis_key)
    baseline = by_kernel.get(baseline_kernel, {})
    speedups: dict[str, dict[int, float]] = {}
    for kernel, kernel_records in by_kernel.items():
        if kernel == baseline_kernel:
            continue
        speedups[kernel] = {}
        for x_value, record in kernel_records.items():
            baseline_record = baseline.get(x_value)
            if baseline_record is None:
                continue
            speedups[kernel][x_value] = (
                baseline_record["median_ms"] / record["median_ms"]
                if record["median_ms"] > 0
                else float("nan")
            )
    return speedups


def _sorted_kernel_records(
    case_spec: BenchmarkCaseSpec, records: list[dict], kernel: str, sweep_axis: str
) -> list[dict]:
    axis_key = case_spec.get_axis(sweep_axis).parameter
    return sorted(
        (
            record
            for record in records
            if "error" not in record
            and resolve_kernel_name(case_spec.operation, record["kernel"]) == kernel
        ),
        key=lambda record: int(record[axis_key]),
    )


def _format_tick_labels(tick_format: str, values: Iterable[int]) -> list[str]:
    if tick_format == "seq_k":
        labels = []
        for value in values:
            if value >= 1024 and value % 1024 == 0:
                labels.append(f"{value // 1024}k")
            else:
                labels.append(str(value))
        return labels
    return [str(value) for value in values]


def print_legacy_banner(
    case_spec: BenchmarkCaseSpec,
    metadata: ReportMetadata,
    report_style: str,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
) -> None:
    context = case_spec.format_context(
        metadata.sweep_axis, metadata.fixed_params, metadata.dtype
    )
    print()
    print("=" * 95)
    if report_style == "sweep":
        print(f"  {case_spec.suite_label} -- Benchmark Run")
    else:
        print(
            f"  {case_spec.operation_label} Compare Run  |  "
            f"{hardware.name}  ({hardware.compute_capability})"
        )
    print(
        f"  Hardware : {hardware.name}  ({hardware.compute_capability})"
    )
    print(
        f"  Peak     : {hardware.peak_flops_fp32_tflops} TFLOPS FP32  |  "
        f"{hardware.peak_bandwidth_gbs} GB/s"
    )
    print(
        f"  Sweep    : {context['banner_desc']}  dtype={metadata.dtype}  "
        f"cold_l2={metadata.cold_l2}"
    )
    print(f"  Kernels  : {metadata.kernels}")
    print(f"  {metadata.sweep_axis} values : {metadata.x_values}")
    print("=" * 95)
    header = (
        f"  {'Run':>6}  {'Kernel':<20}  {metadata.sweep_axis:>5}  "
        f"{'median_ms':>10}  {'TFLOPS':>9}  {'BW GB/s':>9}  "
        f"{'AI FLOPs/B':>11}  {'SOL_TF%':>8}  {'SOL_BW%':>8}  "
        f"{'CV%':>6}  {'iters':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))


def print_legacy_progress_row(
    case_spec: BenchmarkCaseSpec,
    event: Any,
) -> None:
    if event.record is None:
        return
    record = event.record
    axis_key = case_spec.get_axis(event.sweep_axis).parameter
    if "error" in record:
        print(
            f"  [{event.index:>2}/{event.total}]  {event.kernel:<20}  "
            f"{event.x_value:>5}  SKIP  {record['error']}  ({event.elapsed_s:.1f}s)"
        )
        return

    print(
        f"  [{event.index:>2}/{event.total}]  "
        f"{event.kernel:<20}  {int(record[axis_key]):>5}  "
        f"{record['median_ms']:>10.4f}  "
        f"{record['achieved_tflops']:>9.5f}  "
        f"{record['achieved_bw']:>9.3f}  "
        f"{record['arithmetic_intensity']:>11.3f}  "
        f"{record['sol_compute_pct']:>8.3f}  "
        f"{record['sol_bw_pct']:>8.3f}  "
        f"{record['cv_pct']:>6.2f}  "
        f"{record['num_iters']:>6}  "
        f"({event.elapsed_s:.1f}s)"
    )


def print_legacy_sweep_summary(
    case_spec: BenchmarkCaseSpec,
    metadata: ReportMetadata,
    records: list[dict],
) -> None:
    baseline = metadata.baseline_kernel
    axis_key = case_spec.get_axis(metadata.sweep_axis).parameter
    lookup = _record_lookup(case_spec, records, axis_key)
    baseline_records = lookup.get(baseline, {})
    print()
    print("  SUMMARY TABLE")
    header = (
        f"  {'Kernel':<20}  {metadata.sweep_axis:>5}  {'median_ms':>10}  "
        f"{'TFLOPS':>8}  {'AI FLOPs/B':>11}  {'SOL%':>7}  {'speedup':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kernel in metadata.kernels:
        for record in _sorted_kernel_records(case_spec, records, kernel, metadata.sweep_axis):
            x_value = int(record[axis_key])
            speedup = "--"
            baseline_record = baseline_records.get(x_value)
            if (
                baseline_record is not None
                and kernel != baseline
                and record["median_ms"] > 0
            ):
                speedup = f"{baseline_record['median_ms'] / record['median_ms']:.3f}x"
            print(
                f"  {kernel:<20}  {x_value:>5}  {record['median_ms']:>10.4f}  "
                f"{record['achieved_tflops']:>8.5f}  "
                f"{record['arithmetic_intensity']:>11.3f}  "
                f"{record['sol_compute_pct']:>6.3f}%  "
                f"{speedup:>8}"
            )
    print()


def print_legacy_compare_summary(
    case_spec: BenchmarkCaseSpec,
    metadata: ReportMetadata,
    records: list[dict],
) -> None:
    axis_key = case_spec.get_axis(metadata.sweep_axis).parameter
    print()
    print("  SUMMARY")
    header = (
        f"  {'Kernel':<20}  {metadata.sweep_axis:>5}  {'median_ms':>10}  "
        f"{'TFLOPS':>9}  {'BW GB/s':>9}  {'AI FLOPs/B':>11}  "
        f"{'SOL_TF%':>8}  {'SOL_BW%':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kernel in metadata.kernels:
        for record in _sorted_kernel_records(case_spec, records, kernel, metadata.sweep_axis):
            print(
                f"  {kernel:<20}  {int(record[axis_key]):>5}  "
                f"{record['median_ms']:>10.4f}  "
                f"{record['achieved_tflops']:>9.5f}  "
                f"{record['achieved_bw']:>9.3f}  "
                f"{record['arithmetic_intensity']:>11.3f}  "
                f"{record['sol_compute_pct']:>8.3f}  "
                f"{record['sol_bw_pct']:>8.3f}"
            )
    print()


def legacy_sweep_report(
    case_spec: BenchmarkCaseSpec,
    metadata: ReportMetadata,
    records: list[dict],
    plot_dir: Path,
    timing_dir: Path,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
    plot_path_override: Path | None = None,
    json_path_override: Path | None = None,
) -> list[Path]:
    context = case_spec.format_context(
        metadata.sweep_axis, metadata.fixed_params, metadata.dtype
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{case_spec.legacy_report_prefix}_sweep{metadata.sweep_axis}_{metadata.run_id}"
    json_path = save_results_json(
        metadata,
        records,
        json_path_override if json_path_override is not None else timing_dir / f"{stem}.json",
    )

    BG_DARK = "#ffffff"
    BG_PANEL = "#f6f8fa"
    GRID_COL = "#d0d7de"
    TEXT_COL = "#1f2328"
    SPINE_COL = "#d0d7de"
    REF_TFLOPS_COL = "#b45309"
    REF_BW_COL = "#1a7f37"
    REF_BASELINE_COL = "#0969da"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)
    axis_spec = case_spec.get_axis(metadata.sweep_axis)
    grouped = _records_by_kernel(case_spec, records)

    def _panel_setup(ax: Any, title: str, ylabel: str) -> None:
        ax.set_facecolor(BG_PANEL)
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(axis_spec.x_label, color=TEXT_COL, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        if axis_spec.log_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda value, _pos: f"{int(value)}")
            )
        ax.grid(True, which="both", color=GRID_COL, lw=0.5, alpha=0.9)
        ax.grid(True, which="minor", color=GRID_COL, lw=0.3, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)

    ax0 = axes[0, 0]
    _panel_setup(ax0, f"Execution Time vs {axis_spec.x_label}", "Time (ms)")
    ax0.set_yscale("log")
    for kernel in metadata.kernels:
        kernel_records = sorted(grouped.get(kernel, []), key=lambda item: item[axis_spec.parameter])
        if not kernel_records:
            continue
        style = kernel_line_style(case_spec.operation, kernel)
        ax0.plot(
            [record[axis_spec.parameter] for record in kernel_records],
            [record["median_ms"] for record in kernel_records],
            color=style["color"],
            marker=style["marker"],
            lw=style["lw"],
            ms=style["ms"],
            ls=style["ls"],
        )

    ax1 = axes[0, 1]
    _panel_setup(ax1, f"Achieved TFLOPS vs {axis_spec.x_label}", "Achieved TFLOPS")
    ax1.set_yscale("log")
    ax1.axhline(
        hardware.peak_flops_fp32_tflops,
        color=REF_TFLOPS_COL,
        lw=2.0,
        ls="--",
        alpha=0.8,
    )
    for kernel in metadata.kernels:
        kernel_records = sorted(grouped.get(kernel, []), key=lambda item: item[axis_spec.parameter])
        if not kernel_records:
            continue
        style = kernel_line_style(case_spec.operation, kernel)
        ax1.plot(
            [record[axis_spec.parameter] for record in kernel_records],
            [record["achieved_tflops"] for record in kernel_records],
            color=style["color"],
            marker=style["marker"],
            lw=style["lw"],
            ms=style["ms"],
            ls=style["ls"],
        )

    ax2 = axes[1, 0]
    _panel_setup(ax2, f"Achieved Memory BW vs {axis_spec.x_label}", "Bandwidth (GB/s)")
    ax2.set_yscale("log")
    ax2.axhline(
        hardware.peak_bandwidth_gbs,
        color=REF_BW_COL,
        lw=2.0,
        ls="--",
        alpha=0.8,
    )
    for kernel in metadata.kernels:
        kernel_records = sorted(grouped.get(kernel, []), key=lambda item: item[axis_spec.parameter])
        if not kernel_records:
            continue
        style = kernel_line_style(case_spec.operation, kernel)
        ax2.plot(
            [record[axis_spec.parameter] for record in kernel_records],
            [record["achieved_bw"] for record in kernel_records],
            color=style["color"],
            marker=style["marker"],
            lw=style["lw"],
            ms=style["ms"],
            ls=style["ls"],
        )

    ax3 = axes[1, 1]
    _panel_setup(
        ax3,
        f"Speedup vs {kernel_label(case_spec.operation, metadata.baseline_kernel)} ({axis_spec.x_label})",
        "Speedup (x)",
    )
    ax3.set_yscale("linear")
    ax3.axhline(
        1.0,
        color=REF_BASELINE_COL,
        lw=1.5,
        ls="--",
        alpha=0.9,
    )
    speedups = _speedup_map(case_spec, records, metadata.sweep_axis, metadata.baseline_kernel)
    if speedups:
        for kernel, kernel_speedups in speedups.items():
            xs = sorted(kernel_speedups)
            style = kernel_line_style(case_spec.operation, kernel)
            ax3.plot(
                xs,
                [kernel_speedups[x_value] for x_value in xs],
                color=style["color"],
                marker=style["marker"],
                lw=style["lw"],
                ms=style["ms"],
                ls="-",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "baseline not available",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            color="#57606a",
            fontsize=10,
        )

    legend_handles = []
    legend_labels = []
    for kernel in metadata.kernels:
        if not grouped.get(kernel):
            continue
        style = kernel_line_style(case_spec.operation, kernel)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                lw=style["lw"],
                ms=style["ms"],
                ls=style["ls"],
            )
        )
        legend_labels.append(kernel_label(case_spec.operation, kernel))
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color=REF_TFLOPS_COL, lw=2.0, ls="--"),
            plt.Line2D([0], [0], color=REF_BW_COL, lw=2.0, ls="--"),
            plt.Line2D([0], [0], color=REF_BASELINE_COL, lw=1.5, ls="--"),
        ]
    )
    legend_labels.extend(
        [
            f"Peak FP32 ({hardware.peak_flops_fp32_tflops} TFLOPS)",
            f"Peak BW ({hardware.peak_bandwidth_gbs} GB/s)",
            f"{kernel_label(case_spec.operation, metadata.baseline_kernel)} (baseline)",
        ]
    )
    fig.legend(
        legend_handles,
        legend_labels,
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

    fig.suptitle(
        f"{case_spec.operation_label} Kernel Benchmarks  --  {hardware.name}  "
        f"({hardware.compute_capability})\n"
        f"{context['fixed_desc']}  dtype={metadata.dtype}",
        color=TEXT_COL,
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    plot_path = plot_path_override if plot_path_override is not None else plot_dir / f"{stem}.png"
    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=BG_DARK,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"\n  Plot saved -> {plot_path}")
    print_legacy_sweep_summary(case_spec, metadata, records)
    return [json_path, plot_path]


def legacy_compare_report(
    case_spec: BenchmarkCaseSpec,
    metadata: ReportMetadata,
    records: list[dict],
    plot_dir: Path,
    timing_dir: Path,
    metrics: Iterable[str] = ("latency",),
    annotate: bool = False,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)
    context = case_spec.format_context(
        metadata.sweep_axis, metadata.fixed_params, metadata.dtype
    )
    stem = f"compare_{case_spec.operation}_{metadata.sweep_axis}_{metadata.run_id}"
    written: list[Path] = [
        save_results_json(metadata, records, timing_dir / f"{stem}.json")
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#111111",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
        }
    )

    axis_spec = case_spec.get_axis(metadata.sweep_axis)
    x_values = list(metadata.x_values)
    active_kernels = [
        kernel
        for kernel in metadata.kernels
        if _sorted_kernel_records(case_spec, records, kernel, metadata.sweep_axis)
    ]
    n_kernels = len(active_kernels)
    x = np.arange(len(x_values))
    bar_width = 0.8 / max(n_kernels, 1)
    offsets = (np.arange(n_kernels) - (n_kernels - 1) / 2) * bar_width
    tick_labels = _format_tick_labels(axis_spec.tick_format, x_values)

    for metric_name in metrics:
        if metric_name not in COMPARE_METRICS:
            raise ValueError(
                f"Unknown compare metric {metric_name!r}. "
                f"Known metrics: {sorted(COMPARE_METRICS)}"
            )
        metric_meta = COMPARE_METRICS[metric_name]
        fig, ax = plt.subplots(figsize=(max(9, len(x_values) * 0.9), 5))
        fig.subplots_adjust(bottom=0.18)

        for idx, kernel in enumerate(active_kernels):
            values = []
            positions = []
            for jdx, x_value in enumerate(x_values):
                candidates = [
                    record
                    for record in _sorted_kernel_records(
                        case_spec, records, kernel, metadata.sweep_axis
                    )
                    if int(record[axis_spec.parameter]) == int(x_value)
                ]
                if not candidates:
                    continue
                values.append(candidates[0][metric_meta["key"]])
                positions.append(x[jdx] + offsets[idx])

            bars = ax.bar(
                positions,
                values,
                width=bar_width * 0.92,
                color=kernel_bar_color(case_spec.operation, kernel),
                label=kernel_label(case_spec.operation, kernel),
                zorder=3,
                linewidth=0.4,
                edgecolor="white",
            )
            if annotate:
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{value:.1f}" if value >= 1 else f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="#333333",
                        rotation=90,
                        clip_on=True,
                    )

        ref_line = metric_meta["ref_line"]
        if isinstance(ref_line, str):
            ref_value = float(metadata.hardware[ref_line])
            ax.axhline(
                ref_value,
                color="#cc4444",
                lw=1.0,
                ls="--",
                zorder=2,
                label=f"HW peak ({ref_value})",
            )

        ax.set_title(
            f"{case_spec.operation_label} {metric_meta['title_prefix']} "
            f"({hardware.name}  ({hardware.compute_capability}))  --  "
            f"{context['compare_config']}",
            pad=10,
        )
        ax.set_ylabel(metric_meta["ylabel"])
        ax.set_xlabel(axis_spec.x_label)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0)
        if metric_meta["log_y"]:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5, len(x_values) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", ncol=1)
        ax.text(
            0.99,
            -0.14,
            f"{context['footnote']}  |  {hardware.name}  |  {metadata.run_id}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#666666",
        )

        filename = (
            f"compare_{metric_meta['filename']}_sweep{metadata.sweep_axis}_"
            f"{context['dim_tag']}_{context['causal_tag']}_{metadata.run_id}.png"
        )
        out_path = plot_dir / filename
        fig.savefig(
            out_path,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        print(f"  Saved -> {out_path}")
        written.append(out_path)

    print_legacy_compare_summary(case_spec, metadata, records)
    return written
