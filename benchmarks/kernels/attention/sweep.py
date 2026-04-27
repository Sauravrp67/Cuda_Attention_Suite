"""Canonical attention benchmark entrypoint with legacy-style reporting."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.harness.baselines import normalize_kernel_list
from benchmarks.harness.cases import ATTENTION_CASE_SPEC
from benchmarks.harness.reporting import (
    build_report_metadata,
    ensure_report_dirs,
    legacy_compare_report,
    legacy_sweep_report,
    print_legacy_banner,
    print_legacy_progress_row,
)
from benchmarks.harness.runner import ProgressEvent, run_case


DEFAULT_COMPARE_METRICS = ("latency",)


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype: {name!r}") from exc


def _parser() -> argparse.ArgumentParser:
    case_spec = ATTENTION_CASE_SPEC
    parser = argparse.ArgumentParser(
        description="Canonical attention benchmark runner with legacy-style reports."
    )
    parser.add_argument("--kernels", nargs="+", default=list(case_spec.default_kernels))
    parser.add_argument("--report-style", choices=["sweep", "compare", "both"], default="both")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--sweep-axis",
        choices=sorted(case_spec.supported_sweep_axes),
        default="N",
    )
    parser.add_argument(
        "--N",
        nargs="+",
        type=int,
        default=list(case_spec.default_axis_values("N")),
        dest="n_values",
        help="Sequence-length sweep values when --sweep-axis N.",
    )
    parser.add_argument(
        "--B",
        nargs="+",
        type=int,
        default=list(case_spec.default_axis_values("B")),
        dest="b_values",
        help="Batch-size sweep values when --sweep-axis B.",
    )
    parser.add_argument("--B-fixed", type=int, default=int(case_spec.default_params["B"]))
    parser.add_argument("--N-fixed", type=int, default=int(case_spec.default_params["N"]))
    parser.add_argument("--H", type=int, default=int(case_spec.default_params["H"]))
    parser.add_argument("--D", type=int, default=int(case_spec.default_params["D"]))
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--compare-metrics",
        nargs="+",
        default=list(DEFAULT_COMPARE_METRICS),
        choices=["latency", "tflops", "bandwidth"],
    )
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--warmup-ms", type=float, default=25.0)
    parser.add_argument("--timed-ms", type=float, default=100.0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--no-l2-flush", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-l2", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_attention_benchmark(
    *,
    kernels: Iterable[str],
    report_style: str,
    sweep_axis: str,
    n_values: Iterable[int],
    b_values: Iterable[int],
    B_fixed: int,
    N_fixed: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    causal: bool,
    compare_metrics: Iterable[str],
    annotate: bool,
    warmup_ms: float,
    timed_ms: float,
    cold_l2: bool,
    use_cuda_graph: bool,
    out_dir: str | None = None,
    sweep_plot_path_override: Path | None = None,
    sweep_json_path_override: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    case_spec = ATTENTION_CASE_SPEC
    kernels_normalized = normalize_kernel_list(case_spec.operation, list(kernels))
    if sweep_axis == "N":
        x_values = list(n_values)
        fixed_params = {
            "B": B_fixed,
            "H": H,
            "D": D,
            "causal": causal,
        }
    else:
        x_values = list(b_values)
        fixed_params = {
            "N": N_fixed,
            "H": H,
            "D": D,
            "causal": causal,
        }

    dtype_name = str(dtype).replace("torch.", "")
    metadata = build_report_metadata(
        case_spec=case_spec,
        report_style=report_style,
        dtype=dtype_name,
        sweep_axis=sweep_axis,
        x_values=x_values,
        kernels=kernels_normalized,
        fixed_params=fixed_params,
        cold_l2=cold_l2,
        use_cuda_graph=use_cuda_graph,
    )
    timing_dir, plot_dir = ensure_report_dirs(
        PROJECT_ROOT,
        case_spec.operation,
        metadata.run_id,
        out_dir,
    )
    print_legacy_banner(
        case_spec=case_spec,
        metadata=metadata,
        report_style="compare" if report_style == "compare" else "sweep",
    )

    def _progress(event: ProgressEvent) -> None:
        print_legacy_progress_row(case_spec, event)

    records = run_case(
        case_spec=case_spec,
        kernels=kernels_normalized,
        sweep_axis=sweep_axis,
        x_values=x_values,
        fixed_params=fixed_params,
        dtype=dtype,
        warmup_ms=warmup_ms,
        timed_ms=timed_ms,
        cold_l2=cold_l2,
        use_cuda_graph=use_cuda_graph,
        progress_callback=_progress,
    )
    print("=" * 95)

    outputs: list[Path] = []
    if report_style in ("sweep", "both"):
        outputs.extend(
            legacy_sweep_report(
                case_spec=case_spec,
                metadata=replace(metadata, report_style="sweep"),
                records=records,
                plot_dir=plot_dir,
                timing_dir=timing_dir,
                plot_path_override=sweep_plot_path_override,
                json_path_override=sweep_json_path_override,
            )
        )
    if report_style in ("compare", "both"):
        outputs.extend(
            legacy_compare_report(
                case_spec=case_spec,
                metadata=replace(metadata, report_style="compare"),
                records=records,
                plot_dir=plot_dir,
                timing_dir=timing_dir,
                metrics=compare_metrics,
                annotate=annotate,
            )
        )

    return {
        "metadata": metadata,
        "records": records,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    use_cuda_graph = args.cuda_graph or args.use_cuda_graph
    cold_l2 = True if args.cold_l2 else not args.no_l2_flush
    dtype = _dtype_from_name(args.dtype)
    return run_attention_benchmark(
        kernels=args.kernels,
        report_style=args.report_style,
        sweep_axis=args.sweep_axis,
        n_values=args.n_values,
        b_values=args.b_values,
        B_fixed=args.B_fixed,
        N_fixed=args.N_fixed,
        H=args.H,
        D=args.D,
        dtype=dtype,
        causal=args.causal,
        compare_metrics=args.compare_metrics,
        annotate=args.annotate,
        warmup_ms=args.warmup_ms,
        timed_ms=args.timed_ms,
        cold_l2=cold_l2,
        use_cuda_graph=use_cuda_graph,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
