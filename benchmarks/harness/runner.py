"""Generic benchmark runner for case-driven workloads."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from benchmarks.harness.baselines import make_case_callable, resolve_kernel_name
from benchmarks.harness.cases import BenchmarkCaseSpec
from benchmarks.utils.hardware_constants import HardwareSpecs, RTX_4050_LAPTOP
from benchmarks.utils.timer import bench_gpu_time


CaseParams = Mapping[str, Any]
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    """Per-run event emitted by the generic benchmark runner."""

    index: int
    total: int
    kernel: str
    sweep_axis: str
    x_value: int
    params: dict[str, Any]
    record: dict | None
    elapsed_s: float


def _timing_record(times: list[float]) -> dict[str, float]:
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    return {
        "median_ms": round(float(np.median(times)), 6),
        "mean_ms": round(mean_ms, 6),
        "std_ms": round(std_ms, 6),
        "min_ms": round(float(np.min(times)), 6),
        "max_ms": round(float(np.max(times)), 6),
        "cv_pct": round(float(std_ms / mean_ms * 100.0), 6) if mean_ms else 0.0,
        "num_iters": len(times),
    }


def _measure(
    fn: Callable[[], torch.Tensor],
    warmup_ms: float,
    timed_ms: float,
    cold_l2: bool,
    use_cuda_graph: bool,
) -> tuple[torch.Tensor, list[float]]:
    out = fn()
    torch.cuda.synchronize()
    times = bench_gpu_time(
        fn,
        warmup_time_ms=warmup_ms,
        timed_time_ms=timed_ms,
        cold_l2_cache=cold_l2,
        use_cuda_graph=use_cuda_graph,
    )
    return out, times


def derive_performance_metrics(
    flops: float,
    algo_bytes: float,
    median_ms: float,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
) -> dict[str, float]:
    """Derive roofline-style metrics from analytical accounting and timing."""

    median_s = median_ms * 1e-3
    if median_s <= 0:
        raise ValueError(f"median_ms must be positive, got {median_ms!r}")

    achieved_tflops = flops / median_s / 1e12
    achieved_bw = algo_bytes / median_s / 1e9
    arithmetic_intensity = flops / algo_bytes if algo_bytes else float("inf")
    sol_compute_pct = (
        achieved_tflops / hardware.peak_flops_fp32_tflops * 100.0
        if hardware.peak_flops_fp32_tflops
        else 0.0
    )
    sol_bw_pct = (
        achieved_bw / hardware.peak_bandwidth_gbs * 100.0
        if hardware.peak_bandwidth_gbs
        else 0.0
    )
    return {
        "achieved_tflops": round(float(achieved_tflops), 6),
        "achieved_bw": round(float(achieved_bw), 6),
        "arithmetic_intensity": round(float(arithmetic_intensity), 6),
        "sol_compute_pct": round(float(sol_compute_pct), 6),
        "sol_bw_pct": round(float(sol_bw_pct), 6),
    }


def benchmark_case_once(
    case_spec: BenchmarkCaseSpec,
    kernel_name: str,
    params: dict[str, Any],
    dtype: torch.dtype,
    warmup_ms: float,
    timed_ms: float,
    cold_l2: bool,
    use_cuda_graph: bool,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
    device: str = "cuda",
) -> dict:
    """Benchmark a single kernel/shape combination for a case spec."""

    canonical_kernel = resolve_kernel_name(case_spec.operation, kernel_name)
    try:
        inputs = case_spec.input_builder(params, dtype, device)
        fn = make_case_callable(case_spec.operation, canonical_kernel, inputs, params)
        out, times = _measure(fn, warmup_ms, timed_ms, cold_l2, use_cuda_graph)
        case_spec.output_validator(out, inputs, params)
        timing = _timing_record(times)
        analytics = case_spec.metric_model.account(params, dtype)
        derived = derive_performance_metrics(
            flops=float(analytics["flops"]),
            algo_bytes=float(analytics["algo_bytes"]),
            median_ms=timing["median_ms"],
            hardware=hardware,
        )
    except NotImplementedError as exc:
        return {
            "operation": case_spec.operation,
            "benchmark": case_spec.operation,
            "kernel": canonical_kernel,
            "dtype": str(dtype).replace("torch.", ""),
            **{k: v for k, v in params.items() if isinstance(v, (int, float, bool, str))},
            "error": f"not_implemented: {exc}",
        }
    except Exception as exc:
        return {
            "operation": case_spec.operation,
            "benchmark": case_spec.operation,
            "kernel": canonical_kernel,
            "dtype": str(dtype).replace("torch.", ""),
            **{k: v for k, v in params.items() if isinstance(v, (int, float, bool, str))},
            "error": str(exc),
        }

    return {
        "operation": case_spec.operation,
        "benchmark": case_spec.operation,
        "kernel": canonical_kernel,
        **{k: v for k, v in params.items() if isinstance(v, (int, float, bool, str))},
        "dtype": str(dtype).replace("torch.", ""),
        "output_shape": list(out.shape),
        "flops": int(analytics["flops"]),
        "algo_bytes": int(analytics["algo_bytes"]),
        **timing,
        **derived,
    }


def run_case(
    case_spec: BenchmarkCaseSpec,
    kernels: list[str] | tuple[str, ...],
    sweep_axis: str,
    x_values: list[int] | tuple[int, ...],
    fixed_params: dict[str, Any],
    dtype: torch.dtype,
    warmup_ms: float,
    timed_ms: float,
    cold_l2: bool,
    use_cuda_graph: bool,
    hardware: HardwareSpecs = RTX_4050_LAPTOP,
    device: str = "cuda",
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Run a full sweep for one benchmark case and return flat records."""

    axis_spec = case_spec.get_axis(sweep_axis)
    total = len(kernels) * len(x_values)
    records: list[dict] = []
    done = 0

    for kernel_name in kernels:
        canonical_kernel = resolve_kernel_name(case_spec.operation, kernel_name)
        for x_value in x_values:
            params = dict(case_spec.default_params)
            params.update(fixed_params)
            params[axis_spec.parameter] = x_value
            t0 = time.perf_counter()
            record = benchmark_case_once(
                case_spec=case_spec,
                kernel_name=canonical_kernel,
                params=params,
                dtype=dtype,
                warmup_ms=warmup_ms,
                timed_ms=timed_ms,
                cold_l2=cold_l2,
                use_cuda_graph=use_cuda_graph,
                hardware=hardware,
                device=device,
            )
            elapsed_s = time.perf_counter() - t0
            records.append(record)
            done += 1
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        index=done,
                        total=total,
                        kernel=canonical_kernel,
                        sweep_axis=sweep_axis,
                        x_value=x_value,
                        params=params,
                        record=record,
                        elapsed_s=elapsed_s,
                    )
                )

    return records
