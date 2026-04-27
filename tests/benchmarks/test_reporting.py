import json
from dataclasses import replace

import pytest

from benchmarks.harness.cases import ATTENTION_CASE_SPEC
from benchmarks.harness.reporting import (
    build_report_metadata,
    legacy_compare_report,
    legacy_sweep_report,
)


def _sample_records() -> list[dict]:
    return [
        {
            "operation": "attention",
            "benchmark": "attention",
            "kernel": "naive_attention",
            "B": 1,
            "H": 8,
            "N": 64,
            "D": 64,
            "causal": False,
            "dtype": "float32",
            "median_ms": 2.0,
            "mean_ms": 2.1,
            "std_ms": 0.1,
            "min_ms": 1.9,
            "max_ms": 2.2,
            "cv_pct": 4.76,
            "num_iters": 10,
            "flops": 8388608,
            "algo_bytes": 131072,
            "achieved_tflops": 0.004194,
            "achieved_bw": 0.065536,
            "arithmetic_intensity": 64.0,
            "sol_compute_pct": 0.031067,
            "sol_bw_pct": 0.034133,
        },
        {
            "operation": "attention",
            "benchmark": "attention",
            "kernel": "torch_sdpa",
            "B": 1,
            "H": 8,
            "N": 64,
            "D": 64,
            "causal": False,
            "dtype": "float32",
            "median_ms": 4.0,
            "mean_ms": 4.1,
            "std_ms": 0.1,
            "min_ms": 3.9,
            "max_ms": 4.2,
            "cv_pct": 2.43,
            "num_iters": 10,
            "flops": 8388608,
            "algo_bytes": 131072,
            "achieved_tflops": 0.002097,
            "achieved_bw": 0.032768,
            "arithmetic_intensity": 64.0,
            "sol_compute_pct": 0.015533,
            "sol_bw_pct": 0.017067,
        },
        {
            "operation": "attention",
            "benchmark": "attention",
            "kernel": "naive_attention",
            "B": 1,
            "H": 8,
            "N": 128,
            "D": 64,
            "causal": False,
            "dtype": "float32",
            "median_ms": 8.0,
            "mean_ms": 8.2,
            "std_ms": 0.2,
            "min_ms": 7.8,
            "max_ms": 8.4,
            "cv_pct": 2.44,
            "num_iters": 10,
            "flops": 33554432,
            "algo_bytes": 262144,
            "achieved_tflops": 0.004194,
            "achieved_bw": 0.032768,
            "arithmetic_intensity": 128.0,
            "sol_compute_pct": 0.031067,
            "sol_bw_pct": 0.017067,
        },
        {
            "operation": "attention",
            "benchmark": "attention",
            "kernel": "torch_sdpa",
            "B": 1,
            "H": 8,
            "N": 128,
            "D": 64,
            "causal": False,
            "dtype": "float32",
            "median_ms": 16.0,
            "mean_ms": 16.2,
            "std_ms": 0.2,
            "min_ms": 15.8,
            "max_ms": 16.4,
            "cv_pct": 1.23,
            "num_iters": 10,
            "flops": 33554432,
            "algo_bytes": 262144,
            "achieved_tflops": 0.002097,
            "achieved_bw": 0.016384,
            "arithmetic_intensity": 128.0,
            "sol_compute_pct": 0.015533,
            "sol_bw_pct": 0.008533,
        },
    ]


def test_legacy_sweep_report_writes_expected_json_and_speedup(tmp_path, capsys):
    metadata = build_report_metadata(
        case_spec=ATTENTION_CASE_SPEC,
        report_style="sweep",
        dtype="float32",
        sweep_axis="N",
        x_values=[64, 128],
        kernels=["naive_attention", "torch_sdpa"],
        fixed_params={"B": 1, "H": 8, "D": 64, "causal": False},
        cold_l2=True,
        use_cuda_graph=False,
    )

    outputs = legacy_sweep_report(
        case_spec=ATTENTION_CASE_SPEC,
        metadata=metadata,
        records=_sample_records(),
        plot_dir=tmp_path,
        timing_dir=tmp_path,
    )
    captured = capsys.readouterr().out

    assert len(outputs) == 2
    json_path, plot_path = outputs
    assert json_path.name.startswith("attention_benchmark_sweepN_")
    assert plot_path.name.startswith("attention_benchmark_sweepN_")
    assert plot_path.suffix == ".png"
    assert json_path.parent == tmp_path
    assert plot_path.parent == tmp_path

    payload = json.loads(json_path.read_text())
    assert payload["operation"] == "attention"
    assert payload["report_style"] == "sweep"
    assert payload["sweep_axis"] == "N"
    assert payload["baseline_kernel"] == "torch_sdpa"
    assert payload["x_values"] == [64, 128]
    assert payload["kernels"] == ["naive_attention", "torch_sdpa"]
    assert len(payload["records"]) == 4

    assert "SUMMARY TABLE" in captured
    assert "median_ms" in captured
    assert "TFLOPS" in captured
    assert "AI FLOPs/B" in captured
    assert "speedup" in captured
    assert "2.000x" in captured


def test_legacy_compare_report_builds_metric_specific_filenames(tmp_path, capsys):
    metadata = build_report_metadata(
        case_spec=ATTENTION_CASE_SPEC,
        report_style="compare",
        dtype="float32",
        sweep_axis="N",
        x_values=[64, 128],
        kernels=["naive_attention", "torch_sdpa"],
        fixed_params={"B": 1, "H": 8, "D": 64, "causal": False},
        cold_l2=True,
        use_cuda_graph=False,
    )

    outputs = legacy_compare_report(
        case_spec=ATTENTION_CASE_SPEC,
        metadata=metadata,
        records=_sample_records(),
        plot_dir=tmp_path,
        timing_dir=tmp_path,
        metrics=("latency", "tflops"),
        annotate=False,
    )
    captured = capsys.readouterr().out

    names = [path.name for path in outputs]
    assert any(name.startswith("compare_attention_N_") and name.endswith(".json") for name in names)
    assert any(name.startswith("compare_latency_sweepN_B1_D64_noncausal_") for name in names)
    assert any(name.startswith("compare_tflops_sweepN_B1_D64_noncausal_") for name in names)
    assert all(path.parent == tmp_path for path in outputs)
    assert "SUMMARY" in captured
    assert "BW GB/s" in captured
    assert "SOL_TF%" in captured
    assert "SOL_BW%" in captured
