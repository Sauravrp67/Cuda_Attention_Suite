import pytest
import torch

from benchmarks.harness.cases import ATTENTION_CASE_SPEC, MATMUL_CASE_SPEC
from benchmarks.harness.runner import derive_performance_metrics
from benchmarks.utils.bench_core import HW, attention_bytes, attention_flops


def test_attention_metric_model_matches_legacy_formula():
    params = {"B": 2, "H": 8, "N": 128, "D": 64}
    metrics = ATTENTION_CASE_SPEC.metric_model.account(params, torch.float32)

    assert metrics["flops"] == attention_flops(2, 8, 128, 64)
    assert metrics["algo_bytes"] == attention_bytes(2, 8, 128, 64, 4)

    derived = derive_performance_metrics(
        flops=metrics["flops"],
        algo_bytes=metrics["algo_bytes"],
        median_ms=1.5,
    )
    expected_tflops = metrics["flops"] / (1.5e-3) / 1e12
    expected_bw = metrics["algo_bytes"] / (1.5e-3) / 1e9
    expected_ai = metrics["flops"] / metrics["algo_bytes"]

    assert derived["achieved_tflops"] == pytest.approx(round(expected_tflops, 6))
    assert derived["achieved_bw"] == pytest.approx(round(expected_bw, 6))
    assert derived["arithmetic_intensity"] == pytest.approx(round(expected_ai, 6))
    assert derived["sol_compute_pct"] == pytest.approx(
        round(expected_tflops / HW.peak_flops_fp32_tflops * 100.0, 6)
    )
    assert derived["sol_bw_pct"] == pytest.approx(
        round(expected_bw / HW.peak_bandwidth_gbs * 100.0, 6)
    )


def test_matmul_metric_model_matches_closed_form_formula():
    params = {"M": 128, "K": 1024, "N": 512}
    metrics = MATMUL_CASE_SPEC.metric_model.account(params, torch.float32)

    expected_flops = 2 * 128 * 1024 * 512
    expected_bytes = (128 * 1024 + 1024 * 512 + 128 * 512) * 4

    assert metrics["flops"] == expected_flops
    assert metrics["algo_bytes"] == expected_bytes

    derived = derive_performance_metrics(
        flops=metrics["flops"],
        algo_bytes=metrics["algo_bytes"],
        median_ms=2.0,
    )
    assert derived["arithmetic_intensity"] == pytest.approx(
        round(expected_flops / expected_bytes, 6)
    )
    assert derived["achieved_tflops"] == pytest.approx(
        round(expected_flops / (2.0e-3) / 1e12, 6)
    )


def test_case_registry_exposes_expected_placeholder_cases():
    placeholder_ops = {
        "softmax",
        "online_softmax",
        "convolution",
        "layernorm",
        "rope",
        "attention_layer",
        "model_prefill",
        "model_decode",
    }

    from benchmarks.harness.cases import BENCHMARK_CASES

    assert placeholder_ops.issubset(BENCHMARK_CASES)
    for op in placeholder_ops:
        spec = BENCHMARK_CASES[op]
        with pytest.raises(NotImplementedError):
            spec.metric_model.account({}, torch.float32)


def test_derive_performance_metrics_rejects_non_positive_median():
    with pytest.raises(ValueError, match="median_ms must be positive"):
        derive_performance_metrics(flops=1.0, algo_bytes=1.0, median_ms=0.0)
