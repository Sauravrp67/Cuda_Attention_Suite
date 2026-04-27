"""Case definitions and analytical metric models for benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


CaseParams = Mapping[str, Any]
InputBuilder = Callable[[CaseParams, torch.dtype, str], dict[str, torch.Tensor]]
OutputValidator = Callable[[torch.Tensor, dict[str, torch.Tensor], CaseParams], None]
ContextFormatter = Callable[[str, CaseParams, str], dict[str, str]]
MetricAccountFn = Callable[[CaseParams, torch.dtype], dict[str, float]]


@dataclass(frozen=True)
class SweepAxisSpec:
    """Describes a supported sweep axis for a benchmark case."""

    name: str
    parameter: str
    x_label: str
    default_values: tuple[int, ...]
    log_x: bool = True
    tick_format: str = "plain"


@dataclass(frozen=True)
class MetricModel:
    """Operation-specific FLOP/byte accounting contract."""

    name: str
    account_fn: MetricAccountFn

    def account(self, params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
        metrics = dict(self.account_fn(params, dtype))
        if "flops" not in metrics or "algo_bytes" not in metrics:
            raise ValueError(
                f"Metric model {self.name!r} must provide flops and algo_bytes."
            )
        return metrics


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    """Full benchmark case description for the generic harness."""

    operation: str
    operation_label: str
    suite_label: str
    legacy_report_prefix: str
    default_kernels: tuple[str, ...]
    baseline_kernel: str
    supported_sweep_axes: dict[str, SweepAxisSpec]
    default_params: dict[str, Any]
    input_builder: InputBuilder
    output_validator: OutputValidator
    metric_model: MetricModel
    context_formatter: ContextFormatter

    def get_axis(self, sweep_axis: str) -> SweepAxisSpec:
        if sweep_axis not in self.supported_sweep_axes:
            supported = ", ".join(sorted(self.supported_sweep_axes))
            raise ValueError(
                f"{self.operation} does not support sweep axis {sweep_axis!r}. "
                f"Supported axes: {supported}"
            )
        return self.supported_sweep_axes[sweep_axis]

    def default_axis_values(self, sweep_axis: str) -> tuple[int, ...]:
        return self.get_axis(sweep_axis).default_values

    def format_context(self, sweep_axis: str, params: CaseParams, dtype_str: str) -> dict[str, str]:
        return self.context_formatter(sweep_axis, params, dtype_str)


def _dtype_itemsize(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


def _attention_metric_account(params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
    B = int(params["B"])
    H = int(params["H"])
    N = int(params["N"])
    D = int(params["D"])
    itemsize = _dtype_itemsize(dtype)
    flops = 4 * B * H * N * N * D
    algo_bytes = 4 * B * H * N * D * itemsize
    return {"flops": float(flops), "algo_bytes": float(algo_bytes)}


def _matmul_metric_account(params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    itemsize = _dtype_itemsize(dtype)
    flops = 2 * M * K * N
    algo_bytes = (M * K + K * N + M * N) * itemsize
    return {"flops": float(flops), "algo_bytes": float(algo_bytes)}


def _build_attention_inputs(
    params: CaseParams, dtype: torch.dtype, device: str
) -> dict[str, torch.Tensor]:
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed)
    B = int(params["B"])
    H = int(params["H"])
    N = int(params["N"])
    D = int(params["D"])
    return {
        "q": torch.randn(B, H, N, D, dtype=dtype, device=device),
        "k": torch.randn(B, H, N, D, dtype=dtype, device=device),
        "v": torch.randn(B, H, N, D, dtype=dtype, device=device),
    }


def _validate_attention_output(
    output: torch.Tensor, _inputs: dict[str, torch.Tensor], params: CaseParams
) -> None:
    expected_shape = (
        int(params["B"]),
        int(params["H"]),
        int(params["N"]),
        int(params["D"]),
    )
    if tuple(output.shape) != expected_shape:
        raise AssertionError(
            f"Attention output shape mismatch: got {tuple(output.shape)}, "
            f"expected {expected_shape}"
        )


def _attention_context(
    sweep_axis: str, params: CaseParams, dtype_str: str
) -> dict[str, str]:
    B = int(params["B"])
    H = int(params["H"])
    N = int(params["N"])
    D = int(params["D"])
    causal = bool(params.get("causal", False))
    causal_str = "causal" if causal else "non-causal"
    causal_tag = "causal" if causal else "noncausal"
    if sweep_axis == "N":
        return {
            "banner_desc": f"N sweep  B={B} H={H} D={D}",
            "fixed_desc": f"B={B}",
            "compare_config": f"{causal_str}, B={B}, head dim {D}",
            "footnote": f"B={B}  H={H}  D={D}  dtype={dtype_str}",
            "dim_tag": f"B{B}_D{D}",
            "causal_tag": causal_tag,
        }
    if sweep_axis == "B":
        return {
            "banner_desc": f"B sweep  N={N} H={H} D={D}",
            "fixed_desc": f"N={N}",
            "compare_config": f"{causal_str}, N={N}, head dim {D}",
            "footnote": f"N={N}  H={H}  D={D}  dtype={dtype_str}",
            "dim_tag": f"N{N}_D{D}",
            "causal_tag": causal_tag,
        }
    raise ValueError(f"Unsupported attention sweep axis: {sweep_axis!r}")


def _build_matmul_inputs(
    params: CaseParams, dtype: torch.dtype, device: str
) -> dict[str, torch.Tensor]:
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed)
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    return {
        "a": torch.randn(M, K, dtype=dtype, device=device),
        "b": torch.randn(K, N, dtype=dtype, device=device),
    }


def _validate_matmul_output(
    output: torch.Tensor, _inputs: dict[str, torch.Tensor], params: CaseParams
) -> None:
    expected_shape = (int(params["M"]), int(params["N"]))
    if tuple(output.shape) != expected_shape:
        raise AssertionError(
            f"Matmul output shape mismatch: got {tuple(output.shape)}, "
            f"expected {expected_shape}"
        )


def _matmul_context(
    sweep_axis: str, params: CaseParams, dtype_str: str
) -> dict[str, str]:
    M = int(params["M"])
    K = int(params["K"])
    N = int(params["N"])
    if sweep_axis != "M":
        raise ValueError(f"Unsupported matmul sweep axis: {sweep_axis!r}")
    return {
        "banner_desc": f"M sweep  K={K} N={N}",
        "fixed_desc": f"K={K}, N={N}",
        "compare_config": f"K={K}, N={N}",
        "footnote": f"K={K}  N={N}  dtype={dtype_str}",
        "dim_tag": f"K{K}_N{N}",
        "causal_tag": "none",
    }


def _raise_unimplemented_builder(
    params: CaseParams, dtype: torch.dtype, device: str
) -> dict[str, torch.Tensor]:
    raise NotImplementedError(
        "This benchmark case is only scaffolded for future work and cannot run yet."
    )


def _raise_unimplemented_metrics(params: CaseParams, dtype: torch.dtype) -> dict[str, float]:
    raise NotImplementedError(
        "This benchmark case does not yet define analytical FLOP/byte accounting."
    )


def _placeholder_context(
    sweep_axis: str, params: CaseParams, dtype_str: str
) -> dict[str, str]:
    return {
        "banner_desc": f"{sweep_axis} sweep",
        "fixed_desc": "",
        "compare_config": "placeholder",
        "footnote": f"dtype={dtype_str}",
        "dim_tag": "placeholder",
        "causal_tag": "none",
    }


def _placeholder_case(operation: str, label: str) -> BenchmarkCaseSpec:
    return BenchmarkCaseSpec(
        operation=operation,
        operation_label=label,
        suite_label=f"CUDA {label} Suite",
        legacy_report_prefix=f"{operation}_benchmark",
        default_kernels=(),
        baseline_kernel="",
        supported_sweep_axes={
            "N": SweepAxisSpec(
                name="N",
                parameter="N",
                x_label="Problem Size",
                default_values=(64, 128, 256),
            )
        },
        default_params={},
        input_builder=_raise_unimplemented_builder,
        output_validator=lambda *_args, **_kwargs: None,
        metric_model=MetricModel(
            name=f"{operation}_placeholder", account_fn=_raise_unimplemented_metrics
        ),
        context_formatter=_placeholder_context,
    )


ATTENTION_CASE_SPEC = BenchmarkCaseSpec(
    operation="attention",
    operation_label="Attention",
    suite_label="CUDA Attention Suite",
    legacy_report_prefix="attention_benchmark",
    default_kernels=("naive_attention", "torch_sdpa"),
    baseline_kernel="torch_sdpa",
    supported_sweep_axes={
        "N": SweepAxisSpec(
            name="N",
            parameter="N",
            x_label="Sequence Length N",
            default_values=(2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
            tick_format="seq_k",
        ),
        "B": SweepAxisSpec(
            name="B",
            parameter="B",
            x_label="Batch Size B",
            default_values=(1, 2, 4, 8, 16, 32),
            log_x=True,
        ),
    },
    default_params={"B": 32, "H": 32, "N": 512, "D": 128, "causal": False, "seed": 42},
    input_builder=_build_attention_inputs,
    output_validator=_validate_attention_output,
    metric_model=MetricModel(name="attention", account_fn=_attention_metric_account),
    context_formatter=_attention_context,
)


MATMUL_CASE_SPEC = BenchmarkCaseSpec(
    operation="matmul",
    operation_label="Matmul",
    suite_label="CUDA Matmul Suite",
    legacy_report_prefix="matmul_benchmark",
    default_kernels=("naive_matmul", "tiled_matmul", "torch_matmul"),
    baseline_kernel="torch_matmul",
    supported_sweep_axes={
        "M": SweepAxisSpec(
            name="M",
            parameter="M",
            x_label="Rows M",
            default_values=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096,8192),
        ),
    },
    default_params={"M": 64, "K": 8192, "N": 8192, "seed": 42},
    input_builder=_build_matmul_inputs,
    output_validator=_validate_matmul_output,
    metric_model=MetricModel(name="matmul", account_fn=_matmul_metric_account),
    context_formatter=_matmul_context,
)


BENCHMARK_CASES: dict[str, BenchmarkCaseSpec] = {
    "attention": ATTENTION_CASE_SPEC,
    "matmul": MATMUL_CASE_SPEC,
    "softmax": _placeholder_case("softmax", "Softmax"),
    "online_softmax": _placeholder_case("online_softmax", "Online Softmax"),
    "convolution": _placeholder_case("convolution", "Convolution"),
    "layernorm": _placeholder_case("layernorm", "LayerNorm"),
    "rope": _placeholder_case("rope", "RoPE"),
    "attention_layer": _placeholder_case("attention_layer", "Attention Layer"),
    "model_prefill": _placeholder_case("model_prefill", "Model Prefill"),
    "model_decode": _placeholder_case("model_decode", "Model Decode"),
}


def get_case_spec(operation: str) -> BenchmarkCaseSpec:
    if operation not in BENCHMARK_CASES:
        raise KeyError(f"Unknown benchmark case: {operation!r}")
    return BENCHMARK_CASES[operation]
