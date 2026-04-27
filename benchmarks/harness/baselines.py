"""Callable kernel adapters and plotting metadata for benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch.nn.functional import scaled_dot_product_attention

from attention_variants.backends.cuda.loader import gemm, naive_attn, tiled_gemm


CaseInputs = Mapping[str, torch.Tensor]
CaseParams = Mapping[str, Any]
KernelFactory = Callable[[CaseInputs, CaseParams], Callable[[], torch.Tensor]]


@dataclass(frozen=True)
class KernelMeta:
    """Display and plotting metadata for a benchmarked kernel."""

    label: str
    line_color: str
    bar_color: str
    marker: str
    line_style: str = "-"
    line_width: float = 2.0
    marker_size: float = 7.0
    role: str = "candidate"


def _attention_naive_factory(
    inputs: CaseInputs, _params: CaseParams
) -> Callable[[], torch.Tensor]:
    return lambda: naive_attn(inputs["q"], inputs["k"], inputs["v"])


def _attention_sdpa_factory(
    inputs: CaseInputs, params: CaseParams
) -> Callable[[], torch.Tensor]:
    q = inputs["q"]
    scale = q.shape[-1] ** -0.5
    is_causal = bool(params.get("causal", False))
    return lambda: scaled_dot_product_attention(
        inputs["q"], inputs["k"], inputs["v"], scale=scale, is_causal=is_causal
    )


def _matmul_naive_factory(
    inputs: CaseInputs, _params: CaseParams
) -> Callable[[], torch.Tensor]:
    return lambda: gemm(inputs["a"], inputs["b"])


def _matmul_torch_factory(
    inputs: CaseInputs, _params: CaseParams
) -> Callable[[], torch.Tensor]:
    return lambda: torch.matmul(inputs["a"], inputs["b"])


def _matmul_tiled_factory(
    inputs: CaseInputs, _params: CaseParams
) -> Callable[[], torch.Tensor]:
    return lambda: tiled_gemm(inputs["a"], inputs["b"])


KERNEL_REGISTRY: dict[str, dict[str, KernelMeta]] = {
    "attention": {
        "naive_attention": KernelMeta(
            label="Naive Attention (CUDA)",
            line_color="#f78166",
            bar_color="#4878CF",
            marker="o",
        ),
        "torch_sdpa": KernelMeta(
            label="PyTorch SDPA",
            line_color="#58a6ff",
            bar_color="#D65F5F",
            marker="*",
            line_style="--",
            line_width=2.5,
            marker_size=10.0,
            role="baseline",
        ),
    },
    "matmul": {
        "naive_matmul": KernelMeta(
            label="Naive GEMM (CUDA)",
            line_color="#ffa657",
            bar_color="#6ACC65",
            marker="s",
        ),
        "tiled_matmul": KernelMeta(
            label="Tiled GEMM (CUDA)",
            line_color="#ff7b72",
            bar_color="#8C613C",
            marker="^",
        ),
        "torch_matmul": KernelMeta(
            label="torch.matmul",
            line_color="#7ee787",
            bar_color="#B47CC7",
            marker="D",
            line_style="--",
            role="baseline",
        ),
    },
}


CALLABLE_REGISTRY: dict[str, dict[str, KernelFactory]] = {
    "attention": {
        "naive_attention": _attention_naive_factory,
        "torch_sdpa": _attention_sdpa_factory,
    },
    "matmul": {
        "naive_matmul": _matmul_naive_factory,
        "tiled_matmul": _matmul_tiled_factory,
        "torch_matmul": _matmul_torch_factory,
    },
}


KERNEL_ALIASES: dict[str, dict[str, str]] = {
    "attention": {
        "naive_v1": "naive_attention",
        "sdpa": "torch_sdpa",
    },
    "matmul": {},
}


ATTENTION_BASELINE_LABELS: dict[str, str] = {
    name: meta.label for name, meta in KERNEL_REGISTRY["attention"].items()
}
MATMUL_BASELINE_LABELS: dict[str, str] = {
    name: meta.label for name, meta in KERNEL_REGISTRY["matmul"].items()
}
PLOT_STYLE: dict[str, dict[str, object]] = {
    name: {
        "color": meta.line_color,
        "marker": meta.marker,
        "lw": meta.line_width,
        "ms": meta.marker_size,
        "ls": meta.line_style,
    }
    for registry in KERNEL_REGISTRY.values()
    for name, meta in registry.items()
}


def resolve_kernel_name(operation: str, kernel_name: str) -> str:
    registry = KERNEL_REGISTRY.get(operation, {})
    if kernel_name in registry:
        return kernel_name
    aliases = KERNEL_ALIASES.get(operation, {})
    if kernel_name in aliases:
        return aliases[kernel_name]
    raise ValueError(f"Unknown {operation} kernel: {kernel_name!r}")


def normalize_kernel_list(
    operation: str,
    kernels: list[str] | tuple[str, ...],
    include_baseline: bool = True,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for kernel_name in kernels:
        canonical = resolve_kernel_name(operation, kernel_name)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)

    if include_baseline:
        baseline = default_baseline_kernel(operation)
        if baseline and baseline not in seen:
            normalized.append(baseline)
    return normalized


def make_case_callable(
    operation: str,
    kernel_name: str,
    inputs: CaseInputs,
    params: CaseParams,
) -> Callable[[], torch.Tensor]:
    canonical = resolve_kernel_name(operation, kernel_name)
    try:
        factory = CALLABLE_REGISTRY[operation][canonical]
    except KeyError as exc:
        raise ValueError(f"Unknown {operation} kernel: {kernel_name!r}") from exc
    return factory(inputs, params)


def kernel_meta(operation: str, kernel_name: str) -> KernelMeta:
    canonical = resolve_kernel_name(operation, kernel_name)
    return KERNEL_REGISTRY[operation][canonical]


def kernel_label(operation: str, kernel_name: str) -> str:
    return kernel_meta(operation, kernel_name).label


def kernel_line_style(operation: str, kernel_name: str) -> dict[str, object]:
    meta = kernel_meta(operation, kernel_name)
    return {
        "color": meta.line_color,
        "marker": meta.marker,
        "lw": meta.line_width,
        "ms": meta.marker_size,
        "ls": meta.line_style,
    }


def kernel_bar_color(operation: str, kernel_name: str) -> str:
    return kernel_meta(operation, kernel_name).bar_color


def default_baseline_kernel(operation: str) -> str:
    for name, meta in KERNEL_REGISTRY.get(operation, {}).items():
        if meta.role == "baseline":
            return name
    return ""


def make_attention_callable(
    kernel_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    return make_case_callable(
        "attention",
        kernel_name,
        {"q": q, "k": k, "v": v},
        {"causal": False},
    )


def make_matmul_callable(
    kernel_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    return make_case_callable("matmul", kernel_name, {"a": a, "b": b}, {})
