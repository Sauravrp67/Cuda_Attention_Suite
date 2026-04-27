"""Shared helpers for profiler launch targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.harness.baselines import make_case_callable, resolve_kernel_name
from benchmarks.harness.cases import BenchmarkCaseSpec, get_case_spec


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

PROFILER_KERNEL_REGEX: dict[tuple[str, str], str] = {
    ("attention", "naive_attention"): "naive_attention_kernel",
    ("matmul", "naive_matmul"): "naive_gemm",
    ("matmul", "tiled_matmul"): "tiled_gemm",
}


def add_case_target_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the shared operation/kernel/shape arguments used by profiler targets."""

    parser.add_argument("--operation", choices=["attention", "matmul"], default="attention")
    parser.add_argument("--kernel", required=True)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPE_MAP),
    )
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    return parser


def build_case_context(args: argparse.Namespace) -> dict[str, object]:
    """Resolve a profiler target namespace into a harness-backed callable."""

    case_spec = get_case_spec(args.operation)
    dtype = DTYPE_MAP[args.dtype]
    params = dict(case_spec.default_params)

    for key in ("B", "H", "N", "D", "M", "K"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if "causal" in params or getattr(args, "causal", False):
        params["causal"] = bool(getattr(args, "causal", False))

    kernel = resolve_kernel_name(case_spec.operation, args.kernel)
    inputs = case_spec.input_builder(params, dtype, "cuda")
    fn = make_case_callable(case_spec.operation, kernel, inputs, params)
    return {
        "case_spec": case_spec,
        "kernel": kernel,
        "dtype": dtype,
        "dtype_name": args.dtype,
        "params": params,
        "inputs": inputs,
        "fn": fn,
    }


def validate_case_output(case_spec: BenchmarkCaseSpec, output: torch.Tensor, context: dict[str, object]) -> None:
    """Apply the case-specific output contract check."""

    case_spec.output_validator(output, context["inputs"], context["params"])


def profiler_kernel_regex(operation: str, kernel: str) -> str | None:
    """Return the default CUDA kernel-name regex for a profiler launch, if known."""

    return PROFILER_KERNEL_REGEX.get((operation, kernel))
