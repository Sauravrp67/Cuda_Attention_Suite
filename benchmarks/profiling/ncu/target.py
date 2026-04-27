#!/usr/bin/env python3
"""Minimal Nsight Compute target application backed by harness case specs."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.profiling.common import (
    add_case_target_arguments,
    build_case_context,
    validate_case_output,
)


def _find_stack_size() -> int | None:
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        return None
    stack_size = ctypes.c_size_t()
    cuda_limit_stack_size = 0
    status = cudart.cudaDeviceGetLimit(
        ctypes.byref(stack_size), cuda_limit_stack_size
    )
    if status != 0:
        return None
    return int(stack_size.value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-launch profiler target for Nsight Compute. "
            "Do not add timing or warmup loops here."
        )
    )
    add_case_target_arguments(parser)
    parser.add_argument("--print-stack-limit", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    context = build_case_context(args)
    torch.cuda.synchronize()

    if args.print_stack_limit:
        stack_limit = _find_stack_size()
        if stack_limit is not None:
            print(f"Stack Limit: {stack_limit} bytes")

    out = context["fn"]()
    torch.cuda.synchronize()
    validate_case_output(context["case_spec"], out, context)

    params = context["params"]
    shape_desc = " ".join(
        f"{key}={params[key]}"
        for key in ("B", "H", "N", "D", "M", "K")
        if key in params
    )
    print(
        f"operation={context['case_spec'].operation} "
        f"kernel={context['kernel']} "
        f"{shape_desc} "
        f"dtype={context['dtype_name']} "
        f"output_shape={list(out.shape)} DONE"
    )


if __name__ == "__main__":
    main()
