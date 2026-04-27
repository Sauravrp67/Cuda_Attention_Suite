#!/usr/bin/env python3
"""Harness-backed target process for Nsight Systems tracing."""

from __future__ import annotations

import argparse
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace target for Nsight Systems with optional warmup/trace loops."
    )
    add_case_target_arguments(parser)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    return parser


def _nvtx_push(label: str) -> None:
    try:
        torch.cuda.nvtx.range_push(label)
    except Exception:
        pass


def _nvtx_pop() -> None:
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    context = build_case_context(args)
    fn = context["fn"]

    _nvtx_push("warmup")
    for _ in range(max(args.warmup_iters, 0)):
        fn()
    torch.cuda.synchronize()
    _nvtx_pop()

    last_out = None
    _nvtx_push("profiled_trace")
    for _ in range(max(args.iters, 1)):
        last_out = fn()
    torch.cuda.synchronize()
    _nvtx_pop()

    if last_out is None:
        raise RuntimeError("NSYS target did not execute any iterations.")
    validate_case_output(context["case_spec"], last_out, context)

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
        f"warmup_iters={args.warmup_iters} "
        f"iters={args.iters} "
        f"output_shape={list(last_out.shape)} DONE"
    )


if __name__ == "__main__":
    main()
