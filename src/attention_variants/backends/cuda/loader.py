"""Loader for the structured CUDA extension module."""

from __future__ import annotations

from . import attention_variants_cuda as _backend
from torch.nn.functional import scaled_dot_product_attention


def gemm(A, B, transA: bool = False, transB: bool = False):
    """Structured CUDA wrapper for the naive GEMM reference kernel."""
    return _backend.gemm(A, B, transA, transB)

def sgemm(A, B, transA: bool = False, transB: bool = False):
    """Structured CUDA wrapper for the tiled GEMM reference kernel"""
    return _backend.sgemm(A,B, transA, transB)

def var_sgemm(A,B,transA:bool = False, transB: bool = False):
    """Structured CUDA wrapper for tiled GEMM kernel with non-square TILE Dimension"""
    return _backend.var_sgemm(A, B, transA, transB)

def regtiled2DSgemm(A, B, transA:bool = False, transB: bool = False):
    """Structured CUDA wrapper for 2D register TILED GEMM kernel with non-square TILE dimension"""
    return _backend.reg2DTiledsgemm(A, B, transA, transB)


def naive_attn(q, k, v):
    """Structured CUDA wrapper for the naive SDPA reference kernel."""
    return _backend.naive_attention_fwd(q, k, v)


def sdpa_attn(q, k, v, scaling=None):
    """Torch SDPA baseline for parity tests and benchmarks."""
    scale = scaling if scaling is not None else q.shape[-1] ** -0.5
    return scaled_dot_product_attention(q, k, v, scale=scale)


def fused_attn(*args, **kwargs):
    """Placeholder until the structured fused attention kernel lands."""
    raise NotImplementedError("fused_attn is not implemented in attention_variants yet.")
