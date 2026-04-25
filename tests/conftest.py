import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Tolerance constants

# float32: naive three-pass softmax vs SDPA (different op ordering → ULP drift)
ATOL_FP32 = 1e-4
RTOL_FP32 = 1e-3

# float16: much larger rounding error is expected
ATOL_FP16 = 5e-3
RTOL_FP16 = 5e-3

# BFloat16: Mantissa bits as fp16 (Actually Fewer 7 mantissa for BFloat16 and 10 for FP16)
# More exponent bits -> Less overflow risk.
# Softmax result stays finite more easily.
# Mantissa Precision is less
ATOL_BF16 = 5e-2
RTOL_BF16 = 5e-2

# Hardware availability markers

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cuda: skip if no CUDA device is available",
    )
    config.addinivalue_line(
        "markers",
        "requires_sm89: skip if GPU is not Ada Lovelace (sm_89)",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    if item.get_closest_marker("requires_sm89"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (8, 9):
            pytest.skip(
                f"Test requires sm_89 (Ada Lovelace), "
                f"got sm_{major}{minor}"
            )

# Canonical shapes

SHAPES_SMALL = [
    pytest.param((1, 1,    1,  64), id="B1_H1_N1_D64"),       # single token
    pytest.param((1, 1,   64,  64), id="B1_H1_N64_D64"),      # N==D
    pytest.param((1, 1,  128,  64), id="B1_H1_N128_D64"),
    pytest.param((1, 8,  128,  64), id="B1_H8_N128_D64"),
    pytest.param((2, 8,  128,  64), id="B2_H8_N128_D64"),
    pytest.param((1, 1,  512,  64), id="B1_H1_N512_D64"),
    pytest.param((1, 8,  512,  64), id="B1_H8_N512_D64"),
    pytest.param((2, 8,  512,  64), id="B2_H8_N512_D64"),
    pytest.param((1, 8, 1024,  64), id="B1_H8_N1024_D64"),
    pytest.param((2, 8, 1024,  64), id="B2_H8_N1024_D64"),
]

SHAPES_LARGE = [
    pytest.param((1, 8, 2048, 64), id="B1_H8_N2048_D64"),
    pytest.param((1, 8, 4096, 64), id="B1_H8_N4096_D64"),
    pytest.param((2, 8, 4240, 64), id="B2_H8_N4320_D64"),      # Test for N > MAX_SEQ_LEN
    pytest.param((2, 32, 2048, 128), id="B2_H32_N2048_D128"),  # LLaMA-7B head config
]

SHAPES_ALL = SHAPES_SMALL + SHAPES_LARGE

MATMUL_SHAPES = [
    pytest.param((16, 256, 512), id="M16_K256_N512"),
    pytest.param((64, 512, 512), id="M64_K512_N512"),
    pytest.param((128, 1024, 512), id="M128_K1024_N512"),
    pytest.param((256, 1024, 1024), id="M256_K1024_N1024"),
]

# Tensor factory

def make_qkv(
    B: int,
    H: int,
    N: int,
    D: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (Q, K, V) tensors of shape [B, H, N, D], seeded deterministically.

    Values are drawn from N(0, 0.02) — same initialisation range as GPT-2 —
    so scores after scaling sit in a numerically well-behaved range.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def _make():
        return torch.randn(B, H, N, D, generator=generator).to(
            dtype=dtype, device=device
        )

    return _make(), _make(), _make()

# Reference implementation

def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Calls torch.nn.functional.scaled_dot_product_attention with scale=1/sqrt(D).
    This is the single ground truth for all parity tests.
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def make_matmul_inputs(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (A, B) tensors for GEMM parity tests with shapes [M, K] and [K, N].
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    a = torch.randn(M, K, generator=generator).to(dtype=dtype, device=device)
    b = torch.randn(K, N, generator=generator).to(dtype=dtype, device=device)
    return a, b


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch reference for the naive GEMM kernel."""
    return torch.matmul(a, b)


def transposed_matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """Torch reference for GEMM parity tests with optional transpose flags."""
    lhs = a.transpose(-2, -1) if trans_a else a
    rhs = b.transpose(-2, -1) if trans_b else b
    return torch.matmul(lhs, rhs)
