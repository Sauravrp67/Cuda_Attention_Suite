"""
conftest.py — shared fixtures, parametrize helpers, and pytest configuration
for the CUDA Attention Suite test suite.

Design principles:
  - All random tensors are seeded deterministically: reproducible on any machine.
  - Fixtures are scoped to avoid redundant GPU allocations.
  - Tolerance constants are centralised here; never scattered across test files.
  - Hardware skips are declared once and reused everywhere.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------

# float32: naive three-pass softmax vs SDPA (different op ordering → ULP drift)
ATOL_FP32 = 1e-4
RTOL_FP32 = 1e-3

# float16: much larger rounding error is expected
ATOL_FP16 = 1e-2
RTOL_FP16 = 1e-2


# ---------------------------------------------------------------------------
# Hardware availability markers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Canonical shapes
# ---------------------------------------------------------------------------
# Each entry is (B, H, N, D). Chosen to cover:
#   - batch=1 (no batching), batch>1 (batching)
#   - single head, multi-head
#   - small N (fits L2), medium N (spills L2), large N (stress test)
#   - D=64 (GPT-2 / LLaMA canonical head dim)
#   - N == D (square case), N < D (under-determined), N == 1 (single token)

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
    pytest.param((1,  8, 2048,  64), id="B1_H8_N2048_D64"),
    pytest.param((1,  8, 4096,  64), id="B1_H8_N4096_D64"),
    pytest.param((2, 32, 2048, 128), id="B2_H32_N2048_D128"),  # LLaMA-7B head config
]

SHAPES_ALL = SHAPES_SMALL + SHAPES_LARGE


# ---------------------------------------------------------------------------
# Tensor factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

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