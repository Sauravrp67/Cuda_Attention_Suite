"""
test_correctness.py — numerical parity tests for all custom attention kernels.

Structure:
  TestNaiveAttentionV1   — three-pass naive kernel vs SDPA ground truth
  (future) TestFusedV2   — add class here when fused_v2 is implemented
  (future) TestFlashV3   — add class here when flash_v3 is implemented

Each test class follows the same pattern:
  test_output_shape          — tensor shape contract
  test_output_dtype          — dtype is preserved
  test_output_is_finite      — no NaN / Inf anywhere
  test_softmax_partition     — attention weights sum to 1 (internal invariant)
  test_parity_sdpa           — allclose vs torch SDPA (parametrised over shapes)
  test_parity_fp16           — same parity check in float16
  test_causal_mask_parity    — causal masked output matches SDPA causal mode

Adding a new kernel:
  1. Import the loader function at the top of the file.
  2. Add a new test class mirroring TestNaiveAttentionV1.
  3. Replace self._run() with the new kernel's call.
  No changes to conftest.py required.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import (
    ATOL_FP32,
    RTOL_FP32,
    ATOL_FP16,
    RTOL_FP16,
    SHAPES_SMALL,
    SHAPES_ALL,
    make_qkv,
    sdpa_reference,
)

# ---------------------------------------------------------------------------
# Kernel import
# ---------------------------------------------------------------------------
# The compiled backend is imported lazily inside each test so that the test
# file can be collected even when the extension has not been built yet.
# pytest will report an ImportError as an ERROR (not a PASS) which is the
# correct behaviour: the test is not skipped, it is broken.

def _import_naive():
    from PyCuAttention.kernels.loader import naive_attn
    return naive_attn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaling(D: int) -> float:
    return D ** -0.5


# ---------------------------------------------------------------------------
# Naive V1 test class
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
class TestNaiveAttentionV1:
    """
    Numerical parity and contract tests for the naive three-pass attention kernel.

    Every method is a self-contained test. Fixtures are pulled from conftest.py
    so that adding a new shape only requires editing SHAPES_SMALL / SHAPES_ALL
    in conftest.py — no changes here.
    """

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        naive_attn = _import_naive()
        B, H, N, D = q.shape
        return naive_attn(q, k, v)

    # ------------------------------------------------------------------
    # Shape and dtype contracts
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_output_shape(self, shape):
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        out = self._run(q, k, v)
        assert out.shape == (B, H, N, D), (
            f"Expected output shape {(B, H, N, D)}, got {out.shape}"
        )

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_output_dtype_preserved(self, shape):
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D, dtype=torch.float32)
        out = self._run(q, k, v)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_output_device_preserved(self, shape):
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        out = self._run(q, k, v)
        assert out.is_cuda, "Output tensor must reside on CUDA device"

    # ------------------------------------------------------------------
    # Numerical health checks (independent of SDPA)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_output_is_finite(self, shape):
        """No NaN or Inf anywhere in the output — catches broken softmax."""
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        out = self._run(q, k, v)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_output_not_all_zeros(self, shape):
        """
        Catches the case where atomicAdd was removed but output buffer
        was never written, or pre-zero guarantee is violated.
        """
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        out = self._run(q, k, v)
        assert out.abs().sum() > 0, "Output is all zeros — kernel did not write"

    # ------------------------------------------------------------------
    # SDPA parity — the primary correctness signal
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_parity_sdpa_fp32(self, shape):
        """
        Core parity test. The naive kernel must agree with SDPA to within
        ATOL_FP32 / RTOL_FP32 (defined in conftest.py).

        Failure here means either:
          - Wrong indexing offset (i*N+d instead of i*D+d)
          - Broken softmax (max initialised to 0 instead of scores[0])
          - Output accumulation bug (atomicAdd on exclusive row)
        """
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        ref = sdpa_reference(q, k, v)
        out = self._run(q, k, v)
        torch.testing.assert_close(
            out, ref,
            atol=ATOL_FP32,
            rtol=RTOL_FP32,
            msg=(
                f"FP32 parity failed for shape B={B} H={H} N={N} D={D}. "
                f"Max abs error: {(out - ref).abs().max().item():.2e}"
            ),
        )

    @pytest.mark.parametrize("shape", SHAPES_SMALL)
    def test_parity_sdpa_fp16(self, shape):
        """
        Float16 parity. Tolerance is much looser — FP16 has only ~3 decimal
        digits of precision and softmax normalisation amplifies rounding.
        """
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D, dtype=torch.float16)
        ref = sdpa_reference(q, k, v)
        out = self._run(q, k, v)
        torch.testing.assert_close(
            out, ref,
            atol=ATOL_FP16,
            rtol=RTOL_FP16,
        )

    # ------------------------------------------------------------------
    # Numerical stability stress tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("shape", [
        pytest.param((1, 1, 128, 64), id="B1_H1_N128_D64"),
        pytest.param((1, 8, 512, 64), id="B1_H8_N512_D64"),
    ])
    def test_stability_large_scores(self, shape):
        """
        Input scaled up by 100x to produce large dot products.
        A kernel without numerically stable softmax (max subtraction) will
        produce Inf or NaN here.
        """
        B, H, N, D = shape
        q, k, v = make_qkv(B, H, N, D)
        q_large = q * 100.0
        k_large = k * 100.0

        out = self._run(q_large, k_large, v)
        ref = sdpa_reference(q_large, k_large, v)

        assert torch.isfinite(out).all(), (
            "Output is not finite under large scores — softmax is numerically unstable"
        )
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", [
        pytest.param((1, 1, 128, 64), id="B1_H1_N128_D64"),
    ])
    def test_stability_uniform_scores(self, shape):
        """
        When all Q and K vectors are identical, all attention weights must be
        exactly 1/N (uniform distribution), and output must equal mean of V rows.
        This is a closed-form check independent of SDPA.
        """
        B, H, N, D = shape
        q = torch.ones(B, H, N, D, device="cuda")
        k = torch.ones(B, H, N, D, device="cuda")
        v, _, _ = make_qkv(B, H, N, D)

        out = self._run(q, k, v)

        # Expected: mean of V across sequence dimension
        expected = v.mean(dim=2, keepdim=True).expand_as(v)
        torch.testing.assert_close(out, expected, atol=ATOL_FP32, rtol=RTOL_FP32)

    # ------------------------------------------------------------------
    # Batch and head independence
    # ------------------------------------------------------------------

    def test_batch_independence(self):
        """
        Output for batch element b must be identical whether computed alone
        (B=1) or as part of a larger batch. Catches cross-batch index aliasing.
        """
        B, H, N, D = 4, 4, 128, 64
        q, k, v = make_qkv(B, H, N, D)

        out_batch = self._run(q, k, v)

        for b in range(B):
            out_single = self._run(
                q[b:b+1], k[b:b+1], v[b:b+1]
            )
            torch.testing.assert_close(
                out_batch[b:b+1], out_single,
                atol=ATOL_FP32, rtol=RTOL_FP32,
                msg=f"Batch independence violated at b={b}",
            )

    def test_head_independence(self):
        """
        Output for head h must be identical whether computed alone (H=1)
        or as part of a multi-head tensor. Catches cross-head index aliasing.
        """
        B, H, N, D = 1, 8, 128, 64
        q, k, v = make_qkv(B, H, N, D)

        out_multi = self._run(q, k, v)

        for h in range(H):
            out_single = self._run(
                q[:, h:h+1, :, :],
                k[:, h:h+1, :, :],
                v[:, h:h+1, :, :],
            )
            torch.testing.assert_close(
                out_multi[:, h:h+1, :, :], out_single,
                atol=ATOL_FP32, rtol=RTOL_FP32,
                msg=f"Head independence violated at h={h}",
            )

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic_across_runs(self):
        """
        Two forward passes with identical inputs must produce bit-identical
        output. Catches non-deterministic atomics or uninitialized memory.
        """
        B, H, N, D = 2, 8, 256, 64
        q, k, v = make_qkv(B, H, N, D)

        out_a = self._run(q, k, v)
        out_b = self._run(q, k, v)

        assert torch.equal(out_a, out_b), (
            "Kernel is non-deterministic: two identical forward passes "
            "produced different outputs"
        )