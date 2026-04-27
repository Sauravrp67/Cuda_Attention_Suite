import pytest
import torch

from tests.conftest import (
    ATOL_BF16,
    ATOL_FP16,
    ATOL_FP32,
    RTOL_BF16,
    RTOL_FP16,
    RTOL_FP32,
    SHAPES_ALL,
    SHAPES_SMALL,
    make_qkv,
    sdpa_reference,
)

STRUCTURED_ATTENTION_SHAPES = list(SHAPES_SMALL) + [
    pytest.param((1, 8, 2048, 64), id="B1_H8_N2048_D64"),
    pytest.param((1, 8, 4096, 64), id="B1_H8_N4096_D64"),
]


def _import_structured_naive():
    from attention_variants.backends.cuda.loader import naive_attn

    return naive_attn


@pytest.mark.requires_cuda
class TestStructuredNaiveAttention:
    """Parity and contract tests for the structured naive attention backend."""

    def _run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        naive_attn = _import_structured_naive()
        return naive_attn(q, k, v)

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_output_shape(self, shape):
        q, k, v = make_qkv(*shape)
        out = self._run(q, k, v)
        assert out.shape == q.shape

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_output_dtype_preserved(self, shape):
        q, k, v = make_qkv(*shape, dtype=torch.float32)
        out = self._run(q, k, v)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_output_device_preserved(self, shape):
        q, k, v = make_qkv(*shape)
        out = self._run(q, k, v)
        assert out.is_cuda, "Output tensor must reside on CUDA device"

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_output_is_finite(self, shape):
        q, k, v = make_qkv(*shape)
        out = self._run(q, k, v)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_output_not_all_zeros(self, shape):
        q, k, v = make_qkv(*shape)
        out = self._run(q, k, v)
        assert out.abs().sum() > 0, "Output is all zeros — kernel did not write"

    @pytest.mark.parametrize("shape", STRUCTURED_ATTENTION_SHAPES)
    def test_parity_sdpa_fp32(self, shape):
        q, k, v = make_qkv(*shape, dtype=torch.float32)
        ref = sdpa_reference(q, k, v)
        out = self._run(q, k, v)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", SHAPES_SMALL)
    def test_parity_sdpa_fp16(self, shape):
        q, k, v = make_qkv(*shape, dtype=torch.float16)
        ref = sdpa_reference(q, k, v)
        out = self._run(q, k, v)
        torch.testing.assert_close(out, ref, atol=ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("shape", SHAPES_SMALL)
    def test_parity_sdpa_bf16(self, shape):
        q, k, v = make_qkv(*shape, dtype=torch.bfloat16)
        ref = sdpa_reference(q, k, v)
        out = self._run(q, k, v)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    @pytest.mark.parametrize("shape", [
        pytest.param((1, 1, 128, 64), id="B1_H1_N128_D64"),
        pytest.param((1, 8, 512, 64), id="B1_H8_N512_D64"),
    ])
    def test_stability_large_scores(self, shape):
        q, k, v = make_qkv(*shape)
        q_large = q * 100.0
        k_large = k * 100.0

        out = self._run(q_large, k_large, v)
        ref = sdpa_reference(q_large, k_large, v)

        assert torch.isfinite(out).all(), (
            "Output is not finite under large scores — softmax is numerically unstable"
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-1)

    @pytest.mark.parametrize("shape", [
        pytest.param((1, 1, 128, 64), id="B1_H1_N128_D64"),
    ])
    def test_stability_uniform_scores(self, shape):
        B, H, N, D = shape
        q = torch.ones(B, H, N, D, device="cuda")
        k = torch.ones(B, H, N, D, device="cuda")
        v, _, _ = make_qkv(B, H, N, D)

        out = self._run(q, k, v)
        expected = v.mean(dim=2, keepdim=True).expand_as(v)
        torch.testing.assert_close(out, expected, atol=ATOL_FP32, rtol=RTOL_FP32)

    def test_batch_independence(self):
        B, H, N, D = 4, 4, 128, 64
        q, k, v = make_qkv(B, H, N, D)

        out_batch = self._run(q, k, v)

        for b in range(B):
            out_single = self._run(q[b:b+1], k[b:b+1], v[b:b+1])
            torch.testing.assert_close(
                out_batch[b:b+1],
                out_single,
                atol=ATOL_FP32,
                rtol=RTOL_FP32,
                msg=f"Batch independence violated at b={b}",
            )

    def test_head_independence(self):
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
                out_multi[:, h:h+1, :, :],
                out_single,
                atol=ATOL_FP32,
                rtol=RTOL_FP32,
                msg=f"Head independence violated at h={h}",
            )

    def test_deterministic_across_runs(self):
        B, H, N, D = 2, 8, 256, 64
        q, k, v = make_qkv(B, H, N, D)

        out_a = self._run(q, k, v)
        out_b = self._run(q, k, v)

        assert torch.equal(out_a, out_b), (
            "Kernel is non-deterministic: two identical forward passes "
            "produced different outputs"
        )
