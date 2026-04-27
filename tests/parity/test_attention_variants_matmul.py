import pytest
import torch

from tests.conftest import (
    ATOL_BF16,
    ATOL_FP32,
    MATMUL_SHAPES,
    RTOL_BF16,
    RTOL_FP16,
    RTOL_FP32,
    make_matmul_inputs,
    matmul_reference,
    transposed_matmul_reference,
)

MATMUL_ATOL_FP16 = 4e-2
TILED_MATMUL_SHAPES = MATMUL_SHAPES
TILED_MATMUL_TILE_TAIL_SHAPES = [
    pytest.param((17, 31, 19), id="M17_K31_N19_tile_tail"),
]
TRANSPOSED_MATMUL_CASES = [
    pytest.param(16, 256, 512, False, True, id="M16_K256_N512_ABt"),
    pytest.param(64, 512, 512, True, False, id="M64_K512_N512_AtB"),
    pytest.param(128, 1024, 512, True, True, id="M128_K1024_N512_AtBt"),
]


def _import_structured_gemm():
    from attention_variants.backends.cuda.loader import gemm

    return gemm


def _import_tiled_gemm():
    from attention_variants.backends.cuda.loader import sgemm

    return sgemm


def _make_transposed_operands(
    M: int,
    K: int,
    N: int,
    *,
    dtype: torch.dtype,
    trans_a: bool,
    trans_b: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = make_matmul_inputs(M, K, N, dtype=dtype)
    a_operand = a.t().contiguous() if trans_a else a
    b_operand = b.t().contiguous() if trans_b else b
    return a_operand, b_operand


@pytest.mark.requires_cuda
class TestStructuredNaiveMatmul:
    """Parity and contract tests for the structured naive GEMM backend."""

    def _run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> torch.Tensor:
        gemm = _import_structured_gemm()
        return gemm(a, b, trans_a, trans_b)

    @pytest.mark.parametrize("shape", MATMUL_SHAPES)
    def test_output_shape(self, shape):
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert out.shape == (shape[0], shape[2])

    @pytest.mark.parametrize("shape", MATMUL_SHAPES)
    def test_output_is_finite(self, shape):
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("shape", MATMUL_SHAPES)
    def test_parity_torch_fp32(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float32)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", MATMUL_SHAPES)
    def test_parity_torch_fp16(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("shape", MATMUL_SHAPES[:-1])
    def test_parity_torch_bf16(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.bfloat16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp32(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.float32, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp16(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES[:-1])
    def test_parity_transposed_bf16(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.bfloat16, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    def test_shape_mismatch_raises(self):
        a = torch.randn(16, 32, device="cuda")
        b = torch.randn(31, 64, device="cuda")

        gemm = _import_structured_gemm()
        with pytest.raises(RuntimeError, match="Shape mismatch for GEMM"):
            gemm(a, b)


@pytest.mark.requires_cuda
class TestStructuredTiledMatmul:
    """Parity and contract tests for the structured tiled GEMM backend."""

    def _run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> torch.Tensor:
        tiled_gemm = _import_tiled_gemm()
        return tiled_gemm(a, b, trans_a, trans_b)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_output_shape(self, shape):
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert out.shape == (shape[0], shape[2])

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_output_is_finite(self, shape):
        a, b = make_matmul_inputs(*shape)
        out = self._run(a, b)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_parity_torch_fp32(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float32)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES)
    def test_parity_torch_fp16(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("shape", TILED_MATMUL_SHAPES[:-1])
    def test_parity_torch_bf16(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.bfloat16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    @pytest.mark.parametrize("shape", TILED_MATMUL_TILE_TAIL_SHAPES)
    def test_parity_torch_fp32_tile_tail(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float32)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("shape", TILED_MATMUL_TILE_TAIL_SHAPES)
    def test_parity_torch_fp16_tile_tail(self, shape):
        a, b = make_matmul_inputs(*shape, dtype=torch.float16)
        ref = matmul_reference(a, b)
        out = self._run(a, b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp32(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.float32, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=RTOL_FP32)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES)
    def test_parity_transposed_fp16(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=MATMUL_ATOL_FP16, rtol=RTOL_FP16)

    @pytest.mark.parametrize("M,K,N,trans_a,trans_b", TRANSPOSED_MATMUL_CASES[:-1])
    def test_parity_transposed_bf16(self, M, K, N, trans_a, trans_b):
        a, b = _make_transposed_operands(
            M, K, N, dtype=torch.bfloat16, trans_a=trans_a, trans_b=trans_b
        )
        ref = transposed_matmul_reference(a, b, trans_a, trans_b)
        out = self._run(a, b, trans_a, trans_b)
        torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=RTOL_BF16)

    def test_shape_mismatch_raises(self):
        a = torch.randn(16, 32, device="cuda")
        b = torch.randn(31, 64, device="cuda")

        tiled_gemm = _import_tiled_gemm()
        with pytest.raises(RuntimeError, match="Shape mismatch for GEMM"):
            tiled_gemm(a, b)
