from pathlib import Path

import pytest

from benchmarks.kernels.attention.sweep import main as attention_main
from benchmarks.kernels.primitives.matmul import main as matmul_main


@pytest.mark.requires_cuda
def test_attention_cli_both_report_styles_emit_outputs(tmp_path: Path):
    result = attention_main(
        [
            "--kernels",
            "torch_sdpa",
            "--report-style",
            "both",
            "--N",
            "2",
            "8",
            "--B-fixed",
            "1",
            "--H",
            "2",
            "--D",
            "16",
            "--warmup-ms",
            "1",
            "--timed-ms",
            "1",
            "--out-dir",
            str(tmp_path),
            "--no-l2-flush",
        ]
    )

    outputs = [Path(path) for path in result["outputs"]]
    run_id = result["metadata"].run_id
    assert any(path.name.startswith("attention_benchmark_sweepN_") for path in outputs)
    assert any(path.name.startswith("compare_latency_sweepN_B1_D16_noncausal_") for path in outputs)
    assert any(path.suffix == ".json" for path in outputs)
    assert any(path.parent == tmp_path / "timing" / "attention" / run_id for path in outputs if path.suffix == ".json")
    assert all(path.parent == tmp_path / "plots" / "attention" / run_id for path in outputs if path.suffix == ".png")


@pytest.mark.requires_cuda
def test_matmul_cli_both_report_styles_emit_outputs(tmp_path: Path):
    result = matmul_main(
        [
            "--kernels",
            "tiled_matmul",
            "torch_matmul",
            "--report-style",
            "both",
            "--M",
            "16",
            "32",
            "--K",
            "32",
            "--N",
            "64",
            "--warmup-ms",
            "1",
            "--timed-ms",
            "1",
            "--out-dir",
            str(tmp_path),
            "--no-l2-flush",
        ]
    )

    outputs = [Path(path) for path in result["outputs"]]
    run_id = result["metadata"].run_id
    assert any(path.name.startswith("matmul_benchmark_sweepM_") for path in outputs)
    assert any(path.name.startswith("compare_latency_sweepM_K32_N64_none_") for path in outputs)
    assert any(path.suffix == ".json" for path in outputs)
    assert any(path.parent == tmp_path / "timing" / "matmul" / run_id for path in outputs if path.suffix == ".json")
    assert all(path.parent == tmp_path / "plots" / "matmul" / run_id for path in outputs if path.suffix == ".png")
