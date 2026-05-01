# Attention_Variants

`Attention_Variants` is a CUDA/PyTorch learning repo for experimenting with custom attention and GEMM kernels, validating them against PyTorch references, and benchmarking them with a shared harness.

The current structured Python package lives under `src/attention_variants`, while the CUDA extension is built into `src/attention_variants/backends/cuda/attention_variants_cuda.so`.

## Repo layout

- `csrc/`
  Native CUDA/C++ kernels and pybind bindings.
- `src/attention_variants/`
  Structured Python package and backend loaders.
- `tests/parity/`
  Numerical parity tests against PyTorch references.
- `benchmarks/harness/`
  Shared benchmark case definitions, runner, and reporting.
- `benchmarks/kernels/attention/sweep.py`
  Canonical attention benchmark CLI.
- `benchmarks/kernels/primitives/matmul.py`
  Canonical matmul benchmark CLI.
- `benchmarks/profiling/`
  Nsight Compute and Nsight Systems profiling entrypoints.
- `scripts/build.sh`
  Main extension build script.

## Requirements

This repo assumes:

- Linux
- NVIDIA GPU with CUDA installed
- Python with PyTorch built for CUDA
- `cmake`, `nvcc`, and a working C++ compiler

The build script reads the active Python environment and uses PyTorch’s CMake prefix automatically, so the simplest setup is to activate the environment where `torch.cuda.is_available()` already works.

## Build the CUDA extension

From the repo root:

```bash
bash scripts/build.sh
```

This compiles the structured CUDA extension and places it at:

```bash
src/attention_variants/backends/cuda/attention_variants_cuda.so
```

Optional environment variables:

```bash
BUILD_TYPE=Release bash scripts/build.sh
BUILD_TYPE=Debug bash scripts/build.sh
BUILD_LEGACY_PYCUATTENTION=ON bash scripts/build.sh
```

## Run parity tests

Focused GEMM parity:

```bash
pytest -q tests/parity/test_attention_variants_matmul.py
```

Attention parity:

```bash
pytest -q tests/parity/test_attention_variants_naive_attention.py
```

Full test suite:

```bash
pytest
```

## Run benchmarks

The benchmark harness supports two main workloads:

- `attention`
- `matmul`

Reports are written under:

```bash
benchmarks/reports/timing/<operation>/<run_id>/
benchmarks/reports/plots/<operation>/<run_id>/
```

Each run can emit:

- `.json` raw results
- `.png` plots
- `.txt` final-sweep summary table

You can override the report root with `--out-dir`.

### Attention benchmark

Run the default attention sweep:

```bash
python benchmarks/kernels/attention/sweep.py
```

Example: sweep sequence length `N` for the naive CUDA attention kernel and PyTorch SDPA:

```bash
python benchmarks/kernels/attention/sweep.py \
  --kernels naive_attention torch_sdpa \
  --report-style both \
  --sweep-axis N \
  --N 64 128 256 512 1024 \
  --B-fixed 1 \
  --H 8 \
  --D 64 \
  --dtype float32
```

Example: sweep batch size `B`:

```bash
python benchmarks/kernels/attention/sweep.py \
  --kernels naive_attention torch_sdpa \
  --report-style compare \
  --sweep-axis B \
  --B 1 2 4 8 16 32 \
  --N-fixed 512 \
  --H 32 \
  --D 128 \
  --dtype float16
```

Useful options:

- `--report-style sweep|compare|both`
- `--compare-metrics latency tflops bandwidth`
- `--causal`
- `--warmup-ms <ms>`
- `--timed-ms <ms>`
- `--cuda-graph`
- `--no-l2-flush`
- `--out-dir <path>`

### Matmul benchmark

Run the default matmul sweep:

```bash
python benchmarks/kernels/primitives/matmul.py
```

The current matmul benchmark kernel ids are:

- `naive_matmul`
- `tiled_matmul`
- `coarsened_tiled_matmul`
- `reg1d_tiled_matmul`
- `reg2d_tiled_matmul`
- `torch_matmul`

Example:

```bash
python benchmarks/kernels/primitives/matmul.py \
  --kernels naive_matmul tiled_matmul coarsened_tiled_matmul reg1d_tiled_matmul reg2d_tiled_matmul torch_matmul \
  --report-style both \
  --M 16 32 64 128 256 512 \
  --K 4096 \
  --N 4096 \
  --dtype float32
```

Small smoke run:

```bash
python benchmarks/kernels/primitives/matmul.py \
  --kernels naive_matmul tiled_matmul torch_matmul \
  --report-style compare \
  --M 16 32 \
  --K 32 \
  --N 64 \
  --warmup-ms 1 \
  --timed-ms 1 \
  --no-l2-flush
```

Useful options:

- `--report-style sweep|compare|both`
- `--compare-metrics latency tflops bandwidth`
- `--warmup-ms <ms>`
- `--timed-ms <ms>`
- `--cuda-graph`
- `--no-l2-flush`
- `--out-dir <path>`

## Profiling

The profiling layer is harness-backed and lives under `benchmarks/profiling/`.

Nsight Compute target example for matmul:

```bash
python benchmarks/profiling/ncu/target.py \
  --operation matmul \
  --kernel reg2d_tiled_matmul \
  --M 16 \
  --K 32 \
  --N 64 \
  --dtype float32
```

Nsight Compute shell launcher:

```bash
bash benchmarks/profiling/ncu/run.sh --operation matmul --kernel tiled_matmul --M 256 --K 4096 --N 4096
```

If a new kernel is benchmarked and profiled, it should also have a matching regex entry in `benchmarks/profiling/common.py::PROFILER_KERNEL_REGEX`.

## Quick sanity checks

After changing CUDA kernels:

```bash
bash scripts/build.sh
pytest -q tests/parity/test_attention_variants_matmul.py
```

After changing benchmark/reporting code:

```bash
pytest -q tests/benchmarks/test_reporting.py tests/benchmarks/test_cli_smoke.py
```

## Notes

- The structured backend loader is `src/attention_variants/backends/cuda/loader.py`.
- The benchmark harness uses PyTorch baselines by default for comparison.
- `ManualLatencyVerification.py` is useful for quick one-off timing checks, but the canonical benchmark flows are the CLIs in `benchmarks/kernels/...`.
