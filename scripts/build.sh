#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_LEGACY_PYCUATTENTION="${BUILD_LEGACY_PYCUATTENTION:-OFF}"

LEGACY_BINARY_NAME="cuda_attn_backend.so"
STRUCTURED_TARGET_DIR="${ROOT_DIR}/src/attention_variants/backends/cuda"

symlink_binary() {
    local source_path="$1"
    local target_dir="$2"
    local binary_name="$3"

    mkdir -p "${target_dir}"
    ln -sfv "${source_path}" "${target_dir}/${binary_name}"
}

echo "🛠️ Compiling CUDA extensions for legacy and structured package layouts..."

cmake --fresh -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DATTENTION_VARIANTS_OUTPUT_DIR="${STRUCTURED_TARGET_DIR}" \
    -DATTENTION_VARIANTS_OUTPUT_NAME="attention_variants_cuda" \
    -DBUILD_LEGACY_PYCUATTENTION="${BUILD_LEGACY_PYCUATTENTION}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

if [[ "${BUILD_LEGACY_PYCUATTENTION}" == "ON" ]]; then
    LEGACY_TARGET_BIN_DIR="${ROOT_DIR}/PyCuAttention/kernels/bin"
    symlink_binary "${BUILD_DIR}/lib/${LEGACY_BINARY_NAME}" "${LEGACY_TARGET_BIN_DIR}" "${LEGACY_BINARY_NAME}"
fi

echo "✅ CUDA compilation completed."
if [[ "${BUILD_LEGACY_PYCUATTENTION}" == "ON" ]]; then
    echo "Legacy symlink: ${LEGACY_TARGET_BIN_DIR}/${LEGACY_BINARY_NAME}"
fi
echo "Structured placement: ${STRUCTURED_TARGET_DIR}/attention_variants_cuda.so"
