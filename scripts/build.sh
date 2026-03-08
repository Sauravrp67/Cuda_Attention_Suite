#!/bin/bash
# Exit on error
set -e

echo "Building CUDA Kernels..."

# Build directory
mkdir -p build
cd build

# Run CMake
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
make -j$(nproc)

echo "Build Complete! Shared Object created in build/ directory."