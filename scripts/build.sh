#!/bin/bash
set -e

# Configuration
PACKAGE_NAME="PyCuAttention"
BINARY_NAME="cuda_attn_backend.so"

# The location where Python expects to find the engine
TARGET_BIN_DIR="$(pwd)/$PACKAGE_NAME/kernels/bin"

echo "🛠️ Compiling High-Performance Engine..."

# 1. Build in the root 'build' directory
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# 2. Cleanup and Symlink (The "No Redundancy" Fix)
mkdir -p "$TARGET_BIN_DIR"

# Create a symbolic link: Python sees it in PyCuAttention, but it's physically in build/
# -s: symbolic, -f: force (overwrite existing), -v: verbose
ln -sfv "$(pwd)/build/lib/$BINARY_NAME" "$TARGET_BIN_DIR/$BINARY_NAME"

echo "✅ csrc compilation completed!" 
echo "Binaries at: build/lib/"
echo "SymLink Location at: $TARGET_BIN_DIR"