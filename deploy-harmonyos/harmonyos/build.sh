#!/bin/bash
# Build llama.cpp on HarmonyOS PC (Kirin 9030 CPU-only)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/llama.cpp-master}"
BUILD_DIR="$SRC_DIR/build-release"

echo "=== Building llama.cpp for HarmonyOS (Kirin 9030) ==="

if command -v clang &>/dev/null; then CC=clang; CXX=clang++
elif command -v gcc &>/dev/null; then CC=gcc; CXX=g++
else echo "Error: No compiler found"; exit 1; fi

CORES=$(nproc 2>/dev/null || echo 8)
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake "$SRC_DIR" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_C_FLAGS="-O3 -march=armv8-a" \
    -DCMAKE_CXX_FLAGS="-O3 -march=armv8-a"

cmake --build . -j"$CORES" --target llama-simple llama-bench
echo "=== Build complete: $BUILD_DIR/bin/llama-simple ==="
