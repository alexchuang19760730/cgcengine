#!/bin/bash
# =============================================================================
# CGC Ortho KDA v4 CUDA 编译脚本
# =============================================================================
# 用途: 编译 KDA v4 CUDA 实现
# 需要: CUDA Toolkit (nvcc)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_cuda"
SRC_DIR="$SCRIPT_DIR/src"
INCLUDE_DIR="$SCRIPT_DIR/include"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     CGC Ortho KDA v4 CUDA 编译脚本                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

if ! command -v nvcc &> /dev/null; then
    echo "❌ 未找到 nvcc，请安装 CUDA Toolkit"
    exit 1
fi

echo "✅ CUDA 编译器: $(which nvcc)"
nvcc --version

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CUDA_ARCH="-arch=sm_90"
if [[ "$(uname)" == "Darwin" ]]; then
    CUDA_ARCH="-arch=sm_86"
fi

echo "🔧 编译 KDA v4 CUDA 内核..."

nvcc \
    $CUDA_ARCH \
    -std=c++17 \
    -O3 \
    -Xcompiler "-fPIC" \
    -I"$INCLUDE_DIR" \
    -I"$INCLUDE_DIR/kernels" \
    -lineinfo \
    --ptx \
    "$SRC_DIR/kernels/ortho_kda_v4.cpp" \
    -o ortho_kda_v4_cuda.o

echo "✅ CUDA 目标文件: $BUILD_DIR/ortho_kda_v4_cuda.o"
echo "✅ PTX 代码已生成"
