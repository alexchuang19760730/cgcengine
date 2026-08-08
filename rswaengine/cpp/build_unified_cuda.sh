#!/bin/bash
# =============================================================================
# rswaengine CUDA 构建脚本（需要 CUDA Toolkit / nvcc）
# 编译：
#   build_cuda/libcgc_unified_cuda.so —— 含 window_ortho_kv_compressor.cu
#                                         （与原 v4 .cu 对齐的 CUDA kernel）
#                                         + ortho_kda_v4.cpp (CPU path) 符号
# 若无 nvcc，则跳过并提示（CPU 路径见 build_unified.sh）。
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
INC_DIR="${SCRIPT_DIR}/include"
BUILD_DIR="${SCRIPT_DIR}/build_cuda"
mkdir -p "${BUILD_DIR}"

if ! command -v nvcc &> /dev/null; then
    echo "⚠️  未检测到 nvcc，跳过 CUDA 构建。"
    echo "    CPU 路径（libcgc_unified.so）请运行: bash build_unified.sh"
    exit 0
fi

NVCC="nvcc"
CUDA_ARCH="-arch=sm_90"
if [[ "$(uname)" == "Darwin" ]]; then
    CUDA_ARCH="-arch=sm_86"
fi

echo "=============================================="
echo " rswaengine CUDA build"
echo " nvcc: $(which nvcc)"
echo " arch: ${CUDA_ARCH}"
echo "=============================================="

"${NVCC}" ${CUDA_ARCH} -std=c++17 -O3 -Xcompiler "-fPIC" \
    -I"${INC_DIR}" -I"${INC_DIR}/kernels" \
    --compiler-options '-fPIC' \
    -shared \
    "${SRC_DIR}/kernels/window_ortho_kv_compressor.cu" \
    "${SRC_DIR}/kernels/ortho_kda_v4.cpp" \
    -o "${BUILD_DIR}/libcgc_unified_cuda.so"

if [ -f "${BUILD_DIR}/libcgc_unified_cuda.so" ]; then
    SIZE=$(stat -c%s "${BUILD_DIR}/libcgc_unified_cuda.so" 2>/dev/null || stat -f%z "${BUILD_DIR}/libcgc_unified_cuda.so" 2>/dev/null)
    echo ""
    echo "✅ CUDA 库已生成: ${BUILD_DIR}/libcgc_unified_cuda.so (${SIZE} bytes)"
    echo "   含 window_ortho_kv_compressor.cu (feed/rebuild/attention) + ortho_kda_v4.cpp"
else
    echo "❌ CUDA 构建失败"
    exit 1
fi
