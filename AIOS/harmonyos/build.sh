#!/bin/bash
# ============================================================
# Build llama.cpp for HarmonyOS (Kirin 9030, CPU-only, NEON)
# ============================================================
# Target:    鴻蒙 MateBook 14 (G4AU042K7)
# Arch:      aarch64 (Kirin 9030, Maleoon 935 GPU)
# Features:  expert-cache (CGC), dual-model (Qwen3.6 A3B + MoE)
# Memory:    32GB unified
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/llama.cpp-master}"
BUILD_DIR="$SRC_DIR/build-release"

# --- Configurable flags (override via env) ---
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGML_METAL="${GGML_METAL:-OFF}"       # Kirin 9030 沒有 Metal
GGML_VULKAN="${GGML_VULKAN:-OFF}"     # 暫不啟用
GGML_OPENCL="${GGML_OPENCL:-OFF}"     # 暫不啟用
GGML_BLAS="${GGML_BLAS:-OFF}"         # MUST OFF (IQ3 garbled output)
GGML_ACCELERATE="${GGML_ACCELERATE:-OFF}"
GGML_CPU_REPACK="${GGML_CPU_REPACK:-OFF}" # MUST OFF (IQ3 tensor boundary)
GGML_OPENMP="${GGML_OPENMP:-OFF}"     # OFF for stability
LLAMA_CURL="${LLAMA_CURL:-OFF}"
LLAMA_BUILD_SERVER="${LLAMA_BUILD_SERVER:-OFF}"
LLAMA_BUILD_TESTS="${LLAMA_BUILD_TESTS:-OFF}"
MTP_SUPPORT="${MTP_SUPPORT:-ON}"       # CGC expert-cache + MTP
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
REBUILD="${REBUILD:-1}"

# --- Kirin 9030 NEON/SVE flags ---
# Kirin 9030 支援 ARMv8.2-A + SVE (variable length)
# -march=armv8.2-a 啟用 dotprod/sve，比 armv8-a 快 ~30%
ARCH_FLAGS="-march=armv8.2-a -mtune=cortex-a720"

echo "=========================================="
echo "CGC Fork Build (HarmonyOS / Kirin 9030)"
echo "=========================================="
echo "  Source:     $SRC_DIR"
echo "  Build dir:  $BUILD_DIR"
echo "  Build type: $BUILD_TYPE"
echo "  Metal:      $GGML_METAL (OFF = CPU-only)"
echo "  BLAS:       $GGML_BLAS (MUST OFF)"
echo "  MTP:        $MTP_SUPPORT"
echo "  Arch flags: $ARCH_FLAGS"
echo "  Jobs:       $JOBS"
echo "  Rebuild:    $REBUILD"
echo "=========================================="

cd "$SRC_DIR"

if [ "$REBUILD" = "1" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

# MTP define
if [ "$MTP_SUPPORT" = "ON" ]; then
    CXX_FLAGS_MTP="-DMTP_SUPPORT"
else
    CXX_FLAGS_MTP=""
fi

cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL="$GGML_METAL" \
    -DGGML_VULKAN="$GGML_VULKAN" \
    -DGGML_OPENCL="$GGML_OPENCL" \
    -DGGML_BLAS="$GGML_BLAS" \
    -DGGML_ACCELERATE="$GGML_ACCELERATE" \
    -DGGML_CPU_REPACK="$GGML_CPU_REPACK" \
    -DGGML_OPENMP="$GGML_OPENMP" \
    -DLLAMA_CURL="$LLAMA_CURL" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS="$LLAMA_BUILD_TESTS" \
    -DLLAMA_BUILD_SERVER="$LLAMA_BUILD_SERVER" \
    -DCMAKE_C_FLAGS="-O3 $ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="-O3 $ARCH_FLAGS $CXX_FLAGS_MTP"

cmake --build . -j"$JOBS" --target llama-simple llama-bench

echo "=========================================="
echo "Build complete!"
echo "  llama-simple:  $BUILD_DIR/bin/llama-simple"
echo "  llama-bench:   $BUILD_DIR/bin/llama-bench"
echo "=========================================="
