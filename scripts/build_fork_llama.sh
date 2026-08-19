#!/bin/bash
set -e

# ============================================================
# CGC Fork: llama.cpp build script
# ============================================================
# CRITICAL: GGML_BLAS=OFF is required — BLAS (Accelerate) causes
# IQ3_XXS/IQ2_S garbled output. Build must match build-flat
# configuration (dc605b4 era) for correctness.
# ============================================================

FORK_DIR="$(cd "$(dirname "$0")/../temp/llama_roadB/llama.cpp-master" && pwd)"
BUILD_DIR="${FORK_DIR}/build"

# --- Configurable flags (override via env) ---
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGML_METAL="${GGML_METAL:-ON}"
GGML_BLAS="${GGML_BLAS:-OFF}"           # MUST be OFF — causes IQ3 garbled output
GGML_ACCELERATE="${GGML_ACCELERATE:-OFF}" # MUST be OFF — same issue
GGML_CPU_REPACK="${GGML_CPU_REPACK:-OFF}" # MUST be OFF —踩 IQ3 tensor boundary
GGML_OPENMP="${GGML_OPENMP:-OFF}"        # OFF for stability
LLAMA_CURL="${LLAMA_CURL:-OFF}"
LLAMA_BUILD_SERVER="${LLAMA_BUILD_SERVER:-OFF}"
LLAMA_BUILD_TESTS="${LLAMA_BUILD_TESTS:-OFF}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
REBUILD="${REBUILD:-1}"
# MTP isolation: compile the MTP staging API (release_context in
# common/speculative.cpp) into libllama-common.  Without this flag the
# MTP-only code is excluded (non-MTP builds).  The llama-simple binary itself
# is still compiled without -DMTP_SUPPORT (see build_prod_binary.sh).
MTP_SUPPORT="${MTP_SUPPORT:-ON}"

echo "=========================================="
echo "CGC Fork Build"
echo "=========================================="
echo "  Source:     $FORK_DIR"
echo "  Build dir:  $BUILD_DIR"
echo "  Build type: $BUILD_TYPE"
echo "  Metal:      $GGML_METAL"
echo "  BLAS:       $GGML_BLAS  (MUST be OFF for IQ3_XXS)"
echo "  Accelerate: $GGML_ACCELERATE  (MUST be OFF for IQ3_XXS)"
echo "  Repack:     $GGML_CPU_REPACK  (MUST be OFF for IQ3_XXS)"
echo "  OpenMP:     $GGML_OPENMP"
echo "  Jobs:       $JOBS"
echo "=========================================="

cd "$FORK_DIR"

if [ "$REBUILD" = "1" ]; then
    rm -rf "$BUILD_DIR"
fi

# Only add the define when ON; "-DMTP_SUPPORT=OFF" would still define the macro.
if [ "$MTP_SUPPORT" = "ON" ]; then
    CXX_FLAGS_MTP="-DMTP_SUPPORT"
else
    CXX_FLAGS_MTP=""
fi

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL="$GGML_METAL" \
    -DGGML_BLAS="$GGML_BLAS" \
    -DGGML_ACCELERATE="$GGML_ACCELERATE" \
    -DGGML_CPU_REPACK="$GGML_CPU_REPACK" \
    -DGGML_OPENMP="$GGML_OPENMP" \
    -DLLAMA_CURL="$LLAMA_CURL" \
    -DLLAMA_BUILD_SERVER="$LLAMA_BUILD_SERVER" \
    -DLLAMA_BUILD_TESTS="$LLAMA_BUILD_TESTS" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS_MTP"

cmake --build "$BUILD_DIR" -j"$JOBS"

echo ""
echo "=========================================="
echo "Build complete: $BUILD_DIR/bin/"
echo "=========================================="
ls -la "$BUILD_DIR/bin/llama-simple" 2>/dev/null || echo "WARNING: llama-simple not found"
