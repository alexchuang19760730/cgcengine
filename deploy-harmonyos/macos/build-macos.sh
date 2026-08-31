#!/bin/bash
# ============================================================
# Build and sync the macOS deploy bundle from the current repo
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="${SRC_DIR:-$ROOT_DIR/src/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGML_METAL="${GGML_METAL:-ON}"
LLAMA_BUILD_SERVER="${LLAMA_BUILD_SERVER:-ON}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
REBUILD="${REBUILD:-0}"

echo "=========================================="
echo "CGC Fork Build (macOS / Metal)"
echo "=========================================="
echo "  Source:     $SRC_DIR"
echo "  Build dir:  $BUILD_DIR"
echo "  Output dir: $OUT_DIR"
echo "  Build type: $BUILD_TYPE"
echo "  Metal:      $GGML_METAL"
echo "  Server:     $LLAMA_BUILD_SERVER"
echo "  Jobs:       $JOBS"
echo "  Rebuild:    $REBUILD"
echo "=========================================="

if [ "$REBUILD" = "1" ]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL="$GGML_METAL" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER="$LLAMA_BUILD_SERVER"

cmake --build "$BUILD_DIR" -j"$JOBS" \
    --target llama-simple llama-speculative-simple llama-bench llama-server

copy_bin() {
    local name="$1"
    cp -fL "$BUILD_DIR/bin/$name" "$OUT_DIR/$name"
}

rm -f \
    "$OUT_DIR"/llama-simple \
    "$OUT_DIR"/llama-speculative-simple \
    "$OUT_DIR"/llama-bench \
    "$OUT_DIR"/llama-server \
    "$OUT_DIR"/libggml*.dylib \
    "$OUT_DIR"/libllama*.dylib \
    "$OUT_DIR"/libmtmd*.dylib

# Core executables
copy_bin llama-simple
copy_bin llama-speculative-simple
copy_bin llama-bench
copy_bin llama-server

# Runtime dylibs needed by the mac bundle.
for name in \
    libggml-base.0.19.0.dylib libggml-base.0.dylib libggml-base.dylib \
    libggml-cpu.0.19.0.dylib libggml-cpu.0.dylib libggml-cpu.dylib \
    libggml-metal.0.19.0.dylib libggml-metal.0.dylib libggml-metal.dylib \
    libggml.0.19.0.dylib libggml.0.dylib libggml.dylib \
    libllama-common.0.0.97.dylib libllama-common.0.dylib libllama-common.dylib \
    libllama.0.0.97.dylib libllama.0.dylib libllama.dylib \
    libmtmd.0.0.97.dylib libmtmd.0.dylib libmtmd.dylib \
    libllama-server-impl.dylib; do
    copy_bin "$name"
done

echo "=========================================="
echo "macOS deploy bundle synced!"
echo "  llama-simple:             $OUT_DIR/llama-simple"
echo "  llama-speculative-simple: $OUT_DIR/llama-speculative-simple"
echo "  llama-bench:              $OUT_DIR/llama-bench"
echo "  llama-server:             $OUT_DIR/llama-server"
echo "=========================================="
