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
#
# [CGC 2026-10-06] 版號不再寫死。舊版把三個家族寫成 `*.0.0.97.dylib`，而現在的 build 產出是
# `*.0.0.630.dylib` ⇒ `cp` 失敗 ⇒ `set -e` 中止 —— 而上面的 `rm -f` 已經先把 bundle 刪掉一半
# （2026-10-06 實測：26 個 tracked 檔被刪、16 個被改，驗收 FAIL）。
# 版號是 configure 時由 `git rev-list --count HEAD` 推導的（`cmake/build-info.cmake` 的 BUILD_NUMBER
# → `LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER}`），每次 commit 都可能前進 ⇒ 只能動態解析。
resolve_dylib() {   # $1 = 家族前綴（含版號起頭）；回傳 build/bin 內最新的一個實際檔（跳過 symlink）
    local f hit=""
    for f in "$1"*.dylib; do
        [ -e "$f" ] || continue
        [ -L "$f" ] && continue
        hit="$hit
$(basename "$f")"
    done
    hit="$(printf '%s\n' "$hit" | sed '/^$/d' | sort -V | tail -1)"
    if [ -z "$hit" ]; then
        echo "FAIL: $BUILD_DIR/bin 找不到 $1*.dylib（版號解析失敗；不要把版號寫死）" >&2
        exit 1
    fi
    printf '%s' "$hit"
}

cd "$BUILD_DIR/bin"
BASE_VER="$(resolve_dylib libggml-base.)"
CPU_VER="$(resolve_dylib libggml-cpu.)"
METAL_VER="$(resolve_dylib libggml-metal.)"
GGML_VER="$(resolve_dylib libggml.0.)"
COMMON_VER="$(resolve_dylib libllama-common.0.0.)"
LLAMA_VER="$(resolve_dylib libllama.0.0.)"
MTMD_VER="$(resolve_dylib libmtmd.0.0.)"
cd "$SCRIPT_DIR"
echo "  dylib 版號（動態解析）: base=$BASE_VER cpu=$CPU_VER metal=$METAL_VER ggml=$GGML_VER"
echo "                          common=$COMMON_VER llama=$LLAMA_VER mtmd=$MTMD_VER"

for name in \
    "$BASE_VER" libggml-base.0.dylib libggml-base.dylib \
    "$CPU_VER" libggml-cpu.0.dylib libggml-cpu.dylib \
    "$METAL_VER" libggml-metal.0.dylib libggml-metal.dylib \
    "$GGML_VER" libggml.0.dylib libggml.dylib \
    "$COMMON_VER" libllama-common.0.dylib libllama-common.dylib \
    "$LLAMA_VER" libllama.0.dylib libllama.dylib \
    "$MTMD_VER" libmtmd.0.dylib libmtmd.dylib \
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
