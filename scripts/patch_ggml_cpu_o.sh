#!/bin/bash
# Usage: scripts/patch_ggml_cpu_o.sh
# Restores the working ggml-cpu.c.o (Aug 16 build-flat version) into the build directory
# and relinks libggml-cpu + libllama + llama-simple.
#
# Root cause: ggml-cpu.c was modified on Aug 19 03:33 with CGC changes that corrupt
# IQ3_XXS dequant under -O3. The original working .o is preserved at:
#   flashkv0516/build-flat-ggml-cpu.o
#
# Run this after any `cmake --build build` that recompiles ggml-cpu.c.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/temp/llama_roadB/llama.cpp-master"
BACKUP="$ROOT/build-flat-ggml-cpu.o"
BUILD="$LLAMA/build"

if [ ! -f "$BACKUP" ]; then
    echo "ERROR: $BACKUP not found. Cannot patch."
    exit 1
fi

if [ ! -d "$BUILD/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu" ]; then
    echo "ERROR: build directory not found at $BUILD"
    echo "Run cmake first, then re-run this script."
    exit 1
fi

TARGET="$BUILD/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o"
cp "$BACKUP" "$TARGET"
echo "Patched ggml-cpu.c.o"

# Relink
cd "$LLAMA"
cmake --build build --target ggml-cpu -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
cmake --build build --target llama -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
cmake --build build --target llama-simple -j$(sysctl -n hw.ncpu) 2>&1 | tail -3

echo "Done. Verify with:"
echo "  ./build/bin/llama-simple -m <model> -t 4 -p 'Hello' -n 8 2>/dev/null"
