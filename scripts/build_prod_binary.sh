#!/bin/bash
# build_prod_binary.sh — Build production llama-simple binary
#
# Strategy: compile HEAD's simple.cpp (has expert-cache env var support)
# and link against build-flat's working dylibs (libllama 0.0.4).
# This avoids the IQ3_XXS garbling caused by ggml-cpu.c CGC changes.
#
# Output: build/bin/llama-simple (replaces existing)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/temp/llama_roadB/llama.cpp-master"
FLAT="$LLAMA/build-flat/bin"

if [ ! -f "$FLAT/libllama.0.0.4.dylib" ]; then
    echo "ERROR: build-flat dylibs not found at $FLAT"
    echo "Need build-flat to be intact."
    exit 1
fi

echo "Building production llama-simple..."
/usr/bin/c++ -O3 -DNDEBUG -arch arm64 \
  -I"$LLAMA/ggml/include" -I"$LLAMA/src" -I"$LLAMA/common" -I"$LLAMA/include" \
  -std=c++17 -DGGML_USE_LLAMAFILE -DGGML_USE_ACCELERATE \
  "$LLAMA/examples/simple/simple.cpp" \
  -o "$LLAMA/build/bin/llama-simple" \
  -Wl,-rpath,"$FLAT" \
  "$FLAT/libllama.0.0.4.dylib" \
  "$FLAT/libggml.0.19.0.dylib" \
  "$FLAT/libggml-cpu.0.19.0.dylib" \
  "$FLAT/libggml-metal.0.19.0.dylib" \
  "$FLAT/libggml-base.0.19.0.dylib"

chmod +x "$LLAMA/build/bin/llama-simple"
echo "Done: $LLAMA/build/bin/llama-simple"
echo "Test: $LLAMA/build/bin/llama-simple -m <model> -ngl 99 -c 2048 -t 8 -p 'Hello' -n 4 2>/dev/null"
