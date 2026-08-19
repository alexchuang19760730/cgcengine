#!/bin/bash
# build_prod_binary.sh — Build production llama-simple binary
#
# Strategy: compile HEAD's simple.cpp (has expert-cache env var support)
# and link against build/bin's dylibs (libllama 0.0.5, fresh post-f00c86b).
# libllama 0.0.5 has expert-cache integration (bounded pool, eval callback).
# libggml-cpu from upstream (correct IQ3_XXS dequant).
#
# MTP isolation: llama-simple is built WITHOUT -DMTP_SUPPORT (no MTP code in
# the binary).  llama-speculative-simple is built WITH -DMTP_SUPPORT so the
# draft-mtp code path in speculative-simple.cpp + common/speculative.cpp is
# compiled in.  The shared libllama-common.dylib must be built with
# -DMTP_SUPPORT for release_context() to exist (see build_fork_llama.sh).
#
# Output: build/bin/llama-simple + build/bin/llama-speculative-simple
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/temp/llama_roadB/llama.cpp-master"
DYLIB="$LLAMA/build/bin"

if [ ! -f "$DYLIB/libllama.0.dylib" ]; then
    echo "ERROR: build/bin libllama not found at $DYLIB"
    echo "Need build/bin to be built first (scripts/build_fork_llama.sh)."
    exit 1
fi

echo "Building production llama-simple + llama-speculative-simple..."
# llama-simple: NO -DMTP_SUPPORT (pure non-MTP binary)
/usr/bin/c++ -O3 -DNDEBUG -arch arm64 \
  -I"$LLAMA/ggml/include" -I"$LLAMA/src" -I"$LLAMA/common" -I"$LLAMA/include" \
  -std=c++17 -DGGML_USE_LLAMAFILE -DGGML_USE_ACCELERATE \
  "$LLAMA/examples/simple/simple.cpp" \
  -o "$LLAMA/build/bin/llama-simple" \
  -Wl,-rpath,"$DYLIB" \
  "$DYLIB/libllama.0.dylib" \
  "$DYLIB/libggml.0.dylib" \
  "$DYLIB/libggml-cpu.0.dylib" \
  "$DYLIB/libggml-metal.0.dylib" \
  "$DYLIB/libllama-common.0.dylib" \
  "$DYLIB/libggml-base.0.dylib"

chmod +x "$LLAMA/build/bin/llama-simple"

# Build speculative-simple for MTP mode (WITH -DMTP_SUPPORT)
if [ -f "$LLAMA/examples/speculative-simple/speculative-simple.cpp" ]; then
  /usr/bin/c++ -O3 -DNDEBUG -arch arm64 \
    -I"$LLAMA/ggml/include" -I"$LLAMA/src" -I"$LLAMA/common" -I"$LLAMA/include" \
    -std=c++17 -DGGML_USE_LLAMAFILE -DGGML_USE_ACCELERATE -DMTP_SUPPORT \
    "$LLAMA/examples/speculative-simple/speculative-simple.cpp" \
    -o "$LLAMA/build/bin/llama-speculative-simple" \
    -Wl,-rpath,"$DYLIB" \
    "$DYLIB/libllama.0.dylib" \
    "$DYLIB/libggml.0.dylib" \
    "$DYLIB/libggml-cpu.0.dylib" \
    "$DYLIB/libggml-metal.0.dylib" \
    "$DYLIB/libllama-common.0.dylib" \
    "$DYLIB/libggml-base.0.dylib"
  chmod +x "$LLAMA/build/bin/llama-speculative-simple"
  echo "Built: $LLAMA/build/bin/llama-speculative-simple"
fi

echo "Done: $LLAMA/build/bin/llama-simple"
echo "Test non-MTP: $LLAMA/build/bin/llama-simple -m <model> -ngl 99 -c 2048 -t 8 -p 'Hello' -n 4 2>/dev/null"
echo "Test MTP:      $LLAMA/build/bin/llama-speculative-simple -m <model> --mtp 2 -ngl 99 -c 2048 -t 8 -p 'Hello' -n 4 2>/dev/null"
