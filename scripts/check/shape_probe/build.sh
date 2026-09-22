#!/bin/sh
# Compile the per-shape GEMV probe WITHOUT touching the shared build tree.
#
# IMPORTANT: do NOT use `cmake --build` here. Build products in this repo are
# version-controlled and shared between sessions; a rebuild overwrites the
# dylib that another session may be mapping, which invalidates its A/B run.
# So we compile this one file and link the already-built ggml libraries.
#
# Usage:  ./scripts/check/shape_probe/build.sh
# Output: scripts/check/shape_probe/cgc_shape_probe

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
GGML="${ROOT}/src/llama.cpp/ggml"
LIBDIR="${ROOT}/src/llama.cpp/build/bin"
OUT="${HERE}/cgc_shape_probe"

if [ ! -d "${GGML}/include" ]; then
    echo "FATAL: missing ${GGML}/include -- wrong repo root?" >&2
    exit 2
fi

INC_GGML="${GGML}/include"
INC_SRC="${GGML}/src"

# Link order matters: symbols come from libggml (core), libggml-base (backends),
# libggml-metal (Metal backend implementation).
cc -std=c++17 -O2 -Wall -Wno-unused-function \
    -I"${INC_GGML}" -I"${INC_SRC}" \
    -L"${LIBDIR}" \
    -lggml -lggml-base -lggml-metal \
    -Wl,-rpath,"${LIBDIR}" \
    -o "${OUT}" \
    "${HERE}/dense_gemv_probe.cpp" \
    -lc++

echo "built ${OUT}"
md5 -q "${LIBDIR}/libggml-metal.dylib" | sed 's/^/libggml-metal.dylib md5 /'
md5 -q "${LIBDIR}/libggml.dylib"       | sed 's/^/libggml.dylib       md5 /'
