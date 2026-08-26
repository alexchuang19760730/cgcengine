#!/bin/bash
# Benchmark on HarmonyOS (Kirin 9030) - test different thread counts
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/llama.cpp-master/build-release}"
BIN="$BUILD_DIR/bin/llama-simple"

MODEL="${1:-}"
[ -z "$MODEL" ] && { echo "Usage: $0 <model.gguf>"; exit 1; }

echo "=== Kirin 9030 Benchmark ==="
for T in 4 6 8 10 12; do
    echo "--- threads=$T ---"
    "$BIN" -m "$MODEL" -ngl 0 -t "$T" -c 2048 --no-mmap --flash-attn \
        -n 32 -p "The capital of France is" 2>&1 | grep -E "eval time|speed|tok/s"
    echo ""
done
