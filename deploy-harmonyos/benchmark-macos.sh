#!/bin/bash
# Benchmark llama.cpp on macOS (M4 Max + Metal GPU)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/macos/llama-simple"

MODEL="${1:-}"
[ -z "$MODEL" ] && { echo "Usage: $0 <model.gguf>"; exit 1; }

echo "=== macOS M4 Max Benchmark ==="
echo "Model: $MODEL"
echo ""

echo "--- Baseline (no cache) ---"
DYLD_LIBRARY_PATH="$SCRIPT_DIR/macos" "$BIN" \
    -m "$MODEL" -ngl 99 -t 8 -c 2048 -n 32 \
    -p "The capital of France is" 2>&1 | grep -E "eval time|speed|tok/s"
echo ""

echo "--- Expert-cache ON ---"
CGC_EXPERT_CACHE_BYTES=4294967296 \
DYLD_LIBRARY_PATH="$SCRIPT_DIR/macos" "$BIN" \
    -m "$MODEL" -ngl 99 -t 8 -c 2048 -n 32 \
    -p "The capital of France is" 2>&1 | grep -E "eval time|speed|tok/s|hit rate"
echo ""
