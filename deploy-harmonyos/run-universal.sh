#!/bin/bash
# Universal run script — auto-detects macOS vs Linux/HarmonyOS
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s)"

MODEL=""
N_TOKENS=16
PROMPT="The capital of France is"
NGL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 -m <model.gguf> [-n tokens] [-p prompt]"; exit 1 ;;
        *) shift ;;
    esac
done

[ -z "$MODEL" ] && { echo "Error: -m <model> required"; exit 1; }

case "$OS" in
    Darwin)
        echo "=== macOS (Metal GPU) ==="
        [ -z "$NGL" ] && NGL=99
        BIN="$SCRIPT_DIR/macos/llama-simple"
        THREADS=8
        DYLD_LIBRARY_PATH="$SCRIPT_DIR/macos" "$BIN" \
            -m "$MODEL" -ngl "$NGL" -t "$THREADS" -c 2048 \
            -n "$N_TOKENS" -p "$PROMPT" 2>&1
        ;;
    Linux|*)
        echo "=== Linux/HarmonyOS (CPU-only) ==="
        [ -z "$NGL" ] && NGL=0
        CORES=$(nproc 2>/dev/null || echo 8)
        THREADS=$((CORES > 10 ? 10 : CORES))
        BIN="$SCRIPT_DIR/harmonyos/llama.cpp-master/build-release/bin/llama-simple"
        "$BIN" -m "$MODEL" -ngl "$NGL" -t "$THREADS" -c 2048 \
            --no-mmap --flash-attn -n "$N_TOKENS" -p "$PROMPT" 2>&1
        ;;
esac
