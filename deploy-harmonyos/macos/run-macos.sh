#!/bin/bash
# macOS run script for llama.cpp (M4 Max + Metal GPU)
# Usage: ./run-macos.sh -m <model.gguf> [-n 16] [-p "prompt"]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/llama-simple"
DYLIB_DIR="$SCRIPT_DIR"

MODEL=""
N_TOKENS=16
PROMPT="The capital of France is"
NGL=99
CTX=2048
THREADS=8

usage() {
    echo "Usage: $0 -m <model.gguf> [-n tokens] [-p prompt] [-ngl N] [-c ctx]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

[ -z "$MODEL" ] && { echo "Error: -m <model> required"; usage; }

echo "=== macOS llama.cpp ==="
echo "Model:   $MODEL"
echo "ngl:     $NGL"
echo "Threads: $THREADS"
echo "Context: $CTX"
echo ""

DYLD_LIBRARY_PATH="$DYLIB_DIR" "$BIN" \
    -m "$MODEL" -ngl "$NGL" -t "$THREADS" -c "$CTX" \
    -n "$N_TOKENS" -p "$PROMPT" 2>&1
