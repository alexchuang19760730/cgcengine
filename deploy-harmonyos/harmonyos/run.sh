#!/bin/bash
# Run llama.cpp on HarmonyOS PC (Kirin 9030 CPU-only)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/llama.cpp-master/build-release}"
BIN="$BUILD_DIR/bin/llama-simple"

MODEL=""
N_TOKENS=16
PROMPT="The capital of France is"
NGL=0
CTX=2048
CORES=$(nproc 2>/dev/null || echo 8)
THREADS=$((CORES > 10 ? 10 : CORES))

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 -m <model.gguf> [-n tokens] [-p prompt]"; exit 1 ;;
        *) shift ;;
    esac
done

[ -z "$MODEL" ] && { echo "Error: -m <model> required"; exit 1; }
[ ! -f "$BIN" ] && { echo "Error: Run ./build.sh first"; exit 1; }

echo "=== HarmonyOS llama.cpp (Kirin 9030 CPU-only) ==="
echo "Model:   $MODEL"
echo "Threads: $THREADS"
echo ""

"$BIN" -m "$MODEL" -ngl "$NGL" -t "$THREADS" -c "$CTX" \
    --no-mmap --flash-attn -n "$N_TOKENS" -p "$PROMPT" 2>&1
