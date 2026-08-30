#!/bin/bash
# Run llama.cpp on HarmonyOS NEXT phone (Mate 70 Pro, 12GB RAM)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/llama-server}"
MODEL=""
N_TOKENS=128
PROMPT="The capital of France is"
NGL=0
CTX=2048
PORT=8080
EXPERT_CACHE=0
THREADS=6

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --expert-cache) EXPERT_CACHE="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 -m <model.gguf> [-n tokens] [-p prompt] [--port 8080]"; exit 1 ;;
        *) shift ;;
    esac
done

[ -z "$MODEL" ] && { echo "Error: -m <model> required"; exit 1; }
[ ! -f "$BIN" ] && { echo "Error: Binary not found at $BIN"; exit 1; }

echo "=== CGC Engine - HarmonyOS NEXT Phone ==="
echo "Model:   $MODEL"
echo "Binary:  $BIN"
echo "Threads: $THREADS"
echo "Port:    $PORT"
echo "Cache:   $EXPERT_CACHE bytes"
echo ""

if [ "$EXPERT_CACHE" -gt 0 ] 2>/dev/null; then
    export LLAMA_EXPERT_CACHE_ALLOW_NGL=1
    export LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
    export LLAMA_EXPERT_CACHE_WORKERS=4
    export CGC_OA_ASYNC=1
    export CGC_N_CB=4
    echo "Expert Cache: ENABLED ($EXPERT_CACHE bytes)"
fi

"$BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -ngl "$NGL" \
    -t "$THREADS" \
    -c "$CTX" \
    --no-mmap \
    --flash-attn \
    -n "$N_TOKENS" \
    -p "$PROMPT" \
    ${EXPERT_CACHE:+-expert-cache "$EXPERT_CACHE"} \
    2>&1
