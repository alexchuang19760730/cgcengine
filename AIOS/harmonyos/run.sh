#!/bin/bash
# ============================================================
# Run llama.cpp on HarmonyOS (Kirin 9030, CPU-only)
# ============================================================
# Supports dual-model:
#   -m qwen36   → Qwen3.6-35B-A3B (IQ3_XXS, 12GB, 3B active, SPEED)
#   -m moe      → Qwen3.8 MoE Q3_K_S (12.7GB, 17.8B active, QUALITY)
#   -m <path>   → 任意 GGUF 檔案
# Expert-cache: 4GB bounded pool, skip-load for FFN weights
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/llama.cpp-master/build-release}"
BIN="$BUILD_DIR/bin/llama-simple"

# --- Model paths (adjust to your mount point) ---
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/gguf}"
QWEN36_MODEL="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
MOE_MODEL="$MODEL_DIR/Qwen3.8-Whittle-MoE-27B-A17.8B-Q3_K_S.gguf"

# --- Defaults ---
MODEL_NAME="qwen36"
N_TOKENS=128
PROMPT="The capital of France is"
NGL=99
CTX=2048
CORES=$(nproc 2>/dev/null || echo 8)
THREADS=$((CORES > 8 ? 8 : CORES))
BUDGET="${CACHE_BUDGET:-4294967296}"  # 4GB default
WARM=0
IGNORE_EOS=0

# --- Expert-cache env vars (CGC bounded-residency) ---
export LLAMA_EXPERT_CACHE_ENABLE=1
export LLAMA_EXPERT_CACHE_BUDGET="$BUDGET"
export LLAMA_EXPERT_CACHE_WORKERS=8
export LLAMA_EXPERT_CACHE_ALLOW_NGL=1
export LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
export LLAMA_EXPERT_CACHE_NO_PREWARM=1

usage() {
    echo "Usage: $0 [-m qwen36|moe|<path>] [-n tokens] [-p prompt] [-ngl N]"
    echo ""
    echo "Models:"
    echo "  qwen36  Qwen3.6-35B-A3B IQ3_XXS (12GB, SPEED, ~3-5 t/s on Kirin)"
    echo "  moe     Qwen3.8 MoE Q3_K_S (12.7GB, QUALITY, ~0.5-1 t/s on Kirin)"
    echo ""
    echo "Options:"
    echo "  -m <model>     Model name or path (default: qwen36)"
    echo "  -n <N>         Generate N tokens (default: 128)"
    echo "  -p <prompt>    Input prompt"
    echo "  -ngl <N>       GPU layers (0 for CPU-only, default: 99)"
    echo "  -c <N>         Context size (default: 2048)"
    echo "  -t <N>         Threads (default: auto)"
    echo "  --budget <B>   Cache budget in bytes (default: 4GB)"
    echo "  --warm         Warm model into page cache first"
    echo "  --ignore-eos   Ignore EOS for benchmarking"
    echo "  -h             Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        --budget) CACHE_BUDGET="$2"; export LLAMA_EXPERT_CACHE_BUDGET="$2"; shift 2 ;;
        --warm) WARM=1; shift ;;
        --ignore-eos) IGNORE_EOS=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Resolve model path ---
case "$MODEL_NAME" in
    qwen36|qwen3.6)
        MODEL="$QWEN36_MODEL"
        echo ">>> Using Qwen3.6-35B-A3B (SPEED)"
        ;;
    moe|qwen3.8)
        MODEL="$MOE_MODEL"
        echo ">>> Using Qwen3.8 MoE (QUALITY)"
        ;;
    /*)
        MODEL="$MODEL_NAME"
        ;;
    *)
        echo "Error: Unknown model '$MODEL_NAME'. Use qwen36, moe, or full path."
        exit 1
        ;;
esac

[ ! -f "$MODEL" ] && { echo "Error: Model not found: $MODEL"; exit 1; }
[ ! -f "$BIN" ] && { echo "Error: Binary not found. Run ./build.sh first"; exit 1; }

# --- Warm page cache (optional) ---
if [ "$WARM" = "1" ]; then
    echo ">>> Warming model into page cache..."
    cat "$MODEL" > /dev/null 2>&1
    echo ">>> Warm done."
fi

# --- Build command ---
EXTRA_ARGS=""
[ "$IGNORE_EOS" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --ignore-eos"

mkdir -p /tmp/cgc_log
LOG=/tmp/cgc_log/$(date +%Y%m%d_%H%M%S)_${MODEL_NAME}_ngl${NGL}.log

echo "=== HarmonyOS CGC Engine (Kirin 9030 CPU-only) ==="
echo "Model:       $(basename "$MODEL")"
echo "Threads:     $THREADS"
echo "Cache:       $(($BUDGET / 1048576)) MB"
echo "NGL:         $NGL"
echo "Prompt:      ${PROMPT:0:60}"
echo "Tokens:      $N_TOKENS"
echo "Log:         $LOG"
echo ""

"$BIN" -m "$MODEL" \
    -ngl "$NGL" \
    -t "$THREADS" \
    -c "$CTX" \
    --no-mmap \
    --flash-attn \
    -n "$N_TOKENS" \
    -p "$PROMPT" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG"
