#!/bin/bash
# ============================================================
# Benchmark on HarmonyOS (Kirin 9030, CPU-only)
# ============================================================
# Tests both models across thread counts:
#   Qwen3.6 A3B (SPEED) vs Qwen3.8 MoE (QUALITY)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/llama.cpp-master/build-release}"
BIN="$BUILD_DIR/bin/llama-simple"

# --- Model paths ---
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/gguf}"
QWEN36_MODEL="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
MOE_MODEL="$MODEL_DIR/Qwen3.8-Whittle-MoE-27B-A17.8B-Q3_K_S.gguf"

# --- Expert-cache env vars ---
export LLAMA_EXPERT_CACHE_ENABLE=1
export LLAMA_EXPERT_CACHE_BUDGET="${CACHE_BUDGET:-4294967296}"
export LLAMA_EXPERT_CACHE_WORKERS=8
export LLAMA_EXPERT_CACHE_ALLOW_NGL=1
export LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
export LLAMA_EXPERT_CACHE_NO_PREWARM=1

CORES=$(nproc 2>/dev/null || echo 8)
LOG_DIR=/tmp/cgc_log/benchmark_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# --- Parse args ---
MODE="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --qwen36) MODE="qwen36"; shift ;;
        --moe) MODE="moe"; shift ;;
        -h|--help)
            echo "Usage: $0 [--qwen36|--moe|both]"
            echo "  --qwen36   Benchmark Qwen3.6 A3B only"
            echo "  --moe      Benchmark MoE Q3_K_S only"
            echo "  both       Benchmark both (default)"
            exit 0 ;;
        *) shift ;;
    esac
done

run_bench() {
    local MODEL="$1"
    local NAME="$2"
    local RESULT_FILE="$LOG_DIR/${NAME}.txt"

    echo "" >> "$RESULT_FILE"
    echo "# $(date)" >> "$RESULT_FILE"
    echo "# Model: $NAME" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"

    echo "=== Benchmark: $NAME ==="
    for T in 2 4 6 8; do
        if [ $T -gt $CORES ]; then continue; fi
        echo "--- threads=$T ---"
        echo "threads=$T" >> "$RESULT_FILE"

        RUN_LOG="$LOG_DIR/${NAME}_t${T}.log"
        "$BIN" -m "$MODEL" \
            -ngl 0 \
            -t "$T" \
            -c 2048 \
            --no-mmap \
            --flash-attn \
            -n 64 \
            -p "The capital of France is" \
            2>&1 | tee "$RUN_LOG"

        # Extract speed
        SPEED=$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$RUN_LOG" | tail -1 || echo "N/A")
        echo "speed=$SPEED" >> "$RESULT_FILE"
        echo "  Result: $SPEED"
        echo ""
    done
}

echo "=========================================="
echo "CGC Engine Benchmark (Kirin 9030)"
echo "=========================================="
echo "Model dir:    $MODEL_DIR"
echo "Cache budget: $((${LLAMA_EXPERT_CACHE_BUDGET} / 1048576)) MB"
echo "Cores:        $CORES"
echo "Log dir:      $LOG_DIR"
echo "=========================================="
echo ""

# --- Qwen3.6 A3B (SPEED) ---
if [[ "$MODE" == "both" || "$MODE" == "qwen36" ]]; then
    if [ -f "$QWEN36_MODEL" ]; then
        run_bench "$QWEN36_MODEL" "qwen36_A3B"
    else
        echo "SKIP: $QWEN36_MODEL not found"
    fi
fi

# --- MoE Q3_K_S (QUALITY) ---
if [[ "$MODE" == "both" || "$MODE" == "moe" ]]; then
    if [ -f "$MOE_MODEL" ]; then
        run_bench "$MOE_MODEL" "qwen38_moe"
    else
        echo "SKIP: $MOE_MODEL not found"
    fi
fi

# --- Summary ---
echo "=========================================="
echo "Benchmark complete!"
echo "Results: $LOG_DIR/"
echo "=========================================="

# Print comparison table
echo ""
echo "=== Comparison ==="
for f in $LOG_DIR/*.txt; do
    [ -f "$f" ] && cat "$f"
done
