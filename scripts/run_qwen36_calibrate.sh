#!/bin/bash
# run_qwen36_calibrate.sh — 產生 qwen36 的 PIN_PROFILE（LLAMA_EXPERT_CACHE_RECORD_ALL calibration）
#
# 執行一次 calibration：用代表 prompt 跑一小段生成，LLAMA_EXPERT_CACHE_ROUTE_RECORD=1
# 記錄每層路由頻率，teardown 時 LLAMA_EXPERT_CACHE_ROUTE_DUMP 寫出 per-layer top-K 高頻
# profile + stderr 印 coverage verdict（mean/min/max vs uniform baseline K/128）→
# profiles/qwen36_calib.pin。
# run_n30cache.sh（qwen36）預設載入此 profile pin 住 decode working set（§8.55）。
#
# 用法：
#   ./scripts/run_qwen36_calibrate.sh [-p PROMPT_FILE] [-n tokens]
#   N30CACHE_PROMPT_FILE=... ./scripts/run_qwen36_calibrate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-simple"
Q36="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
PROMPT_FILE="${N30CACHE_PROMPT_FILE:-$ROOT/scripts/prompts/calib_qwen36.txt}"
N="${N30CACHE_N:-64}"
OUT="$ROOT/profiles/qwen36_calib.pin"

[ -f "$BIN" ] || { echo "error: binary not found: $BIN" >&2; exit 2; }
[ -f "$Q36" ] || { echo "error: model not found: $Q36" >&2; exit 2; }
if [ -n "${1:-}" ] && [ "$1" != "-p" ]; then PROMPT_FILE="$1"; fi
[ -f "$PROMPT_FILE" ] || { echo "error: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"

echo "=== qwen36 calibration → $OUT (n=$N) ==="
env CGC_EXPERT_CACHE_BYTES=4294967296 \
    LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
    LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 \
    LLAMA_EXPERT_CACHE_WORKERS=8 \
    CGC_WAKE_POLL_US=15 \
    CGC_OA_ASYNC=1 \
    CGC_N_CB=8 \
    LLAMA_EXPERT_CACHE_ROUTE_RECORD=1 \
    LLAMA_EXPERT_CACHE_ROUTE_DUMP="$OUT" \
    "$BIN" -m "$Q36" -ngl 99 --no-mmap -t 8 \
        --prompt-file "$PROMPT_FILE" -n "$N" \
        > /tmp/qwen36_calib.out 2> /tmp/qwen36_calib.err || { tail -5 /tmp/qwen36_calib.err >&2; exit 1; }
echo "profile written: $OUT ($(awk '{n+=NF} END{print n}' "$OUT") experts, $(wc -l < "$OUT") layers)"
grep "ROUTE-DUMP" /tmp/qwen36_calib.err || true
