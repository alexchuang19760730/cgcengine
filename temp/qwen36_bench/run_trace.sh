#!/bin/bash
# Collect qwen36 r4 routing traces (code + prose) for hot-pool profile generation.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench

rm -f "$OUT/trace_code.csv" "$OUT/trace_prose.csv"

export TURBO_FIELDFARE_EXPERT_SLOTS=48
export TURBO_FIELDFARE_EXPERT_STATS=1
export MTP_MODEL=""

echo "== code trace =="
TURBO_FIELDFARE_EXPERT_TRACE="$OUT/trace_code.csv" \
  "$BIN" --model "$MODEL" --trust-receipt --messages-file "$OUT/msg_code.json" \
  --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
  > "$OUT/trace_code.log" 2>&1 || echo "code trace exit=$?"

echo "== prose trace =="
TURBO_FIELDFARE_EXPERT_TRACE="$OUT/trace_prose.csv" \
  "$BIN" --model "$MODEL" --trust-receipt --messages-file "$OUT/msg_prose.json" \
  --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
  > "$OUT/trace_prose.log" 2>&1 || echo "prose trace exit=$?"

echo "== trace line counts =="
wc -l "$OUT/trace_code.csv" "$OUT/trace_prose.csv" 2>/dev/null || true
