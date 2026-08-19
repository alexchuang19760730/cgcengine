#!/bin/bash
# A/B: qwen36 r4 expert hot pool — baseline vs top64 pinned profile.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
PROF=$OUT/qwen36_top64.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

echo "== A: baseline (EXPERT_SLOTS=96, no pool) =="
TURBO_FIELDFARE_EXPERT_SLOTS=96 \
  "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
  --max-new 96 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
  > "$OUT/poolA.log" 2>&1 || echo "A exit=$?"

echo "== B: top64 pinned pool (EXPERT_SLOTS=96, HOT_POOL=1) =="
TURBO_FIELDFARE_EXPERT_SLOTS=96 \
  TURBO_FIELDFARE_HOT_POOL=1 \
  TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 \
  TURBO_FIELDFARE_HOT_POOL_PROFILE="$PROF" \
  TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
  "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
  --max-new 96 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
  > "$OUT/poolB.log" 2>&1 || echo "B exit=$?"

echo "=== A summary ==="
grep -E 'Q36STEP|stop=' "$OUT/poolA.log" | tail -6
echo "=== B summary ==="
grep -E 'Q36STEP|stop=' "$OUT/poolB.log" | tail -6
