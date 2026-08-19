#!/bin/bash
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P96=$OUT/qwen36_top96.json
/Users/alexchuang/Documents/flashkv0516/.venv-cgc/bin/python3 $ROOT/prime-agent-worktrees/turbo-fieldfare/Scripts/gen_hotpool_profile.py $OUT/trace_all.csv 96 $OUT/qwen36_top96.json >/dev/null 2>&1
export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""
echo "=== 128 slots + pool96 ==="
/usr/bin/time -l env TURBO_FIELDFARE_EXPERT_SLOTS=128 \
  TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=96 \
  TURBO_FIELDFARE_HOT_POOL_PROFILE=$P96 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
  "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
  --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
  > "$OUT/mp_s128p96.log" 2> "$OUT/mp_s128p96.time" || echo exit=$?
grep -E 'maximum resident|tok/s' "$OUT/mp_s128p96.time"
echo done
