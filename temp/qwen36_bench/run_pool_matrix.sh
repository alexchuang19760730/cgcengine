#!/bin/bash
# Pool matrix: slots x profile for qwen36 r4, 96 tok each, averaged.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

run_case() {
  local name=$1; shift
  echo "== $name =="
  env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 96 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/mx_$name.log" 2>&1 || echo "$name exit=$?"
}

# baseline sweep (no pool)
for s in 48 64 80; do
  run_case "base_s$s" TURBO_FIELDFARE_EXPERT_SLOTS=$s
done
# pool sweep: slots = pool + 16 LRU margin
run_case "pool48_s64"  TURBO_FIELDFARE_EXPERT_SLOTS=64  TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=48 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
run_case "pool64_s80"  TURBO_FIELDFARE_EXPERT_SLOTS=80  TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
run_case "pool80_s96"  TURBO_FIELDFARE_EXPERT_SLOTS=96  TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top80.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync

echo
echo "===== SUMMARY (avg over last 48 steady-state steps) ====="
for f in "$OUT"/mx_*.log; do
  name=$(basename "$f" .log)
  # compute averages from Q36STEP lines, skipping first 48 (cold start)
  awk -v name="$name" '
    /Q36STEP/ {
      match($0, /wall=([0-9]+)ms/, w)
      match($0, /gpu=([0-9]+)ms/, g)
      match($0, /fill=([0-9]+)ms/, f)
      match($0, /hit=([0-9]+)%/, h)
      if (++n > 48) { sw+=w[1]; sg+=g[1]; sf+=f[1]; sh+=h[1]; m++ }
    }
    END { if (m>0) printf "%-14s wall=%4dms gpu=%4dms fill=%4dms hit=%3d%% (n=%d)\n", name, sw/m, sg/m, sf/m, sh/m, m }
  ' "$f"
done | sort -t= -k2 -n
