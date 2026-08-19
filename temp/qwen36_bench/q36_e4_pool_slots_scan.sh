#!/bin/bash
# §13.108: r4q4la_ga_e4 的 pool × slots 聯合掃描
# 問題: e4 的 resident mmap 縮小 (2.15→1.41GB) 能否支撐 pool112/128 轉正
# 格子: (slots, pool) ∈ {(96,96), (96,112), (64,96), (64,112), (96,128)}
# 每格 2 輪交錯 (R1-R5 第一輪, R6-R10 第二輪反序), 64 tok, 監看 swap
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
M=prime-agent-worktrees/qwen36-r4q4la_ga_e4.gturbo
MSG=temp/qwen36_bench/msg_code.json
declare -a CONFIGS=("96 96" "96 112" "64 96" "64 112" "96 128")
declare -a NAMES=("s96p96" "s96p112" "s64p96" "s64p112" "s96p128")
# R1-R5: config order 0..4; R6-R10: reverse
ORDER=(0 1 2 3 4 4 3 2 1 0)
R=1
for ci in "${ORDER[@]}"; do
  read -r SLOTS POOL <<< "${CONFIGS[$ci]}"
  NAME="${NAMES[$ci]}"
  echo "=== R$R $NAME (slots=$SLOTS pool=$POOL) ==="
  TURBO_FIELDFARE_EXPERT_SLOTS=$SLOTS TURBO_FIELDFARE_HOT_POOL=1 \
    TURBO_FIELDFARE_HOT_POOL_EXPERTS=$POOL \
    TURBO_FIELDFARE_HOT_POOL_PROFILE=$M/profiles/top${POOL}_code_prose.json \
    TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
    "$BIN" --model "$M" --trust-receipt --messages-file "$MSG" --max-new 64 \
      --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 \
      | grep 'stop=' | tail -1
  sysctl vm.swapusage | grep -oE 'used = [0-9.]+[A-Z]' | head -1
  R=$((R+1))
done
