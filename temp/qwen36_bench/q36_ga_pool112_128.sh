#!/bin/bash
# §13.104: r4q4la_ga pool112/pool128 可行性測試（不靠 embed q4）
# 交錯：96 → 112 → 96 → 128 → 96，各 128 tok；監看 swap（vm.swapusage）與 tok/s
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
MODEL=prime-agent-worktrees/qwen36-r4q4la_ga.gturbo
MSG=temp/qwen36_bench/msg_code.json
for R in 1 2 3 4 5; do
  case $R in
    1|3|5) EXP=96;  PROF=$MODEL/profiles/top96_code_prose.json;  L=pool96 ;;
    2)     EXP=112; PROF=$MODEL/profiles/top112_code_prose.json; L=pool112 ;;
    4)     EXP=128; PROF=$MODEL/profiles/top128_code_prose.json; L=pool128 ;;
  esac
  echo "=== R$R $L (EXPERTS=$EXP) ==="
  TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 \
    TURBO_FIELDFARE_HOT_POOL_EXPERTS=$EXP TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROF \
    TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 128 \
      --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 \
      | grep -E '\[stop=' | tail -1
  sysctl vm.swapusage | awk '{print "swap: "$1" "$2" "$3}'
done
