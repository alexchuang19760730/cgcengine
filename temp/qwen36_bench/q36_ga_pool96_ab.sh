#!/bin/bash
# §13.103: r4q4la_ga pool80 (top80) vs pool96 (top96) 128 tok 交錯 A/B
# Hypothesis: GatedAttn q4 的 392MB RAM 餘裕讓 pool96 可安全 preload，hit↑ → tok/s 轉正
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
MODEL=prime-agent-worktrees/qwen36-r4q4la_ga.gturbo
P80=$MODEL/profiles/top80_code_prose.json
P96=$MODEL/profiles/top96_code_prose.json
MSG=temp/qwen36_bench/msg_code.json
for R in 1 2 3 4; do
  if [ $((R % 2)) -eq 1 ]; then
    EXP=80; PROF=$P80; L=pool80
  else
    EXP=96; PROF=$P96; L=pool96
  fi
  echo "=== R$R $L (EXPERTS=$EXP) ==="
  TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 \
    TURBO_FIELDFARE_HOT_POOL_EXPERTS=$EXP TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROF \
    TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 128 \
      --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 \
      | grep -E '\[stop=|expert-cache|hit=' | tail -3
done
