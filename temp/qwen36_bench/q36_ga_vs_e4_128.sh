#!/bin/bash
# §13.110: r4q4la_ga (fp16 embed) vs r4q4la_ga_e4 (q4 embed) — 128 tok 交錯 A/B
# 目的: 定案 §13.105 的 +6.7% 是真效應還是 load 運氣 (§13.106 在 r3q4 只看到 +0.6%)
# 方法: 2 輪 × 2 模型正反序交錯 [ga, e4, e4, ga], 128 tok, 同 top96 profile (md5 一致),
#       同生產 env (slots96/pool96/sync preload/read workers 8), 乾淨窗 load<2
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
GA=prime-agent-worktrees/qwen36-r4q4la_ga.gturbo
E4=prime-agent-worktrees/qwen36-r4q4la_ga_e4.gturbo
MSG=temp/qwen36_bench/msg_code.json
declare -a MODELS=("$GA" "$E4")
declare -a NAMES=("ga_fp16embed" "e4_q4embed")
# [ga, e4, e4, ga]
ORDER=(0 1 1 0)
R=1
for mi in "${ORDER[@]}"; do
  M="${MODELS[$mi]}"
  NAME="${NAMES[$mi]}"
  echo "=== R$R $NAME ==="
  TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 \
    TURBO_FIELDFARE_HOT_POOL_EXPERTS=96 \
    TURBO_FIELDFARE_HOT_POOL_PROFILE=$M/profiles/top96_code_prose.json \
    TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 \
    "$BIN" --model "$M" --trust-receipt --messages-file "$MSG" --max-new 128 \
      --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 \
      | grep 'stop=' | tail -1
  R=$((R+1))
done
