#!/bin/bash
# base(prod-new, SpAc EMA) vs PIN_PROFILE(prod-new + LLAMA_EXPERT_CACHE_PIN_PROFILE) ABBA×2
# 目的：定 75PCT §4 的 12.57~14.24 真值區間。同 build、同 cell、只差一個 env。
# 兩臂都帶 CGC_MASSCOV=1（75PCT §1：路由是數學，MASSCOV 不影響行為）→ pin 臂可對照 89.4%。
set -u
REPO="/Users/alexchuang/Documents/flashkv-devserver"
BIN="$REPO/src/llama.cpp/build/bin/llama-bench"
MODEL="$REPO/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
PROFILE="$REPO/scripts/check/pin_profiles/route_top142_p0_2026-09-23.txt"
OUT=/tmp/pin_abba.jsonl
: > "$OUT"
swap_mb() { sysctl vm.swapusage | awk -F'used = ' '{split($2,a," "); print a[1]}' | sed 's/[MG]$//; s/\..*//'; }
BASE=(CGC_SERVER_MTP=0 CGC_EXPERT_CACHE_BYTES=8589934592 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
      LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0 LLAMA_EXPERT_CACHE_WORKERS=8 CGC_WAKE_POLL_US=15 \
      CGC_EVICTED_RING=0 CGC_N_CB=8 CGC_OA_ASYNC=1 CGC_SERVER_AUTO_ANCHOR=0 \
      CGC_SERVER_DEFAULT_MARKER_STOPS=1 CGC_GATHER_SLAB_CAP=256 CGC_PREFILL_STREAM=1 \
      CGC_DBUF=1 CGC_SPAC=1 CGC_SPAC_ALPHA=0.75 CGC_SOFT_POOL_L0=0 CGC_SOFT_POOL_L1=0 \
      CGC_LOOP_GUARD=1 CGC_GLU_FUSED_DOWN=1 CGC_WATCHDOG=1 CGC_MM_BITIDENT=1 CGC_MASSCOV=1)
run_arm() {
  local TAG=$1 PIN_FLAG=$2
  local SW0 SW1 RC
  SW0=$(swap_mb)
  if [ "$PIN_FLAG" = "1" ]; then
    env "${BASE[@]}" LLAMA_EXPERT_CACHE_PIN_PROFILE="$PROFILE" "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
      -expert-cache 8589934592 --cache-type-k q8_0 --cache-type-v q8_0 \
      -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 2 -o json \
      > /tmp/pin_abba_${TAG}.json 2> /tmp/pin_abba_${TAG}.log
  else
    env "${BASE[@]}" "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
      -expert-cache 8589934592 --cache-type-k q8_0 --cache-type-v q8_0 \
      -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 2 -o json \
      > /tmp/pin_abba_${TAG}.json 2> /tmp/pin_abba_${TAG}.log
  fi
  RC=$?
  SW1=$(swap_mb)
  if [ $RC -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ${TAG} ${PIN_FLAG} FAILED rc=${RC} swap0=${SW0} swap1=${SW1}" >> "$OUT"
    return
  fi
  local TPS PP HIT COV
  TPS=$(python3 -c "
import json
d=json.load(open('/tmp/pin_abba_${TAG}.json'))
rows=d if isinstance(d,list) else d.get('results',[])
for r in rows:
    if r.get('n_prompt',0)==0 and r.get('n_gen',0)>0: print(round(r.get('avg_ts') or 0,2)); break
" 2>/dev/null || echo NA)
  PP=$(python3 -c "
import json
d=json.load(open('/tmp/pin_abba_${TAG}.json'))
rows=d if isinstance(d,list) else d.get('results',[])
for r in rows:
    if r.get('n_prompt',0)>0 and r.get('n_gen',0)==0: print(round(r.get('avg_ts') or 0,2)); break
" 2>/dev/null || echo NA)
  HIT=$(grep -oE 'hit rate [0-9.]+%' /tmp/pin_abba_${TAG}.log | tail -1 | grep -oE '[0-9.]+' || echo NA)
  COV=$(grep -oE 'coverage [0-9.]+%' /tmp/pin_abba_${TAG}.log | tail -1 | grep -oE '[0-9.]+' || echo NA)
  echo "[$(date +%H:%M:%S)] ${TAG} pin=${PIN_FLAG} decode=${TPS} prefill=${PP} hit=${HIT} cov=${COV} swap0=${SW0} swap1=${SW1} dswap=$((SW1 - SW0))" >> "$OUT"
}
echo "[$(date +%H:%M:%S)] 冷卻 300s 後開始 A=base(SpAc EMA) B=pin(PIN_PROFILE) ABBA×2" >> "$OUT"
sleep 300
for PAIR in 1 2; do
  run_arm base_${PAIR} 0
  sleep 240
  run_arm pin_${PAIR} 1
  sleep 240
done
echo "=== DONE ===" >> "$OUT"
cat "$OUT"
