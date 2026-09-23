#!/bin/bash
# 4G vs 8G ABBA 判別：A B A B A B（3 對），每臂 r=2，臂間 240s 冷卻（M4 Air 無風扇）
# 判定：4G decode t/s 能否達到 8G —— 若能 ⇒ 平行雙實驗（兩 agent 各 4G）成立
set -u
REPO="/Users/alexchuang/Documents/flashkv-devserver"
BIN="$REPO/src/llama.cpp/build/bin/llama-bench"
MODEL="$REPO/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
OUT=/tmp/sweet_abba.jsonl
: > "$OUT"
swap_mb() { sysctl vm.swapusage | awk -F'used = ' '{split($2,a," "); print a[1]}' | sed 's/[MG]$//; s/\..*//'; }
BASE=(CGC_SERVER_MTP=0 CGC_EXPERT_CACHE_BYTES=8589934592 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
      LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0 LLAMA_EXPERT_CACHE_WORKERS=8 CGC_WAKE_POLL_US=15 \
      CGC_EVICTED_RING=0 CGC_N_CB=8 CGC_OA_ASYNC=1 CGC_SERVER_AUTO_ANCHOR=0 \
      CGC_SERVER_DEFAULT_MARKER_STOPS=1 CGC_GATHER_SLAB_CAP=256 CGC_PREFILL_STREAM=1 \
      CGC_DBUF=1 CGC_SPAC=1 CGC_SPAC_ALPHA=0.75 CGC_SOFT_POOL_L0=0 CGC_SOFT_POOL_L1=0 \
      CGC_LOOP_GUARD=1 CGC_GLU_FUSED_DOWN=1 CGC_WATCHDOG=1 CGC_MM_BITIDENT=1)
run_arm() {
  local TAG=$1 SIZE=$2
  local BYTES=$((SIZE * 1073741824)) SW0 SW1 RC
  SW0=$(swap_mb)
  env "${BASE[@]}" "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
    -expert-cache "$BYTES" --cache-type-k q8_0 --cache-type-v q8_0 \
    -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 2 -o json \
    > /tmp/sweet_abba_${TAG}.json 2> /tmp/sweet_abba_${TAG}.log
  RC=$?
  SW1=$(swap_mb)
  if [ $RC -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ${TAG} pool=${SIZE}G FAILED rc=${RC} swap0=${SW0} swap1=${SW1}" >> "$OUT"
    return
  fi
  local TPS PP HIT
  TPS=$(python3 -c "
import json
d=json.load(open('/tmp/sweet_abba_${TAG}.json'))
rows=d if isinstance(d,list) else d.get('results',[])
for r in rows:
    if r.get('n_prompt',0)==0 and r.get('n_gen',0)>0: print(round(r.get('avg_ts') or 0,2)); break
" 2>/dev/null || echo NA)
  PP=$(python3 -c "
import json
d=json.load(open('/tmp/sweet_abba_${TAG}.json'))
rows=d if isinstance(d,list) else d.get('results',[])
for r in rows:
    if r.get('n_prompt',0)>0 and r.get('n_gen',0)==0: print(round(r.get('avg_ts') or 0,2)); break
" 2>/dev/null || echo NA)
  HIT=$(grep -oE 'hit rate [0-9.]+%' /tmp/sweet_abba_${TAG}.log | tail -1 | grep -oE '[0-9.]+' || echo NA)
  echo "[$(date +%H:%M:%S)] ${TAG} pool=${SIZE}G decode=${TPS} prefill=${PP} hit=${HIT} swap0=${SW0} swap1=${SW1} dswap=$((SW1 - SW0))" >> "$OUT"
}
echo "[$(date +%H:%M:%S)] 冷卻 300s 後開始 A=4G B=8G ABBA×3" >> "$OUT"
sleep 300
for PAIR in 1 2 3; do
  run_arm a4_${PAIR} 4
  sleep 240
  run_arm b8_${PAIR} 8
  sleep 240
done
echo "=== DONE ===" >> "$OUT"
cat "$OUT"
