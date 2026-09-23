#!/bin/bash
# pool 甜點曲線：4/5/6/8GB expert-cache × (t/s, hit%, Δswap, thermal)
# 交錯順序 4 8 6 5（避免隨時間單調漂移）；每臂間 90s 冷卻（純 decode 不重熱）
# 產出：/tmp/pool_sweet_spot.jsonl（每臂一行）+ stderr log /tmp/pool_sweet_spot_<size>g.log
set -u
REPO="/Users/alexchuang/Documents/flashkv-devserver"
BIN="$REPO/src/llama.cpp/build/bin/llama-bench"
MODEL="$REPO/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
OUT=/tmp/pool_sweet_spot.jsonl
: > "$OUT"
swap_mb() { sysctl vm.swapusage | awk -F'used = ' '{split($2,a," "); print a[1]}' | sed 's/[MG]$//; s/\..*//'; }
# 完整 prod-new env（CGC_DUMP_ENV=1 解析自 run_server.sh —— 缺任一個都會 Metal OOM，見 §EN-47x）
# 與 commit_bench 同口徑：-p 2048；PREFILL_STREAM=1 + slab 是 prod-new 的一部分
ENVS=(CGC_SERVER_MTP=0 CGC_EXPERT_CACHE_BYTES=8589934592 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
      LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0 LLAMA_EXPERT_CACHE_WORKERS=8 CGC_WAKE_POLL_US=15 \
      CGC_EVICTED_RING=0 CGC_N_CB=8 CGC_OA_ASYNC=1 CGC_SERVER_AUTO_ANCHOR=0 \
      CGC_SERVER_DEFAULT_MARKER_STOPS=1 CGC_GATHER_SLAB_CAP=256 CGC_PREFILL_STREAM=1 \
      CGC_DBUF=1 CGC_SPAC=1 CGC_SPAC_ALPHA=0.75 CGC_SOFT_POOL_L0=0 CGC_SOFT_POOL_L1=0 \
      CGC_LOOP_GUARD=1 CGC_GLU_FUSED_DOWN=1 CGC_WATCHDOG=1 CGC_MM_BITIDENT=1)

for SIZE in 8 6 5 4; do
  BYTES=$((SIZE * 1073741824))
  SW_BEFORE=$(swap_mb)
  echo "[$(date +%H:%M:%S)] arm pool=${SIZE}G start swap=${SW_BEFORE}" >> "$OUT"
  env "${ENVS[@]}" "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
    -expert-cache "$BYTES" --cache-type-k q8_0 --cache-type-v q8_0 \
    -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 3 -o json \
    > /tmp/pool_sweet_spot_${SIZE}g.json 2> /tmp/pool_sweet_spot_${SIZE}g.log
  RC=$?
  SW_AFTER=$(swap_mb)
  if [ $RC -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] arm pool=${SIZE}G FAILED rc=${RC} (見 stderr 尾部) swap_before=${SW_BEFORE} swap_after=${SW_AFTER} dswap=$((SW_AFTER - SW_BEFORE))" >> "$OUT"
    [ "$SIZE" != "4" ] && sleep 90
    continue
  fi
  TPS=$(python3 - <<PY
import json
try:
    d=json.load(open('/tmp/pool_sweet_spot_${SIZE}g.json'))
    rows = d if isinstance(d, list) else d.get('results', [])
    for r in rows:
        if r.get('n_prompt',0)==0 and r.get('n_gen',0)>0:
            print(round(r.get('avg_ts') or 0, 2)); break
    else:
        print('NA')
except Exception:
    print('ERR')
PY
)
  HIT=$(grep -oE 'hit rate [0-9.]+%' /tmp/pool_sweet_spot_${SIZE}g.log | tail -1 | grep -oE '[0-9.]+')
  echo "[$(date +%H:%M:%S)] arm pool=${SIZE}G t/s=${TPS} hit=${HIT} swap_before=${SW_BEFORE} swap_after=${SW_AFTER} dswap=$((SW_AFTER - SW_BEFORE))" >> "$OUT"
  [ "$SIZE" != "4" ] && sleep 90
done
echo "=== DONE ===" >> "$OUT"
cat "$OUT"
