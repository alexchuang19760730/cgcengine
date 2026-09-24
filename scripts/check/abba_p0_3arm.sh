#!/bin/bash
# ABBA 兩組三臂測速：{ctrl, p0} → {p0, p0+PIN_PROFILE}
# 目的：回答「P0 的代價在乾淨/同環境下多少；pin（B 近似）能吃掉多少」
# 協議：ABBA 交錯、每 launch 前深冷卻 ≥300s、配對 per-rep 比率中位、窗口守門、build 指紋
# 用法: bash scripts/check/abba_p0_3arm.sh [--pairs N] [--cool N]

set -u
REPO=/Users/alexchuang/Documents/flashkv-devserver
PAIRS=${PAIRS:-3}
COOL_S=${COOL_S:-300}
PROMPT="Write a detailed technical analysis of memory hierarchy in modern GPU architectures."
NP=160
OUT=/tmp/abba_p0_3arm
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/abba.log"; }

window_free() {
  # 有其他 llama-server 在跑 → 不搶
  n=$(pgrep -f llama-server 2>/dev/null | grep -v $$ | wc -l | tr -d ' ')
  [ "$n" = "0" ]
}

fingerprint() {
  echo "server=$(md5 -q $REPO/src/llama.cpp/build/bin/llama-server 2>/dev/null) impl=$(md5 -q $REPO/src/llama.cpp/build/bin/libllama-server-impl.dylib 2>/dev/null)"
}

launch() { # launch <arm> <extra_env>
  local arm=$1 extra=$2
  pkill -9 -f llama-server 2>/dev/null; sleep 3
  eval "CGC_PREFLIGHT_SKIP_STALE_CHECK=1 CGC_SERVER_PROFILE=prod-new $extra nohup $REPO/scripts/run_server.sh --detach > $OUT/$arm.ctrl.log 2>&1 &"
  # 等 ready
  for t in $(seq 1 60); do
    s=$(curl -s --noproxy '*' -m 2 http://127.0.0.1:8080/health 2>/dev/null)
    [ "$s" = '{"status":"ok"}' ] && return 0
    sleep 2
  done
  log "  ✗ $arm 啟動失敗"; return 1
}

measure() { # measure <arm> → t/s
  local arm=$1
  curl -s --noproxy '*' -m 240 http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$NP,\"temperature\":0.0,\"cache_prompt\":true}" > "$OUT/$arm.json"
  # 測完立即停自己的 server：否則 window_free 會把「自己的 server」當「窗口被佔」，
  # 每 launch 前白等 120×10s。
  pkill -9 -f llama-server 2>/dev/null
  /opt/homebrew/bin/python3 -c "
import json,sys
d=json.load(open('$OUT/$arm.json'))
print(d.get('timings',{}).get('predicted_per_second',0))" 2>/dev/null
}

run_pair() { # run_pair <name> <envA> <envB>
  local name=$1 envA=$2 envB=$3
  log "== 組 ${name}（${PAIRS} 對，冷卻 ${COOL_S}s）=="
  for i in $(seq 1 $PAIRS); do
    for arm in A B; do
      local this=$envA
      [ $arm = B ] && this=$envB
      # 窗口守門：被佔就等（最長 20 分鐘）
      for w in $(seq 1 120); do
        window_free && break
        log "  窗口被佔，等 10s（$w/120）…"; sleep 10
      done
      log "  launch ${name}-${arm} rep${i}（$(fingerprint)）"
      launch "$name-$arm-rep$i" "$this" || continue
      local v=$(measure "$name-$arm-rep$i")
      log "  $name-$arm rep$i = ${v} t/s"
    done
    log "  冷卻 ${COOL_S}s…"
    sleep $COOL_S
  done
  # 配對比率
  /opt/homebrew/bin/python3 -c "
import json,glob,statistics
def get(p):
    try:
        d=json.load(open(p)); return d.get('timings',{}).get('predicted_per_second',0)
    except: return 0
As=sorted(get(p) for p in glob.glob('$OUT/$name-A-*.json'))
Bs=sorted(get(p) for p in glob.glob('$OUT/$name-B-*.json'))
n=min(len(As),len(Bs))
rat=[Bs[i]/As[i] for i in range(n) if As[i]>0]
print(f'== $name 判決: A中位={statistics.median(As):.2f} B中位={statistics.median(Bs):.2f} 配對比率中位={statistics.median(rat):.3f} (n={len(rat)})' if rat else f'== $name 判決: 無配對')
"
}

log "ABBA P0 三臂測速開始（$(fingerprint)）"
if [ "${RUN_GROUP:-all}" = "all" ] || [ "${RUN_GROUP:-all}" = "cP0" ]; then
    run_pair "cP0" "" "CGC_EXPERT_SKIP_READRAW=1 "
fi
if [ "${RUN_GROUP:-all}" = "all" ] || [ "${RUN_GROUP:-all}" = "p0HOT" ]; then
    run_pair "p0HOT" "CGC_EXPERT_SKIP_READRAW=1 " "CGC_EXPERT_SKIP_READRAW=1 CGC_SPAC_HOT=1 "
fi
log "完成"
