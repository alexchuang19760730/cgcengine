#!/bin/bash
# q36_ab.sh — 共用模型級 A/B 定案腳本（§13.110 方法，兩模型同 env 自動交錯）
#
# 方法（定案自 §13.110）:
#   1. 乾淨窗檢查（load1 < 2 才跑；--wait N 每 30s 輪詢最多 N 次，--force 跳過）
#   2. 正反序交錯 [A,B,B,A] × rounds（抵銷時間漂移）
#   3. 兩側同生產 env（slots96/pool96/各自模型 top96 profile/sync preload/RW8/
#      trust-receipt）——模型是唯一變數
#   4. 每輪抓 tok/s + decode + ttft；coherence gate（!!!!/nan/unused/缺 footer
#      → 該輪排除 + [warn]）
#   5. 輸出：輪次表 + 各模型 mean ± std + 配對勝者
#   6. 決定性證據：decode 時間（不含 TTFT 的純每 token 成本）對比——§13.110
#      證明 decode 的 mean+spread 是比 tok/s 更穩的判據（真效應的 decode
#      時間又低又穩；load 偽影則又高又抖）
#
# 用法:
#   MODEL_A=/path/a.gturbo MODEL_B=/path/b.gturbo \
#   MODEL_A_NAME=ga MODEL_B_NAME=e4 \
#   bash temp/qwen36_bench/q36_ab.sh [--max-new 128] [--rounds 2] \
#       [--msg temp/qwen36_bench/msg_code.json] [--wait 10] [--force]
#
# 輸出 exit: 0 = 完成（verdict 在 stdout）；1 = 未跑（load 髒/參數錯/全 degraded）
set -euo pipefail

MAX_NEW=128
ROUNDS=2
MSG="${Q36_AB_MSG:-temp/qwen36_bench/msg_code.json}"
FORCE=0
WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --msg) MSG="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --wait) WAIT="$2"; shift 2 ;;
    *) echo "error: unknown arg $1" >&2; exit 2 ;;
  esac
done

: "${MODEL_A:?error: set MODEL_A (path to model A)}"
: "${MODEL_B:?error: set MODEL_B (path to model B)}"
NAME_A="${MODEL_A_NAME:-$(basename "$MODEL_A")}"
NAME_B="${MODEL_B_NAME:-$(basename "$MODEL_B")}"

[ -d "$MODEL_A" ] || { echo "error: MODEL_A dir not found: $MODEL_A" >&2; exit 1; }
[ -d "$MODEL_B" ] || { echo "error: MODEL_B dir not found: $MODEL_B" >&2; exit 1; }
[ -f "$MSG" ] || { echo "error: messages file not found: $MSG" >&2; exit 1; }
case "$ROUNDS" in [1-4]) ;; *) echo "error: --rounds must be 1-4" >&2; exit 1 ;; esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${ROOT}/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI"
[ -x "$BIN" ] || BIN="${ROOT}/prime-agent-worktrees/turbo-fieldfare/.build/release/TurboFieldfareCLI"
[ -x "$BIN" ] || { echo "error: no release TurboFieldfareCLI build" >&2; exit 1; }

# ---- 乾淨窗檢查 ----
load1() { uptime | grep -oE 'load averages?: [0-9.]+' | grep -oE '[0-9.]+$'; }
if [ "$FORCE" -ne 1 ]; then
  n=0
  while [ "$n" -le "$WAIT" ]; do
    L=$(load1)
    if awk "BEGIN{exit !($L < 2.0)}"; then break; fi
    if [ "$n" -eq "$WAIT" ]; then
      echo "error: load $L still >= 2.0 after ${WAIT}x30s wait — re-run in a clean window (--force to override)" >&2
      exit 1
    fi
    echo "[q36_ab] load $L — waiting for clean window (${n}/${WAIT})" >&2
    sleep 30
    n=$((n+1))
  done
fi

# ---- 交錯順序 [A,B,B,A] × rounds ----
declare -a ORDER=()
r=0
while [ "$r" -lt "$ROUNDS" ]; do ORDER+=(A B B A); r=$((r+1)); done

echo "=== A/B: $NAME_A (A) vs $NAME_B (B)  max-new=$MAX_NEW rounds=$ROUNDS ==="
declare -a RUN_M RUN_TPS RUN_DEC RUN_TTFT
R=1
for m in "${ORDER[@]}"; do
  if [ "$m" = "A" ]; then M="$MODEL_A"; N="$NAME_A"; else M="$MODEL_B"; N="$NAME_B"; fi
  PROF="$M/profiles/top96_code_prose.json"
  [ -f "$PROF" ] || PROF="$M/profiles/top96_code.json"
  OUT=$(TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 \
    TURBO_FIELDFARE_HOT_POOL_EXPERTS=96 TURBO_FIELDFARE_HOT_POOL_PROFILE="$PROF" \
    TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 \
    "$BIN" --model "$M" --trust-receipt --messages-file "$MSG" --max-new "$MAX_NEW" \
      --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1) || true
  TPS=$(printf '%s\n' "$OUT" | grep -oE 'tok/s=[0-9.]+' | tail -1 | cut -d= -f2)
  DEC=$(printf '%s\n' "$OUT" | grep -oE 'decode=[0-9.]+s' | tail -1 | sed 's/decode=//;s/s//')
  TTF=$(printf '%s\n' "$OUT" | grep -oE 'ttft=[0-9.]+' | tail -1 | cut -d= -f2)
  COH="ok"
  if ! printf '%s\n' "$OUT" | grep -q '\[stop='; then COH="NOSTOP"
  elif printf '%s\n' "$OUT" | grep -qE '!!!!|<unused>|nan'; then COH="DEGRADED"; fi
  [ -n "$TPS" ] || TPS="ERR"
  [ -n "$DEC" ] || DEC="-"
  [ -n "$TTF" ] || TTF="-"
  echo "  R$R [$N] tok/s=$TPS decode=${DEC}s ttft=${TTF}s coh=$COH"
  if [ "$COH" != "ok" ]; then echo "    [warn] round degraded ($COH) — excluded"; TPS="ERR"; DEC="-"; TTF="-"; fi
  RUN_M[$R]="$N"; RUN_TPS[$R]="$TPS"; RUN_DEC[$R]="$DEC"; RUN_TTFT[$R]="$TTF"
  R=$((R+1))
done

# ---- 統計 ----
stat_of() { # name → "mean std n" over tok/s; then decode mean std
  local name="$1"
  local ts=() ds=() tn=0 dn=0 tsum=0 dsum=0
  for ((i=1; i<=${#ORDER[@]}; i++)); do
    [ "${RUN_M[$i]}" = "$name" ] || continue
    [ "${RUN_TPS[$i]}" = "ERR" ] || { ts+=("${RUN_TPS[$i]}"); tsum=$(awk "BEGIN{printf \"%.4f\", $tsum + ${RUN_TPS[$i]}}"); tn=$((tn+1)); }
    [ "${RUN_DEC[$i]}" = "-" ] || { ds+=("${RUN_DEC[$i]}"); dsum=$(awk "BEGIN{printf \"%.4f\", $dsum + ${RUN_DEC[$i]}}"); dn=$((dn+1)); }
  done
  local tmean dmean tstd dstd
  if [ "$tn" -gt 0 ]; then
    tmean=$(awk "BEGIN{printf \"%.3f\", $tsum / $tn}")
    tvar=0
    for v in "${ts[@]}"; do tvar=$(awk "BEGIN{printf \"%.4f\", $tvar + ($v - $tmean)^2}"); done
    tstd=$(awk "BEGIN{printf \"%.3f\", sqrt($tvar / $tn)}")
  else tmean="-"; tstd="-"; fi
  if [ "$dn" -gt 0 ]; then
    dmean=$(awk "BEGIN{printf \"%.2f\", $dsum / $dn}")
    dvar=0
    for v in "${ds[@]}"; do dvar=$(awk "BEGIN{printf \"%.4f\", $dvar + ($v - $dmean)^2}"); done
    dstd=$(awk "BEGIN{printf \"%.2f\", sqrt($dvar / $dn)}")
  else dmean="-"; dstd="-"; fi
  echo "$tmean $tstd $tn $dmean $dstd $dn"
}

echo "=== summary ==="
read -r TA TA_S TN DA DA_S DN <<< "$(stat_of "$NAME_A")"
read -r TB TB_S TN2 DB DB_S DN2 <<< "$(stat_of "$NAME_B")"
echo "  $NAME_A (A): tok/s=$TA ± $TA_S (n=$TN)   decode=${DA}s ± $DA_S (n=$DN)"
echo "  $NAME_B (B): tok/s=$TB ± $TB_S (n=$TN2)  decode=${DB}s ± $DB_S (n=$DN2)"

# ---- verdict ----
if [ "$TA" = "-" ] || [ "$TB" = "-" ]; then
  echo "verdict: [warn] one side has no valid rounds — cannot decide"
  exit 1
fi
DELTA=$(awk "BEGIN{printf \"%.1f\", (($TB - $TA) / $TA) * 100}")
DDELTA=$(awk "BEGIN{printf \"%.1f\", (($DA - $DB) / $DA) * 100}")  # >0 = B decode 快
if awk "BEGIN{exit !(($DELTA > 2.0))}"; then
  echo "verdict: B ($NAME_B) wins by ${DELTA}% tok/s"
elif awk "BEGIN{exit !(($DELTA < -2.0))}"; then
  echo "verdict: A ($NAME_A) wins by ${DELTA#-}% tok/s"
else
  echo "verdict: within noise (<2%) on tok/s"
fi
echo "  decode 對比（§13.110 決定性證據）: A=${DA}s±${DA_S}  B=${DB}s±${DB_S}  （B 快 ${DDELTA}%）"
if [ "$DA" != "-" ] && [ "$DB" != "-" ]; then
  if awk "BEGIN{exit !($DDELTA > 3.0)}"; then echo "  → decode 層面 B 明確較快（差 ${DDELTA}%）——tok/s 若僅 <2% 差距，以 decode 為準（TTFT 稀釋）"
  elif awk "BEGIN{exit !($DDELTA < -3.0)}"; then echo "  → decode 層面 A 明確較快"
  else echo "  → decode 層面接近（|Δ|<3%）"; fi
fi
exit 0
