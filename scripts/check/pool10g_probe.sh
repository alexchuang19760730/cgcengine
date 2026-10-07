#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# pool10g_probe.sh —— 專人手動跑：MTP 機制（temp × n-max 雙軸）／產能基線／池大小（已結論）
#
# 為什麼要手動：這台機器的量測窗口會被 WorkBuddy 本身佔用（視窗合成 + GPU 行程）。
# ★ WorkBuddy 必須**完全關閉**再跑這個腳本，否則熱態與 swap 都是被自己干擾的。
#
# ★★ 預設模式（mtp）—— 判據預先寫死，不做事後附會：
#
#   四臂 ABBA（off / on / on / off），池固定 8 GiB（=(default) 權威值）。
#   ★ 交錯的理由：記憶 10-06 實測**位置效應 ~14%**，而 MTP on/off 預期差異只有幾個百分點
#     ⇒ 只跑 off→on 量到的是「位置」不是「MTP」。10-03 那兩版讀數（0.755× vs 0.933×）
#     的差異就是這個，所以這次兩邊都堵（交錯 + attribution）。
#   ★ 判的是**比值 tg_on/tg_off**，不是絕對值。
#
#   M_gain   比值 > 1 且超出兩臂散布  ⇒ MTP 有正收益
#   M_flat   比值 ≈ 1（落在散布內）    ⇒ 攤薄與成本抵銷
#   M_loss   比值 < 1 且超出散布      ⇒ ★ MTP 淨虧
#   M_dirty  任一臂 attribution ≠ none ⇒ 不可引用，先降溫重跑
#
#   對照（記憶 10-03 誠實配對）：ON 9.306 / OFF 9.974 = **0.933×（淨負 −6.7%）**；
#   同一測試的 v1（按臂 shim、沒暖身）報 0.755× —— **那是偽影，不可引用**。
#
# ★ 池大小（8G vs 10G）不再是預設：11:2x 已量到結論 ——
#     model 13030 MiB（不可回收）＋ pool 10240 MiB ＝ 23270 vs 實體 16384
#     ⇒ OVERSUBSCRIBED by 6886 MiB；10 GiB 臂 rows=0 / max_llama_procs=0（起跑前被擋）。
#   再跑只會重現那行字 ⇒ 要重跑得明確指定 --pool10g。
#
# 用法（WorkBuddy 關閉後，在 Terminal 裡 —— zsh / bash 都行，腳本會自舉到 bash）
#
# 用法（WorkBuddy 關閉後，在 Terminal 裡）
#   cd /Users/alexchuang/Documents/flashkv-devserver
#   bash scripts/check/pool10g_probe.sh            # 預設：MTP ABBA 四臂（約 7 分鐘）
#   bash scripts/check/pool10g_probe.sh --off      # 只 MTP off 一趟乾淨基線（約 2 分鐘）
#   bash scripts/check/pool10g_probe.sh --axis       # ★ temp(0.0/0.4) × n-max(1/2/3)，9 臂交錯（約 25 分鐘）
#   bash scripts/check/pool10g_probe.sh --hot        # ★ 同 --axis 但用 hotcache 格（warm_skip 64），抓熱 cache（約 29 分鐘）
#   bash scripts/check/pool10g_probe.sh --pool10g# 已結論的 8G vs 10G（會被 budget-gate 擋）
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ★ 這個腳本用 bash 語法（陣列、`<<<` herestring）。zsh 也多數能跑，但 `set -u` 下
#   `"${arr[@]}"` 對**空陣列**在 zsh 會 `unbound variable` 而在 bash 不会 ——
#   同一份檔案、同一個 set -u，兩種 shell 的行為不同（2026-10-07 實測）。
#   所以：不是「哪個 shell 比較對」，是**根本別讓外部 shell 決定**。
#   `bash "$0" "$@"` 重啟一次，確定執行語言。
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
PY=/opt/homebrew/bin/python3
export PYTHONPATH=/opt/homebrew/lib/python3.14/site-packages
export BUDGET_GATE=warn

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="Backup/spec_tree/pool10g_${STAMP}"
mkdir -p "$OUT"

# ★★★ 機制真相（2026-10-07 實測確認，這是本腳本最關鍵的一段知識）★★★
#
# ❌ `CGC_SERVER_EXPERT_CACHE_BYTES` **不是**送進引擎的 env 名。
#    實測：帶這個 env 跑，harness 直接報
#      「環境變數沒有送達引擎（量到的會是 profile 預設值）：CGC_SERVER_EXPERT_CACHE_BYTES=...」
#      然後 fail-closed 拒跑。⇒ 兩個臂都 rc=1、14 秒。
#    原因：它只是 run_server.sh:543 的** shell 變數**來源，會被 `SERVER_ENV` 消費。
#
# ✅ 真正的兩條通道：
#   (1) `SERVER_ARGS+=(-expert-cache "$BUDGET")`  ← :1342，**主要路徑，CLI 參數**
#   (2) `CGC_EXPERT_CACHE_BYTES="$BUDGET"`         ← :1441，進 SERVER_ENV 給下游讀
#   而 `BUDGET="${CGC_SERVER_EXPERT_CACHE_BYTES:-$BUDGET_DEFAULT}"`（:543）
#   ⇒ **要改 pool 大小，設 `CGC_SERVER_EXPERT_CACHE_BYTES`（shell 變數）就會同時驅動兩條。**
#
# ⚠ 陷阱：`BUDGET_DEFAULT` 已經是 **10 GiB**（:510），是 profile 層在 :555/:582/:627
#   把它壓成 8 GiB（8589934592）。所以「10 GiB」其實是**未覆寫時的預設值**，
#   而 prod-new/prod25 兩��� profile 都不會用到它。
#   本腳本顯式設兩個值，不依賴預設。
POOL_8G=8589934592
POOL_10G=10737418240

# ★ 模式（2026-10-07 11:2x 新增）。預設是 mtp，因為那是唯一還沒答案的實驗。
#   mtp      ABBA 交錯：off/on/on/off，池固定 8 GiB（=(default) 權威值）⇒ 不需要新格
#   off      只跑一趟 MTP off 的乾淨基線（約 110s）
#   pool10g  已結論的 8G vs 10G；要跑必須明確指定（見 main() 的註解）
MODE=mtp
# TEMP/NMAX 預設空字串 ⇒ run_arm 裡的 `${TEMP:-}`／`${NMAX:-3}` 會走交付設定
# （temp 不傳 ⇒ run_server.sh 的 0.4；nmax=3）。這讓舊的 3 欄臂字串語意完全不变。
TEMP=""
NMAX=3
# HOT=1 ⇒ 每個臂用 hotcache 格（warm_skip 64），抓熱 cache performance（見 run_arm 的註解）
HOT=0
case "${1:-}" in
# --help is honoured from ANY position, not just $1: without this, `--axis --help` silently started
# a 25-minute run because $1 was `--axis`. Scanning all args costs one loop and removes the trap.
# This is the same shape as the zsh empty-array bug fixed at 11:2x -- a convenient shorthand that
# silently does something else when the caller combines it with another flag.
  "") ;;
  --help|-h) sed -n '2,32p' "$0"; exit 0 ;;
esac
for _a in "$@"; do
  case "$_a" in
    --help|-h) sed -n '2,32p' "$0"; exit 0 ;;
  esac
done
case "${1:-}" in
  ""|--mtp)      MODE=mtp ;;
  --off|--quick) MODE=off ;;   # --quick 保留為 off 的別名（沿用舊用法）
  --pool10g)     MODE=pool10g ;;
  --axis)        MODE=axis ;;  # temp x draft-n-max 雙軸，9 臂 ABBA 交錯
  --hot)         MODE=axis; HOT=1 ;;  # ★ 同 --axis，但每臂用 hotcache 格（warm_skip 64）
  *) echo "未知選項：$1（可用：--mtp / --off / --pool10g / --axis）" >&2; exit 2 ;;
esac

# ── 陷阱六：環境閘門（listener／殘留行程／熱態／壓縮機）────────────────────
say() { printf '%s\n' "$*"; }
hr()  { say "──────────────────────────────────────────────────────────────────────"; }

preflight() {
  hr; say "[1/5] 環境閘門"
  local n8080 nsrv
  n8080=$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | wc -l | tr -d ' ')
  nsrv=$(pgrep -x llama-server 2>/dev/null | wc -l | tr -d ' ')
  say "  8080 listener : $n8080   llama-server : $nsrv"
  if [ "$n8080" != "0" ] || [ "$nsrv" != "0" ]; then
    say "  ✗ 有殘留 server/listener。先確認不是別條 session 的行程 —— 那是別人的東西。"
    say "    8080=$(lsof -nP -iTCP:8080 -sTCP:LISTEN)   srv=$(pgrep -fl llama-server)"
    exit 2
  fi
  say "  熱態         : $(/usr/bin/pmset -g therm 2>/dev/null | tail -1)"
  say "  ★ 確認 WorkBuddy 已關閉（視窗合成會吃 GPU）"
  hr; say "[2/5] 壓縮機等 QUIET（最多 180s）"
  "$PY" scripts/check/compressor_pressure.py --seconds 12 \
      --json "$OUT/compressor_before.json" 2>&1 | tail -3
  if ! "$PY" scripts/check/compressor_pressure.py --seconds 12 --require 2>/dev/null; then
    say "  ✗ 壓縮機不安靜。**不要**加 CGC_WINDOW_OVERRIDE 硬跑（operator 立規：不得為量測加它）。"
    say "    等它降下來再跑，或關掉其他吃 CPU 的程式。"
    exit 2
  fi
  say "  ✓ 壓縮機安靜"
}

# ── 單臂執行 ────────────────────────────────────────────────────────────
# $1=arm tag  $2=pool bytes  $3=MTP(0|1)
#
# ★★ 2026-10-07 第一次實跑的三個 fail-closed，兩個是本腳本的缺陷、第三個是口徑阻礙：
#
#   (1) `cell_contract` 釘死 (default) 的 `expert_cache_bytes: 8589934592`
#       —— 那是測試卡 §2.5 對 (default) 的**權威宣告值**。
#       ⇒ **operator 已簽：新增一格 `pool10g`（§2.5 的 `cells` 區塊）**，
#         與 (default) 逐鍵相同、只有 expert_cache_bytes 不同 ⇒ (default) 一個鍵都沒動。
#   (2) MTP 臂需要 `--spec-type draft-mtp`
#       —— `CGC_SERVER_MTP=1` 只匯出 MTP 的 env block，spec 本身是 llama-bench 的旗標，
#       沒有它量到的不是 MTP（llama_bench_matrix:1793 fail-closed）。⇒ 已補。
#   (3) `--cell '(default)'` 會被 cell_contract 擋下（`resolve_cell` 只認 `None`）
#       —— ★ 這是 cell_contract 的一個 bug：`cell_names()` 一直把 "(default)" 列為可選，
#       所以「讀清單再把名字傳回去」這個合理動作會被拒。已修（None 與 "(default)" 同義）。
#
# ⇒ 現在兩臂都跑同樣的 `--cell pool10g` / 8GiB 臂跑 `--cell (default)`，
#    **差異只有 expert_cache_bytes 這一個鍵**，這正是這個實驗要控制的東西。
run_arm() {
  local tag="$1" pool="$2" mtp="$3"
  # 選格：8 GiB 用 (default)（它就是 (default) 的權威值），10 GiB 用新增的 pool10g 格。
  # ⛔ 不傳 --cell '(default)' 給 8G 臂：harness 預設就是那一格，不傳最不容易出錯。
  # ★★ CELL_AXIS（2026-10-07）：`--hot` 模式把每個臂切到 hotcache 格（warm_skip 64）。
  #   理由見測試卡該格的 _why_not_default：MTP 臂的前 1-2 個 rep 明顯低於後 3 個（斜坡 8-13%），
  #   而 MTP-off 臂沒有斜坡 ⇒ 斜坡是 MTP draft 碰到的、off 臂永不碰的專家造成的池填充。
  #   `--warm-skip N`丟掉的是**每個** rep 的前 N 個不計時 token（llama-bench.cpp:474-488），
  #   正是為此設計 ⇒ 它讓 avg_ts 量到熱狀態，而 avg_ts 正是 gate 讀的欄位。
  local cellarg=()
  if [ "$pool" = "$POOL_10G" ]; then cellarg=(--cell pool10g)
  elif [ "$HOT" = "1" ];     then cellarg=(--cell hotcache)
  fi
  # ★★ 2026-10-07 14:5x：`--cell` 只帶「格的名字」，**不带格的值**。
  #   14:49 那趟 `--hot` 九臂全被 cell_contract 擋下：
  #     「warm_skip: 實際 0 ≠ 權威 64」—— 因為 harness 的 `--warm-skip` 預設來自
  #     `_BENCH_DEFAULTS`，而**沒有任何機會**把 cell 的值餵回去。
  #   `cell_contract.bench_flags(cell)`（:622）就是那個轉換器，它甚至會印出完整命令；
  #   ★ 它有四個呼叫者（skip_probe / rho_price_authrow / l2010_window / CLI），
  #   ★★ **harness.py 不在裡面** ⇒ 每個「格宣告了但命令列表達不到的值」都會這樣被擋。
  #   ⇒ 這裡直接呼叫它，讓 cell 的值與 `--cell` 一起進 argv。
  if [ ${#cellarg[@]} -gt 0 ]; then
    local cname="${cellarg[1]}"
    local cflags
    # shellcheck disable=SC2207  # 這裡用 command substitution 展開陣列是刻意的
    cflags=$("$PY" - "$cname" <<'PYEOF2'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "check"))
import runpy
m = runpy.run_path("scripts/check/cell_contract.py", run_name="not_main")
name, cell = m["resolve_cell"](m["load_contract"](), sys.argv[1])
flags, warn = m["bench_flags"](cell)
if warn:
    sys.stderr.write("cell_contract warning: " + "; ".join(warn) + "\n")
print("\n".join(flags))
PYEOF2
)
    local -a cflagarr=()
    [ -n "$cflags" ] && cflagarr=($cflags)
    cellarg+=(${cflagarr[@]+"${cflagarr[@]}"})
    say "  cell ${cname} ⇒ flags: ${cflagarr[*]:-(none)}"
  fi

  local envs="CGC_SERVER_EXPERT_CACHE_BYTES=${pool};CGC_SERVER_MTP=${mtp}"
  # [CGC 2026-10-07] TEMP_AXIS. The MTP accept step compares the draft against the TARGET's
  # SAMPLED token, so accept rate == target sharpness (llama-bench.cpp:2810-2818, verbatim). Every
  # MTP reading in memory (10-03's 0.755/0.933, 12:0x's 0.853) was taken at temp=0.4, which is a
  # FLAT distribution -- so "accept rate is only 0.25-0.66" was never evidence about the head.
  # TEMP=greedy is the arm that can move the verdict from "MTP is a loss" to "our production temp
  # is too flat", and those are different conclusions.
  # ⚠️ The value must be last-wins-correct: run_server.sh emits `--temp 0` at argv[34] for the MTP
  # path itself and `--temp <ours>` at argv[40]; llama_bench_matrix.sampling_env takes the LAST
  # occurrence (verified 2026-10-07: TEMP=0.0 -> sampling_env CGC_SERVER_TEMP=0.0). Verified.
  [ -n "$TEMP" ] && envs="${envs};CGC_SERVER_TEMP=${TEMP}"
  # CGC_SERVER_SEED 釘住：逐位元 A/B 必須固定 seed（記憶 10-04 立的規矩）。
  [ "$mtp" = "1" ] && envs="${envs};CGC_SERVER_SEED=1"
  # [CGC 2026-10-07] MTP mechanism instruments, MTP arms only.
  #   LLAMA_BENCH_SPEC_DUMP -> llama-bench prints `CGC-BENCH-ACCEPT ... emit_per_round=n_done/n_rounds`
  #     (llama-bench.cpp:4151). ALL-COUNTER, no clock: the division cancels time entirely, so it
  #     survives a swap-tainted window and is the ONLY half of the diagnosis that does.
  #   CGC_MTP_PERF -> speculative.cpp:3313 prints acc_rate / emit_tok_per_round / ms_per_round.
  #     ⚠️ `ms_per_round` and `t_draft_ms` are WALL CLOCK (common_time_meas, :3112) => swap inflates
  #     them, so under BUDGET_GATE=warn the COST half is not trustworthy while `acc_rate` is.
  #     Read it as a bound, not a measurement. Both are on (run_server.sh:1576 allowlists both).
  [ "$mtp" = "1" ] && envs="${envs};LLAMA_BENCH_SPEC_DUMP=1;CGC_MTP_PERF=1"
  say ""
  say "▶ arm ${tag}   pool=$((pool/1073741824))GiB  MTP=$mtp   cell=${cellarg[1]:-(default)}"
  say "  env: $envs"
  # --charter 的正確用法是 `waive` + `--waive-reason`（harness:482），不是 `none`。
  # 這張卡是「推翻既有判斷」的測量，不是新增交付，所以走急件豁免並留下理由。
  local reason
  reason="pool10g probe: 8GiB-vs-10GiB capacity-ceiling test, overturns or confirms run_server.sh:568-574"
  # (2) MTP 臂一定要 spec-type，否則量到的不是 MTP。
  # [CGC 2026-10-07] NMAX_AXIS. draft-n-max is a llama-bench CLI flag (not an env knob), and it is
  # the only lever that cuts the per-token forward count: speculative.cpp:322 runs one
  # `llama_decode(ctx_dft)` PER DRAFTED TOKEN, so forwards/token = (drafted + rounds)/gen_tokens,
  # measured median 2.03 at n-max=3 vs 1.00 for plain decode (12:6x). Lowering n-max removes
  # forwards that were mostly rejected anyway (draft_ratio was 0.25-0.66).
  # ⚠️ n-max and temp INTERACT, which is why they are swept together rather than one after the
  # other: a flat target accepts fewer drafts, so the best n-max at temp=0.4 is not the best at 0.
  local spec=()
  if [ "$mtp" = "1" ]; then spec=(--spec-type draft-mtp --spec-draft-n-max "${NMAX:-3}"); fi
  local t0 t1 rc
  t0=$(date +%s)
  "$PY" scripts/check/harness.py bench \
      --arm "prod-new:${envs}" \
      --charter waive --waive-reason "$reason" \
      ${cellarg[@]+"${cellarg[@]}"} ${spec[@]+"${spec[@]}"} \
      --json "$OUT/${tag}.json" > "$OUT/${tag}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  say "  rc=$rc   $((t1-t0))s   log=$OUT/${tag}.log"
  # harness 的 exit code 會把 OOM 與量測失敗混在一起，所以另外抓特徵字串。
  if grep -qE "OutOfMemory|Insufficient Memory|rc=-6|rc=-11" "$OUT/${tag}.log" 2>/dev/null; then
    say "  ✗ 抓到 OOM / Insufficient Memory 特徵"
    grep -oE "OutOfMemory|Insufficient Memory|rc=-?[0-9]+" "$OUT/${tag}.log" | sort -u | head -4
  fi
  return $rc
}

# ── 讀回結果 ────────────────────────────────────────────────────────────
report_arm() {
  local tag="$1" pool="$2" mtp="$3"
  # SKIP 與「量了但失敗」是兩件事：前者結構上不存在這個實驗，後者有產物可讀。
  if [ -f "$OUT/${tag}.skipnote" ]; then
    say "  ${tag}  SKIPPED（by design，見 .skipnote）—— 沒有產物，這不是失敗也不是通過"
    return 0
  fi
  "$PY" - "$OUT/${tag}.json" "$(($pool/1073741824))" "$mtp" "$OUT/${tag}.stderr.log" <<'PYEOF' 2>&1
import json, sys
p, gib, mtp, _stderr_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  {gib}GiB mtp={mtp}: 無法讀結果 ({e})"); sys.exit(0)
a = (d if isinstance(d, list) else [d])[0] if isinstance(d, (list, dict)) else None
if not isinstance(a, dict):
    print(f"  {gib}GiB mtp={mtp}: 產物格式非預期"); sys.exit(0)
rows = [r for r in (a.get("rows") or []) if isinstance(r, dict)]
pp = next((r.get("avg_ts") for r in rows if r.get("n_gen") == 0), None)
tg = next((r.get("avg_ts") for r in rows if r.get("n_gen") == 128), None)
at = a.get("attribution")
av = at.get("verdict") if isinstance(at, dict) else None
why = at.get("why") if isinstance(at, dict) else ""
print(f"  {gib}GiB mtp={mtp}: pp={pp} tg={tg}  attribution={av}  {why}")
c = a.get("cell") or {}
print(f"      contract.ok={(a.get('contract') or {}).get('ok')}  cell={c.get('named_cell')}"
      f"  prompt={c.get('prompt')}")
# [CGC 2026-10-07] The 12:2x B-plan field, reported so ONE run answers both questions. A
# product written BEFORE the fix has no `oversubscribed` key at all -- which is how you tell
# "the box was clean" from "the tool that would have said so is older than the fix".
if isinstance(at, dict) and "oversubscribed" in at:
    print(f"      B-plan: oversubscribed={at.get('oversubscribed')}  "
          f"(swap_growth={at.get('swap_growth_mb')} MiB, max_swap={at.get('max_swap_mb')} MiB)")
else:
    print("      B-plan: (no `oversubscribed` key) => this arm predates the 2026-10-07 fix, so its"
          " `none` may be the false-clean the fix was written to catch.")
# [CGC 2026-10-07] MTP mechanism, split by whether swap can touch it. CGC-BENCH-ACCEPT and the
# token ratios are divisions of COUNTS, so the clock cancels and they survive a tainted window;
# ms_per_round / t_draft_ms are wall clock and do not. Printing them together without this split is
# how a swap-inflated cost gets read as "the draft head is expensive".
import os as _os, re as _re
_stderr = _stderr_path
try:
    _txt = open(_stderr, encoding="utf-8", errors="replace").read()
except Exception:
    _txt = ""
if _txt:
    _acc = _re.findall(r"CGC-BENCH-ACCEPT[^\n]*", _txt)
    _perf = _re.findall(r"CGC-MTP-PERF[^\n]*", _txt)
    if _acc:
        for _l in _acc[-2:]:
            print("      MTP counts (swap-immune):", _l.strip())
    if _perf:
        for _l in _perf[-1:]:
            print("      MTP perf (acc_rate valid / ms_per_round NOT):", _l.strip()[:200])
    if not _acc and not _perf:
        print("      MTP: no CGC-BENCH-ACCEPT / CGC-MTP-PERF line in this arm"
PYEOF
}

verdict() {
  hr
  if [ "$MODE" = "axis" ]; then
    say "══════ 預先寫死的判讀（temp × draft-n-max 雙軸）══════"
    say "  ★ 這九臂是交錯的，不是「一個 temp 一塊」——位置效應實測 ~14%，與效應同量級。"
    say "  ★ 兩組 off 臂（a1/a9 同為 temp0.4、a4 為 temp0.0）可交叉驗證漂移："
    say "     若 off_a 與 off_b 差很多 ⇒ 本run 有漂移，比值全部不可引用。"
    say ""
    say "  A_gain  greedy(temp=0) 的 MTP 比值 > 1 且超出散布"
    say "         ⇒ ★ 生產 temp=0.4 選得太平；MTP 的問題是採樣設定不是 head"
    say "  A_flat  greedy 比值 ≈ 1（落在散布內）"
    say "         ⇒ temp 不是槓桿；成本在每 token 的 forward 數（2.03 vs 1.00）"
    say "  A_loss  greedy 比值 < 1 且超出散布"
    say "         ⇒ MTP 在 greedy 下也是淨虧 ⇒ head/forward 成本是結構性的"
    say "  A_dirty 任一臂 attribution ≠ none"
    say "         ⇒ 不可引用，先降溫重跑（**不要**加 BUDGET_GATE=warn 硬跑）"
    say ""
    say "  ★ n-max 的讀法：同一 temp 下 n1/n2/n3 的 tg 應隨 forward 數下降而上升。"
    say "     若 n1 沒比 n3 快 ⇒ forward 數不是瓶頸，桿桿方向要換。"
    say ""
    say "  ★★ 交付口徑提醒：服務端生產是 temp=0.4（run_server.sh:1412）。"
    say "     所以 A_gain 也不代表交付會變快——它代表**診斷**變了（是 temp 選得太平）。"
    hr
    return 0
  fi
  if [ "$MODE" = "mtp" ]; then
    say "══════ 預先寫死的判讀（MTP on/off）══════"
    say "  ★ 這四臂是 ABBA（off/on/on/off），所以**位置效應被對消**。看的是配對中位，不是四個平均。"
    say "  ★ 判的是**比值 tg_on/tg_off**，不是絕對值。"
    say ""
    say "  M_gain   tg_on/tg_off > 1 且超出兩臂散布  ⇒ MTP 在這台機器上有正收益"
    say "  M_flat   比值 ≈ 1（落在散布內）            ⇒ MTP 沒賺也沒賠（攤薄與成本抵銷）"
    say "  M_loss   比值 < 1 且超出散布              ⇒ ★ MTP 淨虧（成本大於攤薄）"
    say "  M_dirty  任一臂 attribution ≠ none        ⇒ 不可引用，先降溫重跑"
    say ""
    say "  ★ 為什麼值得量：記憶 10-03 誠實配對得 **ON 9.306 / OFF 9.974 = 0.933×（淨負 −6.7%）**，"
    say "    而同一個測試的另一版（按臂 shim、v1）報 0.755× —— **兩版差異就是位置／熱態**，"
    say "    所以這次用 ABBA + attribution 雙重把關。舊的 8.698 那筆是偽影，不可引用。"
    say "  ★ 池大小固定 8 GiB：10 GiB 已被 budget-gate 以 6886 MiB 靜態超訂擋下（11:19 量到），"
    say "    與本實驗無關。"
  else
    say "══════ 預先寫死的判讀（池大小）══════"
    say "  H_oom   10 GiB 起不來（OOM）        ⇒ 確認放不進"
    say "  H_nowin 10 GiB 活著但沒贏           ⇒ 確認容量已過熱集"
    say "  H_gain  贏且超出散布                ⇒ ★ 推翻「熱集已裝進 8 GiB」"
    say ""
    say "★ 2026-10-07 11:2x：**H_oom 已成立**，budget-gate 算出靜態需求 23270 MiB vs 實體 16384"
    say "  ＝ OVERSUBSCRIBED by 6886 MiB，10 GiB 臂 rows=0、max_llama_procs=0（起跑前被擋）。"
  fi
  hr
  say "完整結果："
  "$PY" scripts/check/bench_ledger.py --json > "$OUT/ledger_snapshot.json" 2>/dev/null || true
  say ""
  say "★ 收工後請把這幾件事記進 memory："
  say "  1. 各臂的 pp/tg 與 attribution（quotable 或否）"
  say "  2. 命中哪一個結局（mtp 模式：M_gain / M_flat / M_loss / M_dirty）"
  say "  3. 熱態與壓縮機讀數（$OUT/compressor_before.json、compressor_after.json）"
  say ""
  say "產物：$OUT"
}

# ── main ────────────────────────────────────────────────────────────────
main() {
  say "═══ pool10g probe  $(date '+%Y-%m-%d %H:%M:%S')  [mode=${MODE}] ═══"
  say "WorkBuddy 必須已關閉。產物 → $OUT"
  preflight

  hr; say "[3/5] 跑臂"
  # ★★ 2026-10-07 11:2x 模式改寫。理由：10 GiB 已經有結論了。
  #
  # 【已結論，不再重跑】pool-gate 的硬數字：
  #     model 13030 MiB（load_mode=none ⇒ 不可回收）＋ pool 10240 MiB ＝ 23270 vs 實體 16384
  #     ⇒ OVERSUBSCRIBED by 6886 MiB ⇒ 只能靠記憶體壓縮+swap，數字不可引用。
  #   11:19 那趟 10 GiB 臂 rows=0 / max_llama_procs=0 ⇒ **起跑前**就被擋，連 OOM 都沒走到。
  #   ⇒ 再跑只會重現這行字，所以 pool10g 模式改成要明確指定。
  #
  # 【仍未答】使用者的原始問題是「池 10GB、MTP on/off 會快多少」，而 **MTP on/off 那兩臂
  #   從來沒有讀數**：11:0x 那趟 MTP 臂被 cell_contract 擋、11:2x 手動跑的是 MTP off。
  #   而 8 GiB = `(default)` 的權威值 ⇒ **MTP 這個實驗不需要新格**，是產能。
  #
  # 【位置效應】記憶 10-06 實測位置效應 ~14%，而 MTP on/off 的預期差異只有幾個百分點
  #   ⇒ **必須交錯（ABBA），不能只跑 off→on**，否則量到的是「位置」不是「MTP」。
  local arms=()
  case "$MODE" in
    mtp)
      arms=("m1_off_a:${POOL_8G}:0" "m2_on_a:${POOL_8G}:1"
            "m3_on_b:${POOL_8G}:1" "m4_off_b:${POOL_8G}:0")
      ;;
    off)
      arms=("base_off:${POOL_8G}:0")
      ;;
    pool10g)
      arms=("ctl8g:${POOL_8G}:0" "try10g:${POOL_10G}:0")
      ;;
    # [CGC 2026-10-07] The two-axis sweep: temp x draft-n-max, ABBA-interleaved.
    #
    # WHY THIS SHAPE. Every MTP reading in memory was taken at temp=0.4, a FLAT distribution, and
    # the accept step compares the draft against the TARGET's sampled token (llama-bench.cpp:2810
    # -2818) -- so "draft_ratio is only 0.25-0.66" was never evidence about the head. The greedy arm
    # is the one that can move the verdict from "MTP is a loss on this box" to "our production temp
    # is too flat", and those lead to opposite decisions about draft-n-max.
    #
    # ★ The two axes INTERACT (accept rate sets how many drafts are worth making), so sweeping them
    #   one-at-a-time would attribute an interaction effect to whichever axis ran second.
    #
    # ★ Interleaving, not one block per temp: the 10-06 measurement put the POSITION effect at ~14%,
    #   which is the same order as the effects being measured. Both temperatures' off arms sit at
    #   the two ends and again in the middle, so a drift shows up as off_a != off_b rather than as
    #   a spurious temp effect. On arms alternate around them.
    #
    # Verdicts (pre-registered, all three count as information):
    #   A_gain   greedy MTP ratio > 1 beyond spread => the flat production temp was the problem
    #   A_flat   greedy ratio ~ 1                   => temp is not the lever; forward count is
    #   A_loss   greedy ratio < 1 beyond spread    => MTP is a loss even at temp=0
    #   A_dirty  any arm attribution != none       => not quotable, cool down and re-run
    axis)
      arms=(
        "a1_t04_off:${POOL_8G}:0:0.4"
        "a2_t00_n3:${POOL_8G}:1:0.0:3"
        "a3_t04_n1:${POOL_8G}:1:0.4:1"
        "a4_t00_off:${POOL_8G}:0:0.0"
        "a5_t04_n3:${POOL_8G}:1:0.4:3"
        "a6_t00_n1:${POOL_8G}:1:0.0:1"
        "a7_t00_n2:${POOL_8G}:1:0.0:2"
        "a8_t04_n2:${POOL_8G}:1:0.4:2"
        "a9_t04_off2:${POOL_8G}:0:0.4"
      )
      ;;
    *) say "未知模式：$MODE（可用：mtp / off / pool10g / axis）" >&2; exit 2 ;;
  esac
  say "模式 $MODE ⇒ ${#arms[@]} 臂：${arms[*]}"

  local results=()
  # [CGC 2026-10-07] 這裡原本是 `for spec in ...` 而 `spec` 在 run_arm 內是陣列名 —— 撞名。
  # zsh 的函式作用域預設不繼承 local，於是 `spec` 漏到全域、下一輪迭代讀到被覆寫的值。
  # 改成 `_armspec`／`local`，兩層都對。
  local _armspec tag pool mtp
  # [CGC 2026-10-07] Arm spec gained two optional fields: tag:pool:mtp[:temp[:nmax]].
  # Both default to the DELIVERY config when absent (temp unset => run_server.sh's 0.4, nmax=>3),
  # so every pre-existing arm string keeps its exact old meaning. `read` leaves the extra vars
  # EMPTY rather than unset when the field is missing, which is why the defaults are applied with
  # `:-` below and not by initialising them once outside the loop.
  for _armspec in ${arms[@]+"${arms[@]}"}; do
    IFS=':' read -r tag pool mtp _t _n <<< "$_armspec"
    TEMP="$_t" NMAX="$_n" run_arm "$tag" "$pool" "$mtp"; results+=("$tag:$?")
  done

  hr; say "[4/5] 逐臂讀數"
  for _armspec in ${arms[@]+"${arms[@]}"}; do
    IFS=':' read -r tag pool mtp _t _n <<< "$_armspec"
    report_arm "$tag" "$pool" "$mtp"
  done

  hr; say "[5/5] 收尾"
  "$PY" scripts/check/compressor_pressure.py --seconds 12 \
      --json "$OUT/compressor_after.json" 2>&1 | tail -2
  say "臂結果：${results[*]}"
  verdict
}

main "$@"
