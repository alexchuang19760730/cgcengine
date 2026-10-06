#!/usr/bin/env bash
# 看板產生鏈的單一入口（圖一出來就同步全部頁面）。
#   ./scripts/check/board_pipeline.sh          重建：看板 ＋ 逐子目標頁 ＋ 總圖 ＋ 逐節點白皮書
#   ./scripts/check/board_pipeline.sh --check   只驗（rc=1 ⇒ 有漂移／有頁面落後）
# 重建模式會先跑 D15 的機器路徑（decode_board_build --promote：判準成立 ⇒ 自己升進 certified；
# 寫完重建＋--check，沒過整檔回滾）。
# 驗的九條閘門（判準各自只有一份定義）：void_check（成績面：主張了吞吐就得有產物、量具綁上、成對 log）、
# doc_claim_gate（**交付文件與 commit 訊息裡的 t/s 要有主**：對得上一個通過閘門的產物，否則紅。
# commit-msg／pre-push 兩支掛勾由 `--install-hook` 裝，裝了才會在 commit／push 當下攔；現況見 --hooks-status）、
# quote_gate（可不可引用）、budget_gate（是能力還是足跡）、
# fake25_regression（假 25 的回歸測試）、runnable_gate（這一格要的判決是不是已經存在）、
# fill_term_ab（交付 cell 上 fill term 的定價；它的**現判**由看板 D7b 當場重跑）、
# fill_split（交付 cell 上那筆代價的逐桶分解，§71：開檔／解碼／喚醒）、
# decode_board_build（看板 D1–D15）。
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python3
if [[ "${1:-}" == "--check" ]]; then
  $PY scripts/check/decode_board_build.py --check
  $PY scripts/check/mindmap_build.py --check
  # [CGC F(b) 2026-10-06] 這兩條補的是「來源 ↔ 寫回目標」那一段的洞：
  #   --check 逐子目標×options 的歸屬缺口；--drift 只改看板 YAML 的 dispositions:、
  #   而重建路徑漏跑 `--apply` 時，mindmap.json（與由它渲染的 index.html／逐節點白皮書）
  #   不會跟上 —— 而 mindmap_build.py --check 比的是 mindmap.json ↔ 渲染，兩邊都舊 ⇒
  #   一律 PASS，漂移就靜默穿過去。歷史上真的發生過一次（2026-10-06 措辭更正）。
  $PY scripts/check/mindmap_subgoal_sync.py --check
  $PY scripts/check/mindmap_subgoal_sync.py --drift
  $PY scripts/check/mindmap_void_check.py --self-test  # 成績面判準的 fixture（先驗判準本身）
  $PY scripts/check/mindmap_void_check.py              # 現判：09-30 起 rc=1 不再是「預期」——成績面上有主張就必須有據
  $PY scripts/check/doc_claim_gate.py --selftest       # 交付面判準的 fixture（先驗判準本身）
  $PY scripts/check/doc_claim_gate.py                  # 現判：交付型文件裡的每個 t/s 都要有主（未登記 ⇒ 紅）
  $PY scripts/check/doc_claim_gate.py --hooks-status   # commit-msg／pre-push 掛勾的現況（rc 一律 0：hook 不進版控）
  $PY scripts/check/quote_gate.py --selftest        # 引用判準的校準 fixture（必須全過；不過就別引用任何 t/s）
  $PY scripts/check/budget_gate.py --selftest       # 預算判準的 fixture（同上：不過就別把它當能力數）
  $PY scripts/check/fake25_regression.py --selftest # 假 25 回歸測試自己的 fixture（含反空洞：乾淨的 ≥25 必須過）
  $PY scripts/check/fake25_regression.py --check    # 全語料 decode ≥20 必須一個都不 admissible（§62）
  $PY scripts/check/runnable_gate.py --selftest     # 前置判準的 fixture（needed／duplicate／blocked 三態）
  $PY scripts/check/runnable_gate.py                # 逐格問「判決是否已存在」：宣告 needed 卻掃到判決 ⇒ 紅（§64）
  $PY scripts/check/fill_term_ab.py --selftest      # 定價端點的 fixture（七條結構閘；PRICED 的現判在 D7b 裡重跑）
  $PY scripts/check/samecell_margin.py --selftest   # 同格邊際端點的 fixture（現判也在 D7b 裡重跑）
  $PY scripts/check/fill_split.py --selftest        # fill 路徑分解的 fixture（恆等式／陰性對照／七閘）
  $PY scripts/check/fill_split.py --dir Backup/fill_split_delivery_2026-09-30   # 對冷凍產物的現判（§71）
  echo "[pipeline] check 完成"
  exit 0
fi
$PY scripts/check/quote_gate.py --selftest       # 引用閘門的校準 fixture（先驗判準本身）
$PY scripts/check/doc_claim_gate.py --selftest   # 交付面閘門的 fixture（同上）
$PY scripts/check/budget_gate.py --selftest      # 預算閘門的 fixture（必然 swap ⇒ 讀數是足跡，不是能力）
$PY scripts/check/budget_gate.py check --glob 'Backup/**/*.json' \
    --tally --json Backup/budget_gate_ledger.json || true   # 帳面登記：哪些讀數在赤字下取得（rc=1 是預期）
$PY scripts/check/fake25_regression.py --check --json Backup/fake25_regression.json \
    || true   # 假 25 回歸測試：rc=1 ⇒ 出現了 admissible 的 ≥20 讀數（那是好消息也是決策點，見該檔頭）
$PY scripts/check/runnable_gate.py --selftest    # 前置閘門的 fixture（先驗判準本身）
$PY scripts/check/fill_term_ab.py --selftest     # 定價端點的 fixture（七條結構閘）
$PY scripts/check/samecell_margin.py --selftest  # 同格邊際端點的 fixture（三態：POSITIVE／EXCLUDED／NOT_SEPARATED）
$PY scripts/check/fill_split.py --selftest      # fill 路徑分解的 fixture（恆等式／陰性對照）
# D15 的機器路徑：pending_promotion 的判準當場成立 ⇒ 自己升進 certified（沒成立＝一位元組不動；
# 寫完重建＋--check，沒過整檔回滾）。跑在 build 之前 ⇒ 「下一次重建就自己升」，不必等人抄。
$PY scripts/check/decode_board_build.py --promote
$PY scripts/check/decode_board_build.py          # 看板 ＋ docs/mindmap/subgoals/*.html|md（含 D7 引用閘門、D12 前置閘門、D15 待升級）
# [CGC F(b) 2026-10-06] 這一行是關鍵缺片：看板（YAML）→ mindmap.json 的寫回由它負責。
# 少了它，只改 YAML 的 `dispositions:` 時看板會更新、而 mindmap.json 與所有由它渲染的頁面
# 停在舊版；`mindmap_build.py --check` 抓不到（它比的是 mindmap.json ↔ 渲染，兩邊都舊）。
# 它自己會順帶跑 mindmap_build，重建路徑因此多一次等冪的重建（無害）。
$PY scripts/check/mindmap_subgoal_sync.py --apply   # YAML dispositions → mindmap.json（＋立項卡）：少了它渲染永遠落後
$PY scripts/check/mindmap_build.py               # docs/mindmap/index.html ＋ TAXONOMY
$PY scripts/check/mindmap_brief_build.py         # docs/mindmap/briefs/*（逐節點白皮書）
echo "[pipeline] 全部頁面已同步（看板 ← YAML；總圖與白皮書 ← mindmap.json）"
