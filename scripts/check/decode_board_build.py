#!/usr/bin/env python3
"""decode 子目標看板的產生器（單一來源 ⇒ 三個產物）。

唯一來源
--------
  scripts/check/decode_board_2026-09-29.yaml   ← 判定、預期、立項、結案狀態（人寫）
  docs/mindmap/mindmap.json                    ← 節點的 name/tier/sub/targets/res/docs（機器寫）

產物（全部**生成**，不要手改）
------------------------------
  docs/mindmap/prefill250decode20.html         子目標看板（掛進 mindmap 總圖）
  docs/mindmap/subgoals/<id>.html|.md          逐子目標頁（預期／現況／驗收／立項／結案 ＋ 對應節點與 log 連結）
  docs/mindmap/subgoals/index.html|.md         左表：15 個子目標 × 結案狀態

用法
----
  python3 scripts/check/decode_board_build.py             # 生成全部產物
  python3 scripts/check/decode_board_build.py --check      # 只驗：YAML 合法 ＋ 無漂移 ＋ 產物未落後（rc=1 有問題）
  python3 scripts/check/decode_board_build.py --selftest   # 內建案例（不碰 repo 檔案）

漂移（drift）＝ --check rc=1 的六類
----------------------------------
  D1 引用了 mindmap 裡不存在的節點
  D2 有「未結案」節點既沒被子目標覆蓋、也沒被分類（＝完備性稽核）
  D3 欄位不合規（缺預期／分類缺欄／立項不存在…）
  D4 產物落後於 YAML（有人手改 HTML，或忘了重跑）
  D5 立項（charter）指到的檔案不存在
  D6 標了「結案」卻不滿足結案規則（prod-new ＋ harness bench ＋ 達標）——唯一例外是「結案（排除）」要有出處
  D7 引用閘門（quote_gate）：子目標宣告的可引用性必須與閘門**當場**判的一致；且「結案」不得
     建立在不可引用的讀數上（`attribution=none` 不再是充分的理由 —— 見 §55）
      〔2026-10-02 operator 裁定〕唯一例外：`certified` 列明文帶 `waiver:` 且宣告 QUOTABLE、
      而閘門（**不帶** `CGC_QUOTE_WAIVE_ATTRIB`）判 **DIRTY** ⇒ 放行，理由是「operator 顯式放行的
      可引用」而非「閘門說這一場乾淨」。只放行 DIRTY（R4 窗口那一條）；UNSTABLE／REFUSE 一律照擋。
      揭露必須落在該列的 `item`／`value`／`waiver`／`why`（看板渲染前三者）。C13 是第一件。
  D8 主節點（L20／L25）的 goal／evidence 必須一致：綠燈只能建在可引用的讀數上
  D9 產物結構（表格標籤配對）
  D10 臂身分（quote_gate 的 R5／R6 在**看板面**）：`options.arms` 不准留著「量不到交付目標」的臂
      （單次提交臂跳過 per-layer hook ⇒ 計數器回傳常數；或源碼明文 never-quote 的量具）。
      要嘛刪掉，要嘛搬進 `options.arms_removed` 附規則與出處（那是留下來的否證，不是待跑的路）。
  D15 待升級（operator 2026-09-30：「<b>搬成功要放到已認證</b>」）：`pending_promotion` 的每一格
      必須**跑前**就寫死驗收（`accept`）與機器判準（`judge`），而且候選臂必須在認可口徑上
      （quote_gate 的 R7）——否則這一格永遠升不了級、是白等的 pending。判準當場成立卻還停在
      pending ⇒ **紅**（該升進 `certified` 了），失敗方向與 runnable_gate 的「該結案了」一致。
      [2026-10-01 operator：「D15 這要解決」] 紅訊息指向**機器路徑** `--promote`：判定成立 ⇒
      機械合成 `certified` 列（值＝合格場次中位數＋區間、quote＝中位那一場、棘輪＝分布豁免）
      並移除該 pending 條目；寫完重建＋`--check`，沒過 ⇒ **整檔回滾**。`board_pipeline.sh`
      的重建模式會先跑它 ⇒ 下一次重建就自己升，不再需要人手抄。
  D17 棘輪 vs 認證表（2026-10-01）：`certify_anchor` 必須等於 `certified` 表上**最大的 decode 讀數**
      （那也必須是最高那一列的 `quote.artifact`），而且每一列入表時都**嚴格高於當時的上限**。
      有 decode 讀數的認證列一定要宣告 `ratchet:`（參與）或 `ratchet_exempt: <why>`（基準，不參與）
      —— 兩個都沒有 ⇒ 紅（不寫欄位不能是繞過棘輪的方式）。判準見 `ratchet_consistency()`。
  D16 臂類別 vs 卡點型別（2026-09-30）：`options.arms` 是空的（每一支臂都被 R5／R6 刪掉）、又還沒了結
      的格，**不准**把卡點寫成 `window`。窗口是抽籤（同一支臂抽到 `attribution=none` 就引用得到？
      不 —— 見 L20-1），臂的類別是**樹上的事實**：一個 window 永遠修不好臂的身分。這條同時要求
      `block` 存在、型別落在 decision／code／mechanism，且內文**指名**擋住它的是哪一條規則
      （R5／R6／R7）⇒ 讀的人一眼看到解除條件，而不是去等一個永遠不會到的窗口。
"""

import argparse
import glob as _glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_YAML = os.path.join(HERE, "decode_board_2026-09-29.yaml")
MINDMAP = os.path.join(ROOT, "docs", "mindmap", "mindmap.json")
PAGES_DIR = os.path.join(ROOT, "docs", "mindmap", "subgoals")

LIVE_TIERS = {"1", "2", "3a", "3b"}

# 徽章色 → CSS 類別。`danger` 是**別名**：2026-09-29 另一個執行緒在 L20-3 寫了
# `tone: danger`（「前置否證」），而合法的既有紅色類別是 `b-dead`。把別名加進來
# 比改別人的資料列安全，也比讓整個 build 掛掉好（視覺與 b-dead 相同）。
TONE_CLASS = {"ok": "b-ok", "warn": "b-warn", "info": "b-info", "dead": "b-dead", "danger": "b-dead"}
BADGE_TONES = tuple(TONE_CLASS) + (None,)
CLASSES = {"handled", "done", "goal", "c-axis", "account", "no-claim", "recheck"}
NEEDS = {
    "handled": ("by",), "done": ("why",), "goal": ("why",), "c-axis": ("why",),
    "account": ("why", "impact"), "no-claim": ("why",), "recheck": ("why",),
}
# `runnable`（09-30 operator）：每格必須宣告「現在能不能跑＋前置是啥」——
# 放在現況欄最前面。缺它＝D3（fail-closed：新增格子忘了寫，看板會紅，不會默默少資訊）。
TARGET_FIELDS = ("id", "title", "nodes", "now", "action", "accept", "falsify",
                 "expect_ms", "expect_tps", "state", "evidence", "badge", "stage", "runnable")

CSS = """  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans TC',Roboto,sans-serif; line-height:1.6; color:#1a1b1c; background:#f8f9fa; }
  .container { max-width:1280px; margin:0 auto; padding:36px 20px; }
  h1 { font-size:27px; font-weight:700; margin-bottom:8px; }
  .subtitle { font-size:14.5px; color:#6b7280; margin-bottom:24px; }
  h2 { font-size:20px; font-weight:600; margin:34px 0 12px; padding-bottom:7px; border-bottom:2px solid #e5e7eb; }
  h3 { font-size:16.5px; font-weight:600; margin:20px 0 9px; }
  .card { background:#fff; border-radius:12px; padding:18px 22px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.08); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:14px; }
  .stat { text-align:center; padding:16px; }
  .stat-value { font-size:29px; font-weight:700; color:#2563eb; }
  .stat-value.ok { color:#059669; } .stat-value.bad { color:#dc2626; } .stat-value.warn { color:#b45309; }
  .stat-label { font-size:12.5px; color:#6b7280; margin-top:4px; }
  table { width:100%; border-collapse:collapse; margin:12px 0; }
  th,td { padding:8px 10px; text-align:left; border-bottom:1px solid #e5e7eb; vertical-align:top; }
  th { background:#f3f4f6; font-weight:600; font-size:12.5px; white-space:nowrap; }
  td { font-size:13px; }
  code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-family:'SF Mono',Monaco,monospace; font-size:12px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; white-space:nowrap; }
  table.queue td.qnext { text-align: left; max-width: 46em; }
table.queue td { vertical-align: top; }
.layer-met { border-left:6px solid #059669; background:#ecfdf5; padding:6px 12px; border-radius:6px; }
  .b-ok { background:#d1fae5; color:#065f46; } .b-warn { background:#fef3c7; color:#92400e; }
  .b-info { background:#dbeafe; color:#1e40af; } .b-dead { background:#f3f4f6; color:#6b7280; }
  /* 09-30 operator：已了結的格（結案／判死）整列綠底；進行中不變色 */
  tr.row-done td { background:#ecfdf5; }
  tr.row-done td:first-child { border-left:4px solid #059669; }
  /* 每格開頭的「現在能不能跑＋前置」宣告 pill（資料來源＝target.runnable） */
  .runnable { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11.5px; font-weight:600;
              background:#eef2ff; color:#312e81; }
  ul,ol { margin:8px 0 12px 22px; } li { margin-bottom:5px; font-size:13.5px; }
  .warning { background:#fffbeb; border-left:4px solid #f59e0b; padding:13px 15px; border-radius:0 8px 8px 0; margin:12px 0; font-size:13.5px; }
  .success { background:#ecfdf5; border-left:4px solid #10b981; padding:13px 15px; border-radius:0 8px 8px 0; margin:12px 0; font-size:13.5px; }
  .info { background:#eff6ff; border-left:4px solid #3b82f6; padding:13px 15px; border-radius:0 8px 8px 0; margin:12px 0; font-size:13.5px; }
  .danger { background:#fef2f2; border-left:4px solid #dc2626; padding:13px 15px; border-radius:0 8px 8px 0; margin:12px 0; font-size:13.5px; }
  .steps { display:inline-flex; gap:3px; vertical-align:middle; }
  .steps i { width:13px; height:7px; border-radius:2px; background:#e5e7eb; display:inline-block; }
  .steps i.on { background:#2563eb; }
  .legend { font-size:12.5px; color:#6b7280; }
  nav { background:#fff; border-bottom:1px solid #e5e7eb; padding:10px 20px; font-size:13.5px; margin-bottom:0; }
  nav a { color:#2563eb; text-decoration:none; margin-right:16px; }
  footer { margin-top:40px; padding-top:14px; border-top:1px solid #e5e7eb; font-size:12.5px; color:#6b7280; }"""

PAGE_CSS = CSS + """
  .kv { width:100%; } .kv th { width:150px; background:#fff; font-size:13px; }
  .verdict { font-size:15px; font-weight:600; }"""


# ────────────────────────── 讀取 ──────────────────────────
def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_mindmap(path=None):
    with open(path or MINDMAP, encoding="utf-8") as fh:
        return json.load(fh)


def index_nodes(mm):
    return {e.get("id"): e for e in mm.get("entries", [])}


def live_nodes(nodes):
    return sorted(i for i, e in nodes.items() if str(e.get("tier")) in LIVE_TIERS)


def esc(t):
    return str(t)


def rel_from(here, dest):
    """產物搬到哪裡都成立：連結一律相對『本頁所在目錄』計算。"""
    return os.path.relpath(dest, here)


def nav(here, board_out, n_targets=15):
    return ('<nav><a href="%s">◀ 攻關總圖</a><a href="%s">子目標看板</a>'
            '<a href="%s">逐子目標（%d）</a></nav>'
            % (rel_from(here, os.path.join(ROOT, "docs/mindmap/index.html")),
               rel_from(here, board_out),
               rel_from(here, os.path.join(PAGES_DIR, "index.html")), n_targets))


# ────────────────────────── 驗證 ──────────────────────────
def layer_goal(layer, root=ROOT):
    """主節點（L20／L25）是否達標：**只有可引用的讀數能讓它變綠**（§55 的閘門精神）。

    回 (met, why)。met=False ⇒ 不變色、不加任何記號（「若無則沒有變化」）；
    met=None ⇒ 這一層沒有宣告 goal／evidence（不變色）。`why` 一律寫進 <h2 title="…">，
    所以「為什麼沒變綠」在 HTML 裡查得到，不必靠推測。
    """
    g = layer.get("goal") or {}
    ev = layer.get("evidence") or {}
    if not g and not ev:
        return None, ""
    if not ev:
        return None, "本層未宣告 evidence（達標必須指名一場可引用的產物）"
    target, val = g.get("target"), ev.get("value")
    metric = g.get("metric", "tps")
    crit = ("（%s）" % g["crit"]) if g.get("crit") else ""
    verdict = ev.get("quote_verdict")
    head = "目標 %s ≥ %s%s" % (metric, target, crit)
    if target is None or val is None:
        return None, head + "；本層的目標或實測值缺一 ⇒ 不判定"
    met = (float(val) >= float(target)) and verdict == "QUOTABLE"
    body = "；目前最好的可引用讀數 %s（引用閘門 %s）" % (val, verdict or "未判")
    note = ("；" + str(ev["note"])) if ev.get("note") else ""
    return met, head + body + (" ⇒ 達標" if met else " ⇒ 未達標（不變色）") + note


def gate_verdict(root, artifact):
    """跑引用閘門判該產物：回 (verdict, why)。

    verdict=None ⇒ 判不了（產物不存在／沒有可判的 row／載不進模組）。
    verdict="MIXED(...)" ⇒ 檔內 row 判詞不一致（預期／解碼兩軸分開判，一致才算一件事）。
    """
    path = artifact if os.path.isabs(artifact) else os.path.join(root, artifact)
    if not os.path.exists(path):
        return None, "產物不存在：%s" % artifact
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import quote_gate
    except Exception as exc:  # noqa: BLE001
        return None, "載不進 quote_gate：%s" % exc
    try:
        recs = quote_gate.scan([path])
    except Exception as exc:  # noqa: BLE001
        # 閘門自己爆掉 = 「判不了」，不是「build 跟著爆」。D7／D8 會把這句當錯誤報出來，
        # 所以失敗方向仍然是不放行（fail-closed），只是訊息比 traceback 有用。
        return None, "quote_gate 判不了 %s：%s: %s" % (artifact, type(exc).__name__, exc)
    if not recs:
        return None, "產物裡沒有可判的 llama-bench row：%s" % artifact
    verdicts = sorted(set(r["verdict"] for r in recs))
    if len(verdicts) == 1:
        return verdicts[0], ""
    return "MIXED(%s)" % "/".join(verdicts), "檔內 row 判詞不一致"


def _axis_verdict(recs, axis):
    """從 quote_gate.scan 的 recs 取某軸（"decode"／"prefill"）的 verdict。

    軸識別靠 row 的 `shape`（"p<n_prompt>/n<n_gen>/d<n_depth>"）：
    - decode 軸：n_prompt=0 且 n_gen>0 ⇒ shape 以 "p0/" 開頭
    - prefill 軸：n_gen=0 ⇒ shape 含 "/n0/"
    回 None 表示沒找到該軸 row。
    """
    for r in recs or []:
        shp = r.get("shape", "")
        if axis == "decode" and shp.startswith("p0/"):
            return r.get("verdict")
        if axis == "prefill" and "/n0/" in shp:
            return r.get("verdict")
    return None


def _qg():
    """懶載 quote_gate —— 判準（含 R7 的認可口徑白名單）只有一份定義。"""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import quote_gate
    return quote_gate


PENDING_JUDGE_KINDS = ("quotable_launch_cluster",)


def cluster_fields(root, judge):
    """`quotable_launch_cluster` 的**結構化**結果：回 (n_cell, fields, err)。

    fields ＝ 每一場一個 dict：{file, verdict, value, cold, split, spread_all, spread_kept, reasons}。
    `value` ＝ 引用值（`metrics.quoted_ts`；pp-less 拆欄後＝steady 平均，沒有才退回 `avg`）。

    為什麼要結構化：D15 的判詞（`pending_cluster`）與機器升級（`cmd_promote`）**必須走同一支**
    ——不然「升級」就是另一套規則，而這裡的每一條規則都是跑前寫死的。
    """
    pat = judge.get("glob")
    if not pat:
        return None, [], "judge 缺 glob"
    cell = judge.get("cell") or "delivery"
    files = sorted(f for f in _glob.glob(os.path.join(root, pat), recursive=True)
                   if os.path.isfile(f))
    try:
        qg = _qg()
    except Exception as exc:  # noqa: BLE001
        return None, [], "載不進 quote_gate：%s" % exc
    fields = []
    for f in files:
        try:
            recs = qg.scan([f])
        except Exception as exc:  # noqa: BLE001
            return None, [], "quote_gate 判不了 %s：%s" % (os.path.basename(f), exc)
        dec = [r for r in recs
               if (r.get("cell") or "") == cell and (r.get("shape") or "").startswith("p0/")]
        if not dec:
            continue
        best = max(dec, key=lambda r: ((r["metrics"] or {}).get("quoted_ts")
                                       if (r["metrics"] or {}).get("quoted_ts") is not None
                                       else (r["metrics"] or {}).get("avg")) or 0)
        m = best["metrics"] or {}
        val = m.get("quoted_ts") if m.get("quoted_ts") is not None else m.get("avg")
        fields.append({"file": os.path.relpath(f, root) if os.path.isabs(f) else f,
                       "verdict": best.get("verdict"), "value": val,
                       "cold": m.get("cold_ts"),
                       "split": bool(m.get("regime_split")) and m.get("cold_ts") is not None,
                       "spread_all": m.get("all_spread"), "spread_kept": m.get("kept_spread"),
                       "samples": m.get("samples"), "reps": m.get("reps"),
                       "reasons": best.get("reasons") or []})
    return len(fields), fields, ""


def pending_cluster(root, judge):
    """判一個「待升級」候選的機器判準：回 (n_launches, n_quotable, detail)。

    `quotable_launch_cluster` 問的是「這一批**獨立啟動**裡，有幾場真的過得了引用閘門」。
    為什麼不能升級一個單點：§73.5 —— 同一支臂、同一格連三場是 8.933／8.486／10.923，而前兩場
    各有一顆 rep 塌到 3.8–3.9 ⇒ 10.923 是那條分布的右尾之一，不是中心。

    2026-10-01（§3.3b 拆欄）：交付 cell（pp-less）的第 1 個 rep 天生冷 ⇒ 引用值是 **steady**
    那一欄（`metrics.quoted_ts`；cold 值只當診斷並在點名時一起印）。
    """
    n_cell, fields, err = cluster_fields(root, judge)
    if n_cell is None:
        return None, None, err
    cell = judge.get("cell") or "delivery"
    hits = []
    for f in fields:
        if f["verdict"] != "QUOTABLE":
            continue
        note = ("（steady；cold %.2f 未計入）" % f["cold"]) if (f["split"] and f["cold"] is not None) else ""
        hits.append("%s %.3f%s" % (os.path.basename(f["file"]), f["value"] or 0, note))
    if hits:
        return n_cell, len(hits), "；".join(hits)
    return n_cell, 0, "目前 0 場可引用（掃到 %d 場帶 %s decode 列）" % (n_cell, cell)


# ── D15 的機器路徑：--promote（2026-10-01；operator：「D15 這要解決」）────────────────────
# 規矩不變（條件**跑前**寫死）；變的只是「條件成立之後不必等人」：
#   * `--promote [ID]`：對每一個 judge 當場成立的 pending 項，機械合成一列 `certified`
#     （值＝合格場次的**中位數**＋區間；`quote`＝中位那一場；棘輪＝`ratchet_exempt`（分布是基準類）），
#     插進 `certified:` 的**數字序**位置，並把那一條從 `pending_promotion` 移除。
#   * fail-closed 三段（與 caliber_certify --update-board 同一套紀律）：
#     ① 推之前看板已經紅 ⇒ 不動 —— **例外**：紅的**全部**是「我們正要升的那些 D15」時繼續
#     （與 caliber 不同：D15 的紅本身就是「條件成立卻沒升」的觸發器，這正是要解的那件事）；
#     ② 寫完重建＋`--check`，沒過 ⇒ YAML **逐字回滾**（產物也重建回去）；
#     ③ judge 沒成立 ⇒ 不動（印出門檻差距）。
#   * 它**不搬任何門檻**：合格與否仍由 quote_gate（R1–R8）與 pending 條目裡跑前寫死的
#     `accept`／`judge` 判 —— 合成的是**必要欄位**，散文（敘述、反例、卡點）要再人工補。
#   * D15 的紅訊息會直接叫你把這條指令跑起來（見 `validate()`）；`board_pipeline.sh`（重建模式）
#     也會先跑它 ⇒ 下一次重建就自己升。
# 用途上的唯一「新規則」：`--promote` 不改 `certify_anchor`（棘輪上限與分布無關）。
def _indent(text) -> str:
    return "\n".join("    " + l for l in str(text or "").split("\n"))


def _yq(s) -> str:
    """寫進看板的字串：轉義反斜線與雙引號；換行折成 `<br>`（看板的散文本來就用它分段）。"""
    return '"%s"' % str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "<br>")


def run_gate(yaml_path=None, build=False):
    """當場重跑看板閘門（D1–D17，含 D7 引用閘門）：回 (rc, 尾段輸出)。

    build=False ⇒ 只 `--check`（**不寫任何檔**；升級前的預檢用）。
    build=True  ⇒ 先重建產物（build 驗完才寫檔），再 `--check` 逐字對 —— 寫完 YAML 用：
                   改了 YAML 就一定會讓 HTML「落後」（D4），不重建就不算驗完。

    驗證器的**唯一實作**＝本檔（不另寫第二套判準）：這是子行程自己呼叫自己。
    """
    base = [sys.executable, os.path.join(HERE, "decode_board_build.py")]
    extra = ["--yaml", yaml_path] \
        if (yaml_path and os.path.abspath(yaml_path) != os.path.abspath(DEFAULT_YAML)) else []

    def _run(cmd):
        try:
            cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        except Exception as exc:  # noqa: BLE001
            return 99, "閘門跑不起來：%s: %s" % (type(exc).__name__, exc)
        out = (cp.stdout or "").strip() or (cp.stderr or "").strip()
        return cp.returncode, "\n".join(out.split("\n")[-8:])

    if build:
        rc, tail = _run(base + extra)
        if rc == 0:
            rc, tail = _run(base + ["--check"] + extra)
        return rc, tail
    return _run(base + ["--check"] + extra)


def promotion_row(pend, fields):
    """pending 條目 → `certified:` 的一列（list[str]）＋摘要；門檻沒到 ⇒ (None, 原因)。

    值＝合格場次的**中位數**（偶數取小的那一半）＋區間 min–max；`quote` 指中位那一場
    （同值多場 ⇒ 檔名序第一場）—— 這是 pending 條目 `accept` 寫的「值是那 ≥2 場的中位數並附區間」。
    """
    j = pend.get("judge") or {}
    try:
        mn_l = int(j.get("min_launches") or 1)
        mn_q = int(j.get("min_quotable") or 1)
    except (TypeError, ValueError):
        return None, "judge 的門檻不是整數（min_launches=%r、min_quotable=%r）" \
            % (j.get("min_launches"), j.get("min_quotable"))
    q = [f for f in fields if f["verdict"] == "QUOTABLE" and f["value"] is not None]
    if len(fields) < mn_l or len(q) < mn_q:
        return None, "門檻未達（可引用 %d／需 %d；場數 %d／需 %d）" % (len(q), mn_q, len(fields), mn_l)
    vals = sorted(float(f["value"]) for f in q)
    med, lo, hi = vals[(len(vals) - 1) // 2], vals[0], vals[-1]
    mid = min((f for f in q if float(f["value"]) == med), key=lambda f: f["file"])
    prof = str((pend.get("arms") or ["prod-new"])[0]).split(":", 1)[0].strip() or "prod-new"
    per = "／".join("%s <b>%.3f</b>" % (os.path.basename(f["file"]), f["value"])
                   for f in sorted(q, key=lambda f: f["file"]))
    miss = sorted(f for f in fields if f["verdict"] != "QUOTABLE")
    tail_note = ("（%s 未過：%s；判詞與理由見 quote）"
                 % ("、".join(os.path.basename(f["file"]) for f in miss),
                    "／".join(str(f["verdict"]) for f in miss))) if miss else ""
    val_line = ("中位 <b>%.3f</b> t/s（區間 <b>%.3f–%.3f</b>）＝ %d 場獨立啟動裡 <b>%d 場</b>通過引用閘門；"
                "逐場引用值（steady）：%s%s" % (med, lo, hi, len(fields), len(q), per, tail_note))
    seg = []
    for f in sorted(fields, key=lambda f: f["file"]):
        s = "%s <b>%.3f</b>（cold %s" % (os.path.basename(f["file"]), f["value"],
                                          ("%.3f" % f["cold"]) if f.get("cold") is not None else "—")
        if f.get("spread_all") is not None and f.get("spread_kept") is not None:
            s += "、全 rep %.3f→steady %.3f" % (f["spread_all"], f["spread_kept"])
        s += "、%s" % f["verdict"]
        if f["verdict"] != "QUOTABLE" and f["reasons"]:
            s += "〔%s〕" % "；".join(str(x) for x in f["reasons"])
        seg.append(s + "）")
    why = ("分布的中位那一場（引用值 <b>%.3f</b>；cold %.3f 留診斷）。逐場（steady／cold／全 rep→steady／判詞）：%s"
           "<br>⚠ 這一列由 <code>decode_board_build.py --promote</code> 機械合成：值＝合格場次的<b>中位數</b>＋區間、"
           "quote＝中位那一場；分母（哪些場算）＝pending 條目裡跑前寫死的 <code>judge</code>，R1–R8 由 "
           "<code>quote_gate</code> 當場複判（D7）。"
           % (med, mid["cold"] if mid.get("cold") is not None else 0.0, "；".join(seg)))
    src = "%s（%d 場；judge.glob=%s）" % ("、".join(f["file"] for f in sorted(fields, key=lambda f: f["file"])),
                                          len(fields), j.get("glob"))
    if pend.get("source"):
        src += "；原 pending 條目的出處：%s" % pend["source"]
    row = ["  - id: %s" % pend["id"],
           "    item: %s" % _yq(pend.get("item") or ""),
           "    value: %s" % _yq(val_line),
           "    profile: %s" % prof,
           "    entry: \"harness bench\"",
           "    meets: true",
           "    ratchet_exempt: %s" % _yq(
               "基準（非棘輪）：這一列登記的是一條<b>分布</b>（值＝合格場次的中位數並附區間），"
               "不是單場讀數 ⇒ 不參與棘輪（上限＝<code>certify_anchor.decode_ts</code>）。"
               "由 <code>decode_board_build.py --promote</code> 機械升級：入表條件跑前寫死在 pending 條目"
               "（<code>judge</code>＋<code>accept</code>）。"),
           "    source: %s" % _yq(src),
           "    quote: {artifact: %s, verdict: %s, why: %s}" % (_yq(mid["file"]), mid["verdict"], _yq(why))]
    return row, "中位 %.3f（%d／%d 場可引用；quote=%s）" % (med, len(q), len(fields), mid["file"])


def cert_text_insert(text, row_id, snippet):
    """把一列插進 `certified:` 清單的**數字序**位置（不是尾端）：回新全文。

    看板是手寫的（註解就是文件）⇒ 只能用行級手術，不能用 yaml round-trip。同 id 已在表上 ⇒
    ValueError（不重複插入；那是 YAML 漂移，要人判）。
    """
    m0 = re.match(r"C(\d+)$", str(row_id))
    if not m0:
        raise ValueError("id 不是 C<數字> 形式：%r" % row_id)
    want = int(m0.group(1))
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("certified:")), None)
    if i is None:
        raise ValueError("看板沒有 certified: 這一節")
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or lines[j][:1] in (" ", "\t")):
        j += 1
    anchors = []
    for k in range(i + 1, j):
        m = re.match(r"^  - id: C(\d+)\s*$", lines[k])
        if m:
            anchors.append((k, int(m.group(1))))
    for _k, num in anchors:
        if num == want:
            raise ValueError("%s 已經在 certified 表上（不重複插入）" % row_id)
    ins = next((k for k, num in anchors if num > want), None)
    if ins is not None:
        return "\n".join(lines[:ins] + list(snippet) + [""] + lines[ins:])
    k = i
    for x in range(i + 1, j):
        if lines[x].strip():
            k = x
    tail = lines[j:]
    while tail and not tail[0].strip():
        tail.pop(0)
    return "\n".join(lines[:k + 1] + [""] + list(snippet) + [""] + tail)


def pending_text_drop(text, pid):
    """把 `pending_promotion` 的那一條移除（最後一條 ⇒ 整節收成 `pending_promotion: []`）：回新全文。"""
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("pending_promotion:")), None)
    if i is None:
        raise ValueError("看板沒有 pending_promotion: 這一節")
    if lines[i].split(":", 1)[1].strip() not in ("", "[]"):
        raise ValueError("pending_promotion: 不是清單（%r）⇒ 不敢動" % lines[i])
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or lines[j][:1] in (" ", "\t")):
        j += 1
    starts = [k for k in range(i + 1, j) if re.match(r"^  - id: \S", lines[k])]
    if not starts:
        if pid in text:
            raise ValueError("pending_promotion 裡找不到 %s 的條目（節是空的？）" % pid)
        return text
    spans = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else j
        while e > s and not lines[e - 1].strip():
            e -= 1
        spans.append((s, e))
    tgt = next((sp for sp in spans if re.match(r"^  - id: %s\s*$" % re.escape(str(pid)), lines[sp[0]])), None)
    if tgt is None:
        raise ValueError("pending_promotion 裡沒有 %s" % pid)
    block = []
    for s, e in spans:
        if (s, e) == tgt:
            continue
        if block:
            block.append("")
        block += lines[s:e]
    new_block = ["pending_promotion:"] + block if block else ["pending_promotion: []"]
    return "\n".join(lines[:i] + new_block + lines[j:])


def cmd_promote(yaml_path=DEFAULT_YAML, only_id=None, gate=None, root=ROOT):
    """D15 的機器路徑：判準成立 ⇒ 機械升進 `certified`（does not touch `certify_anchor`）。

    回 rc：0 ＝ 照做了或不用做；1 ＝ 寫了但閘門沒過 ⇒ **已回滾**；2 ＝ 用法錯（指名的不存在／
    還沒到門檻）。看板紅在 D15 待升級**以外**的地方 ⇒ 不碰（rc=0）；只紅在「要升的那幾條」⇒
    照升（那正是 D15 的觸發條件）。
    """
    gate = gate or run_gate
    if not os.path.exists(yaml_path):
        print("--promote：看板檔不存在（%s）" % yaml_path)
        return 2
    try:
        board = load_yaml(yaml_path)
    except Exception as exc:  # noqa: BLE001
        print("--promote：YAML 載不進（%s: %s）⇒ 不碰" % (type(exc).__name__, exc))
        return 2
    pend = [p for p in (board.get("pending_promotion") or []) if isinstance(p, dict)]
    if only_id is not None:
        pend = [p for p in pend if str(p.get("id")) == str(only_id)]
        if not pend:
            print("--promote：pending_promotion 裡沒有 %s ⇒ 不碰" % only_id)
            return 2
    else:
        only_id = None
    if not pend:
        print("--promote：pending_promotion 是空的 ⇒ 沒有事要做")
        return 0
    cert_ids = set(c.get("id") for c in (board.get("certified") or []))
    rc, tail = gate(yaml_path, False)
    if rc != 0:
        # D15 的紅＝「條件成立卻還躺在 pending」——那正是這一支要解的，不是阻擋條件。
        # 所以只看「紅是不是**全部**落在我們正要升的 id 上」；有一條別的紅 ⇒ 不碰。
        ours = set(str(p.get("id")) for p in pend)
        errs_all, other = [], []
        try:
            errs_all, _ = validate(board, index_nodes(load_mindmap()), root=root)
        except Exception as exc:  # noqa: BLE001
            print("--promote：看板是紅的，而且判不了紅在哪（%s: %s）⇒ 不碰" % (type(exc).__name__, exc))
            return 0
        for e in errs_all:
            if str(e).startswith("D4 "):
                continue      # 「產物落後於 YAML」⇒ 寫完那一列的重建就會修好（post-check 會驗）
            m = re.match(r"D15 (?:pending )?(\S+) ", str(e))
            if not (m and m.group(1) in ours):
                other.append(e)
        if other:
            print("--promote：⛔ 看板**現在就是紅的**（而且是 D15 待升級以外的紅）⇒ 不推"
                  "（先修綠再來；隨後的 build／--check 會說紅在哪）：\n%s\n%s"
                  % (_indent("\n".join("✗ " + str(e) for e in other[:6])), _indent(tail)))
            return 0
        print("--promote：看板現在是紅的，但紅的只有 %s 的 D15（條件已達成卻還沒升）"
              "（允許附帶 D4：寫完重建就修好）⇒ 正是要升的那一條 ⇒ 繼續" % "、".join(sorted(ours)))
    with open(yaml_path, encoding="utf-8") as fh:
        original = fh.read()
    text, done, missed = original, [], []
    for p in pend:
        pid = str(p.get("id"))
        if pid in cert_ids:
            print("--promote：⛔ %s 同時在 certified 與 pending_promotion ⇒ 不碰（YAML 漂移，要人判）" % pid)
            missed.append(pid)
            continue
        n_l, fields, err = cluster_fields(root, p.get("judge") or {})
        if n_l is None:
            print("--promote：%s 判不了（%s）⇒ 不碰" % (pid, err))
            missed.append(pid)
            continue
        row, why = promotion_row(p, fields)
        if row is None:
            print("--promote：%s 還沒到（%s）⇒ 不碰" % (pid, why))
            missed.append(pid)
            continue
        try:
            text2 = cert_text_insert(text, pid, row)
            text2 = pending_text_drop(text2, pid)
        except ValueError as exc:
            print("--promote：⛔ %s 寫不進去（%s）⇒ 不碰" % (pid, exc))
            missed.append(pid)
            continue
        text = text2
        done.append(pid)
        print("--promote：合成 %s：%s" % (pid, why))
    if not done:
        print("--promote：沒有可升級的條目（%s）⇒ 看板一個位元組都沒動"
              % ("、".join(missed) or "沒有條件達成的"))
        return 2 if only_id else 0
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    rc2, tail2 = gate(yaml_path, True)
    if rc2 != 0:
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(original)
        rc3, tail3 = gate(yaml_path, True)
        print("--promote：⛔ 寫完重跑閘門沒過 ⇒ **已回滾**（看板逐字回到升級前）：\n%s" % _indent(tail2))
        if rc3 != 0:
            print("  ⚠ 回滾後的重建也紅了（那代表升級之前就沒有真的綠；看它的輸出）：\n%s" % _indent(tail3))
        return 1
    print("--promote：✅ 升進 certified：%s；閘門重跑（重建＋--check）PASS：\n%s"
          % ("、".join(done), _indent(tail2)))
    print("  ⚠ 合成的是**必要欄位**（值＝中位＋區間、quote＝中位那一場、棘輪＝分布豁免）；要補敘述"
          "（逐場卡點、反例…）直接編輯那一列 —— 判準不看散文。")
    print("  ⚠ 頁面上的字面（summary／certified_note 的「最高…」）請順手同步：閘門不管字面，"
          "但同一頁要講同一個數。")
    if missed:
        print("  ⚠ 沒動的：%s" % "、".join(missed))
    return 0 if (only_id is None or only_id in done) else 2


D7_EXCLUDED_CLOSURES = ("結案（排除）",)


def arm_census(board):
    """回 (live, removed, bare)：可跑的臂數、已刪的臂數、**已無可跑臂**的子目標 id。

    這三個數是 D10 的儀表：`bare` 不為 0 時，看板上一部分格子「沒有任何一條量得到交付目標的路」，
    這件事必須在頁面上看得見，而不是埋在某一格的 options 裡。
    """
    live = removed = 0
    bare = []
    for layer in board.get("layers") or []:
        for t in layer.get("targets") or []:
            o = t.get("options") or {}
            n_live = len([a for a in (o.get("arms") or [])
                          if isinstance(a, str) or (isinstance(a, dict) and a.get("arm"))])
            n_rm = len([r for r in (o.get("arms_removed") or []) if isinstance(r, dict)])
            live += n_live
            removed += n_rm
            if n_rm and not n_live:
                bare.append(t.get("id"))
    return live, removed, bare


def precondition_census(board):
    """回 (needed, dup, blocked, drift, drift_ids)：前置閘門的**儀表**。

    這是 D12 的對稱物：D12 逐格判「宣告與現判是否一致」，這裡把 16 格的結果**彙總成一行放在頁首**
    —— 「有幾格會產生新資訊／幾格只是重印／幾格卡在人與機制」。有了它，「還能做什麼」第一次變成
    可核對的清單，而不是 16 條看起來都能跑的路。判準呼叫 scripts/check/runnable_gate.py（不重寫）。
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import runnable_gate as RG
    except Exception:  # noqa: BLE001
        return None
    class RC:                       # 判詞常數（唯一定義在 runnable_gate，這裡只取用）
        EXISTS, ABSENT = RG.EXISTS, RG.ABSENT
    needed = dup = blocked = drift = 0
    drift_ids = []
    for layer in board.get("layers") or []:
        for t in layer.get("targets") or []:
            pre = t.get("precondition") or {}
            if not pre:
                continue
            if pre.get("rerun") == "blocked":
                blocked += 1
                continue
            lv, _why, _ = RG.scan(pre, ROOT)
            ok = ((pre.get("rerun") == "needed" and lv == RC.ABSENT)
                  or (pre.get("rerun") == "duplicate" and lv == RC.EXISTS))
            if not ok:
                drift += 1
                drift_ids.append(t.get("id"))
            elif pre.get("rerun") == "needed":
                needed += 1
            else:
                dup += 1
    return needed, dup, blocked, drift, drift_ids


def blocked_queue(board):
    """回 (rows, counts)：`blocked` 那幾格的**形別化決策佇列**（判準定義在 runnable_gate.BLOCK_*）。

    rows ＝ [(kind, owner, id, next), ...]，排序＝決定→改動→機制→窗口（同型再按 id）。
    理由：**最便宜又解鎖最多的先做**。這張表回答的是「誰該動、動什麼」，而不是「哪一格比較重要」。
    欄位由 D13 強制存在（缺了 build 就紅）⇒ 這裡不防缺。
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import runnable_gate as RG
    except Exception:  # noqa: BLE001
        return [], {}
    rows, counts = [], {}
    for layer in board.get("layers") or []:
        for t in layer.get("targets") or []:
            pre = t.get("precondition") or {}
            if pre.get("rerun") != "blocked":
                continue
            blk = pre.get("block") or {}
            k = blk.get("kind") or "?"
            _lv, _lw, _ = RG.scan_block_assert(blk, ROOT)
            rows.append((k, blk.get("owner") or "?", t.get("id"), blk.get("next") or "",
                         _lv, _lw))
            counts[k] = counts.get(k, 0) + 1
    rows.sort(key=lambda r: (RG.BLOCK_ORDER.get(r[0], 99), str(r[2])))
    return rows, counts


def arm_void_flags(arm):
    """回 (flags, why)：這支臂身上有哪些「**量不到交付目標**」的東西真的開著。

    `flags = {旗標: "R5"｜"R6"}`；判不了 ⇒ `(None, 說明)`。
    判準的**唯一定義**在 scripts/check/quote_gate.py（R5 量具登記表 THROUGHPUT_VOID_INSTRUMENTS
    ＋ R6 輸出見證登記表 UNVERIFIED_OUTPUT_ARMS）。這裡只做轉接、不重寫清單 ——
    兩份清單一旦漂移，D10 就會跟同一格產物的判詞說不同的話。
    臂的字串形式與看板一致（`prod-new:FLAG=1;FLAG=1`）⇒ 走 quote_gate 的同一條解析路徑。
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import quote_gate
    except Exception as exc:  # noqa: BLE001
        return None, "載不進 quote_gate：%s: %s" % (type(exc).__name__, exc)
    try:
        prod = {"tag": str(arm)}
        flags = {k: "R5" for k in quote_gate.arm_instruments(prod)}
        flags.update({k: "R6" for k in quote_gate.unverified_output_arms(prod)})
    except Exception as exc:  # noqa: BLE001
        return None, "解析臂失敗：%s: %s" % (type(exc).__name__, exc)
    return flags, ""


# ── 計數器端點的結案路徑（2026-09-29，L20-2 是第一例）────────────────────────
# 為什麼需要第二條路：D7 的引用閘門（quote_gate）判的是**時間**讀數（逐 rep 離散、樣本數、
# 窗口、參考帶）。可是有些子目標的主端點**按設計就是計數器** —— 判準是二值翻轉，不需要乾淨窗，
# 而它們的 t/s 反而**不可引用**（單段提交臂輸出是 garbage）。硬要這種格拿 QUOTABLE，
# 唯一的出路是去捏一個時間讀數，或讓閘門永久紅著。
# 所以補一條**對稱**的路：宣告 `counter_quote`（見證產物 ＋ 判它的檢查器），build 當場重跑。
# 一樣 fail-closed：產物不見了、或判詞不再是宣告的那個，board 就紅。
COUNTER_METRICS = ("zero_slot_flip", "m_ratio", "fill_term_ms", "margin_ms", "feed_flip",
                   "rho_price")
# 每一種 counter metric 允許的**結案判詞**（D7b）：翻轉類＝FLIP；門檻／比值類＝WITHIN；
# 定價類＝PRICED（2026-09-30 §69：交付 cell 上一個 term 的價格，端點是**區間**）。
COUNTER_CLOSE_VERDICTS = {"zero_slot_flip": "FLIP", "m_ratio": "WITHIN",
                          "fill_term_ms": "PRICED",
                          # 餵料端點（2026-09-30 L20-5）：也是翻轉型（對照臂 prefetch=0/0、
                          # 實驗臂除了 >0 之外還有 CGC-RB-FEED）——見 scripts/check/p1_feed_ab.py。
                          "feed_flip": "FLIP",
                          # 同格邊際（2026-09-30 §70）：要以「划算」結案，邊際必須**分離為正**
                          # ——`NOT_SEPARATED` 不能當達標（那正是 L20-3 的判詞）。
                          "margin_ms": "POSITIVE",
                          # ρ 的權威 row 價格（2026-10-01，L20-7）：要以「價格成立」結案才需要
                          # `PRICE`；`NO_EFFECT`（量不出價格）走**排除**路徑（否證式），不在此表。
                          "rho_price": "PRICE"}


def counter_verdict(root, cq):
    """重跑計數器／儀器端點的檢查器 → (verdict, why)；判不了回 (None, why)。"""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    tool = cq.get("tool") or ""
    # ── 儀器端點：門檻／比值（2026-09-30 §63，L25-5 是第一例）─────────────────
    # 為什麼需要第二**種**：`m`（每輪 draft 成本 ÷ 一個 plain step）不是時間讀數
    # ⇒ quote_gate 判不了它；但它也不是二值翻轉，而是一條**門檻**（m ≤ 0.30）。
    # 判它的東西已經存在（mtp_round_split.py 的五條閘）⇒ 這裡只做兩件事：當場重跑
    # 那五條閘 ＋ 檢查各自宣告的門檻。fail-closed：產物不在、閘不過、窗口不是
    # none（若宣告 require_attrib）、cell 不對、m 算不出來 ⇒ 一律不放行。
    if tool == "mtp_round_split":
        art = cq.get("artifact")
        if not art:
            return None, "counter_quote 缺 artifact（儀器端點必須指名產物）"
        ap_ = os.path.join(root, art)
        if not os.path.exists(ap_):
            return None, "見證產物不存在：%s" % art
        if cq.get("limit") is None:
            return None, "counter_quote 缺 limit（門檻類端點必須寫死上限）"
        lp = cq.get("log")
        if lp and not os.path.exists(os.path.join(root, lp)):
            return None, "見證 log 不存在：%s" % lp
        try:
            import mtp_round_split as M
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 mtp_round_split：%s" % exc
        try:
            res = M.judge(ap_, os.path.join(root, lp) if lp else None)
        except Exception as exc:  # noqa: BLE001
            return None, "mtp_round_split 判不了：%s: %s" % (type(exc).__name__, exc)
        if res.get("verdict") != "OK" or res.get("m") is None:
            return "REFUSE", "五閘不過 ⇒ %s" % (res.get("why") or "m 算不出來")
        if cq.get("require_attrib") and res.get("attrib") != cq["require_attrib"]:
            return "REFUSE", "窗口 attribution=%s ≠ 要求的 %s" % (res.get("attrib"), cq["require_attrib"])
        if cq.get("cell") and res.get("cell") != cq["cell"]:
            return "REFUSE", "cell=%s ≠ 宣告的 %s" % (res.get("cell"), cq["cell"])
        if float(res["m"]) > float(cq["limit"]):
            return "REFUSE", "m=%.3f > 門檻 %.3f" % (res["m"], float(cq["limit"]))
        return "WITHIN", ""
    # ── 定價端點：交付 cell 上一個 term 的價格（2026-09-30 §69，L20-4 是第一例）──────
    # 為什麼需要第三種：`ms/step` 的**價格**既不是時間讀數（正確性未證的臂不給引用）
    # 也不是二值翻轉。判它的東西已經存在（fill_term_ab.py 的七條結構閘）⇒ 這裡只做
    # 兩件事：當場重跑那七條閘 ＋ 檢查價格區間與宣告一致（±tol_pct）。fail-closed：
    # 產物／log 不在、cell 不是交付 cell、窗口髒、步數不配、B 沒關掉讀取 ⇒ 一律不放行。
    if tool == "fill_term_ab":
        art, lga, lgb = cq.get("artifact"), cq.get("log_a"), cq.get("log_b")
        if not (art and lga and lgb):
            return None, "counter_quote 缺 artifact／log_a／log_b（定價端點必須指名三個見證檔）"
        for p_ in (art, lga, lgb):
            if not os.path.exists(os.path.join(root, p_)):
                return None, "見證檔不存在：%s" % p_
        band = cq.get("band")
        if not (isinstance(band, list) and len(band) == 2):
            return None, "counter_quote 缺 band（價格是區間 [lo, hi]，不是單點）"
        try:
            import fill_term_ab as F
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 fill_term_ab：%s" % exc
        try:
            res = F.judge(os.path.join(root, art), os.path.join(root, lga), os.path.join(root, lgb))
        except Exception as exc:  # noqa: BLE001
            return None, "fill_term_ab 判不了：%s: %s" % (type(exc).__name__, exc)
        if res.get("verdict") != "PRICED":
            return "REFUSE", "結構閘不過 ⇒ %s" % (res.get("why") or "未定價")
        tol = float(cq.get("tol_pct", 10)) / 100.0
        for i, key in enumerate(("lo", "hi")):
            live = float(res["band"][i])
            if abs(live - float(band[i])) > tol * max(abs(float(band[i])), 1e-9):
                return "REFUSE", "價格區間[%d] 現算 %.2f ≠ 宣告 %.2f（差 > %.0f%%）" % (
                    i, live, float(band[i]), tol * 100)
        return "PRICED", ""
    # ── 同格邊際端點：B 半（靜態寬度重算）在**同一格**上的邊際（2026-09-30 §70）──────
    # 為什麼需要第四種：上面兩種都是**單側**定價（價格／門檻）。L20-3 的錯誤是**跨格**
    # 比較（交付 cell 的重算 vs (default) cell 的 fill）⇒ 這一端點要求兩側都指名、且由
    # 檢查器當場現算「逐 rep x 逐趟」的符號一致性，判詞三態（POSITIVE／EXCLUDED／
    # NOT_SEPARATED）。fail-closed：產物不在、cell 不是交付、窗口髒、探針窗口不乾淨／
    # 幾何不符、steady 步數不足 ⇒ 一律不放行。
    if tool == "samecell_margin":
        ab = cq.get("artifact")
        lga = cq.get("log_a")
        probes = cq.get("probes") or []
        hist = cq.get("hist")
        if not (ab and lga and probes and hist):
            return None, "counter_quote 缺 artifact／log_a／probes／hist（同格邊際要四個見證）"
        for p_ in [ab, lga, hist] + list(probes):
            if not os.path.exists(os.path.join(root, p_)):
                return None, "見證檔不存在：%s" % p_
        try:
            import samecell_margin as S
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 samecell_margin：%s" % exc
        try:
            res = S.judge(os.path.join(root, ab), os.path.join(root, lga),
                          [os.path.join(root, p_) for p_ in probes],
                          os.path.join(root, hist))
        except Exception as exc:  # noqa: BLE001
            return None, "samecell_margin 判不了：%s: %s" % (type(exc).__name__, exc)
        if res.get("verdict") not in ("POSITIVE", "EXCLUDED", "NOT_SEPARATED"):
            return "REFUSE", "同格邊際判不了 ⇒ %s" % (res.get("why") or "未判")
        want_k = cq.get("k_star")
        if want_k is not None and int(want_k) != int(res["k_star"]):
            return "REFUSE", "k* 現算 %s ≠ 宣告 %s（覆蓋率表變了）" % (res["k_star"], want_k)
        band = cq.get("margin_ms")
        if isinstance(band, list) and len(band) == 2:
            tol = float(cq.get("tol_ms", 0.5))
            for i, key in enumerate(("lo", "hi")):
                live = float(res["k_star_margin_ms"][i])
                if abs(live - float(band[i])) > tol:
                    return "REFUSE", "k* 邊際[%d] 現算 %.2f ≠ 宣告 %.2f（容差 %.2f ms）" % (
                        i, live, float(band[i]), tol)
        return res["verdict"], ""
    # ── 翻轉端點（第二種）：餵料在**沒有 debug 插樁**的臂上活著（2026-09-30，L20-5）──────
    # 與 zero_slot_flip 同一**型**（計數從 0 翻成 >0），但見證形狀不同：那一種是兩份 artifact
    # （A/B 各一），這一種是**一份成對產物 ＋ 產物自己記下的兩臂 live_log**—— 因為「臂上沒有
    # R5 列印」這件事只能從 log 判，而 live_log 記在產物裡就不会漏配。
    # fail-closed：產物不在、兩臂分不出來、跨 build、窗口不是 none、對照臂不再 0/0、
    # 實驗臂餵料消失、或 log 出現 MISSMASK 列印 ⇒ 一律不放行。
    if tool == "p1_feed_ab":
        art = cq.get("artifact")
        if not art:
            return None, "counter_quote 缺 artifact（翻轉端點必須指名產物）"
        ap_ = os.path.join(root, art)
        if not os.path.exists(ap_):
            return None, "見證產物不存在：%s" % art
        try:
            import p1_feed_ab as F
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 p1_feed_ab：%s" % exc
        logs = [os.path.join(root, x) for x in (cq.get("logs") or [])] or None
        try:
            res = F.judge(ap_, logs, root=root)
        except Exception as exc:  # noqa: BLE001
            return None, "p1_feed_ab 判不了：%s: %s" % (type(exc).__name__, exc)
        if res.get("verdict") == "REFUSE":
            return "REFUSE", res.get("why") or "端點未成立"
        if res.get("verdict") != "FLIP":
            return None, "判詞不認得：%r" % res.get("verdict")
        return "FLIP", ""
    # ── 見證端點：R6 的**輸出見證**（2026-09-30，L20-1 是第一例）────────────────────
    # 為什麼需要這一種：L20-1 的卡點是 R6（單次提交臂的輸出未驗）。「見證是紅的」原本只是
    # block.next 裡的一句散文，誰都能寫、也沒人驗。這裡把它變成**當場重跑**：兩份 m123
    # summary（對照臂不開旗標、見證臂開旗標）丟進 r6_witness.judge ⇒
    # WITNESS_GREEN／WITNESS_RED／REFUSE。
    # ⚠ 與其他端點不同：判詞直接回傳三態，不映射成 FLIP／WITHIN —— D7b 拿它跟宣告的
    #   verdict 比，所以宣告 WITNESS_RED 而判詞變 GREEN／REFUSE ⇒ 看板紅（該回來重判 L20-1）。
    # fail-closed：產物不在、缺 control／arm、載不進、判詞不認得 ⇒ 一律不放行。
    if tool == "r6_witness":
        ctl, armo = cq.get("control"), cq.get("arm")
        if not (ctl and armo):
            return None, "counter_quote 缺 control／arm（見證端點必須指名兩份 summary）"
        for p_ in (ctl, armo):
            if not os.path.exists(os.path.join(root, p_)):
                return None, "見證產物不存在：%s" % p_
        try:
            import r6_witness as W
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 r6_witness：%s" % exc
        try:
            res = W.judge(ctl, armo, root=root)
        except Exception as exc:  # noqa: BLE001
            return None, "r6_witness 判不了：%s: %s" % (type(exc).__name__, exc)
        v = res.get("verdict")
        if v not in ("WITNESS_GREEN", "WITNESS_RED", "REFUSE"):
            return None, "判詞不認得：%r" % v
        return v, res.get("why") or ""
    # ── 儀器端點（第三種）：ρ 的**收貨端**活著且可複現（2026-09-30，L20-7 是第一例）────────
    # 為什麼需要這一種：S3-b 的預註冊把端點拆成「① 計數器（髒窗可引用）／② 時間」。那一趟
    # ② 判不可辨識（五支臂無一 attribution=none），但 ① **成立**——而 ① 若只留在散文裡，
    # 下一個人就無法分辨「量具跑了」與「量具回常數」（2026-09-26 的 4.76 ms 就是估值的路徑）。
    # ⚠ 它判的是**量具**：控制臂不得有 `CGC-RHO*`、見證臂每步 n_layer 層、`skip=0`、覆蓋
    #   過源頭飽和門檻、多場次一致、fill 記帳全符。它**不**判 t/s、也**不**判價格。
    # fail-closed：檔不在、缺 control／arm、判詞不認得 ⇒ 一律不放行。
    if tool == "rho_instrument_ab":
        ctl = cq.get("control")
        arms_ = cq.get("arm")
        if isinstance(arms_, str):
            arms_ = [arms_]
        if not (ctl and arms_):
            return None, "counter_quote 缺 control／arm（ρ 儀器端點必須指名對照臂與見證臂）"
        for p_ in [ctl] + list(arms_) + ([cq["fill"]] if cq.get("fill") else []):
            if not os.path.exists(os.path.join(root, p_)):
                return None, "見證 log 不存在：%s" % p_
        try:
            import rho_instrument_ab as R
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 rho_instrument_ab：%s" % exc
        try:
            v, why, _ = R.judge(ctl, list(arms_), cq.get("fill"), root=root)
        except Exception as exc:  # noqa: BLE001
            return None, "rho_instrument_ab 判不了：%s: %s" % (type(exc).__name__, exc)
        if v not in ("INSTRUMENT_LIVE", "REFUSE"):
            return None, "判詞不認得：%r" % v
        return v, why
    # ── 判詞重播端點（第五種）：ρ 的**價格**（2026-10-01，L20-7 的第二個端點）───────
    # 為什麼需要這一種：L20-7 的價格端點是一份**判詞**（四條權威 row、兩趟反序），判它的
    # 是 `rho_price_authrow.judge()`（預註冊）。若不當場重跑，`verdict.json` 只是一句散文、
    # 誰都能改。這裡只做一件事：從四份產物**當場重算**判詞，並要求它等於宣告的 verdict
    # （fail-closed：產物不齊、判不了、判詞不認得 ⇒ 不放行）。
    if tool == "rho_price_authrow":
        art = cq.get("artifact")
        if not art:
            return None, "counter_quote 缺 artifact（判詞端點必須指名 verdict.json）"
        ap_ = os.path.join(root, art)
        if not os.path.exists(ap_):
            return None, "見證產物不存在：%s" % art
        try:
            import rho_price_authrow as RPA
        except Exception as exc:  # noqa: BLE001
            return None, "載不進 rho_price_authrow：%s" % exc
        by = {}
        for c in RPA.plan(root):
            cmd = list(c.get("cmd") or [])
            if "--json" not in cmd:
                return None, "plan() 的命令沒有 --json ⇒ 判不了（工具漂了）"
            by[(int(c["order"]), c["arm"])] = os.path.join(root, cmd[cmd.index("--json") + 1])
        for k in ((1, "A"), (1, "B"), (2, "A"), (2, "B")):
            if not os.path.exists(by.get(k) or ""):
                return None, "四份權威 row 產物不齊：缺 order%d_%s" % k
        try:
            oa = RPA.judge(by[(1, "A")], by[(1, "B")])
            ob = RPA.judge(by[(2, "A")], by[(2, "B")])
            res = RPA.both_orders(oa, ob)
        except Exception as exc:  # noqa: BLE001
            return None, "rho_price_authrow 判不了：%s: %s" % (type(exc).__name__, exc)
        v = res.get("verdict")
        if v not in ("PRICE", "NO_EFFECT", "REFUSE"):
            return None, "判詞不認得：%r" % v
        return v, res.get("why") or ""
    if tool != "g3_zeroslot_ab":
        return None, ("counter_quote.tool 不認得：%r（已知：g3_zeroslot_ab、mtp_round_split、"
                      "fill_term_ab、samecell_margin、p1_feed_ab、r6_witness、rho_instrument_ab、"
                      "rho_price_authrow）") % tool
    a, b = cq.get("artifact_a"), cq.get("artifact_b")
    if not (a and b):
        return None, "counter_quote 缺 artifact_a／artifact_b"
    try:
        import g3_zeroslot_ab as G
    except Exception as exc:  # noqa: BLE001
        return None, "載不進 g3_zeroslot_ab：%s" % exc
    try:
        pa, pb = G.resolve_log(os.path.join(root, a)), G.resolve_log(os.path.join(root, b))
        if pa is None or pb is None:
            return None, "見證產物不存在：%s" % (a if pa is None else b)
        verdict, _ = G.classify(G.parse(pa), G.parse(pb))
    except Exception as exc:  # noqa: BLE001
        return None, "g3_zeroslot_ab 判不了：%s: %s" % (type(exc).__name__, exc)
    return verdict, ""


# ── D17：棘輪 vs 認證表（2026-10-01；operator：「--check 要驗『棘輪與已認證表一致』」）────────
# 為什麼要這一條：棘輪的單一來源是看板 `certify_anchor`，而入表的紀錄在 `certified` 那一份清單
# —— 兩邊各自手寫就會漂移，而漂移是**靜默**的（上限指的不是表上最高的一件；或某一列宣稱
# 「入表時高過當時的上限」，其實沒有）。判準（全部機器強制，違反 ⇒ --check rc=1）：
#   ① 有 decode 讀數的認證列，必須宣告 `ratchet:`（參與棘輪）或 `ratchet_exempt: <why>`
#      —— 兩個都沒有 ⇒ 紅（否則「不寫欄位」就是繞過棘輪的方式）。
#   ② `ratchet.decode_ts` 必須**逐字**等於它自己 `quote.artifact` 的 decode 讀數（容差＝
#      caliber_certify.HOLDER_EPS，與認證入口同一個值；不四捨五入）。
#   ③ 每一列 decode_ts **嚴格高於**自己宣告的 `entered_above`（入表當時的上限；0＝棘輪未立）。
#   ④ 入表時的上限只能是**表上已經有的讀數**（或 0）：把列依 entered_above 分組，任一組的上限必須
#      等於「所有更小上限組」的最大 decode_ts ⇒ 沒有自創的錨、沒有跳號。
#   ⑤ `certify_anchor.decode_ts` 必須等於表上最大的 decode_ts，且 `certify_anchor.artifact` 必須
#      就是那一列的 `quote.artifact`（上限指的就是那一件）。
RATCHET_EPS_FALLBACK = 1e-9
RATCHET_TOL_FALLBACK = 1e-4


def ratchet_constants():
    """棘輪的兩個浮點常數：**唯一來源**＝ caliber_certify（認證入口）；載不進才用同名退路。"""
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import caliber_certify as CC
        return float(CC.RATCHET_EPS), float(CC.HOLDER_EPS)
    except Exception:  # noqa: BLE001
        return RATCHET_EPS_FALLBACK, RATCHET_TOL_FALLBACK


def ratchet_consistency(board, root=ROOT):
    """D17：看板 `certify_anchor`（棘輪上限）與 `certified` 清單必須互相講得通 ⇒ [errs]。"""
    errs = []
    entries = [c for c in (board.get("certified") or []) if isinstance(c, dict)]
    if not entries:
        return errs
    eps, tol = ratchet_constants()
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import quote_gate as QG
    except Exception as exc:  # noqa: BLE001
        return ["D17 載不進 quote_gate：%s: %s" % (type(exc).__name__, exc)]

    def decode_readings(rel):
        """那個產物裡的 decode 讀數（判準＝quote_gate；decode row＝shape `p0/...`）⇒ [(avg, verdict)]。"""
        if not rel:
            return []
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            return []
        out = []
        for r in QG.scan([p]):
            if not str(r.get("shape") or "").startswith("p0/"):
                continue
            avg = (r.get("metrics") or {}).get("avg")
            if avg is not None:
                out.append((float(avg), r.get("verdict")))
        return out

    rows = []
    for c in entries:
        cid = c.get("id") or "?"
        q = c.get("quote") or {}
        reads = decode_readings(q.get("artifact"))
        if not reads:
            continue                       # 這一列沒有可判的 decode 讀數（例如 prefill／比值類）
        rat, exempt = c.get("ratchet"), c.get("ratchet_exempt")
        if rat is None and exempt is None:
            errs.append("D17 %s 有 decode 讀數（%s）卻沒宣告 ratchet／ratchet_exempt ⇒ 棘輪的一致性"
                        "沒得驗（參與棘輪的給 ratchet，基準的給 ratchet_exempt＋why）" % (cid, q.get("artifact")))
            continue
        if rat is None:
            if not str(exempt or "").strip():
                errs.append("D17 %s 的 ratchet_exempt 是空的（要寫 why：為什麼這一列不必高過上限）" % cid)
            continue
        if not isinstance(rat, dict) or rat.get("decode_ts") in (None, ""):
            errs.append("D17 %s 的 ratchet 缺 decode_ts" % cid)
            continue
        if q.get("verdict") != "QUOTABLE":
            errs.append("D17 %s 宣告參與棘輪，但 quote.verdict=%r ≠ QUOTABLE（棘輪只認可引用的讀數）"
                        % (cid, q.get("verdict")))
            continue
        try:
            ts = float(rat["decode_ts"])
            above = float(rat.get("entered_above") or 0.0)
        except (TypeError, ValueError):
            errs.append("D17 %s 的 ratchet.decode_ts／entered_above 不是數字（%r／%r）"
                        % (cid, rat.get("decode_ts"), rat.get("entered_above")))
            continue
        if not [a for a, _v in reads if abs(a - ts) <= tol]:
            errs.append("D17 %s 宣告的 decode_ts=%s 不在 %s 的 decode 讀數裡（有 %s）⇒ 逐字抄產物的 avg_ts"
                        % (cid, ts, q.get("artifact"), "、".join("%.6f" % a for a, _v in reads)))
            continue
        if ts <= above + eps:
            errs.append("D17 %s 的 decode_ts=%s 沒有嚴格高於它入表時的上限 %s ⇒ 這一列當時進不去"
                        % (cid, ts, above))
        rows.append((cid, ts, above))

    if not rows:
        return errs
    # ④ 入表時的上限只能是表上已有的讀數（或 0）
    groups = {}
    for cid, ts, above in rows:
        groups.setdefault(above, []).append((cid, ts))
    running = 0.0
    for key in sorted(groups):
        if key > 0 and abs(key - running) > eps:
            errs.append("D17 有一組列的 entered_above=%s 不等於當時表上的最大值 %s ⇒ 上限只能是表上已有的"
                        "讀數（或 0）：%s" % (key, running, "、".join(c for c, _t in groups[key])))
        running = max(running, max(t for _c, t in groups[key]))
    # ⑤ 上限＝表上最高那一件
    top = max(rows, key=lambda r: r[1])
    anchor = board.get("certify_anchor") or {}
    try:
        a_ts = float(anchor.get("decode_ts"))
    except (TypeError, ValueError):
        errs.append("D17 certify_anchor.decode_ts 不是數字（%r）⇒ 棘輪沒有單一來源"
                    % (anchor.get("decode_ts"),))
        return errs
    if abs(a_ts - top[1]) > eps:
        errs.append("D17 certify_anchor.decode_ts=%s ≠ 表上最大的 decode_ts=%s（%s）⇒ 上限必須等於"
                    "已認證表裡最高的那一件" % (a_ts, top[1], top[0]))
    top_art = next((((c.get("quote") or {}).get("artifact")) for c in entries
                    if c.get("id") == top[0]), None)
    a_art = anchor.get("artifact")
    if a_art and top_art and os.path.normpath(a_art) != os.path.normpath(top_art):
        errs.append("D17 certify_anchor.artifact=%s ≠ 最高那一列（%s）的 quote.artifact=%s ⇒ 上限指的"
                    "不是那一件" % (a_art, top[0], top_art))
    return errs


def validate(board, nodes, root=ROOT):
    errs, targets, by_id = [], [], {}
    rgate_cache = {}          # D12 的掃描快取（同一輪 build 內共用，避免重複掃全語料）
    for layer in board.get("layers") or []:
        for t in layer.get("targets") or []:
            t = dict(t); t["_layer"] = layer.get("id"); targets.append(t)
    for t in targets:
        if not t.get("id"):
            errs.append("D3 子目標缺 id"); continue
        if t["id"] in by_id:
            errs.append("D3 子目標 id 重複：%s" % t["id"])
        by_id[t["id"]] = t
        for f in TARGET_FIELDS:
            if f not in t or t[f] in (None, "", [], {}):
                errs.append("D3 %s 缺欄位 %s" % (t["id"], f))
        if not isinstance(t.get("stage"), int) or not (0 <= t["stage"] <= 5):
            errs.append("D3 %s 的 stage 必須是 0..5 的整數" % t["id"])
        b = t.get("badge") or {}
        if not b.get("text"):
            errs.append("D3 %s 缺 badge.text" % t["id"])
        if b.get("tone") not in BADGE_TONES:
            errs.append("D3 %s 的 badge.tone 不合法：%r" % (t["id"], b.get("tone")))
        # D5 立項
        ch = t.get("charter")
        for c in ([ch] if isinstance(ch, str) else (ch or [])):
            if not os.path.exists(os.path.join(root, c)):
                errs.append("D5 %s 的立項卡不存在：%s" % (t["id"], c))
        # D6 結案規則
        ev = t.get("evidence") or {}
        st = str(t.get("state") or "")
        # D11（2026-09-30 §63）：`state`（判定）與 `settled`（看板顯示：整列綠底）必須一致。
        # 兩邊各自演化就會出現「結案但沒綠底」或「綠底但紀錄說還在跑」——兩種都比少一條規則糟。
        _settled = bool(t.get("settled"))
        if st.startswith("結案") and not _settled:
            errs.append("D11 %s 的 state 以『結案』開頭卻沒有 settled ⇒ 看板不會整列綠底"
                        "（顯示與判定不一致）" % t["id"])
        if _settled and st.startswith("未結案"):
            errs.append("D11 %s 有 settled（已了結綠底）但 state 仍以『未結案』開頭" % t["id"])
        # D12 前置閘門（2026-09-30 §64）：宣告「可跑」之前，先證明**這一格要的判決還不存在**。
        # 為什麼：L25-6 一直寫「✅ 現在能跑 — 前置＝加大配對 n」，而它要的那個判決
        # （事先登記的 N=12 檢定）**兩晚前就跑完並留在樹上** ⇒ 差一點用 ~40 趟 launch 重印一次。
        # 判準在 scripts/check/runnable_gate.py（當場重跑；判不了就紅 ⇒ fail-closed）。
        try:
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            import runnable_gate as RG
            errs.extend(RG.check_target(t, root, rgate_cache)["errs"])
        except Exception as exc:  # noqa: BLE001
            errs.append("D12 %s 的前置閘門載不進／跑不動：%s: %s"
                        % (t["id"], type(exc).__name__, exc))
        if st.startswith("結案"):
            if st.startswith("結案（排除）") or ev.get("cert_by") == "排除附機制":
                if not (ev.get("cert_by") and ev.get("source")):
                    errs.append("D6 %s 是『結案（排除）』但缺 cert_by／source（排除也要指出出處）" % t["id"])
            else:
                if ev.get("profile") != "prod-new":
                    errs.append("D6 %s 標結案，但 evidence.profile=%r ≠ prod-new" % (t["id"], ev.get("profile")))
                if ev.get("entry") != "harness bench":
                    errs.append("D6 %s 標結案，但 evidence.entry=%r ≠ `harness bench`" % (t["id"], ev.get("entry")))
                if ev.get("meets") is not True:
                    errs.append("D6 %s 標結案，但 evidence.meets 不是 true（未達預期值）" % t["id"])
                if str(ev.get("value") or "").strip() in ("", "—"):
                    errs.append("D6 %s 標結案，但 evidence.value 為空" % t["id"])
        # D7 引用閘門（2026-09-29 §55）：宣告的可引用性必須與閘門當場的判定一致。
        # 為什麼要這條：`attribution=none` 曾是唯一的准入條件，但交付 cell 唯一一場
        # `none` 的讀數是**最慢且離散 46%** 的那一場（rep 3 慢了 2.7 倍）⇒ 標籤必要不充分。
        # 判準現在由 scripts/check/quote_gate.py 持有（逐 rep 離散 ＋ 樣本數 ＋ 窗口 ＋ 參考帶）。
        q = ev.get("quote")
        closed_real = st.startswith("結案") and not any(st.startswith(x) for x in D7_EXCLUDED_CLOSURES) \
            and ev.get("cert_by") != "排除附機制"
        if q:
            if not isinstance(q, dict) or not q.get("artifact") or not q.get("verdict"):
                errs.append("D7 %s 的 evidence.quote 缺 artifact／verdict" % t["id"])
            else:
                live, why = gate_verdict(root, q["artifact"])
                if live is None:
                    errs.append("D7 %s 的引用判不了（%s）" % (t["id"], why))
                elif live != q["verdict"]:
                    errs.append("D7 %s 宣告 verdict=%s，但閘門現在判 %s（%s）"
                                % (t["id"], q["verdict"], live, q["artifact"]))
        # D7b 計數器端點：宣告 counter_quote 的格，當場重跑檢查器驗它（見 counter_verdict）。
        # ⚠ 它與時間引用**互斥但不強迫**：有 counter_quote 的格不必再拿一個時間引用（那正是
        #   本條存在的理由）；反之亦然。兩者都宣告就兩者都驗。
        # [2026-10-01] 一格可以有**兩個以上**端點（L20-7 是第一例：① ρ 的儀器端點、② ρ 的價格端點）
        # ⇒ `counter_quotes`（複數）與單數 `counter_quote` 並存，**每一個都當場重跑**（fail-closed）。
        #   為什麼不是「留一個就好」：兩個端點判的是兩件事（量具活著／價格是多少），砍掉任一個
        #   就等於那個端點的漂移不再有人看見 —— 那正是 D7b 存在的理由。
        cqs = ([ev["counter_quote"]] if ev.get("counter_quote") else []) \
            + list(ev.get("counter_quotes") or [])
        for cq in cqs:
            if not isinstance(cq, dict) or not cq.get("verdict"):
                errs.append("D7b %s 的 counter_quote 缺 verdict" % t["id"])
            else:
                live, why = counter_verdict(root, cq)
                if live is None:
                    errs.append("D7b %s 的計數器判詞判不了（%s）" % (t["id"], why))
                elif live != cq["verdict"]:
                    errs.append("D7b %s 宣告 counter verdict=%s，但檢查器現在判 %s"
                                % (t["id"], cq["verdict"], live))
        if closed_real and cqs:
            want = COUNTER_CLOSE_VERDICTS.get(ev.get("metric"))
            have = [c.get("verdict") for c in cqs if isinstance(c, dict)]
            if want is None:
                errs.append("D7b %s 用 counter_quote 結案，但 evidence.metric=%r 不在 %s"
                            % (t["id"], ev.get("metric"), list(COUNTER_METRICS)))
            elif want not in have:
                errs.append("D7b %s 用端點結案，但宣告的端點判詞 %s 裡沒有 %s（metric=%s）"
                            % (t["id"], have, want, ev.get("metric")))
        if closed_real and not cqs:
            if not q:
                errs.append("D7 %s 標結案但沒有 evidence.quote（結案必須有一場可引用的讀數）" % t["id"])
            elif q.get("verdict") != "QUOTABLE":
                errs.append("D7 %s 標結案，但引用判詞是 %s ≠ QUOTABLE" % (t["id"], q.get("verdict")))
        # D10 臂必須量得到交付目標（2026-09-30）—— R5／R6 的**看板面**。
        # 為什麼：20+／25+ 目前沒有一列是在交付目標的函數上、用活著的計數器量到的，
        # 而 `options.arms` 是「跑這一格要開什麼」的唯一來源 ⇒ 它留著單次提交臂
        # （per-layer hook 被跳過、計數器回傳常數）或源碼明文 never-quote 的量具時，
        # 下一個人照著跑只會再得到一個不可引用的數。這種臂不准當**選項**。
        o = t.get("options") or {}
        raw_arms = o.get("arms") or []
        live_arms = []          # [(名字, 臂字串, 宣告 dict 或 None)]
        for a in raw_arms:
            if isinstance(a, str):
                live_arms.append((a, a, None))
            elif isinstance(a, dict) and a.get("arm"):
                live_arms.append((str(a["arm"]), str(a["arm"]), a))
            else:
                errs.append("D10 %s 的 options.arms 每一項要是字串或 {arm, role, why, source}"
                            "（收到 %r）" % (t["id"], a))
        lived = {a.lstrip("-").strip() for a, _, _ in live_arms}
        for name, arm, rec in live_arms:
            flags, why = arm_void_flags(arm)
            if flags is None:
                errs.append("D10 %s 的臂判不了（%s）" % (t["id"], why)); continue
            if not flags:
                continue
            # `role: reference`（2026-09-30）：這一支是**成對量測的基準端**（終點上界或底），
            # 不是通往目標的路 —— 它帶著 R5／R6 的量具是**本來就如此**。放行的代價是它必須
            # 附 why ＋ source（例外要留得下理由與出處），而且畫面上標成「基準臂」，
            # 免得下一個人把它報出來的 t/s（例：NOFILL 的 13.75）當成交付讀數。
            if rec is not None and str(rec.get("role") or "") == "reference":
                if not rec.get("why") or not rec.get("source"):
                    errs.append("D10 %s 的基準臂 %s 缺 why／source（例外必須留下理由與出處）"
                                % (t["id"], arm))
                continue
            errs.append("D10 %s 的 options.arms 留著不能量交付目標的臂（%s）：%s"
                        % (t["id"], "、".join("%s=%s" % kv for kv in sorted(flags.items())), arm))
        removed = o.get("arms_removed")
        if removed is not None and not isinstance(removed, list):
            errs.append("D10 %s 的 arms_removed 必須是清單" % t["id"])
        recs = [r for r in (removed if isinstance(removed, list) else []) if isinstance(r, dict)]
        for rec in recs:
            if not rec.get("arm"):
                errs.append("D10 %s 的 arms_removed 每筆要有 arm" % t["id"]); continue
            arm = str(rec["arm"])
            if arm.lstrip("-").strip() in lived:
                errs.append("D10 %s 的 %s 同時列在 arms 與 arms_removed（要嘛能跑、要嘛已刪）"
                            % (t["id"], arm))
            flags, why = arm_void_flags(arm)
            if flags is None:
                errs.append("D10 %s 的 arms_removed %s 判不了（%s）" % (t["id"], arm, why)); continue
            if not flags:
                errs.append("D10 %s 的 arms_removed 收了沒有 R5／R6 量具的臂（%s）"
                            "—— 這一欄只能放判準擋掉的臂" % (t["id"], arm))
            if not str(rec.get("why") or "").strip():
                errs.append("D10 %s 的 arms_removed %s 缺 why（刪除必須留理由）" % (t["id"], arm))
            got = {str(k): str(v) for k, v in (rec.get("flags") or {}).items()}
            if got != flags:
                errs.append("D10 %s 的 arms_removed %s 的 flags=%r 與登記表算出的 %r 不符"
                            "（刪除紀錄必須與 quote_gate 的 R5／R6 登記表一致）"
                            % (t["id"], arm, got, flags))
            rules = [str(r) for r in (rec.get("rules") or [])]
            if not rules or sorted(set(rules)) != sorted(set(flags.values())):
                errs.append("D10 %s 的 arms_removed %s 的 rules=%r 與 flags 不符 %r"
                            % (t["id"], arm, rules, sorted(set(flags.values()))))
            for r in rules:
                if r not in ("R5", "R6"):
                    errs.append("D10 %s 的 arms_removed %s 的 rule=%r 不合法（只收 R5／R6）"
                                % (t["id"], arm, r))
        if len({str(r.get("arm")) for r in recs}) != len(recs):
            errs.append("D10 %s 的 arms_removed 有重複的臂" % t["id"])
        # D16 臂類別 vs 卡點型別（2026-09-30）——L20-1 的實例，也是我自己踩的坑：
        # 那一格 `options.arms` 是空的（唯一一支臂被 R6 刪掉），但 `precondition.block.kind=window`
        # ⇒ 看板同時說「不能出可引用讀數」和「等窗口就好」。順著那句話讀，結論會是「它只差窗口」——
        # 而 R6（輸出未驗）的解**不是窗口**：抽到再乾淨的窗口，這支臂的 t/s 仍然不可引用。
        # 這一條不重寫判準（R5／R6 的定義仍在 quote_gate 的登記表），只把兩欄的事實對齊，且三件都要：
        #   ① 空臂 ＋ 未了結 ⇒ 不准沉默（block 必須存在 —— 卡點是臂的類別，不是判決）
        #   ② `block.kind` 不准是 window（window 只在「臂是對的、只是窗口髒」時才是正確答案）
        #   ③ 內文必須指名擋住它的規則，否則讀的人找不到登記表裡寫死的解除條件
        if live_arms == [] and recs and not t.get("settled"):
            _blk = (t.get("precondition") or {}).get("block")
            if not isinstance(_blk, dict):
                errs.append("D16 %s 的 options.arms 是空的（每一支臂都被 R5／R6 刪掉），但沒有 "
                            "precondition.block —— 這一格跑不了的原因不是判決，也不是窗口，是臂的類別"
                            % t["id"])
            else:
                _bk = str(_blk.get("kind") or "")
                if _bk == "window":
                    errs.append("D16 %s 的 arms 空（全被 R5／R6 刪除）卻寫 block.kind=window —— "
                                "窗口修不好臂的身分：抽到再乾淨的窗口，這支臂的讀數仍然不可引用"
                                "（要嘛換一支不在登記表裡的臂、要嘛走登記表寫死的解除條件）" % t["id"])
                elif _bk not in ("decision", "code", "mechanism"):
                    errs.append("D16 %s 的 block.kind=%r 不合法（空臂的格只收 decision／code／mechanism）"
                                % (t["id"], _bk))
                _txt = " ".join(str(_blk.get(k) or "") for k in ("why", "next")) \
                    + " " + str((_blk.get("assert") or {}).get("why") or "") \
                    + " " + str((t.get("precondition") or {}).get("why") or "")
                if not re.search(r"\bR[567]\b", _txt):
                    errs.append("D16 %s 的卡點內文沒有指名擋住它的規則（R5／R6／R7）—— 解除條件"
                                "寫在 quote_gate 的登記表裡，這裡至少要指得出來" % t["id"])
        for f in ("expect_ms", "expect_tps"):
            v = str(t.get(f) or "").strip()
            if v.startswith("—") or v == "":
                continue          # 「—（說明）」＝本格不主張量化預期，允許
            if not any(c.isdigit() for c in v):
                errs.append("D3 %s 的 %s 既無數字也非「—」（預期必須可量化）" % (t["id"], f))
        for n in t.get("nodes") or []:
            if n not in nodes:
                errs.append("D1 %s 指向不存在的節點 %s" % (t["id"], n))
    covered = {n for t in targets for n in (t.get("nodes") or [])}
    seen = set()
    # ⚠ `notes` **仍然必需**，雖然 09-30 之後不再渲染：它是 D2 的唯一依據（每個未結案節點
    #   必須「被子目標覆蓋」或「被分類」）。拿掉它會讓下面那條 D2 對所有里程碑／A 軸節點報紅，
    #   或者（若順手也拿掉 D2）讓「全數有歸屬」從一條會紅的閘門降級成一句沒人驗的聲明。
    for note in board.get("notes") or []:
        n = note.get("node")
        if n not in nodes:
            errs.append("D1 分類指向不存在的節點 %s" % n); continue
        if n in seen:
            errs.append("D3 節點 %s 有兩筆分類" % n)
        seen.add(n)
        cls = note.get("class")
        if cls not in CLASSES:
            errs.append("D3 %s 的分類不合法：%r" % (n, cls)); continue
        for f in NEEDS[cls]:
            if not note.get(f):
                errs.append("D3 %s（class=%s）缺欄位 %s" % (n, cls, f))
        if cls == "handled" and note.get("by"):
            for tok in [x.strip() for x in str(note["by"]).split("／") if x.strip()]:
                if tok not in by_id:      # 複合格（L20-1／2／3）逐段驗
                    errs.append("D3 %s 的 by 指到不存在的子目標：%s" % (n, tok))
        covered.add(n)
    # `stale`（節點文字就地更新的紀錄）已於 2026-09-30 隨「本頁只保留三段」一併從 YAML 移除；
    # 它沒有任何閘門依賴它（純敘述），所以這裡不留死碼。
    for n in [x for x in live_nodes(nodes) if x not in covered]:
        errs.append("D2 未結案節點既沒被子目標覆蓋、也沒被分類：%s" % n)
    cr = board.get("closure_rule") or {}
    if not cr.get("text") or not cr.get("requires"):
        errs.append("D3 缺 closure_rule.text／requires（結案規則必須明寫）")
    # D8 主節點（L20／L25）的目標／證據：綠燈只能建在**可引用的讀數**上（§55 的精神）。
    for layer in board.get("layers") or []:
        lg, lev = layer.get("goal") or {}, layer.get("evidence") or {}
        if not lg and not lev:
            continue          # 未宣告目標的圖層：不判定、不變色（「若無則沒有變化」）
        if lg and lg.get("target") is None:
            errs.append("D8 %s 的 goal 缺 target" % layer.get("id"))
        if lg and lev.get("value") is None:
            errs.append("D8 %s 有 goal 卻沒有 evidence.value（達標必須有一場實測）" % layer.get("id"))
        if lev:
            if not lev.get("artifact"):
                errs.append("D8 %s 的 layer.evidence 缺 artifact（達標的讀數必須指名產物）" % layer.get("id"))
            else:
                live, why = gate_verdict(root, lev["artifact"])
                if live is None:
                    errs.append("D8 %s 的 layer.evidence 判不了（%s）" % (layer.get("id"), why))
                elif lev.get("quote_verdict") != live:
                    errs.append("D8 %s 宣告 quote_verdict=%s，但閘門現在判 %s"
                                % (layer.get("id"), lev.get("quote_verdict"), live))
            if lev.get("met") is True and lev.get("quote_verdict") != "QUOTABLE":
                errs.append("D8 %s 宣告達標，但引用判詞是 %s ≠ QUOTABLE（不可引用的讀數不能變綠）"
                            % (layer.get("id"), lev.get("quote_verdict")))
        # met 只能由 layer_goal() 算出來；YAML 明文宣告 met=false 卻算成 true 也要擋
        met, _why = layer_goal(layer, root)
        if lev.get("met") is not None and bool(lev.get("met")) != bool(met):
            errs.append("D8 %s 宣告 met=%s，但依 goal ∩ evidence 算是 %s"
                        % (layer.get("id"), lev.get("met"), met))

    for c in board.get("certified") or []:
        for f in ("id", "item", "value", "profile", "entry", "meets", "source"):
            if c.get(f) in (None, "", []):
                errs.append("D3 certified %s 缺欄位 %s" % (c.get("id"), f))
        if c.get("quote"):
            live, why = gate_verdict(root, (c["quote"] or {}).get("artifact"))
            if live is None:
                errs.append("D7 certified %s 的引用判不了（%s）" % (c.get("id"), why))
            elif live != (c["quote"] or {}).get("verdict"):
                # D7-waiver（operator 2026-10-02）：本列明文帶 `waiver` 且宣告 QUOTABLE、而閘門（**不帶**
                # 放行 env）判 DIRTY ⇒ 那是「被 operator 顯式放行的可引用」，不是「閘門說這一場乾淨」。
                # 揭露落在本列的 item／value／waiver／why（看板渲染前三者）。刻意只放行 DIRTY（＝R4 窗口
                # 那一條），UNSTABLE／REFUSE 一律照擋 —— 同 `quote_ungateable` 的樣式：要放行就必須寫理由。
                # ── 2026-10-06 擴充：waiver_axis（按軸放行）──
                # 若本列聲明 `waiver_axis`（如 "decode"），則只看該軸 row 的 verdict；該軸純 DIRTY 即放行，
                # 其餘軸（如 prefill）的 UNSTABLE 不連坐——前提是那些軸不在本列聲明範圍內（由 why 揭露）。
                # 不帶 waiver_axis 時退化為原語意（整檔純 DIRTY 才放），不削弱其他認證。
                _decl = (c["quote"] or {}).get("verdict") == "QUOTABLE"
                _waived = bool(c.get("waiver") and _decl)
                if _waived and live == "DIRTY":
                    pass
                elif _waived and c.get("waiver_axis"):
                    _ax = c["waiver_axis"]
                    try:
                        _qg_mod = _qg()
                        _recs = _qg_mod.scan([os.path.join(root, (c["quote"] or {}).get("artifact"))])
                        _ax_v = _axis_verdict(_recs, _ax)
                    except Exception as exc:  # noqa: BLE001
                        errs.append("D7 certified %s 按軸 %s 判讀失敗：%s" % (c.get("id"), _ax, exc))
                        _ax_v = None
                    if _ax_v != "DIRTY":
                        errs.append("D7 certified %s 宣告 verdict=%s，但閘門現在判 %s（按軸 %s 判 %s，非純 DIRTY）"
                                    % (c.get("id"), (c["quote"] or {}).get("verdict"), live, _ax, _ax_v))
                else:
                    errs.append("D7 certified %s 宣告 verdict=%s，但閘門現在判 %s"
                                % (c.get("id"), (c["quote"] or {}).get("verdict"), live))
        elif c.get("meets") is True and not c.get("quote_ungateable"):
            errs.append("D7 certified %s 標 meets 卻沒有 evidence 可判（要嘛給 quote、"
                        "要嘛明文 quote_ungateable 附理由）" % c.get("id"))
        if c.get("meets") is True and (c.get("profile") != "prod-new" or c.get("entry") != "harness bench"):
            errs.append("D6 certified %s 標 meets，但 profile／entry 不是 prod-new／harness bench" % c.get("id"))

    # D17 棘輪 vs 認證表（2026-10-01；operator：「--check 要驗『棘輪與已認證表一致』」）
    errs.extend(ratchet_consistency(board, root))

    # ── D15：待升級（operator 2026-09-30「搬成功要放到已認證」）────────────────────────
    # 這一格的價值在於**跑之前就把條件寫死**，而且跑完之後**不會被忘記**：判準一旦當場成立
    # 而它還躺在 pending，`--check` 就紅。升級的是一個**分布**，不是一個單點（§73.5）。
    cert_ids = set(c.get("id") for c in (board.get("certified") or []))
    try:
        adm_profiles = list(_qg().QUOTABLE_PROFILES)
    except Exception:  # noqa: BLE001
        adm_profiles = ["prod-new"]
    for p in board.get("pending_promotion") or []:
        if p.get("id") in cert_ids:
            errs.append("D15 %s 同時在 certified 與 pending_promotion（升級要二選一）" % p.get("id"))
        for f in ("id", "item", "why", "accept", "judge", "state", "arms"):
            if p.get(f) in (None, "", []):
                errs.append("D15 pending %s 缺欄位 %s" % (p.get("id"), f))
        if p.get("charter") and not os.path.exists(os.path.join(root, p["charter"])):
            errs.append("D15 pending %s 的 charter 不存在（%s）" % (p.get("id"), p["charter"]))
        # 候選臂必須在認可口徑上（R7）⇒ 這一格量到什麼都升不了級的話，就不該擺在這裡
        for arm in (p.get("arms") or []):
            prof = str(arm).split(":", 1)[0].strip()
            if prof not in adm_profiles:
                errs.append("D15 pending %s 的候選臂 %r 不在認可口徑 %s 上（R7 ⇒ 量到也不能升）"
                            % (p.get("id"), arm, adm_profiles))
        j = p.get("judge") or {}
        if j.get("kind") not in PENDING_JUDGE_KINDS:
            errs.append("D15 pending %s 的 judge.kind=%r 不認得（只收 %s）"
                        % (p.get("id"), j.get("kind"), list(PENDING_JUDGE_KINDS)))
            continue
        try:
            mn_l = int(j.get("min_launches") or 0)
            mn_q = int(j.get("min_quotable") or 0)
        except (TypeError, ValueError):
            errs.append("D15 pending %s 的 judge 數字不是整數（min_launches=%r、min_quotable=%r）"
                        % (p.get("id"), j.get("min_launches"), j.get("min_quotable")))
            continue
        if mn_l < 1 or mn_q < 1 or mn_q > mn_l:
            errs.append("D15 pending %s 的 judge 數字不合理（min_launches=%s、min_quotable=%s）"
                        % (p.get("id"), mn_l, mn_q))
            continue
        n_l, n_q, how = pending_cluster(root, j)
        if n_l is None:
            errs.append("D15 pending %s 判不了：%s" % (p.get("id"), how))
            continue
        if n_l >= mn_l and n_q >= mn_q:
            errs.append("D15 pending %s **條件已達成**（%d 場可引用／門檻 %d、共 %d 場）卻還在 pending"
                        " ⇒ 升進 certified（附 quote）：%s"
                        " ▶ 機器路徑：python3 scripts/check/decode_board_build.py --promote %s"
                        "（或跑一次 board_pipeline.sh：重建模式會先自動 promote）"
                        % (p.get("id"), n_q, mn_q, n_l, how, p.get("id")))
    out = board.get("out") or ""
    if not out.startswith("docs/") or not out.endswith(".html"):
        errs.append("D3 out 必須是 docs/*.html（得到 %r）" % out)
    return errs, by_id


# ────────────────────────── 渲染：看板 ──────────────────────────
def render_board(board, nodes, here, out_path):
    L = []; A = L.append
    A("<!DOCTYPE html>")
    A("<!-- generated by scripts/check/decode_board_build.py — 請勿手改本檔 -->")
    A('<html lang="zh-Hant"><head><meta charset="UTF-8">')
    A('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    A("<title>%s</title><style>%s</style></head><body>" % (board.get("title", "board"), CSS))
    A(nav(here, out_path, len(collect_targets(board))))
    A('<div class="container">')
    A("<h1>%s</h1>" % board.get("title", ""))
    A('<p class="subtitle">%s</p>' % board.get("subtitle", ""))
    cr = board.get("closure_rule") or {}
    A('<div class="success"><b>結案規則（機器強制）</b>：%s<br><span class="legend">條件：%s ｜ %s</span></div>'
      % (cr.get("text", ""), "、".join(cr.get("requires") or []), cr.get("note", "")))
    _live, _rm, _bare = arm_census(board)
    if _rm:
        A('<div class="warning"><b>臂的身分（D10）</b>：可跑 %d 支 ／ 已依 <b>R5／R6</b> 刪除 %d 支。'
          '%s判準的唯一定義在 <code>scripts/check/quote_gate.py</code>（被刪的臂在逐子目標頁有規則與理由）。'
          '</div>'
          % (_live, _rm,
             ("其中 <b>%d 格（%s）已沒有任何可跑的臂</b> ⇒ 目前沒有任何一條已知的路能量到它的交付數字。"
              % (len(_bare), "、".join(_bare))) if _bare else ""))
    _pc = precondition_census(board)
    if _pc:
        _ne, _du, _bl, _dr, _drids = _pc
        _tone = "warning" if _dr else "success"
        A('<div class="%s"><b>前置（判決存不存在）（D12）</b>：'
          '可產出新資訊 <b>%d</b> 格 ／ 只是重印已存在判決 <b>%d</b> 格 ／ 卡在人與機制 <b>%d</b> 格（共 %d 格）。'
          '%s判準的唯一定義在 <code>scripts/check/runnable_gate.py</code>；逐格的現判在逐子目標頁。'
          '</div>'
          % (_tone, _ne, _du, _bl, _ne + _du + _bl,
             ("其中 <b>%d 格（%s）宣告與現判不符</b>（fail-closed，正常不會發生）。"
              % (_dr, "、".join(_drids))) if _dr else ""))
    _q, _qc = blocked_queue(board)
    if _q:
        _LABELS = _RG.BLOCK_LABELS if "_RG" in dir() else {}
        try:
            import runnable_gate as _RG3
            _LABELS = _RG3.BLOCK_LABELS
        except Exception:  # noqa: BLE001
            _LABELS = {}
        _order = " → ".join("%s %d" % (k, _qc[k]) for k in
                            ("decision", "code", "mechanism", "window") if _qc.get(k))
        A('<div class="warning"><b>決策佇列（D13）</b>：會產出新資訊的 <b>%d</b> 格（不在佇列裡），'
          '其餘 <b>%d</b> 格各卡在一種東西上 —— %s。<br>排序＝<b>決定 → 改動 → 機制 → 窗口</b>'
          '（最便宜又解鎖最多的先做）；每一項的「下一個動作」如下，'
          '判準的唯一定義在 <code>scripts/check/runnable_gate.py</code>。</div>'
          % ((_pc[0] if _pc else 0), len(_q), _order))
        _nscan = sum(1 for r in _q if r[4] in ("HELD", "GONE", "CANNOT_JUDGE"))
        _nheld = sum(1 for r in _q if r[4] == "HELD")
        _ngone = sum(1 for r in _q if r[4] == "GONE")
        _nun = sum(1 for r in _q if r[4] == "UNSCANNED")
        if _ngone:
            A('<div class="warning"><b>⚠ 有 %d 格的卡點**已經不在樹上**</b>（宣告過期，該改可跑或結案）。</div>'
              % _ngone)
        A('<div class="legend">卡點斷言（D14）：<b>可掃描 %d／%d</b> —— 其中 <b>%d 格現判「卡點還在」</b>'
          '（HELD，每次 build 當場重掃）；<b>%d 格樹上掃不到</b>（UNSCANNED，只靠散文，逐列標示）。</div>'
          % (_nscan, len(_q), _nheld, _nun))
        A('<table class="queue"><thead><tr><th>型別</th><th>誰</th><th>格</th>'
          '<th>卡點斷言</th><th>下一個動作</th></tr></thead><tbody>')
        for _k, _ow, _tid, _nx, _lv, _lw in _q:
            _tone = {"HELD": "success", "GONE": "warning"}.get(_lv, "legend")
            A('<tr><td><b>%s</b><br><span class="legend">%s ×%d</span></td><td><code>%s</code></td>'
              '<td><code>%s</code></td><td><div class="%s"><b>%s</b><br>%s</div></td>'
              '<td class="qnext">%s</td></tr>'
              % (_k, _LABELS.get(_k, ""), _qc.get(_k, 0), _ow, _tid, _tone, _lv, _lw, _nx))
        A('</tbody></table>')
    A('<div class="grid">')
    for s in board.get("summary") or []:
        tone = (' %s' % s["tone"]) if s.get("tone") else ""
        A('<div class="card stat"><div class="stat-value%s">%s</div><div class="stat-label">%s</div></div>'
          % (tone, s.get("value", ""), s.get("label", "")))
    A("</div>")
    # `reading`（①–⑧ 的敘述區塊）已於 2026-09-30 從 YAML 刪除：它與 `certified_note` 講同一件事
    #   （認證只有三格、09-27 的 +25.7% 是冷啟 rep），同一頁講兩次就是「很亂」的來源。
    #   可操作的兩句已併入 `certified_note`；其餘 §50–§58 的細節留在各自的 docs 與 git 歷史裡。
    A('<div class="legend"><span class="steps"><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i></span> 進展＝%s ／ '
      '<span class="badge b-ok">已成立</span><span class="badge b-warn">有前置／待跑</span><span class="badge b-info">未立卡</span><span class="badge b-dead">已排除</span></div>'
      '<br>主節點（L20／L25）<span class="badge b-ok">達標</span>＝該層已有一場<b>可引用</b>''（引用閘門 <code>QUOTABLE</code>）的讀數達到目標；未達標<b>不變色</b>，理由寫在標題的 <code>title</code> 裡。''<br><span style="padding:2px 10px;border-left:6px solid #059669;background:#ecfdf5;border-radius:6px">綠底＝達標</span>　<span style="padding:2px 10px;border-left:6px solid #059669;background:#ecfdf5;border-radius:6px">整列綠底＝已了結（結案／判死，資料旗標 <code>settled</code>）</span>　<span class="runnable">每格開頭</span>＝現在能不能跑＋前置是啥（<code>runnable</code>）</div>'
      % " ／ ".join("①②③④"[i] + " " + x for i, x in enumerate(board.get("stage_legend") or [])))
    A("<h2>已認證（符合結案規則的實測）</h2>")
    cn = board.get("certified_note") or {}
    if cn:
        A('<div class="warning"><b>%s</b>%s</div>'
          % (cn.get("title", ""), "".join("<br>%s" % x for x in (cn.get("lines") or []))))
    A("<table><thead><tr><th>id</th><th>項目</th><th>實測</th><th>profile</th><th>入口</th><th>達標</th>"
      "<th>引用閘門</th><th>出處</th></tr></thead><tbody>")
    for c in board.get("certified") or []:
        q = c.get("quote") or {}
        if q.get("verdict"):
            gate = '<span class="badge %s">%s</span>' % (
                "b-ok" if q["verdict"] == "QUOTABLE" else "b-warn", q["verdict"])
            if q.get("why"):
                gate += '<br><span class="legend">%s</span>' % q["why"]
        elif c.get("quote_ungateable"):
            gate = '<span class="badge b-info">不可判</span><br><span class="legend">%s</span>' % c["quote_ungateable"]
        else:
            gate = "—"
        A("<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td><code>%s</code></td><td><code>%s</code></td>"
          "<td>%s</td><td>%s</td><td><code>%s</code></td></tr>"
          % (c["id"], c["item"], c["value"], c["profile"], c["entry"],
             "✅" if c.get("meets") else "—", gate, c.get("source", "")))
    # ⚠ 這一行的縮排是**表格完不完整的開關**：`</tbody></table>` 若留在上面那個 `for c` 的迴圈裡，
    #   每一列都會各自關一次表格 ⇒ 瀏覽器把 C2／C3 擠出表格外（2026-09-30 的 7b374d03b 就是這樣壞的：
    #   縮排 4 → 8 把關閉標籤吃進了迴圈）。它必須與 `A("<table>…")` 同層。
    A("</tbody></table>")
    # ── 待升級（D15）：operator 2026-09-30「搬成功要放到已認證」──────────────────────
    # 畫這一區的理由：一個「待跑」的承諾如果不寫在**唯一來源**裡，就會變成人腦記著；
    # 而條件寫死之後，`validate` 會在它成立的那一輪把 `--check` 打紅 ⇒ 升級不會被忘記。
    pend = board.get("pending_promotion") or []
    if pend:
        A('<h3>待升級（D15）—— 條件<b>跑前</b>寫死；一旦當場成立，這一頁就會讓 '
          '<code>--check</code> 變紅直到它升上去</h3>')
        A('<div class="warning"><b>operator 2026-09-30：「搬成功要放到已認證」。</b>'
          '所以這一格<b>先寫死怎麼算搬成功</b>（不是跑完再挑一個好看的），而且判準當場成立而它還躺在'
          '這裡沒升進上表時，<code>scripts/check/decode_board_build.py --check</code> 就紅。'
          '升級的對象是一個<b>分布</b>，不是一個單點——同一支臂同一格連三場是 8.933／8.486／10.923，'
          '而前兩場各有一顆 rep 塌到 3.8–3.9（§73.5）。'
          '<br><b>✅ 2026-10-01 起不必等人抄</b>（operator：「D15 這要解決」）：'
          '<code>decode_board_build.py --promote [ID]</code> 會把成立的那一條機械合成一列 '
          '（值＝合格場次的<b>中位數</b>＋區間、<code>quote</code>＝中位那一場、棘輪＝分布豁免）'
          '插進「已認證」並移除 pending；寫完重建＋<code>--check</code>，沒過 ⇒ <b>整檔回滾</b>。'
          '<code>board_pipeline.sh</code>（重建模式）會先跑它 ⇒ 下一次重建就自己升。</div>')
        A("<table><thead><tr><th>id</th><th>項目</th><th>怎麼才算搬成功（跑前寫死）</th>"
          "<th>現在</th><th>候選臂</th><th>現狀</th></tr></thead><tbody>")
        for p in pend:
            j = p.get("judge") or {}
            n_l, n_q, how = pending_cluster(ROOT, j)
            mn_l, mn_q = j.get("min_launches"), j.get("min_quotable")
            if n_l is None:
                now = '<span class="badge b-warn">判不了</span><br><span class="legend">%s</span>' % how
            else:
                done = (mn_l is not None and mn_q is not None and n_l >= mn_l and n_q >= mn_q)
                now = ('<span class="badge %s">%d／%d 場可引用</span>'
                       '<br><span class="legend">%s</span>'
                       % ("b-ok" if done else "b-info", n_q, n_l, how))
            A("<tr><td><b>%s</b></td><td>%s</td><td>%s<br><span class=\"legend\">%s</span></td>"
              "<td>%s</td><td><code>%s</code></td><td>%s</td></tr>"
              % (p.get("id"), p.get("item"), p.get("accept"), p.get("why", ""),
                 now, "、".join(p.get("arms") or []), p.get("state", "")))
        A("</tbody></table>")
    # [CGC 2026-09-30 版面收斂] 本頁＝**三段**：已認證／L20／L25。operator 的指令：
    # 「三個『結案』到現在只成立三格；以下（冷啟 rep 的產物，不是機器變快）以上的內容請整理在
    #   已認證/L20/L25 上面，其餘刪除。」
    # ⇒ 兩句必須留在**已認證**段裡（它們正是「什麼才叫認證」的判準），而不是散在後面的敘述段。
    #   資料來源＝`certified_note`（YAML）；其餘段落的 `order`／`calls`／`stale` 已從資料裡移除。
    for layer in board.get("layers") or []:
        met, why = layer_goal(layer)
        head = layer.get("title", layer.get("id", ""))
        if met:
            # 只有「可引用的讀數達到目標」才變綠（見 layer_goal；D8 會複判）
            A('<h2 class="layer-met" title="%s">%s <span class="badge b-ok">達標</span></h2>'
              % (_attr(why), head))
        else:
            # 未達標／未宣告 ⇒ 完全不動（「若無則沒有變化」）；理由只放在 title，可稽核但不喧賓奪主
            A('<h2 title="%s">%s</h2>' % (_attr(why), head))
        if layer.get("note"):
            A('<p class="legend">%s</p>' % layer["note"])
        A("<table><thead><tr><th>id</th><th>子目標</th><th>節點</th><th>預期 ms/step</th><th>預期 t/s</th>"
          "<th>現況（已量）</th><th>action</th><th>驗收／否證</th><th>立項</th><th>結案</th><th>進展</th></tr></thead><tbody>")
        for t in layer.get("targets") or []:
            cell = "／".join("<code>%s</code>" % n for n in t.get("nodes") or [])
            ch = t.get("charter")
            chs = [ch] if isinstance(ch, str) else (ch or [])
            chcell = "／".join('<a href="%s">%s</a>' % (rel_from(here, os.path.join(ROOT, c)),
                                                        os.path.basename(c).replace(".yaml", ""))
                               for c in chs) or "—"
            ev = t.get("evidence") or {}
            st = str(t.get("state") or "")
            # [09-30 operator] 已了結的格（結案／判死）整列綠底；進行中不變色。
            # `settled` 是**資料裡的旗標**（YAML），不是從 state 字串猜的——判死格（L20-3）
            # 的 state 仍以「未結案」開頭（前置否證），但它在格線上已經死了，
            # 不該再用紅色暗示「有待辦」。
            settled = bool(t.get("settled"))
            cls = "b-ok" if (st.startswith("結案") or settled) else "b-warn"
            rowcls = ' class="row-done"' if settled else ""
            # 每格「現在能不能跑＋前置是啥」放在現況欄最前面（operator 09-30）
            nowcell = '<span class="runnable">%s</span><br>%s' % (t["runnable"], t["now"])
            cert = ""
            if ev.get("cert_by"):
                cert = '<br><span class="legend">排除依據：%s</span>' % ev.get("source", "")
            else:
                cert = ('<br><span class="legend"><code>%s</code>／<code>%s</code>／%s</span>'
                        % (ev.get("profile", "—"), ev.get("entry", "—"), ev.get("value", "—")))
            if ev.get("quote"):
                qv = ev["quote"].get("verdict", "—")
                cert += ('<br><span class="legend">引用閘門：<b>%s</b></span>' % qv)
            A("<tr%s><td><b>%s</b></td><td><b>%s</b><br><a class=\"legend\" href=\"%s\">逐條頁 ›</a></td><td>%s</td>"
              "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><b>驗收</b>：%s<br><b>否證</b>：%s</td>"
              "<td>%s</td><td><span class=\"badge %s\">%s</span>%s</td>"
              "<td><span class=\"steps\">%s</span></td></tr>"
              % (rowcls, t["id"], t["title"], rel_from(here, os.path.join(PAGES_DIR, t["id"] + ".html")),
                 cell, t["expect_ms"], t["expect_tps"], nowcell, t["action"],
                 t["accept"], t["falsify"], chcell, cls, st, cert,
                 "".join('<i class="%s"></i>' % ("on" if i < t["stage"] else "") for i in range(4))))
        A("</tbody></table>")
    # ── 以下是 09-30 之前會渲染、現在**整段移除**的區塊（operator：其餘刪除）─────────
    #   ① `calls`（要點）：其中兩句已整理進 已認證 段的 `certified_note`；
    #   ② `order`（建議施工順序）：活項都已寫在各格的 `action` 裡；
    #   ③ `notes` 的五個分類段（C 軸上界／已仲裁帳目／節點文字就地更新／對位表／其餘未結案節點）
    #      —— 這些是**導覽視圖**，資料仍在 `mindmap.json` 與各節點頁。
    #   ⚠ `notes` **刻意留在 YAML 裡**：它同時是 D2 的依據（未結案節點必須有歸屬），
    #     刪它等於刪掉一條會紅的閘門、把「全數有歸屬」變成沒人驗的聲明。
    #     所以本檔只停渲染，不動那份分類表；`stale`（節點文字更新紀錄）純敘述 ⇒ 已從 YAML 移除。
    A("<footer>本檔由 <code>scripts/check/decode_board_build.py</code> 從 <code>%s</code> ＋ <code>docs/mindmap/mindmap.json</code> "
      "<b>生成</b>（勿手改）；資料截止 <b>%s</b>；本頁<b>只保留三段</b>（已認證／L20／L25）——"
      "未結案節點共 <b>%d</b> 個，歸屬仍在 <code>mindmap.json</code> 與各節點頁（<code>--check</code> 可驗）。<br>%s</footer>"
      % (os.path.relpath(DEFAULT_YAML, ROOT), board.get("data_asof", "—"), len(live_nodes(nodes)), board.get("footer", "")))
    A("</div></body></html>")
    return "\n".join(L) + "\n"


# ────────────────────────── 渲染：逐子目標頁 ──────────────────────────
def node_doc_links(entry, here=PAGES_DIR):
    """節點的產物／log 連結（evid ＋ docs 清單），全部相對本頁目錄。"""
    out = []
    if not entry:
        return out
    ev = entry.get("evid")
    if ev:
        for piece in str(ev).split("／"):
            piece = piece.strip()
            if piece:
                out.append((piece, None))
    for d in entry.get("docs") or []:
        name = str(d)
        cand = [name] if name.startswith("docs/") else ["docs/%s" % name, name]
        hit = next((c for c in cand if os.path.exists(os.path.join(ROOT, c))), None)
        out.append((name, rel_from(here, os.path.join(ROOT, hit)) if hit else None))
    return out


def render_page(board, t, nodes):
    L = []; A = L.append
    ev = t.get("evidence") or {}
    A("<!DOCTYPE html>")
    A("<!-- generated by scripts/check/decode_board_build.py — 請勿手改本檔 -->")
    A('<html lang="zh-Hant"><head><meta charset="UTF-8">')
    A('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    A("<title>%s — %s</title><style>%s</style></head><body>" % (t["id"], t["title"], PAGE_CSS))
    A(nav(PAGES_DIR, os.path.join(ROOT, board.get("out", "")), len(collect_targets(board))))
    A('<div class="container">')
    A("<h1>%s ｜ %s</h1>" % (t["id"], t["title"]))
    st = str(t.get("state") or "")
    A('<p class="subtitle">結案狀態：<span class="badge %s">%s</span> ｜ 依 <b>%s</b> 的結案規則（prod-new ＋ harness bench ＋ 達標）</p>'
      % ("b-ok" if (st.startswith("結案") or t.get("settled")) else "b-warn", st, board.get("id", "")))
    A('<div class="card"><table class="kv">')
    A("<tr><th>預期（ms/step）</th><td>%s</td></tr>" % t["expect_ms"])
    A("<tr><th>預期（t/s）</th><td>%s</td></tr>" % t["expect_tps"])
    # 09-30 operator：能不能跑／前置是啥，放在現況之前，一眼可判
    A('<tr><th>現在能跑？</th><td><span class="runnable">%s</span></td></tr>' % t["runnable"])
    _pre = t.get("precondition") or {}
    if _pre:
        try:
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            import runnable_gate as _RG
            _lv, _lwhy, _ = _RG.scan(_pre, ROOT)
        except Exception as exc:  # noqa: BLE001
            _lv, _lwhy = "CANNOT_JUDGE", "閘門載不進：%s" % exc
        _cls = "success" if ((_pre.get("rerun") == "needed" and _lv == "ABSENT")
                            or (_pre.get("rerun") == "duplicate" and _lv == "EXISTS")) else "warning"
        A('<tr><th>前置（判決存不存在）</th><td><div class="%s"><b>%s</b>／<code>%s</code> ⇒ '
          '現判 <b>%s</b><br>%s<br><span class="legend">宣告理由：%s</span></div></td></tr>'
          % (_cls, _pre.get("rerun", "—"), _pre.get("kind", "—"), _lv, _lwhy, _pre.get("why", "")))
        _blk = _pre.get("block") or {}
        if _pre.get("rerun") == "blocked" and _blk.get("kind"):
            import runnable_gate as _RG2
            A('<tr><th>卡在哪一型（D13）</th><td><b>%s</b>（<code>%s</code>／誰＝<code>%s</code>）'
              '<br><b>下一個動作</b>：%s</td></tr>'
              % (_RG2.BLOCK_LABELS.get(_blk.get("kind"), _blk.get("kind")),
                 _blk.get("kind"), _blk.get("owner"), _blk.get("next", "")))
            _blv, _blw, _ = _RG2.scan_block_assert(_blk, ROOT)
            A('<tr><th>卡點斷言（D14）</th><td><b>%s</b><br>%s<br>'
              '<span class="legend">宣告的理由：%s</span></td></tr>'
              % (_blv, _blw, _blk.get("assert_why", "")))
    A("<tr><th>現況（已量）</th><td>%s</td></tr>" % t["now"])
    A("<tr><th>action</th><td>%s</td></tr>" % t["action"])
    A("<tr><th>驗收</th><td>%s</td></tr>" % t["accept"])
    A("<tr><th>否證</th><td>%s</td></tr>" % t["falsify"])
    A("</table></div>")
    A("<h3>結案判準的現況</h3><table><thead><tr><th>profile</th><th>入口</th><th>實測值</th><th>metric</th><th>達標</th></tr></thead><tbody>")
    A("<tr><td><code>%s</code></td><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td></tr>"
      % (ev.get("profile", "—"), ev.get("entry", "—"), ev.get("value", "—"), ev.get("metric", "—"),
         "✅" if ev.get("meets") else "❌"))
    A("</tbody></table>")
    if ev.get("cert_by"):
        A('<div class="success"><b>排除依據（%s）</b>：%s</div>' % (ev["cert_by"], ev.get("source", "")))
    q = ev.get("quote")
    if q:
        art = str(q.get("artifact", ""))
        artp = os.path.join(ROOT, art)
        art_html = ('<a href="%s"><code>%s</code></a>' % (rel_from(PAGES_DIR, artp), art)
                    if os.path.exists(artp) else '<code>%s</code>' % art)
        A('<div class="%s"><b>引用閘門：%s</b>（判準＝<code>scripts/check/quote_gate.py</code>，'
          '逐 rep 離散 ＋ 樣本數 ＋ 窗口 ＋ 參考帶）<br>產物 %s<br>%s</div>'
          % ("success" if q.get("verdict") == "QUOTABLE" else "warning",
             q.get("verdict", "—"), art_html, q.get("why", "")))
    elif ev.get("quote_ungateable"):
        A('<div class="warning"><b>引用閘門：無從判</b> —— %s</div>' % ev["quote_ungateable"])
    for cq in ([ev["counter_quote"]] if ev.get("counter_quote") else []) \
            + list(ev.get("counter_quotes") or []):
        okv = cq.get("verdict") in set(COUNTER_CLOSE_VERDICTS.values())
        A('<div class="%s"><b>端點判詞：%s</b>（檢查器 <code>%s</code>，build 當場重跑；'
          '這一格的端點不是時間讀數）<br>%s</div>'
          % ("success" if okv else "warning", cq.get("verdict", "—"),
             str(cq.get("tool", "")), cq.get("why", "")))
    o = t.get("options") or {}
    A("<h3>臂（options）</h3>")
    arms = o.get("arms") or []
    if arms:
        out = []
        for a in arms:
            if isinstance(a, dict) and a.get("arm"):
                tag = "（<b>基準臂</b>：成對量測的端點，報出來的 t/s 不是交付讀數）" \
                    if str(a.get("role") or "") == "reference" else ""
                out.append("<li><code>%s</code>%s%s</li>"
                           % (esc(a["arm"]), tag,
                              ("<br>%s" % esc(a.get("why", ""))) if a.get("why") else ""))
            else:
                out.append("<li><code>%s</code></li>" % esc(a))
        A("<ul>%s</ul>" % "".join(out))
    else:
        A('<div class="warning">本格<b>沒有可跑的臂</b> —— 宣告過的臂全部被 R5／R6 擋掉（見下），'
          '所以 20+／25+ 不可能靠它們量到。</div>')
    for rec in (o.get("arms_removed") or []):
        A('<div class="warning"><b>已刪除的臂</b> <code>%s</code>（%s）<br>%s</div>'
          % (esc(rec.get("arm", "")), "／".join(rec.get("rules") or []), esc(rec.get("why", ""))))
    if o.get("no_knob"):
        A("<p>%s</p>" % o["no_knob"])
    if o.get("note"):
        A("<p>%s</p>" % o["note"])
    ch = t.get("charter")
    chs = [ch] if isinstance(ch, str) else (ch or [])
    A("<h3>立項（charter）</h3>")
    if chs:
        A("<ul>")
        for c in chs:
            A('<li><a href="%s"><code>%s</code></a></li>' % (rel_from(PAGES_DIR, os.path.join(ROOT, c)), c))
        A("</ul>")
    else:
        A('<div class="warning">未立項（本輪未指派 arm）</div>')
    A("<h3>對應節點與其產物／log</h3>")
    for nid in t.get("nodes") or []:
        A('<div class="card"><b><code>%s</code></b> ｜ <a href="../briefs/%s.html">逐條白皮書（html）</a> '
          '／ <a href="../briefs/%s.md">md</a></div>' % (nid, nid, nid))
        A('<table><thead><tr><th>產物／log</th><th>連結</th></tr></thead><tbody>')
        for label, href in node_doc_links(nodes.get(nid)):
            if href:
                A('<tr><td><code>%s</code></td><td><a href="%s">開啟</a></td></tr>' % (label, href))
            else:
                A('<tr><td><code>%s</code></td><td><span class="legend">（文件引用，非檔案路徑）</span></td></tr>' % label)
        A("</tbody></table>")
    A("<footer>本頁由 <code>scripts/check/decode_board_build.py</code> 從 <code>scripts/check/decode_board_2026-09-29.yaml</code> "
      "＋ <code>docs/mindmap/mindmap.json</code> 生成（勿手改）。</footer>")
    A("</div></body></html>")
    return "\n".join(L) + "\n"


def render_page_md(board, t):
    ev = t.get("evidence") or {}
    ch = t.get("charter")
    chs = [ch] if isinstance(ch, str) else (ch or [])
    L = ["# %s ｜ %s" % (t["id"], re_plain(t["title"])), "",
         "- **結案狀態**：%s" % t.get("state", ""),
         "- **預期 ms/step**：%s" % re_plain(t["expect_ms"]),
         "- **預期 t/s**：%s" % re_plain(t["expect_tps"]),
         "", "## 現況（已量）", re_plain(t["runnable"]), "", re_plain(t["now"]), "",
         "## 前置（判決存不存在）", "",
         "- **rerun**：`%s`／`%s`（判準＝`scripts/check/runnable_gate.py`）——%s"
         % ((t.get("precondition") or {}).get("rerun", "—"),
            (t.get("precondition") or {}).get("kind", "—"),
            re_plain((t.get("precondition") or {}).get("why", ""))),
         ("- **卡在哪一型（D13）**：`%s`／`%s`；**下一個動作**：%s"
          % (((t.get("precondition") or {}).get("block") or {}).get("kind", "—"),
             ((t.get("precondition") or {}).get("block") or {}).get("owner", "—"),
             re_plain(((t.get("precondition") or {}).get("block") or {}).get("next", "—")))
          if ((t.get("precondition") or {}).get("block") or {}).get("kind") else ""),
         "## 動作與判準", "- **action**：%s" % re_plain(t["action"]),
         "- **驗收**：%s" % re_plain(t["accept"]), "- **否證**：%s" % re_plain(t["falsify"]), "",
         "## 結案判準現況", "",
         "| profile | 入口 | 實測值 | metric | 達標 |", "|---|---|---|---|---|",
         "| `%s` | `%s` | %s | %s | %s |" % (ev.get("profile", "—"), ev.get("entry", "—"),
                                             ev.get("value", "—"), ev.get("metric", "—"),
                                             "✅" if ev.get("meets") else "❌"), ""]
    if ev.get("cert_by"):
        L += ["> **排除依據（%s）**：%s" % (ev["cert_by"], re_plain(ev.get("source", ""))), ""]
    q = ev.get("quote")
    if q:
        L += ["### 引用閘門", "",
              "- **verdict**：`%s`（判準＝`scripts/check/quote_gate.py`）" % q.get("verdict", "—"),
              "- **產物**：`%s`" % q.get("artifact", ""),
              "- **為什麼**：%s" % re_plain(q.get("why", "")), ""]
    elif ev.get("quote_ungateable"):
        L += ["### 引用閘門", "", "> **無從判** —— %s" % re_plain(ev["quote_ungateable"]), ""]
    cq = ev.get("counter_quote")
    if cq:
        L += ["### 端點判詞", "",
              "- **verdict**：`%s`（檢查器 `%s`，build 當場重跑；這一格的端點不是時間讀數）"
              % (cq.get("verdict", "—"), cq.get("tool", "")),
              "- **為什麼**：%s" % re_plain(cq.get("why", "")), ""]
    o = t.get("options") or {}
    L += ["## 臂（options）", ""]
    arms = o.get("arms") or []
    def _arm_md(a):
        if isinstance(a, dict) and a.get("arm"):
            tag = "（**基準臂**：成對量測的端點，報出來的 t/s 不是交付讀數）" \
                if str(a.get("role") or "") == "reference" else ""
            return "- `%s`%s%s" % (a["arm"], tag,
                                   ("：%s" % re_plain(a["why"])) if a.get("why") else "")
        return "- `%s`" % a
    L += [_arm_md(a) for a in arms] or ["（本格沒有可跑的臂 —— 宣告過的臂全被 R5／R6 擋掉）"]
    for rec in (o.get("arms_removed") or []):
        L.append("- **已刪除** `%s`（%s）：%s"
                 % (rec.get("arm", ""), "／".join(rec.get("rules") or []), re_plain(rec.get("why", ""))))
    if o.get("no_knob"):
        L += ["", "> %s" % re_plain(o["no_knob"])]
    if o.get("note"):
        L += ["", re_plain(o["note"])]
    L += [""]
    L += ["## 立項（charter）", ""]
    L += ["- `%s`" % c for c in chs] or ["（未立項）"]
    L += ["", "## 對應節點", ""]
    for nid in t.get("nodes") or []:
        L.append("- `%s`（逐條白皮書：`docs/mindmap/briefs/%s.html`）" % (nid, nid))
    L += ["", "_本頁由 scripts/check/decode_board_build.py 生成，勿手改。_", ""]
    return "\n".join(L)


def _attr(text):
    """HTML 屬性值（title 用）：只要跳出雙引號與角括號就夠，不引入新依賴。"""
    return (str(text).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def re_plain(s):
    import re as _re
    s = _re.sub(r"<br\s*/?>", " ", str(s))
    s = _re.sub(r"<[^>]+>", "", s)
    return s


def render_index(board, targets):
    L = []; A = L.append
    A("<!DOCTYPE html>")
    A("<!-- generated by scripts/check/decode_board_build.py — 請勿手改本檔 -->")
    A('<html lang="zh-Hant"><head><meta charset="UTF-8">')
    A('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    A("<title>逐子目標（L20／L25）</title><style>%s</style></head><body>" % CSS)
    A(nav(PAGES_DIR, os.path.join(ROOT, board.get("out", "")), len(targets)))
    A('<div class="container"><h1>逐子目標頁：L20（8 格）／L25（6 格）</h1>')
    A('<p class="subtitle">每個子目標一頁：預期 ms/step 與 t/s、現況、驗收／否證、立項卡、結案判準（profile／入口／實測）。'
      '看板：<a href="%s">%s</a>。全部由 YAML ＋ mindmap.json 生成。</p>'
      % (rel_from(PAGES_DIR, os.path.join(ROOT, board.get("out", ""))), os.path.basename(board.get("out", ""))))
    A("<table><thead><tr><th>id</th><th>子目標</th><th>預期 ms/step</th><th>預期 t/s</th><th>立項</th><th>結案</th></tr></thead><tbody>")
    for t in targets:
        ch = t.get("charter")
        chs = [ch] if isinstance(ch, str) else (ch or [])
        st = str(t.get("state") or "")
        A("<tr><td><b><a href=\"%s.html\">%s</a></b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
          "<td><span class=\"badge %s\">%s</span></td></tr>"
          % (t["id"], t["id"], t["title"], t["expect_ms"], t["expect_tps"],
             "／".join("<code>%s</code>" % os.path.basename(c) for c in chs) or "—",
             "b-ok" if st.startswith("結案") else "b-warn", st))
    A("</tbody></table>")
    A('<div class="warning"><b>結案規則</b>：%s</div>' % (board.get("closure_rule") or {}).get("text", ""))
    A("<footer>生成器：<code>scripts/check/decode_board_build.py</code>；<code>--check</code> 會驗這 15 頁是否落後。</footer>")
    A("</div></body></html>")
    return "\n".join(L) + "\n"


def render_index_md(board, targets):
    L = ["# 逐子目標頁（L20／L25）", "",
         "| id | 子目標 | 預期 ms/step | 預期 t/s | 結案 |", "|---|---|---|---|---|"]
    for t in targets:
        L.append("| [%s](%s.html) | %s | %s | %s | %s |" % (t["id"], t["id"], re_plain(t["title"]),
                                                           re_plain(t["expect_ms"]), re_plain(t["expect_tps"]), t.get("state", "")))
    L += ["", "> **結案規則**：%s" % re_plain((board.get("closure_rule") or {}).get("text", "")), ""]
    return "\n".join(L)


# ────────────────────────── 主流程 ──────────────────────────
def collect_targets(board):
    return [dict(t, _layer=l.get("id")) for l in (board.get("layers") or []) for t in (l.get("targets") or [])]


def artifacts(board, nodes):
    board_out = os.path.join(ROOT, board.get("out", ""))
    here = os.path.dirname(board_out)
    out = {board_out: render_board(board, nodes, here, board_out)}
    out[os.path.join(PAGES_DIR, "index.html")] = render_index(board, collect_targets(board))
    out[os.path.join(PAGES_DIR, "index.md")] = render_index_md(board, collect_targets(board))
    for t in collect_targets(board):
        out[os.path.join(PAGES_DIR, "%s.html" % t["id"])] = render_page(board, t, nodes)
        out[os.path.join(PAGES_DIR, "%s.md" % t["id"])] = render_page_md(board, t)
    return out


def build(yaml_path=DEFAULT_YAML, root=ROOT):
    board = load_yaml(yaml_path)
    nodes = index_nodes(load_mindmap())
    errs, by_id = validate(board, nodes, root)
    return board, nodes, errs, artifacts(board, nodes)


def html_structure_problems(text):
    """回產物的**結構**問題（空＝每一列的表格歸屬都成立）。

    為什麼需要這條：`7b374d03b` 把 `</tbody></table>` 多縮排兩格、吃進了「逐列」的迴圈
    ⇒ **每一列各自關一次表格**，瀏覽器於是把 C2／C3 擠出表格外。但頁面照樣生成、
    `--check` 全綠、D4 也說「產物不落後」 —— 這類破壞**只有人眼看得到**，所以要有閘門看它。

    只判「表格歸屬」（`<tr>/<td>/<th>` 在不在 `<table>` 裡、`<table>` 有沒有關完）；
    不比對內容、不碰樣式 —— 這是結構閘門，不是渲染器。
    """
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.tables = 0
            self.problems = []

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.depth += 1
                self.tables += 1
            elif tag in ("tr", "td", "th") and self.depth == 0:
                self.problems.append("<%s> 出現在 <table> 之外 ⇒ 這一列會被擠出表格" % tag)

        def handle_endtag(self, tag):
            if tag == "table":
                self.depth -= 1
                if self.depth < 0:
                    self.problems.append("多餘的 </table>（表格已經關過了）")
                    self.depth = 0

    p = _P()
    p.feed(text)
    if p.depth:
        p.problems.append("有 %d 個 <table> 沒有關閉" % p.depth)
    seen, out = set(), []
    for x in p.problems:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def cmd_check(yaml_path, quiet=False):
    board, nodes, errs, arts = build(yaml_path)
    for path, text in arts.items():
        rel = os.path.relpath(path, ROOT)
        if not os.path.exists(path):
            errs.append("D4 產物不存在：%s" % rel)
        else:
            with open(path, encoding="utf-8") as fh:
                if fh.read() != text:
                    errs.append("D4 產物落後於 YAML（跑一次 build 即可）：%s" % rel)
        for prob in html_structure_problems(text):
            errs.append("D9 產物結構壞了（%s）：%s" % (prob, rel))
    if not quiet:
        tg = collect_targets(board)
        closed = [t["id"] for t in tg if str(t.get("state", "")).startswith("結案")]
        print("[decode-board] 子目標 %d 個；結案 %d 個（%s）" % (len(tg), len(closed), "、".join(closed) or "—"))
        print("[decode-board] 產物 %d 個（看板 1 ＋ 逐子目標頁 %d）" % (len(arts), len(arts) - 1))
        for e in errs:
            print("  ✗ %s" % e)
        print("VERDICT: %s（%d 個問題）" % ("PASS" if not errs else "FAIL", len(errs)))
    return 0 if not errs else 1


def cmd_build(yaml_path=DEFAULT_YAML):
    board, nodes, errs, arts = build(yaml_path)
    if errs:
        for e in errs:
            print("  ✗ %s" % e)
        print("VERDICT: FAIL — 有漂移，未寫檔。")
        return 1
    os.makedirs(PAGES_DIR, exist_ok=True)
    for path, text in arts.items():
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    tg = collect_targets(board)
    closed = [t["id"] for t in tg if str(t.get("state", "")).startswith("結案")]
    print("[decode-board] 寫入 %d 個產物：%s ＋ docs/mindmap/subgoals/（%d 頁 html＋md）"
          % (len(arts), board["out"], len(tg)))
    print("[decode-board] 子目標 %d；結案 %d（%s）；認證 %d 格"
          % (len(tg), len(closed), "、".join(closed) or "—", len(board.get("certified") or [])))
    _live, _rm, _bare = arm_census(board)
    if _rm:
        _q, _qc = blocked_queue(board)
        if _q:
            print("[decode-board] 卡點斷言：可掃描 %d／%d（HELD %d）；樹上掃不到 %d"
                  % (sum(1 for r in _q if r[4] in ("HELD", "GONE", "CANNOT_JUDGE")), len(_q),
                     sum(1 for r in _q if r[4] == "HELD"),
                     sum(1 for r in _q if r[4] == "UNSCANNED")))
            print("[decode-board] 決策佇列：%s（共 %d 格；排序＝決定→改動→機制→窗口）"
                  % ("／".join("%s %d" % (k, _qc[k]) for k in
                                ("decision", "code", "mechanism", "window") if _qc.get(k)), len(_q)))
        _pc = precondition_census(board)
        if _pc:
            print("[decode-board] 前置：可產新資訊 %d／重印 %d／card 卡住 %d；宣告與現判不符 %d"
                  % (_pc[0], _pc[1], _pc[2], _pc[3]))
        print("[decode-board] 臂：可跑 %d／已刪 %d（R5／R6）；無可跑臂的格 %d（%s）"
              % (_live, _rm, len(_bare), "、".join(_bare)))
    print("VERDICT: PASS")
    return 0


def cmd_selftest():
    import contextlib, copy, io, tempfile
    import yaml as _y
    ok = 0
    base = load_yaml(DEFAULT_YAML)
    nodes = index_nodes(load_mindmap())

    total = 0

    def case(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += 1 if cond else 0
        print("  %-46s -> %s%s" % (name, "PASS" if cond else "FAIL", "" if cond else " " + str(detail)))

    errs, _ = validate(base, nodes)
    case("clean base", not errs, errs)
    b = copy.deepcopy(base); b["layers"][0]["targets"][0]["nodes"] = ["no-such-node"]
    case("D1 unknown node", any(e.startswith("D1") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base); b["notes"] = [n for n in b["notes"] if n["node"] != "c-bandwidth"]
    case("D2 uncovered live node", any(e.startswith("D2") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base); b["layers"][0]["targets"][0]["charter"] = "scripts/check/charters/nope.yaml"
    case("D5 missing charter", any(e.startswith("D5") for e in validate(b, nodes)[0]))
    # [09-30 operator] runnable／settled 兩條：缺宣告要紅；已了結的格要整列綠底。
    b = copy.deepcopy(base); del b["layers"][0]["targets"][0]["runnable"]
    case("D3 缺 runnable（現在能跑宣告）", any(e.startswith("D3") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["state"] = "結案"
    b["layers"][0]["targets"][0]["evidence"] = {"profile": "prod25", "entry": "decode_sweep", "value": "8.43", "metric": "tps", "meets": True}
    case("D6 closure without prod-new/harness", any(e.startswith("D6") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["state"] = "結案"
    b["layers"][0]["targets"][0]["evidence"] = {"profile": "prod-new", "entry": "harness bench", "value": "20.5", "metric": "tps", "meets": True}
    case("D6 closure with prod-new + harness", not any(e.startswith("D6") for e in validate(b, nodes)[0]))
    # --- D7 引用閘門（fixture 自帶，不依賴 Backup/ 的實跑產物） ---
    gate_tmp = tempfile.mkdtemp()
    art_ok = os.path.join(gate_tmp, "gate_ok.json")
    art_dirty = os.path.join(gate_tmp, "gate_dirty.json")

    def _fixture(path, attrib):
        # ⚠ `profile` 必要（R7，2026-09-30）：真產物的臂一定帶它，fixture 少了就會被判 DIRTY
        #   ——那樣這一組案例會變成在測 R7，而不是在測 D7／D8（第一次上線時正是這樣紅的）。
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"rows": [{"n_prompt": 0, "n_gen": 64, "n_depth": 512,
                                 "avg_ts": 11.703, "stddev_ts": 0.348,
                                 "samples_ts": [11.65, 11.38, 12.07]}],
                       "attribution": {"verdict": attrib, "thermal_worst": "NOMINAL"},
                       "profile": "prod-new", "tag": "prod-new",
                       "cell": {"named_cell": "delivery"}}, fh)

    _fixture(art_ok, "none")     # 逐 rep 穩 ＋ 窗口乾淨 ⇒ QUOTABLE
    _fixture(art_dirty, "swap")  # 逐 rep 穩但換頁 ⇒ DIRTY
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["evidence"]["quote"] = {"artifact": art_dirty, "verdict": "QUOTABLE"}
    case("D7 declared verdict ≠ live", any(e.startswith("D7") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["state"] = "結案"
    b["layers"][0]["targets"][0]["evidence"] = {"profile": "prod-new", "entry": "harness bench",
                                                "value": "20.5", "metric": "tps", "meets": True}
    case("D7 closure without quote", any(e.startswith("D7") for e in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["state"] = "結案"
    b["layers"][0]["targets"][0]["evidence"] = {"profile": "prod-new", "entry": "harness bench",
                                                "value": "11.703", "metric": "tps", "meets": True,
                                                "quote": {"artifact": art_ok, "verdict": "QUOTABLE"}}
    case("D7 closure with QUOTABLE quote", not any(e.startswith("D7") or e.startswith("D6")
                                                   for e in validate(b, nodes)[0]))
    # --- D7b 儀器端點（metric=m_ratio，2026-09-30 §63；L25-5 是第一例） ---
    mtp_run = os.path.join(gate_tmp, "mtp_run.json")
    mtp_log = os.path.join(gate_tmp, "mtp_run.stderr.log")

    def _mtp_fixture(diff_ms, attrib="none"):
        with open(mtp_run, "w", encoding="utf-8") as fh:
            json.dump([{"rows": [{"n_prompt": 2048, "n_gen": 0, "avg_ts": 305.0},
                                 {"n_prompt": 0, "n_gen": 128, "avg_ts": 9.9}],
                        "cache": {"hit_rate_pct": 91.6}, "incomplete": False,
                        "cell": {"named_cell": "(default)", "depths": "512"},
                        "attribution": {"verdict": attrib}}], fh)
        def line(cd, t):
            return ("CGC-MTP-PERF type=draft-mtp calls_begin=%d calls_draft=%d calls_accept=%d "
                    "gen_tokens=%d acc_tokens=%d t_begin_ms=0.0 t_draft_ms=%.1f t_accept_ms=0.2 "
                    "acc_rate=1.0000 gen_tok_per_round=1.000 acc_tok_per_round=1.000 "
                    "emit_tok_per_round=2.000 ms_per_round=8.0\n" % (cd - 1, cd, cd, cd, cd, t))
        with open(mtp_log, "w", encoding="utf-8") as fh:
            fh.write(line(160, 1000.0 - diff_ms * 1.0) if False else
                     line(160, 1000.0) + line(192, 1000.0 + diff_ms))

    # --- D12／前置閘門的儀表：precondition_census（頁首那一行帳目的算術） ---
    def _census(rerun, globpat, kind="verdict_file"):
        # ⚠ 用**最小**的假看板：`base` 是真 YAML（16 格都有自己的前置），拿它當底會把 16 格一起算進來。
        bb = {"layers": [{"id": "L-selftest", "targets": [{
            "id": "T1", "precondition": {"rerun": rerun, "kind": kind,
                                         "spec": {"glob": globpat, "contains": "board"}}}]}]}
        return precondition_census(bb)

    _OK = "scripts/check/decode_board_2026-09-29.yaml"      # 一定存在
    _NO = "no_such_dir_for_selftest_*/**"                  # 一定不存在
    case("census：needed ＋ 判決不存在 ⇒ 算「可產新資訊」",
         _census("needed", _NO) == (1, 0, 0, 0, []))
    case("census：needed 但判決已存在 ⇒ 算漂移並點名",
         _census("needed", _OK) == (0, 0, 0, 1, ["T1"]))
    case("census：duplicate ＋ 判決存在 ⇒ 算「只是重印」",
         _census("duplicate", _OK) == (0, 1, 0, 0, []))
    case("census：duplicate 但掃不到 ⇒ 漂移",
         _census("duplicate", _NO) == (0, 0, 0, 1, ["T1"]))
    case("census：blocked ⇒ 歸「卡在人與機制」且不掃描",
         _census("blocked", _NO) == (0, 0, 1, 0, []))

    # --- D15／待升級：跑前寫死 ＋ 成立就紅（operator 2026-09-30「搬成功要放到已認證」）---
    def _pend(globpat=None, **over):
        spec = {"id": "PX", "item": "item", "why": "why", "accept": "accept",
                "state": "待跑", "arms": ["prod-new"],
                "judge": {"kind": "quotable_launch_cluster",
                          "glob": globpat or "pend_run_*/**/*.json", "cell": "delivery",
                          "min_launches": 3, "min_quotable": 2}}
        if "judge" in over:
            spec["judge"] = over.pop("judge")
        spec.update(over)
        bb = {"out": "docs/x.html", "layers": [], "certified": [],
              "pending_promotion": [spec]}
        # ⚠ root=gate_tmp：真 YAML 的其他格會因為路徑找不到而紅 ⇒ 只看這一格的 D15。
        return [e for e in validate(bb, {}, root=gate_tmp)[0] if e.startswith("D15")]

    case("D15 現況（還沒有這一輪的產物）⇒ 不紅", not _pend())
    case("D15 缺 accept（驗收沒寫死）⇒ 紅", bool(_pend(accept="")))
    case("D15 候選臂非認可口徑（R7）⇒ 紅", bool(_pend(arms=["prod25:!CGC_X=1"])))
    case("D15 judge.kind 不認得 ⇒ 紅", bool(_pend(judge={"kind": "nope", "glob": "x/*.json"})))
    case("D15 門檻不合理（min_quotable > min_launches）⇒ 紅",
         bool(_pend(judge={"kind": "quotable_launch_cluster", "glob": "x/*.json",
                           "min_launches": 3, "min_quotable": 5})))
    # 三場真產物形狀：可引用／可引用／不可引用（第三場 spread 1.27）。
    # ⚠ `stddev_ts` 必須與 `samples_ts` 反算值相符（R0 內部自洽）——不然閘門判 REFUSE，
    #   這一組案例就會變成「在測 R0」而不是在測 D15（第一次寫的時候正是這樣錯的）。
    def _sd(xs):
        # ⚠ **樣本**標準差（除以 n−1）：quote_gate 的 R0 用樣本 CV 對回報的 `stddev_ts`，
        #   用母體標準差（除以 n）會差 sqrt(n/(n−1))＝1.22× ⇒ 每個 fixture 都被判 REFUSE。
        _m = sum(xs) / len(xs)
        return (sum((x - _m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    for _i, (_avg, _sm) in enumerate(((10.9, [10.34, 11.34, 11.08]),
                                      (9.0, [8.83, 9.59, 9.70]),
                                      (7.0, [7.00, 8.90, 7.10]))):
        _dd = os.path.join(gate_tmp, "pend_run_%d" % _i)
        os.makedirs(_dd, exist_ok=True)
        with open(os.path.join(_dd, "run.json"), "w", encoding="utf-8") as fh:
            json.dump([{"rows": [{"n_prompt": 0, "n_gen": 64, "n_depth": 512,
                                  "avg_ts": _avg, "stddev_ts": _sd(_sm), "samples_ts": _sm}],
                        "attribution": {"verdict": "none", "thermal_worst": "NOMINAL"},
                        "profile": "prod-new", "tag": "prod-new",
                        "cell": {"named_cell": "delivery"}}], fh)
    case("D15 3 場 2 可引用 ≧ 門檻 ⇒ **紅（該升級了）**", bool(_pend()))
    case("D15 同樣 3 場但門檻要 4 場 ⇒ 不紅（場數還不夠）",
         not _pend(judge={"kind": "quotable_launch_cluster", "glob": "pend_run_*/**/*.json",
                          "cell": "delivery", "min_launches": 4, "min_quotable": 2}))
    case("D15 同樣 3 場但門檻要 3 場可引用 ⇒ 不紅（可引用數還沒到）",
         not _pend(judge={"kind": "quotable_launch_cluster", "glob": "pend_run_*/**/*.json",
                          "cell": "delivery", "min_launches": 3, "min_quotable": 3}))

    # --- D15 的機器路徑（--promote，2026-10-01 operator：「D15 這要解決」）---
    #   同一批 pend_run_* fixture：run0／run1 可引用（各有一顆冷 rep ⇒ §3.3b 拆欄），run2 不可。
    def _pspec(**over):
        s = {"id": "C99", "item": "分布（fixture）", "why": "why", "accept": "accept",
             "state": "待跑", "arms": ["prod-new"],
             "judge": {"kind": "quotable_launch_cluster", "glob": "pend_run_*/**/*.json",
                       "cell": "delivery", "min_launches": 3, "min_quotable": 2}}
        s.update(over)
        return s

    def _raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    _pn, _pf, _pe = cluster_fields(gate_tmp, _pspec()["judge"])
    _prow, _pwhy = promotion_row(_pspec(), _pf)
    _pq = sorted(f["value"] for f in _pf if f["verdict"] == "QUOTABLE")
    _pmed = _pq[(len(_pq) - 1) // 2]
    _pmidf = min((f for f in _pf if f["verdict"] == "QUOTABLE" and float(f["value"]) == _pmed),
                 key=lambda f: f["file"])
    case("--promote：3 場 2 可引用 ⇒ 合成一列（值＝中位、quote＝中位那一場）",
         _pn == 3 and _prow is not None and ("%.3f" % _pmed) in _prow[2]
         and _pmidf["file"] in _prow[-1] and "ratchet_exempt" in _prow[6], (_prow, _pwhy))
    case("--promote：合成列的 verdict 逐字＝QUOTABLE（D7 會當場複判）",
         _prow is not None and "verdict: QUOTABLE" in _prow[-1])
    case("--promote：門檻拉高（要 3 場可引用）⇒ 不合成",
         promotion_row(_pspec(judge={"kind": "quotable_launch_cluster",
                                     "glob": "pend_run_*/**/*.json", "cell": "delivery",
                                     "min_launches": 3, "min_quotable": 3}), _pf)[0] is None)
    _fy = ("out: docs/x.html\n"
           "certified:\n"
           "  - id: C1\n"
           '    item: "a"\n'
           "\n"
           "  - id: C4\n"
           '    item: "b"\n'
           "\n"
           "  - id: C6\n"
           '    item: "c"\n'
           "\n"
           "# ── 待升級（D15）：條件**跑前**寫死 ──\n"
           "pending_promotion:\n"
           "  - id: C5\n"
           '    item: "x"\n'
           "    judge: {kind: quotable_launch_cluster, glob: \"pend_run_*/**/*.json\", "
           "cell: delivery, min_launches: 3, min_quotable: 2}\n")
    _fy2 = cert_text_insert(_fy, "C5", ['  - id: C5', '    item: "new"'])
    case("--promote：插入位置＝數字序（C5 落在 C4 與 C6 之間）",
         _fy2.index("C1") < _fy2.index("C4") < _fy2.index("C5") < _fy2.index("C6"), _fy2)
    case("--promote：同 id 已在表上 ⇒ 拒絕（不重複插入）",
         _raises(lambda: cert_text_insert(_fy2, "C5", ['  - id: C5'])))
    _fy3 = pending_text_drop(_fy2, "C5")
    case("--promote：最後一條 pending ⇒ 收成 `pending_promotion: []`（註解留著）",
         "pending_promotion: []" in _fy3 and 'item: "x"' not in _fy3 and "待升級" in _fy3, _fy3)
    _fy4 = _fy + "  - id: C7\n    item: \"y\"\n"      # 第二條（沒有 judge ⇒ 判不了、不碰）
    _fy5 = pending_text_drop(_fy4, "C5")
    case("--promote：還有別條 ⇒ 只移除指名的那一條（C7 還在）",
         'item: "y"' in _fy5 and 'item: "x"' not in _fy5, _fy5)

    # 回滾：stub 閘門（不真的跑子行程）——① 看板已紅 ⇒ 不碰；② 寫完紅 ⇒ 逐字回滾。
    _ptmp = os.path.join(gate_tmp, "promote_board.yaml")
    with open(_ptmp, "w", encoding="utf-8") as _fh:
        _fh.write(_fy4)
    _porig = open(_ptmp, encoding="utf-8").read()

    def _gate_red(bp, build=False):
        return 1, "紅（fixture）"

    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _rc = cmd_promote(_ptmp, None, gate=_gate_red, root=gate_tmp)
    case("--promote：看板現在就是紅的 ⇒ 不碰（一位元組不動、rc=0）",
         _rc == 0 and open(_ptmp, encoding="utf-8").read() == _porig and "不推" in _buf.getvalue())

    _seq = {"n": 0}

    def _gate_flaky(bp, build=False):
        if not build:
            return 0, "綠（fixture）"
        _seq["n"] += 1
        return (1, "紅（fixture）") if _seq["n"] == 1 else (0, "綠（fixture）")

    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _rc = cmd_promote(_ptmp, None, gate=_gate_flaky, root=gate_tmp)
    case("--promote：寫完閘門紅 ⇒ **整檔回滾**（逐字＝升級前）＋ rc=1",
         _rc == 1 and open(_ptmp, encoding="utf-8").read() == _porig and "已回滾" in _buf.getvalue())
    case("--promote：指名一條還沒到門檻 ⇒ rc=2（不靜默）",
         cmd_promote(_ptmp, "C7", gate=_gate_flaky, root=gate_tmp) == 2)

    # --- D13／決策佇列：blocked_queue 的排序與計數 ---
    def _queue(pairs, assertion=None):
        bb = {"layers": [{"id": "L", "targets": [
            {"id": i, "precondition": {"rerun": "blocked", "kind": "none", "why": "f",
                                       "block": {"kind": k, "owner": "operator", "next": "x",
                                                 "assert": assertion or
                                                 {"kind": "none", "why": "fixture"}}}}
            for i, k in pairs]}]}
        return blocked_queue(bb)

    _q, _qc = _queue([("T-w", "window"), ("T-m", "mechanism"), ("T-c", "code"),
                      ("T-d", "decision")])
    case("queue：排序＝決定→改動→機制→窗口（不是 YAML 順序）",
         [r[0] for r in _q] == ["decision", "code", "mechanism", "window"], _q)
    case("queue：計數逐型分開", _qc == {"window": 1, "mechanism": 1, "code": 1, "decision": 1}, _qc)
    _q2, _qc2 = _queue([("T-2", "decision"), ("T-1", "decision")])
    case("queue：同型內按 id 排序", [r[2] for r in _q2] == ["T-1", "T-2"], _q2)
    _q3, _qc3 = _queue([("T-dupe", "mechanism"), ("T-dupe", "mechanism")])
    case("queue：不吞重複的格（守門失敗時看得見）", len(_q3) == 2, _q3)
    # --- D14：佇列每列要帶上「卡點斷言」的現判 ---
    _qs, _ = _queue([("T-ok", "code")],
                    assertion={"kind": "present_in_file", "glob": "scripts/check/runnable_gate.py",
                               "contains": "BLOCK_HELD"})
    case("queue：斷言命中 ⇒ 該列標 HELD（卡點還在）", _qs[0][4] == "HELD", _qs[0])
    _qg, _ = _queue([("T-gone", "code")],
                    assertion={"kind": "present_in_file", "glob": "scripts/check/runnable_gate.py",
                               "contains": "NO_SUCH_MARKER_ANYWHERE"})
    case("queue：斷言不再命中 ⇒ 該列標 GONE（宣告過期，頁首會警示）", _qg[0][4] == "GONE", _qg[0])
    _qu, _ = _queue([("T-un", "mechanism")])
    case("queue：assert(none) ⇒ 該列標 UNSCANNED（只靠散文，要被數出來）", _qu[0][4] == "UNSCANNED", _qu[0])

    def _d7b(cq, metric="m_ratio", state="結案", cert_by=None):
        # ⚠ root=gate_tmp 會讓**其他**格既有的產物路徑一起找不到 ⇒ 只看這一格的 D7b。
        bb = copy.deepcopy(base)
        tt = bb["layers"][0]["targets"][0]
        tid = tt["id"]
        tt["state"] = state
        if cert_by:
            tt["evidence"]["cert_by"] = cert_by
        tt["evidence"] = {"profile": "prod-new", "entry": "harness bench", "value": "fixture",
                          "metric": metric, "meets": True, "counter_quote": cq}
        return [e for e in validate(bb, nodes, root=gate_tmp)[0] if e.startswith("D7b " + tid)]

    CQ = {"tool": "mtp_round_split", "artifact": "mtp_run.json", "log": "mtp_run.stderr.log",
          "limit": 0.30, "cell": "(default)", "require_attrib": "none"}
    _mtp_fixture(0.16 * 86.13 * 32)          # m = 0.16
    case("D7b m_ratio + WITHIN（m=0.16 ≤ 0.30）結案",
         not _d7b(dict(CQ, verdict="WITHIN")))
    case("D7b m_ratio 宣告 FLIP（判詞錯）⇒ 紅", bool(_d7b(dict(CQ, verdict="FLIP"))))
    case("D7b m_ratio 門檻 0.10（m=0.16 > 0.10）⇒ REFUSE ⇒ 紅",
         bool(_d7b(dict(CQ, verdict="WITHIN", limit=0.10))))
    case("D7b m_ratio metric 沒登記（tps）⇒ 紅", bool(_d7b(dict(CQ, verdict="WITHIN"), metric="tps")))
    case("D7b 未知 tool ⇒ 判不了 ⇒ 紅", bool(_d7b(dict(CQ, verdict="WITHIN", tool="nope"))))
    case("D7b 產物不存在 ⇒ 判不了 ⇒ 紅",
         bool(_d7b(dict(CQ, verdict="WITHIN", artifact="missing.json"))))
    _mtp_fixture(0.16 * 86.13 * 32, attrib="swap")
    case("D7b 窗口 attribution=swap 但宣告要求 none ⇒ 紅",
         bool(_d7b(dict(CQ, verdict="WITHIN"))))
    case("D7b cell 不符 ⇒ 紅", bool(_d7b(dict(CQ, verdict="WITHIN", cell="delivery"))))
    # --- D7b 複數端點（counter_quotes，2026-10-01；L20-7 是第一例：同一格兩個端點） ---
    def _d7b2(cqs, metric="m_ratio", state="結案"):
        bb = copy.deepcopy(base)
        tt2 = bb["layers"][0]["targets"][0]
        tid2 = tt2["id"]
        tt2["state"] = state
        tt2["evidence"] = {"profile": "prod-new", "entry": "harness bench", "value": "fixture",
                           "metric": metric, "meets": True, "counter_quotes": cqs}
        return [e for e in validate(bb, nodes, root=gate_tmp)[0] if e.startswith("D7b " + tid2)]

    _mtp_fixture(0.16 * 86.13 * 32)
    case("D7b counter_quotes：兩個端點都過 ⇒ 不紅",
         not _d7b2([dict(CQ, verdict="WITHIN"), dict(CQ, verdict="WITHIN")]))
    case("D7b counter_quotes：其中一個判不了 ⇒ 紅（另一個過救不了它）",
         bool(_d7b2([dict(CQ, verdict="WITHIN"),
                     dict(CQ, verdict="WITHIN", artifact="missing.json")])))
    case("D7b counter_quotes：真結案時，metric 要的那個判詞必須在宣告裡",
         bool(_d7b2([dict(CQ, verdict="WITHIN")], metric="rho_price")))
    case("D7b counter_quotes：排除結案不套門檻映射 ⇒ 不紅",
         not _d7b2([dict(CQ, verdict="WITHIN")], metric="rho_price", state="結案（排除）"))
    # --- D17 棘輪 vs 認證表（2026-10-01；operator：「--check 要驗一致」） ---
    r17a = os.path.join(gate_tmp, "ratchet_a.json")
    r17b = os.path.join(gate_tmp, "ratchet_b.json")

    def _r17(path, avg):
        sm = [avg * 0.995, avg, avg * 1.005]      # 全 rep max/min 1.010 ⇒ QUOTABLE
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([{"rows": [{"n_prompt": 0, "n_gen": 64, "n_depth": 512, "avg_ts": avg,
                                  "stddev_ts": avg * 0.005, "samples_ts": sm}],
                        "attribution": {"verdict": "none", "thermal_worst": "NOMINAL"},
                        "profile": "prod-new", "tag": "prod-new",
                        "cell": {"named_cell": "(default)"}}], fh)

    _r17(r17a, 11.8)
    _r17(r17b, 12.0)

    def _r17row(cid, art, ts, above=None, exempt=None):
        e = {"id": cid, "item": "fixture", "value": "%s t/s" % ts, "profile": "prod-new",
             "entry": "harness bench", "meets": True, "source": art,
             "quote": {"artifact": art, "verdict": "QUOTABLE"}}
        if above is not None:
            e["ratchet"] = {"decode_ts": ts, "entered_above": above}
        if exempt is not None:
            e["ratchet_exempt"] = exempt
        return e

    def _r17board(rows_, a_ts, a_art):
        bb = copy.deepcopy(base)
        bb["certified"] = rows_
        bb["certify_anchor"] = {"decode_ts": a_ts, "cell": "(default)", "artifact": a_art,
                                "at": "selftest", "why": "fixture"}
        return [e for e in validate(bb, nodes, root=gate_tmp)[0] if e.startswith("D17 ")]

    _ok17 = [_r17row("C90", r17a, 11.8, 0.0), _r17row("C91", r17b, 12.0, 11.8)]
    case("D17 一致（上限＝表上最高、每列都高於入表時的上限）", not _r17board(_ok17, 12.0, r17b))
    case("D17 上限 ≠ 表上最高的 decode ⇒ 紅", bool(_r17board(_ok17, 11.8, r17a)))
    case("D17 上限 artifact 不是最高那一件 ⇒ 紅", bool(_r17board(_ok17, 12.0, r17a)))
    case("D17 某列沒嚴格高於入表時的上限 ⇒ 紅",
         bool(_r17board([_r17row("C90", r17a, 11.8, 11.9), _r17row("C91", r17b, 12.0, 11.8)],
                        12.0, r17b)))
    case("D17 入表時的上限不是表上的讀數（自創的錨）⇒ 紅",
         bool(_r17board([_r17row("C90", r17a, 11.8, 0.0), _r17row("C91", r17b, 12.0, 11.5)],
                        12.0, r17b)))
    case("D17 有 decode 讀數卻沒宣告 ratchet／ratchet_exempt ⇒ 紅",
         bool(_r17board([_r17row("C90", r17a, 11.8), _r17row("C91", r17b, 12.0, 11.8)], 12.0, r17b)))
    case("D17 baseline（ratchet_exempt 附 why）⇒ 不紅",
         not _r17board([_r17row("C90", r17a, 11.8,
                                exempt="基準：交付 cell 的第一列可引用讀數，入表條件是可引用、不是高過上限"),
                        _r17row("C91", r17b, 12.0, 0.0)], 12.0, r17b))
    case("D17 宣告的 decode_ts 不是產物自己的讀數（沒逐字抄）⇒ 紅",
         bool(_r17board([_r17row("C90", r17a, 11.8, 0.0), _r17row("C91", r17b, 12.05, 11.8)],
                        12.05, r17b)))
    # --- D7b 定價端點（metric=fill_term_ms，2026-09-30 §69；L20-4 是第一例） ---
    fill_ab = os.path.join(gate_tmp, "fill_ab.json")
    fill_la = os.path.join(gate_tmp, "fill_a.log")
    fill_lb = os.path.join(gate_tmp, "fill_b.log")

    def _fill_fixture(ts_a=(10.773,) * 3, ts_b=(13.733,) * 3, attrib=("none", "none"),
                      io_b=0, log_b=True, cell="delivery"):
        def _arm(tag, env, ts, av, iob):
            return {"tag": tag, "extra_env": env, "contract": {"cell": cell, "ok": True},
                    "attribution": {"verdict": av}, "cache": {"io_bytes": iob},
                    "rows": [{"n_gen": 64, "n_prompt": 0, "warm_skip": 64, "avg_ts": ts[0],
                              "samples_ts": list(ts)}]}
        data = [_arm("prod-new:CGC_EB_TIMER=1", {"CGC_EB_TIMER": "1"}, ts_a, attrib[0], 4864417792),
                _arm("prod-new:CGC_EB_TIMER=1;CGC_EB_NOFILL=1",
                     {"CGC_EB_TIMER": "1", "CGC_EB_NOFILL": "1"}, ts_b, attrib[1], io_b)]
        with open(fill_ab, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with open(fill_la, "w", encoding="utf-8") as fh:
            fh.write("CGC-EBTIMER: step_usec=7000 calls=40 miss=8 n_sum=320 seg=decode n_pf=0 n_dec=40\n" * 384)
        if log_b:
            with open(fill_lb, "w", encoding="utf-8") as fh:
                fh.write("CGC-EBTIMER: step_usec=60 calls=40 miss=8 n_sum=320 seg=decode n_pf=0 n_dec=40\n" * 384)

    CQF = {"tool": "fill_term_ab", "artifact": "fill_ab.json", "log_a": "fill_a.log",
           "log_b": "fill_b.log", "band": [7.0, 20.68], "verdict": "PRICED"}
    _fill_fixture()
    case("D7b fill_term_ms + PRICED（儀器 7.0／牆鐘 20.0）結案",
         not _d7b(dict(CQF), metric="fill_term_ms"))
    case("D7b fill_term_ms 宣告 WITHIN（判詞錯）⇒ 紅",
         bool(_d7b(dict(CQF, verdict="WITHIN"), metric="fill_term_ms")))
    case("D7b fill_term_ms metric 沒登記（tps）⇒ 紅", bool(_d7b(dict(CQF), metric="tps")))
    _fill_fixture(cell="(default)")
    case("D7b fill_term_ms 非交付 cell ⇒ REFUSE ⇒ 紅（訊息指名 REFUSE）",
         any("REFUSE" in e for e in _d7b(dict(CQF), metric="fill_term_ms")))
    _fill_fixture()
    case("D7b fill_term_ms 價格宣告偏離 >10% ⇒ 紅",
         bool(_d7b(dict(CQF, band=[3.0, 8.0]), metric="fill_term_ms")))
    case("D7b fill_term_ms band 不是兩元素 ⇒ 判不了 ⇒ 紅",
         bool(_d7b(dict(CQF, band=[7.0]), metric="fill_term_ms")))
    case("D7b fill_term_ms 沒宣告 log_b ⇒ 判不了 ⇒ 紅",
         bool(_d7b({k: v for k, v in CQF.items() if k != "log_b"}, metric="fill_term_ms")))
    case("D7b fill_term_ms log_b 不存在 ⇒ 判不了 ⇒ 紅",
         bool(_d7b(dict(CQF, log_b="nope.log"), metric="fill_term_ms")))
    _fill_fixture(attrib=("none", "swap"))
    case("D7b fill_term_ms B 臂髒窗口 ⇒ REFUSE ⇒ 紅",
         bool(_d7b(dict(CQF), metric="fill_term_ms")))
    # --- D7b 同格邊際端點（metric=margin_ms，2026-09-30 §70；L20-3 是第一例） ---
    sc_ab = os.path.join(gate_tmp, "sc_ab.json")
    sc_log = os.path.join(gate_tmp, "sc.log")
    sc_hist = os.path.join(gate_tmp, "sc_hist.log")
    sc_p1 = os.path.join(gate_tmp, "sc_probe1.json")
    sc_p2 = os.path.join(gate_tmp, "sc_probe2.json")

    def _sc_fixture(fam1=7.62, fam2=8.71, cell="delivery", attrib="none", cls="clean"):
        with open(sc_ab, "w", encoding="utf-8") as fh:
            json.dump([{"tag": "prod-new:CGC_EB_TIMER=1", "extra_env": {"CGC_EB_TIMER": "1"},
                        "contract": {"cell": cell}, "attribution": {"verdict": attrib},
                        "cache": {"io_bytes": 4864417792, "misses": 4300},
                        "rows": [{"n_gen": 64, "warm_skip": 64, "avg_ts": 10.773,
                                  "samples_ts": [1, 1, 1]}]}], fh)
        with open(sc_log, "w", encoding="utf-8") as fh:
            for rep, us in ((0, 9000), (1, 5200), (2, 5200)):
                for _ in range(128):
                    fh.write("CGC-EBTIMER: step_usec=%d calls=40 miss=7 n_sum=320 seg=decode n_pf=0 n_dec=40\n" % us)
        with open(sc_hist, "w", encoding="utf-8") as fh:
            for st_ in range(1, 387):
                mx = 6 if st_ in (100, 200) else 2
                fh.write("CGC-MISSMASK-HIST: step=%d nrec=39 max=%d il_max=39 unknown=0 per=1\n" % (st_, mx))
        for pp, fam in ((sc_p1, fam1), (sc_p2, fam2)):
            with open(pp, "w", encoding="utf-8") as fh:
                json.dump({"used": 8, "layers": 40, "window": {"class": cls},
                           "rows": {"gate_exps_t1": {"tokens": 1, "ms_per_step": fam / 2},
                                    "down_exps_t1": {"tokens": 1, "ms_per_step": fam / 2}}}, fh)

    CQS = {"tool": "samecell_margin", "artifact": "sc_ab.json", "log_a": "sc.log",
           "probes": ["sc_probe1.json", "sc_probe2.json"], "hist": "sc_hist.log",
           # 邊際宣告的次序是 [min, max]（見 samecell_margin.judge 的 k_star_margin_ms）
           "k_star": 6, "margin_ms": [-1.33, 3.29], "tol_ms": 0.5, "verdict": "NOT_SEPARATED"}
    _sc_fixture()
    # ⚠ 這一格是以**排除**結案（L20-3 的形狀）⇒ `margin_ms -> POSITIVE` 那條「達標才准」
    #   的規則不套用；分離為正的判詞仍然不准拿來當排除結案（下面那條紅）。
    case("D7b margin_ms + NOT_SEPARATED（k*=6）以排除結案",
         not _d7b(dict(CQS), metric="margin_ms", state="結案（排除）"))
    case("D7b margin_ms 宣告 POSITIVE（判詞錯）⇒ 紅",
         bool(_d7b(dict(CQS, verdict="POSITIVE"), metric="margin_ms")))
    case("D7b margin_ms metric 沒登記（tps）⇒ 紅", bool(_d7b(dict(CQS), metric="tps")))
    case("D7b margin_ms k* 宣告不符 ⇒ 紅", bool(_d7b(dict(CQS, k_star=3), metric="margin_ms")))
    case("D7b margin_ms k* 邊際宣告超容差 ⇒ 紅",
         bool(_d7b(dict(CQS, margin_ms=[0.0, 0.0]), metric="margin_ms")))
    case("D7b margin_ms 缺 probes ⇒ 判不了 ⇒ 紅",
         bool(_d7b({k: v for k, v in CQS.items() if k != "probes"}, metric="margin_ms")))
    _sc_fixture(cell="(default)")
    case("D7b margin_ms A 臂非交付 cell ⇒ REFUSE ⇒ 紅",
         any("REFUSE" in e for e in _d7b(dict(CQS), metric="margin_ms")))
    _sc_fixture(cls="busy-overridden")
    case("D7b margin_ms 探針窗口不乾淨 ⇒ REFUSE ⇒ 紅",
         any("REFUSE" in e for e in _d7b(dict(CQS), metric="margin_ms")))
    _sc_fixture()
    # --- D8／主節點綠燈（L20／L25） ---
    def _board_html(b):
        b = copy.deepcopy(b)
        b["goal_of"] = None if False else b.get("goal_of")
        path = os.path.join(gate_tmp, "lvl.yaml")
        b["out"] = "docs/__selftest_board.html"
        with open(path, "w", encoding="utf-8") as fh:
            _y.safe_dump(b, fh, allow_unicode=True, sort_keys=False)
        _, _, e, arts = build(path)
        html = next((t for p, t in arts.items() if p.endswith("__selftest_board.html")), "")
        return html, e

    b = copy.deepcopy(base)
    b["layers"][0]["goal"] = {"metric": "decode_tps", "target": 10.0, "crit": "fixture"}
    b["layers"][0]["evidence"] = {"value": 11.703, "artifact": art_ok, "quote_verdict": "QUOTABLE"}
    html, e = _board_html(b)
    case("L20 達標 ⇒ 綠（<h2 class=\"layer-met\">）", '<h2 class="layer-met"' in html and not e, e)
    b = copy.deepcopy(base)
    b["layers"][0]["targets"][0]["evidence"] = {
        "profile": "prod-new", "entry": "harness bench", "value": "fixture", "metric": "m_ratio",
        "meets": True,
        "counter_quotes": [{"tool": "t1", "verdict": "WITHIN", "why": "第一個端點"},
                           {"tool": "t2", "verdict": "NO_EFFECT", "why": "第二個端點"}]}
    _cqy = os.path.join(gate_tmp, "cq.yaml")
    b["out"] = "docs/__selftest_cq.html"
    with open(_cqy, "w", encoding="utf-8") as fh:
        _y.safe_dump(b, fh, allow_unicode=True, sort_keys=False)
    _, _, _cqe, _cqa = build(_cqy)
    _cqpages = " ".join(_cqa.values())
    case("render：counter_quotes 每個端點都印（不是只印第一個）",
         "第一個端點" in _cqpages and "第二個端點" in _cqpages, _cqe)
    b = copy.deepcopy(base)
    b["layers"][0]["goal"] = {"metric": "decode_tps", "target": 20.0, "crit": "fixture"}
    b["layers"][0]["evidence"] = {"value": 11.703, "artifact": art_ok, "quote_verdict": "QUOTABLE"}
    html, e = _board_html(b)
    case("未達標 ⇒ 不變色（無綠標題）", '<h2 class="layer-met"' not in html and not e, e)
    # D11 之後這個 case 必須「兩邊一致」：settled 要跟 state=結案 一起寫，
    # 而且要有能過 D6／D7 的 evidence（否則 validate 會因為別的原因紅）。
    b = copy.deepcopy(base)
    tt = b["layers"][0]["targets"][0]
    tt["settled"] = True
    tt["state"] = "結案（fixture）"
    tt["evidence"] = {"profile": "prod-new", "entry": "harness bench", "value": "11.703",
                      "metric": "tps", "meets": True,
                      "quote": {"artifact": art_ok, "verdict": "QUOTABLE"}}
    html, e = _board_html(b)
    case("settled ＋ 結案 ⇒ 整列綠底（row-done）且無錯", 'class="row-done"' in html and not e, e)
    # --- D11：state（判定）與 settled（顯示）必須一致 ---
    b = copy.deepcopy(base)
    tt = b["layers"][0]["targets"][0]
    tt["state"] = "結案（fixture）"
    tt["settled"] = False
    case("D11 結案卻沒 settled ⇒ 紅", any(x.startswith("D11") for x in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    tt = b["layers"][0]["targets"][0]
    tt["state"] = "未結案（fixture）"
    tt["settled"] = True
    case("D11 settled 但 state 說未結案 ⇒ 紅", any(x.startswith("D11") for x in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    tt = b["layers"][0]["targets"][0]
    tt["state"] = "結案（fixture）"
    tt["settled"] = True
    case("D11 結案 ＋ settled ⇒ 不紅", not any(x.startswith("D11") for x in validate(b, nodes)[0]))
    b = copy.deepcopy(base)
    b["layers"][0]["goal"] = {"metric": "decode_tps", "target": 10.0}
    b["layers"][0]["evidence"] = {"value": 11.703, "artifact": art_dirty,
                                  "quote_verdict": "DIRTY", "met": True}
    html, e = _board_html(b)
    case("宣告達標但讀數不可引用 ⇒ D8 擋", any(x.startswith("D8") for x in e), e)
    tmp = tempfile.mkdtemp()
    y = os.path.join(tmp, "b.yaml")
    b2 = copy.deepcopy(base); b2["out"] = "docs/__selftest_board.html"
    with open(y, "w", encoding="utf-8") as fh:
        _y.safe_dump(b2, fh, allow_unicode=True, sort_keys=False)
    probe = os.path.join(ROOT, "docs", "__selftest_board.html")
    had = os.path.exists(probe)
    old = open(probe, encoding="utf-8").read() if had else None
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("<!-- stale -->")
        case("D4 stale artifact", cmd_check(y, quiet=True) == 1)
    finally:
        if had:
            open(probe, "w", encoding="utf-8").write(old)
        elif os.path.exists(probe):
            os.remove(probe)
    # --- D10 臂必須量得到交付目標（R5／R6 的看板面）---
    def _d10(arms=None, removed="keep"):
        b = copy.deepcopy(base)
        o = dict(b["layers"][0]["targets"][0].get("options") or {})
        if arms is not None:
            o["arms"] = arms
        if removed != "keep":
            o["arms_removed"] = removed
        b["layers"][0]["targets"][0]["options"] = o
        return [e for e in validate(b, nodes)[0] if e.startswith("D10")]

    case("D10 現況看板不擋", not _d10())
    case("D10 單次提交臂不准留在 options", bool(_d10(arms=["prod-new:CGC_SEG_BATCH=1"])))
    case("D10 診斷量具臂不准留在 options", bool(_d10(arms=["prod-new:CGC_RHO_PROBE=1"])))
    case("D10 已刪除的臂可以留（附規則＋出處）",
         not _d10(arms=[], removed=[{"arm": "prod-new:CGC_SEG_BATCH=1", "rules": ["R6"],
                                     "flags": {"CGC_SEG_BATCH": "R6"}, "why": "fixture"}]))
    case("D10 刪除紀錄說謊（規則與登記表不符）",
         bool(_d10(arms=[], removed=[{"arm": "prod-new:CGC_SEG_BATCH=1", "rules": ["R6"],
                                      "flags": {"CGC_SEG_BATCH": "R5"}, "why": "fixture"}])))
    case("D10 同一支臂兩邊都列",
         bool(_d10(arms=["prod-new:CGC_SEG_BATCH=1"],
                   removed=[{"arm": "prod-new:CGC_SEG_BATCH=1", "rules": ["R6"],
                             "flags": {"CGC_SEG_BATCH": "R6"}, "why": "fixture"}])))
    case("D10 刪掉沒有 R5／R6 量具的臂",
         bool(_d10(arms=[], removed=[{"arm": "prod-new", "rules": ["R5"], "flags": {},
                                      "why": "fixture"}])))
    # [09-30] 基準臂（role=reference）：成對量測的端點，帶量具是本來就如此 ⇒ 附上 why＋source 才放行
    _ref = {"arm": "prod-new:CGC_EB_NOFILL=1", "role": "reference",
            "why": "成對量測的底／終點上界", "source": "docs/FILL_SPLIT_DELIVERY_2026-09-30.md"}
    case("D10 基準臂（附 why＋source）放行", not _d10(arms=[_ref]))
    case("D10 基準臂缺 why 就擋",
         bool(_d10(arms=[dict(_ref, why="")])))
    case("D10 基準臂缺 source 就擋",
         bool(_d10(arms=[{k: v for k, v in _ref.items() if k != "source"}])))
    case("D10 沒有 role 的 dict 臂一樣擋",
         bool(_d10(arms=[{k: v for k, v in _ref.items() if k != "role"}])))
    case("D10 role=option 一樣擋",
         bool(_d10(arms=[dict(_ref, role="option")])))
    case("D10 壞掉的 arms 項（dict 沒 arm）",
         bool(_d10(arms=[{"role": "reference", "why": "fixture"}])))

    # --- D16 臂類別 vs 卡點型別（空臂的格不准等窗口）---
    def _d16(kind="code", why="解除條件見 quote_gate 的 R6 登記表", settled=False, block=True):
        b = copy.deepcopy(base)
        tgt = b["layers"][0]["targets"][0]
        o = dict(tgt.get("options") or {})
        o["arms"] = []
        o["arms_removed"] = [{"arm": "prod-new:CGC_SEG_BATCH=1", "rules": ["R6"],
                              "flags": {"CGC_SEG_BATCH": "R6"}, "why": "fixture"}]
        tgt["options"] = o
        if settled:
            tgt["settled"] = True
            tgt["state"] = "結案（fixture）"
        pre = {"rerun": "blocked", "kind": "none", "why": "fixture"}
        if block:
            pre["block"] = {"kind": kind, "owner": "harness", "next": why,
                            "assert": {"kind": "none", "why": "fixture"}}
        tgt["precondition"] = pre
        return [e for e in validate(b, nodes)[0] if e.startswith("D16")]

    case("D16 空臂 ＋ block.kind=code 且指名 R6 ⇒ 放行", not _d16())
    case("D16 空臂卻等窗口（window 修不好臂的身分）⇒ 紅", bool(_d16(kind="window")))
    case("D16 空臂卻沒寫卡點 ⇒ 紅", bool(_d16(block=False)))
    case("D16 卡點內文沒指名 R5／R6／R7 ⇒ 紅", bool(_d16(why="再等等看")))
    case("D16 已了結（settled）的格不必再寫卡點 ⇒ 不紅", not _d16(block=False, settled=True))
    case("D16 block.kind 不合法（收 window／decision／code／mechanism 之外的）⇒ 紅",
         bool(_d16(kind="arm_class")))

    # --- D9 產物結構（fixture；不回讀檔案，直接餵字串）---
    good = '<table><thead><tr><th>a</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>'
    case("D9 完整表格", not html_structure_problems(good))
    # 這正是 7b374d03b 的破壞形態：關閉標籤被縮排吃進逐列迴圈 ⇒ 每列各關一次
    broken = ('<table><tbody><tr><td>C1</td></tr></tbody></table>'
              '<tr><td>C2</td></tr></tbody></table>')
    case("D9 逐列各關一次表格（7b374d03b 的形態）", bool(html_structure_problems(broken)))
    case("D9 未關閉的表格", bool(html_structure_problems('<table><tr><td>a</td></tr>')))
    case("D9 乾淨的字串不誤報", not html_structure_problems('<p>沒有表格</p><br><div>x</div>'))
    print("SELFTEST %s（%d/%d）" % ("PASS" if ok == total else "FAIL", ok, total))
    return 0 if ok == total else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 decode 子目標看板與逐子目標頁")
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--promote", nargs="?", const="", metavar="ID",
                    help="D15 的機器路徑：pending_promotion 的 judge 當場成立 ⇒ 機械合成 certified 列"
                         "（值＝合格場次中位＋區間、quote＝中位那一場）並移除該條目；寫完重建＋--check，"
                         "沒過就整檔回滾。省略 ID＝全部適用的都升；給 ID＝只升那一條")
    a = ap.parse_args(argv)
    if a.selftest:
        return cmd_selftest()
    if a.promote is not None:
        return cmd_promote(a.yaml, a.promote or None)
    if a.check:
        return cmd_check(a.yaml)
    return cmd_build(a.yaml)


if __name__ == "__main__":
    sys.exit(main())
