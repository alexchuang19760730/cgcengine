#!/usr/bin/env python3
"""mindmap 逐條目技術白皮書生成器（**html ＋ md 成對**）。

每個條目一對檔案（放在 `docs/mindmap/briefs/`）：
    <id>.html　樣式對齊 `docs/S1_ASYNC_GATHER_PIPELINE_2026-09-25.html`
    <id>.md　　樣式對齊 `docs/S1_ASYNC_GATHER_PIPELINE_2026-09-25.md`
並生一份總目錄：`briefs/index.html` ＋ `briefs/index.md`。

呈現骨架（對齊樣例）：
    標題 ／ 一句話 → 【三卡】目標·判準｜結果（大字）｜判定
    → 【四段卡】①目標 → ②判準 → ③結果 → ④判定
    → 【表 1】與其它條目的關係（同軸／同階段，自動對照）
    → 【表 2】依據 · 備註 · 軸性質 · 報告份數
    → 【結論框】關鍵結論（判定 ＋ 理由）
    → 對應報告（可點）＋ 導航（上一份／總目錄／下一份／另一種格式）

資料全部來自 `docs/mindmap/mindmap.json`（`goal`/`crit`/`res`/`evid`/`note`/`tier`/`sub`），
機械生成、不手寫內容。條目可另加兩個**可選**欄位（沒有就自動省略該表）：
    `diff`  : [[對象, 它的做法／結果, 本條目差別], ...]   → 渲染「與其它路線的關鍵差異」
    `risks` : [[風險, 驗證方式], ...]                    → 渲染「成敗點（風險 → 驗證）」

用法：
    python3 scripts/check/mindmap_brief_build.py             # 生成
    python3 scripts/check/mindmap_brief_build.py --selftest  # 自測
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mindmap_build as MB  # noqa: E402  (共用盤點／映射／校驗)

MM_DIR = ROOT / "docs/mindmap"
BRIEF_DIR = MM_DIR / "briefs"
DATA = MM_DIR / "mindmap.json"

STEPS = [("1", "目標", "goal"), ("2", "判準", "crit"),
         ("3", "結果", "res"), ("4", "判定", None)]


def load() -> tuple[dict, dict[str, str]]:
    data = MB.load_data()
    docs = MB.scope_docs(MB.branch_docs())
    mapping, _ = MB.map_docs(data["entries"], docs)
    return data, mapping


def tier_of(data: dict, tid: str) -> dict:
    """TIER 在 mindmap_build 裡只在 JS 側存在，Python 側從 data["tiers"] 建。"""
    return {t["id"]: t for t in data["tiers"]}.get(
        tid, {"id": tid, "label": tid, "color": "#888"})


def sub_of(data: dict, sid: str) -> dict:
    return next((s for s in data["subgoals"] if s["id"] == sid),
                {"id": sid, "label": sid, "color": "#888"})


def plain(s: str) -> str:
    """去掉 HTML 標籤，給純文字（MD）與表格預覽用。"""
    s = re.sub(r"<[^>]+>", "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def detail_href(detail: str, start: Path) -> tuple[str, bool]:
    """`detail_html` 記的是相對 `docs/mindmap/` 的路徑；換算到 start 目錄下可用的相對路徑。"""
    t = (MM_DIR / detail).resolve()
    if t.exists():
        return os.path.relpath(t, start=start), True
    return "", False


def rel_link(doc: str, start: Path) -> tuple[str, bool]:
    """doc 是 repo 相對路徑（docs/xxx.md）。回傳 (href 或 '', 檔案是否在工作區內)。"""
    target = ROOT / doc
    if target.exists():
        return os.path.relpath(target, start=start), True
    return "", False


def big_number(res: str) -> str:
    """從結果字串裡抓一個可放大的數字：t/s ＞ ×倍數 ＞ 百分比。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*t/s", res)
    if m:
        return m.group(1) + " t/s"
    m = re.search(r"[×x]\s*(\d+(?:\.\d+)?)", res)
    if m:
        return "×" + m.group(1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", res)
    if m:
        return m.group(1) + "%"
    return ""


# ───────────────────── HTML（inline style，對齊 S1_ASYNC_GATHER 樣例）────

def card(title: str, body: str, big: str = "", bg: str = "#f8fafc",
         bd: str = "#e2e8f0", fg: str = "#334155") -> str:
    big_html = (f'<div style="margin-top:8px;font-size:22px;font-weight:700;color:{fg};">'
                f'{big}</div>') if big else ""
    return (f'<div style="flex:1;min-width:180px;background:{bg};border:1px solid {bd};'
            f'border-radius:8px;padding:12px;">'
            f'<div style="font-size:12.5px;font-weight:600;margin-bottom:6px;color:{fg};">'
            f'{title}</div>'
            f'<div style="font-size:11.5px;line-height:1.7;">{body}</div>{big_html}</div>')


def arrow() -> str:
    return '<div style="display:flex;align-items:center;color:#94a3b8;font-size:18px;">→</div>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f'<td style="padding:6px 8px;font-weight:600;">{h}</td>' for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f'<td style="padding:6px 8px;border-bottom:1px solid #eee;">{c}</td>'
                      for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return ('<table style="width:100%;border-collapse:collapse;font-size:11.5px;'
            'margin-bottom:14px;">'
            f'<tr style="background:#f1f5f9;">{th}</tr>{"".join(trs)}</table>')


def section_title(t: str) -> str:
    return f'<div style="font-size:13px;font-weight:600;margin-bottom:6px;">{t}</div>'


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_brief_html(e: dict, data: dict, mapping: dict[str, str],
                      order: list[str], idx: int) -> str:
    tier = tier_of(data, e["tier"])
    sub = sub_of(data, e["sub"])
    stage_id = MB.STAGE_OF.get(e["tier"], "closed")
    st = next((s for s in data["stages"] if s["id"] == stage_id), None)
    ent = {x["id"]: x for x in data["entries"]}

    # 原始方案書（條目若有 detail_html）
    det_link = ""
    if e.get("detail_html"):
        dh, ok = detail_href(e["detail_html"], BRIEF_DIR)
        if ok:
            det_link = (f'　｜　<a href="{dh}" target="_blank" rel="noopener" '
                        f'style="color:#1d4ed8;">原始方案書 ↗</a>')

    # 三卡：目標·判準 ／ 結果 ／ 判定
    cards = (card("目標 · 判準（要達成什麼、怎麼算達成）",
                  f'{e.get("goal", "—")}<br><span style="color:#64748b;">判準：{esc(e["crit"])}</span>',
                  bg="#fef2f2", bd="#fecaca", fg="#991b1b")
             + card("結果（實測／推算，依據見下）", e.get("res", "—"),
                    big=big_number(e.get("res", "")),
                    bg="#eff6ff", bd="#93c5fd", fg="#1e40af")
             + card("判定（契約 §5 四級）",
                    f'{esc(tier["label"])}<br><span style="color:#64748b;">{esc(tier["def"])}</span>',
                    big=esc(e["tier"]),
                    bg="#f0fdf4", bd="#86efac", fg="#166534"))

    # 四段卡（①→②→③→④）
    steps_html = []
    for i, (num, label, key) in enumerate(STEPS):
        if key:
            body = e.get(key, "—")
        else:
            body = (f'<b style="color:{tier["color"]}">{esc(e["tier"])} · {esc(tier["label"])}</b>'
                    + (f'<br>{e["note"]}' if e.get("note") else ""))
        steps_html.append(
            f'<div style="flex:1;min-width:170px;background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:8px;padding:10px;">'
            f'<div style="font-size:12px;font-weight:600;margin-bottom:4px;">{num} {label}</div>'
            f'<div style="font-size:11px;line-height:1.65;color:#334155;">{body}</div></div>')
    flow = arrow().join(steps_html)

    # 表 1：與其它條目的關係（同軸／同階段，自動對照）
    peers = [x for x in data["entries"]
             if x["id"] != e["id"] and x["sub"] == e["sub"]
             and MB.STAGE_OF.get(x["tier"]) == stage_id][:6]
    if peers:
        rows = [[f'<a href="{p["id"]}.html">{esc(p["name"])}</a>',
                 f'<b style="color:{tier_of(data, p["tier"])["color"]}">{esc(p["tier"])}</b>',
                 esc(plain(p.get("res", ""))[:90])] for p in peers]
        rel_tbl = section_title(f"與其它條目的關係（同軸 {esc(sub['label'])}／同階段，自動對照）") \
            + table(["條目", "級", "結果（摘）"], rows)
    else:
        rel_tbl = section_title("與其它條目的關係") \
            + table(["說明"], [["本軸／本階段沒有其它條目（本條目是唯一一格）"]])

    # 表 2（可選）：與其它路線的關鍵差異
    diff_tbl = ""
    if e.get("diff"):
        # 這幾欄的內容是 mindmap.json 裡的作者文字（跟 goal 一樣帶 <b>/<code> 標記）⇒ 原樣輸出。
        # 先前這裡用 esc()、goal 卻原樣 ⇒ 同一頁一半的粗體變成可見的 `<b>` 字標。
        diff_tbl = section_title("與其它路線的關鍵差異") + table(
            ["對象", "它的做法／結果", "本條目差別"],
            [[d[0], d[1], d[2]] for d in e["diff"]])

    # 表 3（可選）：成敗點（風險 → 驗證）
    risk_tbl = ""
    if e.get("risks"):
        risk_tbl = section_title("成敗點（風險 → 驗證方式）") + table(
            ["#", "風險", "驗證"],
            [[f"<b>{i + 1}</b>", r[0], r[1]] for i, r in enumerate(e["risks"])])

    # 表 3b（可選）：生產設置（要進生產必須滿足什麼）—— 分級是 ③a 時，這張表就是「升級條件」
    prod_tbl = ""
    if e.get("prod_setup"):
        prod_tbl = section_title("生產設置（要進生產必須滿足什麼）") + table(
            ["項目", "驗收條件（可否證）", "現況"],
            [[r[0], r[1], r[2]] for r in e["prod_setup"]])

    # 表 3c（可選）：驗收測試（完整 test cases）—— 每格都要有「先寫死的通過條件」與「產物」
    test_tbl = ""
    if e.get("tests"):
        test_tbl = section_title("驗收測試（完整 test cases：E 系列）") + table(
            ["#", "測什麼", "通過條件（先寫死）", "成本／指令", "結果"],
            [[f'<b>{esc(t.get("id", ""))}</b>', t.get("what", ""), t.get("pass", ""),
              f'<code>{esc(t.get("cmd", ""))}</code>'
              + (f'<br>{t["cost"]}' if t.get("cost") else ""),
              t.get("result") or '<span style="color:#94a3b8;">未跑</span>']
             for t in e["tests"]])

    # 表 4：依據 · 備註 · 軸性質 · 報告
    doclist = sorted(d for d, eid in mapping.items() if eid == e["id"])
    meta_rows = [
        ["依據", f'<code>{esc(e.get("evid", "—"))}</code>'],
        ["備註", e.get("note") or "—"],
        ["軸性質", f'<span style="color:{sub["color"]}">●</span> {esc(sub.get("role_label") or sub.get("def", ""))}'],
        ["階段", f'{esc(st["label"]) if st else "—"}'
                 + (f'　·　{esc(st["desc"])}' if st else "")],
        ["對應報告", f'{len(doclist)} 份'],
    ]
    meta_tbl = section_title("依據 · 備註 · 軸性質") + table(["項目", "內容"], meta_rows)

    # 結論框
    conclusion = (f'<b>關鍵結論：</b>判定 <b style="color:{tier["color"]}">'
                  f'{esc(e["tier"])} · {esc(tier["label"])}</b>'
                  f'（{esc(tier["def"])}）'
                  + (f'　—　{e.get("note")}' if e.get("note") else "")
                  + f'　｜　本條目屬 <b>{esc(sub["label"])}</b>'
                  + (f'（{esc(sub.get("role_label", ""))}）' if sub.get("role_label") else "")
                  + "。")

    # 對應報告
    if doclist:
        items = []
        for d in doclist:
            href, ok = rel_link(d, BRIEF_DIR)
            name = Path(d).name
            items.append(f'<li><a href="{href}" target="_blank" rel="noopener">{esc(name)}</a></li>'
                         if ok else
                         f'<li>{esc(name)} <span style="color:#94a3b8;">'
                         f'（僅存在於分支，不在工作區）</span></li>')
        docs_html = (section_title(f"對應報告（{len(doclist)} 份）")
                     + f'<ol style="margin:0;padding-left:20px;font-size:12px;line-height:1.7;">'
                     f'{"".join(items)}</ol>')
    else:
        docs_html = section_title("對應報告") + \
            '<div style="font-size:12px;color:#94a3b8;">（無）</div>'

    # 導航：上一份／總目錄／下一份／另一種格式
    prev_id = order[idx - 1] if idx > 0 else None
    next_id = order[idx + 1] if idx + 1 < len(order) else None
    prev_html = (f'<a href="{prev_id}.html">← {esc(ent[prev_id]["name"])}</a>'
                 if prev_id else "<span></span>")
    next_html = (f'<a href="{next_id}.html">{esc(ent[next_id]["name"])} →</a>'
                 if next_id else "<span></span>")
    nav = (f'<div style="margin-top:16px;padding-top:10px;border-top:1px solid #e2e8f0;'
           f'display:flex;justify-content:space-between;gap:10px;font-size:12px;">'
           f'{prev_html}<span><a href="index.html">總目錄</a>　·　'
           f'<a href="{e["id"]}.md">MD 版</a></span>{next_html}</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{esc(e["name"])} — 技術白皮書（{esc(e["tier"])} {esc(tier["label"])}）</title>
</head>
<body style="margin:0;padding:0;">
<div style="width:100%;box-sizing:border-box;padding:18px;font-family:-apple-system,'PingFang TC',sans-serif;color:#1a1a1a;">
  <h2 style="margin:0 0 4px;font-size:19px;">{esc(e["name"])}</h2>
  <div style="font-size:12px;color:#666;margin-bottom:14px;line-height:1.6;">{esc(plain(e.get("goal", "")))}
　｜　主題 {esc(e.get("theme", "—"))}　｜　子目標 <span style="color:{sub["color"]}">●</span> {esc(sub["label"])}{det_link}</div>

  <div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;">{cards}</div>

  {section_title("四段：目標 → 判準 → 結果 → 判定")}
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;align-items:stretch;">{flow}</div>

  {rel_tbl}
  {diff_tbl}
  {risk_tbl}
  {prod_tbl}
  {test_tbl}
  {meta_tbl}
  {docs_html}

  <div style="padding:11px 13px;background:#fffbeb;border:1px solid #fcd34d;border-radius:7px;font-size:12px;color:#78350f;line-height:1.65;margin-top:12px;">
    {conclusion}
  </div>
  <div style="margin-top:8px;font-size:11px;color:#94a3b8;">本頁由
    <code>scripts/check/mindmap_brief_build.py</code> 從 <code>docs/mindmap/mindmap.json</code>
    機械生成；改內容請改 JSON 後重跑，勿直接編輯本頁。</div>
  {nav}
</div>
</body>
</html>
"""


# ───────────────────── MD（對齊 S1_ASYNC_GATHER_PIPELINE md 樣例）────

def render_brief_md(e: dict, data: dict, mapping: dict[str, str],
                    order: list[str], idx: int) -> str:
    tier = tier_of(data, e["tier"])
    sub = sub_of(data, e["sub"])
    stage_id = MB.STAGE_OF.get(e["tier"], "closed")
    st = next((s for s in data["stages"] if s["id"] == stage_id), None)
    ent = {x["id"]: x for x in data["entries"]}

    out = [f'# {plain(e["name"])} — 技術白皮書　·　{e["tier"]} {plain(tier["label"])}', "",
           f'> **一句話**：{plain(e.get("goal", "—"))}', "",
           f'- 主題：{plain(e.get("theme", "—"))}　·　子目標：**{plain(sub["label"])}**'
           + (f'（{plain(sub.get("role_label", ""))}）' if sub.get("role_label") else ""),
           f'- 階段：{plain(st["label"]) if st else "—"}'
           + (f'　·　{plain(st["desc"])}' if st else "")]
    if e.get("detail_html"):
        dh, ok = detail_href(e["detail_html"], BRIEF_DIR)
        if ok:
            out.append(f'- 原始方案書：[{Path(e["detail_html"]).name}]({dh})')
    out += ["", "---", ""]

    for num, label, key in STEPS:
        out += [f"## {num}. {label}", ""]
        if key:
            out += [plain(e.get(key, "—")), ""]
        else:
            out += [f'**{e["tier"]} · {plain(tier["label"])}** — {plain(tier["def"])}', ""]
            if e.get("note"):
                out += [f'> {plain(e["note"])}', ""]

    peers = [x for x in data["entries"]
             if x["id"] != e["id"] and x["sub"] == e["sub"]
             and MB.STAGE_OF.get(x["tier"]) == stage_id][:6]
    # 節號從 5 起**遞增**（不是硬寫）：可選區塊（diff／risks／prod_setup／tests）缺席時，
    # 舊版會留下跳號（5 → 8）或重號（兩個 ## 8）—— 一份自己編號對不上的文件，讀者就沒法引用它。
    sec = 5
    out += ["---", "", f"## {sec}. 與其它條目的關係（同軸／同階段，自動對照）", "",
            "| 條目 | 級 | 結果（摘） |", "|---|---|---|"]
    sec += 1
    if peers:
        for p in peers:
            out.append(f'| [{plain(p["name"])}]({p["id"]}.md) | {p["tier"]} '
                       f'| {plain(p.get("res", ""))[:90]} |')
    else:
        out.append("| （本軸／本階段沒有其它條目） | — | — |")
    out.append("")

    if e.get("diff"):
        out += [f"## {sec}. 與其它路線的關鍵差異", "", "| 對象 | 它的做法／結果 | 本條目差別 |", "|---|---|---|"]
        out += [f"| {plain(d[0])} | {plain(d[1])} | {plain(d[2])} |" for d in e["diff"]]
        out.append("")
        sec += 1

    if e.get("risks"):
        out += [f"## {sec}. 成敗點（風險 → 驗證）", "", "| # | 風險 | 驗證 |", "|---|---|---|"]
        out += [f"| {i + 1} | {plain(r[0])} | {plain(r[1])} |" for i, r in enumerate(e["risks"])]
        out.append("")
        sec += 1

    if e.get("prod_setup"):
        out += [f"## {sec}. 生產設置（要進生產必須滿足什麼）", "",
                "| 項目 | 驗收條件（可否證） | 現況 |", "|---|---|---|"]
        out += [f"| {plain(r[0])} | {plain(r[1])} | {plain(r[2])} |" for r in e["prod_setup"]]
        out.append("")
        sec += 1

    if e.get("tests"):
        out += [f"## {sec}. 驗收測試（完整 test cases：E 系列）", "",
                "| # | 測什麼 | 通過條件（先寫死） | 成本／指令 | 結果 |", "|---|---|---|---|---|"]
        out += [f"| **{plain(t.get('id', ''))}** | {plain(t.get('what', ''))} | "
                f"{plain(t.get('pass', ''))} | `{plain(t.get('cmd', ''))}`"
                + (f"（{plain(t['cost'])}）" if t.get("cost") else "")
                + f" | {plain(t.get('result') or '未跑')} |" for t in e["tests"]]
        out.append("")
        sec += 1

    doclist = sorted(d for d, eid in mapping.items() if eid == e["id"])
    out += [f"## {sec}. 依據 · 備註 · 對應報告", "",
            "| 項目 | 內容 |", "|---|---|",
            f'| 依據 | `{plain(e.get("evid", "—"))}` |',
            f'| 備註 | {plain(e.get("note", "—"))} |',
            f'| 軸性質 | {plain(sub.get("role_label") or sub.get("def", ""))} |',
            f'| 對應報告 | {len(doclist)} 份 |', ""]
    if doclist:
        for d in doclist:
            href, ok = rel_link(d, BRIEF_DIR)
            name = Path(d).name
            out.append(f"- [{name}]({href})" if ok
                       else f"- {name}（僅存在於分支，不在工作區）")
    else:
        out.append("- （無）")
    out.append("")

    prev_id = order[idx - 1] if idx > 0 else None
    next_id = order[idx + 1] if idx + 1 < len(order) else None
    nav = []
    if prev_id:
        nav.append(f'← [{plain(ent[prev_id]["name"])}]({prev_id}.md)')
    nav.append("[總目錄](index.md)")
    nav.append(f"[HTML 版]({e['id']}.html)")
    if next_id:
        nav.append(f'[{plain(ent[next_id]["name"])} →]({next_id}.md)')
    out += ["---", "", "　·　".join(nav), "",
            "本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` "
            "機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。", ""]
    return "\n".join(out)


# ───────────────────── 總目錄（html ＋ md）─────────────────────────

def render_index_html(data: dict, mapping: dict[str, str]) -> str:
    groups = []
    for st in data["stages"]:
        es = [x for x in data["entries"] if MB.STAGE_OF.get(x["tier"]) == st["id"]]
        rows = []
        for x in sorted(es, key=lambda v: v["tier"]):
            n = sum(1 for d, eid in mapping.items() if eid == x["id"])
            sub = sub_of(data, x["sub"])
            t = tier_of(data, x["tier"])
            rows.append(
                f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee;">'
                f'<b style="color:{t["color"]}">{esc(x["tier"])}</b></td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #eee;">'
                f'<a href="{x["id"]}.html">{esc(x["name"])}</a>'
                f'<div style="font-size:11px;color:#64748b;">{esc(plain(x.get("goal", ""))[:80])}…</div></td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #eee;">'
                f'<span style="color:{sub["color"]}">●</span> {esc(sub["label"])}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center;">{n}</td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #eee;">'
                f'<a href="{x["id"]}.md">MD</a></td>'
                f'<td style="padding:6px 8px;border-bottom:1px solid #eee;">'
                f'{esc(plain(x.get("res", ""))[:70])}</td></tr>')
        groups.append(
            f'<div style="font-size:14px;font-weight:600;margin:18px 0 6px;">'
            f'{esc(st["label"])}（{len(es)}）'
            f'<span style="font-size:12px;font-weight:400;color:#64748b;margin-left:8px;">'
            f'{esc(st["desc"])}</span></div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            f'<tr style="background:#f1f5f9;">'
            f'<td style="padding:6px 8px;font-weight:600;">級</td>'
            f'<td style="padding:6px 8px;font-weight:600;">條目（點開白皮書）</td>'
            f'<td style="padding:6px 8px;font-weight:600;">子目標</td>'
            f'<td style="padding:6px 8px;font-weight:600;">報告</td>'
            f'<td style="padding:6px 8px;font-weight:600;">MD</td>'
            f'<td style="padding:6px 8px;font-weight:600;">結果</td></tr>'
            f'{"".join(rows)}</table>')
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>250 / 25 攻關 — 逐條技術白皮書總目錄（{len(data["entries"])} 條）</title>
</head>
<body style="margin:0;padding:0;">
<div style="width:100%;box-sizing:border-box;padding:18px;font-family:-apple-system,'PingFang TC',sans-serif;color:#1a1a1a;">
  <h2 style="margin:0 0 4px;font-size:19px;">250 / 25 攻關 — 逐條技術白皮書（{len(data["entries"])} 條）</h2>
  <div style="font-size:12px;color:#666;margin-bottom:14px;line-height:1.6;">
    每一條實驗／嘗試一對檔案：<b>目標 → 判準 → 結果 → 判定</b> ＋ 依據 ＋ 對應報告。
    　<a href="../index.html">← 回 mindmap 總圖</a>　·　<a href="index.md">MD 版總目錄</a></div>
  {"".join(groups)}
  <div style="margin-top:14px;font-size:11px;color:#94a3b8;">由
    <code>scripts/check/mindmap_brief_build.py</code> 從
    <code>docs/mindmap/mindmap.json</code> 機械生成。</div>
</div>
</html>
"""


def render_index_md(data: dict, mapping: dict[str, str]) -> str:
    out = [f'# 250 / 25 攻關 — 逐條技術白皮書總目錄（{len(data["entries"])} 條）', "",
           "> 每一條實驗／嘗試一對檔案（`html` ＋ `md`）：**目標 → 判準 → 結果 → 判定** ＋ 依據 ＋ 對應報告。",
           "> 回 [mindmap 總圖](../index.html)　·　[HTML 版總目錄](index.html)", ""]
    for st in data["stages"]:
        es = [x for x in data["entries"] if MB.STAGE_OF.get(x["tier"]) == st["id"]]
        out += [f'## {plain(st["label"])}（{len(es)}）', "", f'*{plain(st["desc"])}*', "",
                "| 級 | 條目 | 子目標 | 報告 | MD | 結果 |", "|---|---|---|---|---|---|"]
        for x in sorted(es, key=lambda v: v["tier"]):
            n = sum(1 for d, eid in mapping.items() if eid == x["id"])
            sub = sub_of(data, x["sub"])
            out.append(f'| {x["tier"]} | [{plain(x["name"])}]({x["id"]}.html) '
                       f'| {plain(sub["label"])} | {n} | [{x["id"]}.md]({x["id"]}.md) '
                       f'| {plain(x.get("res", ""))[:70]} |')
        out.append("")
    out += ["---", "",
            "由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成。", ""]
    return "\n".join(out)


# ───────────────────── 生成 ──────────────────────────────────────

def build(data=None, mapping=None) -> list[str]:
    if data is None or mapping is None:
        data, mapping = load()
    problems: list[str] = []
    for e in data["entries"]:
        if not e.get("goal"):
            problems.append(f"條目 {e['id']} 缺 goal（目標）")
    if problems:
        return problems

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    order = [e["id"] for e in data["entries"]]
    for i, e in enumerate(data["entries"]):
        (BRIEF_DIR / f"{e['id']}.html").write_text(
            render_brief_html(e, data, mapping, order, i), encoding="utf-8")
        (BRIEF_DIR / f"{e['id']}.md").write_text(
            render_brief_md(e, data, mapping, order, i), encoding="utf-8")
    (BRIEF_DIR / "index.html").write_text(render_index_html(data, mapping), encoding="utf-8")
    (BRIEF_DIR / "index.md").write_text(render_index_md(data, mapping), encoding="utf-8")
    return problems


def selftest() -> bool:
    ok = True

    def chk(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    data, mapping = load()
    chk("每條目都有 goal", all(e.get("goal") for e in data["entries"]))
    chk("每條目都有 crit / res / evid",
        all(e.get("crit") and e.get("res") for e in data["entries"]))
    chk("globs 可解析出對應報告", len(mapping) > 0)

    problems = build()
    chk("生成無問題", not problems)
    if problems:
        print("   ", problems[:3])

    files = sorted(BRIEF_DIR.glob("*.html"))
    mds = sorted(BRIEF_DIR.glob("*.md"))
    n = len(data["entries"])
    chk(f"每條目一份 html（{n} 份 + index）",
        len([f for f in files if f.name != "index.html"]) == n)
    chk(f"每條目一份 md（{n} 份 + index）",
        len([f for f in mds if f.name != "index.md"]) == n)
    chk("index.html 存在", (BRIEF_DIR / "index.html").exists())
    chk("index.md 存在", (BRIEF_DIR / "index.md").exists())

    # utf-8 宣告不是菜：沒有 `<meta charset="utf-8">` 的頁面用 HTTP 開就是整頁亂碼
    # （實測：瀏覽器以 latin-1 解讀 ⇒ 中文全爛，而「產生器 selftest 綠」完全看不到這件事）。
    missing = [f.name for f in files if "charset" not in
               f.read_text(encoding="utf-8")[:400].lower()]
    chk("每一份 html 都宣告 utf-8", not missing)
    if missing:
        print("   缺宣告：", missing[:5])

    sample = data["entries"][3]
    h = (BRIEF_DIR / f"{sample['id']}.html").read_text(encoding="utf-8")
    m = (BRIEF_DIR / f"{sample['id']}.md").read_text(encoding="utf-8")
    chk("html 含四段（目標/判準/結果/判定）", all(f"{num} {l}" in h for num, l, _ in STEPS))
    chk("md 含四段（1~4 節）", all(f"## {num}. {l}" in m for num, l, _ in STEPS))
    chk("html 對齊樣例骨架（三卡＋四段卡＋結論框）",
        h.count("border-radius:8px") >= 7 and "#fffbeb" in h)
    chk("html 含依據與備註", "依據" in h and "備註" in h)
    chk("html 含對應報告", "對應報告" in h)
    chk("html 有回總目錄的連結", 'href="index.html"' in h)
    chk("html 有 MD 版連結", f'href="{sample["id"]}.md"' in h)
    chk("md 有 HTML 版連結", f"({sample['id']}.html)" in m)
    chk("md 首行是 H1 且有一句話", m.startswith("# ") and "**一句話**" in m)

    idx = (BRIEF_DIR / "index.html").read_text(encoding="utf-8")
    idxm = (BRIEF_DIR / "index.md").read_text(encoding="utf-8")
    chk("html 總目錄連到每一份白皮書",
        all(f'href="{e["id"]}.html"' in idx for e in data["entries"]))
    chk("md 總目錄連到每一份 md",
        all(f']({e["id"]}.md)' in idxm for e in data["entries"]))

    # 負例：缺 goal 要紅（且不該寫出任何檔案）
    bad = json.loads(json.dumps(data))
    bad["entries"][0].pop("goal", None)
    probs = build(bad, mapping)
    chk("缺 goal 會被抓到", any("缺 goal" in x and "m-prefill250" in x for x in probs))

    # 連結有效性：html 的 href 與 md 的 () 都指向存在的檔
    broken = []
    for f in files + mds:
        txt = f.read_text(encoding="utf-8")
        hrefs = re.findall(r'href="([^"]+)"', txt) + re.findall(r"\]\(([^)]+)\)", txt)
        for href in hrefs:
            if href.startswith("http") or href.startswith("#"):
                continue
            if not (f.parent / href).resolve().exists():
                broken.append(f"{f.name} → {href}")
    chk(f"所有相對連結都有效（壞連結 {len(broken)}）", not broken)
    if broken:
        print("   ", broken[:3])
    return ok


def main() -> int:
    if "--selftest" in sys.argv:
        print("════ mindmap_brief_build selftest ════")
        ok = selftest()
        print(f"\n{'SELFTEST OK' if ok else 'SELFTEST FAIL'}")
        return 0 if ok else 1

    data, mapping = load()
    problems = build()
    if problems:
        for p in problems:
            print("  ✗", p)
        return 1
    n = len(data["entries"])
    print(f"  已生成 {n} 對白皮書（html ＋ md）→ docs/mindmap/briefs/"
          f"（對映報告 {len(mapping)} 份）")
    print("  總目錄 → docs/mindmap/briefs/index.html、index.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
