#!/usr/bin/env python3
"""mindmap_build.py — 250/25 攻關總圖：校驗 + 渲染（兩個維度）

維度 1（**階段**，決定分支）：生產（① ② ③b）／ 實驗階段（③a）／ 已結案（④ ＋ 不適用）
維度 2（**子目標**，決定顏色）：S 序列化消減＝粉紅／M MTP on 加速＝黃／兩者＝橙／不適用＝灰

資料：`docs/mindmap/mindmap.json`
產物：`docs/mindmap/index.html`（互動 mindmap）、`docs/mindmap/TAXONOMY_2026-09-25.md`
規則來源：`docs/MEASUREMENT_CONTRACT_2026-09-25.md` §4（子目標）／§5（四級）／§7（作廢數字）

用法：
    python3 scripts/check/mindmap_build.py            # 校驗 + 渲染
    python3 scripts/check/mindmap_build.py --check    # 只校驗
    python3 scripts/check/mindmap_build.py --selftest
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MM_DIR = ROOT / "docs/mindmap"
DATA = MM_DIR / "mindmap.json"

STAGE_OF = {"1": "prod", "2": "prod", "3b": "prod", "3a": "exp", "4": "closed", "na": "closed"}

SCOPE_RE = re.compile(
    r"PREFILL250|DECODE25|DECODE_25|25TPS|_25_|MILE|M25|M-25|TARGET|ARITHMETIC|STRATEGY"
    r"|VERDICT|LEDGER|TPOT|MTP|SHAPE|G7|ORNITH|CROSS_LINE|TODAY_TECH|ROADMAP|IO_PATH"
    r"|SWAP|MISS|S1_|SEG_BATCH|K3_|KNOB|CEILING|JOINT_|EXPERT_CACHE|SOFTPOOL|STEP23"
    r"|EXPERT_IO|IOCACHE|G1_|G4_|K5_|PREBIND|RHO|M_W_|NEIGHBOUR|H_MEASURED|PROGRESS_25|IDEAL_SHAPE"
    r"|BANDWIDTH|GPU_CEILING|FP_ORDER|BOX_ADMISSION|SERVER_WINDOW|OMLX|BEST_SHAPE|REUSE_DISTANCE"
    r"|PROD_NEW_MTP|M3_VERDICT|BIGMOMO|GAP_|L3_M1|MMAP_CACHE|CGC_Decode_Version|Gemma4|M4_SETUP"
    r"|DEVICE_BUSY|CB_IS_THE_DEVICE|DENSE_GEMV|DENSE_NSG",
    re.I)
BRANCHES = ("demo/sweet-spot-windows-fix", "devserver")


# ───────────────────── 盤點 ─────────────────────

def load_data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def branch_docs() -> list[str]:
    out: set[str] = set()
    for b in BRANCHES:
        try:
            r = subprocess.run(["git", "ls-tree", "-r", "--name-only", b, "--", "docs"],
                               cwd=ROOT, capture_output=True, text=True, timeout=60)
            out |= {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
        except Exception:
            pass
    if not out:
        out = {str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*") if p.is_file()}
    return sorted(out)


def scope_docs(docs: list[str]) -> list[str]:
    return [d for d in docs if SCOPE_RE.search(Path(d).name)]


def map_docs(entries: list[dict], docs: list[str]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    for d in docs:
        name = Path(d).name
        for e in entries:
            if any(fnmatch.fnmatch(name, pat) for pat in e.get("docs") or []):
                mapping[d] = e["id"]
                break
    return mapping, [d for d in docs if d not in mapping]


# ───────────────────── 校驗 ─────────────────────

def validate(data: dict) -> list[str]:
    problems: list[str] = []
    tier_ids = [t["id"] for t in data["tiers"]]
    sub_ids = [s["id"] for s in data.get("subgoals", [])]
    stage_ids = [s["id"] for s in data.get("stages", [])]
    if len(tier_ids) != len(set(tier_ids)):
        problems.append("tiers 有重複 id")
    if len(sub_ids) != len(set(sub_ids)):
        problems.append("subgoals 有重複 id")
    used = {STAGE_OF.get(t) for t in tier_ids}
    if None in used:
        problems.append("有 tier 沒有對應階段")
    if used - set(stage_ids):
        problems.append(f"階段清單缺 {sorted(x for x in used - set(stage_ids) if x)}")
    if set(stage_ids) - used:
        problems.append(f"階段清單有未使用的階段 {sorted(set(stage_ids) - used)}")
    briefs = data.get("subgoal_briefs", {})
    if set(briefs) != set(sub_ids):
        problems.append(f"subgoal_briefs 覆蓋不全，缺 {sorted(set(sub_ids) - set(briefs))}")
    for sid, b in briefs.items():
        ks = [str(x.get("k", "")).split()[0] for x in b.get("sections", [])]
        if ks != ["1", "2", "3", "4"]:
            problems.append(f"子目標 {sid} 的說明不是「1 現狀/2 推論/3 量測/4 目標」四段：{ks}")
        if b.get("ref") and not (MM_DIR / b["ref"]).resolve().exists():
            problems.append(f"子目標 {sid} 的原始報告不存在：{b['ref']}")
        if not b.get("headline"):
            problems.append(f"子目標 {sid} 缺 headline")
    for e in data["entries"]:
        if not str(e.get("goal") or "").strip():
            problems.append(f"條目 {e.get('id', '?')} 缺 goal（目標）")
    seen: set[str] = set()
    for e in data["entries"]:
        for f in ("id", "name", "theme", "tier", "crit", "res", "evid", "sub"):
            if not e.get(f):
                problems.append(f"條目 {e.get('id','?')} 缺欄位 {f}")
        if e["id"] in seen:
            problems.append(f"條目 id 重複：{e['id']}")
        seen.add(e["id"])
        if e.get("tier") not in tier_ids:
            problems.append(f"條目 {e.get('id','?')} 的 tier「{e.get('tier')}」不在 {tier_ids}")
        if e.get("sub") not in sub_ids:
            problems.append(f"條目 {e.get('id','?')} 的 sub「{e.get('sub')}」不在 {sub_ids}")
        if e.get("tier") == "1" and e.get("id") != "m-total":
            problems.append(f"① 只允許 m-total（唯一空位）：{e['id']}")
    return problems


def selftest() -> int:
    ok: list[tuple[str, bool]] = []

    def chk(n: str, c: bool) -> None:
        ok.append((n, bool(c)))

    data = load_data()
    chk("6 個 tier / 5 個子目標 / 3 個階段",
        len(data["tiers"]) == 6 and len(data["subgoals"]) == 5 and len(data["stages"]) == 3)
    roles = {x["id"]: x.get("role") for x in data["subgoals"]}
    chk("S／M 是活躍攻關軸、C 是天花板軸",
        roles.get("S") == "active" and roles.get("M") == "active" and roles.get("C") == "ceiling")
    chk("both 是耦合類且保留", roles.get("both") == "couple")
    chk("每個子目標都有軸性質說明", all(x.get("role_label") for x in data["subgoals"]))
    chk("meta.status 三條（兩軸未交付／耦合／C 定位）", len(data["meta"].get("status", [])) >= 3)
    chk("正式資料零問題", validate(data) == [])
    chk("每個條目都有 sub", all(e.get("sub") for e in data["entries"]))
    chk("① 只出現在 prod 階段", STAGE_OF["1"] == "prod")
    chk("③a 在 exp、③b 在 prod", STAGE_OF["3a"] == "exp" and STAGE_OF["3b"] == "prod")
    chk("④ 與不適用在 closed", STAGE_OF["4"] == "closed" and STAGE_OF["na"] == "closed")

    chk("5 個子目標都有四段說明",
        all([str(x["k"]).split()[0] for x in data["subgoal_briefs"][sg]["sections"]] == ["1", "2", "3", "4"]
            for sg in data["subgoal_briefs"]))
    chk("S/M 兩個子目標都指向存在的原始報告",
        all((MM_DIR / data["subgoal_briefs"][sg]["ref"]).resolve().exists() for sg in ("S", "M")))

    bad5 = json.loads(json.dumps(data))
    bad5["subgoal_briefs"]["M"]["sections"] = bad5["subgoal_briefs"]["M"]["sections"][:2]
    chk("說明缺段被抓", any("四段" in q for q in validate(bad5)))

    bad = json.loads(json.dumps(data))
    bad["entries"][0]["tier"] = "9"
    chk("非法 tier 被抓", any("不在" in p for p in validate(bad)))

    bad2 = json.loads(json.dumps(data))
    bad2["entries"][0]["sub"] = "X"
    chk("非法子目標被抓", any("的 sub" in p for p in validate(bad2)))

    bad3 = json.loads(json.dumps(data))
    bad3["entries"].append({"id": "x", "name": "假①", "theme": "t", "tier": "1",
                            "sub": "S", "crit": "c", "res": "r", "evid": "e"})
    chk("多餘的 ① 被抓", any("只允許 m-total" in p for p in validate(bad3)))

    bad4 = json.loads(json.dumps(data))
    bad4["entries"].append({"id": "y", "name": "缺 sub", "theme": "t", "tier": "4",
                            "crit": "c", "res": "r", "evid": "e"})
    chk("缺 sub 被抓", any("缺欄位 sub" in p for p in validate(bad4)))

    docs = ["docs/PREFILL250_CERTIFIABILITY_20260916.html", "docs/MTP_K_AB_2026-09-18.md", "docs/無關.md"]
    m, un = map_docs(data["entries"], docs)
    chk("globs 可映射", len(m) == 2 and un == ["docs/無關.md"])
    chk("scope 過濾正確", len(scope_docs(docs)) == 2 and not scope_docs(["docs/random.md"]))

    for n, c in ok:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"selftest {sum(c for _, c in ok)}/{len(ok)}")
    return 0 if all(c for _, c in ok) else 1


# ───────────────────── 渲染：MD ─────────────────────

def render_md(data: dict, mapping: dict[str, str], unmapped: list[str]) -> str:
    tiers = {t["id"]: t for t in data["tiers"]}
    subs = {s["id"]: s for s in data["subgoals"]}
    stage_of = STAGE_OF
    CN = {"S": "粉紅", "M": "黃", "both": "橙", "C": "天藍", "na": "灰"}

    out = [f"# {data['meta']['title']}", "",
           f"> 規則：`{data['meta']['rule']}`", f"> 範圍：{data['meta']['scope']}",
           f"> 分支：{'、'.join(data['meta']['branches'])}", "",
           "## 兩個維度（＋子目標的軸性質）", "",
           "| 維度 | 取值 |", "|---|---|",
           "| **階段**（分支） | " + "／".join(x["label"] for x in data["stages"]) + " |",
           "| **子目標**（顏色） | " + "／".join(
               f"{x['label']}（{CN.get(x['id'], x['id'])}）"
               for x in data["subgoals"]) + " |", ""]
    out += ["| 子目標 | 顏色 | 軸性質 |", "|---|---|---|"]
    for x in data["subgoals"]:
        out.append(f"| {x['label']} | {CN.get(x['id'], x['id'])} "
                   f"| {x.get('role_label') or x.get('def', '')} |")
    out += ["", "> **狀態（三條）**", ""]
    for line in data["meta"].get("status", []):
        out.append("> - " + line)
    out.append("")

    out += ["## 統計", "", "| 階段 / 子目標 | " + " | ".join(s["label"] for s in data["subgoals"]) + " | 合計 |",
            "|---|" + "---|" * (len(data["subgoals"]) + 1)]
    for st in data["stages"]:
        row = [f"**{st['label']}**"]
        for sg in data["subgoals"]:
            row.append(str(sum(1 for e in data["entries"]
                               if stage_of[e["tier"]] == st["id"] and e["sub"] == sg["id"])))
        row.append(str(sum(1 for e in data["entries"] if stage_of[e["tier"]] == st["id"])))
        out.append("| " + " | ".join(row) + " |")
    out.append("| **合計** | " + " | ".join(
        str(sum(1 for e in data["entries"] if e["sub"] == sg["id"])) for sg in data["subgoals"])
        + " | " + str(len(data["entries"])) + " |")
    out.append("")

    out += ["## 子目標說明（現狀／推論／量測／目標）", ""]
    for sg in data["subgoals"]:
        b = data.get("subgoal_briefs", {}).get(sg["id"])
        if not b:
            continue
        cnt = sum(1 for e in data["entries"] if e["sub"] == sg["id"])
        out += [f"### {sg['label']}（{cnt} 條目）", "", f"*{sg['def']}*", "",
                f"**{b['headline']}**", ""]
        if b.get("ref"):
            out += [f"- 原始報告：[{b['ref_title']}]({b['ref']})", ""]
        for sec in b["sections"]:
            out += [f"**{sec['k']}**", ""] + [f"- {v}" for v in sec["v"]] + [""]

    out += ["## 逐條條目（依階段 → 分級）", ""]
    for st in data["stages"]:
        out += [f"### {st['label']}", "", f"*{st['desc']}*", ""]
        for t in data["tiers"]:
            if stage_of[t["id"]] != st["id"]:
                continue
            ents = [e for e in data["entries"] if e["tier"] == t["id"]]
            out += [f"#### {t['label']}（{len(ents)}）", "", f"*{t['def']}*", "",
                    "| 條目 | 子目標 | 主題 | 判準 | 結果 | 依據 |", "|---|---|---|---|---|---|"]
            for e in ents:
                out.append(f"| [**{e['name']}**](briefs/{e['id']}.html) | "
                           f"{subs[e['sub']]['label']} | {e['theme']} | "
                           f"{e['crit']} | {e['res']} | {e['evid']} |")
            out.append("")

    out += ["## 逐條白皮書（目標 → 判準 → 結果 → 判定）", "",
            "每條一份 HTML：`docs/mindmap/briefs/<id>.html`；總目錄 `briefs/index.html`。", ""]
    for st in data["stages"]:
        ents_st = [e for e in data["entries"] if stage_of[e["tier"]] == st["id"]]
        if not ents_st:
            continue
        out += [f"### {st['label']}（{len(ents_st)}）", ""]
        for e in ents_st:
            t = next(x for x in data["tiers"] if x["id"] == e["tier"])
            out += [f"#### [{e['name']}](briefs/{e['id']}.html)　·　{e['tier']} {t['label']}"
                    f"　·　{subs[e['sub']]['label']}"
                    f"　·　[MD](briefs/{e['id']}.md)", "",
                    f"- **目標**：{e.get('goal', '—')}",
                    f"- **判準**：{e['crit']}",
                    f"- **結果**：{e['res']}",
                    f"- **判定**：**{e['tier']} {t['label']}**"
                    + (f" — {e['note']}" if e.get("note") else ""),
                    f"- **依據**：{e['evid']}", ""]
    out += ["", ""]

    out += ["## 報告對映（自動；250/25 相關報告 → 條目）", "",
            f"- 命中 **{len(mapping) + len(unmapped)}** 份；已映射 **{len(mapping)}**；未映射 **{len(unmapped)}**", "",
            "| 報告 | 條目 | 階段 | 子目標 | 級 |", "|---|---|---|---|---|"]
    ent = {e["id"]: e for e in data["entries"]}
    for d in sorted(mapping):
        e = ent[mapping[d]]
        st = next(s["label"] for s in data["stages"] if s["id"] == stage_of[e["tier"]])
        out.append(f"| `{d}` | {e['name']} | {st} | {subs[e['sub']]['label']} | {e['tier']} |")
    if unmapped:
        out += ["", "**未映射（需人工歸類或補 glob）**", ""] + [f"- `{d}`" for d in unmapped]
    return "\n".join(out) + "\n"


# ───────────────────── 渲染：HTML ─────────────────────

HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2430;--dim:#5b6472;--line:#dfe3ea;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#11151c;--panel:#171d26;--ink:#e8ecf3;--dim:#9aa5b4;--line:#2a3341;--accent:#7aa2ff}}
body.dark{--bg:#11151c;--panel:#171d26;--ink:#e8ecf3;--dim:#9aa5b4;--line:#2a3341;--accent:#7aa2ff}
body.light{--bg:#f6f7f9;--panel:#fff;--ink:#1f2430;--dim:#5b6472;--line:#dfe3ea;--accent:#2563eb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{padding:16px 22px 10px;border-bottom:1px solid var(--line);background:var(--panel);position:relative}
h1{margin:0 0 6px;font-size:19px}
.meta{color:var(--dim);font-size:12.5px}
.meta code{background:var(--bg);padding:1px 5px;border-radius:4px}
#btns{position:absolute;right:18px;top:16px}
button{font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
.legend{padding:8px 22px 10px;background:var(--panel);border-bottom:1px solid var(--line)}
.lrow{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:3px 0}
.lrow .lab{color:var(--dim);font-size:12px;min-width:86px}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;border:1px solid var(--line);border-radius:999px;padding:2px 9px;cursor:pointer;user-select:none}
.chip.off{opacity:.3}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.ring{width:10px;height:10px;border-radius:3px;display:inline-block;border:2px solid}
.wrap{display:flex;align-items:stretch;min-height:calc(100vh - 190px)}
#mind{flex:1 1 auto;overflow:auto;padding:8px 4px 30px}
aside{flex:0 0 400px;border-left:1px solid var(--line);background:var(--panel);padding:14px 16px;overflow:auto;max-height:calc(100vh - 190px)}
@media (max-width:1100px){.wrap{flex-direction:column}aside{flex:none;max-height:none;border-left:0;border-top:1px solid var(--line)}}
.node{cursor:pointer}
.node rect.box{fill:var(--panel);stroke:var(--line)}
.node:hover rect.box{stroke:var(--accent);stroke-width:2}
.node.dim{opacity:.16}
.node text{fill:var(--ink);font-size:12.5px;pointer-events:none}
.node .tier{fill:var(--dim);font-size:10.5px}
.link{fill:none;stroke:var(--line);stroke-width:1.4}
.link.closed{stroke-dasharray:3 3}
h2{font-size:13px;color:var(--dim);margin:16px 0 6px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--bg)}
.card h3{margin:0 0 6px;font-size:14px}
.kv{display:grid;grid-template-columns:58px 1fr;gap:2px 8px;font-size:12.5px}
.kv b{color:var(--dim);font-weight:600}
code{font-size:11.5px;word-break:break-all}
.pill{font-size:11px;border-radius:999px;padding:1px 7px;color:#fff;white-space:nowrap}
input[type=search]{width:100%;padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit;margin-bottom:8px}
.note{color:var(--dim);font-size:12px}
.chip .t{cursor:pointer}
.chip .x{margin-left:3px;padding:0 4px;border-radius:5px;color:var(--dim);font-size:11.5px}
.chip .x:hover{background:var(--bg);color:var(--ink)}
.chip.active{box-shadow:0 0 0 2px var(--accent)}
.brief h4{margin:13px 0 4px;font-size:12.5px;color:var(--accent);border-left:3px solid var(--accent);padding-left:7px}
.brief ul{margin:2px 0 0;padding-left:17px}
.brief li{margin:3px 0;font-size:12.5px}
.brief .head{font-size:12.5px;font-weight:600;margin:2px 0 9px;line-height:1.55}
.bars{display:flex;height:9px;border-radius:5px;overflow:hidden;margin:7px 0 5px;border:1px solid var(--line)}
.bars i{display:block;height:100%}
.bk{display:flex;flex-wrap:wrap;gap:5px 10px;margin:3px 0 9px}
.bk span{font-size:11.5px;color:var(--dim)}
.grp{border-left:3px solid var(--line);padding-left:8px;margin:9px 0}
.grp .gt{font-size:12px;font-weight:600;color:var(--dim)}
.grp .gi{cursor:pointer;font-size:12.5px;margin:3px 0}
.grp .gi:hover{color:var(--accent);text-decoration:underline}
iframe{width:100%;height:360px;border:1px solid var(--line);border-radius:8px;background:#fff;margin-top:8px}
.wpbar{display:flex;align-items:center;gap:8px;margin:6px 0 10px}
.wpbar a.wp{flex:1;font-size:12.5px;font-weight:600;color:var(--accent);text-decoration:none;
  border:1px solid var(--accent);border-radius:6px;padding:5px 9px;text-align:center}
.wpbar a.wp:hover{background:var(--accent);color:#fff}
.wpbar button{font:inherit;font-size:12px;padding:5px 9px;border:1px solid var(--line);
  border-radius:6px;background:var(--bg);color:var(--ink);cursor:pointer}
.hdr a{color:var(--accent)}
.mtx{border-collapse:collapse;font-size:12px;margin:6px 0}
.mtx th,.mtx td{border:1px solid var(--line);padding:3px 8px;text-align:center}
.mtx th{color:var(--dim);font-weight:600}
.banner{margin:10px 22px 0;border-left:4px solid #f59e0b;background:var(--panel);
  border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line);
  border-radius:7px;padding:8px 12px;font-size:12.5px;color:var(--dim)}
.banner b{color:var(--ink)}
.banner ul{margin:5px 0 0;padding-left:18px}
.banner li{margin:2px 0}
</style>
</head>
<body>
<header>
  <div id="btns"><button id="theme">切換主題</button></div>
  <h1>__TITLE__</h1>
  <div class="meta">範圍：__SCOPE__<br>分支：__BRANCHES__<br>規則：<code>__RULE__</code></div>
  <div class="meta hdr">逐條技術白皮書：<a href="briefs/index.html">總目錄（__NENTRY__ 條：目標 → 判準 → 結果 → 判定）</a></div>
</header>
<div class="legend">
  <div class="lrow"><span class="lab">階段（分支）</span><span class="note">生產 ＝ ① ② ③b｜實驗階段 ＝ ③a｜已結案 ＝ ④ ＋ 不適用</span></div>
  <div class="lrow" id="subLegend"><span class="lab">子目標（顏色）</span></div>
  <div class="lrow" id="tierLegend"><span class="lab">分級（外框）</span></div>
</div>
__BANNER__
<div class="wrap">
  <div id="mind"></div>
  <aside>
    <input type="search" id="q" placeholder="搜尋條目 / 報告檔名…">
    <div id="detail"><p class="note">載入中…</p></div>
    <h2>報告對映（<span id="nmap"></span>）</h2>
    <div id="doclist"></div>
  </aside>
</div>
<script>
const DATA = __DATA__, MAP = __MAP__, UNMAPPED = __UNMAPPED__;
const TIER = Object.fromEntries(DATA.tiers.map(t => [t.id, t]));
const SUB  = Object.fromEntries(DATA.subgoals.map(s => [s.id, s]));
const STAGE = DATA.stages;
const STAGE_OF = __STAGE_OF__;
const ENT = Object.fromEntries(DATA.entries.map(e => [e.id, e]));
const BRIEF = __BRIEFS__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const hidTier = new Set(), hidSub = new Set();
const filt = e => !hidTier.has(e.tier) && !hidSub.has(e.sub);

function chipRow(el, items, hidden, onOpen){
  el.querySelectorAll('.chip').forEach(n => n.remove());
  items.forEach(o => {
    const c = document.createElement('span');
    c.className = 'chip';
    const mark = o.color
      ? `<span class="dot" style="background:${o.color}"></span>`
      : `<span class="ring" style="border-color:${o.color}"></span>`;
    const xbtn = onOpen ? `<span class="x" title="過濾：顯示／隱藏這個子目標">&#8856;</span>` : '';
    c.innerHTML = mark + `<span class="t">${o.label}</span>` + xbtn;
    const toggle = () => {
      hidden.has(o.id) ? hidden.delete(o.id) : hidden.add(o.id);
      c.classList.toggle('off'); draw();
    };
    if (onOpen) {
      c.title = (o.def || '') + (o.role_label ? '　｜　軸性質：' + o.role_label : '')
        + '　｜　點標籤＝看說明（現狀／推論／量測／目標）　·　點 ⊘＝過濾';
      c.querySelector('.t').onclick = () => onOpen(o.id, c);
    } else {
      c.title = o.def || '';
      c.onclick = toggle;
    }
    const xn = c.querySelector('.x');
    if (xn) xn.onclick = ev => { ev.stopPropagation(); toggle(); };
    el.appendChild(c);
  });
}
chipRow(document.getElementById('subLegend'), DATA.subgoals, hidSub, showSub);
chipRow(document.getElementById('tierLegend'), DATA.tiers, hidTier, null);

const NW = 254, NH = 30, VG = 9, HG = 96, PAD = 18;

function draw(){
  const q = (document.getElementById('q').value || '').trim().toLowerCase();
  const hit = e => !q || (e.name + e.theme + e.res + e.sub + e.tier + (e.docs||[]).join(' ')).toLowerCase().includes(q);
  const shown = DATA.entries.filter(filt);

  const rows = []; let y = PAD;
  STAGE.forEach(st => {
    const es = shown.filter(e => STAGE_OF[e.tier] === st.id);
    st._es = es; st._y = y;
    y += NH + VG;                                  // 階段標題
    es.forEach(e => { rows.push({e, y}); y += NH + VG; });
    y += 12;                                       // 階段間距
  });
  const H = Math.max(340, y + PAD);
  const xRoot = PAD, xStage = PAD + 200 + HG, xEnt = xStage + NW + HG;
  const W = xEnt + NW + PAD;
  const rootY = H/2 - NH/2;

  const P = [];
  P.push(`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`);
  const link = (x1,a,x2,b,dash) =>
    `<path class="link${dash?' closed':''}" d="M${x1} ${a} C${(x1+x2)/2} ${a} ${(x1+x2)/2} ${b} ${x2} ${b}"/>`;
  // 線
  STAGE.forEach(st => {
    const mid = st._y + NH/2, h = Math.max(NH, st._es.length * (NH+VG));
    P.push(link(xRoot+200, rootY+NH/2, xStage, st._y + (h - VG)/2, st.id === 'closed'));
    rows.filter(r => STAGE_OF[r.e.tier] === st.id).forEach(r =>
      P.push(link(xStage+NW, st._y + NH/2, xEnt, r.y + NH/2, st.id === 'closed')));
  });
  const node = (x, yy, w, txt, opt = {}) => {
    const sub = SUB[opt.sub] || null, tier = TIER[opt.tier] || null;
    const dim = opt.eid && !hit(ENT[opt.eid]) ? ' dim' : '';
    const attrs = `data-id="${opt.eid||''}"`
      + (opt.stage ? ` data-stage="${opt.stage}"` : '') + (opt.root ? ` data-root="1"` : '');
    return `<g class="node${dim}" ${attrs} transform="translate(${x},${yy})">
      <rect class="box" x="0" y="0" width="${w}" height="${NH}" rx="7"
        ${tier?`stroke="${tier.color}" stroke-width="2"`:''}/>
      ${sub?`<rect x="0" y="0" width="5" height="${NH}" rx="2.5" fill="${sub.color}"/>`:''}
      <text x="14" y="19">${esc(txt)}</text>
      ${opt.tierTxt?`<text class="tier" x="${w-9}" y="19" text-anchor="end">${esc(opt.tierTxt)}</text>`:''}
    </g>`;
  };
  P.push(node(xRoot, rootY, 200, '250 / 25 攻關（點看總覽）', {tier: null, root: true}));
  STAGE.forEach(st => {
    const h = Math.max(NH, st._es.length * (NH+VG));
    P.push(node(xStage, st._y + (h - VG)/2 - NH/2, NW, st.label,
                {tierTxt: st._es.length ? String(st._es.length) : '0',
                 sub: null, eid: '', stage: st.id}));
  });
  rows.forEach(r => {
    const e = r.e;
    P.push(node(xEnt, r.y, NW, e.name, {sub: e.sub, tier: e.tier, tierTxt: e.tier, eid: e.id}));
  });
  P.push('</svg>');
  document.getElementById('mind').innerHTML = P.join('');
  document.querySelectorAll('.node[data-id]').forEach(g => {
    if (g.dataset.id) g.onclick = () => show(g.dataset.id);
  });
  document.querySelectorAll('.node[data-stage]').forEach(g => {
    g.onclick = () => showStage(g.dataset.stage, g);
  });
  const r = document.querySelector('.node[data-root]');
  if (r) r.onclick = () => showOverview();
}

function show(id){
  const e = ENT[id], t = TIER[e.tier], s = SUB[e.sub];
  const st = STAGE.find(x => x.id === STAGE_OF[e.tier]);
  const docs = MAP.filter(m => m.entry === id).map(m => m.doc);
  const brief = 'briefs/' + id + '.html';
  const briefMd = 'briefs/' + id + '.md';
  const dsrc = e.detail_html || brief;
  const rich = !!e.detail_html;
  document.getElementById('detail').innerHTML = `
    <div class="card">
      <h3>${esc(e.name)}</h3>
      <div class="wpbar">
        <a class="wp" href="${dsrc}" target="_blank" rel="noopener"
           title="另開完整頁面">${rich?'方案完整視圖 ↗':'技術白皮書 ↗'}</a>
        <a class="wp" href="${briefMd}" target="_blank" rel="noopener"
           title="另開 MD 版（同一份內容的純文字版）">MD ↗</a>
        <button id="wpemb" title="在本頁內嵌">${rich?'收起豐富視圖':'內嵌'}</button>
      </div>
      <div class="kv">
        <b>階段</b><span>${esc(st.label)}</span>
        <b>分級</b><span><span class="pill" style="background:${t.color}">${esc(t.label)}</span></span>
        <b>子目標</b><span><span class="pill" style="background:${s.color}">${esc(s.label)}</span></span>
        <b>主題</b><span>${esc(e.theme)}</span>
        <b>目標</b><span>${e.goal || '—'}</span>
        <b>判準</b><span>${esc(e.crit)}</span>
        <b>結果</b><span>${esc(e.res)}</span>
        <b>依據</b><span><code>${esc(e.evid)}</code></span>
        ${e.note?`<b>註</b><span>${esc(e.note)}</span>`:''}
      </div>
      <h2>對應報告（${docs.length}）</h2>
      ${docs.length?docs.map(d=>`<div><a href="../${esc(d.replace('docs/',''))}" target="_blank"
        rel="noopener"><code>${esc(d)}</code></a></div>`).join('')
        :'<p class="note">（無對應報告檔；判準／結果見上方欄位與技術白皮書）</p>'}
    </div>`;
  const wb = document.getElementById('wpemb');
  const card = document.querySelector('#detail .card');
  function toggleEmbed(){
    const old = card.querySelector('iframe');
    if (old) { old.remove(); wb.textContent = rich ? '展開豐富視圖' : '內嵌'; return; }
    const f = document.createElement('iframe');
    f.src = dsrc; f.title = e.name + ' — 完整視圖';
    if (rich) f.style.height = '700px';
    card.appendChild(f); wb.textContent = rich ? '收起豐富視圖' : '收起';
  }
  if (wb) wb.onclick = toggleEmbed;
  if (rich) toggleEmbed();   // 點此節點即內嵌呈現豐富頁面
}

function asideTop(){ const a = document.querySelector('aside'); if (a) a.scrollTop = 0; }

function tierBars(es){
  const tc = {}; es.forEach(e => tc[e.tier] = (tc[e.tier] || 0) + 1);
  const tot = es.length || 1;
  return `<div class="bars">${Object.keys(tc).sort().map(t =>
    `<i style="width:${(tc[t] / tot * 100).toFixed(2)}%;background:${TIER[t].color}"
       title="${esc(TIER[t].label)} ${tc[t]}"></i>`).join('')}</div>
    <div class="bk">${Object.keys(tc).sort().map(t =>
      `<span><span class="pill" style="background:${TIER[t].color}">${esc(t)}</span>
       ${esc(TIER[t].label)} ${tc[t]}</span>`).join('')}</div>`;
}

// ★ 子目標說明：現狀 / 推論 / 量測 / 目標 ＋ 拆解的子目標狀況
function showSub(id, chip){
  const s = SUB[id]; if (!s) return;
  const b = BRIEF[id] || {headline: '（缺說明）', sections: []};
  document.querySelectorAll('#subLegend .chip').forEach(c => c.classList.remove('active'));
  if (chip) chip.classList.add('active');
  const es = DATA.entries.filter(e => e.sub === id);
  const groups = STAGE.map(st => ({st, es: es.filter(e => STAGE_OF[e.tier] === st.id)}))
                      .filter(g => g.es.length);
  document.getElementById('detail').innerHTML = `
    <div class="card brief">
      <h3><span class="pill" style="background:${s.color}">${esc(s.label)}</span></h3>
      <div class="head">${b.headline}</div>
      <div class="kv">
        <b>定義</b><span>${esc(s.def)}</span>
        <b>軸性質</b><span>${esc(s.role_label || '（未標注）')}</span>
        <b>條目</b><span>${es.length} 條｜${groups.map(g => esc(g.st.label) + ' ' + g.es.length).join(' ／ ')}</span>
        <b>報告</b><span>${b.ref
          ? `<a href="${b.ref}" target="_blank">${esc(b.ref_title)}</a> <button id="emb">內嵌檢視</button>`
          : '<span class="note">（無專屬報告）</span>'}</span>
      </div>
      ${b.sections.map(sec =>
        `<h4>${esc(sec.k)}</h4><ul>${sec.v.map(v => `<li>${v}</li>`).join('')}</ul>`).join('')}
      <h4>拆解的子目標狀況（${es.length} 條目）</h4>
      ${tierBars(es)}
      ${groups.map(g => `<div class="grp" style="border-left-color:${s.color}">
        <div class="gt">${esc(g.st.label)}（${g.es.length}）</div>
        ${g.es.map(e => `<div class="gi" data-e="${esc(e.id)}">
          <span class="pill" style="background:${TIER[e.tier].color}">${esc(e.tier)}</span>
          ${esc(e.name)} — ${esc(e.res)}</div>`).join('')}
      </div>`).join('')}
    </div>`;
  const emb = document.getElementById('emb');
  if (emb) emb.onclick = () => {
    const card = document.querySelector('#detail .brief');
    const old = card.querySelector('iframe');
    if (old) { old.remove(); emb.textContent = '內嵌檢視'; return; }
    const f = document.createElement('iframe');
    f.src = b.ref; f.title = b.ref_title || '';
    card.appendChild(f); emb.textContent = '收起內嵌';
  };
  document.querySelectorAll('#detail .gi[data-e]').forEach(x => x.onclick = () => show(x.dataset.e));
  asideTop();
}

function showStage(id){
  const st = STAGE.find(x => x.id === id); if (!st) return;
  const es = DATA.entries.filter(e => STAGE_OF[e.tier] === id);
  const bySub = DATA.subgoals.map(sg => ({sg, n: es.filter(e => e.sub === sg.id).length}))
                             .filter(x => x.n);
  document.getElementById('detail').innerHTML = `
    <div class="card brief">
      <h3>${esc(st.label)}</h3>
      <div class="head">${esc(st.desc)}</div>
      <div class="kv"><b>條目</b><span>${es.length}</span>
        <b>子目標</b><span>${bySub.map(x =>
          `<span class="pill" style="background:${x.sg.color}">${esc(x.sg.id)}</span> ${x.n}`).join('　')}</span></div>
      ${tierBars(es)}
      ${bySub.map(x => `<div class="grp" style="border-left-color:${x.sg.color}">
        <div class="gt">${esc(x.sg.label)}（${x.n}）
          <a href="#" data-sub="${esc(x.sg.id)}" style="font-weight:400">看說明 ›</a></div>
        ${es.filter(e => e.sub === x.sg.id).map(e => `<div class="gi" data-e="${esc(e.id)}">
          <span class="pill" style="background:${TIER[e.tier].color}">${esc(e.tier)}</span>
          ${esc(e.name)} — ${esc(e.res)}</div>`).join('')}
      </div>`).join('')}
    </div>`;
  document.querySelectorAll('#detail a[data-sub]').forEach(a =>
    a.onclick = ev => { ev.preventDefault(); showSub(a.dataset.sub); });
  document.querySelectorAll('#detail .gi[data-e]').forEach(x => x.onclick = () => show(x.dataset.e));
  asideTop();
}

function showOverview(){
  const subs = DATA.subgoals, sts = STAGE;
  const n = (st, sg) => DATA.entries.filter(e => STAGE_OF[e.tier] === st.id && e.sub === sg.id).length;
  const rowTot = st => subs.reduce((a, sg) => a + n(st, sg), 0);
  const colTot = sg => sts.reduce((a, st) => a + n(st, sg), 0);
  document.getElementById('detail').innerHTML = `
    <div class="card brief">
      <h3>250 / 25 攻關總覽</h3>
      <div class="head">${DATA.entries.length} 條目 × ${MAP.length} 份報告（映射 ${MAP.length}／未映射 ${UNMAPPED.length}）</div>
      <table class="mtx">
        <tr><th>階段 / 子目標</th>${subs.map(s =>
          `<th style="color:${s.color}">${esc(s.label)}</th>`).join('')}<th>合計</th></tr>
        ${sts.map(st => `<tr><th>${esc(st.label)}</th>${subs.map(sg => {
          const c = n(st, sg);
          return `<td>${c ? `<a href="#" data-sub="${esc(sg.id)}"
            style="color:${sg.color};font-weight:600">${c}</a>` : '·'}</td>`;
        }).join('')}<td><b>${rowTot(st)}</b></td></tr>`).join('')}
        <tr><th>合計</th>${subs.map(sg => `<td><b>${colTot(sg)}</b></td>`).join('')}
          <td><b>${DATA.entries.length}</b></td></tr>
      </table>
      <p class="note">點上方<b>子目標標籤</b>看「現狀／推論／量測／目標」＋拆解狀況；點標籤旁的
        <b>⊘</b> 只做過濾；點左側任一節點看單條細節；點<b>根節點</b>回總覽。</p>
    </div>`;
  document.querySelectorAll('#detail a[data-sub]').forEach(a =>
    a.onclick = ev => { ev.preventDefault(); showSub(a.dataset.sub); });
  asideTop();
}

document.getElementById('nmap').textContent =
  MAP.length + (UNMAPPED.length ? ` + 未映射 ${UNMAPPED.length}` : '');
document.getElementById('doclist').innerHTML = MAP.map(m => {
  const e = ENT[m.entry], s = SUB[e.sub];
  return `<div><code>${esc(m.doc)}</code>
    <span class="pill" style="background:${TIER[e.tier].color}">${e.tier}</span>
    <span class="pill" style="background:${s.color}">${s.id}</span></div>`;
}).join('') + (UNMAPPED.length
  ? `<h2>未映射</h2>` + UNMAPPED.map(d=>`<div><code>${esc(d)}</code></div>`).join('') : '');

document.getElementById('q').addEventListener('input', draw);
document.getElementById('theme').onclick = () => {
  document.body.classList.toggle('dark'); document.body.classList.toggle('light'); draw();
};
draw();
showOverview();
</script>
</body>
</html>
"""


def render_html(data: dict, mapping: dict[str, str], unmapped: list[str]) -> str:
    h = HTML_TMPL
    h = h.replace("__TITLE__", data["meta"]["title"])
    h = h.replace("__SCOPE__", data["meta"]["scope"])
    h = h.replace("__BRANCHES__", "、".join(data["meta"]["branches"]))
    h = h.replace("__RULE__", data["meta"]["rule"])
    h = h.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    h = h.replace("__MAP__", json.dumps(
        [{"doc": d, "entry": e} for d, e in sorted(mapping.items())], ensure_ascii=False))
    h = h.replace("__UNMAPPED__", json.dumps(unmapped, ensure_ascii=False))
    h = h.replace("__STAGE_OF__", json.dumps(STAGE_OF, ensure_ascii=False))
    h = h.replace("__BRIEFS__", json.dumps(data.get("subgoal_briefs", {}), ensure_ascii=False))
    h = h.replace("__NENTRY__", str(len(data["entries"])))
    st = data["meta"].get("status", [])
    h = h.replace("__BANNER__", (
        '<div class="banner"><b>狀態（三條）</b><ul>'
        + "".join(f"<li>{x}</li>" for x in st) + "</ul></div>") if st else "")
    return h


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    data = load_data()
    problems = validate(data)
    if problems:
        print("⛔ 資料問題：")
        for p in problems:
            print("   -", p)
        return 1

    docs = scope_docs(branch_docs())
    mapping, unmapped = map_docs(data["entries"], docs)
    print(f"報告盤點：命中 250/25 報告 {len(docs)}；已映射 {len(mapping)}；未映射 {len(unmapped)}")
    for d in unmapped:
        print("   · 未映射:", d)
    for st in data["stages"]:
        n = sum(1 for e in data["entries"] if STAGE_OF[e["tier"]] == st["id"])
        print(f"  階段 {st['label']}: {n} 條目")
    for sg in data["subgoals"]:
        n = sum(1 for e in data["entries"] if e["sub"] == sg["id"])
        print(f"  子目標 {sg['label']}: {n} 條目")
    if args.check:
        return 0

    (MM_DIR / "index.html").write_text(render_html(data, mapping, unmapped), encoding="utf-8")
    (MM_DIR / "TAXONOMY_2026-09-25.md").write_text(render_md(data, mapping, unmapped), encoding="utf-8")
    print("渲染完成：docs/mindmap/index.html、docs/mindmap/TAXONOMY_2026-09-25.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
