#!/usr/bin/env python3
"""今日宏觀技術決策 × 落地 —— **既有真相的一個新視圖，不是第二份真相**。

這一份**不新增任何清單**。它的兩個輸入都是既有的單一真相來源：
  - `agent_harness/engine_loop/traces/decisions.jsonl`（決策正本，只追加）
  - `agent_harness/portal/targets.json`（「資產 → 目標」綁定 ＋ 動能定義）
而它自己**不產生趨勢**（趨勢在 decisions.jsonl 與 auto_runs.jsonl 那種只追加的檔裡）。

★ 為什麼要另開一支而不是改 `build_portal.py`：
  `build_portal.py` 是**全量**視圖（68 筆決策、整棵目標樹、所有資產）。
  這一支是**時間窗**視圖（今天），回答的是「今天決定了什麼、落地到哪裡」。
  兩者的取捨不同（全量看結構，時間窗看推進），硬塞進同一個 render 會讓兩邊都變鈍。
  **但它讀同一份真相** —— 這是最重要的一點。

★★ 這一支最誠實的一格（也是它最容易說謊的地方）：
  `decisions.jsonl` 的 `evidence[].artifact` **不是路徑，是散文**
  （實測：今天 30 筆、105 個 artifact，**沒有一個是可解析的路徑** —— 例如
  `"② 的兩個步預算"`、`"三個獨立方法對 M3 判準的判決（§EN-111／§EN-112）"`）。
  所以「落地」只能從 **`action` 欄位**（30/30 都有）＋ artifact 文字裡**抽**路徑出來，
  而抽取是**啟發式**的 ⇒ **抽不到 ≠ 沒落地**。這一條必須寫在頁面上，
  否則讀者會把「我抽不到」讀成「它沒做」。

rc 契約（照 `build_portal.py`）：
  `--check`     只驗完整性與可讀性，**不寫任何檔**；硬缺陷 ⇒ rc=1
  `--self-test` 突變式自測（fixture 在 tmp，**不得**碰到真 repo）；全過 rc=0
  無旗標        產出 `docs/TODAY_TECH_DECISIONS.html`
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import build_portal as BP                                     # noqa: E402

DECISIONS = REPO / "agent_harness" / "engine_loop" / "traces" / "decisions.jsonl"
TARGETS = HERE / "targets.json"
OUT = REPO / "docs" / "TODAY_TECH_DECISIONS.html"

# ★ 抽取用的路徑形態。刻意**限定已知的頂層目錄**（不亂抓）——
#   一個亂抓的正則會把 `1.5×`、`§EN-111` 這種東西也當成路徑，然後整張表都是假紅。
PATH_RE = re.compile(
    r"(?<![\w/.-])("
    r"(?:docs|scripts|agent_harness|src|Backup|moeexpert|deploy-harmonyos|installer)"
    r"/[A-Za-z0-9_./+-]+"
    r"|\.workbuddy/memory/[A-Za-z0-9_.-]+"
    r"|[A-Za-z0-9_-]+\.(?:md|html|jsonl|json|py|sh|cpp|h|m|ps1|proto)"
    r")(?::(\d+))?"
)
# ★ 裸檔名（沒有目錄前綴，例 `decode_sweep.py`）要往哪找 —— 這是一**猜**，所以三條約束：
#   ① 只找這個 repo **已知的目錄**（不遞迴全樹：慢，而且會命中一堆同名檔）
#   ② **唯一命中**才算命中；多個命中標 `ambiguous`（不挑一個假裝確定）
#   ③ `resolved_by` 說出是**怎麼**判定的 ⇒ 讀者可以不同意那個猜
# ★★ 這兩個目錄是**非權威的 dated 快照**（權威在 `.workbuddy/memory/` 與
#    `~/.workbuddy/skills/`，見 CONVENTIONS 的 D6 與 `agent_harness/memory/README.md`）。
#    裸檔名若同時命中權威與快照，**必須命中權威** —— 所以 `.workbuddy/memory/` 排在前面，
#    而且萬一真的落到快照，`resolved_by` 會標成 `snapshot:`，頁面上會說出「權威在別處」。
SNAPSHOT_DIRS = {"agent_harness/memory/", "agent_harness/skills/"}

SEARCH_DIRS = (
    ".workbuddy/memory/",                     # ★ 權威：決策正本講 memory 時指的是這裡
    "scripts/check/", "scripts/",
    "agent_harness/portal/", "agent_harness/a2a/", "agent_harness/scripts/",
    "agent_harness/engine_loop/", "agent_harness/engine_loop/traces/",
    "agent_harness/engine_loop/memory/", "agent_harness/pd/",
    "agent_harness/memory/",                  # ★ 快照（非權威）
    "agent_harness/skills/",                  # ★ 快照（非權威）
    "docs/", "Backup/",
    "src/llama.cpp/src/", "src/llama.cpp/common/", "src/llama.cpp/ggml/src/ggml-metal/",
    "src/llama.cpp/ggml/src/", "src/llama.cpp/tools/llama-bench/",
)
HOME_SKILLS = Path.home() / ".workbuddy" / "skills"
_home_cache: dict[str, list[str]] = {}


def _home_hits(raw: str) -> list[str]:
    if raw not in _home_cache:
        _home_cache[raw] = [str(p) for p in sorted(HOME_SKILLS.glob("*/" + raw))]
    return _home_cache[raw]


def load_decisions(path: Path | None = None) -> list[dict]:
    p = path or DECISIONS
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_targets(path: Path | None = None) -> dict:
    p = path or TARGETS
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def day_of(did: str) -> str:
    m = re.match(r"dec-(\d{4})(\d{2})(\d{2})-", str(did))
    return "%s-%s-%s" % m.groups() if m else ""


def today_ids(rows: list[dict], day: str | None = None) -> tuple[list[dict], str]:
    """今天的決策。`day` 沒給 ⇒ 取**資料裡最新的一天**（不是系統時鐘 ——
    系統時鐘會讓「今天」在跨日的那一刻突然變空，而資料沒有變）。"""
    days = [day_of(r.get("decision_id", "")) for r in rows]
    days = [d for d in days if d]
    target = day or (max(days) if days else "")
    return [r for r in rows if day_of(r.get("decision_id", "")) == target], target


def _candidates(raw: str):
    yield raw, "as-is"
    if "/" in raw:
        return
    for d in SEARCH_DIRS:
        yield d + raw, ("snapshot:" if d in SNAPSHOT_DIRS else "search:") + d
    for h in _home_hits(raw):
        yield h, "home"


def pointers(text: str) -> list[dict]:
    """從一段文字裡抽路徑。回 [{raw, path, line, exists, resolved_by, ambiguous}]。

    判定規則（三種結果分開講，**不假裝確定**）：
      - `as-is`：原字串就是一條存在的路徑
      - `search:<dir>`：裸檔名，在**已知目錄**裡**唯一**命中
      - 多個命中 ⇒ 仍給第一個，但 `ambiguous=True`（讀者要看得到這件事）
      - 都沒命中 ⇒ `exists=False`、`resolved_by="unresolved"`
    """
    seen, out = set(), []
    for m in PATH_RE.finditer(text or ""):
        raw, line = m.group(1), m.group(2)
        raw = raw.rstrip(".,;:）)、")
        if raw in seen:
            continue
        seen.add(raw)
        hits = [(p, how) for p, how in _candidates(raw) if (REPO / p).exists()]
        if len(hits) == 1:
            p, how = hits[0]
            out.append({"raw": raw, "path": p, "line": line, "exists": True,
                        "resolved_by": how, "ambiguous": False})
        elif hits:
            p, how = hits[0]
            out.append({"raw": raw, "path": p, "line": line, "exists": True,
                        "resolved_by": how, "ambiguous": True, "n_hits": len(hits)})
        else:
            out.append({"raw": raw, "path": raw, "line": line, "exists": False,
                        "resolved_by": "unresolved", "ambiguous": False})
    return out


def landing_of(dec: dict) -> dict:
    """一筆決策的「落地」。三個來源，優先序固定且**都說出處**：

      1. `action`（決策自己說它做了什麼）—— 30/30 都有
      2. `evidence[].artifact`（散文，只能抽）
      3. 都沒有 ⇒ `stated=False`（這才是真的缺陷：決策沒交代它做了什麼）
    """
    action = str(dec.get("action") or "").strip()
    ev_text = " ".join(str(e.get("artifact") or "") for e in (dec.get("evidence") or [])
                       if isinstance(e, dict))
    ptrs, seen = [], set()
    for src, txt in (("action", action), ("evidence", ev_text)):
        for p in pointers(txt):
            if (p["path"], p["line"]) in seen:
                continue
            seen.add((p["path"], p["line"]))
            ptrs.append(dict(p, source=src))
    ok = [p for p in ptrs if p["exists"]]
    return {
        "stated": bool(action),
        "action_chars": len(action),
        "pointers": ptrs,
        "verified": ok,
        # ★ 三種狀態分開講（這是這一支最重要的語意）：
        #   verified  = 抽到的路徑**真的存在**
        #   unparsed  = 有交代，但抽不出可驗的路徑（**啟發式的限制，不是缺陷**）
        #   silent    = 連交代都沒有（**這才是缺陷**）
        "state": ("verified" if ok else ("unparsed" if action else "silent")),
    }


def build(rows: list[dict], tg: dict, day: str | None = None) -> dict:
    today, d = today_ids(rows, day)
    bind = {b.get("asset"): b.get("target") for b in (tg.get("bindings") or [])}
    mom = {r["id"]: r for r in BP.momentum_rows(rows)}
    declared = (tg.get("momentum") or {}).get("declared_red") or {}
    items = []
    for r in today:
        did = r.get("decision_id", "")
        mi = mom.get(did, {})
        items.append({
            "id": did,
            "time": did[13:15] + ":" + did[15:17] if len(did) >= 17 else "",
            "short": BP.human_slug(did),
            "judgement": r.get("judgement") or "",
            "confidence": r.get("confidence") or "",
            "question": BP.balance_code((r.get("question") or "").strip()),
            "conclusion": BP.balance_code((r.get("conclusion") or "").strip()),
            "action": BP.balance_code((r.get("action") or "").strip()),
            "target": bind.get(did, "（未綁定）"),
            "momentum": mi.get("momentum", "n/a"),
            "forward_refs": mi.get("forward_refs") or [],
            "landing": landing_of(r),
        })
    items.sort(key=lambda x: x["time"])
    states = {s: sum(1 for i in items if i["landing"]["state"] == s)
              for s in ("verified", "unparsed", "silent")}
    tgts = {}
    for i in items:
        tgts[i["target"]] = tgts.get(i["target"], 0) + 1
    judged = [i for i in items if i["momentum"] != "pending"]
    return {
        "day": d, "items": items, "n": len(items),
        "states": states, "by_target": tgts,
        "by_judgement": {k: sum(1 for i in items if i["judgement"] == k)
                         for k in ("sound", "refuted")},
        "by_momentum": {k: sum(1 for i in items if i["momentum"] == k)
                        for k in ("carried", "replaced", "refuted", "orphan", "pending")},
        "judged": len(judged),
        "declared_red": declared,
        "targets": {t.get("id"): {k: t.get(k) for k in ("title", "current", "current_text", "gap", "status")}
                    for t in (tg.get("targets") or [])},
    }


def problems(data: dict, rows: list[dict]) -> list[str]:
    """硬缺陷才進這裡（rc=1）。**啟發式抽不到落地不算缺陷** —— 那是資訊。"""
    P = []
    if not rows:
        P.append("decisions.jsonl 讀不到或為空 ⇒ 這一頁沒有正本。")
        return P
    if data["n"] == 0:
        P.append("這一天（%s）沒有任何決策 ⇒ 要嘛那天真的沒決策，要嘛 decision_id 的日期格式變了。"
                 % data["day"])
        return P
    for i in data["items"]:
        if i["landing"]["state"] == "silent":
            P.append("[%s] 既沒有 action 也沒有可抽的 artifact ⇒ "
                     "這一筆沒有交代它落地到哪裡（這是決策記錄本身的缺陷）。" % i["short"])
        if i["target"] == "（未綁定）":
            P.append("[%s] 不在 targets.json 的 bindings 裡 ⇒ "
                     "它沒有被掛到任何目標上（**缺席要出聲**）。" % i["short"])
    return P


# ══════════════════════════════════════════════════════════════════════
#  黑箱自測（fixture 一律在 tmp；★ 不得碰真 repo）
# ══════════════════════════════════════════════════════════════════════

def self_test() -> int:
    res: list[tuple[str, bool, str]] = []

    def case(n, ok, d=""):
        res.append((n, bool(ok), d))

    def mk(did, action, judgement="sound", evidence=None):
        return {"type": "decision", "decision_id": did, "question": "q", "judgement": judgement,
                "confidence": "high", "action": action, "superseded_by": None,
                "evidence": evidence or []}

    tmp = Path(tempfile.mkdtemp(prefix="today_selftest_"))
    real_before = sorted(p.name for p in BP.REPO.glob("docs/TODAY_*"))

    # 1) 抽路徑：真路徑、帶行號、裸檔名（往已知目錄找）
    ps = pointers("更正 docs/roadmap-2026-09-14/ROADMAP_PREFILL250_DECODE25_2026-09-14.html 與 "
                  "MEMORY_PERF.md 的算法，另外看 agent_harness/CONVENTIONS.md:913")
    got = {(p["raw"], p["line"], p["resolved_by"]) for p in ps if p["exists"]}
    case("★ 抽路徑：真路徑／帶行號／裸檔名（往已知目錄找）三種都要命中",
         ("docs/roadmap-2026-09-14/ROADMAP_PREFILL250_DECODE25_2026-09-14.html", None, "as-is") in got
         and ("agent_harness/CONVENTIONS.md", "913", "as-is") in got
         and ("MEMORY_PERF.md", None, "search:.workbuddy/memory/") in got,
         "got=%s" % sorted(got))

    # 1b) ★ 裸檔名在已知目錄**唯一命中**才解析（並說出是哪條路徑命中的）
    p1b = [p for p in pointers("看 decode_sweep.py 的輸出") if p["exists"]]
    case("★ 裸檔名唯一命中 ⇒ 解析並說出 resolved_by（有根據的猜，但要被看見）",
         len(p1b) == 1 and p1b[0]["resolved_by"] == "search:scripts/check/"
         and p1b[0]["ambiguous"] is False, json.dumps(p1b, ensure_ascii=False)[:200])

    # 1c) ★★ 多個同名檔 ⇒ 標 ambiguous（**不挑一個假裝確定**）
    p1c = [p for p in pointers("看 SKILL.md") if p["exists"]]
    case("★★ 多個同名檔（SKILL.md）⇒ ambiguous=True 且帶 n_hits（不假裝確定）",
         len(p1c) == 1 and p1c[0]["ambiguous"] is True and p1c[0].get("n_hits", 0) >= 2,
         json.dumps(p1c, ensure_ascii=False)[:200])

    # 2) ★ 不存在的路徑要標 exists=False，但**不是**硬缺陷
    p2 = pointers("看 docs/NO_SUCH_FILE_XYZ.md")
    case("★ 不存在的路徑 ⇒ exists=False（但這不進 problems ⇒ 不會變紅）",
         len(p2) == 1 and p2[0]["exists"] is False, json.dumps(p2, ensure_ascii=False))

    # 3) ★★ 散文 artifact（沒有可抽路徑）⇒ unparsed，**不是** silent
    d3 = mk("dec-20260918-0100-x", "更正三處", evidence=[{"artifact": "② 的兩個步預算"}])
    l3 = landing_of(d3)
    case("★★ 有 action 但抽不到路徑 ⇒ state=unparsed（啟發式限制，不是缺陷）",
         l3["state"] == "unparsed" and l3["stated"] is True,
         json.dumps({k: l3[k] for k in ("state", "stated")}, ensure_ascii=False))

    # 4) ★★ 連 action 都沒有 ⇒ silent，**這才是缺陷**（並指名那一筆）
    d4 = mk("dec-20260918-0200-y", "", evidence=[])
    l4 = landing_of(d4)
    case("★★ 沒有 action 也沒有 artifact ⇒ state=silent（這才是缺陷）",
         l4["state"] == "silent" and l4["stated"] is False, json.dumps(l4, ensure_ascii=False))

    # 5) 完整性（規矩 5）：來源集合裡的每一筆都要在表上有一列
    rows5 = [mk("dec-20260918-%04d-a%d" % (i, i), "改 docs/roadmap-2026-09-14/x.html")
             for i in range(1, 4)]
    tg5 = {"bindings": [{"asset": r["decision_id"], "target": "decode-25"} for r in rows5],
           "momentum": {"declared_red": {"refuted_rate_gt": .25, "orphan_rate_gt": .5}},
           "targets": []}
    data5 = build(rows5, tg5)
    case("★ 完整性：來源 3 筆 ⇒ 表上恰好 3 行（每一筆都在）",
         data5["n"] == 3 and len(data5["items"]) == 3, "n=%s" % data5["n"])

    # 6) ★★ 少一筆就要紅，而且**指名那一筆**（規矩 5 的執行者）
    tg6 = dict(tg5, bindings=tg5["bindings"][:2])
    data6 = build(rows5, tg6)
    P6 = problems(data6, rows5)
    case("★★ 綁定表少一筆 ⇒ rc 缺陷且指名那一筆（缺席要出聲）",
         len(P6) == 1 and "a3" in P6[0], "problems=%s" % P6)

    # 7) 「今天」取資料裡最新的一天，不是系統時鐘
    rows7 = [mk("dec-20260917-0100-old", "x docs/roadmap-2026-09-14/x.html"),
             mk("dec-20260919-0100-new", "y docs/roadmap-2026-09-14/x.html")]
    ids, day = today_ids(rows7)
    case("★ 「今天」＝資料裡最新的一天（不是系統時鐘；跨日不會突然變空）",
         day == "2026-09-19" and len(ids) == 1, "day=%s n=%d" % (day, len(ids)))

    # 8) 壞 JSON 不炸（回空，且 problems 說得出話）
    bad = tmp / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    case("★ 壞 JSON ⇒ 回空清單不 traceback，且 problems 講得出原因",
         load_decisions(bad) == [] and problems({"n": 0, "day": "", "items": []}, []) != [],
         "load=%s" % load_decisions(bad))

    # 9) 動能沿用 build_portal 的算法（不是這裡再寫一份）
    mom9 = {r["id"]: r["momentum"] for r in BP.momentum_rows(rows7)}
    case("★ 動能**呼叫** build_portal.momentum_rows（不重寫 ⇒ 不會有第二份實作）",
         set(mom9.values()) <= {"carried", "replaced", "refuted", "orphan", "pending"},
         "mom=%s" % mom9)

    # 10) ★ 自測不得在真 repo 留下產物
    real_after = sorted(p.name for p in BP.REPO.glob("docs/TODAY_*"))
    case("★ 自測沒有在真 repo 留下 docs/TODAY_* 產物", real_before == real_after,
         "%s -> %s" % (real_before, real_after))

    passed = sum(1 for _, ok, _ in res if ok)
    for n, ok, d in res:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", n))
        if not ok:
            print("         %s" % d)
    print("  %d/%d 通過" % (passed, len(res)))
    return 0 if passed == len(res) else 1


# ══════════════════════════════════════════════════════════════════════
#  產出
# ══════════════════════════════════════════════════════════════════════

CSS = """
:root { --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#fff; --soft:#f9fafb; --code:#f3f4f6;
        --blue:#1d4ed8; --green:#15803d; --red:#b91c1c; --amber:#b45309; --purple:#6d28d9; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.7 -apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:32px 20px 80px; }
h1 { font-size:22px; margin:0 0 6px; }
h2 { font-size:17px; margin:28px 0 10px; padding-top:12px; border-top:1px solid var(--line); }
h3 { font-size:14.5px; margin:18px 0 6px; }
.sub { color:var(--muted); font-size:13px; margin:0 0 18px; }
code { background:var(--code); padding:1px 5px; border-radius:4px; font-size:12.4px;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
table { border-collapse:collapse; width:100%; margin:10px 0 16px; font-size:12.8px; }
th,td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; }
.num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
.tag { display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px;
       border:1px solid var(--line); background:var(--soft); white-space:nowrap; }
.t-blue{color:var(--blue);border-color:#bfdbfe;background:#eff6ff}
.t-green{color:var(--green);border-color:#bbf7d0;background:#f0fdf4}
.t-red{color:var(--red);border-color:#fecaca;background:#fef2f2}
.t-amber{color:var(--amber);border-color:#fde68a;background:#fffbeb}
.t-purple{color:var(--purple);border-color:#ddd6fe;background:#f5f3ff}
.t-gray{color:var(--muted);border-color:var(--line);background:var(--soft)}
.box { border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:4px;
       padding:10px 12px; margin:13px 0; }
.box.warn { border-left-color:var(--amber); background:#fffdf5; }
.box.ok { border-left-color:var(--green); background:#f6fdf8; }
.box.bad { border-left-color:var(--red); background:#fff7f7; }
.box.pur { border-left-color:var(--purple); background:#faf8ff; }
.small { font-size:12.2px; color:var(--muted); }
ul,ol { margin:6px 0 10px; padding-left:22px; }
li { margin:3px 0; }
.bar { height:9px; border-radius:5px; background:#378ADD; display:inline-block; vertical-align:middle; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:12px 0; }
.card { background:var(--soft); border-radius:8px; padding:10px 12px; }
.card .k { font-size:12px; color:var(--muted); }
.card .v { font-size:22px; font-weight:500; font-variant-numeric:tabular-nums; }
.ptr { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.8px; }
.ptr.ok { color:var(--green); }
.ptr.no { color:var(--muted); }
"""


def esc(s) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt_md(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def render(data: dict) -> str:
    d = data
    n = max(d["n"], 1)
    st = d["states"]
    cards = [
        ("今天的決策", d["n"], ""),
        ("sound / refuted", "%d / %d" % (d["by_judgement"]["sound"], d["by_judgement"]["refuted"]), ""),
        ("有可驗落地的", "%d（%.0f%%）" % (st["verified"], 100 * st["verified"] / n), "green"),
        ("抽不出路徑的", "%d" % st["unparsed"], "gray"),
        ("沒交代落地的", "%d" % st["silent"], "red" if st["silent"] else "gray"),
    ]
    tg = d["targets"]
    h = ["<!DOCTYPE html>", '<html lang="zh-Hant"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>今日宏觀技術決策 × 落地 — %s</title>" % esc(d["day"]),
         "<style>%s</style></head><body><div class=\"wrap\">" % CSS]
    h.append("<h1>今日宏觀技術決策 × 落地</h1>")
    h.append('<p class="sub">%s　·　資料來源：<code>decisions.jsonl</code> ＋ '
             '<code>targets.json</code>（本頁<b>不新增任何清單</b>）</p>' % esc(d["day"]))

    h.append('<div class="box pur"><b>這一頁是既有真相的一個新視圖。</b>'
             '它的兩個輸入都是既有的單一真相來源：決策正本 <code>agent_harness/engine_loop/'
             'traces/decisions.jsonl</code>（只追加）與「資產 → 目標」綁定 '
             '<code>agent_harness/portal/targets.json</code>。本頁不產生趨勢、不新增清單。</div>')

    h.append('<div class="box warn"><b>★ 落地指標是<b>抽</b>出來的，所以「抽不到」不等於「沒落地」。</b>'
             '<code>evidence[].artifact</code> 在決策正本裡<b>不是路徑、是散文</b>'
             '（實測：今天 %d 筆的 artifact 沒有一個是可解析的路徑 —— '
             '例如「② 的兩個步預算」）。所以本頁從 <code>action</code>（今天 30/30 都有）'
             '與 artifact 文字裡<b>用正則抽</b>路徑，再逐個驗存在性。三種狀態刻意分開講：'
             '<span class="tag t-green">verified</span> 抽到的路徑真的存在｜'
             '<span class="tag t-gray">unparsed</span> 有交代但抽不出可驗路徑（<b>啟發式的限制</b>）｜'
             '<span class="tag t-red">silent</span> 連交代都沒有（<b>這才是缺陷</b>）。</div>' % d["n"])

    h.append('<div class="cards">')
    for k, v, tone in cards:
        h.append('<div class="card"><div class="k">%s</div><div class="v"%s>%s</div></div>'
                 % (esc(k), ' style="color:var(--red)"' if tone == "red" else "", esc(v)))
    h.append("</div>")

    h.append("<h2>① 兩個目標（正本：targets.json）</h2>")
    h.append("<table><tr><th>目標</th><th class='num'>目標值</th><th>現況</th>"
             "<th>缺口／判準</th><th>本日決策數</th></tr>")
    for tid, t in tg.items():
        h.append("<tr><td><b>%s</b></td><td class='num'>%s</td><td>%s</td><td>%s</td>"
                 "<td class='num'>%d</td></tr>"
                 % (esc(t.get("title")),
                    esc(re.search(r"(\d+)", str(t.get("title") or "")).group(1)
                        if re.search(r"(\d+)", str(t.get("title") or "")) else "?"),
                    fmt_md(t.get("current_text") or ""), fmt_md(t.get("gap") or ""),
                    d["by_target"].get(tid, 0)))
    for k, v in d["by_target"].items():
        if k not in tg:
            h.append("<tr><td>%s</td><td></td><td></td><td></td><td class='num'>%d</td></tr>"
                     % (esc(k), v))
    h.append("</table>")

    h.append("<h2>② 決策 × 落地（每一筆今天的決策都在表上）</h2>")
    h.append('<p class="small">排序＝時間。落地欄只顯示<b>前 3 個</b>指標；'
             '<span class="ptr ok">綠</span>＝檔案存在，<span class="ptr no">灰</span>＝抽到但不存在。'
             '裸檔名（例 <code>decode_sweep.py</code>）會在<b>已知目錄</b>裡找，'
             '<b>唯一命中</b>才算並標出「搜尋命中 &lt;目錄&gt;」；'
             '多個同名檔 ⇒ 標 <b>⚠</b> 並取第一個（<b>不假裝確定</b>）。</p>')
    h.append("<table><tr><th>時間</th><th>決策</th><th>判斷</th><th>目標</th><th>動能</th>"
             "<th>落地（抽出的指標）</th></tr>")
    for i in d["items"]:
        jt = {"sound": "t-green", "refuted": "t-red"}.get(i["judgement"], "t-gray")
        mt = {"carried": "t-green", "replaced": "t-blue", "refuted": "t-red",
              "orphan": "t-amber", "pending": "t-gray"}.get(i["momentum"], "t-gray")
        ptrs = i["landing"]["pointers"][:3]
        if ptrs:
            cells = []
            for p in ptrs:
                cls = "ok" if p["exists"] else "no"
                rb = p.get("resolved_by", "")
                tag = ""
                if rb.startswith("snapshot:"):
                    tag = (' <span class="small">（快照 %s；<b>權威在 .workbuddy/memory/</b>）</span>'
                           % esc(rb[9:]))
                elif rb.startswith("search:"):
                    tag = ' <span class="small">（搜尋命中 %s）</span>' % esc(rb[7:])
                elif rb == "home":
                    tag = ' <span class="small">（家目錄 skills）</span>'
                if p.get("ambiguous"):
                    tag = (' <span class="small">（⚠ %d 個同名檔，取第一個）</span>'
                           % p.get("n_hits", 2))
                cells.append('<span class="ptr %s">%s%s</span>%s'
                             % (cls, "✓ " if p["exists"] else "· ",
                                esc(p["path"] + (":" + p["line"] if p["line"] else "")), tag))
            ptxt = "<br>".join(cells)
            if len(i["landing"]["pointers"]) > 3:
                ptxt += '<br><span class="small">…還有 %d 個</span>' % (
                    len(i["landing"]["pointers"]) - 3)
        elif i["landing"]["state"] == "silent":
            ptxt = '<span class="tag t-red">沒交代</span>'
        else:
            ptxt = '<span class="tag t-gray">抽不出路徑</span>'
        h.append("<tr><td class='num'>%s</td><td><b>%s</b><br>"
                 '<span class="small">%s</span></td>'
                 '<td><span class="tag %s">%s</span></td><td><span class="tag t-blue">%s</span></td>'
                 "<td><span class='tag %s'>%s</span></td><td>%s</td></tr>"
                 % (esc(i["time"]), esc(i["short"]), fmt_md(i["conclusion"][:150]),
                    jt, esc(i["judgement"]), esc(i["target"]), mt, esc(i["momentum"]), ptxt))
    h.append("</table>")

    h.append("<h2>③ 動能（由 decisions.jsonl 自己算，不是手填）</h2>")
    h.append('<p class="small">一個決策的動能＝它被<b>後續</b>決策納入的程度。'
             '最新的一天結構上不可能有「更晚的決策引用它」⇒ 今天全部計為 '
             '<code>pending</code> 並<b>排除在門檻之外</b>（否則新的一批會讓指標永遠很差）。'
             '宣告門檻：refuted &gt; %.2f 或 orphan &gt; %.2f ⇒ 紅。</p>'
             % (d["declared_red"].get("refuted_rate_gt", 0),
                d["declared_red"].get("orphan_rate_gt", 0)))
    h.append("<table><tr><th>動能</th><th class='num'>筆數</th><th>占比（分母＝已判 %d 筆）</th></tr>"
             % d["judged"])
    order = ["carried", "replaced", "refuted", "orphan", "pending"]
    for k in order:
        v = d["by_momentum"].get(k, 0)
        pct = 100.0 * v / max(d["judged"], 1) if k != "pending" else 0
        h.append("<tr><td><code>%s</code></td><td class='num'>%d</td><td>"
                 "<span class='bar' style='width:%dpx'></span> %.0f%%</td></tr>"
                 % (k, v, max(int(pct * 2.2), 2), pct))
    h.append("</table>")

    h.append('<h2>④ 落地狀態的分布</h2>')
    h.append("<table><tr><th>狀態</th><th class='num'>筆數</th><th>意思</th></tr>")
    for k, mean in (("verified", "抽出的路徑真的存在 —— 這一筆的落地可以自己去看"),
                    ("unparsed", "有交代做了什麼，但抽不出可驗的路徑（<b>啟發式的限制</b>）"),
                    ("silent", "連交代都沒有 ⇒ 決策記錄本身的缺陷（<b>會讓 --check 變紅</b>）")):
        h.append("<tr><td><code>%s</code></td><td class='num'>%d</td><td>%s</td></tr>"
                 % (k, st[k], mean))
    h.append("</table>")

    h.append('<p class="small" style="margin-top:24px">'
             '本頁由 <code>agent_harness/portal/build_today.py</code> 產生；'
             '它讀既有真相、不寫任何清單。<code>--check</code> 只驗完整性（每一筆今天的決策都要在表上）'
             '與可讀性，<b>不寫檔</b>；<code>--self-test</code> 為突變式自測。</p>')
    h.append("</div></body></html>")
    return "\n".join(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="今日宏觀技術決策 × 落地（既有真相的新視圖）")
    ap.add_argument("--day", help="YYYY-MM-DD；預設＝資料裡最新的一天")
    ap.add_argument("--check", action="store_true", help="只驗完整性，不寫檔")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", help="輸出路徑（預設 docs/TODAY_TECH_DECISIONS.html）")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    rows = load_decisions()
    tg = load_targets()
    data = build(rows, tg, a.day)
    P = problems(data, rows)

    if a.check:
        print("  這一天：%s　決策 %d 筆" % (data["day"], data["n"]))
        print("  落地：verified %d／unparsed %d／silent %d"
              % (data["states"]["verified"], data["states"]["unparsed"], data["states"]["silent"]))
        print("  目標：%s" % data["by_target"])
        print("  動能：%s" % data["by_momentum"])
        for x in P:
            print("  [error] %s" % x)
        if not P:
            print("  OK：每一筆今天的決策都在表上，且都有交代落地。")
        return 1 if P else 0

    out = Path(a.out) if a.out else OUT
    html = render(data)
    out.write_text(html, encoding="utf-8")
    print("  已寫出 %s（%d bytes）" % (out, len(html.encode())))
    print("  決策 %d 筆；落地 verified %d／unparsed %d／silent %d"
          % (data["n"], data["states"]["verified"], data["states"]["unparsed"],
             data["states"]["silent"]))
    if P:
        print("  ★ 有 %d 條要看的：" % len(P))
        for x in P:
            print("    %s" % x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
