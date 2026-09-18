#!/usr/bin/env python3
"""build_portal.py — 把四類資產收進同一頁：目標、宏觀決策、驗證閘門、資產盤點。

它回答三個問題，每一題都必須有**可重跑的指令**當答案：

  1. 我們在追求什麼、已經決定了什麼？（`goals.json` ＋ `traces/decisions.jsonl`）
  2. 驗證是不是真的跑過、而且能重跑？（`gates.json`；**實跑**，不是宣告）
  3. 資產有多少、有沒有在動？（記憶／skill／代碼／軌跡／衍生物／證據）

設計上的三條硬規矩（都是本 repo 已付過學費的）：

  * **單一真相來源**：閘門清單只在 `gates.json`。portal 顯示的指令與「一鍵復現」跑的是同一份，
    所以不存在「文件寫的指令已經不是實際跑的那條」這種漂移。
  * **不假裝**：`cost=heavy` 的閘門（需要 GPU／build／docker／模型端點）**不會**被自動跑，
    portal 只列指令與前題並標成「未執行」，**不會畫成綠燈**。把跑不動的閘門畫綠，比沒有閘門更糟。
  * **缺席要出聲**：目標樹每一節要嘛有可解析指針、要嘛顯式標記證據形態；閘門每條要交代
    「證什麼／不證什麼」。缺這些不是「沒填」，是**沉默**，而沉默與沒人管在讀者面前同形
    （CONVENTIONS.md:7-14）。

用法：

    python3 agent_harness/portal/build_portal.py              # 跑快速閘門 + 產 portal（預設）
    python3 agent_harness/portal/build_portal.py --gates-only # 只跑閘門並印表；任何紅燈 ⇒ rc=1
    python3 agent_harness/portal/build_portal.py --no-gates   # 不跑閘門（離線／趕時間）
    python3 agent_harness/portal/build_portal.py --check      # 只驗註冊表與目標樹指針，不寫任何檔
    python3 agent_harness/portal/build_portal.py --self-test  # 陰性對照（含「--check 不寫檔」）

產物（三個，全部在 repo 內）：
    docs/AGENT_HARNESS_PORTAL.html     自帶樣式與腳本，離線可開
    agent_harness/portal/data.json     本輪的機器可讀快照（下一輪的「上一次」）
    agent_harness/portal/history.jsonl **只追加**的量化歷史（趨勢圖的來源）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── 路徑 ────────────────────────────────────────────────────────────────────────
# PA_PORTAL_REPO 讓 --self-test 指到 fixture，而不是真的 repo。
REPO = Path(os.environ.get("PA_PORTAL_REPO", Path(__file__).resolve().parents[2])).resolve()
HOME = Path(os.environ.get("PA_PORTAL_HOME", Path.home()))
PORTAL_DIR = REPO / "agent_harness" / "portal"
GATES_JSON = PORTAL_DIR / "gates.json"
GOALS_JSON = PORTAL_DIR / "goals.json"
DATA_OUT = PORTAL_DIR / "data.json"
HISTORY = PORTAL_DIR / "history.jsonl"
HTML_OUT = REPO / "docs" / "AGENT_HARNESS_PORTAL.html"

MEM_DIR = REPO / ".workbuddy" / "memory"
SKILL_DIR = HOME / ".workbuddy" / "skills"
TRACES = REPO / "agent_harness" / "engine_loop" / "traces"
SELF_REL = "agent_harness/portal/build_portal.py"

# ── 資產掃描的範圍 ──────────────────────────────────────────────────────────────
CODE_EXT = {".py", ".sh", ".bash", ".zsh", ".ts", ".js", ".mjs", ".cjs",
            ".cpp", ".cc", ".c", ".h", ".hpp", ".hxx", ".metal", ".swift",
            ".ps1", ".mm", ".cu", ".go", ".rs", ".java", ".lua", ".sql"}
SKIP_DIR_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules",
                  ".mypy_cache", ".pytest_cache", "site-packages", ".build",
                  "CMakeFiles", "Backup", ".cache"}

# 長前綴優先（先匹配者勝）。每一檔只算一次。
AREA_RULES: list[tuple[str, str]] = [
    ("src/llama.cpp", "src/llama.cpp（引擎核心 C++/Metal）"),
    ("src/", "src/（其他）"),
    ("scripts/", "scripts/（量測、閘門、server）"),
    ("agent_harness/engine_loop", "agent_harness/engine_loop"),
    ("agent_harness/tb_loop", "agent_harness/tb_loop"),
    ("agent_harness/shared", "agent_harness/shared"),
    ("agent_harness/portal", "agent_harness/portal"),
    ("agent_harness/", "agent_harness/（其餘）"),
    ("app/", "app/（CGC 上層）"),
    ("tools/", "tools/"),
    ("deploy-harmonyos/", "deploy-harmonyos/"),
    ("hf-space-deploy/", "hf-space-deploy/"),
    ("moeexpert/", "moeexpert/"),
    ("CGC-main/", "CGC-main/"),
    ("CGC_Phase2/", "CGC_Phase2/"),
    # `versions/` 是歷史存檔（3835 檔 / 1.55M 行），**不是現行代碼**。
    # 它一定要單獨成一個桶並被排除在 headline LOC 之外 —— 否則「現行代碼 3.07M 行」
    # 這個數字有一半是舊版複本，而那種數字看起來很厲害、卻什麼都沒說。
    ("versions/", "versions/（歷史存檔，不計入現行）"),
]

# 這些區域是存檔，不算「現行代碼」
ARCHIVE_AREAS = {"versions/（歷史存檔，不計入現行）"}


def area_of(rel: str) -> str:
    for prefix, label in AREA_RULES:
        if rel.startswith(prefix):
            return label
    return "其他"


# ── 小工具 ─────────────────────────────────────────────────────────────────────
def sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            if limit is not None:
                chunk = chunk[: max(0, limit - n)]
                if not chunk:
                    break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest()


def count_lines(p: Path) -> int:
    """二進位安全、分塊計數（不把整個檔讀進記憶體）。"""
    n = 0
    with p.open("rb") as fh:
        first = fh.read(8192)
        if b"\x00" in first:
            return 0  # 二進位，不算 LOC
        n += first.count(b"\n")
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def run(cmd: list[str], timeout: int = 300, cwd: Path | None = None,
        env: dict | None = None) -> tuple[int, float, str]:
    t0 = time.time()
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(cwd or REPO), env=e)
        rc, out = r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, time.time() - t0, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e2:
        return 127, time.time() - t0, f"command not found: {e2}"
    return rc, time.time() - t0, out


def hsize(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def plain(t) -> str:
    """純文字化（給 HTML **屬性**用，例如 `title="…"`）：屬性裡不能放標籤。

    `mdx()` 會轉出 `<b>`／`<code>`；那是 body 用的。屬性只能放純文字，
    所以這一條要把 markdown 記號**拿掉**而不是轉換 —— 否則 tooltip 會顯示兩個星號。
    """
    t = str("" if t is None else t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    return t.replace("`", "").replace('"', "'")


def balance_code(t: str) -> str:
    r"""把 `<code>` 標記平衡化：數量不等就全部拿掉，只留文字。

    **這一條的來歷要說清楚，因為它一開始是被一個假症狀引出來的**：我曾經量到
    「`<code>` 214 個、`</code>` 213 個」，於是寫下「來源語料有不成對的標籤」這個判斷。
    那個判斷是錯的 —— 我是在一份**內嵌 JSON 的 HTML** 上做 `count('</code>')`，
    而 JSON 會把 `</` 轉義成 `<\/`（防 `</script>` 提前結束），所以 payload 裡的 `</code>`
    本來就是 0。**少的那一個是我的量測方法造出來的，不是資料造出來的。**

    修法是把 payload 還原轉義後再數（214/214，本來就是平衡的）。
    這個函式留著當**防禦**（便宜，且另一條線隨時可能寫出不成對的標籤），
    但它不是任何已知缺陷的修法。**一般化：內嵌 JSON 的 HTML 不能直接數標籤。**
    """
    t = str("" if t is None else t)
    if t.count("<code>") != t.count("</code>"):
        t = t.replace("<code>", "").replace("</code>", "")
    return t


def clip(s, n: int) -> str:
    """截斷文字，**但不切斷 HTML 標籤**。

    為什麼需要它：語料本來就混著 markdown 與 HTML（decision 的 `conclusion` 就含
    `<code>…</code>`），而表格只顯示前 N 字 ⇒ 截斷點**可能**落在標籤中間，
    產生一個永遠不會被關起來的 `<code>`（它會把後面所有文字都吃進等寬區塊）。

    **這是潛在風險，不是已發生的缺陷**：本輪的 `<code>`／`</code>` 差 1 是我的量測假象
    （見 `balance_code` 的說明），實際上是 214/214。留著它是因為代價近乎零，
    而那個症狀一旦發生，看得出來的人不多。
    """
    t = str("" if s is None else s)
    if len(t) <= n:
        return balance_code(t)
    t = t[:n]
    if t.rfind("<") > t.rfind(">"):
        t = t[: t.rfind("<")]
    return balance_code(t + "…")


def mdx(s) -> str:
    """把 markdown 的 **粗體**／`code` 轉成 HTML。

    為什麼需要它：這一頁的文字有兩個來源 —— 一份是我自己寫的 JSON，另一份是**從 markdown
    衍生**的（CONVENTIONS 的義務行、lessons 的 rule、decisions 的結論）。兩邊都可能帶 `**`，
    而 `**` 在 HTML 裡就是兩個星號：**看起來像漏排版，實際上是輸出少了一層轉換**。
    （本檔第一版就是這樣把 `**` 直接吐進 HTML 的。）
    """
    t = balance_code(str("" if s is None else s))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    # 落單的 `**`（來源 markdown 自己就不成對，例如 CONVENTIONS 的 "smoke 未跑**（…"）
    # 留在畫面上就是兩個星號 ⇒ 去掉。成對的已經被上面那條轉掉了。
    t = t.replace("**", "")
    return t


def hesc(s) -> str:
    """HTML 轉義。**這一個必須在 Python 這一側** —— f-string 裡的 `{...}` 是 Python 求值，
    不是 JS；在那裡呼叫 JS 的函式會是 NameError（本檔第一版就是這樣壞的）。"""
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ── 輸入指紋：同一指紋 ⇒ 可比較；不同 ⇒ 不可比 ─────────────────────────────────
def fingerprint() -> dict:
    parts, files = [], []
    for rel in ("agent_harness/engine_loop/traces/lessons.jsonl",
                "agent_harness/engine_loop/traces/decisions.jsonl",
                "agent_harness/engine_loop/traces/episodes.jsonl",
                "agent_harness/CONVENTIONS.md",
                "agent_harness/portal/gates.json",
                "agent_harness/portal/goals.json"):
        p = REPO / rel
        if p.exists():
            h = sha256_file(p)[:12]
            parts.append(f"{rel}:{h}")
            files.append({"path": rel, "sha256_12": h, "bytes": p.stat().st_size})
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return {"digest": digest, "inputs": files}


def git_info() -> dict:
    def g(*a: str) -> str:
        rc, _, out = run(["git", *a], timeout=30)
        return out.strip() if rc == 0 else ""
    head = g("rev-parse", "HEAD")
    status = g("status", "--porcelain", "-uall")
    dirty = [l for l in status.splitlines() if l.strip()]
    return {
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "head_short": head[:9],
        "subject": g("log", "-1", "--format=%s"),
        "dirty": len(dirty),
        "dirty_sample": dirty[:8],
    }


# ── 資產盤點 ───────────────────────────────────────────────────────────────────
def scan_memory() -> dict:
    files = []
    if MEM_DIR.is_dir():
        for p in sorted(MEM_DIR.glob("*.md")):
            txt = p.read_text(encoding="utf-8", errors="replace")
            files.append({
                "name": p.name,
                "bytes": p.stat().st_size,
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                "lines": txt.count("\n") + 1,
                "sections": sum(1 for l in txt.splitlines() if l.startswith("## ")),
            })
    return {"files": files, "count": len(files),
            "bytes": sum(f["bytes"] for f in files),
            "lines": sum(f["lines"] for f in files)}


def scan_skills() -> dict:
    items = []
    if SKILL_DIR.is_dir():
        for d in sorted(SKILL_DIR.iterdir()):
            f = d / "SKILL.md"
            if not f.is_file():
                continue
            txt = f.read_text(encoding="utf-8", errors="replace")
            desc = ""
            for line in txt.splitlines():
                if line.startswith("description:"):
                    desc = line[12:].strip()[:150]
                    break
            items.append({
                "name": d.name,
                "bytes": f.stat().st_size,
                "lines": txt.count("\n") + 1,
                "sha8": sha256_file(f)[:8],
                "desc": desc,
            })
    return {"items": items, "count": len(items),
            "bytes": sum(i["bytes"] for i in items),
            "lines": sum(i["lines"] for i in items)}


def scan_code() -> dict:
    agg: dict[str, dict] = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_PARTS]
        for fn in files:
            if Path(fn).suffix.lower() not in CODE_EXT:
                continue
            full = Path(root) / fn
            rel = str(full.relative_to(REPO))
            try:
                loc = count_lines(full)
                nbytes = full.stat().st_size
            except OSError:
                continue
            a = agg.setdefault(area_of(rel), {"area": area_of(rel), "files": 0, "loc": 0, "bytes": 0})
            a["files"] += 1
            a["loc"] += loc
            a["bytes"] += nbytes
    for a in agg.values():
        a["archive"] = a["area"] in ARCHIVE_AREAS
    areas = sorted(agg.values(), key=lambda x: -x["loc"])
    live = [a for a in areas if not a["archive"]]
    return {"areas": areas,
            "files": sum(a["files"] for a in areas),
            "loc": sum(a["loc"] for a in areas),
            "bytes": sum(a["bytes"] for a in areas),
            "files_live": sum(a["files"] for a in live),
            "loc_live": sum(a["loc"] for a in live),
            "bytes_live": sum(a["bytes"] for a in live)}


def scan_dir(p: Path) -> dict:
    n = b = 0
    if p.is_dir():
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_PARTS]
            for fn in files:
                try:
                    b += (Path(root) / fn).stat().st_size
                    n += 1
                except OSError:
                    pass
    return {"files": n, "bytes": b}


def scan_traces() -> dict:
    import collections
    episodes = load_jsonl(TRACES / "episodes.jsonl")
    decisions = load_jsonl(TRACES / "decisions.jsonl")
    lessons = load_jsonl(TRACES / "lessons.jsonl")
    by_class = dict(collections.Counter(l.get("class", "?") for l in lessons))
    usable = sum(1 for e in episodes if e.get("usable_as_evidence"))
    return {
        "episodes": len(episodes),
        "episodes_usable": usable,
        "decisions": len(decisions),
        "lessons": len(lessons),
        "lessons_by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "bytes": sum((TRACES / f).stat().st_size
                     for f in ("episodes.jsonl", "decisions.jsonl", "lessons.jsonl")
                     if (TRACES / f).exists()),
        "schema": (TRACES / "schema").is_dir(),
    }


def scan_derived() -> dict:
    memdir = REPO / "agent_harness" / "engine_loop" / "harness_engine" / "memories" / "engine"
    n_mem = len(list(memdir.glob("*.md"))) if memdir.is_dir() else 0
    out = {"engine_memories": n_mem}
    for name, rel in (("sft_pi_train", "agent_harness/engine_loop/sft_pi/train.jsonl"),
                      ("sft_pi_valid", "agent_harness/engine_loop/sft_pi/valid.jsonl"),
                      ("sft_prime_train", "agent_harness/engine_loop/sft_prime/train.jsonl"),
                      ("sft_prime_valid", "agent_harness/engine_loop/sft_prime/valid.jsonl")):
        p = REPO / rel
        out[name] = len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0
    return out


def scan_corpus() -> dict:
    docs = sorted(p.name for p in (REPO / "docs").glob("*.html"))
    md = sorted(p.name for p in (REPO / "docs").glob("*.md"))
    return {
        "whitepapers": len(docs),
        "latest_whitepaper": docs[-1] if docs else "",
        "docs_md": len(md),
        "evidence": scan_dir(REPO / "agent_harness" / "shared" / "evidence"),
        "backup": scan_dir(REPO / "Backup"),
    }


def scan_assets() -> dict:
    return {
        "memory": scan_memory(),
        "skills": scan_skills(),
        "code": scan_code(),
        "traces": scan_traces(),
        "derived": scan_derived(),
        "corpus": scan_corpus(),
    }


# ── 決策：分組、取代鏈 ─────────────────────────────────────────────────────────
def human_slug(decision_id: str) -> str:
    tail = re.sub(r"^dec-\d{8}-\d{4}-", "", decision_id)
    return tail.replace("-", " ")


def evidence_bucket(art: str) -> str:
    a = str(art)
    for prefix, label in (("src/", "src（引擎核心）"), ("scripts/", "scripts（量測/閘門）"),
                          ("agent_harness/", "agent_harness（harness）"),
                          ("docs/", "docs（白皮書）"), ("Backup/", "Backup（觀測/log）")):
        if a.startswith(prefix):
            return label
    return "其他"


def build_decisions(rows: list[dict]) -> dict:
    import collections
    items = []
    for r in rows:
        did = r.get("decision_id", "")
        ev = r.get("evidence") or []
        arts = [e.get("artifact") for e in ev if isinstance(e, dict) and e.get("artifact")]
        m = re.match(r"dec-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})-", did)
        items.append({
            "id": did,
            "day": f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "?",
            "time": f"{m.group(4)}:{m.group(5)}" if m else "",
            "short": human_slug(did),
            "question": balance_code((r.get("question") or "").strip()),
            "conclusion": balance_code((r.get("conclusion") or "").strip()),
            "judgement": r.get("judgement") or "",
            "confidence": r.get("confidence") or "",
            "action": balance_code((r.get("action") or "").strip()),
            "superseded_by": r.get("superseded_by"),
            "n_evidence": len(ev),
            "artifacts": arts[:6],
            "bucket": evidence_bucket(arts[0]) if arts else "無證據",
            "reasoning_chars": len(str(r.get("reasoning") or "")),
        })
    items.sort(key=lambda x: x["id"])
    by_id = {i["id"]: i for i in items}
    superseded = [i for i in items if i["superseded_by"]]
    # 取代鏈：從沒有「被指向」的起點往下走
    pointed = {i["superseded_by"] for i in superseded}
    chains = []
    for start in [i for i in items if i["id"] not in pointed]:
        chain, cur, seen = [start["id"]], start.get("superseded_by"), {start["id"]}
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = by_id[cur].get("superseded_by")
        if len(chain) > 1:
            chains.append([{"id": c, "short": by_id[c]["short"],
                            "judgement": by_id[c]["judgement"]} for c in chain])
    counts = {
        "total": len(items),
        "sound": sum(1 for i in items if i["judgement"] == "sound"),
        "refuted": sum(1 for i in items if i["judgement"] == "refuted"),
        "superseded": len(superseded),
        "high": sum(1 for i in items if i["confidence"] == "high"),
        "no_evidence": sum(1 for i in items if not i["artifacts"]),
        "buckets": dict(collections.Counter(i["bucket"] for i in items)),
        "days": dict(sorted(collections.Counter(i["day"] for i in items).items())),
        "last_day": max((i["day"] for i in items), default=""),
    }
    return {"items": items, "counts": counts, "chains": chains, "by_id": by_id}


# ── 判準：憲章條文數與 lesson 分類 ─────────────────────────────────────────────
def build_charter() -> dict:
    p = REPO / "agent_harness" / "CONVENTIONS.md"
    sections, total = {}, 0
    if p.exists():
        cur = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^## ([A-E])\.\s*(.+)$", line)
            if m:
                cur = m.group(1)
                sections[cur] = {"key": cur, "title": m.group(2).strip(), "count": 0}
                continue
            m2 = re.match(r"^\*\*([A-E])(\d+)｜(.+)$", line)
            if m2:
                total += 1
                if m2.group(1) in sections:
                    sections[m2.group(1)]["count"] += 1
    return {"sections": [sections[k] for k in sorted(sections)], "total": total}


def extract_obligations() -> list[dict]:
    """從憲章抓出「未完成的義務」——衍生而非手抄，所以它自己不會漂。"""
    p = REPO / "agent_harness" / "CONVENTIONS.md"
    out = []
    if not p.exists():
        return out
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    KEYS = ("未完成的義務", "不得聲稱", "尚未執行", "仍未執行", "未跑", "記在 E2 頭上")
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s.startswith("-") and not s.startswith("*"):
            continue
        if any(k in s for k in KEYS):
            out.append({"line": i, "text": clip(s.lstrip("-* ").strip(), 400)})
    return out


# ── 閘門 ───────────────────────────────────────────────────────────────────────
def load_gates() -> dict:
    return json.loads(GATES_JSON.read_text(encoding="utf-8"))


def registry_problems(reg: dict) -> list[str]:
    problems = []
    seen = set()
    for g in reg.get("gates", []):
        gid = g.get("id", "")
        if not gid:
            problems.append("閘門沒有 id")
        if gid in seen:
            problems.append(f"閘門 id 重複: {gid}")
        seen.add(gid)
        for field in ("group", "title", "cmd", "expect", "proves", "not_proves", "cost"):
            if not g.get(field):
                problems.append(f"{gid}: 缺 {field}")
        if g.get("group") and g["group"] not in reg.get("groups", {}):
            problems.append(f"{gid}: group `{g['group']}` 不在 groups 表裡")
        if g.get("cost") not in ("fast", "heavy"):
            problems.append(f"{gid}: cost 必須是 fast 或 heavy（現在是 {g.get('cost')!r}）")
        # 快速閘門沒有自測時要明說為什麼 —— 這一條是「能繞過的閘門不是閘門」的直接套用。
        # 但**這個閘門本身跑的就是 --self-test** 時（它就是一組注入式違規），不需要再要一次：
        # 那是可機檢的，所以不必靠人記得填理由。
        is_selftest_gate = "--self-test" in " ".join(g.get("cmd") or [])
        if g.get("cost") == "fast" and not g.get("selftest") and not is_selftest_gate:
            if not g.get("no_selftest_reason"):
                problems.append(f"{gid}: fast 但沒有 selftest，也沒有 no_selftest_reason")
    return problems


def run_gates(reg: dict, fast_only: bool = True, timeout: int = 300) -> list[dict]:
    results = []
    for g in reg.get("gates", []):
        is_fast = g.get("cost") == "fast"
        if fast_only and not is_fast:
            results.append({
                "id": g["id"], "group": g["group"], "title": g["title"],
                "cmd": g["cmd"], "cmd_str": " ".join(g["cmd"]),
                "expect": g["expect"], "proves": g.get("proves", ""),
                "not_proves": g.get("not_proves", ""), "cost": g["cost"],
                "needs": g.get("needs", ""), "citation": g.get("citation", ""),
                "selftest": g.get("selftest"), "status": "SKIP",
                "volatile": g.get("volatile_inputs", ""),
                # tail 在 CLI 與 portal 是同一格 ⇒ 理由只寫一次，而且**不含 markdown 星號**
                # （終端輸出與表格都不會渲染 markdown，星號會原樣顯示成排版垃圾）。
                "tail": "heavy：入口刻意不自動跑（需要 " + (g.get("needs") or "外部資源") + "）",
                # ★ 這裡**不要**再寫一次 "tail"：同一個 dict 字面量裡重複的鍵，後者勝，
                #   而那個空字串會把上面那句理由靜默蓋掉（症狀：SKIP 那一列的尾行是空的，
                #   看起來像「沒有理由」—— 而它其實有）。這是本檔第三次踩「同一個東西寫兩次」。
                "rc": None, "seconds": None,
            })
            continue
        rc, secs, out = run(g["cmd"], timeout=timeout)
        tail = [l for l in out.strip().splitlines() if l.strip()]
        results.append({
            "id": g["id"], "group": g["group"], "title": g["title"],
            "cmd": g["cmd"], "cmd_str": " ".join(g["cmd"]),
            "expect": g["expect"], "proves": g.get("proves", ""),
            "not_proves": g.get("not_proves", ""), "cost": g["cost"],
            "needs": g.get("needs", ""), "citation": g.get("citation", ""),
            "selftest": g.get("selftest"),
            "volatile": g.get("volatile_inputs", ""),
            "status": "PASS" if rc == 0 else "FAIL",
            "rc": rc, "seconds": round(secs, 2),
            "tail": tail[-1][:200] if tail else "",
            "output_tail": tail[-14:],
        })
    return results


# ── 心智圖的三棵樹 ─────────────────────────────────────────────────────────────
def goal_tree(node: dict, depth: int = 0) -> dict:
    out = {
        "kind": "goal", "id": node.get("id", ""), "label": node.get("title", ""),
        "status": node.get("status", ""), "why": node.get("why", ""),
        "acceptance": node.get("acceptance", ""),
        "blocked_by": node.get("blocked_by", ""),
        "evidence": node.get("evidence", []),
        "evidence_note": node.get("evidence_note", ""),
        "depth": depth, "children": [],
    }
    for ch in node.get("children", []):
        out["children"].append(goal_tree(ch, depth + 1))
    return out


def decision_tree(dec: dict) -> dict:
    root = {"kind": "group", "id": "DEC-ROOT", "label": f"宏觀技術決策（{dec['counts']['total']}）",
            "why": "每一筆都有 question／evidence／conclusion／judgement。judgement 是**可被推翻的狀態**，"
                   "不是評分；被推翻的仍留在樹上，因為刪掉會讓後人重踩一次。",
            "children": []}
    if dec["chains"]:
        ch = {"kind": "group", "id": "DEC-CHAINS",
              "label": f"取代鏈（{len(dec['chains'])} 條）",
              "why": "A 被 B 取代、B 又被 C 取代 —— 這條線最適合復盤：它顯示理解是怎麼演進的。",
              "children": []}
        for chain in dec["chains"]:
            ch["children"].append({
                "kind": "chain", "id": "CHAIN-" + chain[0]["id"][:24],
                "label": " → ".join(c["short"][:26] for c in chain),
                "chain": chain, "why": "點開看每一環的問句與結論。", "children": [],
            })
        root["children"].append(ch)
    by_day: dict[str, list] = {}
    for it in dec["items"]:
        by_day.setdefault(it["day"], []).append(it)
    for day in sorted(by_day, reverse=True):
        items = by_day[day]
        dnode = {"kind": "group", "id": f"DAY-{day}", "label": f"{day}（{len(items)} 筆）",
                 "why": "同一天的決策通常共用同一批證據。", "children": []}
        groups = [("定案", lambda i: i["judgement"] == "sound" and not i["superseded_by"], "green"),
                  ("已被取代", lambda i: bool(i["superseded_by"]), "amber"),
                  ("被推翻", lambda i: i["judgement"] == "refuted" and not i["superseded_by"], "red")]
        placed = set()
        for gname, pred, tone in groups:
            sel = [i for i in items if pred(i) and i["id"] not in placed]
            placed |= {i["id"] for i in sel}
            if not sel:
                continue
            g = {"kind": "group", "id": f"{day}-{gname}", "label": f"{gname}（{len(sel)}）",
                 "tone": tone, "why": "", "children": []}
            for i in sel:
                g["children"].append({
                    "kind": "decision", "id": i["id"], "label": i["short"][:60],
                    "decision": i, "tone": tone, "children": [],
                })
            dnode["children"].append(g)
        root["children"].append(dnode)
    return root


def charter_tree(charter: dict, lessons: dict, dec: dict) -> dict:
    root = {"kind": "group", "id": "CH-ROOT",
            "label": f"判準與規訓（憲章 {charter['total']} 條 ＋ lesson {lessons.get('lessons', 0)} 條）",
            "why": "憲章是可引用的判準；lesson 是踩過的坑。兩者都是**可以再被推翻**的，不是永恆真理。",
            "children": []}
    ch = {"kind": "group", "id": "CH-A-E", "label": f"CONVENTIONS §A–§E（{charter['total']} 條）",
          "why": "每一條都必須能指到今天的具體證據，否則檢查器會紅。", "children": []}
    for s in charter["sections"]:
        ch["children"].append({
            "kind": "charter", "id": f"CH-{s['key']}",
            "label": (f"{s['key']}. {s['title'][:44]}（{s['count']} 條）" if s["count"]
                      else f"{s['key']}. {s['title'][:44]}（散文式，無編號條文）"),
            "section": s, "children": [],
        })
    root["children"].append(ch)
    cl = {"kind": "group", "id": "CH-LESSONS",
          "label": f"lessons（{lessons.get('lessons', 0)} 條，7 類）",
          "why": "class 是教訓的種類：量測衛生佔最多，說明最貴的錯不是程式錯。",
          "children": []}
    for cls, n in (lessons.get("lessons_by_class") or {}).items():
        cl["children"].append({
            "kind": "lessonclass", "id": f"LC-{cls}", "label": f"{cls}（{n}）",
            "lesson_class": cls, "children": [],
        })
    root["children"].append(cl)
    return root


# ── 欠帳：四類，全部衍生 ───────────────────────────────────────────────────────
def build_obligations(gate_results: list[dict], assets: dict, dec: dict,
                      charter: dict, obligations: list[dict]) -> list[dict]:
    out = []
    reds = [g for g in gate_results if g["status"] == "FAIL"]
    if reds:
        out.append({
            "kind": "gate", "severity": "high",
            "title": f"{len(reds)} 條快速閘門在**乾淨的樹上就是紅的**",
            "detail": "這些閘門存在、跑得動、而且現在是紅的 —— 它們不在提交閘門的 11 段裡，所以沒有人會踩到。"
                      "這是「閘門存在 ≠ 閘門被跑」最具體的形式。",
            "items": [{"id": g["id"], "tail": g["tail"], "cmd": g["cmd_str"]} for g in reds],
        })
    dm = assets["derived"]
    missing = dm.get("engine_memories", 0)
    lessons_n = assets["traces"]["lessons"]
    if missing < lessons_n:
        out.append({
            "kind": "derived", "severity": "high",
            "title": f"衍生記憶落後 lessons {lessons_n - missing} 筆",
            "detail": f"`memories/engine/` 有 {missing} 檔、lessons.jsonl 有 {lessons_n} 條 ⇒ "
                      f"system prompt 的注入來源比教訓本身舊。**重生它是一行指令，但它會改變兩個 loop 的"
                      f"執行時行為**（CONVENTIONS §E 的閉環對照義務），所以不能順手做。",
            "items": [{"id": "build_memories.py", "tail": "見 derived-engine-memories 閘門",
                       "cmd": "python3 agent_harness/engine_loop/harness_engine/build_memories.py"}],
        })
    last = dec["counts"]["last_day"]
    today = time.strftime("%Y-%m-%d")
    if last and last < today:
        gap = sum(1 for f in assets["memory"]["files"]
                  if f["name"][:10] > last and re.match(r"^\d{4}-\d{2}-\d{2}\.md$", f["name"]))
        out.append({
            "kind": "registry", "severity": "medium",
            "title": f"決策 registry 的覆蓋停在 {last}（今天的日誌有 {gap} 天在它之後）",
            "detail": f"`traces/decisions.jsonl` 的最後一筆是 {last}。之後的判準決定只存在於 "
                      f"`.workbuddy/memory/*.md` 與 `CONVENTIONS.md` 的修訂裡 ⇒ 它們**不會進任何資料集**"
                      f"（sft_prime 只從這三個 jsonl 投影）。這是「登記制 vs 日誌制」的落差，不是 bug。",
            "items": [{"id": "decisions.jsonl", "tail": f"最後一筆 {last}",
                       "cmd": "python3 -c \"import json;print(json.loads(open('agent_harness/engine_loop/traces/decisions.jsonl').readlines()[-1])['decision_id'])\""}],
        })
    for ob in obligations:
        out.append({
            "kind": "charter", "severity": "medium",
            "title": f"憲章自報的義務（第 {ob['line']} 行）",
            "detail": ob["text"],
            "items": [{"id": f"CONVENTIONS.md:{ob['line']}", "tail": "",
                       "cmd": f"sed -n '{ob['line']}p' agent_harness/CONVENTIONS.md"}],
        })
    return out


# ── HTML ───────────────────────────────────────────────────────────────────────
CSS = """
:root { --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --soft:#f9fafb; --code:#f3f4f6;
        --blue:#1d4ed8; --green:#15803d; --red:#b91c1c; --amber:#b45309; --purple:#6d28d9; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.7 -apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:26px 20px 90px; }
h1 { font-size:23px; margin:0 0 4px; line-height:1.4; }
h2 { font-size:18px; margin:26px 0 10px; padding-top:12px; border-top:1px solid var(--line); }
h3 { font-size:15.5px; margin:18px 0 6px; }
.sub { color:var(--muted); font-size:13px; margin:0 0 18px; }
a { color:var(--blue); }
code { background:var(--code); padding:1px 5px; border-radius:4px; font-size:12.6px;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
pre { background:var(--soft); border:1px solid var(--line); border-radius:6px; padding:10px 12px;
      overflow-x:auto; margin:8px 0 14px; }
pre code { background:none; padding:0; font-size:12.3px; line-height:1.55; display:block; white-space:pre; }
table { border-collapse:collapse; width:100%; margin:10px 0 16px; font-size:13.2px; }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:14px 0 6px; }
.kpi { border:1px solid var(--line); border-radius:8px; padding:11px 13px; background:var(--soft); }
.kpi .v { font-size:23px; font-weight:650; line-height:1.2; }
.kpi .l { font-size:12px; color:var(--muted); }
.kpi.bad { border-color:#fecaca; background:#fef2f2; }
.kpi.good { border-color:#bbf7d0; background:#f0fdf4; }
.kpi.warn { border-color:#fde68a; background:#fffbeb; }
.tabs { display:flex; gap:6px; flex-wrap:wrap; margin:18px 0 12px; border-bottom:1px solid var(--line); }
.tab { padding:7px 14px; border:1px solid var(--line); border-bottom:none; border-radius:7px 7px 0 0;
       cursor:pointer; font-size:13.5px; background:var(--soft); color:var(--muted); }
.tab.on { background:var(--bg); color:var(--fg); font-weight:600; }
.panel { display:none; } .panel.on { display:block; }
.tags { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
.tag { display:inline-block; font-size:11.5px; padding:1px 8px; border-radius:999px;
       border:1px solid var(--line); background:var(--soft); color:var(--muted); }
.t-green{color:var(--green);border-color:#bbf7d0;background:#f0fdf4}
.t-red{color:var(--red);border-color:#fecaca;background:#fef2f2}
.t-amber{color:var(--amber);border-color:#fde68a;background:#fffbeb}
.t-blue{color:var(--blue);border-color:#bfdbfe;background:#eff6ff}
.t-grey{color:var(--muted);border-color:var(--line);background:var(--soft)}
.box { border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:4px;
       padding:10px 13px; margin:12px 0; background:#fbfdff; }
.box.bad { border-left-color:var(--red); background:#fff7f7; }
.box.warn { border-left-color:var(--amber); background:#fffdf5; }
.box.ok { border-left-color:var(--green); background:#f6fdf8; }
.mm-wrap { display:grid; grid-template-columns:minmax(0,1fr) 400px; gap:14px; align-items:start; }
@media (max-width:1080px){ .mm-wrap { grid-template-columns:1fr; } }
.mm { border:1px solid var(--line); border-radius:8px; background:#fcfcfd; overflow:auto;
      height:620px; position:relative; }
.mm svg { display:block; cursor:grab; }
.mm svg:active { cursor:grabbing; }
.detail { border:1px solid var(--line); border-radius:8px; padding:13px 15px; background:var(--soft);
          height:620px; overflow:auto; font-size:13.4px; }
.detail h4 { margin:0 0 8px; font-size:15px; }
.detail dl { margin:0; } .detail dt { font-weight:600; margin-top:10px; color:var(--muted); font-size:12.3px; }
.detail dd { margin:2px 0 0; }
.bar { height:9px; border-radius:5px; background:#e5e7eb; overflow:hidden; }
.bar > i { display:block; height:100%; background:var(--blue); }
.spark { display:flex; align-items:flex-end; gap:2px; height:38px; margin-top:6px; }
.spark i { width:9px; background:var(--blue); border-radius:2px 2px 0 0; min-height:2px; }
.spark i.r { background:var(--red); }
.foot { color:var(--muted); font-size:12.2px; margin-top:26px; border-top:1px solid var(--line); padding-top:12px; }
.k { font-weight:600; }
.small { font-size:12.3px; color:var(--muted); }
"""

JS = r"""
const D = window.__PORTAL__;
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fmt(s){
  let t = String(s==null?'':s);
  // 白名單消毒：只放行這幾個標籤，其餘 '<' 一律轉義（資料來自 markdown 與 jsonl，不是可信 HTML）。
  t = t.replace(/<[^>]*>/g, function(tag){
    return /^<\/?(code|b|i|em|u|br)\b[^>]*>$/i.test(tag) ? tag : tag.replace(/</g, '&lt;');
  });
  // 落單的 '<'（沒有對應的 '>'）也轉義
  t = t.replace(/</g, function(m, off){ return /^<\/?(code|b|i|em|u|br)\b/i.test(t.slice(off, off+8)) ? '<' : '&lt;'; });
  return t.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/`([^`]+)`/g,'<code>$1</code>');
}
function toneOf(kind, node){
  if(kind==='goal') return {done:'green',active:'blue',blocked:'red',planned:'grey'}[node.status]||'grey';
  if(node.tone) return node.tone;
  if(kind==='chain') return 'purple';
  return 'grey';
}
const FILL = {green:'#f0fdf4',red:'#fef2f2',amber:'#fffbeb',blue:'#eff6ff',grey:'#f3f4f6',purple:'#f5f3ff'};
const STROKE = {green:'#bbf7d0',red:'#fecaca',amber:'#fde68a',blue:'#bfdbfe',grey:'#e5e7eb',purple:'#ddd6fe'};
const TEXT = {green:'#15803d',red:'#b91c1c',amber:'#b45309',blue:'#1d4ed8',grey:'#374151',purple:'#6d28d9'};

function width(s){
  let w=0; for(const ch of String(s)) w += ch.codePointAt(0) > 0x2e80 ? 13.2 : 7.1;
  return Math.max(90, Math.min(330, w + 22));
}

function layout(root, cfg){
  let cursor = 0;
  (function size(n, depth){
    n._depth = depth;
    const kids = (n.children||[]).filter(k => !n._collapsed);
    n._kids = kids;
    if(!kids.length){ n._y = cursor; cursor += cfg.rowH; }
    else { kids.forEach(k => size(k, depth+1)); n._y = (kids[0]._y + kids[kids.length-1]._y)/2; }
  })(root, 0);
  const maxD = Math.max(0, ...(function d(n){ return [n._depth, ...(n._kids||[]).map(d)]; })(root));
  return {h: Math.max(cfg.rowH, cursor + cfg.pad*2), maxDepth: maxD};
}

function draw(){
  const held = document.querySelector('.mm');
  held.innerHTML = '';
  const cfg = {rowH:34, pad:18, colW:300};
  const st = layout(D.__tree, cfg);
  const cols = st.maxDepth + 1;
  const w = cols*cfg.colW + 40, h = st.h;
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('width', w); svg.setAttribute('height', h);
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const g = document.createElementNS('http://www.w3.org/2000/svg','g');
  g.setAttribute('transform', `scale(${D.__zoom})`);
  svg.appendChild(g);
  const nodes = [];
  (function walk(n){ nodes.push(n); (n._kids||[]).forEach(walk); })(D.__tree);
  // 連線
  for(const n of nodes){
    const x0 = n._depth*cfg.colW + 20 + width(n.label), y0 = n._y*1 + cfg.pad + 16;
    for(const k of (n._kids||[])){
      const x1 = k._depth*cfg.colW + 20, y1 = k._y + cfg.pad + 16;
      const p = document.createElementNS('http://www.w3.org/2000/svg','path');
      const mx = (x0+x1)/2;
      p.setAttribute('d', `M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1}`);
      p.setAttribute('fill','none'); p.setAttribute('stroke','#d1d5db'); p.setAttribute('stroke-width','1.2');
      g.appendChild(p);
    }
  }
  // 節點
  for(const n of nodes){
    const tone = toneOf(n.kind, n);
    const bw = width(n.label), x = n._depth*cfg.colW + 20, y = n._y + cfg.pad;
    const rec = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rec.setAttribute('x',x); rec.setAttribute('y',y); rec.setAttribute('rx','7');
    rec.setAttribute('width',bw); rec.setAttribute('height','30');
    rec.setAttribute('fill',FILL[tone]); rec.setAttribute('stroke',STROKE[tone]);
    rec.setAttribute('stroke-width', n.id===D.__sel ? 2.2 : 1.1);
    rec.style.cursor='pointer';
    rec.addEventListener('click', () => { D.__sel = n.id; select(n); draw(); });
    g.appendChild(rec);
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', x+11); t.setAttribute('y', y+20);
    t.setAttribute('font-size','12.8'); t.setAttribute('fill',TEXT[tone]);
    t.setAttribute('font-family','-apple-system,"PingFang TC",sans-serif');
    let s = n.label; if(s.length > 34 && !(n._kids||[]).length) s = s.slice(0,33)+'…';
    t.textContent = s;
    t.style.pointerEvents='none';
    g.appendChild(t);
    if((n.children||[]).length){
      const marker = document.createElementNS('http://www.w3.org/2000/svg','text');
      marker.setAttribute('x', x+bw+5); marker.setAttribute('y', y+20);
      marker.setAttribute('font-size','12'); marker.setAttribute('fill','#9ca3af');
      marker.textContent = n._collapsed ? '▸' : '▾';
      marker.style.cursor='pointer';
      marker.addEventListener('click', (e) => { e.stopPropagation(); n._collapsed = !n._collapsed; draw(); });
      g.appendChild(marker);
    }
    const ttl = document.createElementNS('http://www.w3.org/2000/svg','title');
    ttl.textContent = n.label;
    rec.appendChild(ttl);
  }
  held.appendChild(svg);
}

function dl(rows){ return rows.map(([k,v]) => v ? `<dt>${esc(k)}</dt><dd>${v}</dd>` : '').join(''); }
function evlist(ev){
  if(!ev || !ev.length) return '';
  return '<ul style="padding-left:18px;margin:4px 0">' + ev.map(e =>
    `<li><code>${esc(e.path)}</code>${e.contains ? ' <span class="small">（要求含 '+esc(e.contains)+'）</span>' : ''}${e.ok===false ? ' <span class="tag t-red">懸空</span>' : ''}</li>`).join('') + '</ul>';
}

function select(n){
  const p = document.getElementById('detail');
  const k = n.kind;
  if(k === 'goal'){
    p.innerHTML = `<h4>${esc(n.id)} · ${esc(n.label)}</h4>
      <span class="tag t-${toneOf(k,n)}">${esc(n.status||'')}</span>
      ${dl([['為什麼', fmt(n.why)], ['什麼算完成', fmt(n.acceptance)],
            ['卡在哪', n.blocked_by ? fmt(n.blocked_by) : ''],
            ['證據', evlist(n.evidence) + (n.evidence_note ? '<div class="small">'+fmt(n.evidence_note)+'</div>' : '')]])}`;
  } else if(k === 'decision'){
    const d = n.decision;
    p.innerHTML = `<h4>${esc(d.short)}</h4>
      <span class="tag t-${d.judgement==='refuted'?'red':'green'}">${esc(d.judgement)}</span>
      <span class="tag t-grey">信心 ${esc(d.confidence)}</span>
      <span class="tag t-grey">證據 ${d.n_evidence} 筆</span>
      ${d.superseded_by ? `<span class="tag t-amber">已被取代</span>` : ''}
      <div class="small" style="margin-top:6px">${esc(d.id)}</div>
      ${dl([['問句', fmt(d.question)], ['結論', fmt(d.conclusion)], ['行動', fmt(d.action)],
            ['證據落在', esc(d.bucket) + evlist(d.artifacts.map(a=>({path:a}))),
             ['被誰取代', d.superseded_by ? '<code>'+esc(d.superseded_by)+'</code>' : '']]])}`;
  } else if(k === 'chain'){
    p.innerHTML = `<h4>取代鏈</h4>` + n.chain.map((c,i) =>
      `${i?'<div class="small">↓ 被取代</div>':''}<div style="margin:5px 0"><span class="tag t-${c.judgement==='refuted'?'red':'grey'}">${esc(c.judgement)}</span> <code>${esc(c.id)}</code></div>`).join('');
  } else if(k === 'charter'){
    p.innerHTML = `<h4>§${esc(n.section.key)} ${esc(n.section.title)}</h4>
      ${dl([['條數', esc(n.section.count)], ['怎麼驗', '<code>python3 agent_harness/shared/check_citations.py --check</code>']])}`;
  } else if(k === 'lessonclass'){
    const rows = (D.lessons_by_class_brief[n.lesson_class]||[]);
    p.innerHTML = `<h4>lesson class · ${esc(n.lesson_class)}</h4>
      <div class="small">${rows.length} 條（顯示前 40）</div>
      <ul style="padding-left:18px">${rows.slice(0,40).map(r=>`<li><code>${esc(r[0])}</code> ${fmt(r[1])}</li>`).join('')}</ul>`;
  } else {
    p.innerHTML = `<h4>${esc(n.label)}</h4>${dl([['說明', fmt(n.why||'')]])}`;
  }
}

function tab(name, btn){
  document.querySelectorAll('.panel').forEach(e => e.classList.remove('on'));
  document.getElementById('p-'+name).classList.add('on');
  document.querySelectorAll('.tab').forEach(e => e.classList.remove('on'));
  btn.classList.add('on');
}
function subtree(name, btn){
  tab('mm', document.querySelector('.tab'));
  document.querySelectorAll('.subtabs .tab').forEach(e => e.classList.remove('on'));
  btn.classList.add('on');
  D.__tree = D.trees[name]; D.__sel = null;
  document.getElementById('detail').innerHTML = '<h4>點一個節點</h4><div class="small">左邊的圓角方塊都可以點；有 ▾ 的可以折疊。滑鼠滾輪縮放。</div>';
  draw();
}
document.addEventListener('DOMContentLoaded', () => {
  D.__zoom = 1; D.__sel = null; D.__tree = D.trees.goals;
  document.getElementById('detail').innerHTML = '<h4>點一個節點</h4><div class="small">左邊的圓角方塊都可以點；有 ▾ 的可以折疊。</div>';
  draw();
  document.querySelector('.mm').addEventListener('wheel', (e) => {
    if(!e.ctrlKey && !e.metaKey && !e.shiftKey) return;
    e.preventDefault();
    D.__zoom = Math.min(2.2, Math.max(0.5, D.__zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    draw();
  }, {passive:false});
});
"""


def render_html(data: dict) -> str:
    a = data["assets"]
    g = data["gates"]
    dec = data["decisions"]
    ch = data["charter"]
    reds = [x for x in g["results"] if x["status"] == "FAIL"]
    skips = [x for x in g["results"] if x["status"] == "SKIP"]
    passes = [x for x in g["results"] if x["status"] == "PASS"]

    def tag(txt, cls=""):
        return f'<span class="tag {cls}">{txt}</span>'

    # ── KPI ──
    kpi_cls = "bad" if reds else "good"
    kpis = [
        (f'{g["n_pass"]}/{g["n_fast"]}', "快速閘門通過（實跑）", kpi_cls),
        (f'{dec["counts"]["total"]}', f'宏觀決策（{dec["counts"]["refuted"]} 被推翻）', "warn" if dec["counts"]["refuted"] else ""),
        (f'{a["traces"]["lessons"]}', f'lessons（{len(a["traces"]["lessons_by_class"])} 類）', ""),
        (hsize(a["memory"]["bytes"]), f'記憶 {a["memory"]["count"]} 檔', ""),
        (f'{a["skills"]["count"]}', f'skill（{hsize(a["skills"]["bytes"])}）', ""),
        (f'{a["code"]["loc_live"]:,}', f'現行代碼 LOC（{a["code"]["files_live"]:,} 檔；另有存檔 {a["code"]["loc"] - a["code"]["loc_live"]:,} 行）', ""),
    ]
    kpi_html = "".join(f'<div class="kpi {c}"><div class="v">{v}</div><div class="l">{l}</div></div>'
                       for v, l, c in kpis)

    # ── 閘門表 ──
    prev = data.get("previous") or {}
    prev_by_id = {r["id"]: r for r in (prev.get("gates", {}).get("results") or [])}
    order = [x for x in g["results"] if x["status"] != "SKIP"] + skips
    rows = []
    for r in order:
        cls = {"PASS": "t-green", "FAIL": "t-red", "SKIP": "t-grey"}[r["status"]]
        p = prev_by_id.get(r["id"], {})
        delta = ""
        if p and p.get("status") and p["status"] != r["status"]:
            delta = f'<span class="tag t-amber">{p["status"]} → {r["status"]}</span>'
        st = f'<span class="tag {cls}">{r["status"]}</span>' + (" " + delta if delta else "")
        if r.get("volatile") and r["status"] == "FAIL":
            st += ' <span class="tag t-amber" title="' + hesc(plain(r["volatile"])) + '">輸入易變</span>' 
        secs = f'{r["seconds"]:.2f}s' if isinstance(r.get("seconds"), (int, float)) else "—"
        stc = (f'<code>{" ".join(r["selftest"])}</code>' if r.get("selftest") else
               '<span class="small">—</span>')
        rows.append(
            f'<tr><td><code>{r["id"]}</code><div class="small">{r["group"]}</div></td>'
            f'<td>{r["title"]}</td><td>{st}</td><td class="num">{secs}</td>'
            f'<td><code>{r["cmd_str"]}</code></td><td>{stc}</td>'
            f'<td class="small">{mdx(r["proves"])}</td>'
            f'<td class="small">{mdx(r["not_proves"])}</td>'
            f'<td class="small">{mdx(r["tail"])}</td></tr>')
    gate_table = (
        '<table><thead><tr><th>id／組</th><th>這一條在驗什麼</th><th>本輪</th><th>耗時</th>'
        '<th>復現指令</th><th>它自己的陰性對照</th><th>證什麼</th><th>不證什麼</th><th>尾行</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')

    # ── 資產表 ──
    mem_rows = "".join(
        f'<tr><td><code>{f["name"]}</code></td><td class="num">{f["bytes"]:,}</td>'
        f'<td class="num">{f["lines"]:,}</td><td class="num">{f["sections"]}</td><td>{f["mtime"]}</td></tr>'
        for f in a["memory"]["files"])
    sk_rows = "".join(
        f'<tr><td><code>{s["name"]}</code></td><td class="num">{s["lines"]:,}</td>'
        f'<td class="num">{s["bytes"]:,}</td><td><code>{s["sha8"]}</code></td>'
        f'<td class="small">{mdx(clip(s["desc"], 150))}</td></tr>' for s in a["skills"]["items"])
    max_loc = max((x["loc"] for x in a["code"]["areas"]), default=1)
    code_rows = "".join(
        f'<tr><td>{x["area"]}</td><td class="num">{x["files"]:,}</td><td class="num">{x["loc"]:,}</td>'
        f'<td style="width:190px"><div class="bar"><i style="width:{100*x["loc"]/max_loc:.1f}%;'
        f'{"background:#d1d5db" if x.get("archive") else ""}"></i></div></td>'
        f'<td class="num">{hsize(x["bytes"])}</td></tr>' for x in a["code"]["areas"])
    cls_rows = "".join(
        f'<tr><td><code>{k}</code></td><td class="num">{v}</td>'
        f'<td style="width:190px"><div class="bar"><i style="width:{100*v/max(1,a["traces"]["lessons"]):.1f}%"></i></div></td></tr>'
        for k, v in a["traces"]["lessons_by_class"].items())
    dv = a["derived"]
    dex_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>'
        for k, v in [("memories/engine/（system prompt 注入來源）", dv["engine_memories"]),
                     ("sft_pi/train.jsonl", dv["sft_pi_train"]), ("sft_pi/valid.jsonl", dv["sft_pi_valid"]),
                     ("sft_prime/train.jsonl", dv["sft_prime_train"]),
                     ("sft_prime/valid.jsonl", dv["sft_prime_valid"]),
                     ("shared/evidence 檔數", a["corpus"]["evidence"]["files"]),
                     ("Backup/ 檔數", a["corpus"]["backup"]["files"])])

    # ── 趨勢 ──
    hist = data.get("history") or []
    trend = ""
    if len(hist) > 1:
        reds_s = [h.get("gates_red", 0) for h in hist][-30:]
        mx = max(reds_s) or 1
        bars = "".join(f'<i class="{"r" if v else ""}" style="height:{max(3, 100*v/mx):.0f}%" title="{v} 紅燈"></i>' for v in reds_s)
        ls = [h.get("lessons", 0) for h in hist][-30:]
        lo, hi = min(ls), max(ls)
        bars2 = "".join(f'<i style="height:{max(4, 100*(v-lo)/max(1,hi-lo)):.0f}%" title="{v} lessons"></i>' for v in ls)
        trend = (f'<div class="small">紅燈數（近 {len(reds_s)} 次執行）</div><div class="spark">{bars}</div>'
                 f'<div class="small" style="margin-top:8px">lessons 總數（近 {len(ls)} 次）</div><div class="spark">{bars2}</div>')
    else:
        trend = '<div class="small">首個資料點 —— 趨勢要有第二次執行才畫得出來。這也是為什麼「定時」是這一頁的一部分。</div>'

    # ── 欠帳 ──
    ob_html = ""
    for o in data["obligations"]:
        sev = {"high": "bad", "medium": "warn", "low": ""}[o["severity"]]
        items = "".join(
            f'<li><code>{i["id"]}</code> — <span class="small">{i["tail"]}</span><br>'
            f'<code>{i["cmd"]}</code></li>' for i in o["items"])
        ob_html += (f'<div class="box {sev}"><div class="k">{mdx(o["title"])}</div>'
                    f'<div style="margin-top:5px">{mdx(o["detail"])}</div>'
                    f'<ul style="margin:7px 0 0;padding-left:18px">{items}</ul></div>')
    if not ob_html:
        ob_html = '<div class="box ok">沒有登記在案的欠帳。</div>'

    # ── 決策表 ──
    dec_rows = "".join(
        f'<tr><td><code>{x["day"]} {x["time"]}</code></td>'
        f'<td><span class="tag t-{"red" if x["judgement"]=="refuted" else "green"}">{x["judgement"]}</span>'
        + (f'<span class="tag t-amber">已取代</span>' if x["superseded_by"] else '') + '</td>'
        f'<td>{mdx(x["short"])}</td><td class="small">{mdx(clip(x["question"], 190))}</td>'
        f'<td class="small">{mdx(clip(x["conclusion"], 230))}</td><td class="num">{x["n_evidence"]}</td>'
        f'<td class="small">{x["bucket"]}</td></tr>' for x in dec["items"])

    payload = {
        "trees": data["trees"],
        "lessons_by_class_brief": data["lessons_brief"],
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    hist_meta = (f'{len(hist)} 次執行紀錄' if hist else "本輪是第一次執行")

    # ── 取代鏈（在 f-string 之外組好：鏈的 HTML 內含雙引號，混在 f-string 裡會咬到引號）──
    if dec["chains"]:
        chains_html = ""
        for chain in dec["chains"]:
            cells = []
            for c in chain:
                tone = "red" if c["judgement"] == "refuted" else "grey"
                cells.append(f'<span class="tag t-{tone}">{hesc(c["judgement"])}</span> '
                             f'<code>{hesc(c["short"])}</code>')
            chains_html += "<tr><td>" + ' <span class="small">→ 被取代</span> '.join(cells) + "</td></tr>"
        chain_html = ('<table><thead><tr><th>取代鏈（愈下面愈新）</th></tr></thead><tbody>'
                      + chains_html + "</tbody></table>")
    else:
        chain_html = '<div class="box warn">沒有取代鏈。</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGENT HARNESS PORTAL — 目標／決策／驗證／資產</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<h1>AGENT HARNESS PORTAL — 目標／決策／驗證／資產</h1>
<p class="sub">
產生於 <b>{data['generated_at']}</b>　·　
<code>{data['git']['head_short']}</code> ({hesc(data['git']['branch'])}) {hesc(data['git']['subject'][:78])}
　·　未提交 <b>{data['git']['dirty']}</b> 檔
　·　輸入指紋 <code>{data['fingerprint']['digest']}</code>
　·　{hist_meta}
</p>

<div class="kpis">{kpi_html}</div>

<div class="box {'bad' if reds else 'ok'}">
  <div class="k"><b>一句話</b></div>
  <div style="margin-top:5px">
    {('有 <b>' + str(len(reds)) + '</b> 條快速閘門是紅的（' + "、".join(x['id'] for x in reds) + '）—— 這些閘門存在、跑得動，'
      '但<b>不在提交閘門的 11 段裡</b>，所以沒有人會踩到。') if reds else '所有快速閘門綠。'} 
    另有 <b>{len(skips)}</b> 條 heavy 閘門<b>刻意未自動執行</b>（需要 GPU／build／docker／模型端點）——
    它們在表上是 <span class="tag t-grey">SKIP</span>，不是綠燈。
  </div>
  <div class="small" style="margin-top:7px">
    一鍵復現：<code>python3 agent_harness/portal/build_portal.py --gates-only</code>
    （任何紅燈 ⇒ rc=1）　·　只驗不寫：<code>… --check</code>　·　自我陰性對照：<code>… --self-test</code>
  </div>
</div>

<div class="tabs">
  <div class="tab on" onclick="tab('mm',this)">心智圖</div>
  <div class="tab" onclick="tab('gates',this)">驗證閘門</div>
  <div class="tab" onclick="tab('assets',this)">資產盤點</div>
  <div class="tab" onclick="tab('oblig',this)">欠帳與復盤</div>
</div>

<!-- ══ 心智圖 ══ -->
<div class="panel on" id="p-mm">
  <div class="tabs subtabs" style="border:none;margin:0 0 8px">
    <div class="tab on" onclick="subtree('goals',this)">目標樹</div>
    <div class="tab" onclick="subtree('decisions',this)">宏觀技術決策</div>
    <div class="tab" onclick="subtree('charter',this)">判準與規訓</div>
  </div>
  <div class="mm-wrap">
    <div class="mm"></div>
    <div class="detail" id="detail"></div>
  </div>
  <div class="small" style="margin-top:8px">
    節點可點、可折疊（▾）；<b>Ctrl/⌘ + 滾輪</b>縮放。右側面板是「可解釋」的那一半：
    每個節點都要能回答 <i>為什麼</i>、<i>什麼算完成</i>、<i>證據在哪</i>。
  </div>
</div>

<!-- ══ 閘門 ══ -->
<div class="panel" id="p-gates">
  <h2>驗證閘門（{g['n_fast']} 條快速 ＋ {g['n_heavy']} 條 heavy）</h2>
  <p class="sub">單一真相來源是 <code>agent_harness/portal/gates.json</code>；下面每一條的「復現指令」
  就是入口真的跑的那條 —— 不存在文件與實跑不一致的可能。</p>
  {gate_table}
  <div class="box warn">
    <div class="k">heavy 閘門為什麼不自動跑</div>
    <div style="margin-top:5px">它們需要 GPU／build 產物／docker／模型端點，而且其中幾條會<b>與另一條線正在做的量測互搶</b>
    （本 repo 允許多個 session 共用同一個 worktree）。把跑不動的閘門畫成綠燈，比沒有閘門更糟 —— 所以它們是
    <span class="tag t-grey">SKIP</span>。要跑就照表上的指令，並先確認 8080 空窗與沒有別的 session 在量測。</div>
  </div>
</div>

<!-- ══ 資產 ══ -->
<div class="panel" id="p-assets">
  <h2>記憶（權威：<code>.workbuddy/memory/</code>）</h2>
  <table><thead><tr><th>檔</th><th>bytes</th><th>行</th><th>§ 節</th><th>mtime</th></tr></thead>
  <tbody>{mem_rows}</tbody></table>
  <div class="small">合計 <b>{a['memory']['bytes']:,} B</b> / {a['memory']['lines']:,} 行 / {a['memory']['count']} 檔。
  這批 bytes <b>不在版控內</b>，靠 <code>agent_harness/memory/</code> 的 dated 快照跨機器。</div>

  <h2>skill（權威：<code>~/.workbuddy/skills/</code>）</h2>
  <table><thead><tr><th>名稱</th><th>行</th><th>bytes</th><th>sha8</th><th>description</th></tr></thead>
  <tbody>{sk_rows}</tbody></table>

  <h2>程式碼（LOC 與 bytes，按區域）</h2>
  <table><thead><tr><th>區域</th><th>檔數</th><th>LOC</th><th></th><th>bytes</th></tr></thead>
  <tbody>{code_rows}</tbody></table>
  <div class="small">排除 <code>.git/.venv/node_modules/__pycache__/Backup</code>；二進位檔不計 LOC
  （讀到 NUL 就跳過）。<b>現行代碼 {a['code']['loc_live']:,} 行 / {a['code']['files_live']:,} 檔</b>
  （灰色條 = 歷史存檔，不計入）；含存檔合計 {a['code']['loc']:,} 行 / {a['code']['files']:,} 檔。</div>

  <h2>軌跡（訓練數據的唯一出口）</h2>
  <table><thead><tr><th>種類</th><th>筆數</th><th></th></tr></thead><tbody>
    <tr><td>episode</td><td class="num">{a['traces']['episodes']}</td>
        <td class="small">其中 {a['traces']['episodes_usable']} 標為 usable_as_evidence</td></tr>
    <tr><td>decision</td><td class="num">{a['traces']['decisions']}</td>
        <td class="small">最後一筆 {dec['counts']['last_day']}</td></tr>
    <tr><td>lesson</td><td class="num">{a['traces']['lessons']}</td>
        <td class="small">{len(a['traces']['lessons_by_class'])} 類</td></tr>
  </tbody></table>
  <h3>lesson 分類</h3>
  <table><thead><tr><th>class</th><th>條數</th><th></th></tr></thead><tbody>{cls_rows}</tbody></table>

  <h2>衍生物（會進 system prompt 與資料集的那批）</h2>
  <table><thead><tr><th>項目</th><th>數量</th></tr></thead><tbody>{dex_rows}</tbody></table>

  <h2>交付物</h2>
  <table><thead><tr><th>項目</th><th>數量</th></tr></thead><tbody>
    <tr><td>白皮書 <code>docs/*.html</code></td><td class="num">{a['corpus']['whitepapers']}</td></tr>
    <tr><td>設計文件 <code>docs/*.md</code></td><td class="num">{a['corpus']['docs_md']}</td></tr>
    <tr><td>證據包 <code>shared/evidence</code></td><td class="num">{a['corpus']['evidence']['files']:,} 檔 / {hsize(a['corpus']['evidence']['bytes'])}</td></tr>
    <tr><td>量測原始紀錄 <code>Backup/</code></td><td class="num">{a['corpus']['backup']['files']:,} 檔 / {hsize(a['corpus']['backup']['bytes'])}</td></tr>
  </tbody></table>

  <h2>趨勢（<code>history.jsonl</code>，只追加）</h2>
  {trend}
</div>

<!-- ══ 欠帳 ══ -->
<div class="panel" id="p-oblig">
  <h2>欠帳（{len(data['obligations'])} 項，全部由程式推導，不是手抄）</h2>
  <p class="sub">這一頁刻意全部是<b>衍生</b>的：紅閘門來自本輪實跑，覆蓋落差來自 registry 的最後一筆日期，
  憲章義務來自掃 <code>CONVENTIONS.md</code> 的標記行。手抄的清單會腐爛，衍生的不會。</p>
  {ob_html}

  <h2>復盤：被推翻與被取代的決策</h2>
  <p class="sub">這一節是「可復盤」的核心 —— 不是列出決定了什麼，而是列出<b>哪些理解被後來的證據改掉了</b>。
  被推翻的決策依 §D4 不得當正例訓練。</p>
  {chain_html}

  <h2>全部決策（{dec['counts']['total']} 筆）</h2>
  <table><thead><tr><th>時間</th><th>判決</th><th>主題</th><th>問句</th><th>結論</th><th>證據</th><th>落在</th></tr></thead>
  <tbody>{dec_rows}</tbody></table>
</div>

<div class="foot">
  產生器 <code>agent_harness/portal/build_portal.py</code>　·　閘門註冊表 <code>agent_harness/portal/gates.json</code>
  　·　目標樹 <code>agent_harness/portal/goals.json</code>　·　歷史 <code>agent_harness/portal/history.jsonl</code><br>
  這一頁的每個數字都指到一個可重跑的指令；<b>heavy 閘門一律標 SKIP 而不是綠燈</b>。
  輸入指紋 <code>{data['fingerprint']['digest']}</code> 相同的兩次執行才可比較。
</div>
</div>
<script>window.__PORTAL__ = {payload_json};</script>
<script>{JS}</script>
</body></html>
"""


# ── 目標樹機檢 ─────────────────────────────────────────────────────────────────
def check_goals(goals: dict) -> list[str]:
    problems = []

    def walk(node: dict, path: str) -> None:
        nid = node.get("id") or path
        for field in ("id", "title", "why", "status", "acceptance"):
            if not node.get(field):
                problems.append(f"{nid}: 缺 {field}")
        if node.get("status") not in ("done", "active", "blocked", "planned"):
            problems.append(f"{nid}: status 必須是 done/active/blocked/planned（現在 {node.get('status')!r}）")
        if node.get("status") == "blocked" and not node.get("blocked_by"):
            problems.append(f"{nid}: status=blocked 但沒有 blocked_by（卡在哪是必填）")
        ev = node.get("evidence") or []
        if not ev and not node.get("evidence_note"):
            problems.append(f"{nid}: 既沒有可解析的 evidence，也沒有 evidence_note（沉默）")
        for e in ev:
            rel = e.get("path", "")
            p = (REPO / rel) if rel else None
            if not p or not p.exists():
                problems.append(f"{nid}: 懸空指針 {rel}")
                continue
            needle = e.get("contains")
            if needle:
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    problems.append(f"{nid}: 讀不到 {rel}")
                    continue
                if needle not in txt:
                    problems.append(f"{nid}: {rel} 裡已找不到 {needle!r}（子串漂移）")
        for ch in node.get("children", []):
            walk(ch, f"{path}/{ch.get('id','?')}")

    if not goals.get("root"):
        problems.append("goals.json 沒有 root")
    else:
        walk(goals["root"], "")
    return problems


def annotate_evidence(node: dict) -> dict:
    """回傳給 portal 用的副本，並標記每個指針是否可解析。"""
    out = {k: v for k, v in node.items() if k != "children"}
    ev = []
    for e in node.get("evidence") or []:
        rel = e.get("path", "")
        p = (REPO / rel) if rel else None
        ok = bool(p and p.exists())
        if ok and e.get("contains"):
            try:
                ok = e["contains"] in p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                ok = False
        ev.append({"path": rel, "contains": e.get("contains", ""), "ok": ok})
    out["evidence"] = ev
    out["children"] = [annotate_evidence(c) for c in node.get("children", [])]
    return out


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def assemble(reg: dict, goals: dict, do_gates: bool, fast_only: bool = True) -> dict:
    git = git_info()
    assets = scan_assets()
    dec = build_decisions(load_jsonl(TRACES / "decisions.jsonl"))
    lessons_rows = load_jsonl(TRACES / "lessons.jsonl")
    charter = build_charter()
    results = (run_gates(reg, fast_only=fast_only) if do_gates else
               [{"id": g["id"], "group": g["group"], "title": g["title"], "cmd": g["cmd"],
                 "cmd_str": " ".join(g["cmd"]), "expect": g["expect"], "proves": g.get("proves", ""),
                 "not_proves": g.get("not_proves", ""), "cost": g["cost"], "needs": g.get("needs", ""),
                 "citation": g.get("citation", ""), "selftest": g.get("selftest"),
                 "status": "SKIP", "tail": "--no-gates：本輪未執行", "rc": None,
                 "seconds": None, "tail": ""} for g in reg["gates"]])
    n_fast = sum(1 for g in reg["gates"] if g["cost"] == "fast")
    n_heavy = sum(1 for g in reg["gates"] if g["cost"] == "heavy")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    obligations = build_obligations(results, assets, dec, charter, extract_obligations())
    lessons_brief = {}
    for r in lessons_rows:
        lessons_brief.setdefault(r.get("class", "?"), []).append(
            [r.get("lesson_id", ""), clip(r.get("rule"), 150)])

    goals_annot = annotate_evidence(goals["root"])
    trees = {
        "goals": goal_tree(goals_annot),
        "decisions": decision_tree(dec),
        "charter": charter_tree(charter, {"lessons": assets["traces"]["lessons"],
                                          "lessons_by_class": assets["traces"]["lessons_by_class"]}, dec),
    }
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "fingerprint": fingerprint(),
        "assets": assets,
        "decisions": {"counts": dec["counts"], "chains": dec["chains"],
                      "items": [{k: v for k, v in i.items()} for i in dec["items"]]},
        "charter": charter,
        "gates": {"results": results, "n_fast": n_fast, "n_heavy": n_heavy,
                  "n_pass": n_pass, "n_fail": n_fail},
        "obligations": obligations,
        "trees": trees,
        "lessons_brief": lessons_brief,
    }
    return data


def history_row(data: dict) -> dict:
    a = data["assets"]
    return {
        "ts": data["generated_at"],
        "head": data["git"]["head_short"],
        "fingerprint": data["fingerprint"]["digest"],
        "gates_pass": data["gates"]["n_pass"],
        "gates_fail": data["gates"]["n_fail"],
        "gates_red_ids": [r["id"] for r in data["gates"]["results"] if r["status"] == "FAIL"],
        "episodes": a["traces"]["episodes"],
        "decisions": a["traces"]["decisions"],
        "lessons": a["traces"]["lessons"],
        "engine_memories": a["derived"]["engine_memories"],
        "memory_bytes": a["memory"]["bytes"],
        "memory_files": a["memory"]["count"],
        "skills": a["skills"]["count"],
        "code_loc": a["code"]["loc"],
        "code_loc_live": a["code"]["loc_live"],
        "whitepapers": a["corpus"]["whitepapers"],
        "obligations": len(data["obligations"]),
    }


def verify_output(html: str) -> list[str]:
    r"""驗 HTML 產物本身。回傳問題清單（空 = 好）。

    四件事，每一件都對應今天真的發生過的一次壞法：
    ① HTML 本體不得有 markdown 殘留（`**`）；
    ② 標籤要對稱 —— 但**內嵌 JSON 要先還原**，因為產生 payload 時會把 `</` 轉義成 `<\/`
       （不還原就會得到「`</code>` 少一個」這種**假的**不對稱，我自己就被它騙過一次）；
    ③ 四個分頁錨點要在；
    ④ 不得引用外部資源（必須離線可開）。
    """
    problems = []
    m = re.search(r"<script>window\.__PORTAL__ = (\{.*?\});</script>", html, re.S)
    if not m:
        return ["找不到內嵌的 window.__PORTAL__（前端會整頁空白）"]
    i, j = m.start(1), m.end(1)
    body = html[:i] + html[j:]
    restored = html[:i] + html[i:j].replace("<\\/", "</") + html[j:]

    if "**" in body:
        k = body.find("**")
        problems.append(f"HTML 本體有 markdown 殘留 `**`（{body.count('**')} 處，例："
                        f"{body[max(0, k - 40):k + 30]!r}）")

    for tag in ("code", "b", "div", "table", "tbody", "script", "style"):
        # ★ 開標籤要用「`<tag` 後面接空白或 `>`」來數，**不能用 `"<b>"` 也不能用 `"<b"`**：
        #   `"<b"` 會命中 `<br>`／`<body>`（症狀與「標籤真的少了一個」一模一樣）；
        #   而 `"<div>"` 永遠數到 0，因為 div 一定帶屬性（`<div class=…>`）。
        #   這兩個變體我同一天各踩一次，而且是在「專門抓輸出缺陷的函式」裡 ——
        #   **檢查器的錯與被檢查物的錯同形，是最貴的一種**。
        o = len(re.findall(rf"<{tag}[\s>]", restored))
        c = restored.count(f"</{tag}>")
        if o != c:
            problems.append(f"<{tag}> 不對稱：{o} 開 / {c} 關")

    for anchor in ('id="p-mm"', 'id="p-gates"', 'id="p-assets"', 'id="p-oblig"',
                   "window.__PORTAL__"):
        if anchor not in html:
            problems.append(f"缺少錨點 {anchor}")

    ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    if ext:
        problems.append(f"引用了外部資源（無法離線開啟）：{ext[:3]}")
    return problems


def trim_for_disk(data: dict) -> dict:
    """`data.json` 只留「下一輪要用得到」的部分（閘門的前後對照 ＋ 量化指標）。

    為什麼不整份寫：整份含 179 條 lesson 的全文與三棵樹，實測 **578 KB**（第一版），
    而 commit 一份每天重生的 578 KB JSON 是沒有理由的 —— 大檔要進版控就得有讀者，
    這一份的讀者是「下一次執行」。HTML 才是給人看的。
    """
    return {
        "generated_at": data["generated_at"],
        "git": data["git"],
        "fingerprint": data["fingerprint"],
        "assets": data["assets"],
        "decisions": {
            "counts": data["decisions"]["counts"],
            "chains": data["decisions"]["chains"],
            "n_items": len(data["decisions"]["items"]),
        },
        "charter": data["charter"],
        "gates": {
            "n_fast": data["gates"]["n_fast"], "n_heavy": data["gates"]["n_heavy"],
            "n_pass": data["gates"]["n_pass"], "n_fail": data["gates"]["n_fail"],
            "results": [{k: r.get(k) for k in
                         ("id", "group", "status", "seconds", "rc", "tail", "cost", "needs")}
                        for r in data["gates"]["results"]],
        },
        "obligations": [{"kind": o["kind"], "severity": o["severity"], "title": o["title"]}
                        for o in data["obligations"]],
    }


def cmd_check() -> int:
    problems = []
    for path, label in ((GATES_JSON, "gates.json"), (GOALS_JSON, "goals.json")):
        if not path.exists():
            problems.append(f"{label} 不存在: {path}")
    if problems:
        for p in problems:
            print(f"  [error] {p}")
        return 1
    try:
        reg = load_gates()
    except json.JSONDecodeError as e:
        print(f"  [error] gates.json 不是合法 JSON: {e}")
        return 1
    problems += registry_problems(reg)
    try:
        goals = json.loads(GOALS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  [error] goals.json 不是合法 JSON: {e}")
        return 1
    problems += check_goals(goals)
    if problems:
        for p in problems:
            print(f"  [error] {p}")
        print(f"  -> DRIFT: {len(problems)} 項")
        return 1
    fast = sum(1 for g in reg["gates"] if g["cost"] == "fast")
    heavy = sum(1 for g in reg["gates"] if g["cost"] == "heavy")
    print(f"  -> OK: {len(reg['gates'])} 條閘門（fast {fast}／heavy {heavy}），"
          f"目標樹指針全部可解析、每一節都有 acceptance")
    return 0


def cmd_gates_only(reg: dict) -> int:
    results = run_gates(reg, fast_only=True)
    fails = [r for r in results if r["status"] == "FAIL"]
    skips = [r for r in results if r["status"] == "SKIP"]
    for r in results:
        cls = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[r["status"]]
        secs = f'{r["seconds"]:6.2f}s' if isinstance(r.get("seconds"), (int, float)) else "     —"
        print(f"  {cls}  {r['id']:26s} {secs}  {r['tail'][:96]}")
    print(f"\n  {len(results) - len(skips) - len(fails)}/{len(results) - len(skips)} 快速閘門通過"
          f"（另有 {len(skips)} 條 heavy 刻意未執行）")
    if fails:
        print(f"  -> RED: {len(fails)} 條 —— {', '.join(f['id'] for f in fails)}")
        return 1
    print("  -> 全部綠")
    return 0


def cmd_self_test() -> int:
    me = str(Path(__file__).resolve())
    results: list[tuple[str, bool, str]] = []

    def case(name, ok, detail=""):
        results.append((name, ok, detail))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ── 看門狗：整個自我測試期間，真正的 repo 不得有任何檔案被新增或改動 ──
        #   這一格存在的原因很具體：本檔第一版忘了把 PA_PORTAL_REPO 傳給子行程，
        #   於是「跑自我測試」等於「在真正的 repo 原地產生一份 portal」（data.json ＋ history.jsonl
        #   真的被寫出來了，而 13 個 fixture 全部「通過」）。**自測有副作用**是這個 repo
        #   最不能接受的失效模態之一，所以它必須有自己的陰性對照。
        def snapshot_real() -> dict:
            out = {}
            for base in (PORTAL_DIR, REPO / "docs"):
                if base.is_dir():
                    for p in base.iterdir():
                        if p.is_file():
                            st = p.stat()
                            out[str(p)] = (st.st_mtime_ns, st.st_size)
            return out

        before_real = snapshot_real()

        # 最小但**合法**的閘門註冊表：fixture 一律要有它，否則 --check 會先抱怨「gates.json 不存在」，
        # 而那會讓 A–E 五格全部拿同一個假原因失敗（自測本身變成一個掩蓋真相的東西）。
        MINIMAL_REG = {"schema": 1, "groups": {"g": "g"}, "gates": [
            {"id": "ok", "group": "g", "title": "t", "cmd": ["true"], "expect": "0",
             "proves": "p", "not_proves": "q", "cost": "fast", "selftest": ["true"]}]}

        def fixture(goals_obj, gates_obj=MINIMAL_REG):
            r = tmp / f"repo{len(results)}"
            (r / "agent_harness" / "portal").mkdir(parents=True)
            (r / "docs").mkdir(parents=True)
            (r / "agent_harness" / "trails.txt").write_text("marker-ok\n", encoding="utf-8")
            (r / "agent_harness" / "portal" / "goals.json").write_text(
                json.dumps(goals_obj, ensure_ascii=False), encoding="utf-8")
            (r / "agent_harness" / "portal" / "gates.json").write_text(
                json.dumps(gates_obj, ensure_ascii=False), encoding="utf-8")
            return r

        def goal(ev=None, note=None, status="active"):
            g = {"id": "G0", "title": "t", "why": "w", "status": status, "acceptance": "a"}
            if ev is not None:
                g["evidence"] = ev
            if note:
                g["evidence_note"] = note
            return {"root": g}

        def run_check(repo):
            # ★ 一定要帶 PA_PORTAL_REPO。少了它，子行程會用 **真的 repo** 當 REPO，
            #   於是「跑自我測試」等於「在原地產生一份 portal」。這個 bug 是本檔第一版真的犯過的，
            #   所以下面第 O 格就是它的陰性對照。
            return run(["python3", me, "--check"], cwd=repo, timeout=60,
                       env={"PA_PORTAL_REPO": str(repo)})

        def run_me(repo, *flags):
            return run(["python3", me, *flags], cwd=repo, timeout=180,
                       env={"PA_PORTAL_REPO": str(repo)})

        # 1) 乾淨 ⇒ rc 0
        r1 = fixture(goal([{"path": "agent_harness/trails.txt", "contains": "marker-ok"}]))
        rc, _, out = run_check(r1)
        case("A 乾淨的目標樹 ⇒ rc=0", rc == 0, out.strip()[-120:])

        # 2) 懸空指針 ⇒ rc 1 且指名
        r2 = fixture(goal([{"path": "agent_harness/nope.txt"}]))
        rc, _, out = run_check(r2)
        case("B 懸空指針 ⇒ rc=1 且指名", rc == 1 and "nope.txt" in out, out.strip()[-120:])

        # 3) 沒有 evidence 也沒有 evidence_note ⇒ 沉默要算失敗
        r3 = fixture(goal(None, None))
        rc, _, out = run_check(r3)
        case("C 沉默（無指針也無標記）⇒ rc=1", rc == 1 and "evidence_note" in out, out.strip()[-120:])

        # 4) contains 子串漂移 ⇒ rc 1
        r4 = fixture(goal([{"path": "agent_harness/trails.txt", "contains": "marker-GONE"}]))
        rc, _, out = run_check(r4)
        case("D contains 漂移 ⇒ rc=1", rc == 1 and "GONE" in out, out.strip()[-120:])

        # 5) blocked 沒有 blocked_by ⇒ rc 1
        r5 = fixture(goal([{"path": "agent_harness/trails.txt"}], None, status="blocked"))
        rc, _, out = run_check(r5)
        case("E blocked 缺 blocked_by ⇒ rc=1", rc == 1 and "blocked_by" in out, out.strip()[-120:])

        # 6) 閘門缺 not_proves ⇒ rc 1（「只能證不能否證」的檢查不是檢查）
        bad_reg = {"schema": 1, "groups": {"g": "g"}, "gates": [
            {"id": "x", "group": "g", "title": "t", "cmd": ["true"], "expect": "0",
             "proves": "p", "cost": "fast", "selftest": ["true"], "no_selftest_reason": "n/a"}]}
        r6 = fixture(goal([{"path": "agent_harness/trails.txt"}]), bad_reg)
        rc, _, out = run_check(r6)
        case("F 閘門缺 not_proves ⇒ rc=1", rc == 1 and "not_proves" in out, out.strip()[-120:])

        # 7) fast 閘門沒有 selftest 也沒有理由 ⇒ rc 1
        bad2 = {"schema": 1, "groups": {"g": "g"}, "gates": [
            {"id": "x", "group": "g", "title": "t", "cmd": ["true"], "expect": "0",
             "proves": "p", "not_proves": "q", "cost": "fast"}]}
        r7 = fixture(goal([{"path": "agent_harness/trails.txt"}]), bad2)
        rc, _, out = run_check(r7)
        case("G fast 無 self-test 且無理由 ⇒ rc=1",
             rc == 1 and "no_selftest_reason" in out, out.strip()[-120:])

        # 8) 壞 JSON ⇒ rc 1（不是 traceback）
        r8 = fixture(goal([{"path": "agent_harness/trails.txt"}]))
        (r8 / "agent_harness" / "portal" / "gates.json").write_text("{not json", encoding="utf-8")
        rc, _, out = run_check(r8)
        case("H 壞 JSON ⇒ rc=1 且不 traceback",
             rc == 1 and "Traceback" not in out and "JSON" in out, out.strip()[-120:])

        # 9) --check 不得寫任何檔（陰性對照：先快照 mtime/size，跑完再比）
        r9 = fixture(goal([{"path": "agent_harness/trails.txt"}]),
                     {"schema": 1, "groups": {"g": "g"}, "gates": []})
        before = {str(p): p.stat().st_mtime_ns for p in r9.rglob("*") if p.is_file()}
        run_check(r9)
        after = {str(p): p.stat().st_mtime_ns for p in r9.rglob("*") if p.is_file()}
        case("I --check 不寫任何檔", before == after,
             f"新增 {set(after)-set(before)} 改動 {[k for k in before if k in after and before[k]!=after[k]]}")

        # 10) default：history 只追加（第二次執行不能蓋掉第一次）
        r10 = fixture(goal([{"path": "agent_harness/trails.txt"}]),
                      {"schema": 1, "groups": {"g": "g"}, "gates": []})
        run_me(r10, "--no-gates")
        run_me(r10, "--no-gates")
        hp = r10 / "agent_harness" / "portal" / "history.jsonl"
        lines = hp.read_text(encoding="utf-8").splitlines() if hp.exists() else []
        case("J history 只追加（兩次執行 ⇒ 兩行）", len(lines) == 2, f"{len(lines)} 行")

        # 11) portal HTML 必須產出，且含四個必要錨點
        html = r10 / "docs" / "AGENT_HARNESS_PORTAL.html"
        txt = html.read_text(encoding="utf-8") if html.exists() else ""
        anchors = ["心智圖", "驗證閘門", "資產盤點", "欠帳與復盤", "window.__PORTAL__"]
        miss = [a for a in anchors if a not in txt]
        case("K HTML 產出且四個分頁錨點齊全", html.exists() and not miss, f"缺 {miss}")

        # 12) --no-gates 時不能有任何 PASS（不跑就不准宣稱綠）
        d = json.loads((r10 / "agent_harness" / "portal" / "data.json").read_text(encoding="utf-8"))
        st = {r["status"] for r in d["gates"]["results"]}
        case("L --no-gates ⇒ 沒有任何 PASS（不跑不宣稱）", "PASS" not in st, f"status={st}")

        # 13) --gates-only 在有紅燈時必須 rc=1
        r13 = fixture(goal([{"path": "agent_harness/trails.txt"}]),
                      {"schema": 1, "groups": {"g": "g"}, "gates": [
                          {"id": "ok", "group": "g", "title": "t", "cmd": ["true"], "expect": "0",
                           "proves": "p", "not_proves": "q", "cost": "fast", "selftest": ["true"]},
                          {"id": "bad", "group": "g", "title": "t", "cmd": ["false"], "expect": "0",
                           "proves": "p", "not_proves": "q", "cost": "fast", "selftest": ["true"]}]})
        rc, _, out = run_me(r13, "--gates-only")
        case("M 有紅燈 ⇒ --gates-only rc=1 且列出 id",
             rc == 1 and "bad" in out, f"rc={rc}")

        # 14) heavy 閘門必須被標 SKIP 而不是 PASS
        r14 = fixture(goal([{"path": "agent_harness/trails.txt"}]),
                      {"schema": 1, "groups": {"g": "g"}, "gates": [
                          {"id": "h", "group": "g", "title": "t", "cmd": ["true"], "expect": "0",
                           "proves": "p", "not_proves": "q", "cost": "heavy", "needs": "GPU"}]})
        run_me(r14, "--no-gates")
        d = json.loads((r14 / "agent_harness" / "portal" / "data.json").read_text(encoding="utf-8"))
        gh = d["gates"]["results"][0]
        case("N heavy ⇒ SKIP（不畫成綠燈）", gh["status"] == "SKIP", gh["status"])

        # 15) O. 看門狗（見上面）：真正的 repo 一個檔都不能被動到
        after_real = snapshot_real()
        newf = sorted(set(after_real) - set(before_real))
        chg = sorted(k for k in before_real if k in after_real and before_real[k] != after_real[k])
        case("O 自我測試不寫任何檔到真正的 repo（子行程必須帶 PA_PORTAL_REPO）",
             not newf and not chg,
             f"新增 {[Path(x).name for x in newf]} 改動 {[Path(x).name for x in chg]}")

    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail[:90]}]" if not ok else ""))
    print(f"\n  {npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="產生 agent_harness portal（目標／決策／閘門／資產）")
    ap.add_argument("--check", action="store_true", help="只驗註冊表與目標樹指針，不寫任何檔")
    ap.add_argument("--self-test", action="store_true", help="黑箱自測（含陰性對照）")
    ap.add_argument("--gates-only", action="store_true", help="只跑快速閘門並印表；任何紅燈 ⇒ rc=1")
    ap.add_argument("--no-gates", action="store_true", help="不跑閘門（離線／趕時間）")
    ap.add_argument("--out", default=str(HTML_OUT), help=f"portal HTML 輸出路徑（預設 {HTML_OUT}）")
    ap.add_argument("--timeout", type=int, default=300, help="單條閘門的上限秒數（預設 300）")
    args = ap.parse_args()

    if args.self_test:
        return cmd_self_test()
    if args.check:
        return cmd_check()

    if not GATES_JSON.exists():
        print(f"[portal] 找不到 {GATES_JSON}", file=sys.stderr)
        return 1
    reg = load_gates()
    if args.gates_only:
        return cmd_gates_only(reg)
    goals = json.loads(GOALS_JSON.read_text(encoding="utf-8"))

    data = assemble(reg, goals, do_gates=not args.no_gates)

    prev = None
    if DATA_OUT.exists():
        try:
            prev = json.loads(DATA_OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = None
    data["previous"] = prev
    hist = load_jsonl(HISTORY)
    data["history"] = hist

    # 先算 HTML：render 失敗時不要留下半套產物（data.json 有了、portal 沒有，會讓下一次的
    # 「上一次」指向一個從未被人看過的狀態）。
    html = render_html(data)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.write_text(json.dumps(trim_for_disk(data), ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_row(data), ensure_ascii=False) + "\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    # ★ 驗自己的產物（見 verify_output 的說明）。壞掉就大聲，不要靜靜交一份壞頁面。
    issues = verify_output(html)
    if issues:
        print("[portal] ★ 產物自檢未過：", file=sys.stderr)
        for x in issues:
            print(f"[portal]   - {x}", file=sys.stderr)
        return 1
    print(f"[portal] 產物自檢通過（{len(html):,} B：無 markdown 殘留、標籤對稱、四個分頁齊、離線可開）")

    reds = [r["id"] for r in data["gates"]["results"] if r["status"] == "FAIL"]
    print(f"[portal] {out}")
    print(f"[portal] 閘門 {data['gates']['n_pass']}/{data['gates']['n_fast']} 綠"
          + (f"　紅燈: {', '.join(reds)}" if reds else "　（全綠）"))
    print(f"[portal] 欠帳 {len(data['obligations'])} 項　資產 LOC {data['assets']['code']['loc']:,}"
          f"　記憶 {data['assets']['memory']['count']} 檔 {data['assets']['memory']['bytes']:,} B"
          f"　skill {data['assets']['skills']['count']} 個")
    print(f"[portal] 輸入指紋 {data['fingerprint']['digest']}　history 共 {len(hist) + 1} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
