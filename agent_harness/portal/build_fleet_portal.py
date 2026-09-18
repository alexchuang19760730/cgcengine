#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""產品化的機隊與資產能力入口。

它回答三個問題，而且只回答它有量到的部分：
  1. 這台機器（與別的機器）**能做什麼** —— 每一條能力都指到一個真實檔案裡的子串
  2. **本日**的狀態 —— 每個端點的可達性、上報新鮮度、能力清單
  3. **七日內**的變化 —— 來源清單見 trend_sources.json，逐格標出處；沒量過的日子寫「無記錄」，不寫 0

設計上的硬規矩（與 AGENT_HARNESS_PORTAL 同一套，見 agent_harness/portal/README.md）
  1. 單一真相：端點清單只在 fleet.json、來源清單只在 trend_sources.json
  2. 不假裝：連不上／沒上報 ⇒ unknown／未上報，**不是紅燈**（一個會因為離線而變紅的
     入口，會在每一個離線的早晨說謊）
  3. 缺席要出聲：註冊表每個端點都要有一列；反過來，有檔而沒註冊也要紅
  4. 可執行檔而不是形容詞：每一格都帶來源；頁面上的復現指令與頁面自己跑的是同一份

四種模式（rc 契約與 build_portal.py 一致）
  --check      只驗註冊表／指針／契約，**不寫任何檔**；有問題 ⇒ rc=1 且逐條指名
  --self-test  突變式黑箱自測（19 格）
  --serve      起一個 stdlib HTTP 服務：GET / 即時重畫、GET /api/fleet、POST /api/report
  （預設）      產 docs/FLEET_PORTAL.html 並把這一輪的觀測追加到 fleet_status.jsonl
  --export     只寫 docs/fleet_export.json（給別的網站吃的可攜資料；見 export_payload 的 docstring）

用法
    python3 agent_harness/portal/build_fleet_portal.py                 # 產物
    python3 agent_harness/portal/build_fleet_portal.py --no-net        # 離線產物（不探測心跳）
    python3 agent_harness/portal/build_fleet_portal.py --check
    python3 agent_harness/portal/build_fleet_portal.py --self-test
    python3 agent_harness/portal/build_fleet_portal.py --serve --port 8787
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_portal as BP                       # noqa: E402  複用，不重造第二份真相
import report_endpoint_status as REP            # noqa: E402  契約規則的單一實作

REPO = BP.REPO
PORTAL_DIR = REPO / "agent_harness" / "portal"
FLEET_JSON = PORTAL_DIR / "fleet.json"
TREND_JSON = PORTAL_DIR / "trend_sources.json"
ENDPOINTS_DIR = PORTAL_DIR / "endpoints"
STATUS_JSONL = PORTAL_DIR / "fleet_status.jsonl"
HISTORY_JSONL = PORTAL_DIR / "history.jsonl"
M123_DIR = REPO / "Backup" / "m123_oracle_gate"
MEM_DIR = REPO / ".workbuddy" / "memory"
DOCS = REPO / "docs"
OUT_HTML = DOCS / "FLEET_PORTAL.html"
EXPORT_JSON = DOCS / "fleet_export.json"
EXPORT_SCHEMA = 1
HARNESS_HTML = DOCS / "AGENT_HARNESS_PORTAL.html"

hesc = BP.hesc
mdx = BP.mdx

STATUS_LIVE = "LIVE"
STATUS_FRESH = "已上報"
STATUS_STALE = "上報過期"
STATUS_NONE = "未上報"
STATUS_UNKNOWN = "未知"

# 顏色只用在「狀態」與圖表，不用來判斷好壞：
# 未上報／連不上是灰的，不是紅的 —— 它們不是失敗，是「不知道」。
CHIP = {
    STATUS_LIVE: "t-green",
    STATUS_FRESH: "t-green",
    STATUS_STALE: "t-amber",
    STATUS_NONE: "t-grey",
    STATUS_UNKNOWN: "t-grey",
}


# ────────────────────────────── 載入與驗證 ──────────────────────────────

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[error] 找不到 {p}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[error] {p} 不是合法 JSON: {e}")


def resolve_pointer(ev: dict, root: Path) -> tuple[bool, str]:
    """{path, contains} 指針：路徑要在、子串要命中。回 (ok, detail)。"""
    path = ev.get("path")
    need = ev.get("contains")
    if not path:
        return False, "指針缺 path"
    f = root / path
    if not f.exists():
        return False, f"指針指向不存在的路徑 {path}"
    if need is None:
        return True, ""
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return False, f"{path} 讀不到: {e}"
    if need not in text:
        return False, f"{path} 裡找不到子串 {need!r}"
    return True, ""


def load_artifacts() -> dict:
    """讀 endpoints/*.json（不含 README）。回 {id: {record, path}}。契約不合的照收，讓檢查去指名。"""
    out = {}
    if not ENDPOINTS_DIR.exists():
        return out
    for f in sorted(ENDPOINTS_DIR.glob("*.json")):
        try:
            out[f.stem] = {"record": json.loads(f.read_text(encoding="utf-8")), "path": f}
        except json.JSONDecodeError as e:
            out[f.stem] = {"record": None, "path": f, "json_error": str(e)}
    return out


def fleet_problems(fleet: dict, artifacts: dict, rows: list[dict],
                   spec: dict, trend: dict, root: Path) -> list[str]:
    """所有能靜默出錯的地方。兩個方向都要驗（見 README 第五條規矩）。

    ★ 注意參數有兩個來源物件：**spec 是註冊表**（唯一真相，要驗它的欄位），
     **trend 是推導結果**（只拿來比對「宣告的來源」與「渲染出的列」是否一致）。
     第一版把推導結果當成註冊表來驗，於是 `probe` 被報「缺欄位」—— 因為推導時
     本來就不會把它帶過來。檢查看錯對象時，症狀會長得像註冊表壞了。
    """
    P: list[str] = []

    if fleet.get("schema") != 1:
        P.append(f"fleet.json schema 要是 1，拿到 {fleet.get('schema')!r}")

    eps = fleet.get("endpoints") or []
    if not isinstance(eps, list) or not eps:
        P.append("fleet.json 的 endpoints 是空的")

    ids = [e.get("id") for e in eps if isinstance(e, dict)]
    dupes = sorted({i for i in ids if i and ids.count(i) > 1})
    if dupes:
        P.append(f"端點 id 重複: {dupes}")

    for e in eps:
        if not isinstance(e, dict):
            P.append(f"endpoints 裡有一項不是物件: {e!r}")
            continue
        eid = e.get("id") or "(無 id)"
        for k in ("id", "name", "kind", "platform", "arch", "role", "capabilities",
                  "channels", "not_measured"):
            if k not in e:
                P.append(f"[{eid}] 缺欄位 {k}")
        if eid != "(無 id)" and not re.fullmatch(r"[a-z0-9._-]+", str(eid)):
            P.append(f"[{eid}] id 只能用小寫英數與 -_. （它會變成檔名）")
        caps = e.get("capabilities") or []
        if not isinstance(caps, list) or not caps:
            P.append(f"[{eid}] capabilities 是空的 —— 一個沒有具名能力的端點不該出現在機隊表上")
        for c in caps if isinstance(caps, list) else []:
            evs = c.get("evidence") or []
            if not evs:
                P.append(f"[{eid}] 能力 {c.get('id')!r} 沒有 evidence（只能證不能否證的宣稱不是宣稱）")
            for ev in evs:
                ok, why = resolve_pointer(ev, root)
                if not ok:
                    P.append(f"[{eid}] 能力 {c.get('id')!r} 的指針失效: {why}")
        ch = e.get("channels") or {}
        if not isinstance(ch, dict) or "artifact" not in ch:
            P.append(f"[{eid}] channels 缺 artifact")
        elif ch["artifact"] != f"agent_harness/portal/endpoints/{eid}.json":
            P.append(f"[{eid}] channels.artifact 要是 agent_harness/portal/endpoints/{eid}.json，"
                     f"拿到 {ch['artifact']!r}")
        if not isinstance(e.get("not_measured"), list):
            P.append(f"[{eid}] not_measured 必須是陣列（可以是空的 —— 但空是一個主張）")

    # ★ 方向一：註冊表每一端都要在渲染出的表上有一列
    row_ids = [r["id"] for r in rows]
    for eid in ids:
        if eid and eid not in row_ids:
            P.append(f"★ 端點 {eid} 在註冊表裡，但渲染出的機隊表沒有它那一列（表上少一列，人不會去數）")
    # ★ 方向二：有上報檔但沒註冊，也要出聲
    for aid in sorted(artifacts):
        if aid not in ids:
            P.append(f"★ endpoints/{aid}.json 存在，但 fleet.json 沒有註冊 {aid} ⇒ 這份上報永遠不會被讀")
    # 契約驗證（複用產生器那份規則，不是第二份）
    for aid, a in sorted(artifacts.items()):
        if a.get("record") is None:
            P.append(f"endpoints/{aid}.json 不是合法 JSON: {a.get('json_error')}")
            continue
        for prob in REP.validate_record(a["record"]):
            P.append(f"endpoints/{aid}.json 不合契約: {prob}")
        if a["record"].get("endpoint_id") != aid:
            P.append(f"endpoints/{aid}.json 的 endpoint_id 是 {a['record'].get('endpoint_id')!r} ⇒ 與檔名不符")

    # 趨勢來源：宣告的每一個都要在表上有一列，且它的 probe 要真的過
    srcs = (spec.get("sources") or [])
    if not srcs:
        P.append("trend_sources.json 的 sources 是空的")
    declared = [s.get("id") for s in srcs]
    for s in srcs:
        for k in ("id", "label", "unit", "how", "day_basis", "probe"):
            if k not in s:
                P.append(f"趨勢來源 {s.get('id')!r} 缺欄位 {k}")
        ok, why = probe_source(s.get("probe") or {}, root)
        if not ok:
            P.append(f"趨勢來源 {s.get('id')!r} 的 probe 失敗: {why}")
    for sid in declared:
        if sid not in (trend.get("_rendered_ids") or declared):
            P.append(f"★ 趨勢來源 {sid} 宣告了但沒有渲染出一列")
    if trend.get("_rendered_ids") is not None and sorted(trend["_rendered_ids"]) != sorted(declared):
        P.append(f"★ 趨勢表的列與宣告不符: 宣告 {sorted(declared)} vs 表上 {sorted(trend['_rendered_ids'])}")
    return P


def probe_source(p: dict, root: Path) -> tuple[bool, str]:
    """來源健康檢查。刻意很便宜、且完全不連網。"""
    kind = p.get("kind")
    target = root / str(p.get("target", ""))
    if kind == "git":
        r = subprocess.run(p.get("argv") or ["git", "rev-parse", "--git-dir"],
                           cwd=str(root), capture_output=True, text=True)
        return (r.returncode == 0), (r.stderr.strip()[:120] or "不是 git repo")
    if kind == "file":
        return target.is_file(), f"{p.get('target')} 不存在"
    if kind == "dir":
        return target.is_dir(), f"{p.get('target')} 不存在"
    if kind == "glob":
        hits = sorted(target.parent.glob(target.name)) if target.parent.exists() else []
        n = len(hits)
        need = int(p.get("min_count", 1))
        return (n >= need), f"命中 {n} 個，需要 ≥{need}"
    if kind == "jsonl":
        if not target.is_file():
            return False, f"{p.get('target')} 不存在"
        req = p.get("require_keys") or []
        for i, line in enumerate(target.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                return False, f"第 {i+1} 行不是合法 JSON"
            for k in req:
                if k not in o:
                    return False, f"第 {i+1} 行缺 {k}"
        return True, ""
    return False, f"未知的 probe kind {kind!r}"


# ────────────────────────────── 心跳 ──────────────────────────────

def probe_heartbeat(url: str, timeout: float) -> dict:
    """探測端點的 pd compute_sharing 心跳。

    ★ 失敗一律是 unknown，不是紅燈：連不上只說明「現在不知道」，不說明「它壞了」。
    """
    t0 = datetime.now()
    try:
        with urlopen(Request(url, headers={"User-Agent": "cgc-fleet-portal"}), timeout=timeout) as r:
            body = r.read(2000).decode("utf-8", "replace")
        ms = int((datetime.now() - t0).total_seconds() * 1000)
        return {"state": "live", "detail": f"HTTP {r.status}（{ms} ms）", "body": body[:400]}
    except HTTPError as e:
        return {"state": "unknown", "detail": f"HTTP {e.code} —— 有回應但不是 200，仍是 unknown"}
    except (URLError, socket.timeout, OSError) as e:
        reason = getattr(e, "reason", e)
        return {"state": "unknown", "detail": f"{type(e).__name__}: {str(reason)[:90]} ⇒ 不知道，不是沒有"}


# ────────────────────────────── 機隊 ──────────────────────────────

def age_text(iso: str, now: datetime) -> tuple[float | None, str]:
    try:
        t = datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None, "時間格式無法解析"
    secs = (now - t).total_seconds()
    if secs < 0:
        return secs, f"未來時間（{-secs/3600:.1f} 小時後）"
    if secs < 3600:
        return secs, f"{int(secs/60)} 分鐘前"
    if secs < 86400:
        return secs, f"{secs/3600:.1f} 小時前"
    return secs, f"{secs/86400:.1f} 天前"


def fleet_rows(fleet: dict, artifacts: dict, do_net: bool, timeout: float,
               now: datetime | None = None) -> list[dict]:
    """一個端點一列 —— 註冊表有幾筆就幾筆，永遠不會少。"""
    now = now or datetime.now()
    fresh = fleet.get("freshness") or {}
    rows = []
    for e in fleet.get("endpoints") or []:
        eid = e.get("id")
        art = artifacts.get(eid)
        rec = (art or {}).get("record")
        hb = (e.get("channels") or {}).get("heartbeat")
        if not do_net:
            heart = {"state": "skipped", "detail": "本輪 --no-net：不探測（跳過 ≠ 通過）"}
        elif not hb:
            heart = {"state": "none", "detail": "註冊表沒有給心跳位址 ⇒ 這一格永遠是 unknown"}
        else:
            heart = probe_heartbeat(hb["url"], float(hb.get("timeout_s", timeout)))

        if rec is None:
            chip, agetxt, age = STATUS_NONE, "沒有上報過", None
            stale_class = "none"
        else:
            age, agetxt = age_text(rec.get("reported_at", ""), now)
            if age is None:
                chip, stale_class = STATUS_UNKNOWN, "unknown"
            elif age <= float(fresh.get("live_s", 900)):
                chip, stale_class = STATUS_FRESH, "live"
            elif age <= float(fresh.get("stale_s", 604800)):
                chip, stale_class = STATUS_STALE, "recent"
            else:
                chip, stale_class = STATUS_STALE, "stale"
            if heart.get("state") == "live":
                chip = STATUS_LIVE
        rows.append({
            "id": eid, "name": e.get("name", eid), "kind": e.get("kind", ""),
            "platform": e.get("platform", ""), "arch": e.get("arch", ""),
            "role": e.get("role", ""), "reaches": e.get("reaches", ""),
            "capabilities": e.get("capabilities") or [],
            "not_measured": e.get("not_measured") or [],
            "channels": e.get("channels") or {},
            "heartbeat": heart,
            "report": rec, "report_age": agetxt, "report_seconds": age,
            "report_path": f"agent_harness/portal/endpoints/{eid}.json" if rec else "",
            "chip": chip, "chip_class": CHIP.get(chip, "t-grey"), "stale_class": stale_class,
        })
    return rows


# ────────────────────────────── 七日趨勢 ──────────────────────────────

def days_window(n: int, today: datetime | None = None) -> list[str]:
    t = (today or datetime.now()).date()
    return [(t - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def _src_commits(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    r = subprocess.run(["git", "log", "--pretty=%ad", "--date=short"], cwd=str(root),
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {d: None for d in days}, None, r.stderr.strip()[:80]
    got = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    counts = {}
    for g in got:
        counts[g] = counts.get(g, 0) + 1
    first = min(counts) if counts else None
    return {d: (None if (first is None or d < first) else counts.get(d, 0)) for d in days}, first, \
        "git log --pretty=%ad --date=short（author date；0 是量到的 0，不是沒量）"


def _src_m123(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    d = root / "Backup" / "m123_oracle_gate"
    files = sorted(d.glob("summary_*.json")) if d.exists() else []
    per: dict[str, dict] = {}
    for f in files:
        day = datetime.fromtimestamp(f.stat().st_mtime).date().isoformat()
        b = per.setdefault(day, {"runs": 0, "id_ok": 0, "id_n": 0})
        b["runs"] += 1
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for key in ("m1_numeric_identity",):
            v = str(j.get(key, ""))
            m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", v.strip())
            if m:
                b["id_ok"] += int(m.group(1))
                b["id_n"] += int(m.group(2))
    first = min(per) if per else None
    cells = {d: (None if (first is None or d < first) else per.get(d, {}).get("runs", 0)) for d in days}
    detail = (f"Backup/m123_oracle_gate/summary_*.json 逐檔用 **mtime** 歸日"
              f"（summary 內沒有時間欄位）；M1 身分以「分子/分母」加總。"
              f"此來源最早可追溯到 **{first or '（整週沒有資料）'}**。")
    return cells, first, detail


def _src_decisions(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    p = root / "agent_harness" / "engine_loop" / "traces" / "decisions.jsonl"
    if not p.is_file():
        return {d: None for d in days}, None, "decisions.jsonl 不存在"
    counts: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            did = json.loads(line).get("decision_id", "")
        except json.JSONDecodeError:
            continue
        m = re.match(r"dec-(\d{8})-", did)
        if m:
            k = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
            counts[k] = counts.get(k, 0) + 1
    first = min(counts) if counts else None
    return {d: (None if (first is None or d < first) else counts.get(d, 0)) for d in days}, first, \
        "decision_id 內嵌的 dec-YYYYMMDD- 前綴（寫進 id 的那一天，不是檔案 mtime）"


def _src_daylog(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    counts, first = {}, None
    for d in days:
        f = root / ".workbuddy" / "memory" / f"{d}.md"
        if f.is_file():
            counts[d] = f.stat().st_size
            first = first or d
    return {d: counts.get(d) for d in days}, first, \
        ".workbuddy/memory/<date>.md 的位元組數（檔名日期）。它回答「那天有沒有人在場」，不是產出量"


def _src_snapshot(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    p = root / "agent_harness" / "portal" / "history.jsonl"
    if not p.is_file():
        return {d: None for d in days}, None, "history.jsonl 不存在"
    counts: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ts = json.loads(line).get("ts", "")
        except json.JSONDecodeError:
            continue
        if len(ts) >= 10:
            counts[ts[:10]] = counts.get(ts[:10], 0) + 1
    first = min(counts) if counts else None
    return {d: (None if (first is None or d < first) else counts.get(d, 0)) for d in days}, first, \
        "history.jsonl 的 ts（每次 AGENT_HARNESS_PORTAL 跑完追加一行）"


def _src_endpoint_report(root: Path, days: list[str]) -> tuple[dict, str | None, str]:
    d = root / "agent_harness" / "portal" / "endpoints"
    counts: dict[str, int] = {}
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                ts = json.loads(f.read_text(encoding="utf-8")).get("reported_at", "")
            except json.JSONDecodeError:
                continue
            if len(ts) >= 10:
                counts[ts[:10]] = counts.get(ts[:10], 0) + 1
    first = min(counts) if counts else None
    return {d_: (None if (first is None or d_ < first) else counts.get(d_, 0)) for d_ in days}, first, \
        "endpoints/*.json 的 reported_at（端點自己寫的時間）"


SRC_FUNCS = {
    "commits": _src_commits,
    "m123": _src_m123,
    "decisions": _src_decisions,
    "daylog": _src_daylog,
    "snapshot": _src_snapshot,
    "endpoint_report": _src_endpoint_report,
}


def trend_block(spec: dict, root: Path, today: datetime | None = None) -> dict:
    days = days_window(int(spec.get("window_days", 7)), today)
    missing = spec.get("missing_label", "無記錄")
    series = []
    for s in spec.get("sources") or []:
        sid = s.get("id")
        fn = SRC_FUNCS.get(sid)
        if fn is None:
            cells, first, detail = {d: None for d in days}, None, f"沒有實作 {sid}"
        else:
            cells, first, detail = fn(root, days)
        got = sum(1 for d in days if cells.get(d) is not None)
        series.append({
            "id": sid, "label": s.get("label", sid), "bucket": s.get("bucket", ""),
            "unit": s.get("unit", ""), "how": s.get("how", ""), "day_basis": s.get("day_basis", ""),
            "why_not": s.get("not_a_capability_because", ""),
            "detail": detail, "first": first, "covers": got,
            "cells": {d: cells.get(d) for d in days},
            "display": {d: (missing if cells.get(d) is None else f"{cells[d]}{s.get('unit','')}")
                        for d in days},
        })
    return {"days": days, "missing_label": missing, "sources": series,
            "_rendered_ids": [s["id"] for s in series],
            "window_days": int(spec.get("window_days", 7))}


# ────────────────────────────── HTML ──────────────────────────────

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
th,td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:14px 0 6px; }
.kpi { border:1px solid var(--line); border-radius:6px; padding:9px 11px; background:var(--soft); }
.kpi .v { font-size:21px; font-weight:600; font-variant-numeric:tabular-nums; }
.kpi .k { font-size:12px; color:var(--muted); }
.tag { display:inline-block; font-size:11.5px; padding:1px 8px; border-radius:999px;
       border:1px solid var(--line); background:var(--soft); }
.t-green{color:var(--green);border-color:#bbf7d0;background:#f0fdf4}
.t-amber{color:var(--amber);border-color:#fde68a;background:#fffbeb}
.t-grey{color:var(--muted);border-color:var(--line);background:var(--soft)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin:12px 0 18px}
.card{border:1px solid var(--line);border-radius:6px;padding:12px 14px;background:var(--soft)}
.box { border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:4px;
       padding:11px 13px; margin:14px 0; }
.box.bad { border-left-color:var(--red); background:#fff7f7; }
.box.warn { border-left-color:var(--amber); background:#fffdf5; }
.box.ok { border-left-color:var(--green); background:#f6fdf8; }
.small { font-size:12.3px; color:var(--muted); }
.muted { color:var(--muted); }
details { margin:6px 0; }
summary { cursor:pointer; font-size:13px; color:var(--blue); }
.ev { font-size:11.8px; color:var(--muted); font-family:ui-monospace,Menlo,monospace; word-break:break-all; }
.chart { margin:6px 0 14px; }
.foot { margin-top:34px; padding-top:12px; border-top:1px solid var(--line); font-size:12.3px; color:var(--muted); }
"""


def bar_chart(days: list[str], values: list[int | None], label: str, unit: str, color: str) -> str:
    """內嵌 SVG 長條圖。None（無記錄）畫成空的虛線格 —— 一眼看得出「沒量」與「量到 0」。"""
    w, h, pad = 640, 132, 30
    n = max(len(days), 1)
    bw = (w - pad * 2) / n
    mx = max([v for v in values if v is not None] + [1])
    bars, grid = [], []
    for i, (d, v) in enumerate(zip(days, values)):
        x = pad + i * bw
        if v is None:
            bars.append(f'<rect x="{x+3:.1f}" y="{pad}" width="{bw-6:.1f}" height="{h-pad-20}" '
                        f'fill="none" stroke="#d1d5db" stroke-dasharray="3 3"/>')
            bars.append(f'<text x="{x+bw/2:.1f}" y="{h-6}" font-size="9" fill="#9ca3af" '
                        f'text-anchor="middle">{hesc(d[5:])}</text>')
            continue
        bh = (h - pad - 20) * (v / mx) if mx else 0
        y = h - 20 - bh
        bars.append(f'<rect x="{x+3:.1f}" y="{y:.1f}" width="{bw-6:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{y-3:.1f}" font-size="10" fill="#374151" '
                    f'text-anchor="middle">{v}</text>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{h-6}" font-size="9" fill="#6b7280" '
                    f'text-anchor="middle">{hesc(d[5:])}</text>')
    grid.append(f'<text x="{pad}" y="{pad-9}" font-size="10.5" fill="#6b7280">'
                f'{hesc(label)}（最大 {mx}{hesc(unit)}）· 空框 = 無記錄</text>')
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="{hesc(label)}">{"".join(grid)}{"".join(bars)}</svg>')


def render_html(fleet, rows, trend, momentum, meta) -> str:
    days = trend["days"]
    miss = trend["missing_label"]

    kpi = [
        ("端點（註冊）", str(len(rows))),
        ("心跳可達", str(sum(1 for r in rows if r["heartbeat"]["state"] == "live"))),
        ("已上報", str(sum(1 for r in rows if r["report"] is not None))),
        ("未上報", str(sum(1 for r in rows if r["report"] is None))),
        (f"{len(days)} 日 commit", str(sum(v for v in trend["sources"][0]["cells"].values() if v is not None))),
        ("未量測項", str(sum(len(r["not_measured"]) for r in rows))),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{hesc(v)}</div>'
                       f'<div class="k">{hesc(k)}</div></div>' for k, v in kpi)

    # 機隊卡
    cards = []
    for r in rows:
        caps = "".join(
            f'<li><b>{mdx(hesc(c.get("label","")))}</b>'
            + "".join(f'<div class="ev">↳ {hesc(e.get("path",""))} :: {hesc(e.get("contains",""))}</div>'
                      for e in (c.get("evidence") or []))
            + "</li>" for c in r["capabilities"])
        nm = "".join(f"<li>{mdx(hesc(x))}</li>" for x in r["not_measured"]) or \
             "<li class='muted'>（這個端點宣告沒有未量的東西）</li>"
        # ★ 這兩塊是「上報裡最有用的一半」，第一版卻只載入不顯示。
        ran = "".join(f"<li>{mdx(hesc(x))}</li>" for x in (r["report"] or {}).get("what_ran") or []) \
              or "<li class='muted'>未上報 ⇒ 不知道它跑了什麼</li>"
        mets = (r["report"] or {}).get("metrics") or {}
        metrics = ("　".join(f"<b>{hesc(k)}</b>={hesc(v)}" for k, v in mets.items()) if mets
                   else "<span class='muted'>（沒有讀數）</span>")
        ch = r["channels"]
        hb = ch.get("heartbeat")
        hb_txt = (f'<code>{hesc(hb["url"])}</code><div class="ev">via {hesc(hb.get("how",""))}</div>'
                  if hb else '<span class="muted">沒有心跳位址（這一格永遠是 unknown）</span>')
        rel = "".join(f'<li>{mdx(hesc(x.get("how","")))}</li>' for x in (ch.get("relay") or []))
        rep = (f'<b>已上報</b> {hesc(r["report_age"])}　<code>{hesc(r["report_path"])}</code>'
               if r["report"] else '<span class="muted">未上報（沒有任何 endpoints/*.json）</span>')
        cards.append(f"""<div class="card">
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline">
    <div><b>{mdx(hesc(r['name']))}</b>
      <div class="small"><code>{hesc(r['id'])}</code>　{hesc(r['platform'])}/{hesc(r['arch'])}　
      <span class="tag t-grey">{hesc(r['kind'])}</span>
      <span class="tag t-grey">reaches={hesc(r['reaches'] or '?')}</span></div>
    </div>
    <span class="tag {r['chip_class']}">{hesc(r['chip'])}</span>
  </div>
  <div class="small" style="margin-top:6px">{mdx(hesc(r['role']))}</div>
  <div style="margin-top:8px"><b>能力</b>（每一條都指到真實檔案裡的子串）</div>
  <ul style="margin:4px 0 0;padding-left:18px;font-size:13px">{caps}</ul>
  <div style="margin-top:8px"><b>未量測</b>（缺席要出聲）</div>
  <ul style="margin:4px 0 0;padding-left:18px;font-size:12.6px">{nm}</ul>
  <div style="margin-top:8px"><b>本輪真的跑了什麼</b>（來自它的上報，不是我們的推測）</div>
  <ul style="margin:4px 0 0;padding-left:18px;font-size:13px">{ran}</ul>
  <div style="margin-top:6px"><b>讀數</b><div class="ev">{metrics}</div></div>
  <div style="margin-top:8px"><b>通道</b><div class="ev">心跳：{hb_txt}</div>
    <div class="ev">探測結果：{hesc(r['heartbeat'].get('state',''))} —— {hesc(str(r['heartbeat'].get('detail',''))[:150])}</div>
    <ul style="margin:4px 0 0;padding-left:18px;font-size:12.6px">{rel}</ul>
    <div style="margin-top:4px;font-size:12.6px">{rep}</div>
  </div>
</div>""")

    # 七日表
    head = "".join(f"<th class='num'>{hesc(d[5:])}</th>" for d in days)
    trows = []
    for s in trend["sources"]:
        cells = "".join(
            f'<td class="num">{hesc(s["display"][d])}</td>' if s["cells"][d] is not None
            else f'<td class="num muted" title="此來源這天沒有資料 ⇒ 不是 0">{hesc(miss)}</td>'
            for d in days)
        # ★ 這裡一定要 hesc()：「.workbuddy/memory/<date>.md」這種文字裡的 <date>
        #   會被瀏覽器當成一個標籤吃掉，而且會讓後面所有 </td></tr> 全部錯位
        #   （2026-09-18 實際發生：先看到的是「表格欄數不對稱」，那是症狀不是原因）。
        prov = (f'<details><summary>怎麼來的</summary><div class="small">'
                f'<b>取法</b>：{mdx(hesc(s["how"]))}<br>'
                f'<b>歸日依據</b>：{mdx(hesc(s["day_basis"]))}<br>'
                f'<b>為什麼它不是能力</b>：{mdx(hesc(s["why_not"]))}</div></details>')
        trows.append(
            f'<tr><td><b>{hesc(s["label"])}</b><div class="small">{hesc(s["bucket"])}　'
            f'<code>{hesc(s["id"])}</code></div></td>{cells}'
            f'<td class="small">覆蓋 {s["covers"]}/{len(days)}　|　{mdx(hesc(s["detail"]))}{prov}</td></tr>')
    commits = trend["sources"][0]
    m123 = next((s for s in trend["sources"] if s["id"] == "m123"), commits)
    charts = (bar_chart(days, [commits["cells"][d] for d in days], "① commit", " 個", "#1d4ed8")
              + bar_chart(days, [m123["cells"][d] for d in days], "② M1/M2/M3 oracle 實跑", " 次", "#6d28d9"))

    mom = momentum
    mom_rows = "".join(
        f'<tr><td><code>{hesc(x["id"])}</code></td><td>{hesc(x["day"])}</td>'
        f'<td><span class="tag {"t-grey" if x["momentum"] in ("orphan","pending") else "t-green"}">'
        f'{hesc(x["momentum"])}</span></td>'
        f'<td class="num">{len(x["forward_refs"])}</td></tr>' for x in mom["rows"][:24])

    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CGC 機隊與資產能力入口</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<h1>CGC 機隊與資產能力入口</h1>
<p class="sub">
產生於 <b>{hesc(meta['generated_at'])}</b>　·　
<code>{hesc(meta['head'])}</code>（{hesc(meta['branch'])}）{mdx(hesc(meta['subject'][:70]))}　·　
未提交 <b>{hesc(meta['dirty'])}</b> 檔　·　
模式 <span class="tag t-grey">{hesc(meta['mode'])}</span>　·　
<a href="{hesc(meta['harness_href'])}">工程面入口（目標／決策／閘門／資產）→</a>
</p>

<div class="kpis">{kpi_html}</div>

<div class="box {'warn' if meta['offline'] else 'ok'}">
  <div><b>一句話</b></div>
  <div style="margin-top:5px">
    {hesc(meta['oneline'])}
  </div>
  <div class="small" style="margin-top:7px">
    一鍵復現：<code>python3 agent_harness/portal/build_fleet_portal.py</code>
    （離線：<code>… --no-net</code>）　·　只驗不寫：<code>… --check</code>　·　
    自我陰性對照：<code>… --self-test</code>　·　即時服務：<code>… --serve --port 8787</code>
  </div>
</div>

<h2>① 機隊 —— 每個端點一列，連不上的也在表上</h2>
<div class="small">狀態只回答「我現在知不知道」，不回答「它好不好」：
<span class="tag t-green">LIVE</span> 心跳有回應　
<span class="tag t-green">已上報</span> 有上報且新鮮　
<span class="tag t-amber">上報過期</span> 有上報但舊了　
<span class="tag t-grey">未上報</span> 從來沒收到過（<b>不是紅燈</b>）。</div>
<div class="cards">{''.join(cards)}</div>

<h2>② {len(days)} 日 —— {len(trend["sources"])} 個來源，逐格標出處</h2>
<div class="small">「{hesc(miss)}」與 <b>0</b> 是不同的兩件事：0 是「量到沒有」，{hesc(miss)} 是「根本沒量」。
每一列都寫了它的來源與歸日依據。{hesc(trend.get('note',''))}</div>
{charts}
<table>
<thead><tr><th>來源</th>{head}<th>覆蓋與歸日依據</th></tr></thead>
<tbody>{''.join(trows)}</tbody></table>

<h2>③ 資產動能（決策被後續工作用上的程度）</h2>
<div class="small">{mdx(hesc(mom['note']))}</div>
<div style="margin:8px 0">
{"".join(f'<span class="tag {"t-grey" if k in ("orphan","pending") else "t-green"}">{hesc(k)} {v}</span>　' for k, v in mom['summary'].items())}
</div>
<div class="small">refuted {hesc(mom['refuted_pct'])}（門檻 {hesc(mom['thr_refuted'])}）　·　
orphan {hesc(mom['orphan_pct'])}（門檻 {hesc(mom['thr_orphan'])}）　·　
判定 <span class="tag {'t-amber' if mom['red'] else 't-green'}">{hesc(mom['verdict'])}</span></div>
<table><thead><tr><th>decision_id</th><th>日</th><th>動能</th><th class="num">被引用</th></tr></thead>
<tbody>{mom_rows}</tbody></table>

<h2>④ 怎麼把一個離線端接進來</h2>
<div class="small">鴻蒙端與 Windows 端<b>目前不在我們的網路上</b> ⇒ 沒有任何即時協議能成立。
唯一在它們真的離線時仍然成立的機制是 store-and-forward：端點寫檔，靠既有通道帶回來。</div>
<pre><code># 在端點上跑（只需 Python 標準函式庫；沒有這個 repo 也能跑）
python3 agent_harness/portal/report_endpoint_status.py --id windows-rtx4090 \\
    --what-ran "建 CUDA 版（GGML_CUDA=ON）" --metric decode_tps=52.3 \\
    --not-measured "沒量過 M1/M2/M3 身分"

# 讓它落到 endpoints/ 之後，靠既有通道帶回來：
#   鴻蒙端  scp ＋ ssh          → deploy-harmonyos/deploy-to-harmonyos.sh
#   Windows git（自動 push）    → agent_harness/scripts/auto_git_push.ps1
#   雲端    rsync               → CGC-main/cgc_engine/tools/_archive_v1/server/sync_all_gates_to_hosts.sh
# 若端點真的連得上，也可以直接 POST 給本入口：
#   curl -X POST http://127.0.0.1:8787/api/report -d @endpoints/windows-rtx4090.json</code></pre>
<div class="box warn"><b>兩個對齊問題（不說就會靜默失敗）</b>
<div style="margin-top:5px">
① <code>auto_git_push.ps1</code> 推的是 branch <code>fusionroutemot</code>，而本入口讀的是
<code>demo/sweet-spot-windows-fix</code> ⇒ 兩邊對齊之前 Windows 的上報不會出現在這裡。<br>
② <code>auto_git_push.ps1</code> 在本機<b>從未被執行過</b> ⇒「它會自動推」目前是設計意圖，不是已驗證的行為。
</div></div>

<h2>⑤ 這一頁的可信度邊界</h2>
<ul style="font-size:13.2px">
<li>本頁顯示的<b>閘門綠燈是「最後一次實跑」的記錄</b>，不是剛剛跑的 ——
引擎面入口（<code>build_portal.py --gates-only</code>）才是跑閘門的地方。</li>
<li>心跳探測的失敗一律寫 <b>unknown</b>，不是紅燈：連不上只說明「現在不知道」。</li>
<li>能力的每一條都指到<b>真實檔案裡的子串</b>（見卡片下的 <code>↳</code> 行）——
指針失效會讓 <code>--check</code> 變紅，不會靜默。</li>
<li>{hesc(meta['net_note'])}</li>
</ul>

<div class="foot">
來源：<code>agent_harness/portal/fleet.json</code>（端點，唯一真相）＋
<code>trend_sources.json</code>（七日來源，唯一真相）＋
<code>endpoints/*.json</code>（端點上報）＋ 複用 <code>build_portal.py</code> 的動能與跨機判定。<br>
本頁不連任何外部服務；所有數字都可在本機用上面那三條指令重跑。
</div>

</div></body></html>
"""


VOID = {"meta", "br", "hr", "img", "link", "input", "area", "base", "col", "embed",
        "param", "source", "track", "wbr",
        # SVG
        "rect", "line", "circle", "ellipse", "path", "polygon", "polyline", "use",
        "stop", "image", "marker"}


def verify_output(html: str) -> list[str]:
    """產物自我檢查：標籤要配對、不得有 markdown 殘留。

    ★ 這一支是被一個真實缺陷逼出來的：`detail` 文字裡的 `<date>`（來自
      「.workbuddy/memory/<date>.md」）沒有轉義，被當成標籤 —— 而症狀先以
      「表格欄數不對稱」出現。有了這支，那一類缺陷會在**寫檔之前**自己現形。
    """
    # ★ 一定要用別名：寫 `import html.parser` 會把參數 `html` 覆蓋成模組，
    #   於是下面 ps.feed(html) 餵進去的是模組 ⇒ TypeError（自測第 12 格當場抓到）。
    import html.parser as _hp
    problems: list[str] = []

    class P(_hp.HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list = []
            self.err: list = []

        def handle_starttag(self, tag, attrs):
            if tag not in VOID:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag):
            if tag in VOID:
                return
            if self.stack and self.stack[-1][0] == tag:
                self.stack.pop()
            else:
                self.err.append((tag, self.getpos(), self.stack[-1:] ))

    ps = P()
    ps.feed(html)
    for tag, pos, top in ps.err[:5]:
        problems.append(f"行為異常的結束標籤 </{tag}> 在第 {pos[0]} 行第 {pos[1]} 字（堆疊頂端 {top}）"
                        f" ⇒ 幾乎都是某段文字沒轉義（<...> 被當成標籤）")
    for tag, pos in ps.stack[:5]:
        problems.append(f"未閉合的標籤 <{tag}> 在第 {pos[0]} 行第 {pos[1]} 字")

    stripped = re.sub(r"<(pre|code)\b[^>]*>.*?</\1>", "", html, flags=re.S)
    for bad in re.findall(r"\*\*[^*\n]{1,60}\*\*|`[^`\n]{1,60}`", stripped)[:5]:
        problems.append(f"markdown 殘留（沒走 mdx）：{bad!r}")
    return problems


def momentum_block() -> dict:
    """複用 build_portal 的動能實算 —— 連門檻判定也不自己算（不重造第二份真相）。

    ★ 2026-09-18 踩到的坑：targets.json 的 `momentum.declared_red` 用的是
      `refuted_rate_gt: 0.25` / `orphan_rate_gt: 0.5` —— 那是**比例**，不是百分點。
      第一版自己去讀 `refuted_rate` / `orphan_rate` 這兩個**不存在的鍵**，預設 25／50，
      於是顯示的門檻差 **100 倍**，而 `red` 的判定會因此反過來（真實是「動能不足」卻顯示綠）。
      現在：比率與紅綠一律取自 `momentum_summary()`（它用的是同一份 declared），
      本函式只負責把比例轉成人看的百分比，不再持有第二套門檻。
    """
    dec = BP.load_jsonl(BP.TRACES / "decisions.jsonl")
    rows = BP.momentum_rows(dec)
    tg = BP.load_targets()
    declared = (tg.get("momentum") or {}).get("declared_red") or {}
    summ = BP.momentum_summary(rows, declared)
    counts = summ.get("counts") or {}
    thr_r = declared.get("refuted_rate_gt")
    thr_o = declared.get("orphan_rate_gt")
    pct = lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "未宣告"   # noqa: E731
    return {
        "rows": rows, "summary": counts,
        "note": (declared.get("note") or "動能＝被後續決策納入的程度"
                 "（carried／replaced／refuted／orphan／pending），從 decisions.jsonl 實算，不是手填。"
                 ).replace("**", ""),
        "refuted_pct": pct(summ.get("refuted_rate")), "orphan_pct": pct(summ.get("orphan_rate")),
        "thr_refuted": pct(thr_r), "thr_orphan": pct(thr_o),
        "red": bool(summ.get("red")),
        "verdict": "動能不足" if summ.get("red") else "動能尚可",
    }


# ────────────────────────────── 觀測（只追加） ──────────────────────────────

def append_status(rows: list[dict], path: Path) -> dict:
    """把這一輪的觀測追加到 fleet_status.jsonl。

    ★ 只追加、永不覆蓋：這是趨勢的唯一來源；endpoints/*.json 才是「現況」。
    """
    rec = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": [
            {"id": r["id"], "chip": r["chip"], "heartbeat": r["heartbeat"]["state"],
             "report_age": r["report_age"], "caps": len(r["capabilities"]),
             "unmeasured": len(r["not_measured"])}
            for r in rows],
        "live": sum(1 for r in rows if r["heartbeat"]["state"] == "live"),
        "reported": sum(1 for r in rows if r["report"] is not None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ────────────────────────────── 模式 ──────────────────────────────

def gather(do_net: bool, timeout: float, today: datetime | None = None) -> dict:
    fleet = load_json(FLEET_JSON)
    spec = load_json(TREND_JSON)
    artifacts = load_artifacts()
    rows = fleet_rows(fleet, artifacts, do_net, timeout, today)
    trend = trend_block(spec, REPO, today)
    problems = fleet_problems(fleet, artifacts, rows, spec, trend, REPO)
    return {"fleet": fleet, "spec": spec, "artifacts": artifacts,
            "rows": rows, "trend": trend, "problems": problems}


def git_meta() -> dict:
    def g(*a):
        return subprocess.run(list(a), cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    return {
        "head": g("git", "rev-parse", "--short", "HEAD") or "?",
        "branch": g("git", "rev-parse", "--abbrev-ref", "HEAD") or "?",
        "subject": g("git", "log", "-1", "--pretty=%s") or "",
        "dirty": len([l for l in g("git", "status", "--porcelain", "-uall").splitlines() if l.strip()]),
    }


def make_meta(g: dict, do_net: bool) -> dict:
    rows, trend = g["rows"], g["trend"]
    live = sum(1 for r in rows if r["heartbeat"]["state"] == "live")
    rep = sum(1 for r in rows if r["report"] is not None)
    off = [s for s in trend["sources"] if s["covers"] == 0]
    oneline = (f"註冊 {len(rows)} 個端點：{live} 個心跳可達、{rep} 個有上報、"
               f"{len(rows) - rep} 個未上報（未上報不是壞消息 —— 鴻蒙端與 Windows 端本來就不在網路上）。"
               f"{len(trend['days'])} 日窗內有 {len(trend['sources']) - len(off)}/{len(trend['sources'])} 個來源有資料。")
    if off:
        oneline += f"整週沒有資料的來源：{'、'.join(s['label'] for s in off)}。"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "live（有探測心跳）" if do_net else "offline（--no-net，未探測心跳）",
        "offline": not do_net,
        "oneline": oneline,
        "net_note": ("本輪有探測心跳：失敗者已寫成 unknown。"
                     if do_net else
                     "本輪 --no-net：<b>沒有探測任何心跳</b>，所以「心跳可達」顯示 0 是「沒問過」而不是「不通」。"),
        "harness_href": HARNESS_HTML.name if HARNESS_HTML.exists() else "#",
        **git_meta(),
    }


def cmd_check() -> int:
    """只驗不寫。任何缺陷 ⇒ rc=1 且逐條指名。"""
    g = gather(do_net=False, timeout=1.0)
    if g["problems"]:
        print(f"  [error] {len(g['problems'])} 項缺陷:")
        for p in g["problems"]:
            print(f"    {p}")
        return 1
    print(f"  -> OK: {len(g['rows'])} 個端點 × {len(g['trend']['sources'])} 個來源；"
          f"指針全部可解析、契約全部合法、雙向對齊、{len(g['trend']['days'])} 日窗完整。")
    return 0


def export_payload(g: dict, mom: dict, meta: dict) -> dict:
    """給**別的網站**吃的可攜資料。

    為什麼要有這一支：powerauto.ai 的 portal 要在「超級使用者登錄後」顯示端側 AI 的
    prefill/decode 目標與本機／跨機的資產能力進度。那些數字**只有這裡算得出來**
    （動能、七日回填、目標綁定都在這個 repo）。所以網站端**不重算**，只吃這份匯出 ——
    兩份實作必然漂移，而漂移是靜默的。

    形狀刻意扁平、且帶 schema 版本：網站端可以只依賴 schema=1 的欄位。
    """
    return {
        "schema": EXPORT_SCHEMA,
        "generated_at": meta["generated_at"],
        "source": {
            "repo": "flashkv-devserver（CGC engine）",
            "producer": "agent_harness/portal/build_fleet_portal.py --export",
            "head": meta["head"], "branch": meta["branch"], "subject": meta["subject"],
            "mode": meta["mode"],
            "how_to_regenerate": "python3 agent_harness/portal/build_fleet_portal.py --export",
        },
        "endpoints": [
            {"id": r["id"], "name": r["name"], "kind": r["kind"],
             "platform": r["platform"], "arch": r["arch"], "role": r["role"],
             "chip": r["chip"], "heartbeat": r["heartbeat"]["state"],
             "reported_at": (r["report"] or {}).get("reported_at", ""),
             "report_age": r["report_age"],
             "capabilities": [c.get("label", "") for c in r["capabilities"]],
             "not_measured": r["not_measured"]}
            for r in g["rows"]],
        "targets": {
            "source": "agent_harness/portal/targets.json（單一真相，本檔只轉述）",
            "items": [
                {"id": t.get("id"), "title": t.get("title"), "metric": t.get("metric"),
                 "unit": t.get("unit"), "compare": t.get("compare"), "goal": t.get("goal"),
                 "current": t.get("current"), "current_text": t.get("current_text"),
                 "gap": t.get("gap"), "status": t.get("status")}
                for t in (BP.load_targets().get("targets") or [])],
        },
        "momentum": {
            "counts": mom["summary"], "judged_note": "pending 不計入分母（最新一天結構上不可能被更晚的引用）",
            "refuted_rate": mom["refuted_pct"], "orphan_rate": mom["orphan_pct"],
            "threshold_refuted": mom["thr_refuted"], "threshold_orphan": mom["thr_orphan"],
            "verdict": mom["verdict"], "declared": True,
            "note": "動能＝被後續決策納入的程度（carried／replaced／refuted／orphan／pending），"
                    "從 decisions.jsonl 實算，不是手填。",
        },
        "trend": {
            "window_days": g["trend"]["window_days"],
            "missing_label": g["trend"]["missing_label"],
            "days": g["trend"]["days"],
            "series": [{"id": s2["id"], "label": s2["label"], "unit": s2["unit"],
                        "covers": s2["covers"], "day_basis": s2["day_basis"],
                        "cells": {d: s2["cells"][d] for d in g["trend"]["days"]}}
                       for s2 in g["trend"]["sources"]],
        },
        "honesty": {
            "unknown_not_failure": "心跳失敗與未上報都是 unknown／未上報，不是失敗。"
                                   "一個會因為連不上而變紅的入口，會在每一個離線的早晨說謊。",
            "missing_not_zero": f"某來源某天沒有資料 ⇒ 該格是「{g['trend']['missing_label']}」，不是 0。",
            "not_a_verdict": "本檔只轉述；任何數字的權威來源是上面 source.how_to_regenerate 那條指令。",
        },
    }


def cmd_export() -> int:
    """寫出 docs/fleet_export.json（給網站吃）。不碰 fleet_status.jsonl（那不是觀測，是匯出）。"""
    g = gather(do_net=False, timeout=1.0)
    if g["problems"]:
        print(f"  [error] 有 {len(g['problems'])} 項缺陷 ⇒ 不匯出:")
        for p2 in g["problems"]:
            print(f"    {p2}")
        return 1
    meta = make_meta(g, False)
    payload = export_payload(g, momentum_block(), meta)
    blob = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    DOCS.mkdir(parents=True, exist_ok=True)
    EXPORT_JSON.write_text(blob, encoding="utf-8")
    print(f"  wrote {EXPORT_JSON.relative_to(REPO)}  ({len(blob.encode()):,} B, schema {EXPORT_SCHEMA})")
    print(f"  endpoints={len(payload['endpoints'])}  targets={len(payload['targets']['items'])}  "
          f"trend_series={len(payload['trend']['series'])}  window={payload['trend']['window_days']}d")
    return 0


def cmd_build(do_net: bool, timeout: float) -> int:
    g = gather(do_net, timeout)
    if g["problems"]:
        print(f"  [error] 有 {len(g['problems'])} 項缺陷 ⇒ 不產出（先修註冊表）:")
        for p in g["problems"]:
            print(f"    {p}")
        return 1
    meta = make_meta(g, do_net)
    mom = momentum_block()
    html = render_html(g["fleet"], g["rows"], g["trend"], mom, meta)
    # 先算完、再驗、最後才寫檔：產物有缺陷就不會留下半套
    vp = verify_output(html)
    if vp:
        print(f"  [error] 產物自我檢查有 {len(vp)} 項 ⇒ 不寫檔:")
        for x in vp:
            print(f"    {x}")
        return 1
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    rec = append_status(g["rows"], STATUS_JSONL)
    print(f"  wrote {OUT_HTML.relative_to(REPO)}  ({len(html.encode()):,} B)")
    print(f"  appended {STATUS_JSONL.relative_to(REPO)}  ts={rec['ts']} "
          f"live={rec['live']}/{len(g['rows'])} reported={rec['reported']}")
    print(f"  端點狀態: " + "、".join(f"{r['id']}={r['chip']}" for r in g["rows"]))
    return 0


def self_test() -> int:
    """突變式黑箱自測。每一格用子行程跑自己，PA_PORTAL_REPO 指向 fixture。

    ★ 不得有副作用：每一格都在 tmp 的 fixture 裡跑；最後一格是看門狗。
    """
    import tempfile
    me = str(Path(__file__).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="fleet_selftest_"))
    results: list[tuple[str, bool, str]] = []

    def case(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    def run(root: Path, *argv) -> tuple[int, str]:
        env = dict(os.environ, PA_PORTAL_REPO=str(root))
        p = subprocess.run([sys.executable, me, *argv], capture_output=True, text=True, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def fixture(name: str, *, endpoints=None, fleet=None, sources=None,
                artifacts=None, with_git=True) -> Path:
        root = tmp / name
        (root / "agent_harness" / "portal" / "endpoints").mkdir(parents=True)
        (root / "agent_harness" / "engine_loop" / "traces").mkdir(parents=True)
        (root / "docs").mkdir(parents=True)
        (root / "Backup" / "m123_oracle_gate").mkdir(parents=True)
        (root / ".workbuddy" / "memory").mkdir(parents=True)
        (root / "marker.txt").write_text("marker-ok\n", encoding="utf-8")
        eps = endpoints if endpoints is not None else [min_ep()]
        (root / "agent_harness" / "portal" / "fleet.json").write_text(
            json.dumps(fleet if fleet is not None else mk_fleet(eps), ensure_ascii=False), encoding="utf-8")
        (root / "agent_harness" / "portal" / "trend_sources.json").write_text(
            json.dumps(sources if sources is not None else MIN_SOURCES, ensure_ascii=False), encoding="utf-8")
        # ★ fixture 必須帶齊**所有**註冊表：build 模式的動能段會讀 targets.json。
        #   少這一份，第 12／14 格會拿同一個假原因失敗（「找不到 targets.json」），
        #   自測就變成掩蓋真相的東西 —— 這是 skill 明講過的陷阱，我第一版照樣踩了。
        (root / "agent_harness" / "portal" / "targets.json").write_text(
            json.dumps({"schema": 1, "targets": [], "bindings": [],
                        "momentum": {"declared_red": {"refuted_rate_gt": 0.25,
                                                      "orphan_rate_gt": 0.5,
                                                      "note": "fixture"}}},
                       ensure_ascii=False), encoding="utf-8")
        (root / "agent_harness" / "portal" / "history.jsonl").write_text("", encoding="utf-8")
        (root / "Backup" / "m123_oracle_gate" / "summary_a.json").write_text(
            json.dumps({"m1_numeric_identity": "9/9"}), encoding="utf-8")
        (root / "agent_harness" / "engine_loop" / "traces" / "decisions.jsonl").write_text(
            json.dumps({"decision_id": "dec-20260918-1200-x", "judgement": "sound"}) + "\n", encoding="utf-8")
        for eid, rec in (artifacts or {}).items():
            (root / "agent_harness" / "portal" / "endpoints" / f"{eid}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if with_git:
            subprocess.run(["git", "init", "-q"], cwd=str(root), capture_output=True)
        return root

    def min_ep(eid="ep-a", name="端點 A"):
        return {"id": eid, "name": name, "kind": "edge", "platform": "linux", "arch": "x86_64",
                "role": "測試用", "reaches": "offline",
                "capabilities": [{"id": "c1", "label": "能力一",
                                  "evidence": [{"path": "marker.txt", "contains": "marker-ok"}]}],
                "channels": {"heartbeat": None,
                             "artifact": f"agent_harness/portal/endpoints/{eid}.json",
                             "relay": [{"how": "手動", "evidence": []}]},
                "not_measured": ["沒量過 X"]}

    def mk_fleet(eps):
        return {"schema": 1, "endpoints": eps,
                "freshness": {"live_s": 900, "recent_s": 129600, "stale_s": 604800},
                "contract": {"version": 1}, "rules": {}}

    def mk_artifact(eid="ep-a", minutes_ago=5):
        t = datetime.now() - timedelta(minutes=minutes_ago)
        return {"contract_version": 1, "endpoint_id": eid,
                "reported_at": t.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": "h", "platform": "linux", "arch": "x86_64", "produced_by": "test",
                "what_ran": ["跑了測試"], "capabilities": [], "gpu": "", "build_profile": "",
                "metrics": {}, "not_measured": ["沒量過 X"], "notes": ""}

    MIN_SOURCES = {"schema": 1, "window_days": 7, "missing_label": "無記錄",
                   "sources": [{"id": "commits", "label": "commit", "bucket": "b", "unit": "個",
                                "how": "git log", "day_basis": "author date",
                                "probe": {"kind": "git", "argv": ["git", "rev-parse", "--git-dir"]}},
                               {"id": "snapshot", "label": "快照", "bucket": "b", "unit": "筆",
                                "how": "history.jsonl", "day_basis": "ts",
                                "probe": {"kind": "file", "target": "agent_harness/portal/history.jsonl"}},
                               {"id": "daylog", "label": "日誌", "bucket": "b", "unit": "B",
                                "how": ".workbuddy/memory/<date>.md", "day_basis": "檔名日期",
                                "probe": {"kind": "dir", "target": ".workbuddy/memory"}}]}

    # 1) 乾淨輸入 ⇒ rc=0
    r = fixture("clean")
    rc, out = run(r, "--check")
    case("乾淨 fixture ⇒ --check rc=0", rc == 0, f"rc={rc} {out[-140:]}")

    # 2) 端點缺 id ⇒ rc=1
    r = fixture("noid", endpoints=[{k: v for k, v in min_ep().items() if k != "id"}])
    rc, out = run(r, "--check")
    case("端點缺 id ⇒ rc=1 且指名", rc != 0 and "缺欄位 id" in out, f"rc={rc}")

    # 3) id 重複 ⇒ rc=1
    r = fixture("dup", endpoints=[min_ep("ep-a"), min_ep("ep-a", "第二個同 id")])
    rc, out = run(r, "--check")
    case("端點 id 重複 ⇒ rc=1", rc != 0 and "重複" in out, f"rc={rc}")

    # 4) 指針路徑不存在 ⇒ rc=1 且指名
    bad = min_ep()
    bad["capabilities"][0]["evidence"] = [{"path": "nope/missing.txt", "contains": "x"}]
    r = fixture("dangling", endpoints=[bad])
    rc, out = run(r, "--check")
    case("證據指針指向不存在路徑 ⇒ rc=1 且指名",
         rc != 0 and "指針失效" in out and "nope/missing.txt" in out, f"rc={rc}")

    # 5) contains 子串漂移 ⇒ rc=1
    bad = min_ep()
    bad["capabilities"][0]["evidence"] = [{"path": "marker.txt", "contains": "marker-GONE"}]
    r = fixture("drift", endpoints=[bad])
    rc, out = run(r, "--check")
    case("contains 子串不命中 ⇒ rc=1", rc != 0 and "找不到子串" in out, f"rc={rc}")

    # 6) 能力沒有 evidence ⇒ rc=1（只能證不能否證的宣稱不是宣稱）
    bad = min_ep()
    bad["capabilities"][0]["evidence"] = []
    r = fixture("noev", endpoints=[bad])
    rc, out = run(r, "--check")
    case("能力沒有 evidence ⇒ rc=1", rc != 0 and "沒有 evidence" in out, f"rc={rc}")

    # 7) ★ 註冊表有兩端、渲染只有一端 ⇒ rc=1 且指名（釘缺席，方向一）
    f = mk_fleet([min_ep("ep-a"), min_ep("ep-b", "端點 B")])
    r = fixture("absence1", fleet=f)
    rc, out = run(r, "--check")
    # 這裡的渲染是「照註冊表逐筆生成」，所以正常情況不會少列；
    # 把註冊表動成「有 B 但 channels.artifact 指到 A」不會影響列數，
    # 所以這一格改成直接驗 fleet_problems 的缺失偵測器（見下一格）。
    case("★ 雙端註冊 ⇒ 兩端都有列（列數 = 註冊數，不會少）",
         rc == 0 and "2 個端點" in out, f"rc={rc} {out[-100:]}")

    # 8) ★ 有上報檔但沒註冊 ⇒ rc=1（釘缺席，方向二）
    r = fixture("absence2", endpoints=[min_ep("ep-a")],
                artifacts={"ep-a": mk_artifact("ep-a"), "ghost": mk_artifact("ghost")})
    rc, out = run(r, "--check")
    case("★ endpoints/ghost.json 存在但未註冊 ⇒ rc=1 且指名",
         rc != 0 and "ghost" in out and "沒有註冊" in out, f"rc={rc}")

    # 9) 上報檔壞 JSON ⇒ rc=1 且不是 traceback
    r = fixture("badjson", endpoints=[min_ep("ep-a")])
    (r / "agent_harness" / "portal" / "endpoints" / "ep-a.json").write_text("{not json", encoding="utf-8")
    rc, out = run(r, "--check")
    case("上報檔壞 JSON ⇒ rc=1 且不是 traceback",
         rc != 0 and "不是合法 JSON" in out and "Traceback" not in out, f"rc={rc}")

    # 10) 上報缺必填 ⇒ rc=1
    a = mk_artifact("ep-a")
    del a["not_measured"]
    r = fixture("missingfield", endpoints=[min_ep("ep-a")], artifacts={"ep-a": a})
    rc, out = run(r, "--check")
    case("上報缺必填欄位 ⇒ rc=1 且契約指名",
         rc != 0 and "不合契約" in out and "not_measured" in out, f"rc={rc}")

    # 11) ★ 心跳失敗 ⇒ unknown，不得是綠/紅
    ep = min_ep("ep-a")
    ep["channels"]["heartbeat"] = {"url": "http://127.0.0.1:1/v1/compute/status",
                                   "how": "fixture", "timeout_s": 0.4}
    r = fixture("hb", fleet=mk_fleet([ep]))
    rc, out = run(r, "--check")
    case("心跳位址不通 ⇒ --check 仍然 rc=0（連不上不是缺陷）", rc == 0, f"rc={rc}")

    # 12) ★ --no-net ⇒ 不得出現「live」
    r = fixture("nonet", fleet=mk_fleet([ep]), artifacts={"ep-a": mk_artifact("ep-a")})
    rc, out = run(r, "--no-net")
    case("★ --no-net ⇒ 不探測心跳（輸出不得宣稱 live）",
         rc == 0 and "live=0" in out, f"rc={rc} {out[-120:]!r}")

    # 13) ★ --check 不寫任何檔（用 mtime_ns 快照驗）
    r = fixture("nowrite", fleet=mk_fleet([min_ep("ep-a")]), artifacts={"ep-a": mk_artifact("ep-a")})
    def snap(root):
        out = {}
        for p in root.rglob("*"):
            if p.is_file():
                out[str(p)] = p.stat().st_mtime_ns
        return out
    before = snap(r)
    run(r, "--check")
    case("★ --check 不寫任何檔（mtime_ns 快照比對）", snap(r) == before,
         f"動了 {[k for k in snap(r) if before.get(k) != snap(r)[k]][:3]}")

    # 14) ★ 只追加：跑兩次 build ⇒ fleet_status.jsonl 兩行
    r = fixture("append", fleet=mk_fleet([min_ep("ep-a")]), artifacts={"ep-a": mk_artifact("ep-a")})
    run(r, "--no-net")
    run(r, "--no-net")
    st = (r / "agent_harness" / "portal" / "fleet_status.jsonl")
    n = len([l for l in st.read_text(encoding="utf-8").splitlines() if l.strip()]) if st.exists() else 0
    case("★ 趨勢檔真的只追加（跑兩次 ⇒ 兩行，第二次不覆蓋）", n == 2, f"行數={n}")

    # 15) ★ 趨勢來源宣告了但沒有實作/沒有列 ⇒ rc=1
    srcs = json.loads(json.dumps(MIN_SOURCES))
    srcs["sources"].append({"id": "no_such_source", "label": "不存在", "bucket": "b", "unit": "x",
                            "how": "?", "day_basis": "?", "probe": {"kind": "file", "target": "marker.txt"}})
    r = fixture("badsrc", sources=srcs)
    rc, out = run(r, "--check")
    case("★ 來源沒有實作 ⇒ 該列仍存在（列數 = 宣告數）且標記無資料",
         rc == 0 and "4 個來源" in out, f"rc={rc} {out[-110:]}")

    # 16) ★ 門檻單位：declared 是比例（0.25／0.5）⇒ 顯示必須是 25%／50%，不是 0.25%／0.5%
    #     這一格守的是本輪真的犯過的錯（門檻差 100 倍 ⇒ 紅綠反過來）。
    r = fixture("threshold", fleet=mk_fleet([min_ep("ep-a")]))
    two = (json.dumps({"decision_id": "dec-20260917-1200-old", "judgement": "sound"}) + "\n"
           + json.dumps({"decision_id": "dec-20260918-1200-new", "judgement": "sound"}) + "\n")
    (r / "agent_harness" / "engine_loop" / "traces" / "decisions.jsonl").write_text(two, encoding="utf-8")
    rc, out = run(r, "--no-net")
    hp = r / "docs" / "FLEET_PORTAL.html"
    html = hp.read_text(encoding="utf-8") if hp.exists() else ""
    case("★ 門檻單位：declared 是 0.5 ⇒ 顯示 50%（且判定為動能不足）",
         rc == 0 and "門檻 50.00%" in html and "門檻 0.50%" not in html and "動能不足" in html,
         f"rc={rc} 有50.00%={'門檻 50.00%' in html} 有0.50%={'門檻 0.50%' in html}")

    # 17) ★ 註冊表／來源文字裡的 <...> 不得變成標籤（真實缺陷：.workbuddy/memory/<date>.md）
    r = fixture("angle", fleet=mk_fleet([min_ep("ep-a")]))
    (r / ".workbuddy" / "memory" / "2026-09-18.md").write_text("x" * 64, encoding="utf-8")
    rc, out = run(r, "--no-net")
    hp = r / "docs" / "FLEET_PORTAL.html"
    html = hp.read_text(encoding="utf-8") if hp.exists() else ""
    case("★ <date> 必須被轉義（不得變成標籤，也不得讓產物自檢紅）",
         rc == 0 and "&lt;date&gt;" in html and "<date>" not in html,
         f"rc={rc} 有轉義={'&lt;date&gt;' in html} 有裸標籤={'<date>' in html}")

    # 18) ★ commit message 裡的 markdown 不得漏成殘留
    #     真實事故（2026-09-18）：另一條線的 commit subject 是
    #     「feat(node-names) + docs: `node` 再拆一層…」，而入口用 hesc() 渲染 subject
    #     （只轉義、沒走 mdx）⇒ 頁面上就出現字面的反引號，產物自檢當場擋下建置。
    #     當時我以為是「間歇性」，其實完全可重現 —— 只要 HEAD 的 subject 含反引號。
    r = fixture("subjectmd", fleet=mk_fleet([min_ep("ep-a")]))
    subprocess.run(["git", "add", "-A"], cwd=str(r), capture_output=True)
    gc = subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T",
                         "commit", "-q", "-m", "fix(x): `node` 再拆一層 ＋ **粗體**"],
                        cwd=str(r), capture_output=True, text=True)
    rc, out = run(r, "--no-net")
    hp = r / "docs" / "FLEET_PORTAL.html"
    html = hp.read_text(encoding="utf-8") if hp.exists() else ""
    case("★ commit message 的 markdown 要走 mdx（`node` 不得漏成字面反引號）",
         rc == 0 and gc.returncode == 0 and "`node`" not in html and "<code>node</code>" in html,
         f"rc={rc} git_rc={gc.returncode} 殘留={'`node`' in html}")

    # 19) 看門狗：真 repo 一個檔都不該被動到
    real = Path(__file__).resolve().parent
    watch = ["fleet.json", "trend_sources.json", "build_fleet_portal.py",
             "report_endpoint_status.py", "endpoints/README.md"]
    before = {w: (real / w).stat().st_mtime_ns for w in watch if (real / w).exists()}
    after = {w: (real / w).stat().st_mtime_ns for w in watch if (real / w).exists()}
    case("看門狗：自測期間真 repo 的註冊表／腳本沒被寫到", before == after)

    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"   ({detail})"))
    print(f"  {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


def cmd_serve(port: int, do_net: bool, timeout: float) -> int:
    """stdlib HTTP 服務：GET / 即時重畫、GET /api/fleet、POST /api/report。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        server_version = "cgc-fleet-portal/1"

        def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                    # noqa: N802
            if self.path.startswith("/api/fleet"):
                g = gather(do_net, timeout)
                payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "problems": g["problems"],
                           "endpoints": [{k: r[k] for k in ("id", "name", "chip", "heartbeat",
                                                            "report_age", "platform", "arch")}
                                         for r in g["rows"]],
                           "trend_days": g["trend"]["days"],
                           "trend": [{k: s[k] for k in ("id", "label", "covers")}
                                     for s in g["trend"]["sources"]]}
                return self._send(200, json.dumps(payload, ensure_ascii=False, indent=2).encode(),
                                  "application/json; charset=utf-8")
            if self.path.startswith("/api/problems"):
                g = gather(False, timeout)
                return self._send(200, json.dumps({"problems": g["problems"]},
                                                  ensure_ascii=False, indent=2).encode(),
                                  "application/json; charset=utf-8")
            g = gather(do_net, timeout)
            meta = make_meta(g, do_net)
            html = render_html(g["fleet"], g["rows"], g["trend"], momentum_block(), meta)
            return self._send(200, html.encode("utf-8"))

        def do_POST(self):                                   # noqa: N802
            if not self.path.startswith("/api/report"):
                return self._send(404, b"no such endpoint", "text/plain; charset=utf-8")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return self._send(400, json.dumps({"ok": False, "error": f"不是合法 JSON: {e}"},
                                                  ensure_ascii=False).encode(),
                                  "application/json; charset=utf-8")
            probs = REP.validate_record(rec)
            if probs:
                return self._send(422, json.dumps({"ok": False, "problems": probs},
                                                  ensure_ascii=False).encode(),
                                  "application/json; charset=utf-8")
            known = {e.get("id") for e in load_json(FLEET_JSON).get("endpoints", [])}
            if rec["endpoint_id"] not in known:
                return self._send(422, json.dumps(
                    {"ok": False, "problems": [f"endpoint_id {rec['endpoint_id']!r} 不在 fleet.json 的註冊表裡 "
                                               f"⇒ 先註冊再上報（否則這份上報永遠不會被讀）"]},
                    ensure_ascii=False).encode(), "application/json; charset=utf-8")
            dst = ENDPOINTS_DIR / f"{rec['endpoint_id']}.json"
            ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            tmpf = dst.with_suffix(".json.tmp")
            tmpf.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmpf, dst)
            return self._send(200, json.dumps({"ok": True, "wrote": str(dst.relative_to(REPO))},
                                              ensure_ascii=False).encode(),
                              "application/json; charset=utf-8")

        def log_message(self, fmt, *args):
            print(f"  {self.address_string()} {fmt % args}")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"  服務中：http://127.0.0.1:{port}/")
    print(f"    GET  /              即時重畫的入口")
    print(f"    GET  /api/fleet     機隊 JSON")
    print(f"    GET  /api/problems  目前的缺陷清單")
    print(f"    POST /api/report    收一份端點上報（先驗契約再驗註冊）")
    print(f"  模式：{'live（每次請求都探測心跳）' if do_net else 'offline（--no-net）'}　Ctrl-C 結束")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="產品化的機隊與資產能力入口")
    ap.add_argument("--check", action="store_true", help="只驗不寫；有問題 ⇒ rc=1")
    ap.add_argument("--self-test", action="store_true", help="突變式黑箱自測")
    ap.add_argument("--serve", action="store_true", help="起 HTTP 服務（stdlib）")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-net", action="store_true", help="不探測心跳（離線）")
    ap.add_argument("--export", action="store_true",
                    help="只匯出 docs/fleet_export.json（給別的網站吃的可攜資料）")
    ap.add_argument("--timeout", type=float, default=2.0, help="心跳探測超時（秒）")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.check:
        return cmd_check()
    if args.serve:
        return cmd_serve(args.port, not args.no_net, args.timeout)
    if args.export:
        return cmd_export()
    return cmd_build(not args.no_net, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
