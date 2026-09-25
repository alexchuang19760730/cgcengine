#!/usr/bin/env python3
"""Lane watchdog — 獨立於任何 runner 的「看門人」。

為什麼存在
----------
harness verify 的 G1 只在「走 harness」時生效；各線常用自己的 runner（mw_ab、paired_ab…）
直接調 llama_bench_matrix，繞過 G1，於是在 swap 已高、free 極低的髒環境硬跑：
要嘛 OOM、要嘛把 swap 撐得更大、還把壓力下的數字當成生產成績。

這個看門人不相信任何 runner 的自我標註，直接看三件事：
  1. 系統：thermal / swap used·total / pages free / wired
  2. 進程：正在跑的 llama-bench / llama-server，解析實際命令行（batch、load-mode、pool…）
  3. 產物：最近完成的 matrix 產物，啟動時 swap、swap growth、attribution —— 結論該不該引用

分級（保守，避免誤殺）
  ok   ：乾淨
  warn ：臨界（swap 1024~3072 或 free 150~400）—— 只報告
  kill ：系統已危險、繼續跑只會 OOM / 製造更多 swap —-- 預設報告，--kill 才終止

kill 必須同時滿足（雙重條件，防誤殺正常測量）：
  (a) 進程是 llama-bench/llama-server，
  (b) 當前 swap > SWAP_KILL 或 free < FREE_KILL（系統已在懸崖邊）。
另外「啟動就髒」（產物 launch_swap > LAUNCH_SWAP_KILL）單列為 strong-warning，
因為它在 G1 口徑下根本該拒跑；配合 --kill 也會終止。

用法
  python3 scripts/check/lane_watchdog.py                # 報告（不終止）
  python3 scripts/check/lane_watchdog.py --kill         # 對 kill 級進程終止（SIGTERM→SIGKILL）
  python3 scripts/check/lane_watchdog.py --json
  python3 scripts/check/lane_watchdog.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS_LOG = ROOT / "Backup" / "lane_watchdog" / "watchdog_runs.jsonl"

# ── 閾值（MiB），依本機實測校準 ──
SWAP_WARN, SWAP_KILL = 1024.0, 3072.0
FREE_WARN, FREE_KILL = 400.0, 150.0
LAUNCH_SWAP_KILL = 2048.0      # 產物記錄的啟動 swap > 此值 = 髒環境硬跑
SWAP_GROWTH_KILL = 1500.0      # 單次測量製造的 swap growth > 此值
RECENT_ARTIFACT_MIN = 90       # 檢查最近 N 分鐘完成的產物
SPREAD_UNRELIABLE = 12.0       # rep採樣散度>此%=量具被壓垮、數字作廢（生產 cell 4.3%、壞 cell 38%）
QUAR_DIR = ROOT / "Backup" / "quarantine"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- system
def system_state() -> dict:
    """thermal + swap used/total + pages free/speculative + wired（直接解析，不信 runner）。"""
    st: dict = {}
    try:
        t = _load("tp_wd", "thermal_pressure.py").stamp()
        st["thermal"] = t.get("label")
        st["thermal_level"] = t.get("lv")
    except Exception as e:
        st["thermal"] = f"err:{e}"

    # sysctl vm.swapusage: "total = XM  used = YM  free = ZM"（按標籤解析，勿按位置）
    try:
        import re
        out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                             text=True, timeout=10).stdout
        def _m(name):
            mm = re.search(name + r"\s*=\s*([\d.]+)\s*M", out)
            return float(mm.group(1)) if mm else None
        st["swap_total_mb"] = _m("total")
        st["swap_used_mb"] = _m("used")
        st["swap_free_mb"] = _m("free")
    except Exception:
        st["swap_used_mb"], st["swap_total_mb"], st["swap_free_mb"] = None, None, None

    # vm_stat: page size + free / speculative / wired
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        psize = 16384
        vals: dict = {}
        for line in out.splitlines():
            if "page size of" in line:
                psize = int(line.split("page size of")[1].split()[0])
            low = line.lower()
            for key, names in (("free", ("pages free:",)),
                              ("spec", ("pages speculative:",)),
                              ("wired", ("pages wired down:",))):
                if low.startswith(names):
                    vals[key] = int(line.split(":")[1].strip().rstrip("."))
        mb = lambda n: n * psize / (1024 * 1024)
        st["free_mb"] = mb(vals.get("free", 0)) + mb(vals.get("spec", 0))
        st["wired_mb"] = mb(vals.get("wired", 0))
    except Exception:
        st["free_mb"], st["wired_mb"] = None, None
    return st


# --------------------------------------------------------------------------- processes
def _parse_etime(s: str):
    """ps etime [[dd-]hh:]mm:ss -> seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [int(x) for x in parts]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        h = m = sec = 0
    return days * 86400 + h * 3600 + m * 60 + sec


def running_procs() -> list[dict]:
    """所有 llama-bench / llama-server，解析實際命令行關鍵參數。"""
    out = subprocess.run(
        ["ps", "-Ao", "pid=,etime=,rss=,command="],
        capture_output=True, text=True, timeout=15).stdout
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, etime, rss, cmd = parts
        if not (("llama-bench" in cmd) or ("llama-server" in cmd)):
            continue
        if "grep" in cmd:
            continue
        toks = cmd.split()

        def opt(name, default=None):
            if name in toks:
                i = toks.index(name)
                return toks[i + 1] if i + 1 < len(toks) else True
            return default

        is_bench = "llama-bench" in toks[0]
        procs.append({
            "pid": int(pid), "etime_s": _parse_etime(etime),
            "rss_mb": int(rss) / 1024, "is_bench": is_bench,
            "batch": int(opt("-b", 0) or 0),
            "ubatch": int(opt("-ub", 0) or 0),
            "load_mode": opt("--load-mode"),
            "ngl": opt("-ngl"),
            "expert_cache": opt("-expert-cache"),
            "prompt": opt("-p"), "gen": opt("-n"),
            "raw": cmd,
        })
    return procs


def judge_proc(p: dict, sysst: dict) -> dict:
    """對一個進程下 ok / warn / kill，附理由。kill 用雙重條件防誤殺。"""
    reasons, level = [], "ok"
    swap, free = sysst.get("swap_used_mb"), sysst.get("free_mb")

    dangerous_now = False
    if swap is not None and swap > SWAP_KILL:
        reasons.append(f"swap {swap:.0f} > {SWAP_KILL:.0f} MiB")
        dangerous_now = True
    if free is not None and free < FREE_KILL:
        reasons.append(f"free {free:.0f} < {FREE_KILL:.0f} MiB（瀕臨 jetsam）")
        dangerous_now = True
    if swap is not None and swap > SWAP_WARN and not dangerous_now:
        reasons.append(f"swap {swap:.0f} > warn {SWAP_WARN:.0f}")
        level = "warn"
    if free is not None and free < FREE_WARN and not dangerous_now:
        reasons.append(f"free {free:.0f} < warn {FREE_WARN:.0f}")
        level = "warn"

    # 參數明顯拼錯（會產出無效結果）
    if p["is_bench"] and p["batch"] and p["batch"] >= 4096 and dangerous_now:
        reasons.append(f"大 batch {p['batch']} 在高危記憶體下跑")
    if p["is_bench"] and not p["load_mode"]:
        reasons.append("llama-bench 未帶 --load-mode（可能不是 prod-new 口徑）")
        if level == "ok":
            level = "warn"

    if dangerous_now:
        level = "kill"
    return {"pid": p["pid"], "level": level, "reasons": reasons,
            "dangerous_now": dangerous_now}


# --------------------------------------------------------------------------- artifacts
def judge_artifact(arm: dict) -> dict:
    """純判決：數字作廢與否由「thermal + 量具 rep 散度」裁決，**不由 swap**。
    swap 高只標 stressed（環境承壓、需結構修復），量具穩時數字仍可引用。"""
    mem = arm.get("memory", {}) or {}
    launch, endd = mem.get("launch", {}) or {}, mem.get("end", {}) or {}
    ls, es = launch.get("swap_used_mb"), endd.get("swap_used_mb")
    growth = (es - ls) if (ls is not None and es is not None) else None

    scores, rep_spread, row_sd = {}, {}, {}
    for r in arm.get("rows", []):
        kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
        avg, sams = r.get("avg_ts"), r.get("samples_ts") or []
        scores[kind], row_sd[kind] = avg, r.get("stddev_ts")
        if avg and sams and len(sams) >= 2:
            rep_spread[kind] = (max(sams) - min(sams)) / avg * 100

    th = arm.get("thermal", {}) or {}
    th_worst = (th.get("worst", {}) or {}).get("label")
    thermal_bad = th_worst not in (None, "NOMINAL")
    spread_tg = rep_spread.get("tg")
    spread_high = spread_tg is not None and spread_tg > SPREAD_UNRELIABLE
    unreliable = thermal_bad or spread_high
    stressed = bool(
        (ls is not None and ls > LAUNCH_SWAP_KILL) or
        (growth is not None and growth > SWAP_GROWTH_KILL))

    kill_notes, stress_notes = [], []
    if thermal_bad:
        kill_notes.append(f"thermal worst={th_worst}（GPU 時脈非正常）")
    if spread_high:
        kill_notes.append(f"量具散度 {spread_tg:.0f}%>{SPREAD_UNRELIABLE:.0f}%（不可重複）")
    if ls is not None and ls > LAUNCH_SWAP_KILL:
        stress_notes.append(f"啟動 swap={ls:.0f}")
    if growth is not None and growth > SWAP_GROWTH_KILL:
        stress_notes.append(f"swap growth={growth:+.0f}（環境承壓、需結構修復；數字本身仍有效）")
    return {"launch_swap": ls, "growth": growth, "scores": scores,
            "row_sd": row_sd, "rep_spread": rep_spread, "thermal_worst": th_worst,
            "thermal_bad": thermal_bad, "spread_high": spread_high,
            "unreliable": unreliable, "stressed": stressed,
            "kill_notes": kill_notes, "stress_notes": stress_notes}


def recent_artifacts(minutes: int = RECENT_ARTIFACT_MIN) -> list[dict]:
    """最近完成、含 memory/attribution 的 matrix 產物，判乾淨/污染、結論可否引用。"""
    cutoff = time.time() - minutes * 60
    found = []
    patterns = [str(ROOT / "Backup" / "**" / "*.json"),
                "/tmp/harness_bench/**/*.json", "/tmp/harness_verify/**/matrix.json"]
    paths = set()
    for pat in patterns:
        paths.update(glob.glob(pat, recursive=True))
    for path in sorted(paths):
        try:
            if os.path.getmtime(path) < cutoff:
                continue
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        arms = data if isinstance(data, list) else [data]
        for arm in arms:
            if not isinstance(arm, dict) or "memory" not in arm:
                continue
            v = judge_artifact(arm)
            worst = (arm.get("memory", {}) or {}).get("worst", {}) or {}
            cch = arm.get("cache", {}) or {}
            envd = arm.get("env", {}) or {}
            io_jobs, io_bytes = cch.get("io_jobs"), cch.get("io_bytes")
            bytes_per_job = (io_bytes / io_jobs) if (io_jobs and io_bytes) else None
            cm = {
                "hit_pct": cch.get("hit_rate_pct"),
                "misses": cch.get("misses"),
                "miss_compulsory": cch.get("miss_compulsory"),
                "miss_capacity": cch.get("miss_capacity"),
                "us_per_job": cch.get("io_us_per_job"),
                "per_miss_us": cch.get("per_miss_us"),
                "bytes_per_job": bytes_per_job,
                "effective_mib_s": cch.get("io_effective_mib_s"),
                "resident_mib": cch.get("resident_mib"),
                "workers_claimed": envd.get("LLAMA_EXPERT_CACHE_WORKERS"),
            }
            found.append({
                "path": path, "tag": arm.get("tag"), "scores": v["scores"],
                "row_sd": v["row_sd"], "rep_spread_pct": v["rep_spread"],
                "cache": cm, "thermal_worst": v["thermal_worst"],
                "launch_swap": v["launch_swap"], "swap_growth": v["growth"],
                "worst_swap": worst.get("max_swap_mb"),
                "min_free": worst.get("min_free_mb"),
                "stressed": v["stressed"], "polluted": v["unreliable"],
                "kill_notes": v["kill_notes"], "stress_notes": v["stress_notes"],
                "notes": v["kill_notes"] + v["stress_notes"],
            })
    # 去重（同 path+tag）
    seen, uniq = set(), []
    for a in found:
        k = (a["path"], a["tag"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(a)
    return uniq


# --------------------------------------------------------------------------- consistency
def consistency_check(artifacts: list[dict]) -> list[dict]:
    """宣稱（env/profile 解析值）vs 實際（行為指標）交叉檢查。

    坑（本輪實例）：兩臂 json 都記 WORKERS=8，但一臂 us/job 偏高 ⇒ 實際子進程可能拿到 2。
    json 的 env 是「profile 解析值」、不是實際下發值。驗證開關生效要看行為指標，
    否則「沒生效」和「沒效果」看上去一模一樣。
    """
    groups: dict = {}
    for a in artifacts:
        c = a.get("cache", {})
        uj, wk = c.get("us_per_job"), c.get("workers_claimed")
        if uj is not None and a["scores"].get("tg"):
            groups.setdefault(wk, []).append((a, uj))
    flags = []
    for wk, grp in groups.items():
        if len(grp) < 2:
            continue
        ujs = [u for _, u in grp]
        lo, hi = min(ujs), max(ujs)
        dev = (hi - lo) / lo * 100 if lo else 0
        if dev > 12:  # 與量具失效閾值同級；swap 抖動下保守
            flags.append({
                "level": "warn",
                "issue": f"workers 同宣稱={wk}、但 us/job 離散 {lo:.0f}~{hi:.0f}（{dev:.0f}%）",
                "suspect_paths": [a.get("path") for a, u in grp if u == hi],
                "hint": "高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；"
                        "這是『請驗證』、非定論。需對實際下發值/進程命令行核對。"})
    return flags


# --------------------------------------------------------------------------- quarantine
def quarantine_artifacts(artifacts: list[dict], when: str, do_it: bool) -> list[dict]:
    """對 UNRELIABLE 產物主動移除錯誤數字（STRESSED 但數字有效的**不動**）：
      - 多 arm 混合檔：在檔內給失效 arm 注入 _quarantine、保留其他 arm；
      - 整檔失效：移到 Backup/quarantine、原路徑留指針；
      - 維護 QUARANTINE_REGISTRY.json。
    """
    if not do_it:
        return []
    QUAR_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = QUAR_DIR / "QUARANTINE_REGISTRY.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"quarantined": []}
    by_path: dict = {}
    for a in artifacts:
        if a.get("polluted"):
            by_path.setdefault(a["path"], []).append(a)
    actions = []
    for path, alist in by_path.items():
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        arms = data if isinstance(data, list) else [data]
        tags = {a["tag"] for a in alist}
        all_bad = all(arm.get("tag") in tags for arm in arms)
        for arm in arms:
            if arm.get("tag") in tags:
                arm["_quarantine"] = {
                    "when": when,
                    "reasons": next(a["kill_notes"] for a in alist
                                    if a["tag"] == arm.get("tag")),
                    "note": "量具不可信、數字作廢、不可引用"}
        moved = None
        if all_bad:
            dst = QUAR_DIR / f"{p.stem}.{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}"
            shutil.move(str(p), str(dst))
            moved = str(dst)
            pointer = {"_quarantine_pointer": True, "when": when,
                       "original_path": path, "quarantined_path": moved,
                       "reasons": alist[0]["kill_notes"],
                       "note": "量具不可信、數字作廢；證據見 quarantined_path"}
            p.write_text(json.dumps(pointer, indent=1, ensure_ascii=False))
        else:
            p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        for a in alist:
            reg["quarantined"].append({
                "when": when, "path": path, "tag": a["tag"],
                "scores": a["scores"], "reasons": a["kill_notes"], "moved_to": moved})
        actions.append({"path": path, "arms": sorted(tags), "moved_to": moved})
    reg_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    return actions


# --------------------------------------------------------------------------- actions
def terminate_pid(pid: int) -> dict:
    """SIGTERM 等 8 s，仍活再 SIGKILL。回結果。"""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"pid": pid, "result": "already gone"}
    for _ in range(16):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return {"pid": pid, "result": "terminated (SIGTERM)"}
    try:
        os.kill(pid, signal.SIGKILL)
        return {"pid": pid, "result": "SIGKILL"}
    except ProcessLookupError:
        return {"pid": pid, "result": "terminated (SIGTERM, late)"}


def inspect(do_kill: bool, do_quarantine: bool = False) -> dict:
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    sysst = system_state()
    procs = running_procs()
    judged = [judge_proc(p, sysst) for p in procs]
    artifacts = recent_artifacts()
    consistency = consistency_check(artifacts)
    quar = quarantine_artifacts(artifacts, when, do_quarantine)

    actions = []
    if do_kill:
        for j in judged:
            if j["level"] == "kill":
                r = terminate_pid(j["pid"])
                r["reasons"] = j["reasons"]
                actions.append(r)

    return {"when": when, "system": sysst,
            "procs": [dict(p, verdict=next(j for j in judged if j["pid"] == p["pid"]))
                      for p in procs],
            "artifacts": artifacts, "consistency": consistency,
            "quarantine": quar, "actions": actions}


# 絕對化措辭（這兩天反覆出現的「unavoidable / 一定 / 100%」類）
ABSOLUTE_WORDS = ["必然", "必定", "一定", "肯定", "絕對", "绝对", "unavoidable",
                  "guaranteed", "100%", "不可能", "無可", "无可", "鐵定", "铁定",
                  "永遠", "永远", "零誤差", "零误差", "must be", "絕不", "绝不"]
# 同行出現這些 = 大概率帶證據（測量數字 / 來源標記 / 產物路徑）
EVIDENCE_MARKS = ["【實測】", "【实测】", "【推斷】", "【推断】", "【假設】", "【假设】",
                  "【已證偽】", "【已证伪】", ".json", ".md", "/tmp/", "Backup/",
                  "MiB", "GiB", "ms", "t/s", "pinned"]


def audit_doc(path: str) -> list[dict]:
    """掃一份 md：絕對化措辭但同行無數字/來源標記 → 列為「請人工複核」。保守、只初篩。"""
    import re
    flags = []
    in_code = False
    for i, line in enumerate(Path(path).read_text(errors="ignore").splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        hit = [w for w in ABSOLUTE_WORDS if w in line]
        if not hit:
            continue
        has_number = bool(re.search(r"\d", line))
        has_mark = any(m in line for m in EVIDENCE_MARKS)
        if not (has_number or has_mark):
            flags.append({"line": i, "words": hit, "text": line.strip()[:100]})
    return flags


def cmd_audit(paths: list[str]) -> int:
    """對 memory / SKILL.md / 白皮書做断言健康度核查。"""
    if not paths:
        paths = sorted(glob.glob(str(ROOT / ".workbuddy/memory/*.md")))
        paths += sorted(glob.glob(str(ROOT / "docs/*WHITEPAPER*.md")))
        paths += sorted(glob.glob(str(ROOT / "docs/*白皮*.md")))
    total = 0
    for p in paths:
        flags = audit_doc(p)
        if not flags:
            continue
        print(f"\n{p}")
        for f in flags:
            total += 1
            print(f"  L{f['line']}  絕對化={f['words']}  無同行來源")
            print(f"      {f['text']}")
    print(f"\n共 {total} 處「絕對化但無同行證據」→ 請補數據/來源，或改為【假設】。")
    print("（自動初篩、非定讞；已帶數字/來源的斷言不會被標）")
    return 1 if total else 0


def cmd_gate(args) -> int:
    """run 前 preflight：環境髒（thermal/swap/free/殘留進程/watchdog）→ 非 0，runner 拒跑。"""
    s = system_state()
    bad = []
    if s.get("thermal") != "NOMINAL":
        bad.append(f"thermal={s.get('thermal')}")
    if s.get("swap_used_mb") is not None and s["swap_used_mb"] > args.max_swap_mb:
        bad.append(f"swap {s['swap_used_mb']:.0f} > {args.max_swap_mb:.0f} MiB")
    if s.get("free_mb") is not None and s["free_mb"] < args.min_free_mb:
        bad.append(f"RAM free {s['free_mb']:.0f} < {args.min_free_mb:.0f} MiB")
    if not args.allow_shared:
        procs = running_procs()
        if procs:
            bad.append("殘留 llama 進程: " + ", ".join(str(p["pid"]) for p in procs))
    wd = subprocess.run(["pgrep", "-fl", "auto_bench_watchdog"], capture_output=True,
                        text=True).stdout.strip()
    if wd:
        bad.append(f"rogue watchdog: {wd.splitlines()[0]}")
    if bad:
        print("PREFLIGHT FAIL — 啟動會污染測量，拒跑：")
        for b in bad:
            print(f"  - {b}")
        return 2
    print(f"PREFLIGHT PASS  thermal={s.get('thermal')} swap={s.get('swap_used_mb'):.0f} "
          f"free={s.get('free_mb'):.0f}")
    return 0


def notify_next_actions(rep: dict) -> str | None:
    """kill 了進程 / 發現污染 → 在 NEXT_ACTIONS 寫一條，讓相關 agent 看到、不再重跑錯的。"""
    killed = rep.get("actions", [])
    polluted = [a for a in rep.get("artifacts", []) if a.get("polluted")]
    stressed = [a for a in rep.get("artifacts", [])
                if not a.get("polluted") and a.get("stressed")]
    quar = rep.get("quarantine", [])
    cons = rep.get("consistency", [])
    kill_procs = [p for p in rep.get("procs", []) if p["verdict"]["level"] == "kill"]
    if not (killed or polluted or stressed or quar or cons or kill_procs):
        return None
    lines = ["", f"## [watchdog {rep['when']}] 巡檢問題（看門人自動通知）", ""]
    for k in killed:
        lines.append(f"- 已終止 pid={k['pid']}（{k.get('result')}）："
                     + "; ".join(k.get("reasons", [])))
    for p in kill_procs:
        if not any(k["pid"] == p["pid"] for k in killed):
            lines.append(f"- 高危進程 pid={p['pid']}（報告模式、未終止）："
                         + "; ".join(p["verdict"]["reasons"]))
    for q in quar:
        lines.append(f"- 已隔離失效數字：{q['arms']} @ `{q['path']}`"
                     + (f" → `{q['moved_to']}`" if q.get("moved_to") else "（檔內標廢）"))
    for a in polluted:
        sc = " ".join(f"{k}={v:.2f}" for k, v in a["scores"].items() if v is not None)
        lines.append(f"- 量具失效、數字作廢：{a['tag']} {sc} "
                     f"（{'; '.join(a['kill_notes'])}）@ `{a['path']}`")
    for cf in cons:
        lines.append(f"- 宣稱vs實際待驗：{cf['issue']}；可疑 {cf['suspect_paths']}")
        lines.append(f"    {cf['hint']}")
    for a in stressed:
        sc = " ".join(f"{k}={v:.2f}" for k, v in a["scores"].items() if v is not None)
        lines.append(f"- 環境承壓但**數字有效、可引用**：{a['tag']} {sc} "
                     f"（{'; '.join(a['stress_notes'])}）；swap 結構問題另需修復")
    block = "\n".join(lines) + "\n"
    cand = sorted(glob.glob(str(ROOT / "docs" / "NEXT_ACTIONS_*.md")), reverse=True)
    path = Path(cand[0]) if cand else (ROOT / "docs" / f"NEXT_ACTIONS_{time.strftime('%Y-%m-%d')}.md")
    with path.open("a") as fh:
        fh.write(block)
    return str(path)


def _print_human(rep: dict) -> None:
    s = rep["system"]
    print(f"LANE WATCHDOG  {rep['when']}")
    print(f"  thermal={s.get('thermal')}  swap={s.get('swap_used_mb'):.0f}/"
          f"{s.get('swap_total_mb'):.0f} MiB  free={s.get('free_mb'):.0f}  "
          f"wired={s.get('wired_mb'):.0f}")
    print("\n進程：")
    if not rep["procs"]:
        print("  （無 llama-bench/server 在跑）")
    for p in rep["procs"]:
        v = p["verdict"]
        shape = f"p{p['prompt']}/n{p['gen']}" if p["is_bench"] else "server"
        print(f"  [{v['level'].upper():4}] pid={p['pid']} {shape} b={p['batch']} "
              f"load={p['load_mode']} etime={p['etime_s']:.0f}s")
        for r in v["reasons"]:
            print(f"         - {r}")
    print("\n最近產物（結論可否引用）：")
    if not rep["artifacts"]:
        print("  （近期無完成的 matrix 產物）")
    for a in rep["artifacts"]:
        if a["polluted"]:
            tag = "UNRELIABLE"
        elif a.get("stressed"):
            tag = "STRESSED*"
        else:
            tag = "clean"
        sc = " ".join(f"{k}={v:.2f}" for k, v in a["scores"].items() if v is not None)
        sd = a.get("row_sd", {}).get("tg")
        spread = a.get("rep_spread_pct", {}).get("tg")
        tail = []
        if sd is not None:
            tail.append(f"sd={sd:.2f}")
        if spread is not None:
            tail.append(f"rep散度={spread:.1f}%")
        print(f"  [{tag:8}] {a['tag']}  {sc}  " + " ".join(tail))
        c = a.get("cache", {})
        cp = []
        if c.get("hit_pct") is not None:
            cp.append(f"hit={c['hit_pct']:.1f}%")
        if c.get("misses") is not None:
            cp.append(f"miss={c['misses']}(comp{c.get('miss_compulsory')}/cap{c.get('miss_capacity')})")
        if c.get("us_per_job") is not None:
            cp.append(f"us/job={c['us_per_job']}")
        if c.get("per_miss_us") is not None:
            cp.append(f"us/miss={c['per_miss_us']/1000:.0f}k")
        if c.get("bytes_per_job") is not None:
            cp.append(f"B/job={c['bytes_per_job']/1024:.1f}KiB")
        if c.get("effective_mib_s") is not None:
            cp.append(f"eff={c['effective_mib_s']}MiB/s")
        if c.get("resident_mib") is not None:
            cp.append(f"resident={c['resident_mib']:.0f}")
        if c.get("workers_claimed") is not None:
            cp.append(f"workers宣稱={c['workers_claimed']}")
        if cp:
            print("         " + "  ".join(cp))
        for n in a.get("kill_notes", []):
            print(f"         ✗ {n}")
        for n in a.get("stress_notes", []):
            print(f"         · {n}")
    if rep.get("quarantine"):
        print("\n隔離動作：")
        for q in rep["quarantine"]:
            where = f" → {q['moved_to']}" if q.get("moved_to") else "（檔內標廢）"
            print(f"  {q['arms']} @ {q['path']}{where}")
    if rep.get("consistency"):
        print("\n宣稱 vs 實際（待驗證、非定論）：")
        for cf in rep["consistency"]:
            print(f"  ! {cf['issue']}")
            print(f"      {cf['hint']}")
    if rep["actions"]:
        print("\n止損動作：")
        for a in rep["actions"]:
            print(f"  pid={a['pid']}: {a['result']}  ({'; '.join(a.get('reasons', []))})")


def _append_log(rep: dict) -> None:
    try:
        RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_LOG.open("a") as fh:
            fh.write(json.dumps(rep, ensure_ascii=False) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("乾淨系統 + 正常 bench -> ok，不誤殺")
    clean_sys = {"swap_used_mb": 200.0, "free_mb": 2000.0, "thermal": "NOMINAL"}
    p = {"pid": 1, "is_bench": True, "batch": 5632, "load_mode": "none"}
    j = judge_proc(p, clean_sys)
    check("clean bench => ok", j["level"] == "ok")

    print("swap 爆 + free 極低 + 大 batch -> kill")
    dirty_sys = {"swap_used_mb": 4000.0, "free_mb": 80.0, "thermal": "NOMINAL"}
    j = judge_proc({"pid": 2, "is_bench": True, "batch": 5632,
                    "load_mode": "none"}, dirty_sys)
    check("dangerous => kill", j["level"] == "kill" and j["dangerous_now"])

    print("swap 中等、free 尚可 -> warn（不 kill）")
    mid_sys = {"swap_used_mb": 1500.0, "free_mb": 600.0, "thermal": "NOMINAL"}
    j = judge_proc({"pid": 3, "is_bench": True, "batch": 5632,
                    "load_mode": "none"}, mid_sys)
    check("mid => warn, not kill", j["level"] == "warn")

    print("free 極低即使 swap 不高 -> kill（瀕臨 jetsam）")
    j = judge_proc({"pid": 4, "is_bench": True, "batch": 5632,
                    "load_mode": "none"},
                   {"swap_used_mb": 500.0, "free_mb": 60.0})
    check("free cliff => kill", j["level"] == "kill")

    print("server（非 bench）在高危下也該止損")
    j = judge_proc({"pid": 5, "is_bench": False, "batch": 0,
                    "load_mode": None}, dirty_sys)
    check("server dangerous => kill", j["level"] == "kill")

    print("system_state 真的能解析出數字")
    st = system_state()
    check("swap_used readable", st.get("swap_used_mb") is not None)
    check("free readable", st.get("free_mb") is not None)

    print("etime 解析正確")
    check("mm:ss", _parse_etime("03:30") == 210)
    check("hh:mm:ss", _parse_etime("1:02:03") == 3723)
    check("dd-hh:mm:ss", _parse_etime("2-01:00:00") == 2 * 86400 + 3600)

    print("產物判決：swap 高但量具穩 -> stressed、數字仍有效")
    stressed_arm = {
        "memory": {"launch": {"swap_used_mb": 2200}, "end": {"swap_used_mb": 5000}},
        "thermal": {"worst": {"label": "NOMINAL"}},
        "rows": [{"n_prompt": 0, "avg_ts": 12.2, "stddev_ts": 0.3,
                  "samples_ts": [12.5, 11.98, 12.11]}]}
    j = judge_artifact(stressed_arm)
    check("swap高+低散度 => stressed, not unreliable",
          j["stressed"] and not j["unreliable"])

    print("量具高散度 -> unreliable（即使 thermal NOMINAL）")
    bad_arm = {
        "memory": {"launch": {"swap_used_mb": 100}, "end": {"swap_used_mb": 200}},
        "thermal": {"worst": {"label": "NOMINAL"}},
        "rows": [{"n_prompt": 0, "avg_ts": 8.5, "stddev_ts": 3.0,
                  "samples_ts": [10.5, 6.5, 8.5]}]}
    j = judge_artifact(bad_arm)
    check("高散度 => unreliable", j["unreliable"] and j["spread_high"])

    print("thermal HEAVY -> unreliable")
    hot_arm = {
        "memory": {"launch": {"swap_used_mb": 100}, "end": {"swap_used_mb": 100}},
        "thermal": {"worst": {"label": "HEAVY"}},
        "rows": [{"n_prompt": 0, "avg_ts": 12.0, "stddev_ts": 0.2,
                  "samples_ts": [12.1, 11.9, 12.0]}]}
    j = judge_artifact(hot_arm)
    check("thermal bad => unreliable", j["unreliable"] and j["thermal_bad"])

    print("consistency_check：同 workers、us/job 離散大 -> flag")
    art = [
        {"scores": {"tg": 12.0}, "cache": {"us_per_job": 27000, "workers_claimed": 8}},
        {"scores": {"tg": 11.5}, "cache": {"us_per_job": 31000, "workers_claimed": 8}}]
    check("us/job 離散>12% => flag", len(consistency_check(art)) == 1)

    print("quarantine_artifacts：失效產物被隔離、原路徑留指針（用臨時 QUAR_DIR）")
    import tempfile
    tmpdir = tempfile.mkdtemp()
    global QUAR_DIR
    saved_quar = QUAR_DIR
    QUAR_DIR = Path(tmpdir)
    try:
        f = Path(tmpdir) / "bad.json"
        f.write_text(json.dumps([{
            "tag": "prod-new",
            "memory": {"launch": {"swap_used_mb": 100}, "end": {"swap_used_mb": 200}},
            "thermal": {"worst": {"label": "HEAVY"}},
            "rows": [{"n_prompt": 0, "avg_ts": 12.0, "samples_ts": [12.1, 11.9]}]}]))
        acts = quarantine_artifacts(
            [{"path": str(f), "tag": "prod-new", "polluted": True,
              "kill_notes": ["thermal worst=HEAVY"], "scores": {"tg": 12.0}}],
            "selftest", True)
        check("quarantine acted", len(acts) == 1)
        check("pointer left at origin",
              json.loads(f.read_text()).get("_quarantine_pointer") is True)
    finally:
        QUAR_DIR = saved_quar
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS — 看門人判決與誤殺防護就緒")
    return 0


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # 子命令 audit（memory/skill/白皮書断言核查）
    if raw and raw[0] == "audit":
        return cmd_audit(raw[1:])

    # 子命令 gate（run 前 preflight）
    if raw and raw[0] == "gate":
        gp = argparse.ArgumentParser(description="run 前環境 preflight")
        gp.add_argument("--max-swap-mb", type=float, default=1024)
        gp.add_argument("--min-free-mb", type=float, default=400)
        gp.add_argument("--allow-shared", action="store_true")
        return cmd_gate(gp.parse_args(raw[1:]))

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kill", action="store_true", help="對 kill 級進程終止止損")
    ap.add_argument("--quarantine", action="store_true",
                    help="對 UNRELIABLE 產物主動隔離、移除錯誤數字")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-notify", action="store_true",
                    help="不在 NEXT_ACTIONS 寫通知（預設會寫）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(raw)
    if args.selftest:
        return selftest()

    rep = inspect(do_kill=args.kill, do_quarantine=args.quarantine)
    if args.json:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    else:
        _print_human(rep)
    if not args.no_log:
        _append_log(rep)

    # kill 了進程 / 發現污染 → 通知相關 agent（寫 NEXT_ACTIONS）
    if not args.no_notify:
        note = notify_next_actions(rep)
        if note and not args.json:
            print(f"\n已在 NEXT_ACTIONS 通知相關 agent：{note}")

    # 退出碼：2=kill 級進程；3=污染產物；1=warn；0=乾淨
    if any(p["verdict"]["level"] == "kill" for p in rep["procs"]):
        return 2
    if any(a["polluted"] for a in rep["artifacts"]):
        return 3
    if any(p["verdict"]["level"] == "warn" for p in rep["procs"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
