#!/usr/bin/env python3
"""Compare the CGC_LAYER_AHEAD_PREFETCH arms from the driver logs + the server DECPROF blocks.

Reads:  /tmp/la_<arm>.log          (driver stdout: launch state, t/s, and its own step table)
        Backup/cgc_logs/*.log      (the server logs, for the per-step wait/cb/submit/gap lines)
Prints: per arm -- decode t/s, and the MEDIAN of (step, wait, cb, submit, union, gap) over the
        steady-state decode steps, plus the arm's launch memory state (drift control).
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/Users/alexchuang/Documents/flashkv-devserver")
ARMS = sys.argv[1:] or ["off", "on", "off2"]

STEP = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \(\d+%\) cb=([\d.]+) \(\d+%\) submit=([\d.]+) \(\d+%\) ntok=(\d+) \| "
    r"layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+)")


def launch_state(arm: str) -> dict:
    txt = Path(f"/tmp/la_{arm}.log").read_text(errors="replace") if Path(f"/tmp/la_{arm}.log").exists() else ""
    m = re.search(r"mem=\{[^}]*\}", txt)
    env = Path(f"/tmp/la_{arm}/prof.env.json")
    e = json.loads(env.read_text()) if env.exists() else {}
    return {"mem": m.group(0) if m else None,
            "layer_ahead": e.get("env", {}).get("CGC_LAYER_AHEAD_PREFETCH")}


def driver_table(arm: str) -> str:
    p = Path(f"/tmp/la_{arm}.log")
    if not p.exists():
        return "(no driver log)"
    lines = [l for l in p.read_text(errors="replace").splitlines()
             if l.startswith("| `") or "decode t/s" in l]
    return "\n".join(lines)


def server_rows(arm: str) -> list[dict]:
    """Steady-state decode steps (ntok=1) from the server log of THIS arm.

    The arm's own log is found by the launch timestamp the driver recorded, because
    Backup/cgc_logs holds every arm's log side by side and picking "newest" would mix them.
    """
    # The arm's server log is named in its own launch log (Backup/cgc_logs/llama_server_*.log),
    # which is what /tmp/la_<arm>/prof.launch.log records. Reading "the newest log" instead would
    # mix arms: all three write into the same directory.
    dates = []
    for cand in (Path(f"/tmp/la_{arm}.log"), Path(f"/tmp/la_{arm}") / "prof.launch.log"):
        if cand.exists():
            dates += re.findall(r"llama_server_(\d{8}_\d{6})\.log", cand.read_text(errors="replace"))
    rows = []
    for d in sorted(set(dates)):
        f = ROOT / "Backup" / "cgc_logs" / f"llama_server_{d}.log"
        if not f.exists():
            continue
        for m in STEP.finditer(f.read_text(errors="replace")):
            g = m.groups()
            if int(g[7]) != 1:          # ntok: decode only
                continue
            rows.append({"step": int(g[0]), "segs": int(g[1]),
                         "total": float(g[3]), "wait": float(g[4]), "cb": float(g[5]),
                         "submit": float(g[6]), "gpu": float(g[8]), "union": float(g[9]),
                         "gap": float(g[10])})
    return rows


def med(xs):
    return round(st.median(xs), 2) if xs else None


for arm in ARMS:
    rows = server_rows(arm)
    # first 4 steps after the warm-up are dominated by compulsory misses; report both
    warm = [r for r in rows if r["step"] >= 16]
    print(f"\n===== arm {arm} =====")
    print("launch:", launch_state(arm))
    print("driver:", driver_table(arm) or "(none)")
    if not warm:
        print("server rows: none")
        continue
    print(f"steady decode steps n={len(warm)} (step>=16)")
    for k in ("total", "wait", "cb", "submit", "union", "gap", "gpu"):
        vals = [r[k] for r in warm]
        print(f"  {k:7s} median={med(vals):7} min={min(vals):7} max={max(vals):7}")
    idle = [r["gap"] / r["union"] * 100 for r in warm if r["union"] > 0]
    print(f"  gpuidle median={med(idle)}% of span")
