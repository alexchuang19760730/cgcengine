#!/usr/bin/env python3
"""Run the two re-measurement families the moment the box is clean, then write the report.

Why a gate and not just a command
---------------------------------
The 2026-09-25 pass failed at the box, not at the code: eight P0/P1/P2 arms started with
3849-6004 MiB of swap already used and every one of them ended HEAVY, and the welded arm vs its
own control flipped direction within 40 minutes on one build (p012 -13.7% then +6.9%). An
out-of-session reference cannot answer "which switch costs", so the re-run has to (a) start on a
clean box, (b) keep its arms cooled alike, and (c) carry the session's identity in the artifact.

What this does
  1. waits for the window: 0 llama processes, swap used <= --swap-limit, thermal NOMINAL, and
     `server_window.quiet()` admits the port. Every one of those probes is imported, not
     re-implemented -- the repo already paid for having two definitions of "usable memory".
  2. runs the two drivers sequentially, never concurrently (two drivers interleave launches and
     void both), re-checking the window before each step:
        p0p1p2  Backup/mtpoff_base/p0_vs_madv_iso.py --passes 2 --cool 420
        s1      Backup/eseries/driver.py E0 E4 E2
  3. writes `docs/RERUN_SESSION_<stamp>.md` and the same-stem `.json` sidecar, carrying the
     session fingerprint: boot epoch + first launch, HEAD, engine digest, per-step window
     evidence and box state. `cross_session_rule` in the sidecar states what a verdict from
     another session id is worth (nothing) so the next reader does not have to remember.

Usage
    python3 scripts/check/restart_rerun.py run [--swap-limit 2048] [--wait-min 240]
    python3 scripts/check/restart_rerun.py report          # regenerate from artifacts, no GPU
    python3 scripts/check/restart_rerun.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
sys.path.insert(0, str(CHECK))

import memory_pressure as mp          # noqa: E402  swap_used_mb / llama_procs / top_rss
import thermal_pressure as tp         # noqa: E402  level / label / wait_nominal
import server_window as sw            # noqa: E402  quiet / decision

DOCS = ROOT / "docs"
P0_SUM = Path("/tmp/p0iso/summary.json")
S1_SUM = ROOT / "Backup/eseries/results.json"
DEFAULT_SWAP_LIMIT_MB = 2048.0        # MEASUREMENT_CONTRACT §3.3 start line
CROSS_SESSION_RULE = (
    "a verdict is only comparable to another arm with the SAME session id; the 2026-09-25 pass "
    "flipped the welded arm's sign between two sessions on one build, so a cross-session "
    "comparison is refused rather than caveated")


# ── the window ────────────────────────────────────────────────────────────────

def window_state(swap_limit_mb: float = DEFAULT_SWAP_LIMIT_MB, port: int = sw.PROD_PORT,
                 need_mb: float | None = None, *,
                 swap=None, procs=None, thermal=None, decision=None) -> dict:
    """(ok, reasons, evidence). Probes are injectable so the selftest can feed a busy box.

    The vetoes are the terms that actually dominated the 2026-09-25 readings: carried swap,
    thermal, neighbours, the port, and the launcher's own guard. Memory is EVIDENCE unless
    `need_mb` is given, and that is a deliberate choice rather than a threshold nobody meets:
    measured right after the reboot, `server_window.quiet()`'s reclaimable term reads 5848 MB
    against its own 8000 MB default while swap is 0.00M and no llama process exists. 8000 MB is
    unreachable on this 16 GB box with a desktop logged in (active 5.2 + wired 1.8 + compressed
    2.1 GB), so ANDing it in means the gate never opens and the run it guards never happens.
    The number is recorded either way, and `--need-mb` restores the strict rule on request.
    """
    swap = swap or mp.swap_used_mb
    procs = procs or mp.llama_procs
    thermal = thermal or tp.level
    decision = decision or sw.decision
    swap_mb, n_procs, lv = swap(), procs(), thermal()
    dec = decision(port) or {}
    terms = dec.get("harness_terms") or {}
    reasons = []
    if n_procs != 0:
        reasons.append(f"llama process(es) alive: {n_procs}")
    if swap_mb is None:
        reasons.append("swap unreadable")
    elif swap_mb > swap_limit_mb:
        reasons.append(f"swap {swap_mb:.0f}MB > limit {swap_limit_mb:.0f}MB")
    if lv != 0:
        reasons.append(f"thermal {tp.label(lv)} (need NOMINAL)")
    if terms.get("foreign") is False:
        reasons.append(f"other llama process(es): {dec.get('foreign_llama')}")
    if terms.get("port") is False:
        reasons.append(f"port {port} already listened on")
    if need_mb is not None and not terms.get("memory", True):
        reasons.append(f"reclaimable={dec.get('reclaimable_mb')}MB<{need_mb:.0f}MB "
                       f"(veto you asked for with --need-mb)")
    if dec.get("launcher_admits") is False:
        reasons.append("launcher's own guard refuses: " + ",".join(dec.get("refused_by") or []))
    return {"ok": not reasons, "reasons": reasons,
            "evidence": {"swap_used_mb": swap_mb, "llama_procs": n_procs,
                         "thermal": tp.label(lv), "thermal_level": lv,
                         "reclaimable_mb": dec.get("reclaimable_mb"),
                         "harness_admits": dec.get("harness_admits"),
                         "launcher_admits": dec.get("launcher_admits"),
                         "definitions_agree": dec.get("agree"),
                         "binding_definition": dec.get("binding"),
                         "memory_term_is_veto": need_mb is not None}}


def wait_window(swap_limit_mb: float, wait_min: float, poll_s: float = 30.0, *, need_mb=None,
                log=print) -> dict:
    """Poll the window until it opens or `wait_min` expires. Never launches on a closed box."""
    t0 = time.time()
    n = 0
    while True:
        st = window_state(swap_limit_mb, need_mb=need_mb)
        n += 1
        if st["ok"]:
            log(f"[window] OPEN after {(time.time() - t0) / 60:.1f} min ({n} probes): "
                f"{st['evidence']}")
            st["waited_min"] = round((time.time() - t0) / 60, 1)
            st["probes"] = n
            return st
        if (time.time() - t0) / 60 >= wait_min:
            log(f"[window] still closed after {wait_min:.0f} min: {st['reasons']} -- NOT running")
            st["waited_min"] = round((time.time() - t0) / 60, 1)
            st["probes"] = n
            st["timed_out"] = True
            return st
        if n == 1:
            log(f"[window] closed: {'; '.join(st['reasons'])} (polling every {poll_s:.0f}s)")
        time.sleep(poll_s)


# ── the two families ──────────────────────────────────────────────────────────

def steps(spec: str) -> list[dict]:
    return [{"name": "p0p1p2",
             "cmd": [sys.executable, str(ROOT / "Backup/mtpoff_base/p0_vs_madv_iso.py"),
                     "--passes", "2", "--cool", "420", "--workdir", "/tmp/p0iso"],
             "artifact": str(P0_SUM)},
            {"name": "s1",
             "cmd": [sys.executable, str(ROOT / "Backup/eseries/driver.py")] + spec.split(),
             "artifact": str(S1_SUM)}]


def run_step(step: dict, swap_limit_mb: float, wait_min: float, need_mb=None, log=print) -> dict:
    win = wait_window(swap_limit_mb, wait_min, need_mb=need_mb, log=log)
    tp.wait_nominal(timeout_s=420.0, log=log)          # the ONE cooldown loop, reused
    swap_before, procs_before = mp.swap_used_mb(), mp.llama_procs()
    t0 = time.time()
    log(f"[{step['name']}] $ {' '.join(step['cmd'])}")
    rc = subprocess.run(step["cmd"], cwd=ROOT).returncode
    log(f"[{step['name']}] rc={rc} wall={(time.time() - t0) / 60:.1f} min")
    return {"name": step["name"], "cmd": step["cmd"], "rc": rc,
            "wall_min": round((time.time() - t0) / 60, 1), "window": win,
            "swap_before_mb": swap_before, "swap_after_mb": mp.swap_used_mb(),
            "llama_procs_before": procs_before, "artifact": step["artifact"],
            "status": "ran" if rc == 0 else "failed"}


# ── the session fingerprint and the report ────────────────────────────────────

def boot_epoch() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True,
                             text=True).stdout
        inside = out.split("{", 1)[1]
        return float(inside.split(",")[0].split("sec =")[1])
    except Exception:  # noqa: BLE001 - an unknown boot time must not kill the run
        return None


def engine_digest() -> dict:
    out = {}
    for rel in ("src/llama.cpp/build/bin/libllama-common.0.dylib",
                "src/llama.cpp/build/bin/llama-bench"):
        p = ROOT / rel
        if p.exists():
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[Path(rel).name] = h.hexdigest()
    out["engine_digest"] = out.get("libllama-common.0.dylib", "")[:16]
    return out


def session_id(booted: float | None, first_launch: float) -> str:
    return f"{(int(booted) if booted else 0)}-{int(first_launch)}"


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001 - a missing artifact is reported, not raised
        return {"error": f"{path}: {e}"}


def p0_table(doc: dict) -> list[dict]:
    out = []
    for a in doc.get("arms", []):
        out.append({"tag": a.get("tag"), "pass": a.get("pass"), "order_index": a.get("order_index"),
                    "tg": a.get("tg"), "pp": a.get("pp"), "rc": a.get("rc"),
                    "thermal": a.get("thermal"), "reads": a.get("reads"),
                    "pread_usec": a.get("pread_usec"),
                    "warm_skip_applied": a.get("warm_skip_applied"),
                    "swap_growth_mib": a.get("swap_growth_mib"),
                    "swap_arms": a.get("swap_arms"), "overrides": a.get("overrides")})
    return out


def s1_table(doc: dict) -> list[dict]:
    out = []
    for key in doc.get("_order", []):
        v = doc.get(key) or {}
        row = {"key": key, "rc": v.get("rc")}
        for arm in (v.get("arms") or []):
            rows = arm.get("rows") or []
            tg = next((r for r in rows if not r.get("n_prompt")), {})
            pp = next((r for r in rows if r.get("n_prompt")), {})
            row.update({"tag": arm.get("tag"), "tg": tg.get("avg_ts"), "pp": pp.get("avg_ts"),
                        "samples": tg.get("samples_ts"), "thermal": arm.get("thermal"),
                        "cell": arm.get("cell"), "box_gate": arm.get("box_gate")})
        if "probe_answer" in v:
            row["probe_answer"] = v["probe_answer"]
        if v.get("rc") not in (0, None):
            row["tail"] = (v.get("tail") or "")[-400:]
        out.append(row)
    return out


def render(sess: dict, p0: dict, s1: dict) -> str:
    L = []
    L.append(f"# 重開機後的重測 session `{sess['session']['id']}`\n")
    L.append("由 `scripts/check/restart_rerun.py run` 產生：窗戶一開就跑，兩個家族各跑自己的"
             "驅動，產物是這份 md 與同名 `.json` sidecar。\n")
    L.append("## 0. session 指紋（這份報告的身分）\n")
    L.append("| 項 | 值 |\n|---|---|")
    s = sess["session"]
    L.append(f"| session id | `{s['id']}` |")
    L.append(f"| 開機時刻 | {s.get('booted_at')}（boot epoch {s.get('boot_epoch')}） |")
    L.append(f"| 首次啟動 | {s.get('first_launch_at')} |")
    L.append(f"| HEAD | `{s['head']}` |")
    L.append(f"| engine digest | `{s['engine_digest']}` |")
    L.append(f"| swap 門檻 | {s['swap_limit_mb']:.0f} MiB（契約 §3.3） |")
    L.append(f"| thermal 要求 | NOMINAL |")
    L.append(f"\n> **跨 session 規則**：{CROSS_SESSION_RULE}\n")
    L.append("## 1. 每一步的窗戶證據\n")
    L.append("| step | rc | wall | 開跑前窗戶 | swap 前→後 (MiB) | thermal 開跑前 |"
             "\n|---|---|---|---|---|---|")
    for st in sess["steps"]:
        w = st.get("window", {})
        ev = w.get("evidence", {})
        L.append(f"| `{st['name']}` | {st['rc']} | {st['wall_min']} min | "
                 f"{'OPEN' if w.get('ok') else 'closed/timeout'}"
                 f"{'' if w.get('ok') else ': ' + '; '.join(w.get('reasons', []))} | "
                 f"{st.get('swap_before_mb')} → {st.get('swap_after_mb')} | "
                 f"{ev.get('thermal')} |")
    L.append("\n## 2. P0/P1/P2 四臂（in-session 判定）\n")
    if isinstance(p0, dict) and p0.get("arms"):
        L.append(f"- 判定：**{p0.get('verdict', {}).get('verdict', '(無)')}**")
        L.append(f"- pct_vs_armed：`{p0.get('verdict', {}).get('pct_vs_armed')}`"
                 f"（負 = 該臂比 armed 慢）")
        L.append(f"- 每臂散度：`{p0.get('verdict', {}).get('spread_pct')}`")
        L.append(f"- 臂序（逐 pass 輪替）：`{p0.get('arm_order')}`；冷卻 {p0.get('cool_s')} s")
        L.append("\n| tag | pass | 序 | tg | pp | thermal | reads | pread_us | warm_skip_applied | Δswap | swap_arms |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|")
        for r in p0_table(p0):
            L.append(f"| `{r['tag']}` | {r['pass']} | {r['order_index']} | {r['tg']} | {r['pp']} | "
                     f"{r['thermal']} | {r['reads']} | {r['pread_usec']} | "
                     f"{r['warm_skip_applied']} | {r['swap_growth_mib']} | {r['swap_arms']} |")
    else:
        L.append(f"- **沒有可用產物**（`{P0_SUM}`）⇒ 這一格不可判讀。")
    L.append("\n## 3. S1 的 E0/E4/E2\n")
    if isinstance(s1, dict) and s1.get("_order"):
        L.append("| key | rc | arm | tg | pp | samples | thermal | probe_answer |\n"
                 "|---|---|---|---|---|---|---|---|")
        for r in s1_table(s1):
            L.append(f"| `{r['key']}` | {r['rc']} | {r.get('tag')} | {r.get('tg')} | {r.get('pp')} | "
                     f"{r.get('samples')} | {r.get('thermal')} | {r.get('probe_answer')} |")
    else:
        L.append(f"- **沒有可用產物**（`{S1_SUM}`）⇒ 這一格不可判讀。")
    L.append("\n## 4. 這份報告不主張的事\n")
    L.append(f"- 兩個家族都只在自己的 session `{s['id']}` 內可比；跨 session 的一律不引用。")
    L.append("- 任何一步的 rc ≠ 0，或窗戶未開（timeout）⇒ 該步的數字不列入判定，只留證據。")
    L.append("- 每臂的 `thermal` 若 worst 不是 NOMINAL，該臂依契約 §5.1 屬診斷價。")
    L.append("- S1 的輸出正確性由 E0 的 `probe_answer` 決定，不由 t/s 決定。")
    return "\n".join(L) + "\n"


def write_report(sess: dict, p0: dict, s1: dict, stamp: str | None = None) -> dict:
    stamp = stamp or time.strftime("%Y%m%d_%H%M")
    DOCS.mkdir(exist_ok=True)
    md = DOCS / f"RERUN_SESSION_{stamp}.md"
    js = DOCS / f"RERUN_SESSION_{stamp}.json"
    sidecar = {"kind": "cgc-rerun-session", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "session": sess["session"], "steps": sess["steps"],
               "cross_session_rule": CROSS_SESSION_RULE,
               "p0p1p2": {"verdict": (p0 or {}).get("verdict"), "arms": p0_table(p0 or {}),
                          "artifact": str(P0_SUM)},
               "s1": {"rows": s1_table(s1 or {}), "artifact": str(S1_SUM)},
               "report": md.name}
    md.write_text(render(sess, p0, s1), encoding="utf-8")
    js.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": str(md), "sidecar": str(js)}


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_run(args) -> int:
    booted = boot_epoch()
    first_launch = time.time()
    sess = {"session": {"id": session_id(booted, first_launch), "boot_epoch": booted,
                        "booted_at": (time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(booted)) if booted else None),
                        "first_launch_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "head": head(), "engine_digest": engine_digest()["engine_digest"],
                        "identity": engine_digest(), "swap_limit_mb": args.swap_limit,
                        "thermal_required": "NOMINAL"},
            "steps": []}
    print(f"[session] id={sess['session']['id']} head={sess['session']['head']} "
          f"digest={sess['session']['engine_digest']}")
    stamp = time.strftime("%Y%m%d_%H%M")
    wanted = [s for s in steps(args.spec) if args.only in ("", s["name"])]
    for st in wanted:
        sess["steps"].append(run_step(st, args.swap_limit, args.wait_min, args.need_mb))
        # Write after EVERY step, under one stamp: the family that finishes first must be readable
        # while the second is still waiting for a window (the first pass only wrote at the end,
        # which turned a finished verdict into a 90-minute wait for an unrelated one).
        write_report(sess, read_json(P0_SUM), read_json(S1_SUM), stamp)
    out = write_report(sess, read_json(P0_SUM), read_json(S1_SUM), stamp)
    print(f"[report] {out['report']}\n[sidecar] {out['sidecar']}")
    return 0 if all(s["rc"] == 0 for s in sess["steps"]) else 1


def cmd_report(args) -> int:
    p0, s1 = read_json(P0_SUM), read_json(S1_SUM)
    booted = boot_epoch()
    sess = {"session": {"id": session_id(booted, time.time()), "boot_epoch": booted,
                        "booted_at": None, "first_launch_at": "n/a (report-only)",
                        "head": head(), "engine_digest": engine_digest()["engine_digest"],
                        "identity": engine_digest(), "swap_limit_mb": args.swap_limit,
                        "thermal_required": "NOMINAL"},
            "steps": []}
    out = write_report(sess, p0, s1)
    print(f"[report] {out['report']}\n[sidecar] {out['sidecar']}")
    return 0


def selftest() -> int:
    """Frozen checks: the gate must refuse each closed condition, and the report must carry id."""
    bad = []

    def expect(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r} want {want!r}")

    def dec(d, free=9000.0, launcher=True):
        return lambda port: {"reclaimable_mb": free, "foreign_llama": d, "port_held": d,
                             "harness_terms": {"memory": free >= 8000.0, "foreign": not d,
                                               "port": not d},
                             "harness_admits": free >= 8000.0 and not d, "launcher_admits": launcher,
                             "agree": True, "binding": None, "refused_by": [] if launcher else ["launcher"]}

    clean = dict(swap=lambda: 0.0, procs=lambda: 0, thermal=lambda: 0, decision=dec(False))
    st = window_state(swap=clean["swap"], procs=clean["procs"], thermal=clean["thermal"],
                      decision=clean["decision"])
    expect("clean box -> open", st["ok"], True)
    for name, kw, want in (
            ("swap above limit", {"swap": lambda: 4096.0}, "swap 4096MB > limit 2048MB"),
            ("swap unreadable", {"swap": lambda: None}, "swap unreadable"),
            ("a llama process alive", {"procs": lambda: 1}, "llama process(es) alive: 1"),
            ("thermal MODERATE", {"thermal": lambda: 1}, "thermal MODERATE (need NOMINAL)"),
            ("port held", {"decision": dec(True)}, "port 8080 already listened on"),
            ("launcher's own guard refuses", {"decision": dec(False, launcher=False)},
             "launcher's own guard refuses: launcher")):
        env = dict(clean)
        env.update(kw)
        st = window_state(swap=env["swap"], procs=env["procs"], thermal=env["thermal"],
                          decision=env["decision"])
        expect(name + " -> closed", st["ok"], False)
        if want not in st["reasons"]:
            bad.append(f"{name}: reason missing ({st['reasons']})")

    # This box, right after the reboot: 5848 MB reclaimable, 0 swap, 0 neighbours. It must be
    # OPEN by default (that is the whole point of the run) and CLOSED when the veto is asked for.
    box = dict(swap=lambda: 0.0, procs=lambda: 0, thermal=lambda: 0,
               decision=dec(False, free=5848.0))
    st = window_state(swap=box["swap"], procs=box["procs"], thermal=box["thermal"],
                      decision=box["decision"])
    expect("reclaimable below the shared probe's 8000MB default -> still open (evidence, not veto)",
           st["ok"], True)
    expect("and the number is recorded as evidence", st["evidence"]["reclaimable_mb"], 5848.0)
    st = window_state(need_mb=8000.0, swap=box["swap"], procs=box["procs"], thermal=box["thermal"],
                      decision=box["decision"])
    expect("--need-mb restores the strict veto", st["ok"], False)
    expect("--need-mb states it was asked for",
           any("veto you asked for" in r for r in st["reasons"]), True)

    expect("session id changes with boot", session_id(1000, 2000) != session_id(1001, 2000), True)
    expect("session id changes with first launch", session_id(1000, 2000) != session_id(1000, 2001),
           True)
    expect("cross-session rule is present and refuses",
           "refused" in CROSS_SESSION_RULE and "SAME session id" in CROSS_SESSION_RULE, True)

    # the renderer must mark a missing artifact instead of printing a zero
    md = render({"session": {"id": "t-1", "head": "x", "engine_digest": "y", "swap_limit_mb": 2048.0},
                 "steps": []}, {"error": "no file"}, {"error": "no file"})
    expect("missing artifact is stated, not zeroed",
           "沒有可用產物" in md and "| `p0p1p2` |" not in md, True)
    expect("report carries the session id", "`t-1`" in md, True)
    expect("report carries the refusal rule", "跨 session 規則" in md, True)

    if bad:
        print("SELFTEST FAILED:")
        for b in bad:
            print("  -", b)
        return 1
    print(f"SELFTEST OK ({len(bad)} failures; window gate: 6 refusals + clean + the real box,"
          " session id, renderer honesty)")
    return 0


def detach(argv: list[str], log_path: Path) -> int:
    """Re-exec this script in its own session so it outlives the calling shell.

    macOS has no `setsid` binary, and `nohup cmd &` from a tool-invoked shell dies with that
    shell's process group -- the 2026-09-25 attempt produced a log with one line and no process.
    A two-fork/setsid dance is the only thing that survives, and it belongs next to the runner
    it detaches rather than in a fifth private copy.
    """
    pid = os.fork()
    if pid > 0:
        print(f"[detach] launched pid={pid}\n[detach] log: {log_path}")
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for fd in (1, 2):
        os.dup2(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND), fd)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    # -u: a detached run writes to a file, and buffered prints make a two-hour run look dead
    # (measured 2026-09-25: the log stayed empty 45 s in, with the runner alive).
    os.execv(sys.executable, [sys.executable, "-u", str(Path(__file__).resolve())] + argv)
    os._exit(1)                                   # only reached if execv fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="wait for the window, run both families, write the report")
    p = sub.add_parser("report", help="regenerate the report from the artifacts (no GPU)")
    sub.add_parser("selftest")
    for sp in (r, p):
        sp.add_argument("--swap-limit", type=float, default=DEFAULT_SWAP_LIMIT_MB)
    r.add_argument("--wait-min", type=float, default=240.0,
                   help="give up (without launching) after this long with no window")
    r.add_argument("--spec", default="E0 E4 E2", help="steps for the S1 driver")
    r.add_argument("--only", default="", choices=("", "p0p1p2", "s1"),
                   help="run one family only -- a family the window closed on needs its own "
                        "clean box, and swap does not drain back below the contract limit")
    r.add_argument("--detach", action="store_true",
                   help="re-exec in its own session (survives the calling shell) and return")
    r.add_argument("--need-mb", type=float, default=None,
                   help="also veto on reclaimable memory (off by default: the 8000 MB default of "
                        "the shared probe is unreachable on this box -- see window_state)")
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "run":
        if args.detach:
            log = ROOT / "Backup/rerun" / f"run_{time.strftime('%Y%m%d_%H%M')}.log"
            argv = ["run", "--swap-limit", str(args.swap_limit),
                    "--wait-min", str(args.wait_min), "--spec", args.spec]
            for flag, val in (("--only", args.only), ("--need-mb", args.need_mb)):
                if val not in ("", None):       # never pass an empty value: argparse eats the next arg
                    argv += [flag, str(val)]
            return detach(argv, log)
        return cmd_run(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
