#!/usr/bin/env python3
"""Run the M1/M2/M3 oracle gate inside a quiet window, and record the reading WITH its provenance.

Why a wrapper instead of just running the gate
----------------------------------------------
Three things have gone wrong repeatedly on this line, and all three are structural:

  1. **A launch into a busy box measures the neighbour.** The gate loads a 13 GB model + an 8 GiB
     pool onto a 16 GB machine. Started while another session's server is up, it over-commits and
     the numbers describe a thrash regime (measured spread on this box: 17% on identical configs).
  2. **A reading without its engine digest is not a reading.** `summary_*.json` carries
     `engine_digest`; a number quoted without it cannot be attributed to a binary.
  3. **A non-comparable run looks identical to a real one at a glance.** Three runs at 20:56-20:58
     reported *M1 1/9* against a reference dumped in a different configuration. That is not a
     regression and not a pass — the gate's own `comparable: false` says the comparison is void,
     but the score is what a reader sees. Here it is never published as a reading.

So: wait for a window (chainable behind another job), run the gate, then write a record that
refuses to call a void comparison a reading.

Usage
-----
    # run once at the next quiet window
    python3 scripts/check/m123_gate_window.py run --profile prefill250

    # chain: start only after the null-cell sweep has exited, then wait for a window
    python3 scripts/check/m123_gate_window.py run --after-pid 12345

    python3 scripts/check/m123_gate_window.py selftest
    python3 scripts/check/m123_gate_window.py --status
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "check" / "decode_window_harness.py"
GATE = ROOT / "scripts" / "check" / "m123_oracle_gate.py"
RESULT_DIR = ROOT / "Backup" / "m123_oracle_gate"
STATE = ROOT / "Backup" / "phase_decomp" / "m123_gate_window_state.json"
LOG_DIR = ROOT / "Backup" / "phase_decomp"

# The profile whose numerics the v6 reference was dumped under. Anything else is NOT comparable
# to DEFAULT_REF, and the gate will (correctly) say so -- so it is not a default worth overriding
# silently; --profile exists to make an intentional deviation explicit.
ANCHOR_PROFILE = "prefill250"


def _load_harness():
    spec = importlib.util.spec_from_file_location("cgc_window_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_sw():
    """The shared probe, so this runner's products carry the same window evidence everything else
    does and a busy-box question can be answered from the file rather than from a shell's log."""
    spec = importlib.util.spec_from_file_location("server_window",
                                                  ROOT / "scripts" / "check" / "server_window.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SW = _load_sw()


def _probe(name, fallback):
    """Use the harness's probe when it exists; otherwise fall back ONCE, loudly.

    plain_match_window.py called `_H.llama_pids()`, which the harness does not define, and the
    AttributeError silently dropped it onto a naive `pgrep -f` -- the exact probe the harness had
    already replaced, for the exact bug it documents. So the fallback announces itself.
    """
    try:
        return getattr(_load_harness(), name)
    except AttributeError:
        print(f"[probe] harness has no {name}(); using the local fallback", file=sys.stderr)
        return fallback


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_pids(pids: list[int], timeout_s: float, *, poll=15.0) -> bool:
    """Chain point: block until every pid has exited. Returns False on timeout (never runs anyway)."""
    t0 = time.time()
    while pids:
        pids = [p for p in pids if _pid_alive(p)]
        if not pids:
            return True
        if time.time() - t0 > timeout_s:
            print(f"[chain] still alive after {timeout_s:.0f}s: {pids} -- refusing to start the "
                  f"gate, because a launch into a busy box measures the neighbour", flush=True)
            return False
        print(f"[chain] waiting for {pids} ({(time.time()-t0)/60:.1f} min)", flush=True)
        time.sleep(poll)
    return True


def quiet_sample(port: int, need_mb: float) -> tuple[bool, str]:
    """(is-quiet, reason). Reuses the harness's probes so two guards cannot disagree on 'quiet'."""
    free = float(_probe("vm_free_mb", lambda: 0.0)())
    busy = _probe("foreign_llama", lambda: [])()
    held = _probe("listening", lambda p: False)(port)
    reasons = []
    if held:
        reasons.append(f"port {port} already listened on")
    if busy:
        reasons.append(f"other llama process(es): {busy}")
    if free < need_mb:
        reasons.append(f"reclaimable={free:.0f}MB<{need_mb:.0f}")
    return (not reasons), ("; ".join(reasons) or f"quiet (reclaimable {free:.0f}MB)")


def wait_window(port: int, need_mb: float, timeout_s: float, *, streak_needed=2) -> bool:
    streak = 0
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, why = quiet_sample(port, need_mb)
        if ok:
            streak += 1
            print(f"    quiet sample {streak}/{streak_needed} ({why})", flush=True)
            if streak >= streak_needed:
                return True
        else:
            streak = 0
            print(f"    waiting: {why}", flush=True)
        time.sleep(30)
    return False


def verdict_from_summary(summary: dict) -> dict:
    """Turn a gate summary into a reading, or into a refusal to call it one.

    This is the part the wrapper exists for: `comparable: false` means the run's configuration does
    not match the reference's, so its M1/M2/M3 scores describe a different measurement. They are
    recorded as raw evidence and NOT published as a reading.
    """
    compar = summary.get("comparable")
    m1 = summary.get("m1_numeric_identity")
    m2 = summary.get("m2_decision_agreement")
    m3 = summary.get("m3_topk_set_agreement")
    out = {"comparable": compar, "m1": m1, "m2": m2, "m3": m3,
           "n_compared": summary.get("n_compared"), "ok": summary.get("ok"),
           "engine_digest": summary.get("engine_digest"),
           "ref": summary.get("ref"), "profile": summary.get("profile"),
           "tag": summary.get("tag")}
    if compar is False:
        out["published_reading"] = False
        out["verdict"] = ("NOT A READING: comparable=false -- this run's launch config does not "
                          "match the reference's, so M1/M2/M3 describe a different measurement. "
                          "Raw scores kept as evidence only.")
        out["config_diffs"] = summary.get("config_diffs", [])
    elif compar is True:
        out["published_reading"] = True
        good = (m2 == f"{out['n_compared']}/{out['n_compared']}"
                and m1 == f"{out['n_compared']}/{out['n_compared']}")
        out["verdict"] = ("M1/M2/M3 identical to the reference" if good else
                          f"DIFFERENCE vs the reference (M1 {m1}, M2 {m2}, M3 {m3})")
    else:
        out["published_reading"] = False
        out["verdict"] = "no comparability verdict in the summary (harness error?)"
    return out


SWEEP = ROOT / "scripts" / "check" / "divergence_onset_sweep.py"


def verdict_from_sweep(sw: dict) -> dict:
    """The one claim a sweep product can support: is on/off attributable to the flag?

    The sweep's own verdict says WHERE the divergence starts (a fixed position, or end-tracking).
    Only the NULL CELL says whether the same recipe diverges from ITSELF across launches -- and
    without that, an on/off difference at position p proves only that two launches differ. On this
    box launch-to-launch drift has moved decode by more than any effect being chased, so this is
    the difference between a finding and a guess.
    """
    status = sw.get("null_cell_status")
    cells = sw.get("null_cells") or []
    analysis = sw.get("analysis") or {}
    arms = sw.get("arms") or []
    out = {"null_cell_status": status, "sweep_verdict": analysis.get("state"),
           "onset_positions": analysis.get("firsts"), "null_cells": cells,
           "arms": [a.get("arm") for a in arms],
           "engine_digest": next((a.get("engine_digest") for a in arms if a.get("engine_digest")),
                                 None)}
    if not cells:
        out["onoff_separated"] = None
        out["verdict"] = (f"NOT ESTABLISHED: null_cell_status={status!r} -- the on/off difference is "
                          f"not yet separated from a launch difference")
        return out
    clean = [c for c in cells if c.get("state") == "no-divergence"]
    out["onoff_separated"] = len(clean) == len(cells)
    if out["onoff_separated"]:
        out["verdict"] = ("ATTRIBUTABLE: the same recipe is bit-identical across launches, so the "
                          "on/off onset is the flag's, not the launch's")
    else:
        out["verdict"] = (f"NOT ATTRIBUTABLE: the same recipe diverges from itself across launches "
                          f"({[c.get('state') for c in cells]}) -- launch drift, not the flag")
    return out


def run_job(cmd: list[str], log_path: Path) -> tuple[int, float]:
    """Run one gate/sweep job to completion, capturing its log. Returns (exit, seconds)."""
    t0 = time.time()
    with open(log_path, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return p.returncode, time.time() - t0


def digests_agree(a, b) -> bool | None:
    """Two jobs in one round must report the same binary, or the round compares two engines."""
    if not a or not b:
        return None
    for k, v in a.items():
        if isinstance(v, dict) and k in b and b[k].get("md5") != v.get("md5"):
            return False
    return True


def find_summary(tag: str, after: float) -> Path | None:
    """The gate's own summary path, found by tag first and then by mtime."""
    exact = RESULT_DIR / f"summary_{tag}.json"
    if exact.exists() and exact.stat().st_mtime >= after - 1:
        return exact
    cands = [Path(p) for p in glob.glob(str(RESULT_DIR / "summary_*.json"))
             if Path(p).stat().st_mtime >= after]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def write_state(**kw):
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    prev["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(prev, indent=1, ensure_ascii=False))


def run_once(args) -> dict:
    """One attempt: window -> gate -> record. Returns the record dict."""
    tag = args.tag or f"windowed_{ANCHOR_PROFILE}_{time.strftime('%Y%m%d_%H%M%S')}"
    log = LOG_DIR / f"m123_gate_window_{tag}.log"
    rec = {"tag": tag, "profile": args.profile, "port": args.port, "log": str(log),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "attempt": args.attempt,
           "gate_exit": None, "summary": None, "record": None,
           "window_provenance": SW.provenance(args.port, args.need_mb)}
    cmd = [sys.executable, str(GATE), "--profile", args.profile, "--tag", tag,
           "--port", str(args.port)]
    if args.allow_incomparable:
        cmd.append("--allow-incomparable")
    rec["cmd"] = cmd
    write_state(in_flight={"attempt": args.attempt, "pid": os.getpid(), "tag": tag,
                           "started": rec["started"], "log": str(log)}, blocked_by=None)
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    rec["gate_exit"] = p.returncode
    rec["seconds"] = round(time.time() - t0, 1)
    s = find_summary(tag, t0)
    if s:
        rec["summary"] = str(s)
        rec["record"] = verdict_from_summary(json.loads(s.read_text()))
    write_state(in_flight=None, last_attempt=rec)
    return rec


def cmd_run(args) -> int:
    if args.after_pid:
        pids = [int(x) for x in args.after_pid.replace(",", " ").split()]
        print(f"[chain] waiting for {pids} to exit before looking for a window", flush=True)
        if not wait_for_pids(pids, args.chain_timeout_s):
            write_state(outcome="chain-timeout", blocked_by=f"pids still alive: {pids}")
            return 2
        print("[chain] clear", flush=True)
    for attempt in range(1, args.attempts + 1):
        args.attempt = attempt
        print(f"--- attempt {attempt}/{args.attempts} ---", flush=True)
        ok, why = quiet_sample(args.port, args.need_mb)
        if not ok:
            write_state(blocked_by=why, outcome="waiting", attempts=attempt)
            print(f"[window] {why}", flush=True)
            if not wait_window(args.port, args.need_mb, args.window_timeout_s):
                print("[window] no window inside the timeout; not starting the gate", flush=True)
                write_state(outcome="no-window", blocked_by=why)
                return 2
        rec = run_once(args)
        r = rec["record"] or {}
        print(f"[gate] exit={rec['gate_exit']} summary={rec['summary']}", flush=True)
        if r:
            print(f"[gate] {r.get('verdict')}", flush=True)
            for d in (r.get("config_diffs") or [])[:6]:
                print(f"        {d}", flush=True)
        if rec["summary"] is None:
            print("[gate] no summary written: launch/harness failure -- this is a window problem, "
                  "so it is worth another attempt", flush=True)
            if attempt < args.attempts:
                continue
            write_state(outcome="harness-error", last_attempt=rec)
            return 2
        if r.get("comparable") is False:
            # Terminal by design: retrying cannot change a config mismatch, and the scores are
            # recorded but never published as a reading.
            write_state(outcome="not-comparable", last_attempt=rec)
            return 2
        write_state(outcome="done", last_attempt=rec)
        return 0 if r.get("published_reading") else 1
    return 2


def cmd_round(args) -> int:
    """One window, two jobs, in order: the null cell first, then the gate.

    Order matters. The null cell is what decides whether an on/off difference means anything; the
    gate needs the same sensible machine state. Running them in separate windows is how the two
    numbers end up describing two different machines.
    """
    if args.after_pid:
        pids = [int(x) for x in args.after_pid.replace(",", " ").split()]
        print(f"[chain] waiting for {pids} to exit before looking for a window", flush=True)
        if not wait_for_pids(pids, args.chain_timeout_s):
            write_state(outcome="chain-timeout", blocked_by=f"pids still alive: {pids}")
            return 2
        print("[chain] clear", flush=True)

    ok, why = quiet_sample(args.port, args.need_mb)
    if not ok:
        print(f"[window] {why}", flush=True)
        write_state(blocked_by=why, outcome="waiting")
        if not wait_window(args.port, args.need_mb, args.window_timeout_s):
            write_state(outcome="no-window", blocked_by=why)
            print("[window] none inside the timeout; nothing was run", flush=True)
            return 2
    else:
        print(f"[window] {why}", flush=True)

    # Log the sample that let the launch proceed, so the round's products can answer "was this
    # taken on a busy box?" from inside the file (server_window.provenance reads this log).
    SW.record(args.port, args.need_mb, where="round window: cleared before the first launch",
              gated=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rec = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "window": why,
           "arms": args.arms, "profile": args.profile, "jobs": {}}
    write_state(in_flight={"round": stamp, "pid": os.getpid(), "stage": "sweep"}, blocked_by=None)

    # ---- job 1: the null cell (no --wait-window: this runner owns the window decision)
    sw_json = LOG_DIR / f"null_cell_round_{stamp}.json"
    sw_log = LOG_DIR / f"null_cell_round_{stamp}.log"
    rc, secs = run_job([sys.executable, str(SWEEP), "run", "--arms", args.arms,
                        "--port", str(args.port), "--json", str(sw_json)], sw_log)
    j = {"exit": rc, "seconds": round(secs, 1), "log": str(sw_log), "json": str(sw_json)}
    if sw_json.exists():
        try:
            j["record"] = verdict_from_sweep(json.loads(sw_json.read_text()))
        except Exception as exc:
            j["record"] = {"onoff_separated": None, "verdict": f"unreadable product: {exc}"}
    rec["jobs"]["sweep"] = j
    print(f"[sweep] exit={rc} {j.get('record', {}).get('verdict', 'no product written')}",
          flush=True)

    # ---- window re-check: cheap, and it labels the gate half honestly if the box changed hands
    ok2, why2 = SW.record(args.port, args.need_mb, where="round: before the gate job", gated=False)
    rec["window_at_gate"] = why2
    print(f"[window] before gate: {why2}", flush=True)

    # ---- job 2: the gate
    tag = f"chain_{stamp}"
    gate_log = LOG_DIR / f"m123_gate_window_{tag}.log"
    write_state(in_flight={"round": stamp, "pid": os.getpid(), "stage": "gate"}, blocked_by=None)
    t0 = time.time()
    rc2, secs2 = run_job([sys.executable, str(GATE), "--profile", args.profile, "--tag", tag,
                          "--port", str(args.port)], gate_log)
    gj = {"exit": rc2, "seconds": round(secs2, 1), "log": str(gate_log), "tag": tag}
    s = find_summary(tag, t0)
    if s:
        gj["summary"] = str(s)
        gj["record"] = verdict_from_summary(json.loads(s.read_text()))
    rec["jobs"]["gate"] = gj
    r = gj.get("record") or {}
    print(f"[gate] exit={rc2} {r.get('verdict', 'no summary written (launch/harness failure)')}",
          flush=True)

    # ---- the cross-job check that makes "same round" mean something
    rec["same_engine"] = digests_agree((rec["jobs"]["sweep"].get("record") or {}).get("engine_digest"),
                                       r.get("engine_digest"))
    if rec["same_engine"] is False:
        print("[round] the two jobs report DIFFERENT engine digests: this round compares two "
              "binarys, so neither number carries the other's authority", flush=True)
    rec["window_provenance"] = SW.provenance(args.port, args.need_mb)
    write_state(in_flight=None, outcome="done", last_round=rec)
    print(f"\nround record -> {STATE}", flush=True)
    print(f"[window] provenance class: {rec['window_provenance']['class']} "
          f"({rec['window_provenance']['why']})", flush=True)
    return 0 if r.get("published_reading") else 1


def cmd_status() -> int:
    if not STATE.exists():
        print("no state file: the guard has never run")
        return 5
    d = json.loads(STATE.read_text())
    infl = d.get("in_flight")
    state = "idle"
    if infl:
        alive = _pid_alive(int(infl.get("pid", -1)))
        state = "running" if alive else "stale-running"
    elif d.get("outcome") in ("done", "not-comparable", "harness-error", "no-window", "chain-timeout"):
        state = "done"
    elif infl is None and d.get("outcome") == "waiting":
        state = "waiting"
    print(f"state: {state}")
    print(f"  outcome={d.get('outcome')} blocked_by={d.get('blocked_by')} updated={d.get('updated')}")
    if infl:
        print(f"  in-flight: attempt {infl.get('attempt')} pid {infl.get('pid')} "
              f"driver_alive={_pid_alive(int(infl.get('pid', -1)))}")
    if d.get("last_round"):
        rd = d["last_round"]
        sw = (rd["jobs"].get("sweep") or {}).get("record") or {}
        gt = (rd["jobs"].get("gate") or {}).get("record") or {}
        print(f"  last round: started={rd.get('started')} arms={rd.get('arms')} "
              f"same_engine={rd.get('same_engine')}")
        print(f"    window at gate: {rd.get('window_at_gate')}")
        print(f"    null cell: {sw.get('verdict')}")
        print(f"    M1={gt.get('m1')} M2={gt.get('m2')} M3={gt.get('m3')} "
              f"comparable={gt.get('comparable')} published_reading={gt.get('published_reading')}")
        if gt.get("verdict"):
            print(f"    {gt['verdict']}")
    la = d.get("last_attempt") or {}
    if la and not d.get("last_round"):
        r = la.get("record") or {}
        print(f"  last: exit={la.get('gate_exit')} summary={la.get('summary')}")
        print(f"        M1={r.get('m1')} M2={r.get('m2')} M3={r.get('m3')} "
              f"comparable={r.get('comparable')} published_reading={r.get('published_reading')}")
        if r.get("verdict"):
            print(f"        {r['verdict']}")
    return {"running": 0, "stale-running": 3, "done": 4, "waiting": 5, "idle": 5}[state]


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    print("a void comparison must never be published as a reading")
    void = {"comparable": False, "m1_numeric_identity": "1/9", "m2_decision_agreement": "9/9",
            "m3_topk_set_agreement": "4/9", "n_compared": 9,
            "config_diffs": ["CGCENV.CTX: ref='8192'  now='4096'"]}
    r = verdict_from_summary(void)
    expect("raw scores are kept as evidence", (r["m1"], r["m2"], r["m3"]), ("1/9", "9/9", "4/9"))
    expect("published_reading is False", r["published_reading"], False)
    expect("verdict says NOT A READING", r["verdict"].startswith("NOT A READING"), True)
    expect("config diffs travel with it", r["config_diffs"][0].startswith("CGCENV.CTX"), True)

    good = {"comparable": True, "m1_numeric_identity": "9/9", "m2_decision_agreement": "9/9",
            "m3_topk_set_agreement": "9/9", "n_compared": 9, "ok": True,
            "engine_digest": {"libllama.0.0.279.dylib": {"md5": "aab787412e572550"}}}
    r = verdict_from_summary(good)
    expect("comparable run is published", r["published_reading"], True)
    expect("engine digest travels with the reading",
           r["engine_digest"]["libllama.0.0.279.dylib"]["md5"], "aab787412e572550")

    diff = dict(good, m1_numeric_identity="7/9")
    expect("a comparable run WITH a difference is still a reading", 
           verdict_from_summary(diff)["published_reading"], True)
    expect("and its verdict names the difference",
           verdict_from_summary(diff)["verdict"].startswith("DIFFERENCE"), True)

    expect("a summary with no comparability field is not published",
           verdict_from_summary({})["published_reading"], False)

    print("\nthe window gate must refuse a busy box")
    import types
    global _probe
    saved = _probe
    def fake(name, fallback=None):
        vals = {"vm_free_mb": lambda: 500.0, "foreign_llama": lambda: [999],
                "listening": lambda p: True}
        return vals[name]
    _probe = fake
    ok, why = quiet_sample(8080, 8000.0)
    expect("busy box is not quiet", ok, False)
    expect("all three reasons are reported", why.count(";"), 2)
    _probe = saved

    print("\nthe chain point must not start into a live pid")
    expect("our own pid is alive", _pid_alive(os.getpid()), True)
    expect("an unused high pid is not", _pid_alive(999999), False)
    expect("chain times out instead of running anyway",
           wait_for_pids([os.getpid()], 0.1, poll=0.05), False)

    print("\nthe round must derive on/off attribution from the NULL CELL, not from the sweep verdict")
    base = {"analysis": {"state": "fixed-onset", "firsts": [None, 95, 95, 95]},
            "arms": [{"arm": "on", "engine_digest": {"libllama.0.0.279.dylib": {"md5": "aab"}}},
                     {"arm": "off"}, {"arm": "off"}]}
    v = verdict_from_sweep(dict(base, null_cell_status="requested-but-arms-failed", null_cells=[]))
    expect("arms failed -> attribution NOT established", v["onoff_separated"], None)
    expect("and the onset positions still travel", v["onset_positions"], [None, 95, 95, 95])
    v = verdict_from_sweep(dict(base, null_cell_status="compared",
                                null_cells=[{"mtp": "off", "state": "no-divergence"}]))
    expect("a clean null cell makes the onset the flag's", v["onoff_separated"], True)
    expect("and says ATTRIBUTABLE", v["verdict"].startswith("ATTRIBUTABLE"), True)
    v = verdict_from_sweep(dict(base, null_cell_status="compared",
                                null_cells=[{"mtp": "off", "state": "fixed-onset"}]))
    expect("a diverging null cell kills the attribution", v["onoff_separated"], False)
    expect("and says NOT ATTRIBUTABLE", v["verdict"].startswith("NOT ATTRIBUTABLE"), True)
    expect("a fixed-onset sweep verdict alone never establishes attribution",
           verdict_from_sweep(dict(base, null_cell_status="not-requested"))["onoff_separated"], None)

    print("\ntwo jobs in one round must be the same binary")
    a = {"libllama.0.0.279.dylib": {"md5": "aab"}}
    expect("same digest", digests_agree(a, {"libllama.0.0.279.dylib": {"md5": "aab"}}), True)
    expect("different digest is caught",
           digests_agree(a, {"libllama.0.0.279.dylib": {"md5": "zzz"}}), False)
    expect("missing digest is not a pass", digests_agree(a, None), None)

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- a void comparison is recorded but never published as a reading")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Both spellings work: the Usage block advertises `--status`, and a doc that advertises a flag
    # the parser rejects is the same class of defect as a message that does not match its state.
    ap.add_argument("--status", action="store_true", help="print the guard's state and exit")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--profile", default=ANCHOR_PROFILE)
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--tag", default="")
    r.add_argument("--need-mb", type=float, default=8000.0)
    r.add_argument("--window-timeout-s", type=float, default=3600.0)
    r.add_argument("--chain-timeout-s", type=float, default=7200.0)
    r.add_argument("--after-pid", default="", help="wait for these pids to exit first (the chain)")
    r.add_argument("--attempts", type=int, default=2,
                   help="a launch that never became healthy is a window problem, not a verdict, so "
                        "it is worth another attempt; a config mismatch is terminal")
    r.add_argument("--allow-incomparable", action="store_true")
    rd = sub.add_parser("round", help="null cell THEN the gate, in one window")
    rd.add_argument("--profile", default=ANCHOR_PROFILE)
    rd.add_argument("--port", type=int, default=8080)
    rd.add_argument("--arms", default="1,0,0",
                    help="MTP settings for the null-cell sweep; a repeated setting is what makes "
                         "the launch-drift control exist")
    rd.add_argument("--need-mb", type=float, default=8000.0)
    rd.add_argument("--window-timeout-s", type=float, default=3600.0)
    rd.add_argument("--chain-timeout-s", type=float, default=7200.0)
    rd.add_argument("--after-pid", default="")
    args = ap.parse_args(argv)
    if getattr(args, "status", False) or args.cmd == "status":
        return cmd_status()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "round":
        return cmd_round(args)
    if args.cmd != "run":
        ap.error("give a command: run | round | status | selftest (--status also works)")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
