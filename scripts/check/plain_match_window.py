#!/usr/bin/env python3
"""Run `plain_match_ab` the moment the box is actually free, and requeue if it is not.

WHY THIS EXISTS
---------------
`plain_match` is the one measurement that decides whether the MTP accept ceiling is a property of
the draft head or of a broken verify path (see `plain_match_ab.py`'s own decision rule). It needs a
13 GB model plus an 8 GiB pool on a 16 GB box, so a run that starts while another session holds a
server does not merely get slower: it lands in the documented bimodal regime (measured 2.5x swings,
`DECODE_STEP_BUDGET_2026-09-19.md` Sec.13/15) and the resulting number cannot be attributed to either
arm. Waiting is therefore part of the measurement, not a convenience.

THE GATE (all three must hold, for QUIET_SAMPLES consecutive samples)
  * nothing listens on 8080 (the launcher's production port)
  * no llama process of any kind is running (`llama-server` / `llama-bench`)
  * reclaimable memory >= NEED_MB

Reclaimable, not "free": on this box `Pages free` sits near zero by design while several GB of file
cache is available for reuse -- a gate on `Pages free` blocks forever (`decode_window_harness.py:155`
records the same lesson).

INTERRUPTION IS NOT FAILURE
  * rc == 0                          -> done
  * rc > 0 / killed by a signal      -> the window was lost mid-run: requeue and wait again
  * wall clock exceeded per attempt  -> requeue (the gate re-checks the reasons before re-launching)
State (attempts, outcomes, the last blocking reason) is persisted, because a guard that blocks for an
hour must be answerable from its own files rather than from a shell's scrollback.

"AM I RUNNING RIGHT NOW" IS NOT ANSWERABLE BY THE FILE ALONE
A run leaves three different states that a file written only at the END cannot tell apart: waiting,
running, and dead-with-`running`-left-written. The first version of this guard wrote only on the way
out, so while a 3-minute attempt was in flight the file still said `blocked_by: llama running: [...]`
-- a past state presented as the present one. Fixing that needs two things, and neither is optional:

  1. write the record at the SPAWN instant (pid + log + when), and clear it when the attempt ends;
  2. when reading, combine the record with a liveness probe -- `--status` reports `stale-running`
     (pid gone, or alive but not our driver) rather than echoing the record as `running`. A pid is
     not a run: a killed guard, a killed driver and a recycled pid all leave the same bytes on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Seam for the spawn-ordering test: `run_driver` must record the running state BEFORE it waits on
# the child, and that ordering is the whole point of the change, so it gets a test. Kept as a
# module attribute rather than patching subprocess globally.
_POPEN = subprocess.Popen

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent / "decode_window_harness.py"
DRIVER = Path(__file__).resolve().parent / "plain_match_ab.py"
STATE = ROOT / "Backup" / "phase_decomp" / "plain_match_window_state.json"

PORT = 8080
NEED_MB = 8000.0          # the run wants 13 GB model + 8 GiB pool; below this it over-commits
QUIET_SAMPLES = 2         # consecutive clean samples before launching (one clean sample can be luck)
POLL_S = 30.0
WINDOW_WAIT_S = 3600.0    # how long one wait may take before we give up and report
ATTEMPT_TIMEOUT_S = 3600.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- probes
def _load_harness():
    """Reuse the window probes rather than re-deriving them: two gates that disagree about what
    'quiet' means is exactly how one of them ends up measuring the other session's neighbour."""
    try:
        spec = importlib.util.spec_from_file_location("cgc_window_harness", HARNESS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        return mod
    except Exception as exc:                   # pragma: no cover - fallback path
        log(f"harness probes unavailable ({exc}); using local implementations")
        return None


_H = _load_harness()


def _vm_pages(*names) -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        for n in names:
            if line.strip().startswith(n):
                digits = "".join(c for c in line.split(":")[1] if c.isdigit())
                got[n] = int(digits or 0)
    return sum(got.get(n, 0) for n in names)


def _free_mb() -> float:
    if _H is not None:
        return float(_H.vm_free_mb())
    pages = _vm_pages("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
    return pages * 16384 / 1e6


def _listening(port: int):
    if _H is not None:
        return _H.listening(port)
    p = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                       capture_output=True, text=True)
    return [int(x) for x in p.stdout.split() if x.strip().isdigit()]


# Classify by EXECUTABLE, never by any argv token. `pgrep -f llama-server` matches the polling
# shell whose command line merely *contains* the name; the harness records the same bug live
# (`decode_window_harness.foreign_llama`'s docstring: "pid 60206 blocked a launch while free
# memory was already healthy"), and this file carried a second copy of it -- see _llama_pids.
LLAMA_EXES = {"llama-server", "llama-bench", "llama-cli", "llama-simple", "llama-run",
              "llama-perplexity", "llama-quantize"}


def classify_llama_ps(ps_text: str, self_pid: int) -> list[int]:
    """Pure: which pids in `ps -Ao pid=,args=` are actually llama binaries.

    Separated from the probe so the exact false positive this box produced can be asserted in a
    test rather than described in a comment.
    """
    hits = []
    for line in ps_text.splitlines():
        pid, _, args = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) == self_pid:
            continue
        toks = args.split()
        i = 0
        # see through `env VAR=VAL cmd` and leading switches, as the harness's argv_exe does
        while i < len(toks) and ("=" in toks[i] or toks[i].startswith("-")):
            i += 1
        exe = os.path.basename(toks[i]) if i < len(toks) else ""
        if exe in LLAMA_EXES:
            hits.append(int(pid))
    return hits


def _llama_pids():
    """The harness's probe if it is importable, else an equivalent one.

    BUG THIS FIXES: this used to call `_H.llama_pids()`, which the harness does not define (it is
    `foreign_llama`). The AttributeError was swallowed by `except AttributeError: pass`, so the
    guard silently ran its own `pgrep -f "llama-server|llama-bench"` fallback -- the naive version
    the harness had already replaced -- for every check it ever made. A silently-swallowed
    AttributeError is what made a wrong probe look like the intended one.
    """
    if _H is not None:
        for name in ("foreign_llama", "llama_pids"):
            fn = getattr(_H, name, None)
            if fn is None:
                continue
            try:
                got = fn()
            except Exception:
                continue
            out = []
            for item in got:                      # [pid, ...] or [(pid, exe), ...]
                out.append(int(item[0] if isinstance(item, (list, tuple)) else item))
            return out
    p = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True)
    return classify_llama_ps(p.stdout, os.getpid())


# --------------------------------------------------------------------- the gate
def gate_reasons(port: int = PORT, need_mb: float = NEED_MB, probes: dict | None = None):
    """Why the box is not usable right now. Empty list == usable."""
    pr = probes or {}
    listening = pr.get("listening", _listening)
    llama = pr.get("llama_pids", _llama_pids)
    free = pr.get("free_mb", _free_mb)

    reasons = []
    held = listening(port)
    if held:
        reasons.append(f"port {port} held by {held}")
    pids = llama()
    if pids:
        reasons.append(f"llama running: {pids[:3]}")
    mb = free()
    if mb < need_mb:
        reasons.append(f"reclaimable={mb:.0f}MB<{need_mb:.0f}")
    return reasons, mb


def window_ok(port: int = PORT, need_mb: float = NEED_MB, probes: dict | None = None):
    reasons, mb = gate_reasons(port, need_mb, probes)
    return (not reasons), "; ".join(reasons), mb


# --------------------------------------------------------------------- requeue policy
def requeue_decision(rc: int, attempt: int, max_attempts: int) -> str:
    """What to do after one attempt. Pure, so the policy can be tested without a GPU.

    A killed child (rc < 0) and a failed run (rc > 0) are the SAME case here: the window was not
    ours. Neither is a reason to stop, which is the whole point of the guard -- the alternative is
    a human re-launching a 6-minute run whenever a neighbour session starts up.
    """
    if rc == 0:
        return "done"
    if attempt >= max_attempts:
        return "give-up"
    return "requeue"


# --------------------------------------------------------------------- running
def run_driver(args, log_path: Path, on_spawn=None) -> int:
    """`on_spawn(pid, cmd)` fires between Popen and wait -- i.e. while the run is in flight. It is a
    callback rather than a state write here so that all state mutation stays in one place (and so
    the spawn ordering can be asserted without a GPU)."""
    cmd = [sys.executable, str(DRIVER),
           "--profile", args.profile, "--pool-gb", str(args.pool_gb),
           "--n-predict", str(args.n_predict), "--reps", str(args.reps),
           "--order", args.order, "--port", str(args.port),
           "--json", args.json]
    log(f"    launch: {' '.join(cmd)}")
    with open(log_path, "w") as fh:
        try:
            p = _POPEN(cmd, stdout=fh, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
        except Exception as exc:
            log(f"    launch failed: {exc}")
            return 1
        log(f"    driver pid {p.pid} -> {log_path}")
        if on_spawn is not None:
            try:
                on_spawn(p.pid, cmd)
            except Exception as exc:      # the record must never take the run down
                log(f"    WARNING: could not record the running state: {exc}")
        try:
            return p.wait(timeout=args.attempt_timeout)
        except subprocess.TimeoutExpired:
            log(f"    attempt exceeded {args.attempt_timeout:.0f}s; terminating OUR pid only")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
            time.sleep(5)
            return -signal.SIGTERM


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"attempts": [], "blocked_by": None, "done": False}


def save_state(st: dict) -> None:
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)


# --------------------------------------------------------------------- "am I running"
def _pid_alive(pid) -> bool:
    """Signal 0: existence probe. PermissionError means it exists and is not ours, which is still
    'alive' for our purposes (we only claim our own pid anyway)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, OSError):
        return True


def _pid_cmdline(pid) -> str:
    try:
        p = subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))],
                           capture_output=True, text=True)
        return p.stdout.strip()
    except Exception:
        return ""


def set_running(st: dict, attempt: int, driver_pid: int, log_path, cmd) -> None:
    """Called at the spawn instant, BEFORE waiting on the child. That ordering is the point:
    between spawn and exit there is a multi-minute window in which the only honest description of
    the guard is 'running', and it must already be on disk when that window opens."""
    st["running"] = {
        "attempt": attempt,
        "driver_pid": int(driver_pid),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": str(log_path),
        "expect_in_cmdline": str(DRIVER),
        "launch_cmd": " ".join(str(c) for c in cmd),
    }
    st["blocked_by"] = None      # we are not blocked any more; the same defect, other direction
    save_state(st)


def clear_running(st: dict) -> None:
    st["running"] = None
    save_state(st)


def status_of(st: dict, alive=None, cmdline=None) -> tuple[str, list[str]]:
    """The current state, from the record PLUS a probe. Injectable probes so it is testable.

    Returns (state, evidence). `stale-running` is the verdict that matters: it is the one a bare
    state file would have reported as healthy.
    """
    alive = alive or _pid_alive
    cmdline = cmdline or _pid_cmdline
    ev: list[str] = []

    g = st.get("guard") or {}
    if g.get("pid"):
        g_alive = bool(alive(g["pid"]))
        ev.append(f"guard pid {g['pid']} alive={g_alive}"
                  + ("" if g_alive else "  <-- the guard itself is gone; nothing will advance this file"))

    r = st.get("running")
    if r:
        pid = r.get("driver_pid")
        is_alive = bool(pid) and bool(alive(pid))
        cmd = cmdline(pid) if is_alive else ""
        want = r.get("expect_in_cmdline") or ""
        matches = bool(want) and want in cmd
        ev.append(f"record: attempt {r.get('attempt')}, driver_pid {pid}, started {r.get('started')}")
        ev.append(f"  pid alive={is_alive}  cmdline is our driver={matches}")
        if is_alive and matches:
            return "running", ev
        ev.append("  the record says running but the process is gone, or is something else "
                  "(killed guard / killed driver / recycled pid) -- the file is describing a past "
                  "state, so it must not be read as 'running'")
        return "stale-running", ev

    if st.get("done"):
        return "done", ev + [f"outcome={st.get('outcome')} "
                              f"attempts={len(st.get('attempts') or [])}"]
    if st.get("outcome"):
        return str(st["outcome"]), ev + [f"outcome={st['outcome']}"]
    if st.get("blocked_by"):
        return "waiting", ev + [f"blocked_by: {st['blocked_by']}"]
    if st.get("started"):
        return "idle", ev + ["started but has not recorded a wait reason yet"]
    return "unknown", ev + ["no state file, or it has no usable fields"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="gate port (0 -> 8080, the production port)")
    ap.add_argument("--driver-port", type=int, default=8097, help="the driver's own probe port")
    ap.add_argument("--pool-gb", type=float, default=8.0)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--order", default="1,0,1,0")
    ap.add_argument("--profile", default="prod25")
    ap.add_argument("--need-mb", type=float, default=NEED_MB)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--window-wait-s", type=float, default=WINDOW_WAIT_S)
    ap.add_argument("--attempt-timeout", type=float, default=ATTEMPT_TIMEOUT_S)
    ap.add_argument("--json", default=str(ROOT / "Backup" / "phase_decomp" / "plain_match.json"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="answer 'am I running right now' from the state file PLUS a liveness probe. "
                         "Exit code doubles as a predicate: 0=running, 3=stale-running (record set but "
                         "the process is gone/not ours), 4=finished (done/give-up/no-window), "
                         "5=waiting/idle/unknown. 3 is deliberately NOT 0 and NOT 4: a stale record "
                         "must not be readable as either 'healthy' or 'finished'")
    args = ap.parse_args()
    args.port = args.port or PORT
    # The driver's probe port must differ from the gate port: the gate watches the launcher's
    # production port, and passing 8080 to the driver would make it poll its neighbour's server.
    if args.driver_port == args.port:
        args.driver_port = 8097

    if args.selftest:
        return selftest()

    if args.status:
        st = load_state()
        state, evidence = status_of(st)
        print(f"state: {state}")
        for line in evidence:
            print(f"  {line}")
        if st.get("running"):
            print(f"  log: {st['running'].get('log')}")
        elif st.get("attempts"):
            print(f"  last attempt: {st['attempts'][-1]}")
        return {"running": 0, "stale-running": 3,
                "done": 4, "give-up": 4, "no-window": 4}.get(state, 5)

    st = load_state()
    st["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st["gate_port"] = args.port
    st["guard"] = {"pid": os.getpid(), "started": st["started"],
                   "argv": sys.argv[:8], "need_mb": args.need_mb}
    st.setdefault("running", None)
    attempt = len(st["attempts"])
    while True:
        # ---- wait
        t0 = time.time()
        streak = 0
        while time.time() - t0 < args.window_wait_s:
            ok, why, mb = window_ok(args.port, args.need_mb)
            if ok:
                streak += 1
                log(f"    quiet sample {streak}/{QUIET_SAMPLES} (reclaimable {mb:.0f}MB)")
                if streak >= QUIET_SAMPLES:
                    break
            else:
                if streak:
                    log(f"    window lost before launch: {why}")
                streak = 0
                st["blocked_by"] = why
                save_state(st)                 # answerable from the file, not from scrollback
                log(f"    waiting: {why}")
            time.sleep(POLL_S)
        else:
            log(f"no window in {args.window_wait_s:.0f}s; last reason: {st.get('blocked_by')}")
            st["outcome"] = "no-window"
            save_state(st)
            return 2

        # ---- run
        attempt += 1
        log_path = ROOT / "Backup" / "cgc_logs" / f"plain_match_window_attempt{attempt}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        rec = {"attempt": attempt, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "gate_reclaimable_mb": round(mb, 1), "log": str(log_path)}
        rc = run_driver(args, log_path,
                        on_spawn=lambda pid, cmd: set_running(st, attempt, pid, log_path, cmd))
        rec["rc"] = rc
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        clear_running(st)          # the attempt is over; leaving `running` set would be the same
                                   # past-as-present defect, mirrored
        st["attempts"].append(rec)

        verdict = requeue_decision(rc, attempt, args.max_attempts)
        log(f"    attempt {attempt}: rc={rc} -> {verdict}")
        # A failed attempt is retried because the window was never ours; a failed attempt is ALSO
        # never overwritten, so the log of the arm that died mid-run stays on disk next to the one
        # that produced the number.
        st["outcome"] = verdict
        if verdict == "done":
            st["done"] = True
            save_state(st)
            return 0
        if verdict == "give-up":
            save_state(st)
            return 1
        save_state(st)
        time.sleep(POLL_S)


# --------------------------------------------------------------------- tests
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(f"{name}: got {got!r} want {want!r}")

    def probes(port_pids=(), llama=(), mb=9000.0):
        return {"listening": lambda p: list(port_pids),
                "llama_pids": lambda: list(llama),
                "free_mb": lambda: mb}

    # 1. the three gates, each in isolation and together
    ok, why, mb = window_ok(probes=probes())
    expect("quiet box -> usable", ok, True)
    ok, why, _ = window_ok(probes=probes(port_pids=(27141,)))
    expect("8080 held -> refuse, and name the pid", (ok, "27141" in why), (False, True))
    ok, why, _ = window_ok(probes=probes(llama=(123,)))
    expect("llama running -> refuse", (ok, "llama running" in why), (False, True))
    ok, why, _ = window_ok(probes=probes(mb=1000.0))
    expect("low reclaimable -> refuse, and quote it", (ok, "reclaimable=1000MB<8000" in why), (False, True))
    ok, why, _ = window_ok(probes=probes(port_pids=(1,), llama=(2,), mb=10.0))
    expect("all three broken -> all three named", why.count(";"), 2)

    # 2. free memory is NOT the gate: a free-but-cached box must pass, or the guard blocks forever
    #    (measured on this box: free 68MB while 6.1GB was reclaimable)
    ok, why, mb = window_ok(probes=probes(mb=6100.0), need_mb=6000.0)
    expect("cached-but-reclaimable box passes", ok, True)

    # 3. the requeue policy
    expect("success -> done", requeue_decision(0, 1, 5), "done")
    expect("failed run -> requeue", requeue_decision(1, 1, 5), "requeue")
    expect("killed mid-run -> requeue, not failure", requeue_decision(-15, 2, 5), "requeue")
    expect("attempts exhausted -> give up", requeue_decision(1, 5, 5), "give-up")
    expect("success wins even at the last attempt", requeue_decision(0, 5, 5), "done")

    # 4. state round trip, so a restart resumes instead of re-running attempt 1
    global STATE
    keep = STATE
    try:
        STATE = Path("/tmp/plain_match_window_selftest.json")
        if STATE.exists():
            STATE.unlink()
        st = load_state()
        st["attempts"].append({"attempt": 1, "rc": 1})
        save_state(st)
        st2 = load_state()
        expect("state survives a restart", len(st2["attempts"]), 1)
        # The intent is "never claim success without one", not "the key is absent" -- a fresh
        # state defines done=False, so asserting None here tested my wording instead of the rule.
        expect("a state with no successful attempt must not claim success",
               bool(load_state().get("done")), False)
    finally:
        STATE = keep
        if Path("/tmp/plain_match_window_selftest.json").exists():
            Path("/tmp/plain_match_window_selftest.json").unlink()

    # 5. THE ORDERING CLAIM. `on_spawn` must fire between Popen and wait -- otherwise the record
    #    lands after the run, which is exactly the defect being fixed. A fake Popen observes it.
    global _POPEN
    keep_popen = _POPEN
    order = {}

    class _FakePopen:
        pid = 4242

        def __init__(self, *a, **k):
            pass

        def wait(self, timeout=None):
            order["spawned_by_wait_time"] = order.get("spawned")
            return 0

    lp = Path("/tmp/pmw_spawn_order.log")
    try:
        _POPEN = _FakePopen
        rc = run_driver(argparse.Namespace(profile="x", pool_gb=1.0, n_predict=1, reps=1,
                                           order="1", port=1, json="/tmp/x.json",
                                           attempt_timeout=1.0),
                        lp, on_spawn=lambda pid, cmd: order.__setitem__("spawned", pid))
        expect("on_spawn fires before wait", order.get("spawned_by_wait_time"), 4242)
        expect("run_driver returns the child rc", rc, 0)
    finally:
        _POPEN = keep_popen
        if lp.exists():
            lp.unlink()

    # 6. set_running / clear_running round trip through the FILE, and the past-as-present fix:
    #    the record must clear `blocked_by`, or the file reports a wait while a run is in flight.
    #    (`STATE` is already declared global in scenario 4; a second declaration here is a
    #    SyntaxError, so it is deliberately not repeated.)
    keep2 = STATE
    try:
        STATE = Path("/tmp/plain_match_window_selftest2.json")
        if STATE.exists():
            STATE.unlink()
        s = load_state()
        s["blocked_by"] = "llama running: [34879]"
        set_running(s, attempt=1, driver_pid=4242, log_path=Path("/tmp/l"), cmd=["py", "d.py"])
        on_disk = load_state()
        expect("running is on disk at spawn", on_disk["running"]["driver_pid"], 4242)
        expect("running clears the stale blocked_by", on_disk["blocked_by"], None)
        expect("running carries the log path", on_disk["running"]["log"], "/tmp/l")
        clear_running(s)
        expect("cleared after the attempt", load_state()["running"], None)
    finally:
        STATE = keep2
        if Path("/tmp/plain_match_window_selftest2.json").exists():
            Path("/tmp/plain_match_window_selftest2.json").unlink()

    # 7. status_of with INJECTED probes -- deterministic on any host, and it is the reader side of
    #    the fix: the file alone may not be read as 'running'.
    rec = {"attempt": 1, "driver_pid": 99999, "started": "t", "log": "/tmp/l",
           "expect_in_cmdline": str(DRIVER)}
    yes = lambda p: True
    no = lambda p: False
    our_cmd = lambda p: f"python3 {DRIVER} --profile prod25"
    other_cmd = lambda p: "/usr/bin/something-else"

    st, ev = status_of({"running": dict(rec)}, alive=yes, cmdline=our_cmd)
    expect("alive + our cmdline -> running", st, "running")

    st, ev = status_of({"running": dict(rec)}, alive=no, cmdline=our_cmd)
    expect("pid gone -> stale-running, not running", st, "stale-running")
    expect("stale-running explains itself", any("past " in e for e in ev), True)

    st, ev = status_of({"running": dict(rec)}, alive=yes, cmdline=other_cmd)
    expect("recycled pid -> stale-running", st, "stale-running")

    # A live run must win over a stale outcome field: `done` from the previous attempt must not
    # mask a run that is in flight right now.
    st, _ = status_of({"running": dict(rec), "done": True}, alive=yes, cmdline=our_cmd)
    expect("a live run outranks an old done flag", st, "running")

    st, _ = status_of({"running": None, "done": True, "outcome": "done", "attempts": [{}]},
                      alive=no, cmdline=other_cmd)
    expect("finished -> done", st, "done")
    st, _ = status_of({"running": None, "blocked_by": "reclaimable=1000MB<8000"})
    expect("blocked -> waiting, quoting the reason", st, "waiting")
    st, ev = status_of({})
    expect("empty -> unknown", st, "unknown")

    # 7b. the exit code is a predicate, and `stale` must be its own value: a caller that reads
    #     "not running" as "finished" is the failure mode this whole change exists to prevent.
    codes = {"running": 0, "stale-running": 3, "done": 4, "give-up": 4, "no-window": 4}
    expect("stale is neither healthy nor finished",
           (codes["stale-running"] != codes["running"], codes["stale-running"] != codes["done"]),
           (True, True))
    expect("waiting/unknown share a distinct code",
           codes.get("waiting", 5) not in (0, 3, 4), True)
    st, _ = status_of({"running": None, "started": "t"})
    expect("started but not waiting yet -> idle", st, "idle")

    # 7c. THE PROBE REGRESSION. The guard must use the harness's executable-based probe when it is
    #     importable; a silently swallowed AttributeError used to send it down its own naive
    #     `pgrep -f` path instead, which matches polling shells.
    global _H
    keep_h = _H
    try:
        class _StubH:
            @staticmethod
            def foreign_llama():
                return [(7, "llama-server"), (9, "llama-bench")]

            @staticmethod
            def llama_pids():
                raise AssertionError("should prefer foreign_llama")

        _H = _StubH
        expect("harness probe is used when importable", _llama_pids(), [7, 9])
        # a bare [pid, ...] shape must also work (two shapes exist in the tree)
        class _StubH2:
            @staticmethod
            def foreign_llama():
                return [7, 9]

        _H = _StubH2
        expect("plain pid-list shape also works", _llama_pids(), [7, 9])
    finally:
        _H = keep_h

    # 7d. the fallback classifier, against the FALSE POSITIVE this box produced: a shell whose
    #     argv merely contains `pgrep -f "build/bin/llama-bench"` is NOT a llama process.
    ps_fake = "\n".join([
        "  111 /bin/zsh -c ... pgrep -f 'build/bin/llama-bench' ...",
        "  222 /bin/bash -c pgrep -f 'llama-server|llama-bench'",
        "  333 /path/build/bin/llama-server -m x.gguf --port 8080",
        "  444 LLAMA_ARG=1 /path/build/bin/llama-bench -m x.gguf",
        "  555 /usr/bin/python3 scripts/check/plain_match_window.py --status",
        "  notapid whatever llama-server",
    ])
    expect("argv-mention shells are NOT llama", classify_llama_ps(ps_fake, 999), [333, 444])
    expect("our own pid is excluded", classify_llama_ps(ps_fake, 333), [444])
    expect("empty ps output -> no hits", classify_llama_ps("", 1), [])
    expect("env-prefixed invocation is seen through", classify_llama_ps(ps_fake, 1), [333, 444])

    # 8. guard liveness is part of the answer: a dead guard must not look like a healthy wait.
    _, ev = status_of({"guard": {"pid": 99999}, "blocked_by": "x"}, alive=no, cmdline=other_cmd)
    expect("dead guard is called out", any("guard itself is gone" in e for e in ev), True)

    # 9. the regression in one line: the EXACT state this box was in while attempt 1 ran --
    #    blocked_by written, run in flight -- must no longer be reportable as `waiting`.
    mid_run = {"blocked_by": "llama running: [34879]", "outcome": None,
               "running": dict(rec), "done": False}
    st, _ = status_of(mid_run, alive=yes, cmdline=our_cmd)
    expect("the past-as-present bug is fixed", st, "running")
    expect("and with the driver dead it is flagged, not echoed",
           status_of(mid_run, alive=no, cmdline=our_cmd)[0], "stale-running")

    print("\nSELFTEST " + ("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
