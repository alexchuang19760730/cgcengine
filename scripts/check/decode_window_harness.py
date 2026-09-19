#!/usr/bin/env python3
"""Two decode lanes, run only in a quiet window, with a judged report instead of a pile of numbers.

    python3 scripts/check/decode_window_harness.py run       # waits for a window, runs, saves JSON
    python3 scripts/check/decode_window_harness.py report    # -> docs/DECODE_WINDOW_<date>.html
    python3 scripts/check/decode_window_harness.py run --analyze-only --only lane_a

Why the harness is shaped this way (each point is a defect measured in this project, not a style
preference):

* **A window, not a launch.** This box is shared with parallel sessions that start and stop llama
  servers on their own schedule. An arm is only valid if it ran inside one uninterrupted window, so
  the harness waits for a window and marks an arm `done` ONLY after all of its requests came back.
  Anything interrupted is re-queued and is never recorded as data.
* **We never signal anything that is not ours.** A window requires that no llama process exists at
  all, and cleanup signals only the pid listening on our own dedicated port (CGC_SERVER_PORT). A
  `pkill -f llama-server` here would kill a parallel session's server -- it did, once.
* **Baselines bracket the test.** Lane A runs b1 / b0 / b1 so the two BITIDENT=1 baselines bound the
  box's drift. An effect smaller than that spread is reported INCONCLUSIVE, because a single-pair
  comparison at this box's noise level (±8% between launches) cannot separate them.
* **Mean, never median x count.** A previous version multiplied a median step by the token count and
  compared it to a mean-derived wall, inventing a 20% gap that did not exist.
* **Cold vs warm is a variable, not noise.** Every lane takes at least two requests per launch; the
  first request of a fresh server is a different regime (8.5-9 t/s vs 13.3 t/s) and mixing them has
  already produced one wrong conclusion.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import signal
import statistics as st
import subprocess
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = int(os.environ.get("PROBE_PORT", "8199"))
OTHER_PORT = int(os.environ.get("PROBE_OTHER_PORT", "8080"))   # observed, never signalled
STATE = os.environ.get("PROBE_STATE", "/tmp/decode_window_harness.json")
PROMPT = ("Explain in a few sentences why disk caches help sequential reads more than random "
          "reads.")
N_PREDICT = int(os.environ.get("PROBE_N_PREDICT", "96"))
FILLER = ("Caching is a trade-off between memory and latency. A cache costs space and buys time, "
          "and the exchange rate depends on the access pattern rather than on the hardware alone. ")
THRASH_PROMPT = FILLER * 60 + "Summarise the above in one sentence."
THRASH_N = 8

QUIET_SAMPLES = 4
SAMPLE_S = 20
MAX_WAIT_S = 90 * 60
PASSES = 4
TOTAL_BUDGET_S = float(os.environ.get("PROBE_BUDGET_S", str(4 * 3600)))
STARTED = time.time()

# An effect has to beat the box's own drift to count. 5% is the margin the two BITIDENT=1 baselines
# have to agree within before the BITIDENT=0 arm may be compared against them.
BASELINE_AGREE_PCT = 5.0

# Residue smaller than this (in ms/token) is treated as run-to-run variation rather than an effect
# to explain, and is not given a concentration shape.
MATERIAL_RESIDUE_PCT = 5.0

# The artefacts the M1/M2/M3 oracle gate last passed on. This box is shared: another session can
# rebuild libllama mid-run, and an arm measured across that rebuild would still be stamped with
# *a* digest -- just not the one any gate passed. So the harness checks instead of stamping, and
# marks such an arm rather than recording it as data. PROBE_ANCHOR=none disables the check for a
# deliberate new-build campaign.
ANCHOR = {
    "llama-server": "054fb22f04a01c5c",
    "libllama.0.dylib": "aab787412e572550",
    "libggml-metal.0.dylib": "1be366306c604669",
    "libllama-server-impl.dylib": "a87bfd2e80a086b4",
}
RESIDUE_CONCENTRATION = 0.50     # top-5 share of the total delta above which it is "concentrated"

# A block whose arms span more than this was not measured inside one regime. Interleaving cancels a
# drift that is slow relative to a single arm; it cannot cancel a flip that happens faster than an arm
# takes. On 2026-09-19 this box flipped 2.5x inside ONE block (337.2 -> 133.6 ms/step) with an
# identical cb (9.8 vs 10.8 ms), i.e. the same pool state -- so the pair members were never in the
# same regime, and no number of extra blocks can repair that.
REGIME_FLIP_PCT = 20.0

# The mul_mat family is chosen from the WEIGHT type. Only two disjoint type lists are ever admitted
# to the small-batch family (`mul_mv_ext`); every other type falls through to mul_mv/mul_mm, where
# the batch size does not change the per-row reduction -- so for those types CGC_MM_BITIDENT cannot
# change anything at all. Transcribed from ggml-metal-ops.cpp:2476-2504 (the `if` guarding
# CGC_MM_TRACE("small-batch")) together with its ne11 ranges. Kept as data rather than prose because
# "BITIDENT costs the verify step its small-batch kernel" is an *eligibility* claim, and eligibility
# is a property of the weights the model ships, not of the knob. The check below reads the real GGUF.
SMALL_BATCH_A = {"F32", "F16", "BF16", "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
                 "Q8_0", "MXFP4", "IQ4_NL"}
SMALL_BATCH_A_M = (2, 8)
SMALL_BATCH_B = {"Q4_K", "Q5_K", "Q6_K", "Q2_K", "Q3_K"}
SMALL_BATCH_B_M = (4, 8)

# ggml type ids -> names, straight from ggml/include/ggml.h. Unknown ids keep their number so the
# table says "type_37" instead of silently dropping a tensor it cannot classify.
GGML_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
                   9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
                   15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
                   20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
                   26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
                   41: "Q1_0", 42: "Q2_0"}

LLAMA_NAMES = {"llama-server", "llama-bench", "llama-cli", "llama-simple",
               "llama-speculative-simple", "llama-perplexity", "llama-batched",
               "llama-embedding", "llama-gguf-split"}

STEPRE = re.compile(r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
                    r"wait=([\d.]+) \(\d+%\) cb=([\d.]+) \(\d+%\) submit=([\d.]+) \(\d+%\) ntok=(\d+)")
ALLRE = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
                   r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")

OUR_PIDS: set[int] = set()
OUR_LOGS: set[str] = set()
LOG_LINES: list[str] = []


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    LOG_LINES.append(line)
    print(line, flush=True)


def budget_left():
    return TOTAL_BUDGET_S - (time.time() - STARTED)


# ----------------------------------------------------------------- observations
def _vm_pages(*names):
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        k, _, v = line.partition(":")
        got[k.strip()] = v.strip().rstrip(".")
    total = 0
    for n in names:
        try:
            total += int(got.get(n, "0"))
        except ValueError:
            pass
    return total


def vm_free_mb():
    """Reclaimable memory, NOT `Pages free`.

    macOS drives `Pages free` to near zero by design, holding the model file in file-backed cache:
    measured live with ZERO llama processes running, free was 68 MB while inactive+purgeable+free was
    ~6.1 GB and 5.6 GB of it was reclaimable file cache. A gate on `Pages free` alone therefore
    blocks forever on an idle machine -- it did, for a whole run. `Pages free + purgeable + inactive`
    is the quantity that actually answers "is there room to load a model".
    """
    return _vm_pages("Pages free", "Pages purgeable", "Pages inactive") * 16384 / 1048576


def listening(port):
    out = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if p.isdigit()]


def argv_exe(args):
    """The executable a process will actually run, seeing through `env VAR=VAL cmd`.

    run_server.sh launches its server as `env CGC_... /path/to/llama-server`, so argv[0] is `env` and
    the executable is the first non-assignment, non-flag token after it.
    """
    toks = args.split()
    i = 0
    if toks and os.path.basename(toks[0]) == "env":
        i = 1
        while i < len(toks) and ("=" in toks[i] or toks[i].startswith("-")):
            i += 1
    return os.path.basename(toks[i]) if i < len(toks) else ""


def foreign_llama():
    """llama processes that are not ours, classified by the EXECUTABLE, not by any argv token.

    The first version matched any token's basename, and a parallel session's polling shell is
    `zsh -c '... pgrep -f "build/bin/llama-bench" ...'` -- its command line *contains* a llama name
    while it is not a llama process at all. Those wrappers held the guard shut after the bench
    itself had exited (seen live: pid 60206 blocked a launch while free memory was already healthy).
    run_server.sh's preflight documents the same class of bug and adopted the same rule: match the
    executable, and a real bench is still caught -- its own child has the llama binary as argv[0].
    """
    out = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) in OUR_PIDS or int(pid) == os.getpid():
            continue
        exe = argv_exe(args)
        if exe in LLAMA_NAMES:
            hits.append((int(pid), exe))
    return hits


def newest_log_age():
    """Seconds since the last FOREIGN server log write.

    OUR OWN logs must be excluded. They are written by the arms we just ran, so counting them makes
    the harness treat its own finished arm as an active neighbour: measured live, every arm was
    followed by ~90s of "waiting: foreign log written Ns ago" that was entirely self-inflicted, and
    the reason printed was simply wrong about who wrote it.
    """
    logs = os.path.join(ROOT, "Backup/cgc_logs")
    best = None
    for f in os.listdir(logs):
        if f.startswith("llama_server_") and f.endswith(".log"):
            try:
                p = os.path.realpath(os.path.join(logs, f))
                if p in OUR_LOGS:
                    continue
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if best is None or mt > best:
                best = mt
    return None if best is None else time.time() - best


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else None


def window_state():
    reasons = []
    fl = foreign_llama()
    if fl:
        reasons.append(f"llama running: {fl[:3]}")
    for p in (PORT, OTHER_PORT):
        held = listening(p)
        if held:
            reasons.append(f"port {p} held by {held}")
    free = vm_free_mb()
    if free < 4000:
        reasons.append(f"reclaimable={free:.0f}MB<4000")
    age = newest_log_age()
    if age is not None and age < 90:
        reasons.append(f"foreign log written {age:.0f}s ago")
    return (not reasons), "; ".join(reasons) or "quiet"


def wait_for_window():
    t0 = time.time()
    streak = 0
    if budget_left() <= 0:
        log("    budget exhausted; not waiting")
        return False
    while time.time() - t0 < min(MAX_WAIT_S, budget_left()):
        ok, why = window_state()
        if ok:
            streak += 1
            log(f"    quiet sample {streak}/{QUIET_SAMPLES}")
            if streak >= QUIET_SAMPLES:
                return True
        else:
            if streak:
                log(f"    window lost before launch: {why}")
            streak = 0
            log(f"    waiting: {why}")
        # Refresh the report DURING the wait. Without this the report is a snapshot of the last
        # state change, so a harness blocked for an hour looks identical to one that just started --
        # "why is there no output?" has to be answerable from the report itself, not from /tmp.
        if LATEST:
            LATEST["blocked_by"] = why
            LATEST["waiting_since"] = time.strftime("%H:%M:%S", time.localtime(t0))
            autoreport(LATEST)
        time.sleep(SAMPLE_S)
    return False


def stop_ours():
    """Signal only our tracked pids, and only while they still hold OUR port."""
    for p in sorted(OUR_PIDS & set(listening(PORT))):
        subprocess.run(["kill", "-TERM", str(p)], capture_output=True)
    for _ in range(25):
        time.sleep(1)
        if not (OUR_PIDS & set(listening(PORT))):
            break
    for p in sorted(OUR_PIDS & set(listening(PORT))):
        subprocess.run(["kill", "-9", str(p)], capture_output=True)
    OUR_PIDS.clear()
    time.sleep(2)


# ---------------------------------------------------------------------- parsing
def parse_range(path, lo, hi):
    with open(path, errors="replace") as fh:
        fh.seek(lo)
        chunk = fh.read(max(hi - lo, 0))
    by_ntok, by_layer = {}, {}
    for line in chunk.splitlines():
        m = STEPRE.search(line)
        if m:
            d = dict(step=int(m.group(1)), layers=int(m.group(3)), total=float(m.group(4)),
                     wait=float(m.group(5)), cb=float(m.group(6)), submit=float(m.group(7)),
                     ntok=int(m.group(8)))
            if d["layers"] >= 40 and d["step"] >= 3:
                by_ntok.setdefault(d["ntok"], []).append(d)
            continue
        m = ALLRE.search(line)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(
                dict(wait=float(m.group(2)), cb=float(m.group(3)), submit=float(m.group(4)),
                     gpu=float(m.group(5)), union=float(m.group(6)), gap=float(m.group(7))))
    steps = {n: {k: st.median(r[k] for r in rows) for k in ("total", "wait", "cb", "submit")}
             for n, rows in by_ntok.items()}
    for n, rows in by_ntok.items():
        steps[n]["n"] = len(rows)
    layers = {L: {k: st.median(r[k] for r in rows) for k in ("wait", "cb", "gpu", "union", "gap")}
              for L, rows in by_layer.items()}
    for L, rows in by_layer.items():
        layers[L]["n"] = len(rows)
    return {"steps": steps, "layers": layers}


def digest():
    out = {}
    for rel in ("llama-server", "libllama.0.dylib", "libggml-metal.0.dylib",
                "libllama-server-impl.dylib"):
        p = os.path.join(ROOT, "src/llama.cpp/build/bin", rel)
        if os.path.exists(p):
            out[rel] = subprocess.run(["md5", "-q", p], capture_output=True,
                                      text=True).stdout.strip()[:16]
    return out


def anchor_ok(dg):
    """Is this the build the M1/M2/M3 gate passed on? Returns (ok, why)."""
    if os.environ.get("PROBE_ANCHOR", "strict") == "none":
        return True, "anchor check disabled (PROBE_ANCHOR=none)"
    differ = {k: f"{dg.get(k)} != {v}" for k, v in ANCHOR.items()
              if k in dg and dg[k] != v}
    if differ:
        return False, f"engine differs from the M1/M2/M3 anchor: {differ}"
    missing = [k for k in ANCHOR if k not in dg]
    if missing:
        return False, f"engine digest could not be resolved for {missing}"
    return True, "matches the M1/M2/M3 anchor"


def tree_state():
    out = subprocess.run(["git", "-C", os.path.join(ROOT, "src/llama.cpp"),
                          "status", "--short"], capture_output=True, text=True).stdout
    return {"dirty_tracked": len([x for x in out.splitlines() if x.strip()])}


LAUNCHER_LOG = "/tmp/decode_window_launcher.log"


def launcher_tail(n=6):
    """The launcher's own last words. run_server.sh has a memory guard that refuses to start when
    free% is low or another llama is running, and its refusal is the only place that reason is
    written down -- so it belongs in our log and in the report, not just in a file nobody reads."""
    try:
        with open(LAUNCHER_LOG, errors="replace") as fh:
            lines = [l.rstrip() for l in fh if l.strip()]
        return " | ".join(lines[-n:])
    except OSError:
        return "(no launcher log)"


def launcher_failed(proc):
    """Did the launcher already exit? run_server.sh `wait`s on the server in the foreground and only
    exits 0 early when CGC_DETACHED=1, so an exited launcher that never bound our port is a failure,
    not a slow start. Without this the harness spent ~26 min of timeouts per attempt on a launcher
    that had already died -- the same silent-waiting defect, one level lower."""
    return proc is not None and proc.poll() is not None


# ----------------------------------------------------------------------- launch
def launch(env_extra):
    env = {**os.environ, "CGC_SERVER_PORT": str(PORT),
           "CGC_SERVER_PROFILE": "prefill250", "CGC_FORCE_TEMP0": "1",
           "CGC_DECODE_PROFILE": "1", "CGC_DECODE_PROFILE_ALL": "1"}
    # CGC_DUMP_ENV=1 means "parse and print the resolved env/argv, do NOT start a server"
    # (run_server.sh:762, DUMP_ONLY at :819). Setting it -- or merely inheriting it from the caller's
    # shell -- makes every launch print the ARG dump and exit 0 with no server, which looks exactly
    # like "something killed my launcher". It is not: it is dump-only mode. Force it off.
    env["CGC_DUMP_ENV"] = "0"
    env.update(env_extra)
    lac = LAUNCHER_LOG
    proc = None
    with open(lac, "w") as fh:
        proc = subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    srv_log = None
    for _ in range(150):
        time.sleep(3)
        if launcher_failed(proc):
            log(f"    launcher exited rc={proc.returncode} before creating a server log")
            # An rc=0 exit right after an ARG dump is the dump-only signature, not a kill. Naming it
            # here is what turned 26 minutes of "interrupted" into a 3-second diagnosis.
            if proc.returncode == 0 and "ARG " in launcher_tail(40):
                log("    (rc=0 after an ARG dump = dump-only mode: CGC_DUMP_ENV was set. "
                    "Nothing killed the launcher; it was told not to start a server.)")
            log(f"    launcher said: {launcher_tail()}")
            return None, None, False
        logs = os.path.join(ROOT, "Backup/cgc_logs")
        best = None
        for f in os.listdir(logs):
            if f.startswith("llama_server_") and f.endswith(".log"):
                try:
                    p = os.path.realpath(os.path.join(logs, f))
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                if mt >= t0 - 5 and (best is None or mt > best[0]):
                    best = (mt, p)
        if best:
            srv_log = best[1]
            OUR_LOGS.add(srv_log)          # so it is never counted as a neighbour's log
            break
    pid = None
    for _ in range(120):
        held = listening(PORT)
        if held:
            pid = held[0]
            OUR_PIDS.add(pid)
            break
        if launcher_failed(proc):
            log(f"    launcher exited rc={proc.returncode} without binding port {PORT}")
            log(f"    launcher said: {launcher_tail()}")
            return pid, srv_log, False
        time.sleep(2)
    healthy = False
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read()
            healthy = True
            break
        except Exception:
            if launcher_failed(proc):
                log(f"    launcher exited rc={proc.returncode} while waiting for /health")
                log(f"    launcher said: {launcher_tail()}")
                return pid, srv_log, False
            time.sleep(5)
    return pid, srv_log, healthy


def ask(prompt, n_predict, srv_log, tag):
    lo = os.path.getsize(srv_log) if srv_log else 0
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                              headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(r, timeout=2400).read())
    hi = os.path.getsize(srv_log)
    tm = resp.get("timings") or {}
    rec = {"tokens": resp.get("tokens_predicted"),
           "ms_per_token": tm.get("predicted_per_token_ms"),
           "wall_s": time.time() - t0}
    rec.update(parse_range(srv_log, lo, hi))
    log("      " + tag + f": {rec['tokens']} tok, "
        f"{(rec['ms_per_token'] or float('nan')):.2f} ms/tok | " +
        " ".join(f"ntok={n}:total={v['total']:.1f}/wait={v['wait']:.1f}/cb={v['cb']:.1f}(n={v['n']})"
                 for n, v in sorted(rec["steps"].items())))
    return rec


# ------------------------------------------------------------------------ lanes
LANE_A = [("b1", "1"), ("b0", "0"), ("b1b", "1")]
LANE_B_SEQ = [("rep1_cold", PROMPT, N_PREDICT), ("rep2_warm", PROMPT, N_PREDICT),
              ("rep3_warm", PROMPT, N_PREDICT), ("thrash_slab2000", THRASH_PROMPT, THRASH_N),
              ("rep4_after_thrash", PROMPT, N_PREDICT), ("rep5_after_thrash2", PROMPT, N_PREDICT)]


def run_lane_a(state):
    for name, bitident in LANE_A:
        if state["lane_a"].get(name, {}).get("status") == "done":
            log(f"  [A:{name}] already done, skipping")
            continue
        ok, why = anchor_ok(digest())
        if not ok:
            log(f"  [A:{name}] {why} -> arm marked, not recorded; nothing may be concluded")
            state["lane_a"][name] = {"status": "anchor-mismatch", "bitident": bitident,
                                    "engine_digest": digest(), "anchor": why}
            save(state)
            return
        if not wait_for_window():
            log("  [A] no window within the budget")
            return
        log(f"  [A:{name}] window open, BITIDENT={bitident}")
        pid, srv_log, healthy = launch({"CGC_SERVER_MTP": "1", "CGC_MM_BITIDENT": bitident})
        rec = {"bitident": bitident, "server_log": srv_log, "pid": pid, "reps": {},
               "engine_digest": digest(), "anchor": why, "tree": tree_state()}
        if not healthy:
            rec["status"] = "interrupted"
            rec["launcher_said"] = launcher_tail()
            log(f"  [A:{name}] never became healthy -> interrupted, will re-queue")
        else:
            try:
                rec["reps"]["rep1_cold"] = ask(PROMPT, N_PREDICT, srv_log, "rep1_cold")
                rec["reps"]["rep2_warm"] = ask(PROMPT, N_PREDICT, srv_log, "rep2_warm")
                rec["status"] = "done"
            except Exception as exc:
                rec["status"] = "interrupted"
                rec["error"] = str(exc)
                log(f"  [A:{name}] request failed ({exc}) -> interrupted, will re-queue")
        state["lane_a"][name] = rec
        save(state)
        stop_ours()


DRIFT_REPS = int(os.environ.get("PROBE_DRIFT_REPS", "3"))
DRIFT_MAX_FAILS = 3


def launcher_model_path():
    """The model this profile will load, asked of the launcher instead of guessed.

    run_server.sh resolves the profile itself and prints `CGCENV MODEL <path>` in dump mode; parsing
    that keeps the harness from carrying its own copy of the mapping (which would go stale the day a
    profile changes, and would then describe a model nobody loaded).
    """
    env = {**os.environ, "CGC_DUMP_ENV": "1", "CGC_SERVER_PROFILE": "prefill250"}
    try:
        out = subprocess.run(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None
    m = re.search(r"^CGCENV MODEL\s+(\S+)\s*$", out, re.M)
    return m.group(1) if m else None


def gguf_tensors(path):
    """(name, type_name, ne00, offset) per tensor, read from the file. Enough for an eligibility check.

    The offset is kept so byte sizes can be recovered exactly as the gap to the next tensor, which
    avoids carrying a second table (per-type block sizes) that would be one more thing to keep true.
    """
    import struct
    with open(path, "rb") as f:
        magic, _ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:                       # 'GGUF' little-endian
            raise ValueError(f"not a GGUF file: {path}")

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip_scalar(t):
            size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(t)
            if size is not None:
                f.read(size)
            elif t == 8:
                rstr()
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip_scalar(et)

        for _ in range(n_kv):
            rstr()
            skip_scalar(struct.unpack("<I", f.read(4))[0])
        out = []
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            tid = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            out.append((name, GGML_TYPE_NAMES.get(tid, f"type_{tid}"), dims[0] if dims else 0, off))
        base = f.tell() - out[-1][3] if out else 0           # data section start, for the last tensor
        out.append(("<eof>", "<eof>", 0, os.path.getsize(path) - base))
        return out


def family_eligibility(model_path):
    """Which tensors in this model could EVER take the small-batch family, at any batch size.

    Eligibility = the weight type is on one of the two lists AND ne00 % 128 == 0 (the guard) AND the
    ne11 range is one the experiment actually runs. `verify` is the batch size the item-1 question is
    about (the 4-wide MTP verify step), so the range is checked against that too -- a type admitted
    only at ne11 >= 4 is not admitted for a 3-wide batch, and vice versa.
    """
    rows = gguf_tensors(model_path)
    # Exact byte size of every tensor = the gap to the next one (tensors are laid out in order).
    order = sorted(rows, key=lambda r: r[3])
    size = {}
    for a, b in zip(order, order[1:]):
        size[a[0]] = b[3] - a[3]
    per_type = {}
    for name, tname, ne00, _off in rows:
        if tname == "<eof>":
            continue
        r = per_type.setdefault(tname, {"total": 0, "expert": 0, "ne00_ok": 0, "eligible": 0,
                                        "expert_eligible": 0,
                                        "expert_bytes": 0, "expert_eligible_bytes": 0,
                                        "bytes": 0, "eligible_bytes": 0})
        nb = size.get(name, 0)
        r["total"] += 1
        r["bytes"] += nb
        is_expert = ("ffn_" in name) and ("exps" in name)
        if is_expert:
            r["expert"] += 1
            r["expert_bytes"] += nb
        ne00_ok = (ne00 % 128 == 0)
        if ne00_ok:
            r["ne00_ok"] += 1
        # ne11=4 (the verify batch this question is about) is inside BOTH ranges, so a listed type is
        # admitted for the step being measured. The ranges differ only at ne11=2..3, i.e. not here.
        eligible = ne00_ok and (tname in SMALL_BATCH_A or tname in SMALL_BATCH_B)
        if eligible:
            r["eligible"] += 1
            r["eligible_bytes"] += nb
            if is_expert:
                r["expert_eligible"] += 1
                r["expert_eligible_bytes"] += nb
    return per_type


def verdict_family(state):
    """Does the item-1 knob touch the tensors that dominate the verify step? (Source-level, no run.)

    This one gate exists because the whole item-1 experiment rests on an assumption that is not about
    timing at all: that CGC_MM_BITIDENT=0 would hand the 4-wide verify step a faster kernel. If the
    expert weights are types the small-batch family never admits, the knob is inert for exactly the
    tensors being measured -- and the honest answer is "the experiment asked the wrong question",
    not "the effect is small".
    """
    out = {"rows": [], "verdicts": []}
    mp = state.get("model_path")
    if not mp or not os.path.exists(mp):
        out["verdicts"].append(("INCOMPLETE",
                                f"the model path was not recorded ({mp!r}), so eligibility of the "
                                f"weights actually being measured cannot be checked."))
        return out
    try:
        per_type = family_eligibility(mp)
    except Exception as exc:
        out["verdicts"].append(("INCOMPLETE", f"could not read the GGUF: {exc}"))
        return out
    tot_experts, tot_expert_eligible = 0, 0
    tot_bytes = tot_expert_bytes = tot_expert_eligible_bytes = 0
    for tname, r in sorted(per_type.items(), key=lambda kv: -kv[1]["expert_bytes"]):
        fam = ("A ne11 2-8" if tname in SMALL_BATCH_A else
               "B ne11 4-8" if tname in SMALL_BATCH_B else "--")
        out["rows"].append({"type": tname, "total": r["total"], "expert": r["expert"],
                            "eligible": r["eligible"], "expert_eligible": r["expert_eligible"],
                            "expert_mib": r["expert_bytes"] / 2**20,
                            "expert_eligible_mib": r["expert_eligible_bytes"] / 2**20,
                            "family": fam})
        tot_experts += r["expert"]
        tot_expert_eligible += r["expert_eligible"]
        tot_expert_bytes += r["expert_bytes"]
        tot_expert_eligible_bytes += r["expert_eligible_bytes"]
        tot_bytes += r["bytes"]
    out["model"] = os.path.basename(mp)
    share = (tot_expert_eligible_bytes / tot_expert_bytes) if tot_expert_bytes else 0.0
    out["expert_bytes_gib"] = tot_expert_bytes / 2**30
    out["eligible_share"] = share
    if not tot_experts:
        out["verdicts"].append(("INCOMPLETE", "no 'ffn_*_exps' tensors found; wrong model or name map."))
    elif share == 0.0:
        out["verdicts"].append((
            "premise void",
            f"not one of the {tot_experts} expert tensors in {os.path.basename(mp)} "
            f"({tot_expert_bytes / 2**30:.1f} GiB) is on either small-batch list, so CGC_MM_BITIDENT "
            f"changes nothing for the tensors that dominate the verify step. Item 1's kernel "
            f"hypothesis does not apply to this model at all -- whatever the A/B shows is not a "
            f"kernel-family effect."))
    else:
        out["verdicts"].append((
            "premise bounded",
            f"the knob can only move {share * 100:.1f}% of the expert weight bytes "
            f"({tot_expert_eligible_bytes / 2**20:.0f} MiB of {tot_expert_bytes / 2**20:.0f} MiB; "
            f"{tot_expert_eligible}/{tot_experts} tensors), so any kernel-family saving on the verify "
            f"step is capped near that share of its expert-matmul cost. A measured effect much larger "
            f"than this is not a kernel-family effect -- grid contention, or a regime change."))

    # Cross-instrument check. The ceiling above is derived from the weights; the A/B is a measurement.
    # They are independent, and when the measurement is a large multiple of what the mechanism could
    # possibly produce, the honest reading is that the effect was never the mechanism -- which is a
    # conclusion neither instrument could reach alone. Reported whenever both numbers exist.
    la = state.get("lane_a") or {}

    def _ok4(name):
        rep = ((la.get(name) or {}).get("reps") or {}).get("rep2_warm") or {}
        return ((rep.get("steps") or {}).get("4") or {}).get("total")

    b0, b1 = _ok4("b0"), _ok4("b1")
    if b0 and b1:
        measured = abs(b0 - b1) / b1
        if share == 0.0 or measured > 2 * share:
            ratio_txt = "unbounded (0%)" if share == 0.0 else f"{measured / share:.0f}x"
            out["verdicts"].append((
                "not attributable",
                f"the A/B measures {measured * 100:.1f}% on the verify step (b1 {b1:.1f} -> b0 "
                f"{b0:.1f} ms) while the weights allow at most {share * 100:.1f}%: the effect is "
                f"{ratio_txt} the ceiling, so it is NOT a kernel-family effect. Block 2 of the drift "
                f"lane shows the same arm moving {abs(1 - 337.2 / 133.6) * 100:.0f}% between launches "
                f"-- that is the signature of the shared GPU, not of a kernel."))
    return out


def run_lane_drift(state):
    """Run the Lane A sequence DRIFT_REPS times and keep EVERY occurrence, for a paired analysis.

    Why: the two BITIDENT=1 baselines differed by 17.5% (166.4 vs 139.6 ms/step) while their fill
    cost was identical (cb 9.63 vs 10.15 ms) and all 41 layers were slower in the earlier arm
    (min ratio 1.020, median 1.121). A uniform GPU-side amplification with an unchanged CPU/I-O side
    is a clock/thermal signature, not a pool one -- and the first arm ran minutes after another
    session's GPU-heavy sweep, the third after 13 idle minutes. This box has no readable thermal
    probe (`pmset -g therm` records nothing, `powermetrics` needs root), so rather than trying to
    measure the temperature, the design removes it: arms adjacent in time share a thermal state, so
    the paired difference inside a block cancels the drift.

    The order is reversed on alternate blocks. b1 is first in every forward block, so a slow thermal
    ramp would bias b1 vs b1b by a fixed amount rather than by noise; alternating the order turns
    that bias into something the pairing cancels.
    """
    seq = LANE_A
    want = DRIFT_REPS * len(seq)
    got = state.setdefault("lane_drift", [])
    fails = 0
    while len(got) < want:
        idx = len(got)
        block = idx // len(seq)
        reversed_block = block % 2 == 1
        order = list(reversed(seq)) if reversed_block else seq
        name, bitident = order[idx % len(seq)]
        ok, why = anchor_ok(digest())
        if not ok:
            log(f"  [drift:{name}] {why} -> stopping this lane")
            state["lane_drift_error"] = why
            save(state)
            return
        if not wait_for_window():
            log("  [drift] no window within the budget")
            return
        log(f"  [drift] block {block + 1}/{DRIFT_REPS} slot {idx % len(seq) + 1}/{len(seq)}"
            f"{' (reversed)' if reversed_block else ''}: {name} BITIDENT={bitident}")
        pid, srv_log, healthy = launch({"CGC_SERVER_MTP": "1", "CGC_MM_BITIDENT": bitident})
        rec = {"name": name, "bitident": bitident, "block": block, "index": idx,
               "reversed": reversed_block, "ts": time.strftime("%H:%M:%S"), "pid": pid,
               "server_log": srv_log, "free_mb": vm_free_mb(), "swap_mb": swap_used_mb(),
               "engine_digest": digest(), "anchor": why, "reps": {}}
        if not healthy:
            rec["status"] = "interrupted"
            rec["launcher_said"] = launcher_tail()
        else:
            try:
                rec["reps"]["rep1_cold"] = ask(PROMPT, N_PREDICT, srv_log, "rep1_cold")
                rec["reps"]["rep2_warm"] = ask(PROMPT, N_PREDICT, srv_log, "rep2_warm")
                rec["status"] = "done"
            except Exception as exc:
                rec["status"] = "interrupted"
                rec["error"] = str(exc)
        if rec["status"] == "done":
            got.append(rec)
            fails = 0
        else:
            # Do not consume the slot on a failure: a killed launch is not a measurement, and
            # advancing here would silently thin the design (fewer pairs than requested).
            fails += 1
            log(f"  [drift:{name}] {rec['status']} -> slot NOT consumed (attempt {fails}/{DRIFT_MAX_FAILS})")
            if fails >= DRIFT_MAX_FAILS:
                log(f"  [drift] {DRIFT_MAX_FAILS} consecutive launch failures; stopping this lane")
                state["lane_drift_error"] = rec.get("launcher_said") or "launch failed repeatedly"
                save(state)
                return
        save(state)
        stop_ours()


def run_lane_b(state):
    if state["lane_b"].get("status") == "done":
        log("  [B] already done, skipping")
        return
    ok, why = anchor_ok(digest())
    if not ok:
        log(f"  [B] {why} -> lane marked, not recorded; nothing may be concluded")
        state["lane_b"] = {"status": "anchor-mismatch", "engine_digest": digest(), "anchor": why}
        save(state)
        return
    if not wait_for_window():
        log("  [B] no window within the budget")
        return
    log("  [B] window open (MTP off: every decode step is the same shape)")
    pid, srv_log, healthy = launch({"CGC_SERVER_MTP": "0"})
    rec = {"server_log": srv_log, "pid": pid, "reps": {}, "engine_digest": digest(),
           "anchor": why, "tree": tree_state()}
    if not healthy:
        rec["status"] = "interrupted"
        rec["launcher_said"] = launcher_tail()
        log("  [B] never became healthy -> interrupted, will re-queue")
    else:
        try:
            for tag, prompt, n in LANE_B_SEQ:
                rec["reps"][tag] = ask(prompt, n, srv_log, tag)
            rec["status"] = "done" if len(rec["reps"]) == len(LANE_B_SEQ) else "interrupted"
        except Exception as exc:
            rec["status"] = "interrupted"
            rec["error"] = str(exc)
            log(f"  [B] failed mid-sequence ({exc}) -> interrupted, will re-queue")
    state["lane_b"] = rec
    save(state)
    stop_ours()


LATEST: dict = {}
REPORT_OUT: str | None = None


def report_path():
    return (REPORT_OUT or os.environ.get("PROBE_OUT")
            or os.path.join(ROOT, "docs", f"DECODE_WINDOW_{datetime.now().strftime('%Y-%m-%d')}.html"))


def autoreport(state):
    """Refresh the report from whatever is known right now. Never raises: a report that fails to
    write must not take the measurement down with it."""
    try:
        out = report_path()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(write_html(state))
        return out
    except Exception as exc:                                    # noqa: BLE001 - see docstring
        log(f"    (report write failed: {exc})")
        return None


def save(state):
    """Persist state AND refresh the report.

    The first version wrote the report only when the whole run finished, so a harness that was still
    waiting for a window -- or that got killed while waiting -- produced no report at all, and the
    question "why is there no output?" had no answer on disk. Every state change now refreshes it, so
    the file always reflects the current truth (INCOMPLETE plus the reason), including the waiting
    lines from the log.
    """
    global LATEST
    LATEST = state
    state["log"] = LOG_LINES[-400:]
    # Persist which logs are ours: without this, a restart forgets and then waits out the 90s
    # log-quiet gate against its own previous arm -- a self-inflicted stall on every restart.
    state["our_logs"] = sorted(OUR_LOGS)
    with open(STATE, "w") as fh:
        json.dump(state, fh, indent=1)
    autoreport(state)


def _on_signal(signum=None, _frame=None):
    """Killing the harness must still leave a report; losing the reason is the defect, not the kill."""
    if LATEST:
        LATEST["stopped_by_signal"] = int(signum) if signum else 0
        save(LATEST)
    log(f"=== stopped (signal {signum}); report written from the state as it stands")
    raise SystemExit(0)


def load():
    if os.path.exists(STATE):
        try:
            s = json.load(open(STATE))
            s.setdefault("lane_a", {})
            s.setdefault("lane_b", {})
            OUR_LOGS.update(s.get("our_logs") or [])
            return s
        except Exception:
            pass
    return {"lane_a": {}, "lane_b": {}}


# ----------------------------------------------------------------------- verdicts
def _step(rep, ntok, key="total"):
    if not rep:
        return None
    v = (rep.get("steps") or {}).get(str(ntok)) or (rep.get("steps") or {}).get(ntok)
    return None if not v else v.get(key)


def verdict_lane_a(state):
    """Decide whether the kernel-family hypothesis is supported, with the control as the test.

    The prediction is differential, so the two questions are: did the CONTROL move, and did the
    TEST move. 'Moved' means beyond the spread between the two bracketing baselines -- otherwise the
    effect is not distinguishable from this box's launch-to-launch drift and must be reported as
    such rather than rounded into a conclusion.
    """
    arms = state["lane_a"]
    out = {"rows": [], "verdicts": []}
    bad = [a.get("anchor") for a in arms.values() if a.get("status") == "anchor-mismatch"]
    if bad:
        out["verdicts"].append(("INCOMPLETE",
                                "lane A was measured on a build the M1/M2/M3 gate never passed: "
                                + bad[0] + ". The numbers cannot be attributed to the gated "
                                "engine, so none are reported here."))
        return out
    if any(a.get("status") != "done" for a in arms.values()) or len(arms) < 3:
        out["verdicts"].append(("INCOMPLETE", "lane A has not finished; nothing may be concluded."))
        return out

    for name, _ in LANE_A:
        rep = arms[name]["reps"].get("rep2_warm") or {}
        out["rows"].append({
            "arm": name, "bitident": arms[name]["bitident"],
            "ms_per_token": rep.get("ms_per_token"),
            "ntok1_total": _step(rep, 1), "ntok1_wait": _step(rep, 1, "wait"),
            "ntok1_cb": _step(rep, 1, "cb"),
            "ntok4_total": _step(rep, 4), "ntok4_wait": _step(rep, 4, "wait"),
            "ntok4_cb": _step(rep, 4, "cb"),
            "log": os.path.basename(arms[name].get("server_log") or ""),
        })

    base = [r for r in out["rows"] if r["bitident"] == "1"]
    test = [r for r in out["rows"] if r["bitident"] == "0"]
    if len(base) < 2 or not test:
        out["verdicts"].append(("INCOMPLETE", "missing a baseline or the test arm."))
        return out
    test = test[0]

    def spread(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None, None
        mu = st.mean(vals)
        return mu, (max(vals) - min(vals)) / mu * 100.0 if mu else None

    # The plan's differential control is the plain (ntok=1) step: ne11=1 is not eligible for the
    # small-batch family either way, so it should not move. With MTP on there are no ntok=1 base
    # steps at all -- every base step is a 4-wide verify batch -- so on this server the control does
    # not exist. Report that as a design gap, not as a missing number.
    have_control = any(_step((arms[a].get("reps") or {}).get("rep2_warm") or {}, 1) is not None
                       for a, _ in LANE_A)
    metrics = [("ntok4_total", "ntok=4 verify step (TEST: ne11=4)")]
    if have_control:
        metrics.insert(0, ("ntok1_total",
                           "ntok=1 plain step (CONTROL: ne11=1 -> mul_mv either way)"))
    else:
        out["verdicts"].append(("DESIGN GAP",
            "no ntok=1 plain step exists to act as the control: with MTP on every base step is a "
            "4-wide verify batch. The verify step is judged on its own; the control is recorded as "
            "UNAVAILABLE, not as a pass."))

    for metric, label in metrics:
        mu, sp = spread([r[metric] for r in base])
        tv = test[metric]
        if mu is None or tv is None:
            # Only reachable when the metric was expected to exist; a missing shape is reported in
            # the rows (as an em dash) rather than silently deciding anything.
            log(f"    (lane A: {label} has no comparable value; not judged)")
            continue
        delta = (tv - mu) / mu * 100.0
        # The threshold is a FLOOR, not just the measured spread. With a 0.0% spread (two baselines
        # that happened to agree exactly) using the spread alone makes every 1% wobble "moved" and
        # rounds launch noise into a conclusion -- the exact failure this harness exists to prevent.
        # BASELINE_AGREE_PCT is that floor, and it is also the drift budget: if the two baselines
        # cannot agree within it, the box moved under us and no comparison against them is valid.
        thr = max(sp or 0.0, BASELINE_AGREE_PCT)
        if sp is not None and sp > BASELINE_AGREE_PCT:
            out["drift"] = (f"the two BITIDENT=1 baselines disagree by {sp:.1f}% "
                            f"(budget {BASELINE_AGREE_PCT:.0f}%)")
        moved = abs(delta) > thr
        out["verdicts"].append((
            "moved" if moved else "flat",
            f"{label}: baselines {mu:.1f} ms (spread {sp:.1f}%, threshold {thr:.1f}%), "
            f"BITIDENT=0 {tv:.1f} ms => {delta:+.1f}% -> "
            f"{'BEYOND' if moved else 'within'} baseline spread"))

    control = next((v for k, v in out["verdicts"] if k in ("moved", "flat") and "CONTROL" in v), None)
    testv = next((v for k, v in out["verdicts"] if k in ("moved", "flat") and "TEST" in v), None)
    control_moved = bool(control and "BEYOND" in control)
    test_moved = bool(testv and "BEYOND" in testv)
    # `test` is a summary row, not a rep -- _step() would return None here and make CONFIRMED
    # unreachable, i.e. a verdict function that could only ever say no. Compare the row field.
    base4 = [r["ntok4_total"] for r in base if r["ntok4_total"] is not None]
    test4 = test["ntok4_total"]
    test_faster = bool(test_moved and base4 and test4 is not None and test4 < st.mean(base4))

    # A fixed-shape per-step cost is comparable across arms; ms/token is not, because the BITIDENT=0
    # arm breaks bit-identity, so its token stream (and hence accept rate and tokens/step) diverges.
    # The measured data makes this concrete: b0's verify step is cheaper while its ms/token is much
    # worse, which can only come from fewer tokens per step.
    out["verdicts"].append(("CAVEAT",
        "ms/token is NOT comparable across these arms -- BITIDENT=0 breaks bit-identity, so the "
        "token stream and the accept rate differ. Compare fixed-shape per-step costs only."))

    if not have_control and test_moved:
        out["verdicts"].append(("INCONCLUSIVE",
            "The verify step moved, but with no control arm there is no way to separate a "
            "family-specific effect from a global one. Get a control before claiming either."))
    elif out.get("drift") and not test_moved:
        out["verdicts"].append(("NOT SUPPORTED",
            "The verify step is inside this box's own drift (" + out["drift"] +
            "), so the small-batch family shows no resolvable effect here. The comparison is "
            "underpowered, not negative: tightening the baselines is what would change this."))
    elif control_moved and test_moved:
        out["verdicts"].append(("REFUTED",
                                "Both the control and the test moved. The kernel family is not the "
                                "mechanism; something global differs between the BITIDENT settings."))
    elif test_moved and test_faster:
        out["verdicts"].append(("CONFIRMED",
                                "Control flat, verify step faster => the small-batch mat-mv family is "
                                "the mechanism. An M-invariant version of it is worth building."))
    elif test_moved:
        out["verdicts"].append(("PARTIAL",
                                "Control flat and the verify step moved, but not in the expected "
                                "direction (slower). Re-examine before building anything."))
    else:
        out["verdicts"].append(("NOT SUPPORTED",
                                "The verify step did not move beyond baseline spread: the "
                                "BITIDENT-forced mul_mv path is not what costs the 2.28x."))
    return out


def verdict_lane_b(state):
    """Is the post-thrash residue global or concentrated, and does one more pass clear it?"""
    lb = state.get("lane_b") or {}
    out = {"rows": [], "layers": [], "verdicts": []}
    if lb.get("status") == "anchor-mismatch":
        out["verdicts"].append(("INCOMPLETE",
                                "lane B ran on a build the M1/M2/M3 gate never passed: "
                                + str(lb.get("anchor")) + ". Nothing is reported from it."))
        return out
    if lb.get("status") != "done":
        out["verdicts"].append(("INCOMPLETE", "lane B has not finished; nothing may be concluded."))
        return out
    reps = lb.get("reps") or {}
    warm_names = ["rep2_warm", "rep3_warm"]

    for tag in [t for t, _, _ in LANE_B_SEQ]:
        rep = reps.get(tag) or {}
        out["rows"].append({"tag": tag, "tokens": rep.get("tokens"),
                            "ms_per_token": rep.get("ms_per_token"),
                            "step_total": _step(rep, 1), "step_wait": _step(rep, 1, "wait"),
                            "step_cb": _step(rep, 1, "cb")})

    base = [r for n in warm_names for r in [reps.get(n)] if r]
    if not base:
        out["verdicts"].append(("INCOMPLETE", "no warm baseline requests."))
        return out
    warm_wait = st.mean([_step(r, 1, "wait") for r in base if _step(r, 1, "wait") is not None])
    warm_cb = st.mean([_step(r, 1, "cb") for r in base if _step(r, 1, "cb") is not None])

    # Every residue judgement below uses ms/token -- the clock a user is actually served on -- and not
    # the per-step `wait` median, which excludes work the GPU overlaps and can therefore move several
    # percent while the rate does not. Measured live: rep5's wait was 5.6% off warm while its rate was
    # back on baseline, and judging on `wait` called that "not recovered".
    def _rate(r):
        return (r or {}).get("ms_per_token")

    warm_rates = [_rate(r) for r in base if _rate(r) is not None]
    warm_rate = st.mean(warm_rates) if warm_rates else None
    after_rate = _rate(reps.get("rep4_after_thrash"))

    # Per-layer residue: warm baseline layer table vs the first request after the thrash.
    base_layers = {}
    per_layer_sets = [r.get("layers") or {} for r in base]
    for L in set().union(*[set(d) for d in per_layer_sets]) if per_layer_sets else set():
        vals = [d[L]["wait"] for d in per_layer_sets if L in d]
        if vals:
            base_layers[L] = st.mean(vals)
    after = (reps.get("rep4_after_thrash") or {}).get("layers") or {}
    deltas = {L: after[L]["wait"] - base_layers[L]
              for L in sorted(set(base_layers) & set(after))}
    if deltas:
        pos = {L: d for L, d in deltas.items() if d > 0}
        total = sum(pos.values())
        top5 = sum(sorted(pos.values(), reverse=True)[:5])
        share = (top5 / total) if total else 0.0
        worst = sorted(deltas.items(), key=lambda kv: -kv[1])[:6]
        out["layers"] = [{"layer": L, "warm_wait": base_layers[L], "after_wait": after[L]["wait"],
                          "delta": d} for L, d in worst]
        # Only classify the shape of the residue once there IS a material residue to place. Two
        # milliseconds spread over nineteen layers has a "top-5 share" that says nothing at all --
        # measured live it printed "concentrated in a few layers, top-5 share 79%" for +2 ms total.
        material = bool(warm_rate and after_rate and
                        abs((after_rate - warm_rate) / warm_rate * 100.0) >= MATERIAL_RESIDUE_PCT)
        if not material:
            out["verdicts"].append((
                "no material residue",
                f"per-layer deltas total only +{total:.0f} ms across {len(pos)} layers; not labelled "
                f"concentrated or global, because at this size neither word means anything."))
        else:
            out["verdicts"].append((
                "concentrated" if share >= RESIDUE_CONCENTRATION else "global",
                f"thrash residue: total +{total:.0f} ms across {len(pos)} layers, top-5 share "
                f"{share * 100:.0f}% => "
                + ("concentrated in a few layers" if share >= RESIDUE_CONCENTRATION
                   else "broad-based")))

    rep5 = reps.get("rep5_after_thrash2") or {}
    rep5_rate = _rate(rep5)

    if warm_rate and after_rate:
        res_pct = (after_rate - warm_rate) / warm_rate * 100.0
        if abs(res_pct) < MATERIAL_RESIDUE_PCT:
            out["verdicts"].append((
                "no material residue",
                f"after a 2000-token slab thrash the rate is {after_rate:.2f} vs {warm_rate:.2f} "
                f"ms/token ({res_pct:+.1f}%): below the {MATERIAL_RESIDUE_PCT:.0f}% threshold, so "
                f"there is no residue worth engineering away in this run. The per-layer table below "
                f"is then comparing noise -- read it as such."))
        else:
            out["verdicts"].append((
                "material residue",
                f"after a 2000-token slab thrash the rate is {after_rate:.2f} vs {warm_rate:.2f} "
                f"ms/token ({res_pct:+.1f}%)."))

    if warm_rate and rep5_rate:
        rec_pct = (rep5_rate - warm_rate) / warm_rate * 100.0
        recovered = abs(rec_pct) < MATERIAL_RESIDUE_PCT
        out["verdicts"].append((
            "recovered" if recovered else "not recovered",
            f"second pass after the thrash: {rep5_rate:.2f} ms/token vs warm {warm_rate:.2f} "
            f"({rec_pct:+.1f}%) => "
            + ("one more pass restores the rate, so 're-warm after prefill' is a sufficient fix"
               if recovered else
               "a second pass does NOT restore the rate; the damage is not just pool contents")))
    return out


def verdict_drift(state):
    """Did interleaving give the item-1 comparison the resolution it was missing?

    The test is narrow and has nothing to do with BITIDENT=0: two arms that are IDENTICAL by
    construction must agree within BASELINE_AGREE_PCT, otherwise no effect measured against them
    means anything. The first run's baselines differed by 17.5%, which is why its result had to be
    reported as underpowered rather than as evidence.
    """
    runs = [r for r in (state.get("lane_drift") or []) if r.get("status") == "done"]
    out = {"rows": [], "blocks": [], "verdicts": []}
    if state.get("lane_drift_error"):
        out["verdicts"].append(("INCOMPLETE",
                                "the drift lane stopped early: " + str(state["lane_drift_error"])))
    if not runs:
        out["verdicts"].append(("INCOMPLETE", "the drift lane has no completed arms yet."))
        return out

    def warm_step(r):
        rep = (r.get("reps") or {}).get("rep2_warm") or {}
        return ((rep.get("steps") or {}).get("4") or {}).get("total")

    for r in sorted(runs, key=lambda x: x["index"]):
        rep = (r.get("reps") or {}).get("rep2_warm") or {}
        s = (rep.get("steps") or {}).get("4") or {}
        out["rows"].append({"block": r["block"] + 1, "arm": r["name"],
                            "bitident": r["bitident"], "ts": r.get("ts"),
                            "free_mb": r.get("free_mb"), "swap_mb": r.get("swap_mb"),
                            "step_total": s.get("total"),
                            "cb": s.get("cb"), "wait": s.get("wait"),
                            "ms_per_token": rep.get("ms_per_token")})

    by_block = {}
    for r in runs:
        v = warm_step(r)
        if v:
            by_block.setdefault(r["block"], {})[r["name"]] = v
    ratios = []
    for b, d in sorted(by_block.items()):
        a, c = d.get("b1"), d.get("b1b")
        if a and c:
            ratios.append(a / c)
            out["blocks"].append({"block": b + 1, "b1": a, "b1b": c, "ratio": a / c})

    # A different question from the paired one, and the one that decides whether pairing could ever
    # work here: if the arms inside a SINGLE block are further apart than the effect being hunted, the
    # pair members were not measured in the same regime -- and no number of extra blocks repairs that.
    flips = []
    out["blocks_all"] = []
    for b, d in sorted(by_block.items()):
        vals = list(d.values())
        if len(vals) < 2:
            continue
        sp = (max(vals) - min(vals)) / st.mean(vals) * 100.0
        flips.append(sp)
        out["blocks_all"].append({"block": b + 1, "spread": sp, "arms": dict(d)})
        out["verdicts"].append((
            "regime faster than one arm" if sp > REGIME_FLIP_PCT else "block stable",
            f"block {b + 1}: arms span {sp:.1f}% ("
            + ", ".join(f"{k}={v:.1f}" for k, v in d.items())
            + f") against a {REGIME_FLIP_PCT:.0f}% flip threshold; a pair in this block was "
            + ("NOT " if sp > REGIME_FLIP_PCT else "") + "measured inside one regime."))

    # A single pair has no spread by construction: max(ratios) - min(ratios) is 0 the moment there is
    # one ratio, so any code that reads the spread here would call one pair "controlled" and then
    # "resolved". The block-level verdicts above still stand (they read within-block spread, which
    # one block *can* demonstrate); the paired question needs two or more.
    if len(ratios) < 2:
        out["verdicts"].append(("INCOMPLETE",
                                f"{len(ratios)} pair(s) with both BITIDENT=1 baselines; a paired "
                                f"spread needs at least 2, so the paired question stays open. "
                                f"({len(runs)} arm(s) measured.)"))
        return out

    mu = st.mean(ratios)
    matched_diff = abs(mu - 1.0) * 100.0
    matched_spread = (max(ratios) - min(ratios)) / mu * 100.0
    allb1 = [d["b1"] for d in by_block.values() if "b1" in d]
    allb1b = [d["b1b"] for d in by_block.values() if "b1b" in d]
    unpaired_diff = abs(st.mean(allb1) - st.mean(allb1b)) / st.mean(allb1b) * 100.0
    pooled = allb1 + allb1b
    unpaired_spread = (max(pooled) - min(pooled)) / st.mean(pooled) * 100.0 if len(pooled) > 1 else 0.0

    out["verdicts"].append((
        "interleaving controls drift" if max(matched_diff, matched_spread) < BASELINE_AGREE_PCT
        else "interleaving insufficient",
        f"the two IDENTICAL BITIDENT=1 arms now differ by {matched_diff:.1f}% "
        f"(paired over {len(ratios)} blocks), with a paired ratio spread of {matched_spread:.1f}%. "
        f"Unpaired, the same arms differ by {unpaired_diff:.1f}% with a pooled spread of "
        f"{unpaired_spread:.1f}%. Threshold is {BASELINE_AGREE_PCT:.0f}%."))
    if max(matched_diff, matched_spread) < BASELINE_AGREE_PCT:
        out["verdicts"].append(("resolved",
            "Item 1's comparison is now admissible: an effect of the size the plan looked for "
            "(~2.28x on the verify step) would be visible against these baselines."))
    else:
        out["verdicts"].append(("NOT SUPPORTED",
            f"Still underpowered at {max(matched_diff, matched_spread):.1f}%: the identical baselines "
            f"disagree by more than the {BASELINE_AGREE_PCT:.0f}% margin, so no effect measured "
            f"against them is readable."))
        if flips and max(flips) > REGIME_FLIP_PCT:
            # The distinction that matters: "not enough blocks" and "the regime flips inside a block"
            # call for opposite next actions, and only the second one is true when a flip is present.
            out["verdicts"].append(("NOT SUPPORTED",
                f"And more blocks will not fix it: a block flipped by {max(flips):.1f}% inside itself, "
                f"so the noise timescale is shorter than one arm -- the two members of a pair are never "
                f"in one regime. On a shared box this is two GPU jobs, not a thermal ramp: one llama "
                f"process is enough to move `wait` by 2x with `cb` unchanged. The only measurement shape "
                f"that can answer item 1 is one with no cross-launch component at all (both kernel "
                f"families timed inside one process, seconds apart)."))
            out["verdicts"].append(("not settled by this shape",
                "the interleaved cross-launch A/B is hereby the wrong instrument for this question; "
                "its rows are still useful as a distribution of the box's regimes."))
    out["verdicts"].append(("CAVEAT",
        "the alternating block order is what removes a fixed first-slot bias; a run that lost that "
        "rotation would look identical in the rows and would not be paired-comparable."))
    return out


# ------------------------------------------------------------------------- report
def write_html(state):
    a = verdict_lane_a(state)
    b = verdict_lane_b(state)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    digests = (state.get("lane_a", {}).get("b1", {}).get("engine_digest")
               or state.get("lane_b", {}).get("engine_digest") or {})

    def esc(x):
        return html.escape(str(x))

    def table(headers, rows):
        th = "".join(f"<th>{esc(h)}</th>" for h in headers)
        trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    def badge(kind):
        cls = {"CONFIRMED": "ok", "REFUTED": "bad", "NOT SUPPORTED": "warn",
               "PARTIAL": "warn", "INCONCLUSIVE": "warn", "INCOMPLETE": "warn",
               "DESIGN GAP": "bad", "CAVEAT": "warn", "no material residue": "ok",
               "material residue": "bad", "resolved": "ok",
               "interleaving controls drift": "ok", "interleaving insufficient": "bad",
               "moved": "bad", "flat": "ok", "concentrated": "warn", "global": "warn",
               "recovered": "ok", "not recovered": "bad",
               "premise void": "bad", "premise bounded": "warn", "premise live": "warn",
               "not attributable": "bad",
               "regime faster than one arm": "bad", "block stable": "ok",
               "not settled by this shape": "bad"}.get(kind, "warn")
        return f'<span class="badge {cls}">{esc(kind)}</span>'

    parts = [f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Decode window harness — {esc(ts)}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;
   max-width:1080px;margin:2rem auto;padding:0 1.2rem;line-height:1.65;color:#1a1a1a}}
 h1{{font-size:1.5rem;border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{font-size:1.15rem;margin-top:2rem;border-left:4px solid #555;padding-left:.6rem}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}}
 th,td{{border:1px solid #ddd;padding:.4rem .55rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f4f4f4}}
 code{{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 .badge{{display:inline-block;padding:.12rem .5rem;border-radius:10px;color:#fff;font-size:.8rem;
   font-weight:600}}
 .ok{{background:#1a7f37}} .bad{{background:#b42318}} .warn{{background:#9a6700}}
 .note{{background:#fbf8e5;border-left:4px solid #d4a72c;padding:.6rem .9rem;margin:1rem 0}}
 .v{{margin:.5rem 0 .5rem .6rem}}
</style></head><body>
<h1>Decode window harness</h1>
<p><strong>{esc(ts)}</strong> · port {PORT} (8080 only observed, never signalled) ·
engine digests: {esc(json.dumps(digests))}</p>
<div class="note"><strong>如何讀這份報告：</strong>每個判定都要打贏「這台機器自己的漂移」才算數。
Lane A 的兩個 BITIDENT=1 基線之間的分歧就是那個門檻；效果小於門檻一律標成
INCONCLUSIVE，不會被四捨五入成結論。Lane B 的冷/暖是變數不是雜訊。</div>"""]

    parts.append("<h2>Lane A — item 1：small-batch mat-mv 是不是 2.28× 的成因</h2>")
    parts.append("<p>預測是<b>微分</b>的：原始碼說 <code>ne11=1</code> 本來就不在 "
                 "small-batch 家族裡，所以 BITIDENT=0 應該<b>只動 ntok=4</b>、不動 ntok=1。</p>")
    if a["rows"]:
        parts.append(table(
            ["arm", "BITIDENT", "ms/token", "ntok=1 total", "ntok=1 wait", "ntok=1 cb",
             "ntok=4 total", "ntok=4 wait", "ntok=4 cb", "log"],
            [[r["arm"], r["bitident"], _f(r["ms_per_token"]), _f(r["ntok1_total"]),
              _f(r["ntok1_wait"]), _f(r["ntok1_cb"]), _f(r["ntok4_total"]), _f(r["ntok4_wait"]),
              _f(r["ntok4_cb"]), r["log"]] for r in a["rows"]]))
    for kind, text in a["verdicts"]:
        parts.append(f'<div class="v">{badge(kind)} {esc(text)}</div>')

    parts.append("<h2>Lane B — items 6–8：thrash 的殘留與暖池機制</h2>")
    if b["rows"]:
        parts.append(table(["request", "tokens", "ms/token", "step total", "step wait", "step cb"],
                           [[r["tag"], r["tokens"], _f(r["ms_per_token"]), _f(r["step_total"]),
                             _f(r["step_wait"]), _f(r["step_cb"])] for r in b["rows"]]))
    if b["layers"]:
        parts.append("<p><b>逐層殘留</b>（暖池 vs thrash 後第一個請求，wait 差值最大的幾層）：</p>")
        parts.append(table(["layer", "warm wait", "after-thrash wait", "delta"],
                           [[r["layer"], _f(r["warm_wait"]), _f(r["after_wait"]), _f(r["delta"])]
                            for r in b["layers"]]))
    for kind, text in b["verdicts"]:
        parts.append(f'<div class="v">{badge(kind)} {esc(text)}</div>')

    if state.get("blocked_by") and not state.get("synthetic"):
        parts.append('<div class="note"><strong>還沒開始量測。</strong>harness 正在等一個乾淨窗口，'
                     f'自 {esc(state.get("waiting_since") or "?")} 起被擋住：'
                     f'<code>{esc(state["blocked_by"])}</code>。下面那兩個 INCOMPLETE 就是這個意思。'
                     '要看即時狀態用 <code>tail -f /tmp/harness.log</code>。</div>')
    _said = [a.get("launcher_said") for a in (state.get("lane_a") or {}).values()
             if a.get("launcher_said")]
    if not _said and (state.get("lane_b") or {}).get("launcher_said"):
        _said = [state["lane_b"]["launcher_said"]]
    if _said and not state.get("synthetic"):
        parts.append('<div class="note"><strong>啟動失敗時，launcher 自己的話：</strong>'
                     f'<code>{esc(_said[-1])}</code><br>這通常是 <code>run_server.sh</code> 的 '
                     'memory guard 拒絕啟動（free% 過低、或有別人的 llama 在跑），'
                     '不是 harness 的判定問題。</div>')
    if state.get("synthetic"):
        parts.append('<div class="note" style="background:#fdecec;border-left-color:#b42318">'
                     '<strong>SYNTHETIC RENDER TEST</strong> — 下面每一個數字都是為了驗證報告排版而'
                     '鍵入的，<b>不是量測結果</b>，不可引用。</div>')
    dr = verdict_drift(state)
    parts.append("<h2>Drift lane — 交錯重複的 b1/b1b：item 1 的比較有沒有解析度</h2>")
    parts.append("<p>問的不是 BITIDENT，是兩個<b>完全同配置</b>的臂能不能在 5% 內一致。第一次跑它們差 "
                 "17.5%，所以那個結果只能報成 underpowered。逐層證據：b1/b1b 的 wait 比值中位 1.121、"
                 "<b>最小 1.020（沒有一層更快）</b>、而 cb 比值 0.974 ⇒ 全域 GPU 側因子（時脈/熱），"
                 "不是 pool。所以這條線<b>交錯時序</b>並輪替區塊內順序，讓配對差值抵銷漂移。</p>")
    if dr["blocks"]:
        parts.append("<p><b>配對結果</b>（同一區塊內 b1 vs b1b 的 verify step）：</p>")
        parts.append(table(["block", "b1 step (ms)", "b1b step (ms)", "ratio"],
                           [[r["block"], _f(r["b1"]), _f(r["b1b"]), f"{r['ratio']:.4f}"]
                            for r in dr["blocks"]]))
    if dr.get("blocks_all"):
        parts.append("<p><b>區塊內全臂展幅</b>（這才是「配對有沒有用」的判準：同一區塊內所有臂的"
                     "span。大於門檻就代表這一對<b>不在同一個 regime</b>，再多 block 也補不回來）：</p>")
        parts.append(table(["block", "span %", "arms (ms)"],
                           [[r["block"], _f(r["spread"]),
                             ", ".join(f"{k}={v:.1f}" for k, v in r["arms"].items())]
                            for r in dr["blocks_all"]]))
    if dr["rows"]:
        parts.append("<p>每個臂的原始讀數（時間與 free 是為了讓漂移看得到，不是解釋）：</p>")
        parts.append(table(["block", "arm", "BITIDENT", "ts", "free MB", "swap MB", "step total",
                            "cb", "wait", "ms/token"],
                           [[r["block"], r["arm"], r["bitident"], r["ts"], _f(r["free_mb"], 0),
                             _f(r["swap_mb"], 0), _f(r["step_total"]), _f(r["cb"]), _f(r["wait"]),
                             _f(r["ms_per_token"])] for r in dr["rows"]]))
    for kind, text in dr["verdicts"]:
        parts.append(f'<div class="v">{badge(kind)} {esc(text)}</div>')

    fam = verdict_family(state)
    parts.append("<h2>Kernel family eligibility — item 1 的前提是否成立（讀真的 GGUF，不跑量測）</h2>")
    parts.append("<p>這件事<b>不是量測問題</b>：small-batch 家族（<code>mul_mv_ext</code>）只接受兩份"
                 "互斥的型別白名單，其餘型別一律落到 mul_mv/mul_mm；在那裡 batch size 不改變每列的"
                 "歸約，所以 <code>CGC_MM_BITIDENT</code> 對它們<b>不可能有任何效果</b>。若 expert "
                 "權重的型別一個都不在白名單上，那 item 1 的 kernel 假設對這顆模型根本不適用——"
                 "此時 A/B 裡看到的任何差異都不是 kernel 家族的效應。</p>")
    if fam.get("rows"):
        parts.append(table(["type", "tensors", "of which expert", "eligible",
                            "expert & eligible", "expert MiB", "expert & elig MiB", "family"],
                           [[r["type"], r["total"], r["expert"], r["eligible"],
                             r["expert_eligible"], _f(r["expert_mib"], 0),
                             _f(r["expert_eligible_mib"], 0), r["family"]]
                            for r in fam["rows"]]))
    for kind, text in fam["verdicts"]:
        parts.append(f'<div class="v">{badge(kind)} {esc(text)}</div>')

    if state.get("log"):
        parts.append("<h2>執行日誌（最後 400 行）</h2><pre style='font-size:.82rem;background:#f7f7f7;"
                     "padding:.8rem;overflow:auto;max-height:26rem'>"
                     + esc("\n".join(state["log"])) + "</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def sample_state():
    """A shape-correct state for checking that the report renders, marked synthetic so it can never
    be read as a measurement. Used by `render-sample`; never written to docs/ by default."""
    def rep(ntok1, ntok4, mpt):
        return {"ms_per_token": mpt, "tokens": 92,
                "steps": {"1": {"total": ntok1, "wait": ntok1 - 20, "cb": 20.0, "submit": 4.0, "n": 9},
                          "4": {"total": ntok4, "wait": ntok4 - 30, "cb": 30.0, "submit": 6.0, "n": 30}},
                "layers": {L: {"wait": 1.9, "cb": 0.5, "gpu": 1.4, "union": 3.0, "gap": 0.2}
                           for L in range(40)}}

    def arm(name, bitident, ntok1, ntok4, mpt):
        return {"status": "done", "bitident": bitident, "pid": 2000,
                "server_log": f"/tmp/{name}.log",
                "engine_digest": {"llama-server": "054fb22f04a01c5c",
                                  "libllama.0.dylib": "aab787412e572550"},
                "tree": {"dirty_tracked": 0},
                "reps": {"rep1_cold": rep(ntok1 + 40, ntok4 + 60, mpt * 1.45),
                         "rep2_warm": rep(ntok1, ntok4, mpt)}}

    warm = rep(95.0, 150.0, 75.0)
    # Keep the sample in the regime the box actually measures: a warm pool has cb ~4.8 ms/step, not
    # the ~20 ms a cold first request pays. A sample that ignores this reads as a finding.
    warm["steps"]["1"] = {"total": 75.8, "wait": 63.9, "cb": 4.8, "submit": 5.0, "n": 9}
    warm["steps"]["4"] = {"total": 152.6, "wait": 71.0, "cb": 35.2, "submit": 9.0, "n": 30}
    warm["layers"] = {L: {"wait": 1.9, "cb": 0.4, "gpu": 1.4, "union": 3.0, "gap": 0.2}
                      for L in range(40)}
    after = {"ms_per_token": 101.0, "tokens": 92,
             "steps": {"1": {"total": 94.0, "wait": 79.0, "cb": 4.6, "submit": 5.0, "n": 9}},
             "layers": {L: {"wait": 2.0 + (14.0 if L in (37, 38, 39) else 0.0), "cb": 0.6,
                             "gpu": 1.5, "union": 3.0, "gap": 0.2} for L in range(40)}}
    return {
        "synthetic": True,
        "lane_a": {"b1": arm("b1", "1", 95.0, 152.0, 100.0),
                   "b0": arm("b0", "0", 94.0, 110.0, 88.0),
                   "b1b": arm("b1b", "1", 95.0, 150.0, 101.0)},
        "lane_b": {"status": "done", "engine_digest": {"llama-server": "054fb22f04a01c5c"},
                   "pid": 2001, "server_log": "/tmp/lane_b.log", "tree": {"dirty_tracked": 0},
                   "reps": {"rep1_cold": {**rep(120.0, 0.0, 120.0),
                                          "steps": {"1": {"total": 110.0, "wait": 77.1,
                                                             "cb": 20.4, "submit": 6.0, "n": 7}}},
                            "rep2_warm": warm,
                            "rep3_warm": {**warm, "steps": {"1": {"total": 78.0, "wait": 68.8,
                                                                        "cb": 3.7, "submit": 5.0, "n": 9}}},
                            "thrash_slab2000": {"ms_per_token": 110.0, "tokens": 8, "steps": {}},
                            "rep4_after_thrash": after,
                            "rep5_after_thrash2": {**warm, "ms_per_token": 84.0}}},
        "log": ["[synthetic] render-sample: no server was launched; no lane was run."],
    }


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def cmd_run(args):
    global REPORT_OUT
    REPORT_OUT = args.out
    state = load()
    log(f"=== decode window harness (port {PORT}; {OTHER_PORT} observed only) ===")
    # Resolve the model ONE way, from the launcher, and keep it: the kernel-family gate below reads
    # the real GGUF, so "which model was measured" has to be a recorded fact and not an assumption
    # about the profile. Only asked once; a cached value is reused across restarts.
    if not state.get("model_path"):
        mp = launcher_model_path()
        if mp:
            state["model_path"] = mp
            log(f"--- model (from the launcher): {os.path.basename(mp)}")
        else:
            log("--- model: the launcher did not report CGCENV MODEL; the family gate will say so")
    # Install the handlers and write the first report BEFORE waiting: from here on there is always a
    # report on disk saying what the harness is doing and why, even if it never gets a window.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass
    save(state)
    log("--- report (kept current from now on): " + report_path())
    if args.analyze_only:
        log("--- --analyze-only: running no lanes, reporting from the saved state")
        state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(state)
        write_report(args, state)
        return None
    for p in range(PASSES):
        # "anchor-mismatch" is terminal, not pending: retrying it would just relaunch the same
        # wrong-build server until the budget ran out, and would hide the real reason.
        DONEISH = ("done", "anchor-mismatch")
        a_pending = [n for n, _ in LANE_A
                     if state["lane_a"].get(n, {}).get("status") not in DONEISH]
        b_pending = state["lane_b"].get("status") not in DONEISH
        d_pending = len(state.setdefault("lane_drift", [])) < DRIFT_REPS * len(LANE_A)
        want_a = a_pending and args.only in ("both", "lane_a")
        want_b = b_pending and args.only in ("both", "lane_b")
        want_d = d_pending and args.only in ("both", "drift")
        if not (want_a or want_b or want_d):
            break
        if budget_left() <= 0:
            log(f"--- overall budget of {TOTAL_BUDGET_S / 3600:.1f}h exhausted, stopping")
            break
        log(f"--- pass {p + 1}/{PASSES}: A pending={a_pending} B pending={b_pending} "
            f"drift {len(state.get('lane_drift') or [])}/{DRIFT_REPS * len(LANE_A)} "
            f"(budget left {budget_left() / 60:.0f} min)")
        if want_a:
            run_lane_a(state)
        if want_b and budget_left() > 0:
            run_lane_b(state)
        if want_d and budget_left() > 0:
            run_lane_drift(state)
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(state)
    log("=== run finished; state in " + STATE)
    # The report writes itself: comparing the arms is the deliverable, and leaving it as a second
    # command is how a harness ends up producing numbers nobody ever verdicts.
    write_report(args, state)


def write_report(args, state):
    """Write the HTML verdict report and echo the verdicts. Returns the path."""
    out = getattr(args, "out", None) or report_path()
    with open(out, "w") as fh:
        fh.write(write_html(state))
    a = verdict_lane_a(state)
    b = verdict_lane_b(state)
    log("=== verdicts ===")
    for kind, text in a["verdicts"] + b["verdicts"] + verdict_drift(state)["verdicts"]:
        log(f"  [{kind}] {text}")
    log("report: " + out)
    return out


def cmd_report(args):
    write_report(args, load())
    return None


def cmd_render_sample(args):
    write_report(args, sample_state())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="wait for a window and run the lanes")
    r.add_argument("--only", choices=["both", "lane_a", "lane_b", "drift"], default="both")
    r.add_argument("--analyze-only", action="store_true",
                   help="skip everything and just report from the saved state")
    r.add_argument("--out", default=None,
                   help="where to write the report (default docs/DECODE_WINDOW_<date>.html); an "
                        "--analyze-only pass should point this away from docs/ so a report with no "
                        "measurement in it cannot be mistaken for a result")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="write the HTML verdict report from the saved state")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)
    s = sub.add_parser("render-sample", help="write a SYNTHETIC report to check the layout "
                                             "(never a measurement, never written to docs/)")
    s.add_argument("--out", default="/tmp/DECODE_WINDOW_RENDER_TEST.html")
    s.set_defaults(func=cmd_render_sample)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
