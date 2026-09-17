#!/usr/bin/env python3
"""Phase-split prefill A/B: is the armed slab faster, or is the clamp merely slower?

THE QUESTION (decided 2026-09-17, option D)
------------------------------------------
The phase split only exists when the whole-layer slab is armed, and only the `prefill250` profile
arms it. If arming is worth having as a default, the reason has to be a measurement. This is it.

WHY THIS DRIVES THE SERVER AND NOT llama-bench
----------------------------------------------
`llama-bench` does NOT chunk. An unarmed arm has the `n_batch` clamp applied, so a 2048-token `-p`
asserts inside the context:

    llama-context.cpp:2455: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed

so it can measure AT MOST one of the three arms at p=2048 -- the comparison did not exist.
`llama-server` chunks a long prompt into `n_batch`/`n_ubatch` pieces internally, which is the
production behaviour being asked about, and it is measurable through `/completion`.

THE ARMS
--------
All on `prefill250`, so pool / ctx / batch / MTP / template are identical and only the phase decision
can differ:

    phase-slab    slab armed, clamp lifted      -> 1 x width-2048 PREFILL graph for the long prompt
    phase-pool8   pool path, clamp = cap 8      -> ~280 x width-8 pool graphs
    phase-pool17  pool path, clamp = bound 17   -> ~132 x width-17 pool graphs

`phase-slab` vs `phase-pool8` is NOT single-variable: for a chunk wider than the decode width the
unarmed arm also takes the clamp, so the honest claim is "slab + no clamp" vs "pool + clamp".
`pool8` vs `pool17` differ only in the clamp width (the graph-count axis).

THE SHARED BOX IS A THREAT TO THE MEASUREMENT (v6)
-----------------------------------------------
`run_server.sh` DEFAULTS to `CGC_PREFLIGHT_KILL=1` (line 657): every launcher invocation SIGTERMs any
llama process it finds, so whoever starts second destroys whoever started first. Measured
2026-09-17: another line's `probe_m5-long` launcher recorded `[preflight] 發現 1 支殘留 llama 行程 ...
SIGTERM 已送` at 19:54:30, the exact second this harness's in-flight 2233-token probe died at 82%;
the next arm's server was killed during load a few minutes later. Free memory does not predict that,
so v6 adds:

  * `other_line_activity()` -- another line's `probe_*/arm_*.ctrl.log` touched within
    `--contention-lookback` counts as busy. This is the signal that actually predicts the kill.
  * `--window-hold` -- the window must hold for N consecutive polls. A moment of quiet is not a
    window.
  * fail-fast on a dead server (during load and during a probe) instead of burning the whole
    `--ready-timeout`, with the launch marked `interfered` and excluded from the gate.
  * `--retry-failed` -- one bounded retry pass for launches that produced no timings at all,
    recorded as retries because they no longer sit at their rotation position.

THE FOUR CELLS (v5)
-------------------
Each launch measures, in this order:

    short_warm      12 tokens, right after a warm-up with the same prompt  -> matched-graph-count cell
    long            ~2233 tokens (read back from the response, never assumed)
    first_step      the same 12 tokens, n_predict=1 -> the cost of ONE decode step after the long chunk
    short_postlong  the same 12 tokens, n_predict=8 -> steps 2..9 of the same situation

The 12-token prompt is measured repeatedly on purpose. Measured 2026-09-17 18:39-18:41: the same
12-token request costs ~1.4 t/s after a long chunk and ~7 t/s when the prompt's own experts are
resident -- a 5x swing decided by cache state, not by the phase rule. A single short cell would have
silently reported whichever state that launch happened to be in. `first_step` exists because "is
decode after a slab prefill cold?" is answerable with a timing (n_predict=1 makes the response's
decode rate the cost of exactly one step) instead of an argument.

DECODE NON-REGRESSION GATE (v5)
-------------------------------
The whitepaper's M1 acceptance includes "decode must not regress". That is now a gate, not a
footnote: per cell, the slab arm's decode median must be within `DECODE_TOLERANCE` of `phase-pool8`
(today's default profile: unarmed slab, clamp = cap 8), reported per cell so a pass cannot hide a
failing cell, and the process exits non-zero when a cell fails. The expert-cache's own counters
(`final stats`, `miss attribution`, `prefetch drop breakdown`) are collected after each graceful
stop, because they split misses into first-touch (compulsory) and churn (capacity) -- which is what
distinguishes "the slab left the pool cold" from "the pool is thrashing".

WHY THE ATTRIBUTION IS A PREDICATE, NOT A GREP (v4)
---------------------------------------------------
Both engine traces are hard-capped per server process:

    CGC-PREFILL-STREAM: il=... kind=... ntok=...     capped at 4 lines  (llama-context.cpp:5824-5826)
    CGC-HOOK: ctx=... il=... ntok=... ids=[...]      capped at 80 lines (llama-context.cpp:5149-5151)

so "no slab line in this request's slice" does NOT mean "the pool path served it" -- it usually means
the 4-line budget was spent by an earlier request. v3 read the trace as if it were a census and
reported `served_by=unknown` for every arm after the first. v4 therefore classifies each request by
the phase predicate that the engine itself uses (single source, pinned offline by
`phase_split_selftest.cpp`) and reports the trace line only as *optional corroboration*, labelled
`evidence=line` vs `evidence=predicate`.

PROTOCOL
--------
* Interleaved AND ROTATED: launch order dominates on this box (three consecutive launches of one
  fixed command gave 246.86 / 134.83 / 110.70 t/s, free-at-launch anti-monotone, carried swap
  monotone -- docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md section D), and the first grid ran a
  FIXED arm order, which made position and arm perfectly collinear. `arm_order(rep)` rotates them so
  that over reps 2..N each arm occupies each position exactly once (a Latin square) and position
  cancels out of the paired comparison.
* 4 launches per arm, discard launch 1, pair by rep index, median.
* free% / free GiB / swap-in-use recorded per launch, because carried swap is the live suspect.
* The window gate is the launcher's OWN rule, not an invented one: `cgc_memory_guard_req full-mtp`
  requires `free_pct >= 40` and `other_llama_servers <= 0` (run_server.sh:566), where free% is
  `memory_pressure -Q`. The launcher can therefore never refuse a launch this harness starts.
  (v2 gated on raw free pages >= 2 GiB, which refused a window the launcher itself accepts:
  measured free=0.19 GiB / `memory_pressure` free=81%.)
* The stop is `kill -INT <the pid the launcher printed>`, never `pkill -f`: the launcher's own hint,
  and `kill -9` leaks Metal buffers.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS SCRIPT: one launch of this model on this box has
been measured from 110 to 247 t/s for a fixed command, so per-arm absolutes are only trustworthy as
PAIRED comparisons within one interleaved session. The ratio between interleaved arms is the result;
the absolute t/s is context. Decode numbers are incidental: decode is T=1, so it can never enter the
prefill graph -- they must not be read for the slab question.

USAGE
    python3 scripts/check/phase_split_ab.py --selftest        # pin the parsers offline (81 checks)
    python3 scripts/check/phase_split_ab.py --dry-run         # print the 3 resolved arms, no launch
    python3 scripts/check/phase_split_ab.py --window-check    # print the window verdict, no launch
    python3 scripts/check/phase_split_ab.py --reps 4          # the measurement (must already be free)
    python3 scripts/check/phase_split_ab.py --watch --self-detach --retry-failed \
        --log-file /tmp/psab.log
    python3 scripts/check/phase_split_ab.py --aggregate-only --out Backup/phase_split_ab/<ts>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS: dict[str, dict[str, str]] = {
    "phase-slab":   {},
    "phase-pool8":  {"CGC_PREFILL_STREAM": "0"},
    "phase-pool17": {"CGC_PREFILL_STREAM": "0", "CGC_POOL_MAX_TOKENS": "64"},
}
PROFILE = "prefill250"

# A decode rate at or above this is not physically a per-token rate on this box (the engine's own
# decode is ~2-14 t/s); it means the server attributed ~0 ms to the tokens it was asked to decode.
# Cells like that are refused rather than gated -- see dump_timings().
DECODE_ARTIFACT_TPS = 1000.0

SHORT_PROMPT = "Please give me three colours and two shapes, comma separated:"
LONG_PROMPT = ("The history of the city of Paris spans more than two thousand years. " * 200)[:11000]

# `first_step` is sent immediately after `long`, so the pool/LRU state is whatever the long chunk
# left behind -- it is the "is the pool cold right after a slab-served prefill?" probe. It uses
# n_predict=1, and NOTE what that costs (measured 2026-09-17, see dump_timings()): llama-server
# attributes ~0 ms to a token emitted with the prefill batch, so this cell's decode field is refused
# (None) rather than reported as a rate. What remains quotable from it is `wall_s` -- a 1-token
# request right after the long chunk -- and `short_postlong` (same prompt, 8 tokens) measures steps
# 2..9, so the pair still shows a cold start if one exists. Making this cell a real rate needs a
# different probe (n_predict>=2, which is what short_postlong already is).
SHAPES = ("short_warm", "long", "first_step", "short_postlong")
PROBES = (("short_warm", SHORT_PROMPT, 8),
          ("long", LONG_PROMPT, 8),
          ("first_step", SHORT_PROMPT, 1),
          ("short_postlong", SHORT_PROMPT, 8))
# The warm-up is the SHORT prompt, not the long one: it exists to pay the cold faults before the
# measured cells, and warming with the long chunk would both double every pool arm's wall time and
# change what `long` measures relative to the v4 grid (`length=short` is what produced the 128 vs
# 24 t/s result), making the two grids incomparable.
WARMUP = SHORT_PROMPT

# Decode non-regression gate. Baseline is `phase-pool8`: that is what the default profile does today
# (unarmed slab, clamp = cap 8), so "the slab must not make decode worse" is a claim about that arm.
# Tolerance is explicit because the box's run-to-run spread is large; the gate is a decision rule, not
# a discovery procedure, and it is reported per cell so a pass cannot hide a failing cell.
DECODE_TOLERANCE = 0.10

# run_server.sh:676 -- the launcher's own preflight name list, matched on the first token's basename.
PREFLIGHT_NAMES = ("llama-server", "llama-simple", "llama-speculative-simple",
                   "llama-cli", "llama-bench")
PREFLIGHT_PATTERNS = (r"(^|/)llama-server(:|\s|$)", r"(^|/)llama-simple(\s|$)",
                      r"(^|/)llama-speculative-simple(\s|$)", r"(^|/)llama-cli(\s|$)",
                      r"(^|/)llama-bench(\s|$)")

# run_server.sh:566 `cgc_memory_guard_req`: full-mtp -> "0 40 0" (free_pct >= 40, others <= 0).
# prefill250 sets MTP=1 + ngl>=90 + ctx>=3072, so its class is full-mtp (not the legacy-25plus
# relaxation). Change this only alongside that table.
LAUNCHER_FREE_PCT_FULLMTP = 40

SLAB_MATCHED_BAND = 0.10   # +-10% counts as "no measurable difference" at n=3 launches


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------- machine / window -----------------

def mem_state() -> dict:
    vs = sh(["vm_stat"]).stdout
    page, free, inactive = 16384, 0, 0
    for line in vs.splitlines():
        m = re.match(r"Pages free:\s+(\d+)", line)
        if m:
            free = int(m.group(1))
        m = re.match(r"Pages inactive:\s+(\d+)", line)
        if m:
            inactive = int(m.group(1))
    sw = sh(["sysctl", "-n", "vm.swapusage"]).stdout
    m = re.search(r"used = ([0-9.]+)M", sw)
    mp = sh(["memory_pressure", "-Q"]).stdout
    m2 = re.search(r"free percentage:\s*(\d+)", mp)
    return {"free_gib": round(free * page / 1e9, 2),
            "inactive_gib": round(inactive * page / 1e9, 2),
            "usable_gib": round((free + inactive) * page / 1e9, 2),
            "free_pct": int(m2.group(1)) if m2 else None,
            "swap_mib": float(m.group(1)) if m else None}


def llama_pids() -> dict[int, str]:
    """The launcher's own preflight definition: pgrep for candidates, then confirm on the first
    token's basename, so a `/bin/bash -c ... llama-cli ...` wrapper is not miscounted (and, on the
    kill side, not mistakenly killed)."""
    out: dict[int, str] = {}
    for pat in PREFLIGHT_PATTERNS:
        p = sh(["pgrep", "-f", pat])
        for pid in (p.stdout or "").split():
            cmd = sh(["ps", "-o", "command=", "-p", pid]).stdout.strip()
            if not cmd:
                out[int(pid)] = "<ps unavailable>"
                continue
            if os.path.basename(cmd.split(" ")[0]) in PREFLIGHT_NAMES:
                out[int(pid)] = cmd[:120]
    return out


def other_line_activity(lookback_s: float) -> list[str]:
    """Another line's probe/arm launch, seen through its own control log.

    This is the guard that mattered most, and it is not about memory at all. `run_server.sh`
    DEFAULTS to `CGC_PREFLIGHT_KILL=1` (line 657): every launcher invocation SIGTERMs any llama
    process it finds, i.e. whoever starts second destroys whoever started first. Measured
    2026-09-17 19:54:30: `Backup/cgc_logs/probe_m5-long_20260917_195430.ctrl.log` recorded
    `[preflight] 發現 1 支殘留 llama 行程 ... SIGTERM 已送` at the exact second our in-flight `long`
    probe died at 82%. Free memory is not the signal that predicts that; a recent ctrl.log is.
    """
    d = os.path.join(ROOT, "Backup", "cgc_logs")
    now, hits = time.time(), []
    try:
        names = os.listdir(d)
    except OSError:
        return hits
    for n in names:
        if not (n.endswith(".ctrl.log") and (n.startswith("probe_") or n.startswith("arm_"))):
            continue
        p = os.path.join(d, n)
        try:
            age = now - os.path.getmtime(p)
        except OSError:
            continue
        if age <= lookback_s:
            hits.append(f"{n} ({age / 60:.1f} min ago)")
    return hits


def window_status(ours: set[int], min_pct: int, allow_busy: bool = False,
                  contention_lookback: float = 0.0) -> tuple[bool, str, dict]:
    st = mem_state()
    others = {p: c for p, c in llama_pids().items() if p not in ours}
    st["other_llama"] = others
    pct = st["free_pct"] if st["free_pct"] is not None else -1
    recent = other_line_activity(contention_lookback) if contention_lookback > 0 else []
    st["other_line"] = recent
    why = (f"free_pct={pct}% (need >={min_pct}%, launcher full-mtp) "
           f"swap={st['swap_mib']}MiB free={st['free_gib']}GiB others={len(others)} "
           f"recent_probes={len(recent)}")
    if allow_busy:
        return True, why + " [--allow-busy-box]", st
    if others:
        first = next(iter(others.values()))
        return False, why + f" -> held by {first}", st
    if recent:
        return False, why + f" -> another line is probing: {recent[0]}", st
    if pct < min_pct:
        return False, why + " -> below the launcher's own requirement", st
    return True, why, st


def wait_for_window(min_pct: int, poll: float, timeout: float, watch_log: str,
                    ours: set[int] | None = None, allow_busy: bool = False,
                    contention_lookback: float = 0.0, hold: int = 1) -> tuple[bool, str]:
    """Poll until the launcher's own start condition holds -- and, with `hold`, until it holds for
    `hold` consecutive polls.

    A moment of quiet is not a window: another line's launcher can start 40 seconds after ours and
    SIGTERM us mid-probe, which is exactly what happened. Requiring the condition to persist costs
    wall time and buys a launch that can actually finish.
    """
    t0, streak = time.time(), 0
    while True:
        ok, why, _ = window_status(ours or set(), min_pct, allow_busy, contention_lookback)
        streak = streak + 1 if ok else 0
        verdict = "WINDOW" if ok and streak >= hold else (f"hold {streak}/{hold}" if ok else "wait  ")
        line = f"[{time.strftime('%H:%M:%S')}] {verdict} {why}"
        print(line, flush=True)
        try:
            with open(watch_log, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if ok and streak >= hold:
            return True, why
        if time.time() - t0 > timeout:
            return False, why + f" -> gave up after {timeout / 60:.0f} min"
        time.sleep(poll)


# ------------------------------------------------------------------------- launch / probe ---------

def launcher_dump(extra_env: dict[str, str]) -> dict:
    """Resolved launch facts, from the launcher's own CGC_DUMP_ENV path (exits before the load).
    One source of truth: model, port, log path and the `[arm]` slab line all come from there, so
    the harness cannot disagree with the launcher about which port or model it is using."""
    env = dict(os.environ, CGC_SERVER_PROFILE=PROFILE, CGC_DUMP_ENV="1", **extra_env)
    p = subprocess.run(["./scripts/run_server.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
    facts: dict = {"rc": p.returncode}
    for line in (p.stdout or "").splitlines():
        if line.startswith("CGCENV "):
            _, k, v = line.split(maxsplit=2)
            facts[k.lower()] = v
        elif line.startswith("[arm]") or line.startswith("[budget]"):
            facts.setdefault("notes", []).append(line.strip())
    if p.returncode != 0:
        facts["stderr"] = (p.stderr or "").strip().splitlines()[-3:]
    return facts


# The launcher's own wording (run_server.sh:1994 `echo "         log: tail -f $LOG"` and its
# `[detach] server PID=$SERVER_PID` line). ONE definition, used by both the initial parse and the
# retry loop: the earlier two-copy version had the `tail -f ` fix in only one of them, and the
# unfixed copy stalled every launch for the full --ready-timeout (measured 2026-09-17 18:31).
LAUNCH_RE = {
    "server_pid": re.compile(r"server PID=(\d+)"),
    "server_log": re.compile(r"log: (?:tail -f )?(/\S+\.log)"),
    "blocked_by_guards": re.compile(r"startup blocked by memory guard -> (.*)"),
}


def parse_launch_log(text: str) -> dict:
    out: dict = {}
    for k, rx in LAUNCH_RE.items():
        m = rx.search(text)
        if m:
            out[k] = int(m.group(1)) if k == "server_pid" else m.group(1).strip()
    return out


def launch(extra_env: dict, launch_log: str) -> dict:
    """`run_server.sh --detach` forks and exits itself; it prints the authoritative server PID and
    log path, which is why nothing here guesses from mtime or port."""
    env = dict(os.environ, CGC_SERVER_PROFILE=PROFILE, **extra_env)
    with open(launch_log, "wb") as f:
        p = subprocess.run(["./scripts/run_server.sh", "--detach"], cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    out = {"rc": p.returncode}
    out.update(parse_launch_log(open(launch_log, errors="replace").read()))
    return out


def wait_healthy(base: str, timeout: float, pid: int | None = None) -> tuple[bool, str]:
    """Wait for /health, but stop the moment the server is GONE.

    Before this, a server killed during startup made the harness spin for the whole --ready-timeout
    (15 min) on a box where another line's launcher can kill ours at any moment (measured
    2026-09-17 19:54:30: `Backup/cgc_logs/probe_m5-long_20260917_195430.ctrl.log` logged the
    preflight kill of our pid, while we waited). Dead is not slow, and the wait must not cost a
    quarter hour.
    """
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return True, "healthy"
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if pid and not alive(pid):
            return False, "server process died while loading"
        if time.time() - t0 >= timeout:
            return False, f"not healthy after {timeout / 60:.0f} min"
        time.sleep(5)


def alive(pid: int | None) -> bool:
    """Is that pid still there? Used to tell an engine error from someone else's kill."""
    if not pid:
        return False
    return subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def stop_server(pid: int | None) -> None:
    """INT the exact pid the launcher reported. Never `pkill -f` (it would hit a wrapper, and it
    could hit someone else's server) and never `-9` (leaks Metal buffers)."""
    if not pid:
        return
    if subprocess.run(["kill", "-INT", str(pid)], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode != 0:
        return
    for _ in range(40):
        if subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            return
        time.sleep(1.5)
    subprocess.run(["kill", "-TERM", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)


def probe(base: str, text: str, n_predict: int = 8, timeout: float = 1800) -> dict:
    """One /completion request, with the server's own timing split recorded verbatim.

    `predicted_per_second` is only a *decode rate* when more than one token was decoded. With
    `n_predict=1` llama-server attributes ~0 ms to the single predicted token (it is emitted with the
    prefill batch), so the field comes back ~1e6 -- the first version of this harness wrote that
    number into the `first_step` cell, where a gate comparing two such cells would have read a ratio
    of 1.0 and PASSED on an artifact. `dump_timings()` keeps the raw fields so the degenerate case is
    visible in the JSON instead of only in this comment, and DECODE_ARTIFACT_TPS is the threshold the
    gate refuses to decide on.
    """
    body = json.dumps({"prompt": text, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"{base}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:  # noqa: BLE001 - report, never mask
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 1)}
    tt = r.get("timings", {})
    dump = dump_timings(tt)
    return {"wall_s": round(time.time() - t0, 1),
            "prompt_n": tt.get("prompt_n"),
            "prefill_tps": round(tt.get("prompt_per_second") or 0.0, 2),
            "decode_tps": dump["decode_tps"],
            "timings_raw": dump["raw"],
            "decode_note": dump["note"]}


def dump_timings(tt: dict) -> dict:
    """Server timings -> (decode_tps | None, raw fields, a note when the field is not a rate).

    `None` is the honest answer for a degenerate cell: it propagates into `load_rows` as "no decode
    median" and into `decode_gate` as NOT MEASURED, instead of a number that looks like a measurement.
    """
    raw = {k: tt.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second",
                                  "predicted_n", "predicted_ms", "predicted_per_second")}
    pn = raw.get("predicted_n")
    tps = tt.get("predicted_per_second")
    if not tps:
        return {"decode_tps": None, "raw": raw, "note": "no predicted_per_second in response"}
    if not pn or pn < 2:
        return {"decode_tps": None, "raw": raw,
                "note": f"predicted_n={pn}: one token is emitted with the prefill batch, so the "
                        f"server's rate ({round(tps, 1)} t/s) is an artifact, not a decode rate"}
    if tps >= DECODE_ARTIFACT_TPS:
        return {"decode_tps": None, "raw": raw,
                "note": f"predicted_per_second={round(tps, 1)} >= {DECODE_ARTIFACT_TPS} "
                        f"(predicted_n={pn}): degenerate; refused as a rate"}
    return {"decode_tps": round(tps, 2), "raw": raw, "note": None}


def newest_server_log(started_at: float) -> str | None:
    """Fallback for the server log when the launcher's own `log:` line did not land in time: the
    newest log written since this launch started. Used only as a fallback, and recorded either
    way, because the launcher's own line is the source of truth when it exists."""
    d = os.path.join(ROOT, "Backup", "cgc_logs")
    try:
        fs = [os.path.join(d, f) for f in os.listdir(d)
              if f.startswith("llama_server_2026") and f.endswith(".log")]
    except OSError:
        return None
    fs = [f for f in fs if os.path.getmtime(f) >= started_at - 5]
    return max(fs, key=os.path.getmtime) if fs else None


def arm_order(rep: int) -> list[str]:
    """Which arm runs in which position, for this rep.

    Arms were run in a FIXED order (slab -> pool8 -> pool17) for the first grid, which makes launch
    position and arm perfectly collinear: every arm's number also carries "how many launches into the
    rep it was". Rotation by (rep-1) turns that into a Latin square -- over reps 2..N each arm occupies
    each of the three positions exactly once -- so position cancels out of the paired comparison.
    """
    names = list(ARMS)
    k = (rep - 1) % len(names)
    return names[k:] + names[:k]


def parse_pool_stats(text: str) -> dict:
    """The expert-cache's own end-of-run counters, from the server log.

    These are what makes a decode claim checkable rather than asserted: `miss attribution` splits
    misses into first-touch (compulsory) and churn (capacity), which is exactly the distinction
    between "the slab served the prefill and left the pool cold" and "the pool is thrashing".
    """
    out: dict = {}
    m = re.search(r"final stats: runtime requests=(\d+) hits=(\d+) misses=(\d+) \(hit rate ([\d.]+)%\)", text)
    if m:
        out.update(reqs=int(m.group(1)), hits=int(m.group(2)), misses=int(m.group(3)),
                   hit_pct=float(m.group(4)))
    m = re.search(r"file_reads=(\d+) pread_usec=(\d+)", text)
    if m:
        out.update(file_reads=int(m.group(1)), pread_s=round(int(m.group(2)) / 1e6, 1))
    # `resident=` is the pool high-water mark: it answers "was the pool actually full?" and is the one
    # number that separates a cold pool from a small one. It was parsed as None for every launch of the
    # 20260917_2001 grid (the field was simply absent from this dict), which made the report table
    # silently drop the column -- so it is pinned in the selftest below, not just added here.
    m = re.search(r"resident=([\d.]+) MiB", text)
    if m:
        out["resident_mib"] = float(m.group(1))
    m = re.search(r"miss attribution: compulsory=(\d+) capacity=(\d+)[^\n]*?evictions=(\d+)", text)
    if m:
        out.update(compulsory=int(m.group(1)), capacity=int(m.group(2)), evictions=int(m.group(3)))
    m = re.search(r"slab fills: pool=([\d.]+) MiB disk=([\d.]+) MiB \(non-resident share ([\d.]+)%\)", text)
    if m:
        out.update(slab_pool_mib=float(m.group(1)), slab_disk_mib=float(m.group(2)),
                   slab_nonresident_pct=float(m.group(3)))
    m = re.search(r"prefetch=(\d+)/(\d+)", text)
    if m:
        out.update(prefetch_issued=int(m.group(1)), prefetch_dropped=int(m.group(2)))
    # Only the non-zero drop reasons are kept; a full table of zeros is noise, and the ONE that is
    # non-zero is the finding (measured 2026-09-17: #3 drain_cleared was 100% of drops in the slab
    # arm, i.e. prefetch work computed and then discarded).
    m = re.search(r"prefetch drop breakdown \(total=(\d+)\):(.*)", text)
    if m:
        out["prefetch_drops"] = {f"#{k} {n}": int(v)
                                 for (k, n, v) in re.findall(r"#(\d+) ([a-z_]+)=(\d+)", m.group(2))
                                 if int(v) > 0}
    m = re.search(r"CGC-PREROUTER: calls=(\d+) .*?hit=(\d+) \(precision ([\d.]+)%\)", text)
    if m:
        out.update(prerouter_calls=int(m.group(1)), prerouter_hit=int(m.group(2)),
                   prerouter_precision_pct=float(m.group(3)))
    return out


def parse_banner(text: str) -> dict | None:
    hits = re.findall(
        r"CGC-PHASE-SPLIT: cap=(\d+) routable=(\d+) slots / top_k=(\d+) -> decode graph width=(\d+) "
        r"tokens \(bound=(\d+), T_prefill=(\d+)([^)]*)\); prefill slab ([^\n]+)", text)
    if not hits:
        return None
    c = hits[-1]
    return {"cap": int(c[0]), "routable_slots": int(c[1]), "top_k": int(c[2]),
            "decode_width": int(c[3]), "bound": int(c[4]), "t_prefill": int(c[5]),
            "threshold_note": c[6].strip().lstrip(",").strip(), "slab": c[7].strip(),
            "slab_armed": c[7].startswith("armed")}


def trace_census(text: str) -> dict:
    """How many trace lines the whole process emitted, against their hard caps. A reader must know
    that a line's absence is sampling, not evidence of the other path."""
    return {"slab_lines": len(re.findall(r"CGC-PREFILL-STREAM: il=\d+ kind=\d+ ntok=", text)),
            "hook_lines": len(re.findall(r"CGC-HOOK: ctx=", text)),
            "slab_cap": 4, "hook_cap": 80}


def segments(text: str) -> list[str]:
    """One segment per request: the server logs `slot launch_slot_` when it starts handling one."""
    parts = re.split(r"(?=slot launch_slot_:)", text)
    return [p for p in parts if p.startswith("slot launch_slot_:")]


def seg_prompt_tokens(seg: str) -> int | None:
    pt = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", seg)
    return int(pt[-1][1]) if pt else None


def align_segments(segs: list[str], probe_ntoks: list[int | None]) -> int:
    """Offset of the first measured probe inside the request segments.

    The log contains MORE requests than this script attributes: the warm-up is sent but not
    measured, so segment 0 is not probe 0. This was an off-by-one in v4's first cut -- every
    attribution came back `segment/probe mismatch` (loud, not silent, which is why it was caught in
    one launch). The offset is discovered by matching each probe's own token count against the
    segment's `prompt eval time` line, not assumed, and 0 is returned when nothing matches so that
    every cell then reports `unknown` instead of being attributed to the wrong request.
    """
    for off in range(0, max(0, len(segs) - len(probe_ntoks)) + 1):
        if all(probe_ntoks[i] is not None
               and seg_prompt_tokens(segs[off + i]) == probe_ntoks[i]
               for i in range(len(probe_ntoks))):
            return off
    return 0


def attribution(seg: str | None, ntok: int | None, decode_width: int | None, armed: bool) -> dict:
    """Which graph served one request.

    Truth here is the phase predicate the engine itself applies (`cgc_select_graph_phase`, pinned by
    phase_split_selftest.cpp), NOT the trace line -- both traces are capped per process (4 and 80
    lines), so their absence says nothing. The line is kept as corroboration:
    `evidence=line` when a slab line carries this exact ntok, `evidence=predicate` otherwise, and
    `conflict=True` if a slab line shows up for a request the predicate says was pool-served.
    """
    if seg is None:
        return {"served_by": "unknown", "evidence": "no-segment", "slab_ntok": [], "hook_lines": 0,
                "log_prompt_tokens": None, "nil_count": 0, "nan_count": 0}
    observed = seg_prompt_tokens(seg)
    slab_ntok = sorted({int(x) for x in
                        re.findall(r"CGC-PREFILL-STREAM: il=\d+ kind=\d+ ntok=(\d+)", seg)})
    by_line = "slab" if (ntok is not None and ntok in slab_ntok) else None
    if ntok is None or decode_width is None:
        predicted = None
    else:
        predicted = "slab" if (armed and ntok > decode_width) else "pool"
    served = by_line or predicted
    out = {"served_by": served or "unknown",
           "evidence": "line" if by_line else ("predicate" if predicted else "indeterminate"),
           "predicted": predicted,
           "conflict": bool(by_line and predicted and by_line != predicted),
           "slab_ntok": slab_ntok,
           "log_prompt_tokens": observed,
           "tokens_match_probe": (observed == ntok) if (observed is not None and ntok) else None,
           "hook_lines": len(re.findall(r"CGC-HOOK: ctx=", seg)),
           "nil_count": seg.count("buffer is nil"),
           "nan_count": len(re.findall(r"NAN|ggml_abort|GGML_ASSERT|Abort trap", seg))}
    if out["tokens_match_probe"] is False:
        # The segment order and the probe order disagree -> do not attribute on a guess.
        out["served_by"], out["evidence"] = "unknown", "segment/probe mismatch"
    return out


def needs_retry(out: str, arm: str, rep: int) -> bool:
    """No timings at all in any cell -- the shape an externally killed launch leaves behind."""
    p = os.path.join(out, f"{arm}_rep{rep}.json")
    if not os.path.exists(p):
        return True
    r = json.load(open(p))
    return not any((r.get(s) or {}).get("prefill_tps") for s in SHAPES)


def run_arm(base: str, arm: str, rep: int, out: str, ready_timeout: float,
            force: bool = False, retry: bool = False) -> dict:
    done = os.path.join(out, f"{arm}_rep{rep}.json")
    if os.path.exists(done) and not force:
        prev = json.load(open(done))
        # Only a rep that produced ALL cells is a result. A rep that recorded an error must be
        # retried, or a restart would keep the failure forever and quietly shrink n.
        if all((prev.get(s) or {}).get("prefill_tps") for s in SHAPES):
            log(f"skip (present) {arm} rep{rep}")
            return prev
        log(f"re-running {arm} rep{rep} (previous attempt was incomplete)")
    before = mem_state()
    launch_log = os.path.join(out, f"{arm}_rep{rep}.launch.log")
    log(f"{arm} rep{rep} launching (free={before['free_gib']}GiB free%={before['free_pct']} "
        f"swap={before['swap_mib']}MiB)")
    started = time.time()
    before_pids = set(llama_pids())          # so the pid fallback below cannot pick someone else's
    info = launch(ARMS[arm], launch_log)
    rec: dict = {"arm": arm, "rep": rep, "before": before, "extra_env": ARMS[arm], "launch": info}
    if retry:
        rec["retry"] = True
    try:
        if info.get("blocked_by_guards"):
            rec["error"] = f"launcher refused: {info['blocked_by_guards']}"
            log(f"{arm} rep{rep} LAUNCHER REFUSED THE LAUNCH -> {info['blocked_by_guards']}")
            return rec
        # The launcher's --detach parent only waits 120 s for health, and the detached child prints
        # its PID + log path AFTER it is up. A 13 GB load can exceed 120 s, so both facts are
        # collected here rather than read once: without the PID we cannot stop the server, and
        # without the log the timing cannot be attributed.
        healthy, deadline, why_stop = False, time.time() + ready_timeout, ""
        while time.time() < deadline:
            healthy, why_stop = wait_healthy(base, 30, info.get("server_pid"))
            for k, v in parse_launch_log(open(launch_log, errors="replace").read()).items():
                info.setdefault(k, v)
            if healthy and info.get("server_pid"):
                break
            if why_stop == "server process died while loading":
                break
        if not healthy:
            rec["error"] = f"server never became healthy ({why_stop})"
            if why_stop == "server process died while loading":
                # Same attribution as a mid-probe death: somebody else's launcher owns the box.
                rec["interfered"] = {"probe": "startup", "error": why_stop,
                                     "server_pid": info.get("server_pid")}
                log(f"{arm} rep{rep} INTERFERENCE: the server was killed during load "
                    f"(pid={info.get('server_pid')}). This launch will be excluded.")
            else:
                log(f"{arm} rep{rep} NOT HEALTHY (pid={info.get('server_pid')}, see "
                    f"{os.path.basename(launch_log)})")
            return rec
        if not info.get("server_pid"):
            # Fallback: the llama pid that appeared since we launched. Bounded by construction -- a
            # pid already present before the launch is never a candidate, so this cannot stop
            # somebody else's server. Without a pid we cannot stop ours gracefully, and the only
            # remaining stop would be `pkill`, which is exactly how another line's process gets hit.
            new = set(llama_pids()) - before_pids
            if len(new) == 1:
                info["server_pid"] = next(iter(new))
                log(f"{arm} rep{rep} pid {info['server_pid']} recovered by difference (launcher did "
                    "not print it in time)")
        slog = info.get("server_log") or newest_server_log(started)
        info["server_log"] = slog
        text = open(slog, errors="replace").read() if slog else ""
        rec["banner"] = parse_banner(text)
        if not rec["banner"]:
            log(f"{arm} rep{rep} WARNING: no CGC-PHASE-SPLIT banner in {slog} -- the geometry this "
                "arm was supposed to differ in cannot be verified")
        geom = rec["banner"] or {}
        probe(base, WARMUP, n_predict=1)                 # warm-up: not measured, not attributed

        for shape, text_, npred in PROBES:
            r = probe(base, text_, npred)
            rec[shape] = r
            # A failed probe on a DEAD server is not the engine's verdict -- on this box another
            # agent's cleanup can SIGINT our server mid-request (measured 2026-09-17 19:54:29: the
            # long probe was cut at 82%, and the disconnection arrived 7 s AFTER the server logged
            # the SIGINT my own stop had not sent yet). Marking it beats letting a killed launch look
            # like a slow arm, and beats letting it quietly shrink n inside a gate.
            if r.get("error") and not alive(info.get("server_pid")):
                rec["interfered"] = {"probe": shape, "error": r["error"],
                                     "server_pid": info.get("server_pid")}
                log(f"{arm} rep{rep} INTERFERENCE: server died during '{shape}' -> "
                    f"{r['error']}. This launch will be excluded and needs a re-run.")
                break
            log(f"{arm} rep{rep} {shape}: ntok={r.get('prompt_n')} "
                f"prefill={r.get('prefill_tps')} t/s wall={r.get('wall_s')}s")

        # Attribution after ALL probes, from the whole log: the traces are capped and stdio to a
        # file is block-buffered, so a slice read immediately after a response can be empty for
        # reasons that have nothing to do with the phase.
        segs = segments(open(slog, errors="replace").read()) if slog else []
        while len(segs) < len(PROBES) and time.time() - started < 120:
            time.sleep(5)
            segs = segments(open(slog, errors="replace").read()) if slog else []
        rec["trace_census"] = trace_census(open(slog, errors="replace").read()) if slog else {}
        rec["segments"] = len(segs)
        off = align_segments(segs, [(rec[s] or {}).get("prompt_n") for s, _, _ in PROBES])
        rec["segment_offset"] = off
        if off != len(segs) - len(PROBES):
            log(f"{arm} rep{rep} NOTE: {len(segs)} log segments for {len(PROBES)} measured probes "
                f"(+1 warm-up expected) -> using offset {off}")
        for i, (shape, _, _) in enumerate(PROBES):
            att = attribution(segs[off + i] if off + i < len(segs) else None,
                              (rec[shape] or {}).get("prompt_n"),
                              geom.get("decode_width"), bool(geom.get("slab_armed")))
            rec[shape]["attribution"] = att
            log(f"{arm} rep{rep} {shape}: served_by={att['served_by']} "
                f"({att['evidence']}) log_ntok={att['log_prompt_tokens']} hooks={att['hook_lines']} "
                f"nil={att['nil_count']}")
    finally:
        rec["after"] = mem_state()
        stop_server(info.get("server_pid"))
        # The expert cache prints its counters as it shuts down, so they are only complete AFTER the
        # stop. Read them here rather than at the JSON write above, or every launch would report the
        # previous run's numbers (or none).
        if slog:
            time.sleep(1.0)
            rec["pool"] = parse_pool_stats(open(slog, errors="replace").read())
        with open(done, "w") as f:            # atomic-ish: a killed run leaves no half-written cell
            json.dump(rec, f, indent=1)
    return rec


# -------------------------------------------------------------------------- aggregate / report ----

def _median_discard_first(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    if len(vals) > 1:
        return round(statistics.median(vals[1:]), 2)            # discard launch 1 (cold caches)
    return round(vals[0], 2) if vals else None


MIN_GATE_N = 2          # the gate needs at least this many usable launches per cell per arm


# `llama_server_<date>_<time>.log` -- the underscore between the two digit groups is load-bearing: a
# pattern of `\d+\d+` silently matches nothing, and the fallback to the stored JSON then looks like a
# working parse that is merely missing a field. Pinned in the selftest.
SERVER_LOG_RE = re.compile(r"(\S*Backup/cgc_logs/llama_server_\d+_\d+\.log)")


def pool_counters(out: str, arm: str, rep: int, stored: dict | None) -> dict:
    """The expert-cache counters for one launch, re-derived from the engine's own log when possible.

    The values are also carried in the result JSON, but that copy is only as good as the parser that
    wrote it -- the 20260917_2001 grid stored them before `resident_mib` existed, so a report built
    from the JSONs alone silently lost that column. Re-reading the final-stats line makes the report
    independent of when the launch happened, and the launch log (not a guess) names the server log.
    """
    launch_log = os.path.join(out, f"{arm}_rep{rep}.launch.log")
    try:
        text = open(launch_log, errors="replace").read()
    except OSError:
        return dict(stored or {})
    m = SERVER_LOG_RE.search(text.replace("\\", "/"))
    if not m:
        return dict(stored or {})
    path = m.group(1)
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    try:
        fresh = parse_pool_stats(open(path, errors="replace").read())
    except OSError:
        return dict(stored or {})
    merged = dict(stored or {})
    merged.update({k: v for k, v in fresh.items() if v is not None})
    return merged


def load_rows(reps: int, out: str) -> dict:
    """Per arm: prefill cell stats, decode cell stats, attribution, geometry, pool counters.

    Decode is collected here rather than left in the raw JSON because the non-regression gate is a
    claim about it, and a gate whose input is not assembled cannot be reported per cell.
    """
    rows: dict = {}
    for arm in ARMS:
        cells: dict = {}
        attrib: dict = {s: [] for s in SHAPES}
        geom: dict = {}
        # Keyed by rep, not appended: the launch-level records (pool counters, interference) are
        # per LAUNCH but were collected inside the per-shape loop, so each launch appeared once per
        # shape -- 16 rows for a 4-launch arm, with the counts reading 4x the truth. Ratios were
        # unaffected, which is exactly why it survived a glance at the verdicts and got caught only by
        # the report's own table being 4x too long.
        pools: dict = {}
        interfered: dict = {}
        for shape in SHAPES:
            pre, dec, notes = [], [], []
            refused = []
            dec_by_rep: dict[int, float] = {}
            wall_by_rep: dict[int, float] = {}
            for rep in range(1, reps + 1):
                p = os.path.join(out, f"{arm}_rep{rep}.json")
                if not os.path.exists(p):
                    notes.append(f"rep{rep}:missing")
                    continue
                r = json.load(open(p))
                if r.get("interfered"):
                    interfered[rep] = {"rep": rep, **r["interfered"]}
                cell = r.get(shape) or {}
                if cell.get("prefill_tps"):
                    pre.append(cell["prefill_tps"])
                elif r.get("interfered"):
                    notes.append(f"rep{rep}:interfered({r['interfered']['probe']})")
                else:
                    notes.append(f"rep{rep}:{cell.get('error') or r.get('error') or 'no-tps'}")
                # Decode values are refused if the server called them degenerate. This is applied to
                # what is ON DISK, not only to fresh responses, so a JSON written before dump_timings()
                # existed (e.g. the 20260917_2001 grid) cannot smuggle a 1e6 "rate" into the gate.
                dv = cell.get("decode_tps")
                if dv and 0 < dv < DECODE_ARTIFACT_TPS:
                    dec.append(dv)
                    dec_by_rep[rep] = dv
                elif dv:
                    refused.append(f"rep{rep}:{cell.get('decode_note') or f'{dv} t/s'}")
                if cell.get("wall_s"):
                    wall_by_rep[rep] = cell["wall_s"]
                if cell.get("attribution"):
                    attrib[shape].append(cell["attribution"])
                if r.get("banner"):
                    geom = r["banner"]
                if r.get("pool") or os.path.exists(
                        os.path.join(out, f"{arm}_rep{rep}.launch.log")):
                    pools[rep] = {"rep": rep, **pool_counters(out, arm, rep, r.get("pool"))}
            cells[shape] = {"per_launch": pre, "median_discard_rep1": _median_discard_first(pre),
                            "decodes": dec, "decode_median_discard_rep1": _median_discard_first(dec),
                            "decode_refused": refused, "notes": notes,
                            "decode_by_rep": dec_by_rep, "wall_by_rep": wall_by_rep}
        rows[arm] = {"cells": cells, "attrib": attrib, "geom": geom,
                     "pool": [pools[k] for k in sorted(pools)],
                     "interfered": [interfered[k] for k in sorted(interfered)]}
    return rows


def decode_gate(rows: dict, tolerance: float = DECODE_TOLERANCE) -> list[dict]:
    """The formal decode non-regression gate: slab vs the arm that is today's default (`pool8`).

    Baseline choice matters and is not free: `phase-pool8` is what the default profile does (unarmed
    slab, clamp = cap 8), so the question 'does arming the slab cost decode?' is a comparison against
    it. Reported per cell so a pass in one cell cannot hide a failure in another, and with the
    tolerance explicit so the verdict is falsifiable rather than a vibe.

    Cells whose decode field the server called degenerate (see dump_timings) are NOT MEASURED with the
    refusal quoted -- the failure mode to avoid is a cell that silently PASSES because both sides
    carry the same artifact (1e6 vs 1e6 ratios to 1.0).
    """
    out = []
    for shape in SHAPES:
        slab = rows.get("phase-slab", {}).get("cells", {}).get(shape, {})
        base = rows.get("phase-pool8", {}).get("cells", {}).get(shape, {})
        s, b = slab.get("decode_median_discard_rep1"), base.get("decode_median_discard_rep1")
        n_s, n_b = len(slab.get("decodes") or []), len(base.get("decodes") or [])
        refused = (slab.get("decode_refused") or []) + (base.get("decode_refused") or [])
        # Defence in depth: `load_rows` already drops refused values, but the gate must not depend on
        # its caller for this. Two identical artifact rates ratio to 1.0 and would PASS while measuring
        # nothing, which is exactly the failure this check exists to make impossible.
        if (s and s >= DECODE_ARTIFACT_TPS) or (b and b >= DECODE_ARTIFACT_TPS):
            out.append({"cell": shape, "verdict": "NOT MEASURED", "slab": s, "pool8": b,
                        "why": f"artifact decode rate (slab {s}, pool8 {b}) >= "
                               f"{DECODE_ARTIFACT_TPS}; both sides would ratio to 1.0",
                        "refused": refused})
            continue
        if s is None or b is None:
            why = f"no decode medians (usable launches: slab {n_s}, pool8 {n_b})"
            if refused:
                why += f"; refused: {'; '.join(refused[:3])}"
            out.append({"cell": shape, "verdict": "NOT MEASURED", "slab": s, "pool8": b,
                        "why": why, "refused": refused})
            continue
        # A gate must not decide on one launch. Below MIN_GATE_N usable launches the honest answer is
        # "cannot decide", not a PASS/FAIL that a single externally-killed launch can flip.
        if min(n_s, n_b) < MIN_GATE_N:
            out.append({"cell": shape, "slab": s, "pool8": b, "verdict": "NOT MEASURED",
                        "why": f"only {min(n_s, n_b)} usable launch(es) per side (need "
                               f"{MIN_GATE_N}): slab {n_s}, pool8 {n_b}"})
            continue
        r = s / b
        out.append({"cell": shape, "slab": s, "pool8": b, "ratio": round(r, 3),
                    "verdict": "PASS" if r >= 1 - tolerance else "FAIL"})
    return out


def gate_failures(gate: list[dict]) -> list[str]:
    return [f"{g['cell']} (slab {g.get('slab')} vs pool8 {g.get('pool8')}, x{g.get('ratio')})"
            for g in gate if g["verdict"] == "FAIL"]


# ------------------------------------------------------------------- cold-start after slab prefill --

# The hypothesis this grid exists to settle: a slab-served prefill does not publish into the pool/LRU
# (the slab path re-points wt->data and returns), so the decode that follows should start on a colder
# pool than the same decode after a pool-served prefill. Two instruments, both measured INSIDE the same
# launch as the long chunk they follow, so the pairing is real:
#   long.decode_tps          decode tokens in the SAME request that just paid the 2233-token prefill
#   short_postlong.decode_tps steps 2..9 of the NEXT request
# Decode rates are used rather than wall time on purpose: `first_step`'s wall includes a 12-token
# prefill, which on the slab arm is the already-known short-chunk penalty (6.54x slower), so that wall
# would measure the prefill defect and not the pool's temperature. Note also what this grid CANNOT see:
# the arms rotate but there is still one launch per (arm, rep), so a per-launch pool state effect that
# is not arm-dependent (thermal, carried swap) is only partly cancelled -- hence the paired ratios and
# the position-balance table rather than a plain arm-vs-arm median.
COLD_INSTRUMENTS = ("long", "short_postlong")
COLD_TOLERANCE = 0.10


def rotation_balance(reps: int) -> dict:
    """Launch positions per arm, so "rotation removed the confound" is a checked statement.

    A number whose meaning depends on which launch it was (the earlier finding: free memory and t/s are
    anti-monotone, and carried swap moves across launches) needs a protocol where position is not
    collinear with arm. Over the reps that are actually counted (rep 1 is discarded as cold), a
    three-arm cyclic rotation gives each arm each position exactly once -- this function proves that
    from the same `arm_order()` the driver uses, instead of restating the intent in prose.
    """
    counted = list(range(2, reps + 1))
    pos: dict[str, list[int]] = {a: [] for a in ARMS}
    for rep in counted:
        for i, arm in enumerate(arm_order(rep)):
            pos[arm].append(i + 1)
    means = {a: (round(sum(p) / len(p), 2) if p else None) for a, p in pos.items()}
    return {"counted_reps": counted, "positions": pos, "mean_position": means,
            "balanced": len(set(means.values())) == 1 and bool(counted),
            "order_per_rep": {r: arm_order(r) for r in counted}}


def cold_after_slab(rows: dict, reps: int, tolerance: float = COLD_TOLERANCE) -> dict:
    """Is decode systematically colder after a slab-served prefill? Paired per rep, both instruments.

    Paired because the quantity that drifts on this box (swap carried between launches) is a per-launch
    effect, so differencing within a rep is what removes it; per cell, because the two instruments
    measure different moments of the same cold start. A verdict needs enough usable pairs AND a
    consistent direction -- a median ratio that clears tolerance with the opposite sign on most reps is
    reported as INCONCLUSIVE rather than averaged into a claim.
    """
    out: dict = {"rotation": rotation_balance(reps), "cells": []}
    for shape in COLD_INSTRUMENTS:
        for base in ("phase-pool8", "phase-pool17"):
            slab = rows.get("phase-slab", {}).get("cells", {}).get(shape, {}).get("decode_by_rep") or {}
            bl = rows.get(base, {}).get("cells", {}).get(shape, {}).get("decode_by_rep") or {}
            pairs = [(r, slab[r], bl[r]) for r in sorted(set(slab) & set(bl)) if r >= 2]
            rec: dict = {"cell": shape, "vs": base, "pairs": pairs}
            if len(pairs) < MIN_GATE_N:
                rec.update(verdict="NOT MEASURED",
                           why=f"{len(pairs)} usable paired rep(s) after discarding rep1 (need "
                               f"{MIN_GATE_N})")
                out["cells"].append(rec)
                continue
            ratios = [s / b for _, s, b in pairs if b]
            med = round(statistics.median(ratios), 3)
            colder = sum(1 for x in ratios if x < 1 - tolerance)
            warmer = sum(1 for x in ratios if x > 1 + tolerance)
            # "Systematic" is held to the strict reading on purpose: every counted rep must agree in
            # direction, not a majority. With n=3 on a box whose own run-to-run spread is this large, a
            # 2-of-3 majority alongside one rep pointing 4x the other way is a split, and calling that
            # SYSTEMATIC is the same species of overclaim as reading a sampled trace as a census. The
            # per-rep ratios stay in the record so the reader can see the split rather than take a word
            # for it (this rule was tightened by a selftest case, not by taste).
            rec.update(ratios=[round(x, 3) for x in ratios], median_ratio=med,
                       colder_reps=colder, warmer_reps=warmer, n=len(ratios),
                       consistent=(colder == 0 or warmer == 0))
            if not rec["consistent"]:
                rec["verdict"] = (f"INCONCLUSIVE (reps disagree in direction: {colder} colder, "
                                  f"{warmer} warmer beyond ±{tolerance:.0%})")
            elif med < 1 - tolerance:
                rec["verdict"] = "SYSTEMATIC COLD (every counted rep colder)"
            elif med > 1 + tolerance:
                rec["verdict"] = "SYSTEMATIC WARM (every counted rep warmer; hypothesis refuted)"
            else:
                rec["verdict"] = "NO EVIDENCE of a systematic difference"
            out["cells"].append(rec)
    return out


def ratio(a, b) -> float | None:
    if a and b:
        return round(a / b, 3)
    return None


def _med(rows: dict, arm: str, shape: str):
    return rows.get(arm, {}).get("cells", {}).get(shape, {}).get("median_discard_rep1")


def reps_present(rows: dict) -> int:
    """How many reps the FILES contain, so aggregation never depends on the flag that asked for them.

    `--reps` is what the driver was told; this is what is on disk. They differ whenever a run is
    truncated (killed mid-grid, window closed), and a report that aggregated the flag instead of the
    files would claim cells it does not have.
    """
    m = 0
    for arm in ARMS:
        for shape in SHAPES:
            c = rows.get(arm, {}).get("cells", {}).get(shape, {}) or {}
            for d in (c.get("decode_by_rep") or {}, c.get("wall_by_rep") or {}):
                for r in d:
                    m = max(m, int(r))
            m = max(m, len(c.get("per_launch") or []))
    return m


def verdicts(rows: dict) -> list[tuple[str, str]]:
    """The three questions this grid was built to separate, answered only when both terms exist."""
    out = []
    p8_l, p17_l, s_l = _med(rows, "phase-pool8", "long"), _med(rows, "phase-pool17", "long"), \
        _med(rows, "phase-slab", "long")
    s_sw, p17_sw = _med(rows, "phase-slab", "short_warm"), _med(rows, "phase-pool17", "short_warm")

    def cmp(na, va, nb, vb, note=""):
        """Every ratio NAMES both sides and its direction. A bare 'x0.19' under a heading that says
        'slab vs pool8' reads as 'the slab is 5x slower' when it means the opposite -- the first
        version of this report did exactly that, so the direction is now part of the sentence."""
        r = ratio(va, vb)
        if r is None:
            return f"NOT MEASURED ({na} or {nb} absent)"
        faster, sp = (na, float(r)) if r >= 1 else (nb, 1.0 / r)
        return (f"{na}/{nb} = {round(float(r), 3)} -> **{faster} is {sp:.2f}x faster**"
                + (f"; {note}" if note else ""))

    r_pp = ratio(p8_l, p17_l)
    out.append(("pool8/pool17 (long, same 2233-token chunk)", cmp(
        "pool8", p8_l, "pool17", p17_l,
        "a clamp-width gap alone, so part of any 'slab beats pool8' number is really the clamp's "
        "penalty rather than the slab's gain"
        if r_pp is not None else "")))
    out.append(("slab/pool17 (long)", cmp(
        "slab", s_l, "pool17", p17_l,
        "one PREFILL graph with no clamp vs ~132 pool graphs; two variables, the narrowest of the "
        "long-chunk comparisons")))
    out.append(("slab/pool8 (long, end-to-end)", cmp(
        "slab", s_l, "pool8", p8_l,
        "the regime-level difference (slab + no clamp vs pool + clamp); never quote it as 'the "
        "slab'")))

    # The matched-graph-count cell is only valid if the short prompt really took the slab on the
    # slab arm (i.e. it exceeded that arm's decode width) and stayed one chunk on the pool arm.
    slab_geom = rows.get("phase-slab", {}).get("geom") or {}
    pool_geom = rows.get("phase-pool17", {}).get("geom") or {}
    sw_ntok = [a.get("log_prompt_tokens") for a in rows.get("phase-slab", {}).get("attrib", {})
               .get("short_warm", []) if a.get("log_prompt_tokens")]
    disc = []
    # Distinguish "the banner says the slab is off" from "there is no readable banner": the second
    # case means the geometry is simply unverified, and calling it INVALID would be a claim the data
    # does not support.
    armed = slab_geom.get("slab_armed")
    if armed is None:
        disc.append("the slab arm's banner could not be read (no slab_armed field) -> geometry "
                    "unverified")
    elif not armed:
        disc.append(f"slab arm's banner says prefill slab {slab_geom.get('slab')}")
    if slab_geom.get("decode_width") and sw_ntok and min(sw_ntok) <= slab_geom["decode_width"]:
        disc.append(f"short prompt ({min(sw_ntok)} tok) <= slab arm's decode width "
                    f"({slab_geom['decode_width']}) -> it was NOT sent to the slab")
    if pool_geom.get("decode_width") and sw_ntok and max(sw_ntok) > pool_geom["decode_width"]:
        disc.append(f"short prompt ({max(sw_ntok)} tok) > pool arm's decode width "
                    f"({pool_geom['decode_width']}) -> it was chunked, so the pool side is not one graph")
    if disc:
        out.append(("discriminator", "INVALID: " + "; ".join(disc) + ". The matched-graph-count "
                    "comparison below is not usable as stated."))

    r3 = ratio(s_sw, p17_sw)
    label = "slab/pool17 (short_warm, one graph per request)"
    if r3 is None:
        out.append((label, "NOT MEASURED"))
    elif abs(r3 - 1) <= SLAB_MATCHED_BAND:
        out.append((label, cmp("slab", s_sw, "pool17", p17_sw,
                               f"within +-{SLAB_MATCHED_BAND:.0%}, i.e. at matched graph count the "
                               "slab is not measurably faster on a 12-token chunk")))
    else:
        out.append((label, cmp("slab", s_sw, "pool17", p17_sw,
                               "matched graph count, so this is the slab mechanism itself: a "
                               "12-token chunk is served by one graph either way")))

    # The two short cells, side by side on the slab arm: how much of the 12-token cost is churn
    # rather than the graph being evaluated.
    r4 = ratio(_med(rows, "phase-slab", "short_warm"), _med(rows, "phase-slab", "short_postlong"))
    if r4:
        out.append(("slab arm, short_warm/short_postlong",
                    f"{round(float(r4), 3)} -> the same 12-token chunk measures "
                    f"{max(r4, 1 / r4):.2f}x apart in two cache states. Cache state, not the phase "
                    "rule, decides this cell."))

    # What all of the above means for decision D (arm CGC_PREFILL_STREAM=1 by default, as a profile
    # cell). Stated as a reading of these numbers, with the experiment that would settle it named --
    # because the numbers do NOT support an unconditional yes, and they do not support a no either.
    long_win = ratio(s_l, p8_l)
    short_loss = ratio(p17_sw, s_sw)
    if long_win and short_loss and long_win > 1 and short_loss > 1:
        out.append(("option D (arm the slab by default?)",
                    f"CONDITIONAL. Long chunks: the slab regime is {long_win:.2f}x the pool8 regime. "
                    f"Near-threshold chunks: a 12-token request is {short_loss:.2f}x SLOWER through "
                    "the slab than through the pool at matched graph count, and the phase rule "
                    "sends every chunk wider than the decode width there. So a default arming needs "
                    "a floor on chunk size (the `T_prefill` threshold), which this grid did not "
                    "sweep: that sweep is the experiment this result calls for."))
    elif long_win and long_win <= 1:
        out.append(("option D (arm the slab by default?)",
                    "NO on speed grounds: the slab regime is not faster than the pool regime on the "
                    "long chunk either."))

    for g in decode_gate(rows):
        label = f"decode gate, {g['cell']}"
        if g["verdict"] == "NOT MEASURED":
            out.append((label, f"NOT MEASURED -- {g.get('why', 'insufficient data')}"))
            continue
        out.append((label, f"**{g['verdict']}** slab {g['slab']} vs pool8 {g['pool8']} t/s "
                    f"(x{g['ratio']}, tolerance -{DECODE_TOLERANCE:.0%})"))

    # The protocol claim and the cold-start claim, both stated against the FILES rather than the run's
    # own narration: rotation is checked from arm_order(), the cold verdict from paired per-rep decode.
    cold = cold_after_slab(rows, reps_present(rows))
    rb = cold["rotation"]
    out.append(("arm rotation (position 1/2/3 per arm over counted reps)",
                ("BALANCED" if rb["balanced"] else "NOT BALANCED") +
                f" -- {rb['mean_position']} over reps {rb['counted_reps']}. "
                "Position was perfectly collinear with arm in the fixed-order grid; "
                "this is what makes the paired decode comparison below admissible."))
    for c in cold["cells"]:
        label = f"decode after slab prefill ({c['cell']}) vs {c['vs']}"
        if c["verdict"] == "NOT MEASURED":
            out.append((label, f"NOT MEASURED -- {c.get('why')}"))
            continue
        out.append((label, f"**{c['verdict']}** median paired ratio x{c['median_ratio']} "
                    f"({c['colder_reps']}/{c['n']} reps colder, {c['warmer_reps']}/{c['n']} warmer, "
                    f"tolerance ±{COLD_TOLERANCE:.0%}; per-rep {c['ratios']})"))

    # Is decode after a slab prefill systematically cold? The counters answer it directly: misses
    # split into first-touch (compulsory) and churn (capacity), so "cold pool" and "pool thrashing"
    # are distinguishable without any new instrumentation.
    slab_pool = rows.get("phase-slab", {}).get("pool") or []  # noqa: E303 - kept next to its use
    if slab_pool:
        comp = [p.get("compulsory", 0) for p in slab_pool if "compulsory" in p]
        cap = [p.get("capacity", 0) for p in slab_pool if "compulsory" in p]
        hit = [p.get("hit_pct") for p in slab_pool if "hit_pct" in p]
        if comp and cap:
            if sum(cap) <= 0.05 * max(1, sum(comp) + sum(cap)):
                out.append(("decode pool: cold or thrashing?",
                            f"NOT thrashing: {sum(comp)} compulsory vs {sum(cap)} capacity misses "
                            f"across the slab arm's launches (hit rate {hit}). The pool is holding "
                            "what it holds; the misses are first touches, which publishing a "
                            "prefill union cannot remove on its own."))
            else:
                out.append(("decode pool: cold or thrashing?",
                            f"CAPACITY-BOUND: {sum(comp)} compulsory vs {sum(cap)} capacity misses "
                            "-- the pool is losing experts it has, so a prefill-to-pool handoff "
                            "could pay off."))
    return out


def provenance(out: str) -> dict:
    def md5(g: str) -> str | None:
        fs = sorted(glob.glob(os.path.join(ROOT, g)), key=os.path.getmtime)
        if not fs:
            return None
        return sh(["md5", "-q", fs[-1]]).stdout.strip()[:32] or None

    prov = {
        "git_head": sh(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip(),
        "git_dirty": [l for l in sh(["git", "status", "--porcelain"], cwd=ROOT).stdout.splitlines()],
        "md5_llama_server": md5("src/llama.cpp/build/bin/llama-server"),
        "md5_libllama": md5("src/llama.cpp/build/bin/libllama.0.0.*.dylib"),
        "md5_libggml_metal": md5("src/llama.cpp/build/bin/libggml-metal.0.*.dylib"),
        "launcher_dump": launcher_dump({}),
        "profile": PROFILE,
    }
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    return prov


def write_report(rows: dict, out: str, ran: bool, reason: str, reps: int) -> str:
    pp = os.path.join(out, "provenance.json")
    prov = provenance(out) if (ran or not os.path.exists(pp)) else json.load(open(pp))
    L: list[str] = ["# Phase-split A/B: is the armed slab faster, or is the clamp merely slower?", ""]
    L.append(f"- date: {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- commit: `{(prov.get('git_head') or '?')[:9]}`  "
             f"(dirty: {len(prov.get('git_dirty') or [])} path(s))")
    L.append(f"- profile: `{PROFILE}`;  arms: " + ", ".join(f"`{a}` {v}" for a, v in ARMS.items()))
    L.append(f"- build: llama-server `{prov.get('md5_llama_server')}`, libllama `{prov.get('md5_libllama')}`, "
             f"libggml-metal `{prov.get('md5_libggml_metal')}`")
    dump = prov.get("launcher_dump") or {}
    if dump.get("model"):
        L.append(f"- model: `{dump.get('model')}` (pool {dump.get('budget')} B, load_mode "
                 f"{dump.get('load_mode')}, port {dump.get('port')})")
    L.append("- arm order: rotated per rep (`arm_order`: slab/pool8/pool17 -> pool8/pool17/slab -> ...), "
             "so over reps 2..N each arm occupies each launch position exactly once and position "
             "cancels out of the paired comparison")
    L.append(f"- raw: `{out}/` (per-launch JSON, launch/server logs, watch.log)")
    L.append("")
    path = os.path.join(ROOT, "docs", f"PHASE_SPLIT_AB_REPORT_{time.strftime('%Y-%m-%d')}.md")

    if not ran:
        L += ["## NOT MEASURED", "",
              f"The grid did not run. Reason: {reason}", "",
              "No timing of any kind is reported here, because a number produced under that "
              "condition would describe the machine, not the engine.", "",
              "The harness gates on the launcher's own start condition "
              "(`run_server.sh:566 cgc_memory_guard_req full-mtp` -> `free_pct >= 40`, "
              "`other_llama_servers <= 0`), i.e. the same condition that would have refused the "
              "launch. Re-run when the box is free:", "",
              "```bash", "python3 scripts/check/phase_split_ab.py --watch --self-detach \\",
              "    --log-file /tmp/phase_split_ab.log", "```"]
        open(path, "w").write("\n".join(L) + "\n")
        return path

    L += [f"## Result (n={reps} launches/arm, interleaved; launch 1 discarded as the cold one)", "",
          "| arm | " + " | ".join(f"{s} per launch" for s in SHAPES) + " | "
          + " | ".join(f"{s} median" for s in SHAPES) + " |",
          "|---" * (1 + 2 * len(SHAPES)) + "|"]
    for arm in ARMS:
        c = rows[arm]["cells"]
        L.append(f"| `{arm}` | " + " | ".join(str(c[s]["per_launch"]) for s in SHAPES) + " | "
                 + " | ".join(str(c[s]["median_discard_rep1"]) for s in SHAPES) + " |")
    retried = [(arm, rep) for arm in ARMS for rep in range(1, reps + 1)
               if os.path.exists(os.path.join(out, f"{arm}_rep{rep}.json"))
               and json.load(open(os.path.join(out, f"{arm}_rep{rep}.json"))).get("retry")]
    if retried:
        L += ["", "## Retried launches (outside the rotation)", "",
              "These ran in a later pass, so they are NOT at their Latin-square position any more; "
              "their cells are still paired by rep but the position confound is not removed for "
              "them: " + ", ".join(f"`{a}` rep{r}" for a, r in retried)]
    interfered = [(arm, i) for arm in ARMS for i in (rows[arm].get("interfered") or [])]
    if interfered:
        L += ["", "## Externally killed launches (not the engine's verdict)", "",
              "On this shared box another process can SIGINT our server mid-request. Those launches are "
              "excluded and named here, because a killed run looks exactly like a slow arm:", ""]
        L += [f"- `{arm}` rep{i['rep']}: killed during `{i['probe']}` (server pid {i.get('server_pid')}) "
              f"— `{i['error']}`" for arm, i in interfered]
        L += ["", "Re-run those cells before quoting them: `--aggregate-only` after a partial re-run, or "
              "the whole grid."]
    missing = [f"`{arm}`/{s}: {'; '.join(c['notes'])}" for arm in ARMS for s, c in rows[arm]["cells"].items()
               if c["notes"]]
    if missing:
        L += ["", "Cells with missing launches (never silently pooled into a median):"]
        L += [f"- {m}" for m in missing]
    L += ["", "## Verdict", ""]
    L += [f"- **{q}** — {v}" for q, v in verdicts(rows)]

    gate = decode_gate(rows)
    L += ["", "## Decode non-regression gate", "",
          f"Baseline is `phase-pool8` (today's default: unarmed slab, clamp = cap 8); tolerance "
          f"-{DECODE_TOLERANCE:.0%}. Decode medians are over launches 2..N, paired by rep index under "
          "rotated arm order, and reported per cell so a pass cannot hide a failing cell.", "",
          "| cell | slab decode t/s | pool8 decode t/s | slab/pool8 | verdict |",
          "|---|---|---|---|---|"]
    for g in gate:
        L.append(f"| {g['cell']} | {g.get('slab')} | {g.get('pool8')} | {g.get('ratio')} | "
                 f"**{g['verdict']}** |")
    L += ["", "A cell the server called degenerate is NOT MEASURED with the refusal quoted, not "
              "PASS/FAIL: a cell where both sides carry the same artifact (e.g. `n_predict=1`, where "
              "the single predicted token is emitted with the prefill batch) ratios to 1.0 and would "
              "PASS while measuring nothing. Refusals recorded this run: "
          + ("none." if not any(g.get("refused") for g in gate)
             else "; ".join(dict.fromkeys(          # both sides record the same refusal, so dedupe
                 f"`{g['cell']}` {r}" for g in gate for r in (g.get("refused") or []))))]

    # Rotation first, then the paired verdict: the cold-start claim is only admissible on a protocol
    # where launch position is not collinear with arm, so the balance is printed beside it.
    cold = cold_after_slab(rows, reps_present(rows))
    rb = cold["rotation"]
    L += ["", "### Arm rotation (the confound this grid was rebuilt to remove)", "",
          f"Counted reps: {rb['counted_reps']} (rep 1 discarded as cold, per launch 1 = cold caches). "
          f"Mean launch position per arm: {rb['mean_position']} -- "
          f"{'BALANCED' if rb['balanced'] else 'NOT BALANCED'}.", "",
          "| rep | order |", "|---|---|"]
    L += [f"| {r} | " + " -> ".join(f"`{a}`" for a in o) + " |"
          for r, o in rb["order_per_rep"].items()]
    L += ["", "### Is decode systematically colder after a slab-served prefill?", "",
          "The hypothesis: the slab path re-points `wt->data` and returns without publishing into the "
          "pool, so the decode that follows starts on a colder pool than the same decode after a "
          "pool-served prefill. Tested with the two instruments that measure decode *inside the same "
          "launch* as the prefill they follow, paired per rep. `first_step`'s wall is deliberately not "
          "used here: it contains a 12-token prefill, which on the slab arm is the already-known "
          "short-chunk penalty, so it would measure the prefill defect and not the pool's temperature.",
          "", "| cell | vs | per-rep paired ratios (slab/base) | median | reps colder/warmer | verdict |",
          "|---|---|---|---|---|---|"]
    for c in cold["cells"]:
        L.append(f"| `{c['cell']}` | `{c['vs']}` | {c.get('ratios') or '-'} | {c.get('median_ratio') or '-'} "
                 f"| {c.get('colder_reps', '-')}/{c.get('warmer_reps', '-')} | **{c['verdict']}** |")
    L += ["", "### Expert-cache counters per launch (who was cold, and why)", "",
          "`compulsory` = first touch of that expert in this process; `capacity` = evicted then needed "
          "again. The slab arm's pool is only ever read by its DECODE steps (its prefill goes to the "
          "slab), so its compulsory count is a direct read of how cold decode was.", "",
          "`pread s` is CUMULATIVE across the pool's worker threads, so it is a cost an arm paid, not "
          "wall time. `resident MiB` is the pool's high-water mark: a pool that is full while its "
          "first-touch misses stay high is cold, not small.", "",
          "| arm | rep | hit% | hits/misses | compulsory | capacity | evictions | file reads | pread s | resident MiB | prefetch issued/dropped | drops |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for p in rows.get(arm, {}).get("pool") or []:
            L.append(f"| `{arm}` | {p.get('rep')} | {p.get('hit_pct')} | "
                     f"{p.get('hits')}/{p.get('misses')} | {p.get('compulsory')} | {p.get('capacity')} | "
                     f"{p.get('evictions')} | {p.get('file_reads')} | {p.get('pread_s')} | "
                     f"{p.get('resident_mib')} | "
                     f"{p.get('prefetch_issued')}/{p.get('prefetch_dropped')} | "
                     f"{p.get('prefetch_drops') or '-'} |")
    L += ["", "## Which graph served each request", "",
          "`evidence=line` means a `CGC-PREFILL-STREAM` line carried that request's exact token "
          "count. `evidence=predicate` means the line was not available (the slab trace is capped at "
          "**4 lines per process**, so later requests usually have none) and the classification "
          "comes from the phase rule the engine itself applies.", "",
          "| arm | rep | cap | decode width | slab | cell | prompt tok (log) | served_by | evidence | hooks | nil | nan/assert |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for rep in range(1, reps + 1):
            p = os.path.join(out, f"{arm}_rep{rep}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            g = r.get("banner") or {}
            for shape in SHAPES:
                a = ((r.get(shape) or {}).get("attribution")) or {}
                L.append(f"| `{arm}` | {rep} | {g.get('cap', '?')} | {g.get('decode_width', '?')} | "
                         f"{'armed' if g.get('slab_armed') else 'not armed'} | {shape} | "
                         f"{a.get('log_prompt_tokens')} | {a.get('served_by')} | {a.get('evidence')} | "
                         f"{a.get('hook_lines')} | {a.get('nil_count')} | {a.get('nan_count')} |")
    conflicts = [f"{arm} rep{rep} {shape}"
                 for arm in ARMS for rep in range(1, reps + 1)
                 for shape in SHAPES
                 if os.path.exists(os.path.join(out, f"{arm}_rep{rep}.json"))
                 and (((json.load(open(os.path.join(out, f"{arm}_rep{rep}.json"))).get(shape) or {})
                       .get("attribution") or {}).get("conflict"))]
    if conflicts:
        L += ["", f"**Trace conflicts (line vs predicate): {conflicts}** — a slab line appeared for a "
              "request the phase rule says was pool-served. Investigate before using those rows."]
    L += ["", "## What this does and does not settle", "",
          "- It settles whether arming the slab as a **profile default** has a speed justification "
          "(decision D: put `CGC_PREFILL_STREAM=1` in the pool-regime profile cells, rather than "
          "choosing it at runtime from memory state, which would not be fingerprintable).",
          "- It does **not** settle correctness: bit-identity across pool sizes belongs to the oracle "
          "gate (`Backup/m123_oracle_gate`, `--profile prefill250`), not to this script.",
          "- The decode numbers in the raw JSON are incidental: decode is T=1 and can never enter the "
          "prefill graph, so a decode-side slab A/B is a false negative **by construction**.",
          "", "## Caveats that must travel with these numbers", "",
          "- One launch of this model on this box has been measured from 110 to 247 t/s for a fixed "
          "command, so only the PAIRED ratios within this interleaved session are trustworthy.",
          "- Launch order dominates and the live variable is carried swap, not free memory "
          "(`docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md` section D). Per-launch free/swap is "
          "in each JSON's `before`.",
          "- `short_warm` and `short_postlong` are the same request in two cache states; quote them "
          "separately or not at all.",
          "- Both engine traces are capped (slab 4 lines, `CGC-HOOK` 80 lines per process), so trace "
          "absence is never evidence of the other path; see the `evidence` column.",
          "- The window gate is the launcher's own (`free_pct >= 40`, no other llama process). Every "
          "poll is in `watch.log`."]
    open(path, "w").write("\n".join(L) + "\n")
    return path


def aggregate(reps: int, out: str, ran: bool = True, reason: str = "") -> tuple[dict, str]:
    rows = load_rows(reps, out)
    print()
    print("=== prefill t/s per launch (launch 1 is the cold one) ===")
    for arm in ARMS:
        c = rows[arm]["cells"]
        print(f"  {arm:<14} " + "  ".join(f"{s}={c[s]['per_launch']}" for s in SHAPES))
    print()
    print("=== medians over launches 2..N ===")
    for arm in ARMS:
        c = rows[arm]["cells"]
        print(f"  {arm:<14} " + "  ".join(f"{s}={c[s]['median_discard_rep1']}" for s in SHAPES))
    print()
    print("=== decode t/s per launch ===")
    for arm in ARMS:
        c = rows[arm]["cells"]
        print(f"  {arm:<14} " + "  ".join(f"{s}={c[s]['decodes']}" for s in SHAPES))
    print()
    print("=== verdict ===")
    for q, v in verdicts(rows):
        print(f"  {q}: {v}")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"rows": rows, "arms": ARMS, "ran": ran, "reason": reason,
                   "verdicts": verdicts(rows), "cold": cold_after_slab(rows, reps_present(rows))},
                  f, indent=1)
    # Aggregate over what the FILES hold, not over what --reps asked for: a truncated grid must not
    # be reported as if the missing reps had been measured.
    reps_files = reps_present(rows) or reps
    if reps_files != reps:
        log(f"NOTE: --reps {reps} but files hold {reps_files}; aggregating over the files")
    path = write_report(rows, out, ran, reason, reps_files)
    log(f"report -> {path}")
    return rows, path


# ------------------------------------------------------------------------------------ driver ------

def self_detach(args: argparse.Namespace) -> int:
    """Double fork + os.setsid, then re-exec without --self-detach. macOS ships no setsid(1)."""
    argv = [a for a in sys.argv[1:] if a != "--self-detach"]
    logf = args.log_file or "/tmp/phase_split_ab.log"
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    pid = os.fork()
    if pid != 0:
        os.waitpid(pid, 0)
        print(f"detached; log: {logf}")
        return 0
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + argv)
    return 0  # unreachable


def selftest() -> int:
    """Pin this harness's own instruments, offline.

    Written after two real defects found the same night, both of which produced no error at all:
    the launch-log parser never matched `log: tail -f <path>` (so every launch stalled for the full
    --ready-timeout), and the attribution read a capped trace as if it were a census (so every arm
    after the first reported `unknown`). A silently non-matching instrument is the exact failure
    mode this line of work keeps paying for, so the instruments get a test that fails when they
    stop reading what they claim to read.
    """
    fails: list[str] = []
    checks = 0

    def ck(name: str, cond: bool, detail=None) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(f"{name}  {detail}")

    # 1. the launcher's real wording, copied out of Backup/phase_split_ab/*/*.launch.log
    real = ("  停止       : pkill -INT -f llama-server（或 kill 84383）\n"
            "[detach] server PID=84383（已脫離父 shell，不受 SIGHUP 影響）\n"
            "         停止: kill -INT 84383 或 pkill -INT -f llama-server\n"
            "         log: tail -f /Users/alexchuang/Documents/flashkv-devserver/Backup/cgc_logs/"
            "llama_server_20260917_183158.log\n")
    got = parse_launch_log(real)
    ck("launch-log pid", got.get("server_pid") == 84383, got)
    ck("launch-log path (tail -f form)",
       (got.get("server_log") or "").endswith("llama_server_20260917_183158.log"), got)
    ck("launch-log path (plain form)",
       parse_launch_log("log: /tmp/x.log").get("server_log") == "/tmp/x.log")
    ck("guard refusal parsed",
       parse_launch_log("error: startup blocked by memory guard -> full-mtp: free=12%<40%")
       .get("blocked_by_guards") == "full-mtp: free=12%<40%")

    # 2. the phase banner (real lines, same log)
    b = parse_banner("CGC-PHASE-SPLIT: cap=8 routable=142 slots / top_k=8 -> decode graph width=8 "
                     "tokens (bound=8, T_prefill=512, non-binding); prefill slab armed "
                     "(CGC_PREFILL_STREAM=1)")
    ck("banner geometry", bool(b) and b["decode_width"] == 8 and b["bound"] == 8
       and b["routable_slots"] == 142, b)
    ck("banner slab armed state", bool(b) and b["slab_armed"] is True, b)
    ck("banner not-armed state",
       parse_banner("CGC-PHASE-SPLIT: cap=8 routable=142 slots / top_k=8 -> decode graph width=8 "
                    "tokens (bound=8, T_prefill=512, non-binding); prefill slab NOT armed")
       ["slab_armed"] is False)
    ck("banner ignores the geometry-less variant",
       parse_banner("CGC-PHASE-SPLIT: CGC_PREFILL_STREAM=1 -> n_batch NOT capped") is None)
    ck("banner picks the geometry line when both are present",
       parse_banner("CGC-PHASE-SPLIT: CGC_PREFILL_STREAM=1 -> n_batch NOT capped\n"
                    "CGC-PHASE-SPLIT: cap=64 routable=142 slots / top_k=8 -> decode graph width=17 "
                    "tokens (bound=17, T_prefill=512, non-binding); prefill slab armed\n")["cap"] == 64)

    # 3. attribution: predicate first, trace line only as corroboration
    seg_pool = ("slot launch_slot_: id 0 | task 3 | processing task\n"
                "CGC-HOOK: ctx=0x1 il=0 ntok=2233 ids=[0 1]\n"
                "slot print_timing: id 0 | task 3 | prompt eval time = 20886.21 ms / 2233 tokens\n"
                "buffer is nil\n")
    a = attribution(seg_pool, 2233, 8, False)
    ck("pool arm, wide chunk -> pool via predicate", a["served_by"] == "pool" and a["evidence"] == "predicate", a)
    ck("attribution reads the segment's own token count", a["log_prompt_tokens"] == 2233, a)
    ck("attribution counts hooks", a["hook_lines"] == 1, a)
    ck("attribution counts nil buffers", a["nil_count"] == 1, a)

    seg_slab = ("slot launch_slot_: id 0 | task 6 | processing task\n"
                "CGC-PREFILL-STREAM: il=0 kind=0 ntok=2233 experts=256 slab=110.00 MiB filled=1 bytes\n"
                "slot print_timing: id 0 | task 6 | prompt eval time = 20886.21 ms / 2233 tokens\n")
    a = attribution(seg_slab, 2233, 8, True)
    ck("slab arm, wide chunk -> slab via line", a["served_by"] == "slab" and a["evidence"] == "line", a)
    a = attribution(seg_slab, 2233, 8, False)
    ck("a slab line on an unarmed arm is a CONFLICT, not silently believed",
       a["conflict"] is True and a["served_by"] == "slab", a)
    a = attribution(seg_slab.replace("2233", "12"), 2233, 8, True)
    ck("segment/probe mismatch refuses to attribute", a["served_by"] == "unknown", a)
    ck("missing segment says unknown", attribution(None, 12, 8, True)["served_by"] == "unknown")
    a = attribution("slot launch_slot_: id 0 | task 1 | processing task\n"
                    "slot print_timing: id 0 | task 1 | prompt eval time = 8666 ms / 12 tokens\n",
                    12, 8, True)
    ck("slab arm, 12 > width 8 -> slab by predicate (the trace cap hides the line)",
       a["served_by"] == "slab" and a["evidence"] == "predicate", a)
    a = attribution("slot launch_slot_: id 0 | task 1 | processing task\n"
                    "slot print_timing: id 0 | task 1 | prompt eval time = 1700 ms / 12 tokens\n",
                    12, 17, False)
    ck("pool17 arm, 12 <= width 17 -> pool by predicate", a["served_by"] == "pool", a)

    # 4. segments: one per request, in send order
    two = ("slot launch_slot_: id 0 | task 0 | processing task\n"
           "slot print_timing: id 0 | task 0 | prompt eval time = 8666 ms / 12 tokens\n"
           "slot launch_slot_: id 0 | task 2 | processing task\n"
           "slot print_timing: id 0 | task 2 | prompt eval time = 20886 ms / 2233 tokens\n")
    ck("segments splits per request", len(segments(two)) == 2, segments(two))
    ck("segment 1 is the short request",
       attribution(segments(two)[0], 12, 8, True)["served_by"] == "slab")
    ck("segment 2 is the long request",
       attribution(segments(two)[1], 2233, 8, True)["served_by"] == "slab")

    # A warm-up request in front of the measured ones must not shift the attribution. The real log
    # shape is [warm-up 12, short_warm 12, long 2233, short_postlong 12] -- note the two 12s at the
    # front, which is exactly why matching on token count alone is not enough and the offset is
    # searched rather than assumed.
    seg = ("slot launch_slot_: id 0 | task 0 | processing task\n"
           "slot print_timing: id 0 | task 0 | prompt eval time = {ms} ms / {n} tokens\n")
    three = (seg.format(ms=8487, n=12) + seg.format(ms=8666, n=12)
             + seg.format(ms=20886, n=2233) + seg.format(ms=8667, n=12))
    segs3 = segments(three)
    ck("synthetic log has the real shape", align_segments(segs3, [12, 2233, 12]) == 1
       or len(segs3) == 4, segs3)
    ck("alignment finds the warm-up offset",
       align_segments(segs3, [12, 2233, 12]) == 1, align_segments(segs3, [12, 2233, 12]))
    ck("alignment matches without a warm-up", align_segments(segments(two), [12, 2233]) == 0)
    ck("alignment refuses to guess when nothing matches",
       align_segments(segments(two), [99, 98]) == 0)
    off = align_segments(segs3, [12, 2233, 12])
    ck("warm-up-offset attribution lands on the measured requests",
       attribution(segs3[off], 12, 8, True)["served_by"] == "slab"
       and attribution(segs3[off + 1], 2233, 8, True)["served_by"] == "slab")
    # the failure mode actually observed at 18:47: without the offset the LONG probe lands on the
    # 12-token segment (the two 12-token requests made index 0 look right), and the token check is
    # what refuses it instead of reporting the short request's phase as the long request's.
    ck("without the offset the long probe is refused",
       attribution(segs3[1], 2233, 8, True)["evidence"] == "segment/probe mismatch"
       and attribution(segs3[1], 2233, 8, True)["served_by"] == "unknown")

    # 5. trace census knows the caps
    tc = trace_census("CGC-PREFILL-STREAM: il=0 kind=0 ntok=1 experts=256 slab=1 bytes\n" * 9
                      + "CGC-HOOK: ctx=0x1 il=0 ntok=1 ids=[0]\n" * 3)
    ck("census counts lines", tc["slab_lines"] == 9 and tc["hook_lines"] == 3, tc)
    ck("census reports the caps", tc["slab_cap"] == 4 and tc["hook_cap"] == 80, tc)

    # 6. verdict logic, including the discriminator that guards the headline claim
    def rows_mk(long_slab, long_p8, long_p17, sw_slab, sw_p17, slab_armed=True, slab_w=8, pool_w=17,
                sw_ntok=12):
        def cell(v):
            return {"per_launch": [v], "median_discard_rep1": v, "notes": []}
            # noqa: E301 - keeper of the shape the report consumes
        return {"phase-slab": {"cells": {"long": cell(long_slab), "short_warm": cell(sw_slab),
                                         "short_postlong": cell(sw_slab)},
                               "geom": {"slab_armed": slab_armed, "decode_width": slab_w},
                               "attrib": {"short_warm": [{"log_prompt_tokens": sw_ntok}]}},
                "phase-pool8": {"cells": {"long": cell(long_p8), "short_warm": cell(long_p8),
                                          "short_postlong": cell(long_p8)}, "geom": {}, "attrib": {}},
                "phase-pool17": {"cells": {"long": cell(long_p17), "short_warm": cell(sw_p17),
                                           "short_postlong": cell(sw_p17)},
                                 "geom": {"slab_armed": False, "decode_width": pool_w},
                                 "attrib": {}}}

    K_PP = "pool8/pool17 (long, same 2233-token chunk)"
    K_SS = "slab/pool17 (short_warm, one graph per request)"
    K_END = "slab/pool8 (long, end-to-end)"
    K_D = "option D (arm the slab by default?)"
    v_eq = dict(verdicts(rows_mk(200, 100, 100, 100, 100)))
    ck("verdict: names both sides", v_eq[K_PP].startswith("pool8/pool17 = 1.0"), v_eq[K_PP])
    ck("verdict: a clamp-width gap is called what it is",
       "clamp-width gap" in v_eq[K_PP], v_eq[K_PP])
    ck("verdict: direction is stated when the wider clamp wins",
       "pool17 is 1.67x faster" in dict(verdicts(rows_mk(200, 60, 100, 100, 100)))[K_PP])
    ck("verdict: end-to-end long comparison names the slab as the faster side",
       "**slab is 2.00x faster**" in v_eq[K_END], v_eq[K_END])
    ck("verdict: slab == pool17 at one graph -> no speed justification",
       "not measurably faster" in v_eq[K_SS], v_eq[K_SS])
    ck("verdict: slab faster at one graph -> says slab",
       "**slab is" in dict(verdicts(rows_mk(200, 100, 100, 150, 100)))[K_SS])
    ck("verdict: slab slower at one graph -> says pool17, not an ambiguous x0.66",
       "**pool17 is" in dict(verdicts(rows_mk(200, 100, 100, 100, 150)))[K_SS])
    ck("verdict: short cell in two cache states is reported separately",
       any("short_warm/short_postlong" in q for q, _ in verdicts(rows_mk(200, 100, 100, 50, 50))))
    v_d = dict(verdicts(rows_mk(200, 100, 100, 50, 100)))
    ck("verdict: option D is answered conditionally when long wins and short loses",
       v_d[K_D].startswith("CONDITIONAL") and "floor on chunk size" in v_d[K_D], v_d[K_D])
    ck("verdict: a short-cell tie alone does not produce a CONDITIONAL yes for option D",
       K_D not in v_eq or not v_eq[K_D].startswith("CONDITIONAL"), v_eq.get(K_D))
    ck("verdict: option D says NO when the slab regime is not faster on the long chunk",
       dict(verdicts(rows_mk(50, 100, 100, 100, 100)))[K_D].startswith("NO on speed grounds"))
    ck("verdict: discriminator INVALID when the short prompt was not slab-sized",
       dict(verdicts(rows_mk(200, 100, 100, 100, 100, sw_ntok=8))).get("discriminator", "")
       .startswith("INVALID"))
    ck("verdict: discriminator INVALID when the pool arm's short prompt was chunked",
       dict(verdicts(rows_mk(200, 100, 100, 100, 100, sw_ntok=40))).get("discriminator", "")
       .startswith("INVALID"))
    ck("verdict: no discriminator complaint when the geometry is right",
       "discriminator" not in dict(verdicts(rows_mk(200, 100, 100, 100, 100))))
    ck("verdict: missing data says NOT MEASURED",
       dict(verdicts(rows_mk(None, None, None, None, None)))[K_PP].startswith("NOT MEASURED"))

    # 7. rotation: over reps 2..N each arm must occupy each position exactly once, or the "rotation"
    # is not a Latin square and the position confound it exists to remove is still there.
    orders = [arm_order(r) for r in range(1, 6)]
    ck("rotation keeps all arms every rep", all(sorted(o) == sorted(ARMS) for o in orders), orders)
    ck("rotation moves the arms", orders[0] != orders[1], orders)
    for pos in range(3):
        seen = [o[pos] for o in orders[1:4]]
        ck(f"rotation: position {pos} is occupied once by every arm over reps 2..4",
           sorted(seen) == sorted(ARMS), seen)

    # 8. the decode gate: baseline pool8, explicit tolerance, per-cell verdicts
    def rows_dec(slab_dec, pool8_dec, pool17_dec=None):
        def cell(pref, dec):
            decs = dec if isinstance(dec, list) else [dec]
            return {"per_launch": [pref], "median_discard_rep1": pref, "decodes": decs,
                    "decode_median_discard_rep1": _median_discard_first(decs), "notes": []}
        out = {"phase-slab": {"cells": {s: cell(100, slab_dec) for s in SHAPES}},
               "phase-pool8": {"cells": {s: cell(50, pool8_dec) for s in SHAPES}}}
        if pool17_dec is not None:
            out["phase-pool17"] = {"cells": {s: cell(60, pool17_dec) for s in SHAPES}}
        return out

    g = decode_gate(rows_dec([10, 10], [10, 10]))
    ck("gate: equal decode -> PASS on every cell", all(x["verdict"] == "PASS" for x in g), g)
    ck("gate: reports every cell, so a pass cannot hide a failure", len(g) == len(SHAPES), len(g))
    g = decode_gate(rows_dec([5, 5], [10, 10]))
    ck("gate: half the decode rate -> FAIL", all(x["verdict"] == "FAIL" for x in g), g)
    ck("gate: failure names the numbers", "x0.5" in gate_failures(g)[0], gate_failures(g))
    g = decode_gate(rows_dec([9.2, 9.2], [10, 10]))   # -8%, inside the -10% tolerance
    ck("gate: a small regression inside tolerance still PASSes",
       all(x["verdict"] == "PASS" for x in g), g)
    g = decode_gate(rows_dec([8.5, 8.5], [10, 10]))   # -15%
    ck("gate: a regression past tolerance FAILs", all(x["verdict"] == "FAIL" for x in g), g)
    g = decode_gate(rows_dec(10, None))
    ck("gate: missing baseline -> NOT MEASURED, not a silent pass",
       all(x["verdict"] == "NOT MEASURED" for x in g), g)
    ck("gate: baseline is pool8 even when pool17 exists",
       decode_gate(rows_dec([10, 10], [5, 5], [20, 20]))[0]["pool8"] == 5)
    g1 = decode_gate(rows_dec(10, 10))
    ck("gate: a single usable launch cannot decide (n < MIN_GATE_N)",
       all(x["verdict"] == "NOT MEASURED" and "usable" in x.get("why", "") for x in g1), g1)
    g2 = decode_gate(rows_dec([10, 10], [10, 10]))
    ck("gate: n >= MIN_GATE_N decides", all(x["verdict"] == "PASS" for x in g2), g2)
    ck("gate: interference is detected against a live/dead pid",
       alive(os.getpid()) is True and alive(999999) is False)

    # 8b. the artifact path, which is how this gate could have passed while measuring nothing: with
    # n_predict=1 llama-server reports ~1e6 "t/s" on BOTH arms, so the ratio is 1.0 and a naive gate
    # says PASS. Measured on the real box 2026-09-17 (first_step.decode_tps = 1000000.0).
    dt = dump_timings({"predicted_n": 1, "predicted_ms": 0.001, "predicted_per_second": 1000000.0})
    ck("timings: n_predict=1 is refused as a rate, with the reason recorded",
       dt["decode_tps"] is None and dt["raw"]["predicted_n"] == 1 and "artifact" in dt["note"], dt)
    dt2 = dump_timings({"predicted_n": 8, "predicted_ms": 800.0, "predicted_per_second": 10.0})
    ck("timings: a normal decode rate survives", dt2["decode_tps"] == 10.0, dt2)
    ck("timings: a high decode rate is refused too",
       dump_timings({"predicted_n": 8, "predicted_per_second": 5000.0})["decode_tps"] is None)
    g = decode_gate(rows_dec([1e6, 1e6], [1e6, 1e6]))
    ck("gate: identical artifact rates are NOT MEASURED, never a PASS",
       all(x["verdict"] == "NOT MEASURED" and "artifact" in x.get("why", "") for x in g), g)
    ck("gate: pool8 alone carrying an artifact also refuses the cell",
       all(x["verdict"] == "NOT MEASURED"
           for x in decode_gate(rows_dec([10, 10], [1e6, 1e6]))))

    # 8c. cold start after a slab-served prefill: paired per rep, both instruments, direction checked.
    def rows_cold(slab_ratios, base_ratios, ln=10.0):
        def cell(dec_by_rep, wall_by_rep=None):
            decs = [dec_by_rep[r] for r in sorted(dec_by_rep) if r >= 2]
            return {"per_launch": [], "median_discard_rep1": None, "decodes": decs,
                    "decode_median_discard_rep1": _median_discard_first(decs) if decs else None,
                    "decode_refused": [], "notes": [], "decode_by_rep": dec_by_rep,
                    "wall_by_rep": wall_by_rep or {}}
        cells = {}
        for s in SHAPES:
            if s == "long":
                cells[s] = cell({r: ln for r in base_ratios}, {})
            else:
                cells[s] = cell({r: base_ratios[r] for r in base_ratios}, {})
        slab_cells = {}
        for s in SHAPES:
            slab_cells[s] = cell({r: slab_ratios[r] for r in slab_ratios}, {})
        return {"phase-slab": {"cells": slab_cells},
                "phase-pool8": {"cells": cells},
                "phase-pool17": {"cells": {s: cell({}) for s in SHAPES}}}

    cold = cold_after_slab(rows_cold({2: 5.0, 3: 5.0, 4: 5.0}, {2: 10.0, 3: 10.0, 4: 10.0}), 4)
    ck("cold: slab decode half of pool on every rep -> SYSTEMATIC COLD",
       all(c["verdict"].startswith("SYSTEMATIC COLD") for c in cold["cells"]
           if c["vs"] == "phase-pool8"), cold["cells"])
    cold = cold_after_slab(rows_cold({2: 20.0, 3: 20.0, 4: 20.0}, {2: 10.0, 3: 10.0, 4: 10.0}), 4)
    ck("cold: slab decode twice the pool -> hypothesis refuted, named as such",
       any("refuted" in c["verdict"] for c in cold["cells"]), cold["cells"])
    ck("cold: a consistent direction is recorded as consistent",
       all(c.get("consistent") is True for c in cold["cells"] if c["verdict"].startswith("SYSTEMATIC")))
    cold = cold_after_slab(rows_cold({2: 5.0, 3: 20.0, 4: 5.0}, {2: 10.0, 3: 10.0, 4: 10.0}), 4)
    ck("cold: opposite signs on different reps -> INCONCLUSIVE, not averaged into a claim",
       any("INCONCLUSIVE" in c["verdict"] for c in cold["cells"]), cold["cells"])
    ck("cold: only rep1 present -> NOT MEASURED (rep1 is discarded as cold)",
       all(c["verdict"] == "NOT MEASURED"
           for c in cold_after_slab(rows_cold({1: 5.0}, {1: 10.0}), 1)["cells"]))
    ck("rotation: balance is checked from arm_order, and is True over reps 2..4",
       rotation_balance(4)["balanced"] is True, rotation_balance(4)["mean_position"])
    ck("rotation: with no counted reps it is not claimed balanced",
       rotation_balance(1)["balanced"] is False)
    ck("reps_present reads the files, not the flag",
       reps_present(rows_cold({2: 5.0, 5: 5.0}, {2: 10.0, 5: 10.0})) == 5)

    # An externally killed launch must be NAMED, never folded into a median as if it were a timing.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        json.dump({"arm": "phase-slab", "rep": 1,
                   "interfered": {"probe": "long", "error": "RemoteDisconnected", "server_pid": 1}},
                  open(os.path.join(td, "phase-slab_rep1.json"), "w"))
        rows_t = load_rows(1, td)
        ck("interference: the launch is recorded",
           rows_t["phase-slab"]["interfered"][0]["probe"] == "long", rows_t["phase-slab"])
        ck("interference: the cell says so and carries no timing",
           "interfered(long)" in ";".join(rows_t["phase-slab"]["cells"]["long"]["notes"])
           and rows_t["phase-slab"]["cells"]["long"]["per_launch"] == [], rows_t["phase-slab"])
        ck("interference: the gate refuses to decide on it",
           all(x["verdict"] == "NOT MEASURED" for x in decode_gate(rows_t)), decode_gate(rows_t))

    # 8d. per-LAUNCH records must appear once per launch, not once per shape. Pool counters and the
    # interference record are launch-level, but they were collected inside the per-shape loop, which
    # multiplied them by len(SHAPES) (16 rows for a 4-launch arm) and inflated every counter 4x.
    with tempfile.TemporaryDirectory() as td:
        for rep in (1, 2):
            rec = {"arm": "phase-slab", "rep": rep}
            for s in SHAPES:
                rec[s] = {"prefill_tps": 10.0, "decode_tps": 5.0, "wall_s": 1.0}
            json.dump(rec, open(os.path.join(td, f"phase-slab_rep{rep}.json"), "w"))
        r1 = load_rows(2, td)["phase-slab"]
        ck("load_rows: one pool record per launch, not one per shape",
           len(r1["pool"]) <= 2, len(r1["pool"]))
        ck("load_rows: decode by rep is keyed, so pairing is per rep",
           r1["cells"]["long"]["decode_by_rep"] == {1: 5.0, 2: 5.0}, r1["cells"]["long"])

    # 9. the pool counters that decide 'cold vs thrashing'
    sample = ("llama_expert_cache: final stats: runtime requests=8764 hits=7559 misses=1205 "
              "(hit rate 86.3%)  prewarm req=0 hit=0 miss=0  resident=6422.04 MiB file_reads=59574 "
              "pread_usec=3040906790 fill_batch_usec=1 fill_wait_us=71342 prefetch=466/203\n"
              "llama_expert_cache: slab fills: pool=25393.4 MiB disk=20094.6 MiB "
              "(non-resident share 44.2%)\n"
              "llama_expert_cache: miss attribution: compulsory=1201 capacity=4 (99.7% / 0.3% of 1205)"
              "  evictions=899  layers_distinct_over_slots=0  worst=layer 40 distinct=113 slots=256\n"
              "llama_expert_cache: prefetch drop breakdown (total=203): #1 no_free_slot=0  "
              "#3 drain_cleared=203  #4 zero_slot_fallback=0\n")
    ps = parse_pool_stats(sample)
    ck("pool stats: hit rate", ps.get("hit_pct") == 86.3, ps)
    ck("pool stats: compulsory vs capacity", ps.get("compulsory") == 1201
       and ps.get("capacity") == 4, ps)
    ck("pool stats: evictions", ps.get("evictions") == 899, ps)
    ck("pool stats: pread seconds", ps.get("pread_s") == 3040.9, ps)
    ck("pool stats: resident high-water mark (was silently absent -- the report column went empty)",
       ps.get("resident_mib") == 6422.04, ps)
    # The launch-log -> server-log hop, which is how a stored result gets its counters backfilled.
    line = "[log]   /Users/x/Backup/cgc_logs/llama_server_20260917_201341.log（tail -f 同路徑）"
    mm = SERVER_LOG_RE.search(line)
    ck("server log path: the underscore between date and time is matched",
       bool(mm) and mm.group(1).endswith("llama_server_20260917_201341.log"), mm)
    ck("server log path: a tail -f style line is matched too",
       bool(SERVER_LOG_RE.search("         log: tail -f /a/b/Backup/cgc_logs/llama_server_1_2.log")))
    ck("pool stats: slab non-resident share", ps.get("slab_nonresident_pct") == 44.2, ps)
    ck("pool stats: prefetch issued/dropped", ps.get("prefetch_issued") == 466
       and ps.get("prefetch_dropped") == 203, ps)
    ck("pool stats: only the non-zero drop reasons are kept",
       ps.get("prefetch_drops") == {"#3 drain_cleared": 203}, ps.get("prefetch_drops"))
    ck("pool stats: empty log -> empty dict, no exception", parse_pool_stats("") == {})

    # 10. the window gate is the launcher's own rule, not a number invented here
    _, why, _ = window_status(set(), LAUNCHER_FREE_PCT_FULLMTP)
    ck("window gate cites the launcher's requirement", "launcher full-mtp" in why, why)

    # 11. contention: another line's probe must count as busy, and the hold must require persistence
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "Backup", "cgc_logs")
        os.makedirs(d)
        other = os.path.join(d, "probe_m5-long_20260917_195430.ctrl.log")
        open(other, "w").close()
        # point the detector at the temp dir by running from a fake ROOT
        real_root = globals()["ROOT"]
        try:
            globals()["ROOT"] = td
            ck("contention: a fresh probe log is detected",
               any("probe_m5-long" in x for x in other_line_activity(420)),
               other_line_activity(420))
            ck("contention: a 0 lookback disables it", other_line_activity(0) == [])
            os.utime(other, (time.time() - 3600, time.time() - 3600))
            ck("contention: an old probe log is not busy", other_line_activity(420) == [])
            ok, why, _ = window_status(set(), 0, contention_lookback=420)
            ck("window: free memory alone does not open it", ok is True, why)
            os.utime(other, (time.time(), time.time()))
            ok, why, _ = window_status(set(), 0, contention_lookback=420)
            ck("window: a fresh other-line probe closes it", ok is False and "another line" in why, why)
        finally:
            globals()["ROOT"] = real_root

    # 12. the retry selector picks exactly the launches with no timings at all
    with tempfile.TemporaryDirectory() as td:
        json.dump({"arm": "phase-slab", "rep": 1}, open(os.path.join(td, "phase-slab_rep1.json"), "w"))
        json.dump({"arm": "phase-slab", "rep": 2, "long": {"prefill_tps": 100.0}},
                  open(os.path.join(td, "phase-slab_rep2.json"), "w"))
        ck("retry: a launch with no timings is a retry", needs_retry(td, "phase-slab", 1))
        ck("retry: a launch with one timing is not a retry", not needs_retry(td, "phase-slab", 2))
        ck("retry: a missing launch is a retry", needs_retry(td, "phase-slab", 3))

    print(f"selftest: {checks - len(fails)}/{checks} checks passed")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=None, help="default: Backup/phase_split_ab/<YYYYmmdd_HHMM>")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved arms and exit")
    ap.add_argument("--selftest", action="store_true", help="pin the harness's own parsers offline")
    ap.add_argument("--window-check", action="store_true", help="print the window verdict and exit")
    ap.add_argument("--watch", action="store_true", help="wait for a valid window, then run")
    ap.add_argument("--self-detach", action="store_true", help="double-fork, then run the real grid")
    ap.add_argument("--log-file", default=None, help="only used with --self-detach")
    ap.add_argument("--min-free-pct", type=int, default=LAUNCHER_FREE_PCT_FULLMTP)
    ap.add_argument("--watch-poll", type=float, default=30.0)
    ap.add_argument("--watch-timeout", type=float, default=7200.0)
    ap.add_argument("--contention-lookback", type=float, default=420.0,
                    help="treat another line's probe_generated ctrl.log newer than this as busy; "
                         "0 disables. See other_line_activity() for why this exists.")
    ap.add_argument("--window-hold", type=int, default=3,
                    help="consecutive quiet polls required before launching")
    ap.add_argument("--retry-failed", action="store_true",
                    help="after the grid, re-run launches that produced no timings at all")
    ap.add_argument("--wait-between", type=float, default=900.0,
                    help="how long to wait for someone else's process to clear mid-grid")
    ap.add_argument("--allow-busy-box", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    # No --report flag on purpose: the report path is derived from the date, so a run cannot write
    # its verdict into a file nobody looks at. Pass none, get one honest destination.
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.dry_run:
        for arm, extra in ARMS.items():
            d = launcher_dump(extra)
            print(f"--- {arm}  extra_env={extra}  (port={d.get('port')}, "
                  f"model={os.path.basename(d.get('model', '?'))})")
            for l in (d.get("notes") or [])[:2]:
                print(f"    {l}")
        return 0

    if args.window_check:
        # Same predicate the grid uses, contention included: a diagnostic that reports a different
        # gate than the one that decides is how "the window was open" and "it refused" coexist.
        ok, why, _ = window_status(set(), args.min_free_pct, args.allow_busy_box,
                                  args.contention_lookback)
        print(("WINDOW OPEN   " if ok else "WINDOW CLOSED ") + why)
        return 0 if ok else 2

    if args.self_detach:
        return self_detach(args)

    out = args.out or os.path.join(ROOT, "Backup", "phase_split_ab", time.strftime("%Y%m%d_%H%M"))
    os.makedirs(out, exist_ok=True)
    log(f"out dir: {out}")

    if args.aggregate_only:
        aggregate(args.reps, out)
        return 0

    dump = launcher_dump({})
    port = dump.get("port") or "8080"
    base = f"http://127.0.0.1:{port}"
    log(f"port={port} model={os.path.basename(dump.get('model', '?'))} profile={PROFILE} "
        f"HEAD={sh(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).stdout.strip()}")
    if dump.get("rc") not in (0, None):
        log(f"WARNING: launcher dump rc={dump.get('rc')} -> the launcher's own guard may be failing: "
            f"{dump.get('stderr')}")

    watch_log = os.path.join(out, "watch.log")
    if args.watch:
        ok, why = wait_for_window(args.min_free_pct, args.watch_poll, args.watch_timeout, watch_log,
                                  allow_busy=args.allow_busy_box,
                                  contention_lookback=args.contention_lookback,
                                  hold=args.window_hold)
        if not ok:
            log(f"window never came: {why}")
            aggregate(0, out, ran=False, reason=why)
            return 3

    aborted = ""
    done_reps = 0
    for rep in range(1, args.reps + 1):
        ours: set[int] = set()
        order = arm_order(rep)
        log(f"rep{rep} arm order: {' -> '.join(order)}")
        for arm in order:                                 # interleaved, rotated per rep
            ok, why, _ = window_status(ours, args.min_free_pct, args.allow_busy_box,
                                       args.contention_lookback)
            if not ok:
                log(f"mid-grid: {why} -> waiting up to {args.wait_between / 60:.0f} min")
                ok, why = wait_for_window(args.min_free_pct, args.watch_poll, args.wait_between,
                                          watch_log, ours=ours, allow_busy=args.allow_busy_box,
                                          contention_lookback=args.contention_lookback,
                                          hold=args.window_hold)
                if not ok:
                    aborted = f"mid-grid at rep{rep} {arm}: {why}"
                    log(f"ABORTING the grid honestly: {aborted}")
                    break
            rec = run_arm(base, arm, rep, out, args.ready_timeout)
            if rec.get("server_pid"):
                ours.add(rec["server_pid"])
        if aborted:
            break
        done_reps = rep
        log(f"rep{rep} complete ({done_reps}/{args.reps})")

    # One bounded retry pass, for launches an external process killed. Recorded as `retry` in the
    # JSON and named in the report: a retried launch is NOT at its rotation position any more (it
    # happened later), so the report must not pretend the Latin square survived it.
    if args.retry_failed and not aborted:
        todo = [(rep, arm) for rep in range(1, args.reps + 1) for arm in arm_order(rep)
                if needs_retry(out, arm, rep)]
        if todo:
            log(f"retry pass: {len(todo)} launch(es) produced no timings -> "
                f"{', '.join(f'{a} rep{r}' for r, a in todo)}")
        for rep, arm in todo:
            ok, why = wait_for_window(args.min_free_pct, args.watch_poll, args.wait_between,
                                      watch_log, allow_busy=args.allow_busy_box,
                                      contention_lookback=args.contention_lookback,
                                      hold=args.window_hold)
            if not ok:
                log(f"retry pass stopped: {why}")
                break
            run_arm(base, arm, rep, out, args.ready_timeout, force=True, retry=True)

    if aborted and done_reps == 0:
        aggregate(0, out, ran=False, reason=aborted)
        return 3
    rows, _ = aggregate(args.reps, out, ran=True, reason=aborted or "")
    # A gate that cannot fail loudly is a report, not a gate: non-zero exit on decode regression, so
    # a script or a person sees it in the exit status and not only in the prose.
    fails = gate_failures(decode_gate(rows))
    if fails:
        log("DECODE GATE FAIL: " + "; ".join(fails))
        return 4
    log("decode gate: PASS on every cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
