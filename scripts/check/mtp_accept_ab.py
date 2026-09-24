#!/usr/bin/env python3
"""Measure MTP draft acceptance per carrier, with the head's identity recorded alongside it.

WHY
---
An MTP accept rate is a property of the (base, head) pair, and it is the number the roadmap's
M4 exits on. Until now the harness had no cell for it at all: the five oracle cells all ran the
MTP *engine*, but their probe was a single chat turn, so a green oracle said nothing about the
verify path or about accept. This driver is that missing cell.

It refuses to produce a number for a head that is not alive. That is not ceremony: the Edge0
head artifact carried an F16 payload under a BF16 type tag, which collapsed its router and made
accept a foregone conclusion. A driver that reports "accept 0.0%" for that artifact is reporting
a file defect as a model property. `mtp_head_identity.py` decides liveness; this script obeys it.

WHAT IT REPORTS
---------------
Per carrier: accepted / generated draft tokens (the server's own counters, from the HTTP
`timings.draft_n_accepted` / `timings.draft_n` fields), the ratio, the mean accepted run length,
and prefill/decode tok/s. Summed over the measured requests, with the per-request values kept in
the JSON so a single outlier cannot hide inside an average.

USAGE
-----
    python3 scripts/check/mtp_accept_ab.py                     # all carriers
    python3 scripts/check/mtp_accept_ab.py --arms nail         # one
    python3 scripts/check/mtp_accept_ab.py --pool-gb 8 --n-predict 96
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "check" / "mtp_head_identity.py"
LOGDIR = ROOT / "Backup" / "cgc_logs"
SERVER_BIN = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-server"


def _file_digest(path, sample_size=65536):
    """Size + first/last 64KiB hash. Same sampling as knifeedge_matrix / pool_curve:
    a full sha256 of a 13GB file is too slow for every arm, and the head/tail sample
    catches every real-world swap."""
    p = Path(path)
    if not p.exists():
        return None
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(sample_size))
        if size > sample_size * 2:
            f.seek(-sample_size, 2)
            h.update(f.read(sample_size))
    return {"size": size, "head_hash": h.hexdigest()[:16]}


def _facts_from_log(logpath: Path) -> dict:
    """Extract min_layer_slots / n_slots from the server log, same fields
    knifeedge_matrix.read_launch_facts() reads. Kept local so mtp_accept_ab
    does not grow a dependency on the matrix harness's globals."""
    facts = {}
    try:
        text = logpath.read_text(errors="replace")
    except OSError:
        return facts
    for pat, key in ((r"min (\d+)/layer", "min_layer_slots"),
                     (r"n_slots[= ]+(\d+)", "n_slots")):
        m = re.search(pat, text)
        if m:
            facts[key] = int(m.group(1))
    return facts


def record_provenance(label: str, model: Path, pool_gb: float, mtp: str,
                      extra_env, logpath: Path, thermal_at_launch=None,
                      thermal_gate=None) -> dict:
    """Provenance for one MTP accept arm. Same philosophy as knifeedge_matrix:
    the result carries enough identity that two arms measured weeks apart can
    be certified comparable -- or refused.

    The MTP-specific fields (mtp flag, head identity via the model path) are
    what make an accept rate attributable: a 60% accept from a dead head is
    not the same measurement as 60% from a live one."""
    pool = {"gb": float(pool_gb), "bytes": int(pool_gb * 1024 ** 3)}
    facts = _facts_from_log(logpath)
    if facts:
        pool["launched_min_layer_slots"] = facts.get("min_layer_slots")
        pool["launched_n_slots"] = facts.get("n_slots")
    # [2026-09-23] `thermal_at_launch` is read BEFORE the launch (the criterion is the level at
    # launch, not over the run -- thermal_pressure.py docstring). Without it an arm's t/s is
    # unattributable: six arms of this same configuration spanned 1.79x (5.84-10.46 t/s) with
    # bit-identical outputs (docs/INSTRUMENT_RESOLUTION_2026-09-23.md), and thermal pressure is
    # the one variable this repo has a measured zero-overlap separation for -- on the PREFILL
    # axis (level 0 -> 6/6 runs >= 250 t/s; level 1-2 -> 0/21). It is recorded as a REGIME, not
    # a gate: it did not separate 6.26 from 9.26 t/s in the two cross-axis pairs.
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arm": label,
            # The SERVER log this arm's engine counters went to. Not `logpath`: that is the arm's own
            # stdout capture (`mtp_accept_*.log`), which contains zero CGC lines -- naming it would
            # be a field that answers nothing. Resolved by launch time instead, and the
            # `llama_server_latest.log` symlink is excluded: in the 2026-09-23 run one arm's counters
            # were read through that symlink and silently belonged to a different launch.
            "server_log": server_log_at(launch_ts),
            "thermal_at_launch": thermal_at_launch,
            # Seconds spent waiting for Nominal before this launch, and whether the wait
            # succeeded. Recorded because the wait is a TREATMENT, not overhead: in the
            # certified-window ABBA the k=3 arms' readings rose monotonically with it
            # (41 s -> 9.09, 50 s -> 8.59, 141 s -> 10.72, 181 s -> 11.48 t/s), i.e. the level
            # signal alone could not separate a box idle for six minutes from one idle for 50 s.
            "thermal_gate": thermal_gate,
            "pool": pool,
            "model": {"name": model.name, "realpath": str(model.resolve()),
                      **(_file_digest(model) or {})},
            "binary": {"path": str(SERVER_BIN), **(_file_digest(SERVER_BIN) or {})},
            "launch": {"mtp": mtp, "extra_env": sorted(extra_env or [])}}

# The three carriers that matter, and why each one is here:
#   nail      -- the known-good pair (Nail's own head on Nail's own base). This is the
#                baseline the strategy promises to reach, so it must reproduce.
#   edge0head -- Edge0's own head on Edge0's base, i.e. the artifact the policy calls for.
#                Was reporting a dead router; measure it now that it is alive.
#   graft     -- Nail's head on Edge0's base. The (base, head) mismatch, kept as the control
#                that shows accept is a property of the PAIR and not of the head's precision.
# (model, why, mtp) -- the mtp flag is part of the arm because M4's exit is "MTP-on >= MTP-off",
# and that comparison is only meaningful on the same carrier with the same head.
ARMS = {
    "nail":      ("models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
                  "Nail head on Nail base (known-good pair)", "1"),
    "edge0head": ("models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf",
                  "Edge0's own head on Edge0 base (the policy's carrier)", "1"),
    "graft":     ("models/gguf/Edge0-35B-Q4_0-MTP.gguf",
                  "Nail head on Edge0 base ((base,head) mismatch control)", "1"),
    "nail_nomtp": ("models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
                   "same Nail carrier with MTP OFF -- the denominator of 'MTP-on >= MTP-off'", "0"),
}

REQUIRE_THERMAL_0 = [None]      # --require-thermal-0 TIMEOUT_S, set in main()

PROMPTS = [
    "Write one paragraph describing how a river changes between its source and the sea.",
    "Explain in a few sentences why the sky is blue.",
    "List four differences between a lake and an ocean, with one clause each.",
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_thermal():
    """The OS thermal pressure level, or an explicit refusal to guess one.

    `thermal_pressure.py` is the tree's instrument for this (2 ms, no root, and it refuses an
    unverified key rather than inventing a reading). Imported lazily and never allowed to fail
    an arm: a missing stamp must not become a missing measurement.
    """
    try:
        sys.path.insert(0, str(ROOT / "scripts" / "check"))
        import thermal_pressure as tp
        return tp.stamp()
    except Exception as e:
        return {"level": None, "label": f"UNREADABLE: {e}"}


def sh(*args) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def http(url: str, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def gate(model: Path) -> dict:
    """Identity + liveness, delegated to mtp_head_identity.py.

    Raises if the head is not alive. The delegation matters: the rule about what counts as a
    usable head lives in one place, so this driver cannot quietly measure a dead one.
    """
    out = subprocess.run([sys.executable, str(GATE), "check", "--gguf", str(model)],
                         capture_output=True, text=True)
    sidecar = Path((str(model)[:-5] if str(model).endswith(".gguf") else str(model))
                   + ".mtphead.json")
    for line in (out.stdout + out.stderr).strip().splitlines():
        log(f"    gate: {line}")
    if out.returncode != 0:
        raise SystemExit(f"refusing to measure accept for {model.name}: the head gate failed")
    fp = json.loads(sidecar.read_text())
    return fp


def launch(model: Path, pool_gb: float, port: int, logpath: Path, mtp: str = "1",
           extra_env=()) -> None:
    env = dict(os.environ)
    env.update({
        "CGC_DETACHED": "1",
        "_CGC_DETACHED_MARKER": "1",
        "CGC_SERVER_MODEL": str(model),
        "CGC_SERVER_MTP": mtp,
        "CGC_SERVER_EXPERT_CACHE_BYTES": str(int(pool_gb * 1024 ** 3)),
        "CGC_SERVER_PORT": str(port),
    })
    for kv in extra_env or ():
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    with open(logpath, "wb") as fh:
        subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                         stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
                         start_new_session=True)


def server_pid(port: int | None = None) -> int | None:
    """The listener on `port` if given, else any llama-server. The port-scoped form is the one
    teardown uses: this box runs parallel sessions, so a pid-blind `pkill -f llama-server` kills
    their servers too (the defect http_duo.py:31,285 already recorded)."""
    if port is not None:
        out = sh("lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t").split()
        return int(out[0]) if out else None
    out = sh("pgrep", "-f", "build/bin/llama-server").split()
    return int(out[0]) if out else None


def stop_server(port: int | None = None) -> None:
    pid = server_pid(port)
    if pid is None:
        return
    subprocess.run(["kill", "-9", str(pid)], check=False)
    for _ in range(20):
        if server_pid(port) is None:
            return
        time.sleep(0.5)


# Markers that mean the launch is over rather than still in flight.
#
# Deliberately NOT the bare "[guard]" prefix: run_server.sh prints an informational
# "[guard] memory_mode=... free=74%" line on every successful launch, so matching it made a
# perfectly healthy load look dead. Only the refusal forms count, and those all carry the
# script's own "error:" prefix.
FATAL = ("error: startup blocked", "error: even fallback memory guard",
         "error: model not found", "error: CGC_SERVER_RUNTIME_PROFILE",
         "out of memory", "cannot allocate", "failed to load",
         "Killed:", "terminate called")


def dump_tail(logpath: Path, n=30) -> None:
    log(f"    tail of {logpath.name}:")
    try:
        for line in logpath.read_text(errors="replace").splitlines()[-n:]:
            log(f"      {line}")
    except OSError:
        log("      (unreadable)")


def wait_health(port: int, logpath: Path, timeout=600, startup_grace=150) -> bool:
    """Wait for /health, distinguishing "still loading" from "the launch is dead".

    The trap this avoids: run_server.sh validates, checks memory, computes a profile and only
    then execs the server, so `pgrep` legitimately finds NOTHING for the first several seconds
    after launch. Treating "no process yet" as death made the first run abandon a server that
    was loading normally. Now an absent process only counts after `startup_grace`.
    """
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    last_note = 0.0
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass

        if time.time() - t0 > 5 and logpath.exists():
            try:
                txt = logpath.read_text(errors="replace")
            except OSError:
                txt = ""
            for line in txt.splitlines():
                if any(m in line for m in FATAL):
                    log(f"    launch reported a failure: {line.strip()}")
                    dump_tail(logpath)
                    return False

        el = time.time() - t0
        if el > startup_grace and server_pid() is None:
            log(f"    no server process after {int(el)}s")
            dump_tail(logpath)
            return False

        if el - last_note > 60:
            last_note = el
            log(f"    loading… {int(el)}s (pid {server_pid()})")
        time.sleep(5)

    log(f"    timed out after {timeout}s")
    dump_tail(logpath)
    return False


# Deterministic filler, so a long-prompt prefill measurement is the same bytes every run.
# Prose rather than random tokens: it keeps the router's expert choice realistic (a random token
# soup would drive the union toward "all 256 experts on every layer", which is the pessimistic
# end of the range and not what a real request looks like).
FILLER = (
    "A river system carries sediment from its headwaters to the sea, and the amount it carries "
    "depends on the slope, the rock, and the vegetation along the way. Near the source the channel "
    "is steep and the water moves quickly, so it can move boulders and gravel. Lower down the "
    "slope flattens, the current slows, and the load becomes sand and silt. Where the channel "
    "meets a standing body of water the current drops almost to nothing and the load settles out "
    "as a delta. "
)


def context_prefix(chars: int) -> str:
    if chars <= 0:
        return ""
    return (FILLER * (chars // len(FILLER) + 1))[:chars] + "\n\n"


def complete(port: int, prompt: str, n_predict: int, timeout=900) -> dict:
    return http(f"http://127.0.0.1:{port}/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "stream": False, "cache_prompt": False,
    }, timeout=timeout)


def measure(label: str, model: Path, pool_gb: float, port: int, n_predict: int,
            warm: bool = True, mtp: str = "1", extra_env=(), context_chars: int = 0) -> dict:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logpath = LOGDIR / f"mtp_accept_{label}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log(f"launching {label}: {model.name} pool={pool_gb}GB port={port} mtp={mtp}"
        f" extra={list(extra_env) or '-'} ctx_chars={context_chars}")
    log(f"    log: {logpath}")

    stop_server(port)
    wait_s, waited = 0.0, None
    if REQUIRE_THERMAL_0[0] is not None:
        waited, wait_s, lvl = wait_for_thermal_0(REQUIRE_THERMAL_0[0])
        if not waited:
            log(f"    thermal gate: level {lvl} after {wait_s}s -- measuring anyway, recorded")
        else:
            log(f"    thermal gate: Nominal after {wait_s}s")
    thermal_at_launch = read_thermal()      # the criterion is the level AT LAUNCH
    launch_ts = time.time()
    launch(model, pool_gb, port, logpath, mtp, extra_env)
    try:
        if not wait_health(port, logpath):
            raise SystemExit(f"{label}: server did not become healthy")
        log(f"    healthy (pid {server_pid(port)})")

        prefix = context_prefix(context_chars)
        if warm:
            # Cold-cache first touch: its accept is real but its timings are not comparable.
            log("    warmup request (not counted)")
            complete(port, prefix + PROMPTS[0], 24)

        reqs = []
        for i, p in enumerate(PROMPTS):
            t0 = time.time()
            r = complete(port, prefix + p, n_predict)
            dt = time.time() - t0
            tm = r.get("timings", {})
            d_n = int(tm.get("draft_n", 0))
            d_a = int(tm.get("draft_n_accepted", 0))
            reqs.append({
                "prompt": p[:48], "wall_s": round(dt, 2),
                "draft_n": d_n, "draft_n_accepted": d_a,
                "prompt_n": tm.get("prompt_n"),
                "prefill_tps": tm.get("prompt_per_second"),
                "decode_tps": tm.get("predicted_per_second"),
                "n_predicted": tm.get("predicted_n"),
                "text_chars": len(r.get("content", "")),
            })
            log(f"    req {i+1}: prompt_n={tm.get('prompt_n')} "
                f"prefill={tm.get('prompt_per_second', 0):.2f} t/s "
                f"decode={tm.get('predicted_per_second', 0):.2f} t/s  draft {d_a}/{d_n}  {dt:.1f}s")
    finally:
        stop_server(port)

    tot_n = sum(r["draft_n"] for r in reqs)
    tot_a = sum(r["draft_n_accepted"] for r in reqs)
    dec = [r["decode_tps"] for r in reqs if r["decode_tps"]]
    pre = [r["prefill_tps"] for r in reqs if r["prefill_tps"]]
    return {
        "label": label, "model": model.name, "pool_gb": pool_gb, "mtp": mtp,
        "draft_n": tot_n, "draft_n_accepted": tot_a,
        "accept": (tot_a / tot_n) if tot_n else None,
        "mean_acc_len": (1.0 + tot_a / len(reqs)) if reqs else None,
        "decode_tps_mean": sum(dec) / len(dec) if dec else None,
        "prefill_tps_mean": sum(pre) / len(pre) if pre else None,
        "requests": reqs,
        # Provenance: stamped with the log still warm so it carries the launched
        # slot counts. An accept rate without this is unattributable -- the same
        # number from a different (base, head) pair or a different loader is not
        # the same measurement, and the roadmap's M4 exit depends on being able
        # to prove which pair produced it.
        "provenance": record_provenance(label, model, pool_gb, mtp, extra_env, logpath,
                                        thermal_at_launch,
                                        {"waited_s": wait_s, "reached_nominal": waited}),
    }


def server_log_at(since_ts: float, logdir: Path = None) -> str:
    """Newest real `llama_server_*.log` written at or after `since_ts`, or "".

    run_server.sh writes one timestamped log per launch, so launch time identifies it uniquely;
    the `llama_server_latest.log` symlink is skipped because it re-points at every launch (reading
    it for an older arm returns the newest run's counters -- measured, 2026-09-23).
    """
    logdir = logdir or (ROOT / "Backup" / "cgc_logs")
    cands = [p for p in logdir.glob("llama_server_*.log")
             if not p.is_symlink() and p.stat().st_mtime >= since_ts - 1.0]
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else ""


def wait_for_thermal_0(timeout_s: float, poll_s: float = 10.0):
    """Block until the OS reports Nominal, returning (ok, waited_s, level_seen).

    Measured 2026-09-23 (docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md): three back-to-back arms of
    ONE configuration drove the pressure level 0 -> 0/1 -> 2 HEAVY, and the HEAVY arm read 18%
    slow. So an arm's t/s is only attributable if the box is at Nominal when it launches -- the
    criterion thermal_pressure.py records ("the level AT LAUNCH"), which had never been enforced
    on this path. It is enforced here, with the wait recorded: the wait is itself a state
    variable (in that same window the k=3 arms' readings rose monotonically with it), so an
    undocumented wait would be an undocumented treatment.
    """
    t0 = time.time()
    while True:
        lvl = (read_thermal() or {}).get("level")
        if lvl == 0:
            return True, round(time.time() - t0, 1), lvl
        if time.time() - t0 >= timeout_s:
            return False, round(time.time() - t0, 1), lvl
        time.sleep(poll_s)


def selftest() -> int:
    """The two behaviours added 2026-09-23 that a fixture can hold still: log resolution, and the
    thermal gate's give-up path. Both are the kind that fail silently in a real run (an arm whose
    counters belong to another launch; a gate that blocks forever instead of measuring)."""
    import tempfile
    bad = 0

    def mk(folder: Path, name: str, age_s: float) -> Path:
        p = folder / name
        p.write_text(name)
        os.utime(p, (time.time() - age_s,) * 2)
        return p

    d = Path(tempfile.mkdtemp())
    own = mk(d, "llama_server_20260923_010000.log", 120)
    later = mk(d, "llama_server_20260923_020000.log", 30)
    (d / "llama_server_latest.log").symlink_to(later)
    now = time.time()
    cases = [
        # (since, expected): the symlink is never returned even though it has the newest mtime
        (now - 125, later.name), (now - 60, later.name), (now - 5, ""), (now - 200, later.name),
    ]
    for since, want in cases:
        got = server_log_at(since, logdir=d)
        if Path(got).name != want:
            print(f"FAIL server_log_at(since=-{now-since:.0f}s) = {got!r}, want {want!r}")
            bad += 1
    # and the real file behind the symlink is reachable on its own terms
    solo = Path(tempfile.mkdtemp())
    only = mk(solo, "llama_server_20260923_030000.log", 10)
    (solo / "llama_server_latest.log").symlink_to(only)
    if Path(server_log_at(now - 60, logdir=solo)).name != only.name:
        print("FAIL server_log_at missed the only real log")
        bad += 1

    # the gate: with the level unreadable it must give up at the timeout and say so, not hang
    real = read_thermal
    try:
        globals()["read_thermal"] = lambda: {"level": 2, "label": "HEAVY"}
        ok, waited, lvl = wait_for_thermal_0(0.0, poll_s=0.01)
        if ok or lvl != 2 or waited < 0:
            print(f"FAIL gate give-up: ok={ok} lvl={lvl} waited={waited}")
            bad += 1
        globals()["read_thermal"] = lambda: {"level": 0, "label": "NOMINAL"}
        ok, waited, lvl = wait_for_thermal_0(5.0, poll_s=0.01)
        if not ok or lvl != 0:
            print(f"FAIL gate pass-through: ok={ok} lvl={lvl}")
            bad += 1
    finally:
        globals()["read_thermal"] = real

    print(f"selftest: {7 - bad}/7 passed")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="check log resolution and the thermal gate without launching anything")
    ap.add_argument("--arms", default=",".join(ARMS), help=f"subset of {list(ARMS)}")
    ap.add_argument("--pool-gb", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=9932)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--out", default="Backup/cgc_logs/mtp_accept_ab.json")
    ap.add_argument("--context-chars", type=int, default=0,
                    help="deterministic filler prepended to every prompt, to make the prefill "
                         "measurement a real prefill instead of a ~20-token chat turn")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="KEY=VAL for run_server.sh (repeatable) -- e.g. the pool cap and slab "
                         "capacity being A/B'd")
    ap.add_argument("--require-thermal-0", type=float, default=None, metavar="TIMEOUT_S",
                    help="block until the OS thermal pressure reads Nominal before EACH arm "
                         "(give up after TIMEOUT_S and measure anyway, recorded as waited=false). "
                         "Off by default: this makes an arm take up to minutes longer, and it is "
                         "only worth it for arms whose effect is small. Measured same-config "
                         "spread: 18.5%% across a drifting sequence, 5.8%% (k=2)/13.1%% (k=3) with "
                         "this gate on.")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    REQUIRE_THERMAL_0[0] = args.require_thermal_0

    results = []
    for label in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if label not in ARMS:
            raise SystemExit(f"unknown arm {label!r}; known: {list(ARMS)}")
        rel, why, mtp = ARMS[label]
        model = ROOT / rel
        if not model.exists():
            log(f"SKIP {label}: {rel} not present")
            continue
        log(f"=== {label}: {why}")
        fp = gate(model)
        log(f"    head identity {fp['identity'][:16]}…  degenerate={fp['degenerate'] or '{}'}")
        try:
            res = measure(label, model, args.pool_gb, args.port, args.n_predict, mtp=mtp,
                          extra_env=args.extra_env, context_chars=args.context_chars)
        except SystemExit as e:
            # One carrier failing must not cost the others' measurements: a 3-arm run takes many
            # (the port-scoped teardown below is the same reason: only OUR listener is stopped)
            # minutes and a launcher refusal on arm 2 is not a reason to discard arm 1.
            log(f"    ARM FAILED: {e}")
            stop_server(args.port)
            results.append({"label": label, "model": model.name, "failed": str(e),
                            "head_identity": fp["identity"], "head_types": fp["types"]})
            time.sleep(5)
            continue
        res["head_identity"] = fp["identity"]
        res["head_types"] = fp["types"]
        results.append(res)
        a = res["accept"]
        log(f"    => accept {'n/a' if a is None else f'{a*100:.2f}%'}"
            f" ({res['draft_n_accepted']}/{res['draft_n']}), "
            f"decode {res['decode_tps_mean'] or 0:.2f} t/s")
        time.sleep(10)

    out = ROOT / args.out
    out.write_text(json.dumps(results, indent=2) + "\n")
    print()
    print(f"{'carrier':<10} {'accept':>8} {'acc/gen':>12} {'mean len':>9} "
          f"{'prefill t/s':>12} {'decode t/s':>11} {'head':>14}")
    for r in results:
        if "failed" in r:
            print(f"{r['label']:<10} FAILED  {r['failed'][:60]}")
            continue
        a = r["accept"]
        print(f"{r['label']:<10} {'n/a' if a is None else f'{a*100:7.2f}%':>8} "
              f"{r['draft_n_accepted']:>5}/{r['draft_n']:<6} "
              f"{r['mean_acc_len'] or 0:9.2f} {r['prefill_tps_mean'] or 0:12.2f} "
              f"{r['decode_tps_mean'] or 0:11.2f} "
              f"{r['head_identity'][:12]:>14}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
