#!/usr/bin/env python3
"""Re-read `cb` with the memory water level as an ENFORCED entry condition.

WHY THIS EXISTS. 2026-09-20 21:44, the joint capture (§EN-350) measured cb = 11.26 ms
(9.2% of the step). Another run of nominally the same thing had read cb = 74.18 ms (29.9%).
That is 6.6x on the same engine, and the difference was attributed to the memory water level
(startup free = 84% on one, high swap on the other). Everything downstream of that number --
the priority of the victim rule, whose ceiling is computed as a fraction of cb -- is therefore
resting on a quantity that moves 6x with a variable nobody recorded.

The fix is not a better theory of which variable matters. It is to make the water level an
ENTRY CONDITION the way `--min-headroom-mb` already does: you declare the band, the tool
refuses to measure outside it, and every reading ships with the water vector it was taken in
so two readings can be compared at all.

WHAT IT CANNOT DO. It cannot induce a water level. On this box there is no way to release
swap without root (`purge` needs a password, there is no swapoff), and deliberately
allocating RAM to push the box into a band would corrupt whatever the other session on this
machine is measuring. So the tool CONDITIONS (waits for a band, refuses if it never arrives)
rather than INDUCES. A band that cannot be entered is reported as a refusal, never as a
reading. That refusal is itself a result: a number that can only be reproduced in a band you
cannot re-enter is not a number you can build a priority ordering on.

WHAT IT MEASURES. `cb` -- the host-side expert-cache fill (ensure) cost in the segmented
dispatcher -- from `CGC-DECPROF` step rows, via joint_reconcile's parser (work rows only,
sum_gap > 0, the same rule cb_miss_regression.py uses). Reported for the whole run and split
into thirds, because cb is expected to move within a run as the pool warms, and a single
median hides exactly the drift being investigated.

RESULT OF THAT PLAN (2026-09-20 22:3x, §EN-351) -- READ BEFORE USING A BAND AS A GATE.
The plan was falsified three ways; see docs/CB_WATER_CONTROLLED_REREAD_2026-09-20.md.
  (1) The pressure band is NOT ENTERABLE: `--max-usable-pct 15 --max-free-pct 25` timed out
      after 60 s. The pressured state is produced BY a run and the box self-recovers to
      usable 56-64% within ~2 min of one ending, so you cannot stand in it before starting.
  (2) The water READING is unstable: on an idle box, free moved 5.62 GiB -> 0.06 GiB in 30 s.
      A gate sensitive to when you press enter is not a controlled condition.
  (3) Wrong sign: back-to-back same-carrier runs, run 2 entered BETTER on every axis
      (usable 63.8 vs 51.0, headroom 10447 vs 8360) yet cb rose 23.89 -> 33.87 ms (+42%)
      and the step 139 -> 238 ms. n=4 correlations: usable +0.940, headroom +0.939,
      cached -0.836, swap +0.761 -- none clears the 0.950 critical for n=4.
So: `--min-*`/`--max-*` are RECORDING AND FILTERING, not a gate. What actually controls cb is
the CARRIER plus the POSITION WITHIN THE RUN (first_half -> last_half decays 1.63-1.82x in
5/5 runs, larger than the 1.42x run-to-run spread on one carrier). Always compare cb within
one carrier and quote which half it came from.

USAGE
    python3 scripts/check/cb_headroom_probe.py --selftest
    python3 scripts/check/cb_headroom_probe.py --water                # read the water, launch nothing
    python3 scripts/check/cb_headroom_probe.py --band warm-cache --runs 2
    python3 scripts/check/cb_headroom_probe.py --min-cached-gib 5.0 --tag hot2
    python3 scripts/check/cb_headroom_probe.py --print-registry

EXIT CODES  0 ok | 2 refused (band not enterable / window not exclusive) | 3 ran but no rows
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import joint_reconcile as jr                      # noqa: E402
import prefill_certifiability as pc               # noqa: E402
import window_sentinel as wsent                   # noqa: E402
import decode_step_profile as dsp                 # noqa: E402

LAUNCHER = ROOT / "scripts" / "run_server.sh"
REGISTRY = ROOT / "Backup" / "phase_decomp" / "cb_water_registry.jsonl"
OUT_DIR = ROOT / "Backup" / "phase_decomp"

# The carrier that produced cb = 11.26 ms on 2026-09-20 (docs/JOINT_CAPTURE_2026-09-20.md §1).
# Reproduced verbatim so the new reading is comparable to that one; anything else is a
# different experiment wearing the same name.
CAPTURE_ENV = {
    "CGC_DECODE_PROFILE": "1",
    "CGC_DECODE_PROFILE_ALL": "1",
    "CGC_GPU_TIMING": "1",
    "CGC_HOOK_SPLIT": "1",
    "LLAMA_EXPERT_CACHE_BATCH_DBG": "1",
}
# MTP-on production carrier, for readings that are meant to describe the delivered path.
PROD_ENV = {"CGC_SERVER_MTP": "1", "CGC_SERVER_PROFILE": "prod25",
            "CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256"}

CARRIERS = {"joint": CAPTURE_ENV, "prod25": {**CAPTURE_ENV, **PROD_ENV}}

PROBE_PROMPT = "請用繁體中文寫一段約三百字的短文，介紹巴黎的歷史、建築與文化，並說明它們之間的關係。"
PROBE_MAX_TOKENS = 400

# Named bands. A band is a conjunction of bounds on the water vector; a variable that is not
# mentioned is unconstrained (and still recorded). These are starting points chosen from what
# this box has actually been observed at, not derived quantities -- change them on the command
# line, and read the recorded vector to decide what the band SHOULD have been.
BANDS = {
    "any": {},
    "cold-cache": {"max_cached_gib": 3.0},
    "warm-cache": {"min_cached_gib": 5.0},
    "low-swap": {"max_swap_mib": 1500.0},
    "high-swap": {"min_swap_mib": 3000.0},
    "roomy": {"min_usable_pct": 60.0, "min_headroom_mb": 8000},
}

# min_<k> / max_<k> over the water vector keys.
_BOUND_KEYS = ("usable_pct", "headroom_mb", "swap_mib", "cached_gib", "anon_gib", "free_pct")


# ----------------------------------------------------------------- water level

def water() -> dict:
    """One reading of everything that could plausibly be 'the water level'.

    They disagree with each other by construction -- `memory_pressure` free%, vm_stat
    usable%, swap used, and the page cache move for different reasons -- which is the whole
    reason for recording all of them instead of electing one up front. See
    prefill_certifiability.mem_state for why free% was falsified as a proxy.
    """
    m = pc.mem_state()
    w = {
        "t": round(time.time(), 1),
        "usable_pct": round(m["usable_pct"], 2),
        "usable_gib": round(m["usable_gib"], 2),
        "free_gib": round(m["free_gib"], 2),
        "cached_gib": round(m["cached_gib"], 2),
        "anon_gib": round(m["anon_gib"], 2),
        "swap_mib": round(m["swap_used_mib"], 1),
    }
    try:
        mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True,
                            timeout=20).stdout
        mm = re.search(r"free percentage:\s*(\d+)%", mp)
        w["free_pct"] = int(mm.group(1)) if mm else None
    except Exception as e:  # noqa: BLE001 - a probe must not cancel a measurement
        w["free_pct"] = None
        w["_mp_error"] = str(e)
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=20).stdout
        tot = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            mm = re.search(key + r":\s+(\d+)", vs)
            if mm:
                tot += int(mm.group(1))
        w["headroom_mb"] = tot * 16384 // 1048576
    except Exception:
        w["headroom_mb"] = None
    return w


def check_band(w: dict, band: dict) -> list[str]:
    """Reasons the reading is OUTSIDE the band. Empty list == inside."""
    bad = []
    for k in _BOUND_KEYS:
        v = w.get(k)
        lo, hi = band.get("min_" + k), band.get("max_" + k)
        if v is None:
            if lo is not None or hi is not None:
                bad.append(f"{k}: unreadable (bounded by {lo}/{hi})")
            continue
        if lo is not None and v < lo:
            bad.append(f"{k}={v} < min {lo}")
        if hi is not None and v > hi:
            bad.append(f"{k}={v} > max {hi}")
    return bad


def wait_band(band: dict, timeout: float, poll: float = 10.0):
    """Block until the box is inside `band`. Refuse -- never measure -- on timeout."""
    t0 = time.time()
    while True:
        w = water()
        bad = check_band(w, band)
        if not bad:
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "water": w}
        if time.time() - t0 > timeout:
            return {"ok": False, "waited_s": round(time.time() - t0, 1),
                    "water": w, "reasons": bad}
        print("    water out of band (%s) -- waiting %.0fs/%.0fs"
              % ("; ".join(bad), time.time() - t0, timeout), flush=True)
        time.sleep(poll)


class Sampler(threading.Thread):
    """Sample the water level while the run is in flight.

    Entry state is not enough: the run itself changes the water level (it reads a 13.66 GiB
    model and fills an 8 GiB pool), so a run can leave the band it entered in. Without the
    trajectory that is invisible, and the reading gets filed under a band it was not taken in.
    """

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(water())
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        if not self.samples:
            return {"n": 0}
        out = {"n": len(self.samples)}
        for k in ("usable_pct", "swap_mib", "cached_gib", "free_pct", "headroom_mb"):
            v = [s.get(k) for s in self.samples if s.get(k) is not None]
            if v:
                out[k] = {"min": min(v), "med": round(st.median(v), 2), "max": max(v)}
        return out


# ------------------------------------------------------------------- capture

def launch(env_extra: dict, launch_log: str):
    env = dict(os.environ)
    env.update(env_extra)
    lf = open(launch_log, "w")
    p = subprocess.Popen(["bash", str(LAUNCHER)], cwd=str(ROOT), env=env, stdout=lf,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return p, launch_log


def probe_chat(port: str, prompt: str, max_tokens: int) -> dict:
    """The 884-probe request: same endpoint, same prompt, same temperature as the gate."""
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    tt = r.get("usage", {}) or {}
    return {"wall_s": round(time.time() - t0, 2),
            "completion_tokens": tt.get("completion_tokens"),
            "answer_head": ((r.get("choices") or [{}])[0].get("message", {}).get("content", "")
                            or "")[:40]}


def cb_stats(log: str) -> dict:
    """cb from the step rows, whole run and by thirds."""
    steps = [jr.reconcile_step(s) for s in jr.parse_log(Path(log)) if s["layers_rows"]]
    work = [r for r in steps if r["sum_gap"] > 0]
    if not work:
        return {"n_all_rows": len(steps), "n_work_rows": 0}

    def chunk_stats(rows: list[dict]) -> dict:
        if not rows:
            return {}
        tot = [r["total"] for r in rows]
        cb = [r["cb"] for r in rows]
        return {
            "n": len(rows),
            "cb_ms": round(st.median(cb), 2),
            "cb_pct": round(st.median([100.0 * c / t for c, t in zip(cb, tot) if t]), 2),
            "total_ms": round(st.median(tot), 2),
            "wait_ms": round(st.median([r["wait"] for r in rows]), 2),
            "cb_max_ms": round(max(cb), 2),
        }

    n = len(work)
    third = max(n // 3, 1)
    return {
        "n_all_rows": len(steps),
        "n_work_rows": n,
        "all": chunk_stats(work),
        "first_half": chunk_stats(work[: n // 2]),
        "last_half": chunk_stats(work[n // 2:]),
        "third_1": chunk_stats(work[:third]),
        "third_2": chunk_stats(work[third:2 * third]),
        "third_3": chunk_stats(work[2 * third:]),
    }


# ------------------------------------------------------------------ one run

def run_once(tag: str, args) -> dict:
    port = str(args.port)
    env = CARRIERS[args.carrier]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llog = str(out_dir / f"cbwater_{tag}.launch.log")

    if wsent.others():
        return {"tag": tag, "refused": "another session is running; the reading would not be mine"}

    gate = wait_band(args.band, args.band_timeout, args.band_poll)
    rec: dict = {"tag": tag, "carrier": args.carrier, "env": env,
                 "band": args.band, "entry": gate}
    if not gate["ok"]:
        rec["refused"] = ("band not enterable in %.0fs: %s"
                          % (gate["waited_s"], "; ".join(gate["reasons"])))
        return rec
    print("  water in band after %.0fs: %s" % (gate["waited_s"], gate["water"]), flush=True)

    before = {int(x) for x in subprocess.run(["pgrep", "-f", "build/bin/llama-server"],
                                             capture_output=True, text=True).stdout.split()}
    sampler = Sampler(args.sample_interval)
    sampler.start()
    p, _ = launch(env, llog)
    try:
        ok, why = dsp.wait_healthy(port, args.ready_timeout, llog)
        rec["launch_ok"], rec["launch_why"] = ok, why
        if not ok:
            rec["refused"] = f"launch failed: {why}"
            return rec
        rec["probe"] = probe_chat(port, PROBE_PROMPT, PROBE_MAX_TOKENS)
        time.sleep(2)
    finally:
        sampler.stop()
        sampler.join(timeout=10)
        rec["traj"] = sampler.summary()
        dsp.stop(dsp.server_pid(before))
        time.sleep(1.5)
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.terminate()

    _, srv_log = dsp.arm_text(llog)
    rec["server_log"] = srv_log
    if not srv_log:
        rec["refused"] = "no server log found -- nothing to parse"
        return rec
    rec["cb"] = cb_stats(srv_log)
    # The carrier is only comparable if it is the SAME carrier. `L4 pool capacity=179` is the
    # signature of the joint-capture (profile=off) carrier; if it changed, the two readings
    # are not two samples of one population and must not be averaged.
    try:
        txt = Path(srv_log).read_text(errors="replace")
        mm = re.search(r"L4 pool capacity=(\d+)", txt)
        rec["pool_capacity"] = int(mm.group(1)) if mm else None
        mm = re.search(r"loading model '([^']+)'", txt)
        rec["model"] = mm.group(1).split("/")[-1] if mm else None
    except OSError:
        pass
    return rec


# ------------------------------------------------------------------ registry

def registry_rows() -> list[dict]:
    if not REGISTRY.exists():
        return []
    rows = []
    for line in REGISTRY.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return rows


def append_registry(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_registry() -> None:
    rows = [r for r in registry_rows() if not r.get("refused")]
    if not rows:
        print("registry empty (%s)" % REGISTRY)
        return
    print("\n%-22s %-9s %7s %7s %7s %8s %8s  %s"
          % ("tag", "carrier", "cb_ms", "cb_pct", "total", "swap_mib", "cached", "band"))
    for r in rows:
        w = (r.get("entry") or {}).get("water") or {}
        cb = (r.get("cb") or {}).get("all") or {}
        print("%-22s %-9s %7s %7s %7s %8s %8s  %s"
              % (r.get("tag", "-"), r.get("carrier", "-"), cb.get("cb_ms", "-"),
                 cb.get("cb_pct", "-"), cb.get("total_ms", "-"),
                 w.get("swap_mib", "-"), w.get("cached_gib", "-"),
                 json.dumps(r.get("band", {}), ensure_ascii=False)))
    ms = [(r.get("cb") or {}).get("all", {}).get("cb_ms") for r in rows]
    ms = [x for x in ms if x]
    if len(ms) > 1:
        print("\nspread of cb across %d readings: %.2f .. %.2f ms (%.1fx)"
              % (len(ms), min(ms), max(ms), max(ms) / min(ms)))


# ------------------------------------------------------------------ selftest

def cmd_selftest() -> int:
    ok = 0
    checks = []

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond)))
        if cond:
            ok += 1

    # 1. the gate admits inside the band and refuses outside it
    w = {"usable_pct": 60.0, "headroom_mb": 9000, "swap_mib": 100.0,
         "cached_gib": 2.0, "anon_gib": 4.0, "free_pct": 80}
    check("in band -> no reasons", check_band(w, BANDS["cold-cache"]) == [])
    check("out of band -> reason", len(check_band(w, BANDS["warm-cache"])) == 1)
    check("swap bound fires", any("swap_mib" in r for r in check_band(w, BANDS["high-swap"])))
    # 2. an unreadable bounded variable must refuse, not pass silently
    check("unreadable + bounded -> refuses",
          any("unreadable" in r for r in check_band({"cached_gib": None}, BANDS["warm-cache"])))
    check("unreadable + unbounded -> passes",
          check_band({"cached_gib": None, "swap_mib": 4000.0},
                     {"min_swap_mib": 3000.0}) == [])
    # 3. refusal, not a reading: wait_band must time out rather than measure
    r = wait_band({"min_usable_pct": 999.0}, timeout=0.0, poll=0.01)
    check("wait_band refuses on timeout", r["ok"] is False and r["reasons"])
    check("wait_band admits immediately", wait_band({}, timeout=0.0, poll=0.01)["ok"] is True)
    # 4. every bound key is checked on both sides
    w2 = dict(w)
    check("min side checked", check_band(w2, {"min_free_pct": 90}) != [])
    check("max side checked", check_band(w2, {"max_free_pct": 10}) != [])
    # 5. trajectory summary picks the extremes, not the entry value
    s = Sampler(interval=0.01)
    s.samples = [{"swap_mib": 10.0}, {"swap_mib": 50.0}, {"swap_mib": 30.0}]
    check("traj min/med/max", s.summary()["swap_mib"] == {"min": 10.0, "med": 30.0, "max": 50.0})
    # 6. cb_stats splits into thirds and reports a percentage of the step
    import tempfile
    tf = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    for i in range(12):  # 12 work rows: cb rising 1..12 ms, total fixed
        cb = 1.0 + i
        tf.write(f"CGC-DECPROF: step={i} segs=40 layers=40 total=100.0 ms | "
                 f"wait={80.0 - cb} ({(80.0 - cb):.1f}%) cb={cb} ({cb:.1f}%) "
                 f"submit=20.0 (20.0%) ntok=4\n")
        tf.write("CGC-DECPROF all: L0 wait=1.0 cb=%.3f submit=0.5 ms gpu=2.0 union=2.0 "
                 "gap=1.0 sg=1 n=1 st=1.0 en=2.0\n" % (cb / 40))
    tf.close()
    stt = cb_stats(tf.name)
    check("work rows kept", stt["n_work_rows"] == 12)
    check("thirds split", stt["third_1"]["n"] == 4 and stt["third_3"]["n"] == 4)
    check("cb rises across thirds", stt["third_1"]["cb_ms"] < stt["third_3"]["cb_ms"])
    check("cb_pct computed", abs(stt["all"]["cb_pct"] - 6.5) < 0.6)
    os.unlink(tf.name)
    # 7. registry round-trip (written to a temp path so the real one is not polluted)
    real = REGISTRY
    try:
        tmp = Path(tempfile.mkdtemp()) / "reg.jsonl"
        globals()["REGISTRY"] = tmp
        append_registry({"tag": "t1", "cb": {"all": {"cb_ms": 3.0}}})
        append_registry({"tag": "t2", "cb": {"all": {"cb_ms": 9.0}}})
        check("registry round-trip", len(registry_rows()) == 2)
    finally:
        globals()["REGISTRY"] = real

    for name, good in checks:
        print("  %s %s" % ("ok  " if good else "FAIL", name))
    print("\nselftest: %d/%d checks" % (ok, len(checks)))
    return 0 if ok == len(checks) else 1


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="re-read cb with the water level enforced")
    ap.add_argument("--band", default="any", choices=sorted(BANDS),
                    help="named entry band (default: any -- record, do not gate)")
    ap.add_argument("--min-usable-pct", type=float)
    ap.add_argument("--max-usable-pct", type=float)
    ap.add_argument("--min-headroom-mb", type=int)
    ap.add_argument("--min-swap-mib", type=float)
    ap.add_argument("--max-swap-mib", type=float)
    ap.add_argument("--min-cached-gib", type=float)
    ap.add_argument("--max-cached-gib", type=float)
    ap.add_argument("--min-free-pct", type=float)
    ap.add_argument("--max-free-pct", type=float)
    ap.add_argument("--runs", type=int, default=1,
                    help="consecutive captures; each one re-checks the band before launching")
    ap.add_argument("--carrier", default="joint", choices=sorted(CARRIERS))
    ap.add_argument("--port", default="8080")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--json", default="")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--band-timeout", type=float, default=120.0,
                    help="give up waiting for the band after this many seconds (a refusal)")
    ap.add_argument("--band-poll", type=float, default=10.0)
    ap.add_argument("--sample-interval", type=float, default=5.0)
    ap.add_argument("--water", action="store_true", help="print the water vector and exit")
    ap.add_argument("--print-registry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest()
    if args.print_registry:
        print_registry()
        return 0
    if args.water:
        w = water()
        print("water: %s" % json.dumps(w, ensure_ascii=False))
        print("in band %r: %s" % (args.band,
                                  "yes" if not check_band(w, BANDS[args.band]) else "no"))
        return 0

    band = dict(BANDS[args.band])
    for k in _BOUND_KEYS:
        lo = getattr(args, "min_" + k, None)
        hi = getattr(args, "max_" + k, None)
        if lo is not None:
            band["min_" + k] = lo
        if hi is not None:
            band["max_" + k] = hi
    args.band = band

    w = water()
    print("water now: %s" % json.dumps(w, ensure_ascii=False))
    print("band     : %s" % (json.dumps(band) if band else "{} (record only)"))
    bad = check_band(w, band)
    if bad:
        print("out of band now: %s" % "; ".join(bad))

    results = []
    rc = 0
    for i in range(args.runs):
        tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
        if args.runs > 1:
            tag = f"{tag}_r{i + 1}"
        print("\n=== run %d/%d  tag=%s ===" % (i + 1, args.runs, tag), flush=True)
        rec = run_once(tag, args)
        rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results.append(rec)
        if rec.get("refused"):
            print("REFUSED: %s" % rec["refused"])
            rc = 2
            continue
        cb = rec.get("cb", {}).get("all", {})
        if not cb:
            print("no work rows in %s" % rec.get("server_log"))
            rc = rc or 3
        else:
            print("cb %s ms (%s%% of %s ms)  n=%s  [1st third %s -> last third %s]"
                  % (cb["cb_ms"], cb["cb_pct"], cb["total_ms"], cb["n"],
                     rec["cb"].get("third_1", {}).get("cb_ms"),
                     rec["cb"].get("third_3", {}).get("cb_ms")))
            print("water during run: %s" % json.dumps(rec.get("traj", {}), ensure_ascii=False))
        append_registry(rec)

    out = args.json or str(OUT_DIR / ("cb_water_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
    Path(out).write_text(json.dumps({"band": args.band, "water_before": w,
                                     "runs": results}, ensure_ascii=False, indent=1))
    print("\njson -> %s" % out)
    print_registry()
    return rc


if __name__ == "__main__":
    sys.exit(main())
