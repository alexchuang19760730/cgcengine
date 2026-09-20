#!/usr/bin/env python3
"""A window is not certified by "thermal == NOMINAL and memory is free".

WHY THIS EXISTS. 2026-09-20, 18:0x: three arms of a decode A/B came back at 2.17 / 1.36 / 1.35 t/s
against 10.64 t/s for the SAME cell at 16:10 -- while `notifyutil -g ...thermalpressurelevel` said
NOMINAL for the whole run, `mem_state()` said 54% usable, and no other session was running. Both
conditions the existing gates check were satisfied and the box was ~5x slow. That window then
silently corrupted two rounds of measurement before anyone noticed (one of them mine).

The lesson is that "is the box healthy" is a PHYSICAL question, not a bookkeeping one. It has to be
answered by MEASURING THROUGHPUT against a known-shape reference, on the same box, immediately
before the run that matters.

WHAT IT MEASURES. The prefill cell of the frozen production profile (`-p 2048 -n 0`, `-b 5632`).
Prefill is chosen because it is the axis that is dominated by sustained matmul throughput, so it
tracks clock/power state tightly, and because it is cheap (one pass) and far less noisy than the
decode cell (whose single-arm spread is +/-27%). A healthy box reads ~276-292 t/s; see
`window_sentinel_ref.json` for the band and where each sample came from.

USAGE
    python3 scripts/check/window_sentinel.py                     # measure + verdict (exit 0/3)
    python3 scripts/check/window_sentinel.py --json /tmp/s.json
    python3 scripts/check/window_sentinel.py --dry-run           # print the command, launch nothing
    python3 scripts/check/window_sentinel.py --record --this-box-is-healthy
                                                                 # refresh the reference band

EXIT CODES  0 healthy (or dry-run) | 2 refused (another session / no window) | 3 DEGRADED
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = "/Users/alexchuang/Documents/flashkv-devserver"
HERE = os.path.join(ROOT, "scripts/check")
sys.path.insert(0, HERE)
os.chdir(ROOT)

import llama_bench_matrix as lbm                    # noqa: E402
from prefill_certifiability import mem_state        # noqa: E402
import thermal_pressure as tp                       # noqa: E402

REF_PATH = os.path.join(HERE, "window_sentinel_ref.json")
PROFILE = "prefill250"
SHAPE = ["-p", "2048", "-n", "0"]
BIN = os.path.join(ROOT, "src/llama.cpp/build/bin/llama-bench")
MODEL = os.path.join(ROOT, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")
OTHER = "[l]lama|[p]rod_profile|[m]123_oracle|[s]erver_window|[d]ecode_sweep|[m]tp_accept"
# A healthy box is not "somewhere above a bar" -- it is within a band. 0.85 of the recorded median
# is the line: the recorded healthy samples span 275.65-300.43 (1.09x), so a reading 15% under the
# median cannot be a healthy sample of that same population, and the degraded window measured
# on 2026-09-20 was ~0.2x of it (i.e. nowhere near the boundary -- the line is not a knife edge).
DEFAULT_FRAC = 0.85


def full_command():
    env = {k: str(v) for k, v in lbm.resolve(PROFILE, {})["env"].items() if v not in (None, "")}
    cmd = [BIN, "-m", MODEL, "-ngl", "99", "-t", "8",
           "-expert-cache", env.get("CGC_EXPERT_CACHE_BYTES", "8589934592"),
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--load-mode", "none", "-o", "json",
           "-b", "5632", "-ub", "5632"] + SHAPE
    return cmd, env


PY_SCRIPTS = ("prod_profile", "m123_oracle", "server_window", "decode_sweep",
              "mtp_accept", "llama_bench_matrix", "profile_duo")


def _ps(pid, key):
    return subprocess.run(["ps", "-o", key + "=", "-p", pid],
                          capture_output=True, text=True).stdout.strip()


def others():
    """pids actually RUNNING a measured binary -- not shells whose argv merely quotes one.

    2026-09-20 20:1x: another line's post-run `bash -c '... pgrep -f llama-server ...' sleep 180`
    wrapper kept every gate red for three minutes after its server had already exited, because its
    argv contains the pattern as TEXT. A shell is not a measurement. The binaries (`llama-*`) and
    the python drivers are; anything else is a mention and must not block.
    Being wrong in the other direction is much cheaper: a false positive only means "did not run".
    """
    pids = subprocess.run(["pgrep", "-f", OTHER], capture_output=True, text=True).stdout.split()
    blocking = []
    for pid in pids:
        comm = _ps(pid, "comm")
        if comm.startswith("llama"):
            blocking.append(pid)
        elif comm.startswith("python"):
            if any(s in _ps(pid, "args") for s in PY_SCRIPTS):
                blocking.append(pid)
    return blocking


def measure():
    cmd, env = full_command()
    e = dict(os.environ)
    e.update(env)
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    ts = None
    try:
        arr = json.loads(p.stdout[p.stdout.index("["):p.stdout.rindex("]") + 1])
        for row in arr:
            if int(row.get("n_gen", 0)) == 0:
                ts = float(row.get("avg_ts"))
    except Exception:
        pass
    return ts, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frac", type=float, default=DEFAULT_FRAC)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--this-box-is-healthy", action="store_true",
                    help="required with --record: the reference must come from a box you have "
                         "independently certified, or the sentinel will certify the degradation")
    args = ap.parse_args()

    cmd, _ = full_command()
    print("sentinel shape: -p 2048 -n 0 -b 5632 (prefill cell of profile %s)" % PROFILE)
    print("argv:", " ".join(cmd))
    if args.dry_run:
        return 0

    ps = others()
    if ps:
        print("refused: another session is running (pids %s)" % ",".join(ps))
        return 2

    t = tp.stamp()
    m = mem_state()
    print("thermal %s | usable %.1f%% | free %.2f GiB" % (t.get("label"), m["usable_pct"], m["free_gib"]))
    ts, rc = measure()
    print("measured %.2f t/s (rc=%s)" % (ts or -1.0, rc))

    if args.record:
        if not args.this_box_is_healthy:
            print("refused: --record needs --this-box-is-healthy. A reference recorded on a degraded "
                  "box would certify the degradation instead of catching it.")
            return 2
        if not ts:
            print("refused: no reading to record")
            return 2
        ref = {"shape": " ".join(SHAPE), "batch": 5632, "profile": PROFILE,
               "median": ts, "samples": [ts], "frac": args.frac,
               "note": "recorded %s on a box the operator certified healthy" % t.get("t")}
        with open(REF_PATH, "w") as f:
            json.dump(ref, f, ensure_ascii=False, indent=2)
        print("reference -> %s (median %.2f)" % (REF_PATH, ts))
        return 0

    if not os.path.exists(REF_PATH):
        print("DEGRADED? no reference band at %s -- run --record on a certified-healthy box" % REF_PATH)
        return 3
    ref = json.load(open(REF_PATH))
    floor = ref["median"] * args.frac
    ok = ts is not None and ts >= floor
    verdict = "HEALTHY" if ok else "DEGRADED"
    print("\nreference median %.2f (%s)  floor %.2f (%.0f%%)  ->  %s"
          % (ref["median"], ref.get("note", "recorded"), floor, args.frac * 100, verdict))
    if not ok:
        print("  a window like this is NOT a measurement window: the existing gates (thermal key, "
              "usable memory) both pass on it -- measured 2026-09-20 18:1x")
    if args.json:
        json.dump({"t_s": ts, "ref_median": ref["median"], "floor": floor, "verdict": verdict,
                   "thermal": t, "mem": m}, open(args.json, "w"), ensure_ascii=False, indent=2)
        print("json ->", args.json)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
