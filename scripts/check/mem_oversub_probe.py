#!/usr/bin/env python3
"""Memory-oversubscription probe: does swap/compression sit inside the decode step?

Question
--------
Production runs `--load-mode none` (= --no-mmap): 12.7 GB of weights become
ANONYMOUS (swappable/compressible) pages, plus a ~6-8 GiB expert pool, on a
16 GB box.  If the hot working set does not fit, every step pays decompression
and the "19 GB/s effective bandwidth" is partly inflate cost, not DRAM read.

This instrument measures it with ZERO engine code: it samples the VM counters
that the kernel already keeps (compressions / decompressions / swapins /
swapouts / pageouts / compressor occupancy) on a wall clock, while a decode
run executes, then reports:

  1. total counter deltas across the run,
  2. the per-sample time series (so load phase can be separated from steady
     decode),
  3. the steady-window delta, where "steady" starts at the first bench line
     that proves the model is loaded.

The decisive readout is (3): if `decompressions` is ~0 while decode is running,
oversubscription is NOT on the critical path and the hypothesis is dead.

Guard
-----
Refuses to launch if any other llama measurement process is alive.  This probe
never kills anything -- see the 2026-09-18 cross-session incident.

Usage
-----
    python3 scripts/check/mem_oversub_probe.py --selftest
    python3 scripts/check/mem_oversub_probe.py                     # delivery cell
    python3 scripts/check/mem_oversub_probe.py --reps 5 --gen 256
    python3 scripts/check/mem_oversub_probe.py --arm prefill250 --prompt 2048 --gen 16 --depths 0
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts" / "check" / "llama_bench_matrix.py"

# Counters that matter.  Everything else from vm_stat is noise for this question.
KEYS = [
    "Decompressions",
    "Compressions",
    "Swapins",
    "Swapouts",
    "Pageouts",
    "Pageins",
    "Faults",
    "Pages stored in compressor",
    "Pages occupied by compressor",
]

PAGE = 4096


def vm_snapshot():
    """One sample: returns (t, {key: value}, swap_used_mib)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r"\s*(.+?):\s+([\d.]+)\.?\s*$", line)
        if m:
            d[m.group(1).strip()] = float(m.group(2))
    swap = None
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", s)
    if m:
        swap = float(m.group(1))
    return time.time(), d, swap


class Sampler(threading.Thread):
    def __init__(self, hz):
        super().__init__(daemon=True)
        self.hz = hz
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            t, d, swap = vm_snapshot()
            self.samples.append({"t": t, "vm": d, "swap_used_mib": swap})
            self.stop.wait(1.0 / self.hz)


def guard():
    """Refuse to launch on top of somebody else's measurement."""
    p = subprocess.run(["pgrep", "-fl", "llama-server|llama-bench|run_server"],
                       capture_output=True, text=True)
    mine = {"mem_oversub_probe", "pgrep"}
    others = [l for l in p.stdout.splitlines()
              if l.strip() and not any(m in l for m in mine)]
    return others


def run_bench(args, tag):
    """Launch the delivery decode shape, timestamping every output line."""
    cmd = [sys.executable, str(BENCH),
           "--arms", args.arm,
           "--reps", str(args.reps),
           "--prompt", str(args.prompt),
           "--gen", str(args.gen),
           "--depths", str(args.depths),
           "--batch", str(args.batch),
           "--ctx-size", str(args.ctx_size),
           "--warm-skip", str(args.warm_skip),
           "--spec-type", args.spec_type]
    if args.json:
        cmd += ["--json", str(args.json)]
    # `--load-mode` is NOT a llama_bench_matrix.py flag.  It only forwards the
    # value that run_server.sh resolved for the profile, so the way to change it
    # is an env override consumed by run_server.sh (llama_bench_matrix.py:196 /
    # resolve() injects extra_env into the CGC_DUMP_ENV probe).
    env = None
    if args.load_mode:
        env = dict(__import__("os").environ)
        env["CGC_SERVER_LOAD_MODE"] = args.load_mode
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    lines = []
    for line in proc.stdout:
        lines.append({"t": time.time(), "line": line.rstrip()})
    rc = proc.wait()
    if rc != 0:
        # A rc!=0 bench means NOTHING was measured.  Do not write a probe JSON --
        # a 1-sample file looks like a real result and silently poisons later
        # comparisons (hit 2026-09-21: `--load-mode` was rejected, probe still
        # emitted a "successful" artifact with all-zero deltas).
        sys.stderr.write("\n".join(r["line"] for r in lines[-20:]) + "\n")
        raise SystemExit(f"INVALID: bench exited rc={rc}; nothing was measured.")
    return {"cmd": cmd, "t0": t0, "t1": time.time(), "rc": rc, "lines": lines}


def steady_start(bench):
    """First timestamp at which the model is provably loaded.

    llama-bench prints `load time = ... ms` after the model is resident.
    Fall back to the run start if we never see it.
    """
    for rec in bench["lines"]:
        if re.search(r"load time\s*=", rec["line"]):
            return rec["t"]
    return bench["t0"]


def series_between(samples, t0, t1):
    return [s for s in samples if t0 <= s["t"] <= t1]


def delta(series, key):
    if len(series) < 2:
        return None
    return series[-1]["vm"].get(key, 0) - series[0]["vm"].get(key, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="prod25-stream")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depths", type=int, default=512)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--warm-skip", type=int, default=64)
    ap.add_argument("--spec-type", default="draft-mtp")
    ap.add_argument("--load-mode", default=None,
                    help="forwarded to llama-bench. none (= --no-mmap, today's delivery "
                         "default, anonymous weight pages) vs mmap (file-backed, reclaimable).")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    others = guard()
    if others:
        print("REFUSED: other llama measurement processes are alive:")
        for l in others:
            print("   " + l)
        print("This probe never kills anything. Wait for a clean window.")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    a.json = a.json or str(ROOT / "Backup" / "phase_decomp" / f"mem_oversub_{stamp}.json")
    a.out = a.out or str(ROOT / "Backup" / "phase_decomp" / f"mem_oversub_{stamp}.probe.json")

    s = Sampler(a.hz)
    s.start()
    bench = run_bench(a, stamp)
    time.sleep(0.2)
    s.stop.set()
    s.join(timeout=5)

    t_load = steady_start(bench)
    steady = series_between(s.samples, t_load, bench["t1"])
    whole = s.samples

    print("=== memory oversubscription probe ===")
    print(f"window      : {bench['t1'] - bench['t0']:.1f} s   samples={len(whole)}")
    print(f"load done at: {t_load - bench['t0']:.1f} s into the run")
    print(f"steady sub-window: {len(steady)} samples over "
          f"{steady[-1]['t'] - steady[0]['t']:.1f} s\n")

    hdr = f"{'counter':32s}{'whole run':>14s}{'steady':>14s}{'steady/s':>12s}"
    print(hdr)
    print("-" * len(hdr))
    dur_s = max(steady[-1]["t"] - steady[0]["t"], 1e-9)
    res = {}
    for k in KEYS:
        dw = delta(whole, k)
        ds = delta(steady, k)
        rate = (ds / dur_s) if ds is not None else None
        res[k] = {"whole": dw, "steady": ds, "steady_per_s": rate}
        fmt = (f"{k:32s}"
               f"{'' if dw is None else f'{dw:,.0f}':>14s}"
               f"{'' if ds is None else f'{ds:,.0f}':>14s}"
               f"{'' if rate is None else f'{rate:,.1f}':>12s}")
        print(fmt)

    if delta(steady, "Decompressions") is not None:
        mb = delta(steady, "Decompressions") * PAGE / 1e6
        print(f"\nsteady decompressed volume ~= {mb:,.1f} MB over {dur_s:.1f} s"
              f"  =>  {mb / dur_s:,.1f} MB/s")

    sw0 = steady[0]["swap_used_mib"]
    sw1 = steady[-1]["swap_used_mib"]
    print(f"swap used   : {sw0:.0f} -> {sw1:.0f} MiB  (delta {sw1 - sw0:+.0f})")
    occ0 = steady[0]["vm"].get("Pages occupied by compressor", 0)
    occ1 = steady[-1]["vm"].get("Pages occupied by compressor", 0)
    print(f"compressor occupancy: {occ0 * PAGE / 1e6:,.0f} -> {occ1 * PAGE / 1e6:,.0f} MB")

    payload = {"bench": bench, "samples": s.samples, "deltas": res,
               "t_load": t_load, "steady_n": len(steady),
               "steady_dur_s": dur_s, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {a.out}")
    return 0


def selftest():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok += 1 if cond else 0

    t, d, swap = vm_snapshot()
    chk("vm_stat parses Decompressions", "Decompressions" in d)
    chk("vm_stat parses compressor occupancy", "Pages occupied by compressor" in d)
    chk("swapusage parses used MiB", swap is not None and swap >= 0)
    chk("counters are monotonic-looking (large)", d.get("Compressions", 0) > 0)

    s = [{"t": 0.0, "vm": {"Decompressions": 100}, "swap_used_mib": 0},
         {"t": 10.0, "vm": {"Decompressions": 300}, "swap_used_mib": 0}]
    chk("delta() computes 200", delta(s, "Decompressions") == 200)
    chk("short series -> None", delta([s[0]], "Decompressions") is None)
    chk("series_between windowing", len(series_between(s, 0.0, 10.0)) == 2)

    b = {"t0": 0.0, "lines": [{"t": 5.0, "line": "load time =   1234.56 ms"}]}
    chk("steady_start finds load time", steady_start(b) == 5.0)
    chk("steady_start falls back to t0", steady_start({"t0": 1.0, "lines": []}) == 1.0)

    chk("bench script exists", BENCH.exists())
    total = 10
    print(f"\nselftest: {ok}/{total}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
