#!/usr/bin/env python3
"""Is prefill 250 tok/s a spec, or a lottery?

THE QUESTION
------------
The 250 target has been *reached* repeatedly: 254.74 and 264.78 tok/s in `prefill250`
(`Backup/cgc_logs/llama_server_20260915_011146.log`), peak 276.59. And it has also been
missed by the same command: `pp2048 @ -ub 6144` over four independent launches gave

    276.59 / 198.84 / 176.18 / 122.68  t/s   ->  2.25x spread

while the three samples *inside* each launch agreed to 2.08-15.41 stddev. So the spread is
NOT measurement noise: it is state that differs between processes. A number you can hit
2.25x of the time is not a spec, it is a draw. Nothing downstream can be built on it.

THE HYPOTHESIS THIS TESTS
-------------------------
The candidate mechanism is memory state at launch: 16 GB unified memory has to hold
13.66 GiB of model + an 8 GiB expert pool + the compute buffer for a 6144-wide ubatch, and
`run_server.sh:427-434` already documents that +2 GiB of pool is enough to push the 6144
compute buffer out of the GPU entirely ("command buffer failed with status 5 / Insufficient
Memory ... 2026-09-15 用 CGC_SERVER_PROFILE=prefill250 實跑就是這樣炸的"). So the shape is
over-subscribed, and what the draw actually selects is *how much of the working set is
resident* -- i.e. how much of the run is served from page cache vs. from the 1404 MiB/s cold
device.

If that is the mechanism, throughput should track free memory at launch **monotonically**.
If it does, the fix is to make free memory a precondition (a gate) and 250 becomes a spec.
If it does not, the draw is something else and we should stop guessing.

WHAT IT MEASURES
----------------
N independent processes, same resolved command every time (the env is resolved by
`run_server.sh CGC_DUMP_ENV=1` through `llama_bench_matrix.py`, never re-typed here), with
the launch-time memory state recorded for each. Reported per run: free%, pp t/s, stddev, wall.

VERDICT
-------
    spread < 5%            -> certifiable; the number is a spec
    5% <= spread, monotone -> gated spec; ship the precondition, not the number
    spread >= 5%, not monotone -> still a draw; the mechanism is not launch memory

Usage:
    /opt/homebrew/bin/python3 scripts/check/prefill_certifiability.py --runs 5
    /opt/homebrew/bin/python3 scripts/check/prefill_certifiability.py --runs 5 --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "scripts" / "check" / "llama_bench_matrix.py"
PY = "/opt/homebrew/bin/python3"
TOTAL_BYTES = 16 * 1024**3


def mem_state() -> dict:
    """Launch-time memory state.

    NOTE ON WHICH VARIABLE MATTERS -- this is a correction, not a detail.

    The first version of this script recorded `free + inactive` as "usable" and called that the
    candidate mechanism. Five runs falsified that in the *inverted* direction (rho = -0.70): the
    run with the LEAST free memory was the ONLY one to clear 250 t/s. The reason is that free
    memory is not a proxy for the thing the engine actually depends on. The engine streams a
    13.66 GiB model off a 1404 MiB/s cold device; what matters is how much of that file is RESIDENT
    in the page cache (7721 MiB/s hot). And `Pages free` cannot distinguish "small cache" from
    "another process holds it" -- it moves with both. Measured right now on this box:
    free = 5.94 GiB but File-backed (the actual cache) = only 3.33 GiB, with 3.82 GiB anonymous.
    Low free can mean a LARGE cache (MacOS fills spare RAM with cache) or a LARGE app footprint.
    Those predict opposite throughputs, and the metric averaged them.

    So this records the page cache explicitly. `cached_gib` is the leading candidate; free/inactive
    are kept so the two can be compared on the same runs.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    d = {k.strip(): int(v) for k, v in re.findall(r"([A-Za-z\- ]+):\s+(\d+)\.", out)}
    free = d.get("Pages free", 0) * page
    inactive = d.get("Pages inactive", 0) * page
    speculative = d.get("Pages speculative", 0) * page
    cached = d.get("File-backed pages", 0) * page
    anon = d.get("Anonymous pages", 0) * page
    swap = 0
    try:
        so = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", so)
        if m:
            swap = float(m.group(1)) * 1024**2
    except Exception:
        pass
    return {
        "free_gib": free / 1024**3,
        "inactive_gib": inactive / 1024**3,
        "speculative_gib": speculative / 1024**3,
        "usable_gib": (free + inactive + speculative) / 1024**3,
        "usable_pct": 100.0 * (free + inactive + speculative) / TOTAL_BYTES,
        "cached_gib": cached / 1024**3,      # File-backed pages: the page cache. Leading candidate.
        "anon_gib": anon / 1024**3,          # Anonymous pages: what other processes hold.
        "swap_used_mib": swap / 1024**2,
    }


def pp_from(json_path: Path) -> list[dict]:
    """Pull the pp row(s) out of the matrix harness summary json."""
    if not json_path.exists():
        return []
    try:
        res = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return []
    rows = []
    for arm in res:
        for r in arm.get("rows", []):
            if r.get("n_prompt", 0) > 0:
                rows.append({"avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"],
                             "n_prompt": r["n_prompt"], "n_depth": r["n_depth"],
                             "n_batch": r.get("n_batch"), "incomplete": arm.get("incomplete"),
                             "error": arm.get("error")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5, help="independent processes (default 5)")
    ap.add_argument("--arm", default="prefill250")
    ap.add_argument("--prompt", default="2048", help="llama-bench -p")
    ap.add_argument("--gen", default="16", help="llama-bench -n (pp rows are read; tg is incidental)")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench -r (within-process samples)")
    ap.add_argument("--depths", default="0")
    ap.add_argument("--min-usable-pct", type=float, default=30.0,
                    help="refuse to launch below this free+inactive+speculative floor")
    ap.add_argument("--warm-runs", default="",
                    help="comma list of run indices to PRE-READ the model file before "
                         "(forces the page cache up). This is what breaks the order confound: "
                         "plain runs move cache and free together, warmed runs move cache up "
                         "while free falls.")
    ap.add_argument("--warm-file",
                    default=str(ROOT / "models" / "gguf"
                                / "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"))
    ap.add_argument("--json")
    args = ap.parse_args()

    warm_set = {int(x) for x in args.warm_runs.split(",") if x.strip()}
    runs: list[dict] = []
    for i in range(1, args.runs + 1):
        warm_s = None
        if i in warm_set:
            t = time.time()
            try:
                with open(args.warm_file, "rb") as f:
                    n = 0
                    while True:
                        b = f.read(4 << 20)
                        if not b:
                            break
                        n += len(b)
                warm_s = round(time.time() - t, 1)
                print(f"\n  [warm] read {n/1024**3:.2f} GiB of {Path(args.warm_file).name} "
                      f"in {warm_s}s to force the page cache up", flush=True)
            except OSError as e:
                print(f"  [warm] FAILED: {e}", flush=True)
        # Recorded AFTER warming: the post-warm state IS the launch state.
        pre = mem_state()
        print(f"\n{'='*90}\n  run {i}/{args.runs}   launch state: "
              f"free={pre['free_gib']:.2f} GiB inactive={pre['inactive_gib']:.2f} "
              f"usable={pre['usable_pct']:.1f}%  swap={pre['swap_used_mib']:.0f} MiB\n{'='*90}",
              flush=True)
        if pre["usable_pct"] < args.min_usable_pct:
            print(f"  SKIP: usable {pre['usable_pct']:.1f}% < floor {args.min_usable_pct}%; "
                  f"launching here would measure the OOM cliff, not the shape.", flush=True)
            runs.append({"run": i, "skipped": True, "pre": pre})
            continue

        jpath = Path("/tmp") / f"prefill_certifiability_{i}.json"
        cmd = [PY, str(MATRIX), "--arms", args.arm, "--prompt", args.prompt,
               "--gen", args.gen, "--depths", args.depths, "--reps", str(args.reps),
               "--json", str(jpath)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        wall = time.time() - t0
        sys.stdout.write(proc.stdout[-2500:])
        if proc.returncode != 0:
            sys.stdout.write(proc.stderr[-1200:])
        rows = pp_from(jpath)
        for r in rows:
            print(f"  -> pp{r['n_prompt']} d={r['n_depth']} = {r['avg_ts']:.2f} ± "
                  f"{r['stddev_ts']:.2f} t/s  (b={r['n_batch']}, {wall:.0f}s wall, rc={proc.returncode})",
                  flush=True)
        runs.append({"run": i, "skipped": False, "pre": pre, "wall_s": round(wall, 1),
                     "warm_s": warm_s, "rc": proc.returncode, "pp": rows,
                     "err": None if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1:]})
        time.sleep(5)  # let Metal release the GPU buffers before the next process

    # ---- verdict ---------------------------------------------------------------------------
    got = [r for r in runs if not r["skipped"] and r["pp"] and not r["pp"][0]["incomplete"]]
    print(f"\n{'='*90}\n  SUMMARY -- {args.arm}  pp{args.prompt} @ -ub 6144\n{'='*90}")
    print(f"{'run':>4s} {'usable%':>8s} {'free GiB':>9s} {'cache GiB':>10s} {'anon GiB':>9s} "
          f"{'swap MiB':>9s} {'pp t/s':>9s} {'±':>7s} {'wall s':>7s}")
    for r in runs:
        if r["skipped"]:
            print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
                  f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
                  f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
                  f"{r['pre']['swap_used_mib']:>9.0f}   (skipped: below floor)")
            continue
        if not r["pp"]:
            print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
                  f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
                  f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
                  f"{r['pre']['swap_used_mib']:>9.0f}   FAILED rc={r['rc']}")
            continue
        p = r["pp"][0]
        mark = "  [INCOMPLETE]" if p["incomplete"] else ""
        print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
              f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
              f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
              f"{r['pre']['swap_used_mib']:>9.0f} {p['avg_ts']:>9.2f} {p['stddev_ts']:>7.2f} "
              f"{r['wall_s']:>7.0f}{mark}")

    if len(got) < 2:
        print(f"\nVERDICT: UNDECIDED -- only {len(got)} usable run(s); need >= 2 to talk about spread.")
        return 2

    vals = [r["pp"][0]["avg_ts"] for r in got]
    lo, hi = min(vals), max(vals)
    spread = 100.0 * (hi - lo) / lo
    within = max(r["pp"][0]["stddev_ts"] / r["pp"][0]["avg_ts"] for r in got) * 100.0

    # Monotonicity -- Spearman rho, no scipy needed: rank both, Pearson on ranks.
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        rk = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    def spearman(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else float("nan")

    ts = [r["pp"][0]["avg_ts"] for r in got]
    rho_free = spearman([r["pre"]["usable_pct"] for r in got], ts)
    rho_cache = spearman([r["pre"].get("cached_gib", 0.0) for r in got], ts)
    rho_anon = spearman([r["pre"].get("anon_gib", 0.0) for r in got], ts)
    # The verdict keys off whichever variable actually explains the spread. rho_cache is the
    # leading candidate; rho_free is kept because the first version of this script guessed it.
    rho = rho_cache if abs(rho_cache) >= abs(rho_free) else rho_free
    which = "page cache (File-backed)" if rho is rho_cache else "free+inactive"

    print(f"\n  runs usable      : {len(got)}/{args.runs}")
    print(f"  range            : {lo:.2f} .. {hi:.2f} t/s   (spread {spread:.1f}%)")
    print(f"  worst within-run : {within:.1f}%  (llama-bench -r {args.reps}, same process)")
    print(f"  spearman rho(page-cache GiB, t/s) : {rho_cache:+.2f}   <- leading candidate")
    print(f"  spearman rho(usable%, t/s)        : {rho_free:+.2f}")
    print(f"  spearman rho(anon GiB, t/s)       : {rho_anon:+.2f}")
    print(f"  CONFOUND: n={len(got)} and run order == the order these variables moved. A perfect")
    print(f"            rank agreement cannot separate 'this variable causes it' from 'state")
    print(f"            accumulates with run order'. Treat |rho| > 0.7 here as a HYPOTHESIS.")
    print(f"  target           : 250 t/s")
    for r in got:
        print(f"    run {r['run']}: {r['pp'][0]['avg_ts']:7.2f} t/s  -> "
              f"{'AT/ABOVE' if r['pp'][0]['avg_ts'] >= 250 else 'below':>9s} target")
    print()
    # The useful output is not "the spread" but "what is the REPEATABLE BAND". With one outlier in
    # five, the honest reading is the cluster -- the range just says an outlier exists.
    rest = sorted(vals)[:-1]
    rest_spread = 100.0 * (rest[-1] - rest[0]) / rest[0] if len(rest) > 1 else float("nan")
    n_hit = sum(1 for v in vals if v >= 250.0)
    print(f"  repeatable band  : {rest[0]:.2f} .. {rest[-1]:.2f} t/s  ({len(rest)} runs, excluding "
          f"the single max; spread {rest_spread:.1f}%)")
    print(f"  cleared 250      : {n_hit}/{len(vals)} runs")
    print()
    if spread < 5.0:
        print("VERDICT: CERTIFIABLE -- spread < 5%. The number is a spec; use these runs as the "
              "reference and stop calling it a draw.")
    elif n_hit == 0:
        print(f"VERDICT: DRAW, TARGET UNREACHED AT THIS SHAPE -- 0/{len(vals)} runs cleared 250. "
              f"Quote the repeatable band ({rest[0]:.0f}-{rest[-1]:.0f} t/s), not the target.")
    else:
        print(f"VERDICT: DRAW -- {n_hit}/{len(vals)} runs cleared 250, spread {spread:.1f}%. "
              f"250 is an outlier, not the operating point; the repeatable band is "
              f"{rest[0]:.0f}-{rest[-1]:.0f} t/s ({rest_spread:.1f}% across {len(rest)} runs).")
    if len(got) >= 4 and abs(rho) >= 0.7:
        print(f"  leading explanation: {which} tracks t/s (rho={rho:+.2f}), BUT n={len(got)} with "
              f"run order confounded -- this is a hypothesis to test by forcing the two states "
              f"explicitly (pre-read the model to force cache up; run a hog to force anon up), "
              f"not a measured mechanism.")
    elif len(got) >= 4:
        print(f"  no candidate variable tracked t/s (|rho| < 0.7 for page cache, free memory and "
              f"anonymous pages). The draw is driven by something not measured here; instrument "
              f"the slot/fill path instead of the machine's memory counters.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arm": args.arm, "prompt": args.prompt, "runs": runs,
             "spread_pct": spread, "within_run_pct": within,
             "repeatable_band_ts": [rest[0], rest[-1]], "repeatable_band_spread_pct": rest_spread,
             "n_cleared_250": n_hit,
             "spearman_rho_cache": rho_cache, "spearman_rho_usable": rho_free,
             "spearman_rho_anon": rho_anon},
            ensure_ascii=False, indent=2))
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
