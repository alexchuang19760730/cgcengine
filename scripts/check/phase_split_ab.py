#!/usr/bin/env python3
"""Phase-split prefill A/B: is the armed slab faster, or is the clamp merely slower?

THE QUESTION (decided 2026-09-17, option D)
-------------------------------------------
The phase split only exists when the whole-layer slab is armed, and only the `prefill250` profile
arms it. If arming is worth having as a default, the reason has to be a measurement. This is it.

HOW THE COMPARISON IS MADE HONEST
--------------------------------
1. It is a PREFILL experiment, and it cannot be a decode one. With this pool's routable geometry,
   `decode_width = min(floor(142/8), T_prefill-1) = 17` and decode is T=1 (MTP verify 2-4), so every
   decode step takes the DECODE graph in BOTH arms: the slab cannot participate in decode at all.
   An armed-vs-unarmed number on decode is zero by construction, and reporting that zero as "the
   slab gives nothing" would be a false negative manufactured by the measurement shape. (Its
   empirical half: the accept counters are byte-identical between the two builds, 58.25%, 180/309.)

2. Three arms, all on `prefill250`, so pool / ctx / batch / MTP / template are identical and only
   the phase decision can differ:

       phase-slab    slab armed, clamp lifted      -> 1 x width-2048 PREFILL graph
       phase-pool8   pool path, clamp = cap 8      -> ~256 x width-8 pool graphs
       phase-pool17  pool path, clamp = bound 17   -> ~121 x width-17 pool graphs

   `phase-slab` vs `phase-pool8` is NOT a single-variable comparison: for a chunk wider than the
   decode width, the unarmed arm also takes the `n_batch` clamp (the launcher prints `[arm] slab
   OFF ... with the n_batch clamp on`). So the honest claim is "slab + no clamp" vs "pool + clamp".
   Arms `pool8` and `pool17` differ only in the clamp width, which is the graph-count axis, and at
   `--prompt 17` arm `pool17` is a SINGLE pool-path graph -- i.e. the pool baseline at matched graph
   count, which is what separates "the slab is faster" from "the clamp made it slower".

3. Interleaved, not arm-major. Launch order dominates on this box: three consecutive launches of one
   fixed command gave 246.86 / 134.83 / 110.70 t/s, with free-at-launch ANTI-monotone and carried
   swap monotone (docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md section D). Running one arm to
   completion prices the arm by its position in the sequence.

4. 4 launches per arm, discard launch 1 (cold page cache), pair by rep index, take the median. Free
   GiB and swap-in-use are recorded per launch, because carried swap is the live suspect.

5. A memory precondition. This box is 16 GB, the model is 13 GB and the pool is 8 GiB, so the
   experiment is only meaningful with headroom. Measured 2026-09-17 17:34-17:38: with another
   server already resident, free fell 7.2 -> 1.0 GiB and llama-bench was SIGKILLed mid-prefill
   (`rc=-9` at `ntok=2048 il=12`) -- those t/s numbers would have described jetsam, not the engine.
   So the driver REFUSES to start unless free memory clears `--min-free-gib`, and it reports each
   launch's actual return code instead of quietly aggregating whatever came back.

USAGE
    python3 scripts/check/phase_split_ab.py                 # full grid, refuses if the box is busy
    python3 scripts/check/phase_split_ab.py --aggregate-only # re-read an earlier run's json
    python3 scripts/check/phase_split_ab.py --out DIR        # where the per-launch json goes

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS SCRIPT: a single launch of this model on this box
has been measured from 110 to 247 t/s for one fixed command, so per-arm absolute values here are
still only trustworthy as PAIRED comparisons within one interleaved session. The ratio between
interleaved arms is the result; the absolute t/s is context.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["phase-slab", "phase-pool8", "phase-pool17"]
SHAPES = "2048,17"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_state() -> dict:
    """free GiB / inactive GiB / swap in use MiB, read the way prefill_certifiability reports them."""
    vs = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page, free, inactive = 16384, 0, 0
    for line in vs.splitlines():
        m = re.match(r"Pages free:\s+(\d+)", line)
        if m:
            free = int(m.group(1))
        m = re.match(r"Pages inactive:\s+(\d+)", line)
        if m:
            inactive = int(m.group(1))
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", sw)
    return {"free_gib": round(free * page / 1e9, 2),
            "inactive_gib": round(inactive * page / 1e9, 2),
            "swap_mib": float(m.group(1)) if m else None}


def other_llama_procs() -> list[str]:
    """Anything from this repo's build/bin that holds the model in memory.

    Both binaries, not just the server. Measured 2026-09-17 17:45: the process that starved this
    harness was another line's `llama-bench` (7.5 GB resident, parent = another agent's python), and
    a guard that only looked for `llama-server` said the box was free. The condition this preflight
    is trying to detect is "some other process has the model loaded", not "a server is listening".
    """
    found = []
    for pat in ("build/bin/llama-server", "build/bin/llama-bench"):
        p = subprocess.run(["pgrep", "-fl", pat], capture_output=True, text=True)
        found += [l for l in (p.stdout or "").splitlines() if l.strip()]
    return found


def per_arm(arm: str, reps: int, out: str) -> None:
    path = os.path.join(out, f"{arm}_rep{reps}.json")
    if os.path.exists(path):
        log(f"skip (present) {arm} rep{reps}")
        return
    before = mem_state()
    cmd = [sys.executable, "scripts/check/llama_bench_matrix.py", "--arms", arm,
           "--prompt", SHAPES, "--gen", "0", "--depths", "0", "--reps", "3", "--json", path]
    log(f"{arm} rep{reps} start (free={before['free_gib']}GiB swap={before['swap_mib']}MiB)")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    with open(os.path.join(out, f"{arm}_rep{reps}.stdout.txt"), "w") as f:
        f.write(p.stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(p.stderr or "")
    killed = p.returncode < 0
    log(f"{arm} rep{reps} rc={p.returncode}{' (KILLED -- thrash, not the engine)' if killed else ''} "
        f"{time.time() - t0:.0f}s")
    with open(os.path.join(out, f"{arm}_rep{reps}.mem.json"), "w") as f:
        json.dump({"rc": p.returncode, "killed": killed, "before": before, "after": mem_state()}, f, indent=1)


def load_pp(arm: str, rep: int, out: str) -> dict[int, float]:
    path = os.path.join(out, f"{arm}_rep{rep}.json")
    if not os.path.exists(path):
        return {}
    try:
        j = json.load(open(path))
    except Exception:  # noqa: BLE001 - a truncated stream is a killed launch, reported elsewhere
        return {}
    rows = j if isinstance(j, list) else j.get("rows", [])
    if rows and isinstance(rows[0], dict) and "rows" in rows[0]:
        rows = rows[0]["rows"]
    got = {}
    for r in rows:
        if isinstance(r, dict) and r.get("n_prompt") and not r.get("n_gen"):
            got[int(r["n_prompt"])] = round(r.get("avg_ts", 0.0), 2)
    return got


def aggregate(reps: int, out: str) -> dict:
    table = {arm: {w: [] for w in (2048, 17)} for arm in ARMS}
    rcs = {arm: [] for arm in ARMS}
    for rep in range(1, reps + 1):
        for arm in ARMS:
            mpath = os.path.join(out, f"{arm}_rep{rep}.mem.json")
            # .get(): an earlier revision of this harness wrote mem.json without an rc field, and an
            # aggregator that crashes on its own older output is a reason to skip re-reading data.
            rcs[arm].append(json.load(open(mpath)).get("rc") if os.path.exists(mpath) else None)
            d = load_pp(arm, rep, out)
            for w in (2048, 17):
                if w in d:
                    table[arm][w].append(d[w])
    print()
    print("=== per-launch pp t/s (launch 1 is the cold one) ===")
    for arm in ARMS:
        print(f"  {arm:<14} rc={rcs[arm]}")
        for w in (2048, 17):
            print(f"      p={w:<5} {table[arm][w]}")
    summary = {}
    print()
    print(f"=== paired medians over launches 2..{reps} (launch 1 discarded) ===")
    for arm in ARMS:
        summary[arm] = {}
        for w in (2048, 17):
            vals = table[arm][w][1:]
            if vals:
                summary[arm][w] = round(statistics.median(vals), 2)
        p2048, p17 = summary[arm].get(2048), summary[arm].get(17)
        ms = f"{1000.0 / p2048:.2f} ms/token" if p2048 else "n/a"
        print(f"  {arm:<14} p2048={p2048:<8} p17={p17:<8} {ms}")
    if all(summary[a].get(2048) for a in ARMS):
        print()
        print("=== read this as ===")
        print("  pool8 vs pool17 : if per-token cost is ~equal, the gap is GRAPH COUNT (clamp penalty)")
        print("  slab vs pool17  : slab's single graph against the pool's single graph (p=17) isolates")
        print("                    the slab itself at MATCHED graph count")
    print()
    print("=== launch memory state (the covariate that moved the number in section D) ===")
    for arm in ARMS:
        for rep in range(1, reps + 1):
            p = os.path.join(out, f"{arm}_rep{rep}.mem.json")
            if os.path.exists(p):
                m = json.load(open(p))["before"]
                print(f"  {arm:<14} rep{rep} free={m['free_gib']}GiB swap={m['swap_mib']}MiB")
    result = {"per_launch": table, "rc": rcs, "paired_median_discard_rep1": summary}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(result, f, indent=1)
    log(f"summary -> {os.path.join(out, 'summary.json')}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default="/tmp/phase_split_ab")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--min-free-gib", type=float, default=6.0,
                    help="refuse to launch below this; see the 17:34-17:38 thrash note in the docstring")
    ap.add_argument("--allow-busy-box", action="store_true",
                    help="override the precondition (the numbers then describe jetsam, not the engine)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.aggregate_only:
        st = mem_state()
        busy = other_llama_procs()
        log(f"preflight: free={st['free_gib']}GiB inactive={st['inactive_gib']}GiB swap={st['swap_mib']}MiB, "
            f"other llama processes={len(busy)}")
        if not args.allow_busy_box:
            if busy:
                log(f"REFUSING: {len(busy)} other llama process(es) hold the model (e.g. "
                    f"{busy[0].split(' ', 1)[0]}). On a 16 GB box with a 13 GB model and an 8 GiB pool "
                    "there is no room for two, and the previous attempt at this grid was SIGKILLed "
                    "mid-prefill (rc=-9 at ntok=2048 il=12). Free the box, or pass --allow-busy-box to "
                    "measure thrash -- and label the output as such.")
                return 2
            if st["free_gib"] < args.min_free_gib:
                log(f"REFUSING: free={st['free_gib']}GiB < --min-free-gib={args.min_free_gib}. "
                    "Lower it deliberately if you want the numbers anyway.")
                return 2
        for rep in range(1, args.reps + 1):
            for arm in ARMS:                      # interleaved: rotate arms inside every rep
                per_arm(arm, rep, args.out)

    res = aggregate(args.reps, args.out)
    bad = [f"{a}rep{i+1}rc={rc}" for a in ARMS for i, rc in enumerate(res["rc"][a]) if rc not in (0, None)]
    if bad:
        log(f"ATTENTION: non-zero launch exits: {bad} -- those points are thrash, not engine time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
