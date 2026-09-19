#!/usr/bin/env python3
"""Paired A/B for throughput, with memory pressure made VISIBLE instead of silent.

WHY THIS EXISTS. On 2026-09-18 six interleaved arms of the SAME engine spanned 7.03 - 10.80 t/s
(1.54x) while reps inside one arm agreed to 0.6 - 2%. The noise is therefore not in the
measurement; it is decided before each arm starts. `sysctl vm.swapusage` showed
`used = 14134 / 15360 MiB` with `Swapins = 238e6`, on a 16 GB machine holding a 13.6 GB model plus
an expert pool -- a working set that does not fit. Every arm's speed was partly a function of how
much of it happened to be resident, which is a variable no previous tool recorded.

So this harness does three things the previous ones did not:

1. PROBES memory before and after every single arm (Pageins / Swapins / compressor / swap used)
   and reports the DELTA next to the throughput. An arm that paid 3 GB of swap-in is visibly not
   comparable to one that paid nothing, instead of both silently becoming "the number".
2. PAIRS the two candidates instead of blocking them. Consecutive A,B pairs are analysed as
   within-pair RATIOS, so slow drift in machine state divides out. Order alternates AB, BA, AB,
   BA... so that a monotone drift does not masquerade as an effect.
3. MEASURES ITS OWN NOISE FLOOR with `--null`: both slots configured identically, so any ratio
   away from 1.00 is instrument noise, not a real effect. Run this FIRST. If the null inherits a
   spread wider than the effect you hope to detect, no amount of real A/B can see that effect --
   and the honest answer is "not measurable", not "not measured".

THE THERMAL TRADE-OFF (deliberate, and it reverses the previous tool's rule).
`profile_duo.py` waits for NOMINAL before every arm, and that wait is 35 s to >10 min. For a
paired design that wait is a LIABILITY: it separates the two members of a pair by minutes, which
maximises exactly the drift the pairing is meant to cancel. Measured within-arm repeatability is
good, so here the members run back to back (`--thermal-gate` off by default) and thermal is
recorded per arm rather than enforced. Turn the gate on only if you specifically want both members
of every pair to be individually quotable -- and expect fewer, noisier pairs.

USAGE
    # always start here: how much can this instrument even see?
    python3 scripts/check/paired_ab.py --null --pairs 4

    # then the real question
    python3 scripts/check/paired_ab.py --pairs 5 \
        --b-bin-dir /tmp/bisect_09a/src/llama.cpp/build/bin

Exit code 0 always: the table is the deliverable and "not measurable" is a result.
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import thermal_pressure as tp  # noqa: E402
import prod_matrix as pm  # noqa: E402

PAGE = 16384  # vm_stat page size on this machine


class NS:
    """The duck-typed argparse Namespace `prod_matrix.run_cell` expects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------- memory probe

_NUM = re.compile(r"[-+]?\d+")


def mem_probe() -> dict:
    """Snapshot the VM counters that decide whether an arm's t/s means anything.

    Pageins/Swapins are cumulative-since-boot, so only DIFFERENCES across an arm are meaningful.
    """
    out = {"t": round(time.time(), 1)}
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=20).stdout
        for line in vs.splitlines():
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            m = _NUM.search(rest)
            if m:
                out[key.strip().strip('"')] = int(m.group())
    except Exception as e:  # a probe failure must not cancel the measurement
        out["_vmstat_error"] = str(e)
    try:
        sw = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True,
                            timeout=10).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", sw)
        out["swap_used_mb"] = float(m.group(1)) if m else None
    except Exception as e:
        out["_swap_error"] = str(e)
    return out


def mem_delta(before: dict, after: dict) -> dict:
    """Express the memory cost of one arm in MB actually moved, plus residency headroom."""
    d = {}
    for k in ("Pageins", "Pageouts", "Swapins", "Swapouts"):
        b, a = before.get(k), after.get(k)
        d[k + "_mb"] = round((a - b) * PAGE / 1048576, 1) if (a is not None and b is not None) else None
    # How much memory was available to the arm: reclaimable = free + inactive + speculative.
    for k in ("Pages free", "Pages inactive", "Pages speculative"):
        d[k] = before.get(k)
    reclaim = sum(v for v in (before.get("Pages free"), before.get("Pages inactive"),
                              before.get("Pages speculative")) if v is not None)
    d["reclaimable_mb"] = round(reclaim * PAGE / 1048576) if reclaim else None
    d["compressor_mb"] = round(before.get("Pages occupied by compressor", 0) * PAGE / 1048576)
    d["swap_used_mb"] = before.get("swap_used_mb")
    return d


# -------------------------------------------------------------------- one arm

def set_bin(bin_dir: str | None) -> None:
    """Repoint llama-bench WITHOUT rebuilding. See profile_duo.py --bin-dir for why the rebuild
    itself is the danger here (it overwrites artifacts other sessions are mapping)."""
    if not bin_dir:
        return
    cand = Path(bin_dir) / "llama-bench"
    if not cand.exists():
        raise SystemExit(f"error: no llama-bench in {bin_dir}")
    import llama_bench_matrix as lbm
    lbm.LLAMA_BENCH = cand


def run_arm(profile, cell, slot, cfg, args) -> dict:
    set_bin(cfg["bin_dir"])
    ns = NS(logdir=args.logdir, extra_env=cfg["env"], reps=args.reps,
            min_usable_pct=args.min_usable_pct, dry_run=args.dry_run)
    pre = mem_probe()
    r = pm.run_cell(profile, cell, ns)
    post = mem_probe()
    r["slot"] = slot
    r["mem"] = mem_delta(pre, post)
    pl = list((r.get("platform") or {}).values())
    v = pl[0] if pl else {}
    s = [x for x in (v.get("samples_ts") or []) if x is not None]
    # The first rep pays the cold-start penalty (measured: 8.25 vs 10.84 inside one arm), so the
    # metric is the median of the REST. It is deliberately not the mean of all reps.
    warm = s[1:] if len(s) > 1 else s
    r["metric"] = round(st.median(warm), 3) if warm else None
    r["metric_kind"] = f"median(reps[1:]) n={len(warm)}"
    r["samples"] = [round(x, 2) for x in s]
    th = r.get("thermal") or {}
    r["launch_label"] = (th.get("launch") or {}).get("label", "-")
    r["worst_label"] = (th.get("worst") or {}).get("label", "-")
    return r


# Another line of work: another WorkBuddy session, unshared repo, same GPU, same 13.6 GB of RAM.
# This happened twice on 2026-09-18 while this file was being written.
#
# Scripts are matched BY THEIR `.py` SUFFIX, never by bare name. A bare `prod_matrix` also matches
# the path "--json Backup/prod_matrix/null_floor.json" **in our own argv**, and the first version
# of this gate consequently refused to run for its own user. That failure looked like "someone
# else is hogging the machine" -- the exact opposite of the truth.
_RIVAL_SCRIPTS = ("llama_bench_matrix", "prod_matrix", "profile_duo", "http_duo",
                  "paired_ab", "m123_oracle", "decode_sweep")
_RIVAL_BINS = ("llama-bench", "llama-server")
_RIVAL_PROC = re.compile(
    "(?:" + "|".join(re.escape(s) + r"\.py" for s in _RIVAL_SCRIPTS) + ")"
    "|(?:" + "|".join(re.escape(b) for b in _RIVAL_BINS) + r"\b)")


def _ps_table() -> dict:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,command="], capture_output=True,
                         text=True, timeout=15).stdout
    tbl = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            tbl[int(parts[0])] = (int(parts[1]), parts[2])
        except ValueError:
            continue
    return tbl


def _ancestors(tbl: dict, pid: int) -> set:
    """pid plus every ancestor. We must exclude our ancestor shell too: it exists only to carry
    our command line, so it "matches" whatever we match."""
    seen, cur = set(), pid
    while cur in tbl and cur not in seen:
        seen.add(cur)
        cur = tbl[cur][0]
    return seen


def rivals() -> list:
    try:
        tbl = _ps_table()
    except Exception:
        return []
    mine = _ancestors(tbl, os.getpid())
    found = []
    for pid, (ppid, cmd) in tbl.items():
        if pid in mine:
            continue
        if _RIVAL_PROC.search(cmd):
            found.append({"pid": pid, "ppid": ppid, "cmd": cmd[:120]})
    return found


def preflight(args) -> bool:
    """Refuse to launch when someone else is already using the machine.

    Not a printed warning: the whole point is ABORTING. A check whose failure only prints is a
    placebo -- on 2026-09-17 exactly that pattern let a build overwrite the artifacts another
    session was mapping, turning their A/B into a cross-build comparison.
    """
    rv = rivals()
    if not rv:
        return True
    print("\n!! PREFLIGHT REFUSAL -- other measurement processes are running:", file=sys.stderr)
    for r in rv:
        print(f"!!   pid {r['pid']}: {r['cmd']}", file=sys.stderr)
    print("!! A paired design assumes its own pairs are the only load on the box. With another\n"
          "!! session streaming experts off the same SSD, no ratio from this run is meaningful.\n"
          "!! Wait for the window, then re-run. (--no-preflight to override, which produces a\n"
          "!! number that looks identical to a real one and is not.)", file=sys.stderr)
    return False


def headroom_mb() -> int:
    """Reclaimable memory (free + inactive + speculative), MB."""
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=20).stdout
        tot = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            for line in vs.splitlines():
                if line.startswith(key):
                    m = _NUM.search(line.partition(":")[2])
                    if m:
                        tot += int(m.group())
        return tot * PAGE // 1048576
    except Exception:
        return -1


def wait_headroom(target: int, timeout: float, poll: float) -> dict:
    """Block until the machine has `target` MB reclaimable.

    macOS gives us no way to RELEASE swap without root (`purge` needs a password, there is no
    swapoff), so "freeing swap" is not on offer here. What is on offer is entering every arm from
    a comparable memory state -- i.e. making pressure a CONTROLLED variable instead of a hidden
    one. Observed 2026-09-18: 7842 MB reclaimable when idle vs 1111 MB while another session was
    loading the same 13.6 GB model, and that is a 1.5x throughput difference waiting to happen.
    """
    t0 = time.time()
    while True:
        h = headroom_mb()
        if h >= target:
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "headroom_mb": h}
        if time.time() - t0 > timeout:
            return {"ok": False, "waited_s": round(time.time() - t0, 1), "headroom_mb": h}
        print(f"    memory headroom {h} MB < {target} MB -- waiting "
              f"({time.time() - t0:.0f}s / {timeout:.0f}s)", flush=True)
        time.sleep(poll)


def maybe_gate(args) -> None:
    if not args.thermal_gate:
        return
    t0 = time.time()
    while tp.level() != 0 and time.time() - t0 < args.cooldown_timeout:
        print(f"    thermal {tp.label(tp.level())} -- waiting ({time.time() - t0:.0f}s)",
              flush=True)
        time.sleep(args.poll)


# ------------------------------------------------------------------ analysis

def analyse(pairs, label_a, label_b) -> dict:
    """Within-pair ratios, plus how much of the scatter the memory deltas explain."""
    ratios, rows = [], []
    for i, (a, b) in enumerate(pairs, 1):
        if not (a.get("metric") and b.get("metric")):
            rows.append({"pair": i, "note": "arm missing a metric"})
            continue
        ra = a["metric"] / b["metric"]
        ratios.append(ra)
        rows.append({"pair": i, "a": a["metric"], "b": b["metric"], "ratio_a_over_b": round(ra, 4),
                     "a_swapin_mb": a["mem"].get("Swapins_mb"),
                     "b_swapin_mb": b["mem"].get("Swapins_mb"),
                     "a_pagein_mb": a["mem"].get("Pageins_mb"),
                     "b_pagein_mb": b["mem"].get("Pageins_mb")})
    out = {"rows": rows, "n_pairs": len(ratios)}
    if not ratios:
        return out
    med = st.median(ratios)
    out.update({
        "ratios": [round(x, 4) for x in ratios],
        "median": round(med, 4),
        "geometric_mean": round(st.geometric_mean(ratios), 4),
        "min": round(min(ratios), 4), "max": round(max(ratios), 4),
        "spread": round(max(ratios) / min(ratios), 4),
        "mad_pct": round(100 * st.median([abs(x - med) for x in ratios]) / med, 2),
        "direction": f"{label_a}/{label_b}",
    })
    # Does the confounder actually explain the scatter? Pearson r of |log ratio| against |Δswapin|.
    try:
        xs, ys = [], []
        for a, b in pairs:
            if not (a.get("metric") and b.get("metric")):
                continue
            xs.append(abs((a["mem"].get("Swapins_mb") or 0) - (b["mem"].get("Swapins_mb") or 0)))
            ys.append(abs(a["metric"] / b["metric"] - 1))
        if len(xs) >= 3 and max(xs) > 0 and max(ys) > 0:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sx = sum((x - mx) ** 2 for x in xs) ** 0.5
            sy = sum((y - my) ** 2 for y in ys) ** 0.5
            out["r_swapin_vs_ratio"] = round(cov / (sx * sy), 3) if sx and sy else None
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="paired A/B with memory pressure recorded per arm")
    ap.add_argument("--profile", default="prefill250")
    ap.add_argument("--cell", default="decode")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--logdir", default="Backup/prod_matrix")
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--null", action="store_true",
                    help="configure BOTH slots identically: the ratio spread then IS the "
                         "instrument's noise floor. Do this before any real A/B.")
    ap.add_argument("--a-env", default="")
    ap.add_argument("--b-env", default="")
    ap.add_argument("--a-bin-dir", default=None)
    ap.add_argument("--b-bin-dir", default=None)
    ap.add_argument("--a-label", default=None)
    ap.add_argument("--b-label", default=None)
    ap.add_argument("--thermal-gate", action="store_true",
                    help="wait for NOMINAL before each arm (default OFF -- see module docstring)")
    ap.add_argument("--cooldown-timeout", type=float, default=420.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-headroom-mb", type=int, default=4000,
                    help="every arm must start with this much reclaimable memory. This is how "
                         "memory pressure becomes controlled: we cannot release swap (no root), "
                         "but we can refuse to measure at a pressure we did not choose.")
    ap.add_argument("--mem-timeout", type=float, default=600.0,
                    help="give up waiting for headroom after this many seconds (a refusal)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the other-session check. Unsafe by construction: produces numbers "
                         "indistinguishable from clean ones.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label_a = args.a_label or ("HEAD" if not args.a_bin_dir else f"archive {args.a_bin_dir}")
    label_b = args.b_label or ("HEAD" if not args.b_bin_dir else f"archive {args.b_bin_dir}")
    cfg = {
        "A": {"env": args.a_env, "bin_dir": args.a_bin_dir},
        "B": {"env": args.b_env, "bin_dir": args.b_bin_dir},
    }
    if args.null:
        cfg["B"] = dict(cfg["A"])
        label_b = label_a + " (same binary -- null)"

    print("=" * 78)
    print(f"PAIRED A/B   profile={args.profile}  cell={args.cell}  pairs={args.pairs}  "
          f"reps={args.reps}")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"  thermal gate: {'ON' if args.thermal_gate else 'off (pairs run back to back)'}")
    if args.null:
        print("  NULL MODE -- any spread here is instrument noise, not an effect.")
    print("=" * 78, flush=True)

    known = pm.profiles_from_run_server()
    if args.profile not in known:
        print(f"error: unknown profile {args.profile}; known: {known}", file=sys.stderr)
        return 2

    pairs, flat = [], []
    for i in range(args.pairs):
        # Re-check every pair, not just once: another session can appear mid-run, and a pair whose
        # two halves straddle that arrival is worse than no pair at all.
        if not args.no_preflight and not preflight(args):
            print(f"\nABORTED after {i} pair(s). Partial data below is still recorded.", flush=True)
            break
        if args.min_headroom_mb > 0:
            hw = wait_headroom(args.min_headroom_mb, args.mem_timeout, args.poll)
            if not hw["ok"]:
                print(f"\nABORTED before pair {i + 1}: headroom still {hw['headroom_mb']} MB "
                      f"(want >= {args.min_headroom_mb}) after {hw['waited_s']}s.\n"
                      f"Every arm must start from the same memory state or the ratios are not "
                      f"a measurement.", flush=True)
                break
        # Alternate so a monotone drift cancels: pair order AB, BA, AB, BA ...
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]
        got = {}
        for slot in order:
            maybe_gate(args)
            r = run_arm(args.profile, args.cell, slot, cfg[slot], args)
            got[slot] = r
            flat.append(r)
            m = r["mem"]
            print(f"  pair{i + 1} [{slot}] metric={r['metric']}  samples={r['samples']}  "
                  f"worst={r['worst_label']:<9} swapin={m.get('Swapins_mb')}MB "
                  f"pagein={m.get('Pageins_mb')}MB headroom={m.get('reclaimable_mb')}MB",
                  flush=True)
        pairs.append((got["A"], got["B"]))

    res = analyse(pairs, label_a, label_b)
    print("\n" + "=" * 78)
    if res.get("median") is None:
        print("no usable pairs")
        return 0
    print(f"{'pair':<6}{'A':>9}{'B':>9}{'A/B':>9}{'d.swapin MB':>14}{'d.pagein MB':>14}")
    print("-" * 78)
    for row in res["rows"]:
        if "ratio_a_over_b" not in row:
            print(f"{row['pair']:<6}{'-- ' + row.get('note', '')}")
            continue
        dsw = (row.get("a_swapin_mb") or 0) - (row.get("b_swapin_mb") or 0)
        dpg = (row.get("a_pagein_mb") or 0) - (row.get("b_pagein_mb") or 0)
        print(f"{row['pair']:<6}{row['a']:>9.2f}{row['b']:>9.2f}{row['ratio_a_over_b']:>9.3f}"
              f"{dsw:>14.1f}{dpg:>14.1f}")
    print("-" * 78)
    print(f"median A/B = {res['median']:.4f}   geometric mean = {res['geometric_mean']:.4f}")
    print(f"spread = {res['min']:.3f}..{res['max']:.3f}  ({res['spread']:.3f}x, "
          f"MAD {res['mad_pct']:.2f}%)  over {res['n_pairs']} pairs")
    r = res.get("r_swapin_vs_ratio")
    if r is not None:
        print(f"corr(|Δswapin|, |ratio-1|) = {r:+.3f}"
              f"   <-- near 0 means memory does NOT explain the scatter (good pairing);"
              f"\n{'':>45}near +1 means the pairing failed and any conclusion is unsafe")
    print()
    if args.null:
        print(f"NOISE FLOOR: identical binaries land {res['min']:.3f}..{res['max']:.3f} apart "
              f"({res['spread']:.3f}x, MAD {res['mad_pct']:.2f}%).")
        print("Any real effect smaller than this CANNOT be detected by this instrument.")
    else:
        lo, hi = res["min"], res["max"]
        if lo <= 1.0 <= hi:
            print(f"NOT DECIDED: the interval {lo:.3f}..{hi:.3f} straddles 1.0 -- the two "
                  f"candidates are not separated by this experiment.")
        else:
            print(f"SEPARATED: every pair favours the same side ({lo:.3f}..{hi:.3f}, all on one "
                  f"side of 1.0).")
    print("=" * 78)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"profile": args.profile, "cell": args.cell, "null": args.null,
             "a_label": label_a, "b_label": label_b, "analysis": res, "arms": flat},
            ensure_ascii=False, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
