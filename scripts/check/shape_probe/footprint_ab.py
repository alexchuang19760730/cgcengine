#!/usr/bin/env python3
"""footprint_ab.py -- how much memory does ONE cgc_shape_probe invocation actually cost?

WHY THIS FILE EXISTS
    Two joint (shape x kernel) sweeps were refused by their own instrument on
    2026-09-23 (docs/JOINT_SHAPE_KERNEL_AXIS_2026-09-22.md section 8): the null
    cell spanned 21-242% instead of <=5%, and one arm came out at a NEGATIVE
    percentage of peak bandwidth. Before touching the shared admission constant
    `server_window.NEED_MB`, the cheaper question has to be answered with
    evidence: is the instrument itself what makes the box unable to answer?

    That question has three separable parts, and this file prices all three
    because cutting the wrong one buys nothing:

        ALLOCATION  the pool + bank + host fill buffer the probe memory-maps
        DURATION    how long the pages stay HOT (target-ms: compute keeps touching them)
        GEOMETRY    the bank size itself -- which is NOT ours to cut (it is the
                    model's real expert count; shrinking it changes the price)

    So arms A/B isolate ALLOCATION and arms A/C isolate DURATION. If A ~= B the
    pool size was virtual-only reservation and never resident -- the lever is
    somewhere else and saying so is the useful outcome. If A ~= C then the long
    compute is not what holds pages down and shortening it buys nothing.

WHAT IS MEASURED
    peak_rss_mb  maximum resident set size of the invocation (macOS /usr/bin/time -l),
                 which is what actually has to fit in RAM
    dip_mb       how far `vm_free_mb` (the SHARED window's own probe -- imported,
                 never retyped) falls below the level measured immediately before
                 the invocation started
    wall_s       how long the invocation took

HOW IT REFUSES
    The shared window is asked once before the first invocation. A busy box here
    would mean making someone else's numbers worse in the act of measuring our own
    footprint, so the default is to refuse; only action.

Usage:
    shape_probe/footprint_ab.py --selftest
    shape_probe/footprint_ab.py --arms A,B,C --reps 3 --json Backup/footprint_ab.json
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE / "cgc_shape_probe"
CHECK = ROOT / "scripts" / "check"

sys.path.insert(0, str(CHECK))
import server_window as sw                     # noqa: E402  the SHARED probe + admission bar

MIB = 1024.0 * 1024.0
TIME_BIN = "/usr/bin/time"

# Bytes per block for the two expert-bank types this line prices. Both are exact
# (82 B and 110 B per 256-element block), read back from the probe's own output;
# anything derived from them must stay consistent with that printout.
BLOCK_BYTES = {"iq2_s": 82, "iq3_s": 110, "iq3_xxs": 84, "iq4_xs": 106, "q3_k": 112}
BLOCK_ELEMS = 256

# The shapes the joint sweep actually sweeps (mmid_shapes.SWEEP_SHAPES).
SHAPES = (
    ("gate_exps", "iq2_s", 2048, 512),
    ("down_exps", "iq3_s", 512, 2048),
)
EXPERTS_MODEL = 256
USED_MODEL = 8


def vm_free_mb():
    """The SHARED window's own reclaimable reading -- one definition, no second opinion."""
    return float(sw._probe("vm_free_mb", lambda: 0.0)())


def swap_used_mib():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)([KMG])", out)
    if not m:
        return 0.0
    return float(m.group(1)) * {"K": 1 / 1024.0, "M": 1.0, "G": 1024.0}[m.group(2)]


def bank_mib(tname, k, out_n, experts):
    """The single largest thing any invocation allocates: the expert bank itself."""
    if tname not in BLOCK_BYTES:
        raise ValueError("no block size known for type %r -- add it, do not guess" % tname)
    blocks = (k * out_n * experts + BLOCK_ELEMS - 1) // BLOCK_ELEMS
    return blocks * BLOCK_BYTES[tname] / MIB


def lean_pool_mib(tname, k, out_n, experts, nc, used, tokens, slack_mib=64.0):
    """Smallest pool that can still hold the graph, with room to be wrong loudly.

    The probe exits 2 with "pool exhausted" rather than producing a number, so an
    under-sized pool is a visible failure, not a silent one -- the slack is there
    because being wrong here costs a whole invocation, and there is no prize for
    shaving the last 8 MiB.
    """
    return int(round(bank_mib(tname, k, out_n, experts) + slack_mib))


class Sampler:
    """Poll the shared reclaimable probe on its own thread while an invocation runs.

    One wall-clock number taken after a run cannot see the dip: by then the pages
    have already been recycled. The whole point is how far down it goes DURING.
    """

    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                self.samples.append((t, vm_free_mb()))
            except Exception:
                pass
            self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5)
        return False

    def min_mb(self):
        return min((v for _, v in self.samples), default=None)


def parse_maxrss(text):
    """`/usr/bin/time -l` prints `NNN  maximum resident set size` in BYTES.

    Returns None when absent -- the defect this whole line keeps re-learning is a
    missing reading printed as 0, and a parser that returns 0 on no-match would
    report a zero-footprint probe instead of an unread one.
    """
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    return int(m.group(1)) if m else None


def run_one(cmd, env):
    """One invocation under /usr/bin/time -l, with the memory sampler running."""
    t0 = time.time()
    before = vm_free_mb()
    with Sampler() as s:
        p = subprocess.run([TIME_BIN, "-l"] + cmd, capture_output=True, text=True,
                           timeout=900, env=env)
    wall = time.time() - t0
    combine = p.stdout + p.stderr
    return {
        "peak_rss_mb": (parse_maxrss(combine) or 0) / MIB,
        "peak_rss_readable": parse_maxrss(combine) is not None,
        "usable_before_mb": before,
        "usable_min_mb": s.min_mb(),
        "dip_mb": (before - s.min_mb()) if s.min_mb() is not None else None,
        "n_samples": len(s.samples),
        "swap_used_mib": round(swap_used_mib(), 1),
        "wall_s": wall,
        "rc": p.returncode,
        "err": "" if p.returncode == 0 else (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])[-1],
    }


def probe_cmd(tname, k, out_n, experts, used, tokens, batch, pool_mib, target_ms):
    return [str(PROBE), "--op", "mul_mat_id", "--type", tname,
            "--k", str(k), "--out", str(out_n), "--experts", str(experts),
            "--used", str(used), "--tokens", str(tokens),
            "--mode", "batch", "--batch", str(batch), "--iters", "1",
            "--target-ms", str(target_ms), "--pool-mib", str(pool_mib)]


def arm_config(name, tname, k, out_n, experts, used, tokens, batch, pool_override, target_override):
    """One named configuration. A = today's defaults, B = lean pool, C = short compute."""
    if name == "A":
        pool, target = 320, 1500.0
    elif name == "B":
        pool, target = lean_pool_mib(tname, k, out_n, experts, batch, used, tokens), 1500.0
    elif name == "C":
        pool, target = 320, 100.0
    else:
        raise ValueError("unknown arm %r (A=today, B=lean pool, C=short compute)" % name)
    if pool_override:
        pool = pool_override
    if target_override:
        target = target_override
    return pool, target


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def compare(rows):
    """The verdict, from the numbers only. 'no difference' is a first-class answer.

    Two of this line's past defects came from upgrading an inconclusive comparison
    into a conclusion, so here a sub-noise difference is reported as one rather
    than being rounded into a direction.
    """
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    out = {"per_arm": {}}
    for a, rs in sorted(by_arm.items()):
        ok = [r for r in rs if r["ok"]]
        out["per_arm"][a] = {
            "n": len(rs), "n_ok": len(ok),
            "median_peak_rss_mb": median([r["peak_rss_mb"] for r in ok]),
            "median_dip_mb": median([r["dip_mb"] for r in ok if r["dip_mb"] is not None]),
            "median_wall_s": median([r["wall_s"] for r in ok]),
        }
    # The decision rule must be stated before any number is read, because either
    # outcome is a finding: the lever exists, or it does not.
    spread = {}
    for a, s in out["per_arm"].items():
        vals = [r["dip_mb"] for r in by_arm[a] if r["ok"] and r["dip_mb"] is not None]
        spread[a] = (max(vals) - min(vals)) if len(vals) > 1 else None
    out["dip_spread_mb"] = spread
    base = out["per_arm"].get("A", {}).get("median_dip_mb")
    cuts = {}
    for a in sorted(by_arm):
        if a == "A":
            continue
        theirs = out["per_arm"][a]["median_dip_mb"]
        cut = {"dip_cut_frac": None, "distinguishable": None, "why": None}
        # No null cell, no reading -- the same rule nsg_decide applies to a knob
        # sweep, and it is the only honest one here too: with a single pass there
        # is no estimate of how far two IDENTICAL invocations differ, so calling a
        # 1.7 percent difference real would be inventing precision.
        if base is None or theirs is None:
            cut["why"] = "no usable dip reading on one side"
            cuts[a] = cut
            continue
        n_base = len([r for r in by_arm.get("A", []) if r["ok"] and r["dip_mb"] is not None])
        n_a = len([r for r in by_arm[a] if r["ok"] and r["dip_mb"] is not None])
        if n_base < 2 or n_a < 2:
            cut["dip_cut_frac"] = (base - theirs) / base if base else None
            cut["why"] = ("unjudged: %d A reading(s) vs %d %s -- one cannot estimate the "
                          "spread of two identical invocations from one of them"
                          % (n_base, n_a, a))
            cuts[a] = cut
            continue
        cut["dip_cut_frac"] = (base - theirs) / base
        floor = max(s for s in (spread.get("A"), spread.get(a)) if s is not None)
        cut["distinguishable"] = abs(base - theirs) > floor
        cut["why"] = ("the %.1f%% dip change %s than the %.1f MB either arm spans on its own"
                      % (100 * cut["dip_cut_frac"],
                         "is bigger" if cut["distinguishable"] else "is NOT bigger", floor))
        cuts[a] = cut
    out["vs_A"] = cuts
    return out


def sequence(rows_by_arm):
    """How each arm's usable-memory BASELINE moves across a BACK-TO-BACK sequence.

    WHY THIS IS SEPARATE FROM THE PER-INVOCATION DIP
        The dip measured per invocation (~360-490 MB here) cannot explain the null
        cell that wrecked two joint sweeps. On 2026-09-23 the sweep's own header read
        usable 1.25 GiB while a single invocation only dipped 0.4 GiB -- a 6 GB gap.
        The missing term is accumulation: each invocation leaves residue that takes
        longer to recycle than the next invocation takes to start. Neither RSS nor a
        one-shot dip can see it; only the STARTING level of invocation k+1 can.

    So the number this returns is `drift_mb` = usable before the LAST invocation minus
    usable before the FIRST: how much the box loses over a sequence, which is exactly
    the trend `nsg_decide` calls "the CHANNEL moved".
    """
    out = {}
    for arm, rs in rows_by_arm.items():
        ok = [r for r in rs if r["ok"] and r["usable_before_mb"] is not None]
        if len(ok) < 2:
            out[arm] = {"n_ok": len(ok), "drift_mb": None,
                        "why": "fewer than two readings: no trend to speak of"}
            continue
        first, last = ok[0]["usable_before_mb"], ok[-1]["usable_before_mb"]
        out[arm] = {"n_ok": len(ok), "first_mb": first, "last_mb": last,
                    "drift_mb": last - first, "min_mb": min(r["usable_before_mb"] for r in ok),
                    "swap_start_mib": ok[0]["swap_used_mib"], "swap_end_mib": ok[-1]["swap_used_mib"],
                    "swap_growth_mib": ok[-1]["swap_used_mib"] - ok[0]["swap_used_mib"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--used", type=int, default=USED_MODEL)
    ap.add_argument("--experts", type=int, default=EXPERTS_MODEL)
    ap.add_argument("--pool-mib", type=int, default=0, help="override every arm's pool size")
    ap.add_argument("--target-ms", type=float, default=0.0, help="override every arm's target")
    ap.add_argument("--need-mb", type=float, default=0.0,
                    help="0 = the shared server_window.NEED_MB")
    ap.add_argument("--allow-busy", action="store_true",
                    help="proceed over a busy box and say so (never narrower)")
    ap.add_argument("--sequence", type=int, default=0,
                    help="run this many invocations per arm, INTERLEAVED across arms, and "
                         "report how each arm's usable baseline moves (the accumulation the "
                         "per-invocation dip cannot see)")
    ap.add_argument("--settle-ms", type=int, default=0,
                    help="pause between invocations in sequence mode; 0 = what a sweep does today")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not PROBE.exists():
        sys.exit("FATAL: %s missing -- run shape_probe/build.sh first" % PROBE)

    need = args.need_mb or sw.NEED_MB
    before = sw.decision(need_mb=need)
    print("window  admits=%s  reclaimable=%.0f MB (need %.0f)  foreign=%s  port=%s"
          % (before["admits"], before["reclaimable_mb"], need,
             before["foreign_llama"], before["port_held"]))
    if not before["admits"] and not args.allow_busy:
        print("REFUSED: %s" % "; ".join(before["refused_by"]))
        print("  measuring our own footprint over someone else's server would make THEIR "
              "numbers worse, so this waits. --allow-busy overrides and records it.")
        return 2
    if not before["admits"]:
        print("OVERRIDDEN: %s" % "; ".join(before["refused_by"]))

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    # (label, tname, k, out_n) pairs but restricted to ONE shape in sequence mode: two
    # shapes would double every sequence for no extra answer about the mechanism.
    seq_shapes = SHAPES[:1] if args.sequence else SHAPES
    rows = []
    for (label, tname, k, out_n) in seq_shapes:
        for arm in arms:
            pool, target = arm_config(arm, tname, k, out_n, args.experts, args.used,
                                      args.tokens, args.batch, args.pool_mib,
                                      args.target_ms or None)
            print("%-10s arm %s  pool=%d MiB target=%.0f ms  (bank %.1f MiB)"
                  % (label, arm, pool, target, bank_mib(tname, k, out_n, args.experts)))
            for _ in range(max(1, args.reps)):
                r = run_one(probe_cmd(tname, k, out_n, args.experts, args.used, args.tokens,
                                      args.batch, pool, target), None)
                r.update({"shape": label, "arm": arm, "pool_mib": pool, "target_ms": target,
                          "ok": r["rc"] == 0 and r["peak_rss_readable"]})
                rows.append(r)
                print("   rss %7.1f MB  dip %7.1f MB  (usable %.0f -> %.0f)  wall %5.2f s  swap %.0f MiB"
                      % (r["peak_rss_mb"], r["dip_mb"] or -1, r["usable_before_mb"],
                         r["usable_min_mb"] or -1, r["wall_s"], r["swap_used_mib"]))
                sys.stdout.flush()

    if args.sequence:
        print("-" * 88)
        print("SEQUENCE: %d invocation(s) per arm, interleaved, settle %.0f ms"
              % (args.sequence, args.settle_ms))
        per_arm_rows = {}
        for (label, tname, k, out_n) in seq_shapes:
            cfg = {a: arm_config(a, tname, k, out_n, args.experts, args.used, args.tokens,
                                 args.batch, args.pool_mib, args.target_ms or None)
                   for a in arms}
            for a in arms:
                per_arm_rows[a] = []
            print("%-10s pools %s" % (label, {a: v[0] for a, v in cfg.items()}))
            for i in range(max(1, args.sequence)):
                for a in arms:                      # interleaved: shared drift is shared
                    pool, target = cfg[a]
                    r = run_one(probe_cmd(tname, k, out_n, args.experts, args.used, args.tokens,
                                          args.batch, pool, target), None)
                    r.update({"shape": label, "arm": a, "pool_mib": pool, "target_ms": target,
                              "seq_i": i, "ok": r["rc"] == 0 and r["peak_rss_readable"]})
                    per_arm_rows[a].append(r)
                    rows.append(r)
                    if args.settle_ms:
                        time.sleep(args.settle_ms / 1000.0)
                print("   i=%2d  " % i + "  ".join(
                    "%s base %6.0f MB / dip %5.0f" % (a, per_arm_rows[a][-1]["usable_before_mb"],
                                                      per_arm_rows[a][-1]["dip_mb"] or -1)
                    for a in arms) + "   swap %.0f MiB" % swap_used_mib())
                sys.stdout.flush()
            seq = sequence(per_arm_rows)
            print("-" * 88)
            for a in arms:
                d = seq[a]
                if d.get("drift_mb") is None:
                    print("  arm %s  %s" % (a, d["why"]))
                    continue
                print("  arm %s  pool %3d MiB  target %6.0f ms  baseline %.0f -> %.0f MB  "
                      "DRIFT %+.0f MB  swap %+.0f MiB"
                      % (a, cfg[a][0], cfg[a][1], d["first_mb"], d["last_mb"], d["drift_mb"],
                         d["swap_growth_mib"]))
            base = seq.get("A", {}).get("drift_mb")
            if base is not None and base < 0:
                for a in arms:
                    if a == "A":
                        continue
                    d = seq[a].get("drift_mb")
                    if d is None:
                        continue
                    print("  vs A: arm %s loses %.0f%% as much across the sequence"
                          % (a, 100.0 * d / base))

    after = sw.decision(need_mb=need)
    hijacked = before["admits"] and not after["admits"]
    if hijacked:
        print("HIJACKED: the box became busy during the run -- rows may describe the neighbour")

    cmp_ = compare(rows)
    print("-" * 88)
    for a, s in sorted(cmp_["per_arm"].items()):
        print("  arm %s  n=%d/%d ok   rss %7.1f MB   dip %7.1f MB   wall %5.2f s"
              % (a, s["n_ok"], s["n"], s["median_peak_rss_mb"] or -1,
                 s["median_dip_mb"] if s["median_dip_mb"] is not None else -1,
                 s["median_wall_s"] or -1))
    for a, c in sorted(cmp_["vs_A"].items()):
        verdict = ("cut" if c["dip_cut_frac"] > 0 else "deeper") if c["distinguishable"] \
            else "inside its own spread: NO READING"
        print("  vs A: arm %s changes the dip by %+.1f%%  -> %s"
              % (a, 100 * c["dip_cut_frac"], verdict))

    out = {"instrument": "shape_probe/footprint_ab.py", "window": before, "window_after": after,
           "hijacked": hijacked, "compare": cmp_, "rows": rows,
           "note": "A=today (pool 320/target 1500)  B=lean pool  C=short compute"}
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote %s" % p)
    return 0


def selftest():
    bad = tot = 0

    def check(name, cond):
        nonlocal bad, tot
        tot += 1
        print("  %s %s" % ("ok  " if cond else "FAIL", name))
        bad += 0 if cond else 1

    real = ("        1.55 real         0.00 user         0.00 sys\n"
            "             16252928  maximum resident set size\n"
            "                     0  page reclaims\n")
    check("the maxrss line parses to bytes", parse_maxrss(real) == 16252928)
    check("and to MB at the right scale", abs(parse_maxrss(real) / MIB - 15.5) < 0.05)
    check("a missing maxrss line is None, NOT 0 -- a zero-footprint probe is a lie",
          parse_maxrss("nothing here") is None)

    # geometry: every derived number must agree with the probe's own printout
    check("iq2_s gate bank is 82.0 MiB (what the probe prints)",
          abs(bank_mib("iq2_s", 2048, 512, 256) - 82.0) < 0.05)
    check("iq3_s down bank is 110.0 MiB (what the probe prints)",
          abs(bank_mib("iq3_s", 512, 2048, 256) - 110.0) < 0.05)
    check("the bank scales with the expert count, so cutting E cuts the MODELS geometry",
          abs(bank_mib("iq3_s", 512, 2048, 128) - 55.0) < 0.05)
    try:
        bank_mib("q9_z", 1, 1, 1)
        check("an unknown type raises instead of inventing a size", False)
    except ValueError:
        check("an unknown type raises instead of inventing a size", True)

    check("A is today's defaults: pool 320, target 1500",
          arm_config("A", "iq3_s", 512, 2048, 256, 8, 4, 32, 0, 0) == (320, 1500.0))
    b = arm_config("B", "iq3_s", 512, 2048, 256, 8, 4, 32, 0, 0)
    check("B is lean but still bigger than the bank it must hold", b[0] > 110 and b[0] < 320)
    check("C only shortens the compute, allocation untouched",
          arm_config("C", "iq3_s", 512, 2048, 256, 8, 4, 32, 0, 0) == (320, 100.0))
    check("an unknown arm is refused, not defaulted",
          _raises(lambda: arm_config("Z", "iq3_s", 512, 2048, 256, 8, 4, 32, 0, 0)))
    check("overrides reach the config (so a new arm needs no edit)",
          arm_config("A", "iq3_s", 512, 2048, 256, 8, 4, 32, 128, 250.0) == (128, 250.0))

    check("median of an empty list is None", median([]) is None)
    check("median agrees with the odd/even definitions",
          median([3.0, 1.0, 2.0]) == 2.0 and median([1.0, 2.0, 3.0, 4.0]) == 2.5)

    # The decision rule has to be exercisable in BOTH directions, or it is decoration.
    flat = [{"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 300.0, "wall_s": 2.0},
            {"arm": "A", "ok": True, "peak_rss_mb": 410.0, "dip_mb": 310.0, "wall_s": 2.1},
            {"arm": "B", "ok": True, "peak_rss_mb": 250.0, "dip_mb": 150.0, "wall_s": 2.0},
            {"arm": "B", "ok": True, "peak_rss_mb": 255.0, "dip_mb": 152.0, "wall_s": 2.0}]
    c = compare(flat)
    check("a real 50 percent dip cut is called distinguishable",
          c["vs_A"]["B"]["distinguishable"] is True and abs(c["vs_A"]["B"]["dip_cut_frac"] - 0.5) < 0.05)
    check("it is labelled a cut, not silence", c["vs_A"]["B"]["dip_cut_frac"] > 0)

    nothing = [{"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 300.0, "wall_s": 2.0},
               {"arm": "B", "ok": True, "peak_rss_mb": 250.0, "dip_mb": 295.0, "wall_s": 2.0}]
    c2 = compare(nothing)
    check("a comparison with no null cell on either side is unjudged, not a tiny win",
          c2["vs_A"]["B"]["distinguishable"] is None and "unjudged" in c2["vs_A"]["B"]["why"])
    check("...and the direction it saw is still reported alongside the refusal",
          abs(c2["vs_A"]["B"]["dip_cut_frac"] - 0.0167) < 0.005)

    single = [{"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 300.0, "wall_s": 2.0},
              {"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 310.0, "wall_s": 2.0},
              {"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 290.0, "wall_s": 2.0},
              {"arm": "B", "ok": True, "peak_rss_mb": 250.0, "dip_mb": 295.0, "wall_s": 2.0},
              {"arm": "B", "ok": True, "peak_rss_mb": 250.0, "dip_mb": 297.0, "wall_s": 2.0}]
    c4 = compare(single)
    check("with real spread on both sides, a cut inside it is called NO READING",
          c4["vs_A"]["B"]["distinguishable"] is False)

    bad_row = [{"arm": "A", "ok": False, "peak_rss_mb": 0.0, "dip_mb": None, "wall_s": 0.0},
               {"arm": "A", "ok": True, "peak_rss_mb": 400.0, "dip_mb": 300.0, "wall_s": 2.0}]
    c3 = compare(bad_row)
    check("failed invocations are counted out, not averaged in",
          c3["per_arm"]["A"]["n"] == 2 and c3["per_arm"]["A"]["n_ok"] == 1)
    check("...and the median still comes from the good ones",
          c3["per_arm"]["A"]["median_peak_rss_mb"] == 400.0)

    # The sequence verdict must refuse on thin evidence for the same reason the sweep
    # does: a trend from one point is not a trend.
    check("one reading gives no trend, and says so",
          sequence({"A": [{"ok": True, "usable_before_mb": 100.0, "swap_used_mib": 10.0}]})
          ["A"]["drift_mb"] is None)
    seq = sequence({"A": [{"ok": True, "usable_before_mb": 8000.0, "swap_used_mib": 100.0},
                          {"ok": True, "usable_before_mb": 3000.0, "swap_used_mib": 400.0}],
                    "B": [{"ok": True, "usable_before_mb": 8000.0, "swap_used_mib": 100.0},
                          {"ok": True, "usable_before_mb": 7000.0, "swap_used_mib": 150.0}]})
    check("a baseline that falls 5000 MB over a sequence reports exactly that",
          seq["A"]["drift_mb"] == -5000.0)
    check("and the arm that holds steady is reported as holding steady",
          seq["B"]["drift_mb"] == -1000.0)
    check("swap growth over the sequence is reported too (thrash has a second tell)",
          seq["A"]["swap_growth_mib"] == 300.0)
    check("failed invocations do not contribute a trend",
          sequence({"A": [{"ok": False, "usable_before_mb": None, "swap_used_mib": 0.0},
                          {"ok": True, "usable_before_mb": 1.0, "swap_used_mib": 0.0}]})
          ["A"]["drift_mb"] is None)

    print("selftest: %d/%d passed" % (tot - bad, tot))
    return 0 if bad == 0 else 1


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main() or 0)
