#!/usr/bin/env python3
"""Can one existing server log resolve a given change in `union_sum`? Answer from the log.

WHY THIS EXISTS. G4's target is a -12.8% change in union_sum, and the question that gates the
whole gate is OLDER than the lever: is 12.8% above this instrument's floor on this box? That
question was answered wrong twice in one day, both times by picking the wrong statistic:

  * "NOT MEASURABLE" was written after measuring the spread ACROSS runs (four Ornith servers,
    ~45 shared steps each). But resolvability is set by the standard error of the median
    WITHIN one run -- SE = 1.253*sigma/sqrt(n) -- and pooling several short runs inflates it.
  * "+/-1.9 t/s" was quoted for t/s, which is an order of magnitude coarser than union for the
    same effect. The instrument has to be named along with its floor.

So the floor is a property of (this box, this instrument, this run length) and it is READ OFF
A RUN, not assumed. This tool reads it, states whether the run can resolve the target, and
refuses to publish a number when the inputs cannot support one.

WHAT IT DOES NOT DO. It does not decide whether a *candidate* is worth testing, and it does not
compare two arms -- that is paired_union_ab.py / paired_union_runner.py. This one answers
"could this run have seen 12.8%?" so that a later null result is read as NOT MEASURABLE rather
than as "no effect".

USAGE

    python3 scripts/check/union_floor.py --newest 5                  # which recent log is usable
    python3 scripts/check/union_floor.py Backup/cgc_logs/<x>.log     # one log, full detail
    python3 scripts/check/union_floor.py <log> --resolve 12.8        # verdict against G4's target
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/check is this file's dir
from paired_union_ab import decprof_steps, series, se_median   # noqa: E402  (one parser, not two)

DEFAULT_TARGET_PCT = 12.8      # G4: union <= 38*mean_len at the measured mean_len 2.40
SPIKE_RATIO = 3.0              # union > 3x its own median == a pool-fill/cold step, not a step


def pick_ntok(S, want=None):
    """The modal ntok, or the requested one if the run actually has it."""
    if not S:
        return None, []
    if want is not None:
        steps = sorted(s for s, v in S.items() if v["ntok"] == want)
        return want, steps
    ntok = st.mode([v["ntok"] for v in S.values()])
    return ntok, sorted(s for s, v in S.items() if v["ntok"] == ntok)


def floor_of(values):
    """median, sigma, SE(median), and 3SE as a fraction of the median."""
    med = st.median(values)
    sd = st.pstdev(values)
    se = se_median(values)
    return dict(n=len(values), median=med, sigma=sd, se=se, three_se_pct=100.0 * 3 * se / med)


def report(path, want_ntok=None, target_pct=DEFAULT_TARGET_PCT, trim=SPIKE_RATIO):
    """Everything the log can support, plus an explicit verdict. Returns a dict or None."""
    S = series(path)
    d_steps = decprof_steps(path)
    head = dict(path=str(path), decprof_steps=d_steps, union_steps=len(S))
    if not S:
        why = ("no CGC-DECPROF at all -- the run did not set CGC_DECODE_PROFILE=1" if d_steps == 0
               else f"{d_steps} DECPROF steps but none with the GPU tail -- the run did not set "
                    f"CGC_GPU_TIMING=1, so there is no union to read")
        head["refuse"] = why
        return head

    ntok, steps = pick_ntok(S, want_ntok)
    if not steps:
        head["refuse"] = (f"no union step at ntok={want_ntok}; the run's ntok values are "
                          f"{sorted({v['ntok'] for v in S.values()})}")
        return head

    raw = [S[s]["union"] for s in steps]
    med_all = st.median(raw)
    kept = [S[s]["union"] for s in steps if S[s]["union"] <= trim * med_all]
    dropped = len(raw) - len(kept)
    head.update(ntok=ntok, all_steps=floor_of(raw), spikes_dropped=dropped,
                spike_cut_ms=trim * med_all)
    if kept:
        head["trimmed"] = floor_of(kept)

    # The second half alone: the first half of a server log carries the cold-pool fill, whose
    # steps are the spikes above -- reported separately so the reader can see what the cold
    # start costs the floor instead of having it averaged away.
    half = steps[len(steps) // 2:]
    if len(half) >= 8:
        head["second_half"] = floor_of([S[s]["union"] for s in half])
        head["second_half"]["steps"] = f"{half[0]}..{half[-1]}"

    # `second_half` IS in the candidate set on purpose: the split is by step ORDER, decided by
    # the run's own sequence, not by which half looked better -- so it is a statement about the
    # cold start (which is not part of a steady-state floor), not a data-dependent selection.
    # It is printed with its step range so the reader can reject it if the pool was still filling.
    best = min((head[k] for k in ("trimmed", "all_steps", "second_half") if k in head),
               key=lambda d: d["three_se_pct"])
    head["best_3se_pct"] = best["three_se_pct"]
    head["n"] = best["n"]
    head["median"] = best["median"]
    head["resolves_target"] = best["three_se_pct"] < target_pct
    head["target_pct"] = target_pct
    return head


def _fmt(h):
    p = os.path.basename(h["path"])
    if "refuse" in h:
        return f"{p:<44} REFUSE  {h['refuse']}"
    f = h["trimmed"] if "trimmed" in h else h["all_steps"]
    tag = f"ntok={h['ntok']}"
    line = (f"{p:<44} {tag:<8} n={f['n']:>4}  median {f['median']:>7.2f} ms  "
            f"sigma {f['sigma']:>6.2f}  3SE {f['three_se_pct']:>5.1f}%")
    if h["spikes_dropped"]:
        line += f"  (dropped {h['spikes_dropped']} >{h['spike_cut_ms']:.0f} ms)"
    return line


def cmd_logs(paths, args):
    worst, any_ok = None, False
    for p in paths:
        h = report(p, args.ntok, args.resolve, args.trim_spike)
        print(_fmt(h))
        if "refuse" in h:
            continue
        any_ok = True
        if args.verbose:
            for k in ("all_steps", "trimmed", "second_half"):
                if k in h and isinstance(h[k], dict):
                    d = h[k]
                    print(f"      {k:<12} n={d['n']:>4} median {d['median']:>7.2f} "
                          f"sigma {d['sigma']:>6.2f} SE {d['se']:>5.2f} 3SE {d['three_se_pct']:>5.1f}%"
                          + (f"  steps {d['steps']}" if "steps" in d else ""))
        if worst is None or h["best_3se_pct"] < worst["best_3se_pct"]:
            worst = h
    if not any_ok:
        print("\nno log in this set carries union -- see the REFUSE reason on each line", file=sys.stderr)
        return 2
    print(f"\nbest of this set: {os.path.basename(worst['path'])}  n={worst['n']}  "
          f"3SE {worst['best_3se_pct']:.1f}%  vs target {worst['target_pct']:.1f}%")
    if worst["resolves_target"]:
        print(f"VERDICT  this run can resolve {worst['target_pct']:.1f}% "
              f"(3SE {worst['best_3se_pct']:.1f}% < {worst['target_pct']:.1f}%). A null result "
              f"from a run of this length is a real bound, not an absence.")
    else:
        need = worst["n"] * (worst["best_3se_pct"] / worst["target_pct"]) ** 2
        print(f"VERDICT  NOT MEASURABLE at {worst['target_pct']:.1f}%: 3SE {worst['best_3se_pct']:.1f}% "
              f"is wider than the target. A null result here means NOT MEASURABLE -- do not report "
              f"it as 'no effect'.  Reaching the target at this sigma needs n ~ {need:.0f} steps "
              f"(have {worst['n']}).")
    return 0


def selftest() -> int:
    fails = []

    def expect(what, got, want):
        if got != want:
            fails.append(f"{what}: got {got!r} want {want!r}")
            print(f"  FAIL {what}: got {got!r} want {want!r}")
        else:
            print(f"  ok   {what}")

    print("SE(median) is 1.253*sigma/sqrt(n), not sigma")
    d = [10.0] * 99 + [20.0]                       # one outlier in 100 points
    expect("sigma of that set", round(st.pstdev(d), 3), 0.995)
    expect("SE(median) uses sqrt(n)", round(se_median(d), 4), round(1.253 * st.pstdev(d) / 10, 4))

    print("\nfloor_of: 3SE is reported as a fraction of the median")
    f = floor_of([100.0] * 8)
    expect("zero spread -> zero 3SE", f["three_se_pct"], 0.0)
    f2 = floor_of([100.0 + i for i in range(20)])
    expect("monotone set has n recorded", f2["n"], 20)

    print("\npick_ntok: modal by default, explicit when asked")
    S = {i: dict(ntok=(4 if i < 7 else 1), total=1.0, union=1.0, gap=0.0) for i in range(10)}
    expect("modal ntok", pick_ntok(S)[0], 4)
    expect("explicit ntok", pick_ntok(S, 1)[1], [7, 8, 9])
    expect("absent ntok yields no steps", pick_ntok(S, 2)[1], [])

    print("\nreport refuses on a log with DECPROF but no GPU tail, and names the missing knob")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "bare.log"
        bare.write_text("".join(
            f"CGC-DECPROF: step={i} segs=41 layers=40 total=100.0 ms | wait=1.0 (1%) "
            f"cb=2.0 (2%) submit=3.0 (3%) ntok=4\n" for i in range(5)))
        h = report(bare)
        expect("refused", "refuse" in h, True)
        expect("blames CGC_GPU_TIMING", "GPU_TIMING" in h["refuse"], True)
        empty = Path(td) / "empty.log"
        empty.write_text("nothing here\n")
        expect("blames CGC_DECODE_PROFILE", "DECODE_PROFILE" in report(empty)["refuse"], True)

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", help="server logs; omit with --newest")
    ap.add_argument("--newest", type=int, metavar="N",
                    help="use the N most recently modified Backup/cgc_logs/llama_server_*.log")
    ap.add_argument("--ntok", type=int, help="use this graph width instead of the modal one")
    ap.add_argument("--resolve", type=float, default=DEFAULT_TARGET_PCT, metavar="PCT",
                    help=f"the effect to judge against (default {DEFAULT_TARGET_PCT})")
    ap.add_argument("--trim-spike", type=float, default=SPIKE_RATIO, metavar="K",
                    help=f"treat union > K x median as a pool-fill step (default {SPIKE_RATIO})")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return selftest()

    paths = list(a.logs)
    if a.newest:
        globbed = [f for f in glob.glob("Backup/cgc_logs/llama_server_*.log") if not os.path.islink(f)]
        globbed.sort(key=os.path.getmtime, reverse=True)
        paths += globbed[:a.newest]
    if not paths:
        ap.error("give at least one log, or --newest N")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"no such log: {missing}", file=sys.stderr)
        return 2
    return cmd_logs(paths, a)


if __name__ == "__main__":
    raise SystemExit(main())
