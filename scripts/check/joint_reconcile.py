#!/usr/bin/env python3
"""joint_reconcile.py -- close the decode-step books from ONE server log.

WHY THIS EXISTS
---------------
Two numbers from two different runs were being added together:

    line I : cb   = 74.18 ms   (run A: CGC_HOOK_SPLIT on, high swap)
    line A : union= 104.6 ms   (run B: different profile/regime)

and the sum exceeded the ~250 ms step it was supposed to be part of. A separate
cross-check, `gap subset-of cb+submit`, failed by 10.09 ms. Both failures have the
same shape: **quantities read from different runs, and from different clocks, were
treated as one decomposition.** This tool reads every quantity out of ONE log, per
step, and prints the residuals instead of asserting they add up.

WHAT EACH FIELD ACTUALLY IS (read off ggml-backend.cpp, not guessed)
--------------------------------------------------------------------
The per-layer accumulator block is around ggml-backend.cpp:2530-2545:

    dp_lay_w[il]   += st1 - st0          # host wait before this layer's top-k hook
    dp_lay_cb[il]  += st2 - st1          # the hook itself (callback_eval(ttopk))
    dp_lay_sub[il] += dp_last_submit_us  # the submit THAT QUEUED this layer's segment
    dp_lay_gpu[il] += sg_busy            # GPU busy, summed over the segment's buffers
    dp_lay_uni[il] += sg_union           # GPU span of the segment
    dp_lay_gap[il] += sg_gap             # GPU idle before this layer's segment

Three consequences that decide how the books may be closed:

1. **Two clocks.** wait/cb/submit come from `ggml_time_us()` (CPU). gpu/union/gap come
   from GPU buffer timestamps (ns). `gap subset-of cb+submit` is therefore a
   CROSS-CLOCK comparison: it can only be a plausibility check, never an identity.
2. **`submit` is attributed, not measured.** `dp_last_submit_us` is the submit that
   queued this layer's segment -- it is a queue-level quantity copied onto a layer. If
   a layer is charged it more than once (n>1 or sg>1) it is double-counted.
3. **gpu_sum > union_sum is expected, not a bug.** gpu sums per-buffer busy time while
   union is the segment's span; concurrent buffers overlap, so gpu >= union.

**`cpu_resid` IS A TAUTOLOGY -- DO NOT READ IT AS EVIDENCE**
-----------------------------------------------------------
ggml-backend.cpp:2691 is literally `const int64_t dp_tot = dp_w + dp_cb + dp_sb;`.
The printed `total` IS the sum of the three printed components. So `wait+cb+submit ==
total` reproduces to 0.00 ms by construction and carries **zero information** -- it
cannot fail and it does not show the books close. It is still printed, because a
reader who does not know line 2691 will otherwise take it as the strongest result in
the table. The closure that actually has content is `union + gap` vs `total`: those
two come from GPU buffer timestamps, a clock the CPU decomposition never touches.

THE CLOCK-SAFE CLOSURE (needs st= / en=)
----------------------------------------
`all:` rows also carry `st=` and `en=` -- GPU-timestamp start/end of each layer's
segment, relative to the step's earliest start. Those live on ONE clock, so the
GPU-side books close without touching the CPU clock at all:

    span  = max(en) - min(st)
    union_sum + gap_sum  ==  span      (if segments are serial and gap is the idle
                                        between them, this is an identity)

Older logs do not have st=/en= (the fields were added to the binary later), so this
tool reports the clock-safe closure as UNAVAILABLE rather than silently skipping it.
That is a property of the log, and the tool says so out loud.

USAGE
    python3 scripts/check/joint_reconcile.py --log <server.log> [--json out.json]
    python3 scripts/check/joint_reconcile.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

STEP_RX = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \([\d.]+%\) cb=([\d.]+) \([\d.]+%\) submit=([\d.]+) \([\d.]+%\) ntok=(\d+)"
    r"(?: \| layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms)?")
ALL_RX = re.compile(
    r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
    r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+) sg=(\d+) n=(\d+)"
    r"(?:\s+st=([\d.]+) en=([\d.]+))?")


def parse_log(path: Path) -> list[dict]:
    """Steps in log order. `all:` rows after a step row belong to that step -- the
    binary prints step, then top1..8, then all (ggml-backend.cpp:2701-2762)."""
    steps: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(errors="replace").splitlines():
        ms = STEP_RX.search(line)
        if ms:
            cur = {
                "step": int(ms.group(1)), "segs": int(ms.group(2)),
                "layers": int(ms.group(3)), "total": float(ms.group(4)),
                "wait": float(ms.group(5)), "cb": float(ms.group(6)),
                "submit": float(ms.group(7)), "ntok": int(ms.group(8)),
                "gpu_sum": float(ms.group(9)) if ms.group(9) else None,
                "union_sum": float(ms.group(10)) if ms.group(10) else None,
                "gap_sum": float(ms.group(11)) if ms.group(11) else None,
                "layers_rows": [],
            }
            steps.append(cur)
            continue
        ma = ALL_RX.search(line)
        if ma and cur is not None:
            cur["layers_rows"].append({
                "L": int(ma.group(1)), "wait": float(ma.group(2)),
                "cb": float(ma.group(3)), "submit": float(ma.group(4)),
                "gpu": float(ma.group(5)), "union": float(ma.group(6)),
                "gap": float(ma.group(7)), "sg": int(ma.group(8)),
                "n": int(ma.group(9)),
                "st": float(ma.group(10)) if ma.group(10) else None,
                "en": float(ma.group(11)) if ma.group(11) else None,
            })
    return steps


def reconcile_step(s: dict) -> dict:
    rows = s["layers_rows"]
    sw = sum(r["wait"] for r in rows)
    scb = sum(r["cb"] for r in rows)
    ssub = sum(r["submit"] for r in rows)
    sgpu = sum(r["gpu"] for r in rows)
    suni = sum(r["union"] for r in rows)
    sgap = sum(r["gap"] for r in rows)
    nn = sum(r["n"] for r in rows)
    ssg = sum(r["sg"] for r in rows)

    out = {
        "step": s["step"], "ntok": s["ntok"], "total": s["total"],
        "wait": s["wait"], "cb": s["cb"], "submit": s["submit"],
        "n_layer_rows": len(rows), "sum_n": nn, "sum_sg": ssg,
        "sum_gap": sgap,
        # CPU side: does the step-level decomposition close?
        "cpu_sum": s["wait"] + s["cb"] + s["submit"],
        "cpu_resid": s["total"] - (s["wait"] + s["cb"] + s["submit"]),
        # do the per-layer rolls reproduce the step-level numbers?
        "layer_wait_vs_step": sw - s["wait"],
        "layer_cb_vs_step": scb - s["cb"],
        "layer_submit_vs_step": ssub - s["submit"],
        "layer_gap_vs_step": (sgap - s["gap_sum"]) if s["gap_sum"] is not None else None,
        # the cross-clock check that failed by 10.09 ms
        "cb_plus_submit": s["cb"] + s["submit"],
        "gap_sum": s["gap_sum"],
        "gap_minus_cb_submit": (sgap - (s["cb"] + s["submit"]))
                               if s["gap_sum"] is not None else None,
        "gpu_sum": sgpu, "union_sum": suni,
        "gpu_ge_union": sgpu >= suni - 0.01,
        # How much of the step is OUTSIDE the GPU timeline entirely. If this is
        # large, "union + cb" is not the step -- the step has a third region.
        "total_minus_union_gap": (s["total"] - (suni + sgap))
                                 if s["gap_sum"] is not None else None,
        # The exact sum that was claimed to exceed the step: cb (host) + union (GPU).
        # Read from one step it is bounded by total only because the two overlap.
        "union_plus_cb": suni + s["cb"],
        "union_plus_cb_minus_total": (suni + s["cb"]) - s["total"],
    }

    # Clock-safe GPU closure. Only possible when st=/en= are present.
    sts = [r["st"] for r in rows if r["st"] is not None]
    ens = [r["en"] for r in rows if r["en"] is not None]
    if sts and ens and any(v > 0 for v in sts + ens):
        span = max(ens) - min(sts)
        out["span"] = span
        # The single most informative number in the whole table: how much of the
        # CPU-measured step is NOT covered by the GPU timeline at all.
        out["total_minus_span"] = s["total"] - span
        out["union_plus_gap"] = suni + sgap
        out["gpu_span_resid"] = span - (suni + sgap)
        out["clock_safe"] = True
    else:
        out["clock_safe"] = False
        out["gpu_span_resid"] = None
    return out


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def summarise(recs: list[dict]) -> dict:
    clock_safe = sum(1 for r in recs if r["clock_safe"])
    return {
        "n_steps": len(recs),
        "n_clock_safe": clock_safe,
        "clock_safe_available": clock_safe > 0,
        "median": {
            "total_ms": med([r["total"] for r in recs]),
            "cpu_resid_ms": med([r["cpu_resid"] for r in recs]),
            "cpu_resid_pct": (lambda v: None if v is None else
                              100.0 * v / med([r["total"] for r in recs]))(
                                  med([r["cpu_resid"] for r in recs])),
            "layer_submit_minus_step_ms": med([r["layer_submit_vs_step"] for r in recs]),
            "gap_minus_cb_submit_ms": med([r["gap_minus_cb_submit"] for r in recs]),
            "gpu_span_resid_ms": med([r["gpu_span_resid"] for r in recs]) if clock_safe else None,
            "total_minus_span_ms": med([r["total_minus_span"] for r in recs
                                        if r.get("total_minus_span") is not None]),
            "total_minus_union_gap_ms": med([r["total_minus_union_gap"] for r in recs]),
            "union_plus_cb_minus_total_ms": med([r["union_plus_cb_minus_total"] for r in recs]),
        },
        "submit_double_count": {
            # dp_lay_sub += dp_last_submit_us is charged per layer visit. If sum_n
            # exceeds the row count, submit was charged more than once.
            "n_steps_with_n_gt_rows": sum(1 for r in recs if r["sum_n"] > r["n_layer_rows"]),
            "median_sum_n": med([r["sum_n"] for r in recs]),
            "median_rows": med([r["n_layer_rows"] for r in recs]),
        },
        "gpu_ge_union_all_steps": all(r["gpu_ge_union"] for r in recs),
    }


def cmd_selftest() -> int:
    # Counted, not hard-coded: a hard-coded expected count silently passes when a
    # new check is added and never updated.
    ok = 0
    total = 0

    def check(name, cond):
        nonlocal ok, total
        total += 1
        print("  %-60s %s" % (name, "ok" if cond else "FAIL"))
        ok += 1 if cond else 0

    synthetic = (
        "CGC-DECPROF: step=1 segs=41 layers=2 total=100.00 ms | "
        "wait=40.00 (40%) cb=30.00 (30%) submit=20.00 (20%) ntok=4 | "
        "layer gpu_sum=50.00 union_sum=40.00 gap_sum=30.00 ms\n"
        "CGC-DECPROF all: L0 wait=20.00 cb=15.00 submit=10.00 ms gpu=25.00 union=20.00 "
        "gap=10.00 sg=1 n=1 st=0.000 en=30.000\n"
        "CGC-DECPROF all: L1 wait=20.00 cb=15.00 submit=10.00 ms gpu=25.00 union=20.00 "
        "gap=20.00 sg=1 n=1 st=30.000 en=60.000\n")
    p = Path("/tmp/_joint_reconcile_selftest.log")
    p.write_text(synthetic)
    steps = parse_log(p)
    check("one step parsed", len(steps) == 1)
    check("two layer rows parsed", len(steps[0]["layers_rows"]) == 2)
    r = reconcile_step(steps[0])
    check("cpu resid = 100 - 90 = 10", abs(r["cpu_resid"] - 10.0) < 1e-6)
    check("layer wait sums to step wait", abs(r["layer_wait_vs_step"] - 0.0) < 1e-6)
    check("span = 60 - 0 = 60", abs(r["span"] - 60.0) < 1e-6)
    # union 40 + gap 30 = 70 vs span 60 -> residual +10 (overlapping / not serial)
    check("gpu span resid = 60 - 70 = -10", abs(r["gpu_span_resid"] + 10.0) < 1e-6)
    check("clock_safe detected", r["clock_safe"] is True)
    # a log without st=/en= must say so rather than fake a zero residual
    p2 = Path("/tmp/_joint_reconcile_nost.log")
    p2.write_text("\n".join(synthetic.splitlines()[:1]) + "\n"
                  + "CGC-DECPROF all: L0 wait=20.00 cb=15.00 submit=10.00 ms gpu=25.00 "
                    "union=20.00 gap=10.00 sg=1 n=1\n")
    r2 = reconcile_step(parse_log(p2)[0])
    check("no st/en -> clock_safe False", r2["clock_safe"] is False)
    check("no st/en -> resid None not 0", r2["gpu_span_resid"] is None)
    check("submit charged once (sum_n == rows)", r["sum_n"] == r["n_layer_rows"])
    # synthetic: total 100, union 20+20=40, gap 10+20=30 -> outside = 30
    check("total_minus_union_gap = 100-70 = 30", abs(r["total_minus_union_gap"] - 30.0) < 1e-6)
    # synthetic: union 40 + cb 30 = 70 vs total 100 -> -30 (fits inside the step)
    check("union_plus_cb - total = -30", abs(r["union_plus_cb_minus_total"] + 30.0) < 1e-6)
    print("\nselftest: %d checks" % ok)
    return 0 if ok == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="close the decode-step books from one log")
    ap.add_argument("--log", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--all-rows", action="store_true",
                    help="keep shadow rows too (default: work rows only, sum_gap>0, "
                         "the same rule cb_miss_regression.py uses)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest()
    if not args.log:
        print("need --log (or --selftest)", file=__import__("sys").stderr)
        return 2

    recs = [reconcile_step(s) for s in parse_log(Path(args.log)) if s["layers_rows"]]
    n_all = len(recs)
    if not args.all_rows:
        # CGC_GPU_TIMING makes every turn emit a work row and a shadow row. The
        # shadow row has ~1-2 ms total and one layer row; pooling them produces a
        # median that describes neither population.
        recs = [r for r in recs if r["sum_gap"] > 0]
    if not recs:
        print("no work steps in %s (%d rows before filtering)" % (args.log, n_all))
        return 1
    summ = summarise(recs)

    print("NOTE: cpu_resid is a TAUTOLOGY (total IS wait+cb+submit, ggml-backend.cpp:2691)"
          " -- it cannot fail. The informative closure is total_minus_union_gap.\n")
    print("log: %s" % args.log)
    print("work steps: %d of %d rows  (clock-safe: %d)\n" % (
        summ["n_steps"], n_all, summ["n_clock_safe"]))
    print("%5s %5s %9s %10s %10s %11s %11s" % (
        "step", "ntok", "total", "cpu_resid", "gap-(cb+sub)", "span_resid", "sub_roll"))
    for r in recs[:20]:
        gs = "%11.2f" % r["gpu_span_resid"] if r["gpu_span_resid"] is not None else "%11s" % "-"
        gm = "%11.2f" % r["gap_minus_cb_submit"] if r["gap_minus_cb_submit"] is not None else "%11s" % "-"
        print("%5d %5d %9.2f %10.2f %s %s %11.2f" % (
            r["step"], r["ntok"], r["total"], r["cpu_resid"], gm, gs,
            r["layer_submit_vs_step"]))
    print("\nmedians:")
    for k, v in summ["median"].items():
        print("  %-28s %s" % (k, "n/a" if v is None else round(v, 3)))
    print("\nsubmit attribution: %s" % summ["submit_double_count"])
    if not summ["clock_safe_available"]:
        print("\nNOTE: no st=/en= in this log => the clock-safe GPU closure could not be "
              "read. The 10.09 ms failure is a CROSS-CLOCK comparison and cannot be "
              "settled from this log alone.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"log": args.log, "summary": summ, "steps": recs},
            ensure_ascii=False, indent=2) + "\n")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
