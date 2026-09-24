#!/usr/bin/env python3
"""Parse CGC-NSM (per command buffer) lines from a llama-server log.

CGC-NSM a=<first node> b=<last node> dur_ns=<ns> nk=<nodes> [<kind>:<cnt> ...]

The point of the n_cb=32 + CGC_CB_N_MAIN=1 run is that a segment's command buffers
hold ~3 nodes each instead of >= 64, so a buffer whose nk nodes are ALL of one kind
gives an EXACT per-node GPU duration (dur / nk) with no model and no fit.

Usage:
    python3 scripts/check/per_op_slice_parse.py <log> [--top N]
"""
import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict

NSM = re.compile(
    r"^CGC-NSM a=(\d+) b=(\d+) dur_ns=(\d+) nk=(\d+)(.*)$")
STEP = re.compile(r"^CGC-GPUNODE: step=(\d+) seg_busy=([\d.]+) ms")


def parse_kinds(rest):
    out = {}
    for tok in rest.split():
        if ":" not in tok:
            continue
        k, v = tok.rsplit(":", 1)
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    slices = []
    steps = []
    with open(args.log, errors="replace") as fh:
        for line in fh:
            m = NSM.match(line)
            if m:
                a, b, dur, nk = (int(m.group(1)), int(m.group(2)),
                                 int(m.group(3)), int(m.group(4)))
                slices.append((a, b, dur, nk, parse_kinds(m.group(5))))
                continue
            m = STEP.match(line)
            if m:
                steps.append((int(m.group(1)), float(m.group(2))))

    if not slices:
        print("INVALID: no CGC-NSM lines (is CGC_GPU_NODES_MATRIX=1 set?)")
        return 2

    nk_hist = Counter(s[3] for s in slices)
    print(f"slices={len(slices)}  steps={len(steps)}  "
          f"nk histogram={dict(sorted(nk_hist.items()))}")
    if steps:
        per_step = len(slices) / len(steps)
        print(f"slices/step={per_step:.1f}  "
              f"seg_busy median={statistics.median([s[1] for s in steps]):.2f} ms")

    # ---- exact per-node cost: slices whose nk nodes are all one kind
    exact = defaultdict(list)
    for a, b, dur, nk, kinds in slices:
        if len(kinds) == 1:
            (k, c), = kinds.items()
            if c == nk:
                exact[k].append(dur / nk / 1000.0)  # us per node

    print(f"\n=== EXACT per-node GPU us (uniform slices only: {sum(len(v) for v in exact.values())} samples) ===")
    print(f"{'kind':22s} {'n':>6s} {'median us':>10s} {'mean us':>10s} {'min':>8s} {'max':>9s} {'total ms':>10s}")
    rows = sorted(exact.items(), key=lambda kv: -statistics.median(kv[1]))
    for k, v in rows:
        med = statistics.median(v)
        print(f"{k:22s} {len(v):6d} {med:10.2f} {statistics.fmean(v):10.2f} "
              f"{min(v):8.2f} {max(v):9.2f} {sum(v)/1000.0:10.2f}")

    # ---- duration distribution of ALL slices
    d = sorted(s[2] for s in slices)
    def pct(p):
        return d[int(p / 100 * (len(d) - 1))] / 1000.0
    print("\n=== slice duration distribution (all slices, us) ===")
    print(f"  p1={pct(1):.2f} p10={pct(10):.2f} p50={pct(50):.2f} "
          f"p90={pct(90):.2f} p99={pct(99):.2f} max={d[-1]/1000.0:.2f}")
    tot = sum(d) / 1e6
    print(f"  sum of ALL slice durations = {tot:.2f} ms "
          f"({tot/max(1,len(steps)):.2f} ms per step)")

    # ---- buckets: how much of the total sits in which slice-duration band
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100),
             (100, 300), (300, 1e9)]
    print("\n=== mass by slice-duration band ===")
    for lo, hi in bands:
        sel = [x for x in d if lo <= x / 1000.0 < hi]
        print(f"  [{lo:>4.0f},{hi:>6.0f}) us: n={len(sel):6d} "
              f"({100.0*len(sel)/len(d):5.1f}% of slices)  "
              f"{sum(sel)/1e6:8.2f} ms ({100.0*sum(sel)/sum(d):5.1f}% of time)")

    # ---- hottest slices
    print(f"\n=== hottest {args.top} slices ===")
    for a, b, dur, nk, kinds in sorted(slices, key=lambda s: -s[2])[:args.top]:
        ks = " ".join(f"{k}:{v}" for k, v in kinds.items())
        print(f"  dur={dur/1000.0:9.2f} us  nodes[{a},{b}) nk={nk}  {ks}")

    # ---- kind totals across all slices (node-count weighted, i.e. the old
    #      attribution) for comparison
    print("\n=== kind totals, node-count split over ALL slices (for contrast) ===")
    kind_ms = Counter()
    kind_n = Counter()
    for a, b, dur, nk, kinds in slices:
        tot_k = sum(kinds.values()) or 1
        for k, c in kinds.items():
            kind_ms[k] += dur * c / tot_k / 1e6
            kind_n[k] += c
    for k, ms in kind_ms.most_common(20):
        print(f"  {k:22s} {ms:9.2f} ms  nodes={kind_n[k]:6d}  "
              f"{ms/max(1,len(steps))*1000.0/max(1,kind_n[k])*len(steps):.2f} us/node(naive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
