#!/usr/bin/env python3
"""Split CGC-NSM lines into individual graph computes and profile the decode ones.

A new graph starts whenever a command buffer's first node index `a` goes back down
(segmented computes restart node numbering at 1). Prefill and draft/verify graphs
therefore separate without any timestamp.

Usage: python3 scripts/check/per_op_graphs.py <log> [--min-nodes N] [--top N]
"""
import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict

NSM = re.compile(r"^CGC-NSM a=(\d+) b=(\d+) dur_ns=(\d+) nk=(\d+)(.*)$")


def parse_kinds(rest):
    out = {}
    for tok in rest.split():
        if ":" in tok:
            k, v = tok.rsplit(":", 1)
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--min-nodes", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    graphs = []
    cur = []
    prev_a = -1
    with open(args.log, errors="replace") as fh:
        for line in fh:
            m = NSM.match(line)
            if not m:
                continue
            a = int(m.group(1))
            if a <= prev_a:
                if cur:
                    graphs.append(cur)
                cur = []
            prev_a = a
            cur.append((a, int(m.group(2)), int(m.group(3)),
                        int(m.group(4)), parse_kinds(m.group(5))))
    if cur:
        graphs.append(cur)

    print(f"graphs={len(graphs)}")
    sizes = sorted(len(g) for g in graphs)
    print(f"  slices per graph: p10={sizes[len(sizes)//10]} "
          f"p50={sizes[len(sizes)//2]} p90={sizes[9*len(sizes)//10]} "
          f"max={sizes[-1]}")

    rows = []
    for g in graphs:
        n_nodes = max(s[1] for s in g)
        tot_us = sum(s[2] for s in g) / 1000.0
        rows.append((n_nodes, len(g), tot_us, g))
    rows.sort(key=lambda r: r[0])
    print("\n=== graph census (sorted by node count) ===")
    print(f"{'nodes':>7s} {'slices':>7s} {'sum_us':>12s} {'us/slice':>9s}")
    for n_nodes, nsl, tot, _ in rows:
        if n_nodes < args.min_nodes:
            continue
        print(f"{n_nodes:7d} {nsl:7d} {tot:12.1f} {tot/nsl:9.2f}")

    # the "decode" population: graphs whose node count is at the mode
    mode = Counter(r[0] for r in rows).most_common(3)
    print(f"\n  most common node counts: {mode}")
    target = mode[0][0]
    sel = [r for r in rows if r[0] == target]
    print(f"\n=== decode population: n_nodes=={target} ({len(sel)} graphs) ===")
    sums = [r[2] for r in sel]
    print(f"  sum of slice durations per graph: median={statistics.median(sums):.1f} us "
          f"min={min(sums):.1f} max={max(sums):.1f}")
    nsl = [r[1] for r in sel]
    print(f"  slices per graph: median={statistics.median(nsl):.0f}")

    # per-slice stats inside the decode population
    allsl = [s for r in sel for s in r[3]]
    d = sorted(s[2] for s in allsl)
    def pct(p):
        return d[int(p / 100 * (len(d) - 1))] / 1000.0
    print(f"\n  slice durations (us): p1={pct(1):.2f} p10={pct(10):.2f} "
          f"p50={pct(50):.2f} p90={pct(90):.2f} p99={pct(99):.2f} max={d[-1]/1000.0:.2f}")
    tot = sum(d)
    print(f"  n={len(d)}  sum={tot/1000.0:.1f} us")
    bands = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)]
    print("\n  === mass by slice-duration band (decode only) ===")
    for lo, hi in bands:
        s2 = [x for x in d if lo <= x / 1000.0 < hi]
        if not s2:
            continue
        print(f"   [{lo:>4.0f},{hi:>6.0f}) us: n={len(s2):6d} "
              f"({100.0*len(s2)/len(d):5.1f}% slices)  "
              f"{sum(s2)/1000.0:9.1f} us ({100.0*sum(s2)/tot:5.1f}% time)")

    # exact per-node cost from uniform slices in the decode population
    exact = defaultdict(list)
    for a, b, dur, nk, kinds in allsl:
        if len(kinds) == 1:
            (k, c), = kinds.items()
            if c == nk:
                exact[k].append(dur / nk / 1000.0)
    print("\n  === EXACT per-node us (uniform slices, decode only) ===")
    print(f"  {'kind':20s} {'n':>6s} {'median':>9s} {'mean':>9s} {'min':>7s} {'max':>9s} "
          f"{'us*N/graph':>11s}")
    for k, v in sorted(exact.items(), key=lambda kv: -statistics.median(kv[1])):
        med = statistics.median(v)
        per_graph = len(v) / len(sel)
        print(f"  {k:20s} {len(v):6d} {med:9.2f} {statistics.fmean(v):9.2f} "
              f"{min(v):7.2f} {max(v):9.2f} {med*per_graph:11.1f}")

    print(f"\n=== hottest {args.top} decode slices ===")
    for a, b, dur, nk, kinds in sorted(allsl, key=lambda s: -s[2])[:args.top]:
        ks = " ".join(f"{k}:{v}" for k, v in kinds.items())
        print(f"  dur={dur/1000.0:8.2f} us nodes[{a},{b}) nk={nk}  {ks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
