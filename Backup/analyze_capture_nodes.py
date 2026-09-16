#!/usr/bin/env python3
"""§9.18.6 extension: WHERE does the first divergence occur, per graph, between two arms?

Same pairing rule as scripts/check/ids_capture_diff.py -- graphs are segmented at the
`ffn_moe_gate-1` IDS row and rows are keyed BY NAME (first occurrence wins) -- extended to a list of
captured nodes instead of one.

Two readings, deliberately kept apart:
  * rows with path=MV|MM are the 09-15 ids instrument: what mul_mat_id actually CONSUMED.
  * rows with path=DST are tensor OUTPUTS, one per name in CGC_TENSOR_CAPTURE.

The localisation is the ORDER of the divergence, not the count of them. Rows are emitted in
SUBMISSION order (the dump merges the two streams by seq), so within a graph the earliest DST row
that differs is the earliest point at which the two arms part. Everything downstream of it is then
explained and carries no information -- which is why counting DIFFs across a graph is the wrong
statistic.

Usage:
    analyze_capture_nodes.py LOG_A LOG_B [--graphs N] [--only PREFIX]
"""
import argparse
import re
import sys
from pathlib import Path

RX = re.compile(r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\](?: fuse=(\d+))?")
GRAPH_START = "ffn_moe_gate-1"

# The layer chain, taken from the graph builder (qwen35moe.cpp + llama-graph.cpp), NOT from the
# emission order. Emission order is NOT usable as the localisation key: measured 2026-09-16, the
# same arm captured twice emits the SAME nodes in a DIFFERENT order (e.g. `linear_attn_out-2` before
# and after `attn_residual-2` in different graphs and different runs), while every value is
# reproducible. So the schedule this backend produces is not order-stable, and anything that reads
# "which row came first" as "which node computed first" is reading noise.
CHAIN = [
    "l_out-0.dst",                                                             # layer 0 (last)
    "l_out-1.dst",                                                             # layer 1 (last)
    "attn_norm-2.dst",                                                         # layer 2 attention input
    "z-2.dst", "gate-2.dst",                                                   # its two input projections
    "norm-2.dst",                                                              # gated norm, AFTER the core
    "linear_attn_out-2.dst",                                                   # its output projection
    "attn_residual-2.dst", "attn_post_norm-2.dst",                             # residual + router input
    "ffn_moe_logits_raw-2.dst",                                                # router logits (pre top-k)
    "ffn_moe_gate-2.dst", "ffn_moe_down-2.dst", "ffn_moe_out-2.dst",           # the MoE itself
    "l_out-2.dst",
]


def chain_rank(name):
    try:
        return CHAIN.index(name)
    except ValueError:
        return len(CHAIN) + 1


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if m:
            vals = tuple(int(x) for x in m.group(5).split(",")) if m.group(5) else ()
            rows.append((m.group(4), m.group(2), vals, int(m.group(6) or 0)))
    graphs, cur = [], []
    for name, p, vals, fuse in rows:
        if name == GRAPH_START and cur:
            graphs.append(cur)
            cur = []
        cur.append((name, p, vals, fuse))
    if cur:
        graphs.append(cur)
    return graphs


def split(g):
    """-> (ordered list of (name, vals) for DST rows, {name: vals} for ids rows)."""
    dst, ids = [], {}
    for name, p, vals, _fuse in g:
        if p == "DST":
            dst.append((name, vals))
        else:
            ids.setdefault(name, vals)
    return dst, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--graphs", type=int, default=6, help="how many graphs to tabulate")
    ap.add_argument("--only", default="", help="restrict the table to names with this prefix")
    ap.add_argument("--upto", type=int, default=0,
                    help="consider only graphs [1, --upto). Use it: the IDS destination is capped at "
                         "4096 slots and a full pass costs ~114, so the LAST chunks are not delimited "
                         "by a graph marker at all -- they merge several forward passes, and pairing by "
                         "name inside them compares rows from different passes. 24 (= graphs 1..23) is "
                         "what the 2026-09-16 runs support.")
    args = ap.parse_args()

    a, b = parse(args.a), parse(args.b)
    n = min(len(a), len(b))
    if args.upto:
        n = min(n, args.upto)
    print(f"A graphs={len(a)}  B graphs={len(b)}  comparing {n}"
          + (f" (--upto {args.upto})" if args.upto else ""))
    if len(a) != len(b):
        print("  !! graph counts differ -- compare only the common prefix, and say so")
    print()

    # ---- ids status per graph (the 09-15 instrumentation, re-derived in this very run) ----
    print("=== ids (what mul_mat_id consumed) ===")
    print(f"{'graph':>6} {'ids SAME':>9} {'ids DIFF':>9} {'DST SAME':>9} {'DST DIFF':>9}")
    for i in range(n):
        da, ia = split(a[i])
        db, ib = split(b[i])
        same = sum(1 for k in ia if k in ib and ia[k] == ib[k])
        diff = sum(1 for k in ia if k in ib and ia[k] != ib[k])
        ma, mb = dict(da), dict(db)
        dsame = sum(1 for k in ma if k in mb and ma[k] == mb[k])
        ddiff = sum(1 for k in ma if k in mb and ma[k] != mb[k])
        print(f"{i:>6} {same:>9} {diff:>9} {dsame:>9} {ddiff:>9}")
    print()

    # ---- per-node: the first graph in which it diverges, plus its position in the graph ----
    # Ordered by the SOURCE CHAIN (see CHAIN above), not by emission order.
    order = []
    for name, _v in split(a[min(3, n - 1)])[0]:
        if name not in order:
            order.append(name)
    order.sort(key=chain_rank)

    print("=== localisation: first graph where each DST node DIVERGES, in the SOURCE layer chain ===")
    print("  (pos comes from the builder's chain order, not from emission order -- see CHAIN above)")
    print("  ABSENT means the name was not captured in BOTH logs -- which is NOT 'identical'. A node is")
    print("  missing from the stream whenever it happened to be absorbed as the last node of a FUSED")
    print("  group, and the fusion state depends on node ORDER, which this fork does not keep stable.")
    print(f"  {'pos':>3} {'node':36s} {'first_diff':>10} {'present':>8} {'ids_at_that_graph':>18}")
    first_overall = None
    for pos, name in enumerate(order):
        if args.only and not name.startswith(args.only):
            continue
        fd = None
        present = 0
        for i in range(n):
            ma, mb = dict(split(a[i])[0]), dict(split(b[i])[0])
            if name in ma and name in mb:
                present += 1
                if fd is None and ma[name] != mb[name]:
                    fd = i
        ids_note = ""
        if fd is not None:
            _da, ia = split(a[fd])
            _db, ib = split(b[fd])
            same = sum(1 for k in ia if k in ib and ia[k] == ib[k])
            diff = sum(1 for k in ia if k in ib and ia[k] != ib[k])
            ids_note = f"{same}S/{diff}D"
            if first_overall is None or fd < first_overall[0]:
                first_overall = (fd, name)
        verdict = "never" if fd is None else str(fd)
        if present == 0:
            verdict = "ABSENT"
        print(f"  {pos:>3} {name:36s} {verdict:>10} {present:>8} {ids_note:>18}")
    print()

    if first_overall is not None:
        gi, gname = first_overall
        print(f"=> earliest divergence: graph {gi}, node {gname}")
        ma, mb = dict(split(a[gi])[0]), dict(split(b[gi])[0])
        print(f"   A {list(ma[gname])[:10]}")
        print(f"   B {list(mb[gname])[:10]}")
        # everything BEFORE it in the SOURCE CHAIN that agreed -- that is the exoneration list
        before = []
        for nm in order:
            if nm == gname:
                break
            if nm in ma and nm in mb:
                before.append(f"{nm}={'SAME' if ma[nm] == mb[nm] else 'DIFF'}")
        print(f"   and, earlier in the layer chain: {' '.join(before) if before else '(nothing captured)'}")
    else:
        print("=> every captured node agreed in every graph: this list exonerates all of them")
    print()

    # ---- per-graph detail for the first few, one line per captured node ----
    lim = min(args.graphs, n)
    for i in range(lim):
        da, ia = split(a[i])
        db, _ib = split(b[i])
        mb = dict(db)
        cells = []
        for name, vals in sorted(da, key=lambda nv: chain_rank(nv[0])):
            if args.only and not name.startswith(args.only):
                continue
            if name not in mb:
                cells.append(f"{name}=MISSING")
            else:
                cells.append(f"{name}={'SAME' if vals == mb[name] else 'DIFF'}")
        print(f"-- graph {i}: " + "  ".join(cells))


if __name__ == "__main__":
    sys.exit(main())
