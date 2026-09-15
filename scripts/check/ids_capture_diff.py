#!/usr/bin/env python3
"""Diff two CGC-IDS-CAP captures (kernel-side mul_mat_id ids) graph by graph, node by node.

Why this probe and not CGC-MMID-ASSERT: CGC-IDS-CAP is the only reading taken AT THE MOMENT the
consumer runs -- kernel_cgc_ids_capture is submitted into the same command buffer, immediately
after the kernel that consumes the ids. CGC-MMID-ASSERT reads the same operand from the HOST
during graph ENCODING, where for a device-computed tensor the GPU has not written it yet, so it
reports the buffer's previous occupant. The two disagree in the most misleading way possible:
on the S1 arm ASSERT says id_oob=16/16 (first=1063585220 = the bit pattern of F32 0.90, i.e. a
router probability) while CAP says every id is inside [0, ne02). Both statements are true; they
are about different moments (eng-mh-0008).

Pairing rule: slots are paired by (graph index, node name), NEVER by slot index alone. Each
encode takes a fresh slot from a monotone counter, so equal slot indices across arms only mean
"the Nth mul_mat_id of the run" and stop agreeing the moment one arm runs a different number of
graphs. Graph boundaries are detected by the node name returning to the first layer's gate.

Known instrument limit: the capture reads a fixed stride (8 words) from each ids operand, so for
n_ids = 8*n_tokens only the FIRST token's ids are visible. A divergence confined to tokens 1..N-1
is invisible here; re-run with a non-zero n_skip to reach the tail.

Usage:
    python3 scripts/check/ids_capture_diff.py LOG_A LOG_B [--graph N] [--max N] [--model-token H]
"""

import argparse
import collections
import re
import sys

RX = re.compile(
    r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\]")
# The first MoE layer's gate node marks the start of a new graph. Hard-coded rather than inferred
# because a wrong boundary would silently mis-pair every subsequent slot, which is exactly the
# failure that makes this tool worth having.
GRAPH_START_NODE = "ffn_moe_gate-1"
TOKEN_FORMAT_HEAD = "CGC-MODEL-TOKENS "


class Slot:
    __slots__ = ("slot", "path", "n_ids", "name", "ids")

    def __init__(self, slot, path, n_ids, name, ids):
        self.slot, self.path, self.n_ids, self.name, self.ids = slot, path, n_ids, name, ids

    def key(self):
        return self.name


def parse(path):
    """Return (graphs, tokens). tokens is the model's own token list, when the log carries one."""
    slots = []
    tokens = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if tokens is None and TOKEN_FORMAT_HEAD in line:
                tokens = line.split(TOKEN_FORMAT_HEAD, 1)[1].strip().split(",")
            m = RX.search(line)
            if m:
                ids = tuple(int(x) for x in m.group(5).split(",")) if m.group(5) else ()
                slots.append(Slot(int(m.group(1)), m.group(2), int(m.group(3)),
                                  m.group(4), ids))

    graphs, cur = [], []
    for s in slots:
        if s.name == GRAPH_START_NODE and cur:
            graphs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        graphs.append(cur)
    return graphs, tokens


def graph_map(g):
    """name -> Slot, keeping the FIRST occurrence so a duplicate name cannot silently win."""
    out = {}
    for s in g:
        out.setdefault(s.name, s)
    return out


def fmt(ids):
    return "[" + ",".join(str(v) for v in ids) + "]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_a", help="reference capture (e.g. the baseline arm)")
    ap.add_argument("log_b", help="candidate capture (e.g. the S1 arm)")
    ap.add_argument("--graph", type=int, default=0, help="graph index to compare (default 0)")
    ap.add_argument("--max", type=int, default=12, help="max rows to print per verdict")
    ap.add_argument("--all-graphs", action="store_true",
                    help="compare every graph both captures share, not just --graph")
    args = ap.parse_args()

    ga, ta = parse(args.log_a)
    gb, tb = parse(args.log_b)
    if not ga or not gb:
        print("no CGC-IDS-CAP rows in one of the logs; was CGC_IDS_CAPTURE set?", file=sys.stderr)
        return 2

    print(f"A {args.log_a}")
    print(f"  graphs={len(ga)}  slots={sum(len(g) for g in ga)}  tokens={ta if ta else '<not logged>'}")
    print(f"B {args.log_b}")
    print(f"  graphs={len(gb)}  slots={sum(len(g) for g in gb)}  tokens={tb if tb else '<not logged>'}")
    print()

    if ta and tb and ta != tb:
        # Not fatal: graph 0 is the shared prompt, so it can still be compared. But say so, because
        # every graph after the first is expected to differ for a reason that has nothing to do
        # with the change under test.
        same_head = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"NOTE: token streams differ from index {same_head}. Graphs after the shared prefix")
        print("      are expected to differ; only the shared prefix is evidence.")
        print()

    idxs = range(min(len(ga), len(gb))) if args.all_graphs else [args.graph]
    total_diff = 0
    for gi in idxs:
        if gi >= len(ga) or gi >= len(gb):
            print(f"graph {gi}: not present in both captures (A has {len(ga)}, B has {len(gb)})")
            continue
        ma, mb = graph_map(ga[gi]), graph_map(gb[gi])
        keys = list(ma.keys())
        diffs = []
        for k in keys:
            if k not in mb:
                diffs.append((k, "missing in B", ma[k], None))
                continue
            if ma[k].ids != mb[k].ids:
                diffs.append((k, "ids differ", ma[k], mb[k]))
            elif ma[k].n_ids != mb[k].n_ids:
                diffs.append((k, "n_ids differ", ma[k], mb[k]))
            elif ma[k].path != mb[k].path:
                diffs.append((k, "path differs", ma[k], mb[k]))
        extra = [k for k in mb if k not in ma]
        total_diff += len(diffs) + len(extra)
        status = "IDENTICAL" if not diffs and not extra else f"{len(diffs)+len(extra)} DIFF"
        print(f"graph {gi}: {len(keys)} nodes compared -> {status}")
        for k, why, sa, sb in diffs[: args.max]:
            print(f"    {why:<12} {k}")
            print(f"        A path={sa.path} n_ids={sa.n_ids} ids={fmt(sa.ids)}")
            if sb is not None:
                print(f"        B path={sb.path} n_ids={sb.n_ids} ids={fmt(sb.ids)}")
        if len(diffs) > args.max:
            print(f"    ... and {len(diffs) - args.max} more")
        for k in extra[: args.max]:
            print(f"    only in B  {k} ids={fmt(mb[k].ids)}")

    print()
    print(f"TOTAL differing nodes: {total_diff}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
