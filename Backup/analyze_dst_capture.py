#!/usr/bin/env python3
"""§9.18.6: does `ffn_moe_down-1`'s OUTPUT differ between the baseline and the S1 arm?

Reuses the same pairing rule as scripts/check/ids_capture_diff.py -- graphs are segmented at the
`ffn_moe_gate-1` ids row (each id row is submitted before its graph's dst rows, so the marker still
lands at the head of every graph), and nodes are keyed BY NAME, keeping the first occurrence.

Then answers, per graph, three separate questions:
  1. do the ids still agree?             (re-derives §9.18.3 in this very run)
  2. does `<node>.dst` agree?            (the new reading)
  3. if dst differs, did the ids agree?  (the only combination that exonerates the mapping layer)

Usage: python3 Backup/analyze_dst_capture.py LOG_A LOG_B [node]
"""
import re
import sys
from pathlib import Path

RX = re.compile(r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\]")
GRAPH_START = "ffn_moe_gate-1"


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if m:
            vals = tuple(int(x) for x in m.group(5).split(",")) if m.group(5) else ()
            rows.append((m.group(4), m.group(2), vals))
    graphs, cur = [], []
    for name, path_, vals in rows:
        if name == GRAPH_START and cur:
            graphs.append(cur)
            cur = []
        cur.append((name, path_, vals))
    if cur:
        graphs.append(cur)
    return graphs


def gmap(g):
    out = {}
    for name, p, vals in g:
        out.setdefault(name, (p, vals))
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
node = sys.argv[3] if len(sys.argv) > 3 else "ffn_moe_down-1"
dst = node + ".dst"
print(f"A graphs={len(a)}  B graphs={len(b)}   node={node}")
print(f"{'graph':>6} {'ids SAME':>9} {'ids DIFF':>9} {'dst':>6}  gate/up/down ids of THIS node")
first_dst_diff = None
for i, (ga, gb) in enumerate(zip(a, b)):
    ma, mb = gmap(ga), gmap(gb)
    same = diff = 0
    for k in ma:
        if k.endswith(".dst") or k not in mb:
            continue
        if ma[k][1] == mb[k][1]:
            same += 1
        else:
            diff += 1
    d = "n/a"
    if dst in ma and dst in mb:
        d = "SAME" if ma[dst][1] == mb[dst][1] else "DIFF"
        if d == "DIFF" and first_dst_diff is None:
            first_dst_diff = i
    ids_here = []
    for k in (node, "ffn_moe_gate-1", "ffn_moe_up-1"):
        if k in ma and k in mb:
            ids_here.append(f"{k.split('-')[0][8:]}={'SAME' if ma[k][1] == mb[k][1] else 'DIFF'}")
    print(f"{i:>6} {same:>9} {diff:>9} {d:>6}  {' '.join(ids_here)}")

print()
print(f"first graph where {dst} DIFFERS: {first_dst_diff}")
if first_dst_diff is not None:
    ma, mb = gmap(a[first_dst_diff]), gmap(b[first_dst_diff])
    print(f"  at that graph, {node} ids: "
          f"{'SAME' if ma[node][1] == mb[node][1] else 'DIFF'}")
    print(f"  at that graph, {dst}:")
    print(f"    A {list(ma[dst][1])[:10]}")
    print(f"    B {list(mb[dst][1])[:10]}")
    # The decisive combination.
    if ma[node][1] == mb[node][1] and ma[dst][1] != mb[dst][1]:
        print()
        print("  => DECISIVE: identical ids, identical input, DIFFERENT output. The ids point at")
        print("     different weights, so the carrier is the pool/slot CONTENT, not the mapping.")
    elif ma[node][1] != mb[node][1]:
        print()
        print("  => NOT decisive at this graph: the ids differ here too, so the output difference is")
        print("     already explained upstream. Walk earlier graphs to find the first divergence.")
