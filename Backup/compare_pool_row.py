#!/usr/bin/env python3
"""§9.18.6 r12: do the two arms read the SAME BYTES for the SAME slot?

This reads the `path=POOL` rows (`Backup/run_ids_dst_capture.sh` with `POOL=1`), which carry, per
selected row: the id the consumer used and a device-side digest of the row's first `probe` bytes.

WHY THIS FILE EXISTS, as a decision rule rather than a diff tool:
    ids differ, digests differ  -> the routing/slot mapping is the carrier (a mapping defect)
    ids differ, digests same    -> the ids select different indices that HAPPEN to hold the same
                                   bytes -- i.e. the pool has duplicated content. Rare, but it would
                                   make "same ids" a weaker condition than it looks.
    ids same, digests differ    -> the SAME slot index holds different bytes at consumption time.
                                   That is the residency hypothesis, and it is the last sentence
                                   §9.18.4 could still stand on.
    ids same, digests same      -> that sentence falls too. The gather's two inputs are identical, so
                                   the divergence is NOT in what was read: look AFTER the gather
                                   (allocator aliasing, a view re-pointed per arm, a consumer of a
                                   stale ids buffer, the weighted-sum/combine node).

TWO WAYS THIS INSTRUMENT CAN LIE, both guarded here because both have already happened in this repo:
  1. rows=0 or every group n=0. A slot that was never written is filled with the 0x7fffffff sentinel,
     and a sentinel compares EQUAL to a sentinel. That is how a whole round was once read off
     all-sentinel rows (2026-09-17 00:30, the TAIL window bug). Such a row is reported as NOT-READ,
     never as SAME.
  2. Nothing shared. If no (graph, name) pair exists on both sides -- e.g. the node name differs
     between arms, or the runs captured different layers -- then "no differences" means nothing was
     compared. Reported as NO-OVERLAP and the verdict is withheld.

SAME is strong evidence, not proof: a compensating multiset change can collide. The digest is
(sum, xor, weighted-sum) over BYTES, so it is sensitive to a permuted row as well as to a changed one.
"""
import argparse
import re
import sys
from pathlib import Path

# `path=(\w+)` deliberately matches only the first word: POOL rows carry `from=MV|MM` after the
# geometry fields, and the ids/dst rows carry `fuse=`/`ne=`/`off=`. One regex, three row shapes, and a
# row whose fields do not match at all is NOT silently dropped -- see the unmatched counter below.
RX_POOL = re.compile(
    r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\]"
    r"(?: rows=(\d+) probe=(\d+) nsel=(\d+) from=(\w+))?")
# Graph boundaries live in the IDS rows (see the note in ggml_metal_ops.cpp's dump): every stream is
# merged in submission order, so the marker is seen in order by this reader too.
GRAPH_START = "ffn_moe_gate-1"

SENTINEL = 0x7fffffff  # as int32


def as_i32(v):
    return v - (1 << 32) if v > 0x7FFFFFFF else v


def parse(path):
    """-> [(graph, name, n_groups, groups, rows, probe, nsel, origin)] with groups = [(id, s, x, w, n)]"""
    out = []
    graph = 0
    n_pool = 0
    n_other = 0
    n_unmatched = 0
    for line in open(path, errors="replace"):
        if "CGC-IDS-CAP" not in line:
            continue
        # The instrument also prints INIT BANNERS under the same tag ("CGC-IDS-CAP enabled: slots=...",
        # "CGC-POOL-CAP enabled: slots=..."). They are not rows. Requiring the literal `slot=`
        # (`slots=` does not contain it) is what keeps them out of the unmatched counter -- a counter
        # that fires on every healthy run is worse than no counter, because the next genuinely
        # malformed row hides inside its noise.
        if "slot=" not in line:
            continue
        m = RX_POOL.search(line)
        if m is None:
            n_unmatched += 1
            continue
        _slot, pth, _n_ids, name, ids, rows, probe, nsel, origin = m.groups()
        if pth != "POOL":
            n_other += 1
            # the graph marker is a ffn_moe_gate-1 IDS row: `name=ffn_moe_gate-1` with no suffix
            if name == GRAPH_START:
                graph += 1
            continue
        n_pool += 1
        if rows is None:
            # a POOL row with no geometry is a parser/tooling mismatch, not a data point
            out.append((graph, name, -1, [], 0, 0, 0, "?"))
            continue
        rows = int(rows)
        words = [as_i32(int(t)) for t in ids.split(",")] if ids else []
        # layout: words[0] = groups written, then 5 words per group
        n_groups = words[0] if words else -1
        groups = []
        for r in range(rows):
            b = 1 + 5 * r
            if b + 4 < len(words):
                groups.append(tuple(words[b:b + 5]))
        out.append((graph, name, n_groups, groups, rows, int(probe), int(nsel), origin))
    return out, n_pool, n_other, n_unmatched


def fmt_group(g):
    if g is None:
        return "<none>"
    ident, s, x, w, n = g
    return f"id={ident} sum={s} xor={x} wsum={w} nbytes={n}"


def keyed(rows):
    """(graph, name) -> row, first occurrence wins (same rule as the other comparators)."""
    d = {}
    for r in rows:
        k = (r[0], r[1])
        if k not in d:
            d[k] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--name", help="only this node name (without the .pool suffix)")
    ap.add_argument("--max-graphs", type=int, default=4, help="0 = all")
    ap.add_argument("--rows", type=int, default=0, help="only this many row groups (0 = all)")
    ap.add_argument("--quiet-same", action="store_true")
    args = ap.parse_args()

    A, na, oa, ua = parse(args.log_a)
    B, nb, ob, ub = parse(args.log_b)

    print(f"A = {Path(args.log_a).name}")
    print(f"B = {Path(args.log_b).name}")
    print(f"rows parsed: A pool={na} other={oa} unmatched={ua} | B pool={nb} other={ob} unmatched={ub}")

    if ua or ub:
        print("  !! some CGC-IDS-CAP lines did not match the parser. They are NOT compared, and a "
              "missing comparison is not a SAME. Check the dump format against RX_POOL.")
    if na == 0 or nb == 0:
        print("  !! one side has NO pool rows at all. Either CGC_POOL_CAPTURE did not reach the "
              "server (run_server.sh drops unlisted CGC_* SILENTLY) or the filter matched no node.")
        return 1

    da, db = keyed(A), keyed(B)
    shared = sorted(set(da) & set(db))
    only_a = sorted(set(da) - set(db))
    only_b = sorted(set(db) - set(da))

    maxg = args.max_graphs
    if maxg > 0:
        shared = [k for k in shared if k[0] <= maxg]

    names = sorted({k[1] for k in shared})
    if args.name:
        want = args.name if args.name.endswith(".pool") else args.name + ".pool"
        names = [n for n in names if n == want]
        if not names:
            print(f"  !! no shared row named {want}. Names present: {sorted({k[1] for k in shared})}")
            return 1

    print(f"shared (graph,name) keys: {len(shared)}  names: {names}")
    if only_a or only_b:
        print(f"  only in A: {only_a[:8]}{' ...' if len(only_a) > 8 else ''}")
        print(f"  only in B: {only_b[:8]}{' ...' if len(only_b) > 8 else ''}")
        print("  ABSENT is reported, never counted as identical -- a name missing from one arm means "
              "the two runs are not comparable for it.")

    if not shared:
        print("  !! NO-OVERLAP: nothing is shared, so 'no differences' would be vacuous. Verdict "
              "withheld.")
        return 1

    n_same = n_diff = n_notread = 0
    first = None
    wanted = set(names)
    for k in shared:
        ra, rb = da[k], db[k]
        g, name = k
        if name not in wanted:
            continue
        _ga, _na_, nga, groups_a, rows_a, probe_a, nsel_a, org_a = ra
        _gb, _nb_, ngb, groups_b, rows_b, probe_b, nsel_b, org_b = rb

        nrows = min(rows_a, rows_b)
        if args.rows:
            nrows = min(nrows, args.rows)

        # NOT-READ -- the strongest guard in this file, and the one with a precedent: a round was once
        # read off rows that were entirely sentinel, because a sentinel compares EQUAL to a sentinel.
        # "Nothing was read on either side" must never be able to come out as SAME. Keyed on
        # `nbytes == 0` (the fourth word of a group), because that is the field the KERNEL writes, and
        # the host-side group count cannot distinguish "read nothing" from "wrote nothing".
        read_a = sum(1 for t in groups_a[:nrows] if t[4] > 0)
        read_b = sum(1 for t in groups_b[:nrows] if t[4] > 0)
        if (rows_a == 0 or rows_b == 0 or nga == 0 or ngb == 0 or
                (read_a == 0 and read_b == 0)):
            n_notread += 1
            print(f"  g{g:<3} {name:<22} NOT-READ  rows={rows_a}/{rows_b} groups={nga}/{ngb} "
                  f"read={read_a}/{read_b} probe={probe_a}/{probe_b}  <-- nothing was written/read; "
                  f"NOT a SAME")
            continue

        diffs = []
        for r in range(nrows):
            ga = groups_a[r] if r < len(groups_a) else None
            gb = groups_b[r] if r < len(groups_b) else None
            if ga != gb:
                diffs.append((r, ga, gb))

        # an unread row on exactly ONE side is a difference in the measurement, and it must not be
        # able to masquerade as agreement
        zero_a = nrows - read_a
        zero_b = nrows - read_b
        tag = "" if zero_a == zero_b == 0 else f"  (unread rows: A={zero_a} B={zero_b})"

        if not diffs:
            n_same += 1
            if not args.quiet_same:
                print(f"  g{g:<3} {name:<22} SAME      rows={nrows} probe={probe_a}/{probe_b} "
                      f"nsel={nsel_a}/{nsel_b} from={org_a}/{org_b}{tag}")
        else:
            n_diff += 1
            r, ga, gb = diffs[0]
            print(f"  g{g:<3} {name:<22} DIFF      row={r}/{nrows}  A: {fmt_group(ga)}\n"
                  f"  {'':<6} {'':<22}           B: {fmt_group(gb)}{tag}")
            if len(diffs) > 1:
                print(f"  {'':<6} {'':<22}           +{len(diffs) - 1} more differing row(s)")
            if first is None:
                first = (g, name, r, ga, gb)

    print()
    print(f"verdict rows: SAME={n_same} DIFF={n_diff} NOT-READ={n_notread} "
          f"(graphs<= {maxg if maxg else 'all'}, names={len(names)})")
    if n_same == 0 and n_diff == 0 and n_notread > 0:
        print("NOTHING WAS COMPARED: every shared row is NOT-READ. Verdict WITHHELD -- a sentinel "
              "compares equal to a sentinel, and reading that as agreement is exactly how a round of "
              "this investigation was once spent on empty rows.")
        return 1
    if first is not None:
        g, name, r, ga, gb = first
        ide = "ids differ" if (ga and gb and ga[0] != gb[0]) else "ids SAME"
        print(f"FIRST DIFF: graph {g}, {name}, row {r} -- {ide}. "
              f"{'The index differs: mapping, not bytes.' if ide == 'ids differ' else 'Same index, different bytes: the residency path delivered different content.'}")
    else:
        print("NO DIFFERENCE FOUND in any compared row.")
        print("  Read that as: the gather's ids AND the bytes behind them are identical on both arms "
              "in the compared graphs, so the divergence is not in what the gather read.")
    print()
    print("CONTROL (mandatory): run the SAME arm twice and diff the two POOL streams. If that reports "
          "anything other than SAME, the readout is the finding -- not the engine. The copy reads a "
          "buffer another kernel may still be writing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
