#!/usr/bin/env python3
"""Compare the `CGC-MMID` host-side prints of two capture logs (one per arm).

WHAT IS COMPARED, AND WHY IT IS THE DECISIVE READOUT
----------------------------------------------------
`CGC_MMID_MV_DBG` (in `ggml_metal_op_mul_mat_id`) prints, per MoE node, the weights tensor's shape,
the ids operands, and an FNV-1a fingerprint of THE FIRST <=4096 BYTES OF EACH OF THE FIRST 4 ROWS THAT
THE IDS SELECT. Its own comment states the decision rule:

    "If the hashes match but the ids differ  -> the remap is wrong.
     If the ids match     but the hashes differ -> the POOL CONTENTS are wrong."

That second branch is the one no other instrument in this repo can reach: everything else establishes
that the ids, the mapping and the pool LAYOUT agree, which leaves only "the bytes at that row differ".
This script reads exactly those bytes.

CAVEAT THAT MUST BE STATED WITH ANY RESULT
------------------------------------------
The print is HOST-SIDE and happens at ENCODE time, so it describes the host's view of the weights at
the moment the command buffer was built -- not necessarily what the device gathered. A DIFF here is
therefore conclusive for "the host sees different bytes"; a SAME here does NOT by itself clear the
device path. (This is the same caveat the repo records for `CGC-MMID-ASSERT`.)

PAIRING
-------
Node names repeat once per pass, so rows are paired by (name, occurrence index within the log). The
first occurrence of a name is pass 1 -- the first pool-path pass, which is the pass the divergence is
born in. `--pass N` keeps only occurrences with that index so a later pass can be looked at too.

USAGE
-----
    python3 Backup/compare_mmid_fp.py <A.log> <B.log> [--pass 0] [--all]
"""
import argparse
import re
import sys
from collections import defaultdict

# name=... type=... | src0 ne=[a,b,c] nb01=.. nb02=.. | ids ne=[n,m] nbi1=.. | dst ne=[..,..]
# | ids=[...] id_min= id_max= id_oob_vs_ne02= | fp(NB/row) id<k>:<hex> ...
RX = re.compile(
    r"CGC-MMID name=(\S+) type=(\S+)"
    r" \| src0 ne=\[(\d+),(\d+),(\d+)\] nb01=(\d+) nb02=(\d+)"
    r" \| ids ne=\[(\d+),(\d+)\] nbi1=(\d+)"
    r" \| dst ne=\[(\d+),(\d+)\]"
    r" \| ids=\[([^\]]*)\] id_min=(-?\d+) id_max=(-?\d+) id_oob_vs_ne02=(\d+)"
    r" \| fp\((\d+)B/row\) (.*)$"
)
RX_FP = re.compile(r"id(-?\d+):([0-9a-f]{16})")


def parse(path):
    """-> {name: [ (ids, src0_ne2, oob, {row_id: fp}) ... ]} in emission order."""
    per = defaultdict(list)
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if not m:
            continue
        ids = tuple(int(x) for x in m.group(13).split(",")) if m.group(13) else ()
        fps = {int(a): b for a, b in RX_FP.findall(m.group(18))}
        per[m.group(1)].append({
            "ids":    ids,
            "ne2":    int(m.group(5)),      # src0 ne02 = how many rows the weights tensor has
            "oob":    int(m.group(16)),
            "fp":     fps,
        })
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pass", dest="pass_idx", type=int, default=0,
                    help="which occurrence of each node name to compare (0 = the first pass)")
    ap.add_argument("--all", action="store_true", help="list every differing node, not the first 12")
    args = ap.parse_args()

    a, b = parse(args.a), parse(args.b)
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    na = sum(len(v) for v in a.values())
    nb = sum(len(v) for v in b.values())
    print(f"CGC-MMID rows  A={na}  B={nb}   distinct node names  A={len(a)}  B={len(b)}")
    if not a or not b:
        print("!! one side printed no CGC-MMID rows -- is CGC_MMID_MV_DBG set on BOTH arms?")
        print("!! (its parser reads only the first character: '1'=150 nodes, '2'=512, anything else=4096)")
        return 2
    print()

    names = [n for n in a if n in b]
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"  names only in A: {only_a[:6]}   only in B: {only_b[:6]}")
        print()

    n_same = n_ids = n_fp = 0
    diffs = []
    for name in names:
        ra, rb = a[name], b[name]
        if args.pass_idx >= len(ra) or args.pass_idx >= len(rb):
            continue
        ra, rb = ra[args.pass_idx], rb[args.pass_idx]
        if ra["ne2"] != rb["ne2"]:
            diffs.append((name, "SRC0_SHAPE", f"ne02 {ra['ne2']} vs {rb['ne2']}", ra, rb))
            continue
        # [CGC 2026-09-17 §EN-17, corrected the same hour] The fingerprint comparison must NOT be
        # short-circuited by an ids mismatch. The first version returned IDS_DIFFER and skipped the
        # comparison entirely, which made `BYTES_DIFFER=0` VACUOUS in exactly the case that matters
        # (all 117 rows were IDS_DIFFER, so no row's bytes were ever compared). The fingerprints are
        # keyed BY ROW ID, so they can be paired on their own: if row k hashes differently on the two
        # arms, the bytes at row k differ -- regardless of which ids either arm asked for.
        shared = sorted(set(ra["fp"]) & set(rb["fp"]))
        bad = [k for k in shared if ra["fp"][k] != rb["fp"][k]]
        ids_same = ra["ids"] == rb["ids"]
        if bad:
            n_fp += 1
            k = bad[0]
            diffs.append((name, "BYTES_DIFFER",
                          f"row {k}: {ra['fp'][k]} vs {rb['fp'][k]}"
                          f"   ({len(bad)}/{len(shared)} shared rows differ;"
                          f" ids {'same' if ids_same else 'differ'})", ra, rb))
        elif not ids_same:
            n_ids += 1
            diffs.append((name, "IDS_ONLY",
                          f"bytes agree on all {len(shared)} shared rows; ids "
                          f"{list(ra['ids'])[:6]} vs {list(rb['ids'])[:6]}", ra, rb))
        else:
            n_same += 1

    print(f"compared nodes: {n_same + n_ids + n_fp}   "
          f"identical={n_same}   IDS_ONLY={n_ids}   BYTES_DIFFER={n_fp}")
    print()
    if not diffs:
        print("=> every compared node has the SAME ids AND the same row fingerprints.")
        print("   The host's view of the expert rows is identical at this pass; the pool CONTENT is not")
        print("   the carrier as far as the host can see. Clear the device path, not this one.")
        return 0
    show = diffs if args.all else diffs[:12]
    print(f"{'node':26s} {'verdict':14s} detail")
    for name, kind, detail, ra, rb in show:
        print(f"{name:26s} {kind:14s} {detail}")
        if kind == "BYTES_DIFFER":
            print(f"{'':26s} {'':14s}   ids A={list(ra['ids'])[:10]}")
            print(f"{'':26s} {'':14s}   ids B={list(rb['ids'])[:10]}")
    if len(diffs) > len(show):
        print(f"  ... {len(diffs) - len(show)} more")
    print()
    print("HOW TO READ IT")
    print("  BYTES_DIFFER -> same row id, different bytes: the pool's CONTENT is the carrier. This is")
    print("                  the branch nothing else in this repo can see, and it does not depend on the")
    print("                  ids lists agreeing.")
    print("  IDS_ONLY     -> bytes agree on every shared row, but the two arms asked for different ids.")
    print("                  The mapping differs while the content does not.")
    print("  (both are host-side at ENCODE time -- a DIFF is conclusive, a SAME clears only the host.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
