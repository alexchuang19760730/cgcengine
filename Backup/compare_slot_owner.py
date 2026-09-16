#!/usr/bin/env python3
"""Compare the `CGC-S1: SLOT-OWNER` / `SLOT-SEL` readouts of two capture logs (one per arm).

WHAT EACH READOUT IS
--------------------
`slot_owner[layer][slot]` is the pool's REVERSE map (slot -> expert). The forward map
(expert -> slot) has already been cleared by other instruments: EQUIV-pool shows the published table
equals the live pool at publish time, and SEL-DRIFT shows the mapping of the consumed experts does
not move afterwards. Neither says anything about the pool's CONTENTS. These two readouts are the
readout for that.

  SLOT-OWNER  digests the WHOLE slot vector of a layer. Answers "do the two pools differ at all".
  SLOT-SEL    digests only the slots the consumer reads THIS step, mapped back to expert identity
              (`slot_owner[il][table[ids[j]]]`). Answers "do the two arms FETCH different experts".
              It also counts `wrong` (reverse map disagrees with the forward map for a consumed
              expert => the gather reads another expert's weights) and `unowned` (consumed expert
              resolves to a slot the pool does not own).

READ THE SAME-ARM PAIR FIRST
----------------------------
These reads are unlocked in the hook, so a background fill can tear a digest. That makes the
same-arm control a PREMISE, not a formality: run one arm twice and require the pair to be SAME
before believing any cross-arm DIFF. Same rule as the kernel-side tensor capture.

VERDICT VOCABULARY (this is where a digest can be mis-read)
----------------------------------------------------------
  SAME       byte-identical digest.
  PERMUTED   `sum` and `xor` agree (same MULTISET of owners) but `wsum` does not => the same experts
             are resident / fetched, in a different order or slot placement. Placement, not content.
  DIFFERENT  `sum` or `xor` disagrees => a different multiset. For SLOT-SEL this means the two arms
             fetch DIFFERENT experts' weights, which is the whole question.
  SHAPE      `n` disagrees => not comparable cell by cell.

`wsum` is what makes PERMUTED separable from DIFFERENT: `sum` and `xor` are invariant under a
permutation, `sum(v*(i+1))` is not.

USAGE
-----
    python3 Backup/compare_slot_owner.py <A.log> <B.log> [--graphs N] [--all]

Exit status: 0 = agree, 1 = any digest differs, 2 = the pair cannot be compared (no lines on one
side, or an unequal cell set -- a positional pairing would then be invalid).
"""
import argparse
import re
import sys

RX_OWNER = re.compile(
    r"CGC-S1: SLOT-OWNER graph=(-?\d+) il=(-?\d+) sum=(-?\d+) xor=(-?\d+) wsum=(-?\d+)"
    r" n=(-?\d+) owned=(-?\d+)"
)
RX_SEL = re.compile(
    r"CGC-S1: SLOT-SEL graph=(-?\d+) il=(-?\d+) ntok=(-?\d+) sum=(-?\d+) xor=(-?\d+) wsum=(-?\d+)"
    r" n=(-?\d+) wrong=(-?\d+) unowned=(-?\d+)  idsum=(-?\d+) idsxor=(-?\d+) idwsum=(-?\d+)"
)

# (label, regex, index of the first integer field to keep, note, field names)
SECTIONS = (
    ("SLOT-OWNER", RX_OWNER, 3, "the whole slot vector of a layer: does the pool differ at all",
     "sum, xor, wsum, n, owned"),
    ("SLOT-SEL", RX_SEL, 4, "the slots the consumer reads, mapped back to expert identity",
     "sum, xor, wsum, n, wrong, unowned, idsum, idsxor, idwsum"),
)


def parse(path, rx, start=3):
    """-> {(graph, il): (ints...)}; duplicates overwrite and are counted."""
    rows, dup = {}, 0
    for line in open(path, errors="replace"):
        m = rx.search(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key in rows:
                dup += 1
            rows[key] = tuple(int(m.group(i)) for i in range(start, rx.groups + 1))
    return rows, dup


def classify(a, b, i_n):
    """i_n = index of the `n` field inside the tuple."""
    if a == b:
        return "SAME"
    if a[i_n] != b[i_n]:
        return "SHAPE"
    if a[0] == b[0] and a[1] == b[1]:
        return "PERMUTED"       # same multiset, different placement
    return "DIFFERENT"


def report(label, note, fields, ra, rb, da, db, args, i_n=3, extra=None):
    """Returns (comparable, n_differing)."""
    print("=" * 96)
    print(f"{label}   ({note})")
    print(f"  fields: {fields}")
    print("=" * 96)
    print(f"lines  A={len(ra)}  B={len(rb)}" + (f"   (duplicates A={da} B={db})" if (da or db) else ""))
    if not ra or not rb:
        print("  !! one side printed nothing for this readout.")
        print("  !! both arms must carry CGC_S1_TABLE_CHURN=1 -- a dropped or unlisted CGC_* is silent,")
        print("  !! so 'no effect' and 'never set' look identical.")
        return False, 0
    keys = sorted(set(ra) & set(rb))
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))
    verdicts = {k: classify(ra[k], rb[k], i_n) for k in keys}
    n_same = sum(1 for v in verdicts.values() if v == "SAME")
    counts = {v: sum(1 for x in verdicts.values() if x == v) for v in ("PERMUTED", "DIFFERENT", "SHAPE")}
    print(f"cells  compared={len(keys)}  SAME={n_same}  PERMUTED={counts['PERMUTED']}"
          f"  DIFFERENT={counts['DIFFERENT']}  SHAPE={counts['SHAPE']}")
    if only_a:
        print(f"       present only in A: {len(only_a)}  first={only_a[:4]}")
    if only_b:
        print(f"       present only in B: {len(only_b)}  first={only_b[:4]}")

    if extra is not None:
        print(extra(ra, rb))

    if n_same == len(keys) and not only_a and not only_b:
        print(f"  => SAME on every cell. If these are two DIFFERENT arms, this quantity is not the")
        print(f"     carrier. If they are the SAME arm twice, this is the control and it passing is")
        print(f"     the premise for believing any cross-arm DIFF below.")
        print()
        return True, 0

    per_graph = {}
    for (g, il), v in verdicts.items():
        if v != "SAME":
            per_graph.setdefault(g, []).append((il, v))
    cap = args.graphs if args.graphs else 40
    print(f"  {'graph':>6} {'n_diff':>7} {'permuted':>9} {'first_il':>9}   verdict")
    shown = 0
    for g in sorted(per_graph):
        if shown >= cap:
            print(f"    ... {len(per_graph) - shown} more graphs with differences")
            break
        rows = sorted(per_graph[g])
        n_perm = sum(1 for _il, v in rows if v == "PERMUTED")
        kinds = {v for _il, v in rows}
        tag = kinds.pop() if len(kinds) == 1 else "mixed"
        print(f"  {g:>6} {len(rows):>7} {n_perm:>9} {rows[0][0]:>9}   {tag}")
        show = rows if args.all else rows[:3]
        for il, v in show:
            print(f"          il={il:<4} {v:<10} A={ra[(g, il)]}  B={rb[(g, il)]}")
        shown += 1
    g0 = min(per_graph)
    first = sorted(per_graph[g0])[0]
    print(f"  => FIRST divergence: graph {g0}, il={first[0]}  ({first[1]})")
    print(f"     A={ra[(g0, first[0])]}")
    print(f"     B={rb[(g0, first[0])]}")
    print()
    return True, len(per_graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--graphs", type=int, default=0,
                    help="print per-graph rows for at most this many graphs (0 = up to 40)")
    ap.add_argument("--all", action="store_true",
                    help="list every differing (graph, il) instead of the first three per graph")
    args = ap.parse_args()

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print()

    any_diff = 0
    comparable = True
    ra, da = parse(args.a, RX_OWNER)
    rb, db = parse(args.b, RX_OWNER)

    def owner_extra(x, y):
        return ""

    ok, nd = report("SLOT-OWNER", SECTIONS[0][3], SECTIONS[0][4], ra, rb, da, db, args,
                    i_n=3, extra=owner_extra)
    comparable &= ok
    any_diff += nd

    sa, ds = parse(args.a, RX_SEL, 4)
    sb, dbb = parse(args.b, RX_SEL, 4)

    def sel_extra(x, y):
        # `wrong` is an ABSOLUTE gate, not a comparison: a consumed expert whose reverse-map owner is
        # a different expert means the gather reads the wrong weights, and that is a defect on its own
        # regardless of what the other arm says.
        wa = sum(v[4] for v in x.values())
        wb = sum(v[4] for v in y.values())
        ua = sum(v[5] for v in x.values())
        ub = sum(v[5] for v in y.values())
        # DEGENERACY CONTROL. Whenever wrong == 0 and unowned == 0, `owner == ids[j]` holds
        # elementwise, so the owner digest IS the digest of the top-k ids and this readout says
        # nothing about the pool. Counting the cells where that happens decides it instead of arguing.
        deg_a = sum(1 for v in x.values() if v[0] == v[6] and v[1] == v[7] and v[2] == v[8])
        deg_b = sum(1 for v in y.values() if v[0] == v[6] and v[1] == v[7] and v[2] == v[8])
        out = [f"       absolute gate  wrong (must be 0):   A={wa}  B={wb}"
               f"   -> a consumed expert whose slot holds ANOTHER expert's weights",
               f"       informational  unowned:            A={ua}  B={ub}"
               f"   -> consumed expert -> slot the pool does not own",
               f"       degeneracy     owner==ids:         A={deg_a}/{len(x)}  B={deg_b}/{len(y)}"
               f"   (all cells => this readout is the ids digest, i.e. ROUTING, not pool contents)"]
        if wa or wb:
            out.append("       !! wrong != 0 -> the gather reads another expert's weights. Silent by")
            out.append("       !! construction: the slot index is legal, so nothing asserts.")
        if ua or ub:
            out.append("       NOTE unowned != 0: the hook samples AFTER the fill, so on the pool path this")
            out.append("       NOTE should be 0. A non-zero value here is a finding, not the cold rate.")
        if deg_a == len(x) and deg_b == len(y) and x and y:
            out.append("       => DEGENERATE: every cell has owner == ids. Any difference below is a")
            out.append("       => difference in the top-k ROUTING. Quote it as such -- do not read it as")
            out.append("       => 'the pool hands the consumer a different expert'.")
        return "\n".join(out)

    ok, nd = report("SLOT-SEL", SECTIONS[1][3], SECTIONS[1][4], sa, sb, ds, dbb, args,
                    i_n=3, extra=sel_extra)
    comparable &= ok
    any_diff += nd

    print("=" * 96)
    if not comparable:
        return 2
    if any_diff == 0:
        print("RESULT: both readouts agree on both sides.")
    else:
        print(f"RESULT: differences in {any_diff} graph(s). Read SLOT-SEL first: it is the one that")
        print("        speaks about what the consumer FETCHES; SLOT-OWNER only says the pools differ.")
    return 1 if any_diff else 0


if __name__ == "__main__":
    sys.exit(main())
