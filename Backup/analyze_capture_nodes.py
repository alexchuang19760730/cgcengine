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

## --upto was a BLIND SPOT, not a fix. Read this before using it.

`--upto` exists because the IDS destination is capped at 4096 slots and the LAST chunks merge several
forward passes, so pairing by name inside them compares rows from different passes. Keeping only the
head of the sequence avoids that -- but on 2026-09-16 23:36 it turned out to have silently excluded
EVERY decode step. Measured on a 36-graph capture of this run shape:

    graphs  1- 2  T=2        graphs 28     T=2
    graphs  3-27  T=8        graphs 29-30  T=4
    graph  25     T=6        graphs 31-35  T=1  <- the only decode graphs

`--upto 24` keeps graphs 1..23, i.e. 100% PREFILL. Every localisation this file produced before that
date (r1/r2/r3/r3b/r3c/r3d/r3e) describes prefill, while the thing under investigation is decode --
and on decode the picture is the opposite: EVERY node diverges, starting from layer 0's output.

So the head of the sequence is not a safe default, it is a specific stage. Measure the stage instead:
`--decode-only` selects the T=1 graphs by DERIVING T from the `ne` field, with no hand-entered number.
T = min(ne in a graph) / min(ne over the whole run), because every captured output's element count
scales with the token count of the pass, and the minimum over the run IS the single-token step.

A flag that avoids an artefact by dropping part of the sequence must always be asked: which stage am
I left with? Here the answer was "the wrong one", for seven rounds.

Usage:
    analyze_capture_nodes.py LOG_A LOG_B [--graphs N] [--only PREFIX]
    analyze_capture_nodes.py LOG_A LOG_B --decode-only          # the T=1 graphs
"""
import argparse
import re
import sys
from pathlib import Path

# `ne` is the element count BEFORE the CGC_TENSOR_CAPTURE_WORDS clamp, so it still carries the SHAPE
# of the tensor. n_ids alone cannot: for a tensor with more elements than the word budget, n_ids is
# always saturated at the budget, and the shape -- the only thing that tells a prefill chunk from a
# decode step -- is exactly what the clamp ate.
RX = re.compile(
    r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\](?: fuse=(\d+) ne=(\d+))?(?: off=(\d+))?"
)
GRAPH_START = "ffn_moe_gate-1"

# The layer chain, taken from the graph builder (qwen35moe.cpp + delta-net-base.cpp + llama-graph.cpp),
# NOT from the emission order. Emission order is NOT usable as the localisation key: measured
# 2026-09-16, the same arm captured twice emits the SAME nodes in a DIFFERENT order (e.g.
# `linear_attn_out-2` before and after `attn_residual-2` in different graphs and different runs), while
# every value is reproducible. So the schedule this backend produces is not order-stable, and anything
# that reads "which row came first" as "which node computed first" is reading noise.
#
# [CGC 2026-09-17 00:01 r7] ORDER IS NOW CAUSAL (layer 0 -> 1 -> 2), and that is a deliberate change
# of convention. Previously the list followed the order nodes occur INSIDE one segment, and a segment
# straddles two passes -- the graph boundary is `ffn_moe_gate-1`, so a segment holds {pass P: layer 1's
# MoE onward, all of layer 2} followed by {pass P+1: layer 0, and layer 1 up to its MoE gate}. That
# made `pos` a statement about two different passes and made "earlier in the chain" hard to read. The
# causal order is the one the exclusion argument needs: within one pass, layer 0 runs before layer 1
# before layer 2, so "layer 0 identical, layer 1 not" localises the introduction to layer 1. The
# straddle has NOT gone away -- when comparing, remember `l_out-0` in segment i belongs to a LATER
# pass than the layer-2 rows beside it -- it is only the ordering here that is now per-pass.
#
# Layer 1 is in the list because of the r6 whole-tensor digest: layer 0's twelve nodes are identical
# through graph 30 while layer 1's output and all of layer 2 diverge from graph 1. The instrumented
# set had never included layer 1 -- rounds r1..r4 called it identical from a token-0 window.
CHAIN = [
    "attn_norm-0.dst",                                                         # layer 0
    "z-0.dst", "gate-0.dst",
    "conv_input-0.dst", "conv_output_raw-0.dst", "linear_attn_out-0.dst",
    "attn_residual-0.dst", "attn_post_norm-0.dst",
    "ffn_moe_logits_raw-0.dst",                                                # router logits, PRE top-k
    "ffn_moe_weights_norm-0.dst",                                              # post top-k weights
    "ffn_moe_gate-0.dst", "ffn_moe_up-0.dst", "ffn_moe_down-0.dst", "ffn_moe_out-0.dst",
    "l_out-0.dst",
    "attn_norm-1.dst",                                                         # layer 1
    "z-1.dst", "gate-1.dst",
    "conv_input-1.dst", "conv_output_raw-1.dst", "linear_attn_out-1.dst",
    "attn_residual-1.dst", "attn_post_norm-1.dst",
    "ffn_moe_logits_raw-1.dst",
    "ffn_moe_weights_norm-1.dst",
    "ffn_moe_gate-1.dst", "ffn_moe_up-1.dst", "ffn_moe_down-1.dst", "ffn_moe_out-1.dst",
    "l_out-1.dst",
    "attn_norm-2.dst",                                                         # layer 2
    "z-2.dst", "gate-2.dst",
    "conv_input-2.dst", "conv_output_raw-2.dst", "linear_attn_out-2.dst",
    "attn_residual-2.dst", "attn_post_norm-2.dst",
    "ffn_moe_logits_raw-2.dst",
    "ffn_moe_weights_norm-2.dst",
    "ffn_moe_gate-2.dst", "ffn_moe_up-2.dst", "ffn_moe_down-2.dst", "ffn_moe_out-2.dst",
    "l_out-2.dst",
]


def chain_rank(name):
    try:
        return CHAIN.index(name)
    except ValueError:
        return len(CHAIN) + 1


def parse(path):
    """-> list of graphs; each graph is a list of (name, path, vals, fuse, ne, off)."""
    rows = []
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if m:
            vals = tuple(int(x) for x in m.group(5).split(",")) if m.group(5) else ()
            rows.append((m.group(4), m.group(2), vals, int(m.group(6) or 0),
                         int(m.group(7) or 0), int(m.group(8) or 0)))
    graphs, cur = [], []
    for row in rows:
        if row[0] == GRAPH_START and cur:
            graphs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        graphs.append(cur)
    return graphs


def split(g):
    """-> (ordered list of (name, vals) for DST rows, {name: vals} for ids rows)."""
    dst, ids = [], {}
    for name, p, vals, _fuse, _ne, _off in g:
        if p == "DST":
            dst.append((name, vals))
        else:
            ids.setdefault(name, vals)
    return dst, ids


def t_of(graphs):
    """-> ({graph index: T}, min nelements over the run).

    T is DERIVED, never entered by hand: each captured output's element count scales with the number
    of tokens in the pass, so the smallest `ne` inside a graph divided by the smallest `ne` seen
    anywhere in the run is that graph's token count. The minimum over the run is the single-token
    decode step by construction -- if the run contains no decode step, T has no 1 and --decode-only
    reports an empty selection rather than silently comparing prefill.

    Per-graph MINIMUM rather than one nominated node, so the reading survives a node being absent.
    """
    per, all_min = {}, []
    for i, g in enumerate(graphs):
        nes = [ne for _n, _p, _v, _f, ne, _o in g if ne]
        if nes:
            per[i] = min(nes)
            all_min.append(min(nes))
    if not all_min:
        return {}, 0
    gmin = min(all_min)
    return {i: v // gmin for i, v in per.items()}, gmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--graphs", type=int, default=6, help="how many graphs to tabulate")
    ap.add_argument("--only", default="", help="restrict the table to names with this prefix")
    ap.add_argument("--upto", type=int, default=0,
                    help="consider only graphs [0, --upto). READ THE MODULE DOCSTRING FIRST: on the "
                         "2026-09-16 run shape this keeps PREFILL ONLY -- graphs 1..23 are all T>=2 "
                         "and the 5 decode graphs are 31..35. --upto 0 (the default) means all.")
    ap.add_argument("--decode-only", action="store_true",
                    help="compare ONLY the T=1 graphs, with T derived from `ne` (see the docstring). "
                         "This is the selector to use when the question is about decode.")
    ap.add_argument("--select-t", type=int, default=0, metavar="N",
                    help="compare ONLY the FIRST stage whose derived T == N. Structural (T comes from "
                         "`ne`), which is the point: the question 'which pass' must not be answered by "
                         "a hand-entered stage number. On the current run shape --select-t 2 picks the "
                         "first 2-token POOL-PATH pass, i.e. prompt processing, NOT decode.")
    ap.add_argument("--list-stages", action="store_true",
                    help="print the derived T for every stage and stop (no comparison). Use it to "
                         "confirm a selector picked the stage you think it did.")
    ap.add_argument("--windows", action="store_true",
                    help="print, per node, the element range the 32-word window actually read "
                         "(ne/off). Run this FIRST: a name can report SAME simply because its window "
                         "covers a different token than the node it is being compared against.")
    args = ap.parse_args()

    a, b = parse(args.a), parse(args.b)
    n = min(len(a), len(b))
    if args.upto:
        n = min(n, args.upto)

    sel = list(range(n))
    ta, tb = {}, {}
    if args.decode_only or args.select_t or args.list_stages:
        ta, gmin_a = t_of(a)
        tb, _gmin_b = t_of(b)
        if not ta or not tb:
            print("!! this selector needs the `ne=` field. It is only in captures taken after")
            print("!! 2026-09-16 23:36; older logs carry n_ids (clamped) and cannot report the shape.")
            return 2

    if args.list_stages:
        print(f"=== stage table (T = min ne in graph / {t_of(a)[1]} ne) ===")
        print(f"{'stage':>6} {'A T':>5} {'B T':>5} {'A rows':>7} {'B rows':>7}")
        for i in range(n):
            xa = sum(1 for _nm, p, _v, _f, _ne, _o in a[i] if p == "DST")
            xb = sum(1 for _nm, p, _v, _f, _ne, _o in b[i] if p == "DST")
            print(f"{i:>6} {ta.get(i, '?'):>5} {tb.get(i, '?'):>5} {xa:>7} {xb:>7}")
        print()
        return 0

    if args.decode_only:
        sel = [i for i in range(n) if ta.get(i) == 1 and tb.get(i) == 1]
        if not sel:
            print(f"!! no graph has T=1 in BOTH arms -- n_predict may be too small to reach a decode")
            print(f"!! step, or the arms disagree on the stage. A T: {sorted(set(ta.values()))}")
            print(f"!! B T: {sorted(set(tb.values()))}")
            return 2

    if args.select_t:
        # [CGC 2026-09-17 §EN-16] The FIRST COMPLETE stage with T == N, and only that one.
        #
        # "First" is part of the claim: the question is where the divergence comes from, so the
        # earliest pass that runs the pool path is the one that matters, and picking a later same-T
        # stage would answer a different question. Both arms must agree on the T, otherwise the
        # pairing is meaningless.
        #
        # "Complete" is the correction that came out of doing it once: the capture starts mid-pass,
        # so stage 0 can be a FRAGMENT (measured: 1 DST row where every real pass has 4). A fragment
        # has an incomplete node set, and `present=1` there would read like a node that is absent in
        # the other arm. So the first stage with the MODE row count among T==N stages is taken.
        match = [i for i in range(n) if ta.get(i) == args.select_t and tb.get(i) == args.select_t]
        if not match:
            print(f"!! no stage has T={args.select_t} in BOTH arms.")
            print(f"!! A T: {sorted(set(ta.values()))}   B T: {sorted(set(tb.values()))}")
            return 2

        def rows_of(i):
            return sum(1 for _nm, p, _v, _f, _ne, _o in a[i] if p == "DST")

        full = max(rows_of(i) for i in match)
        complete = [i for i in match if rows_of(i) == full]
        sel = [complete[0]]
        skipped = [i for i in match if i < sel[0]]
        print(f"--select-t {args.select_t}: chose stage {sel[0]} ({rows_of(sel[0])} DST rows). "
              f"{len(match)} stage(s) have T={args.select_t} in both arms; skipped "
              f"{len(skipped)} fragment(s) before it {skipped[:4]}; later ones: {match[len(skipped)+1:][:6]}")
        print()

    print(f"A graphs={len(a)}  B graphs={len(b)}  comparing {len(sel)}"
          + (f"  (--upto {args.upto})" if args.upto else "")
          + ("  [--decode-only]" if args.decode_only else "")
          + (f"  [--select-t {args.select_t}]" if args.select_t else ""))
    if len(a) != len(b):
        print("  !! graph counts differ -- compare only the common prefix, and say so")

    # [CGC 2026-09-16] READ THIS BEFORE BELIEVING ANYTHING BELOW. The pairing here is by GRAPH INDEX,
    # so it is only valid if both logs emitted the same number of DST rows. Fusion state is NOT stable
    # across runs: a node absorbed as the last member of a fused group emits no row of its own, and
    # which node that is depends on node order (see the ABSENT note below). Measured: two runs of the
    # SAME arm produced 492 vs 491 DST rows, with `z-2` present 41 vs 40 times -- ONE row short, which
    # shifts every later graph by one and quietly pairs "run1 graph 2" against "run2 graph 3". That is
    # precisely the clean-looking false DIFF this instrument reported on `gate-2` inside its own
    # same-arm control. So the two counts are printed FIRST.
    ra = sum(1 for l in open(args.a, errors="replace") if "CGC-IDS-CAP" in l and ".dst" in l)
    rb = sum(1 for l in open(args.b, errors="replace") if "CGC-IDS-CAP" in l and ".dst" in l)
    print(f"DST rows  A={ra}  B={rb}" + ("" if ra == rb else "   <<< differ"))
    if ra != rb:
        # [CGC 2026-09-16, corrected the same day it was written] A difference in the TOTALS is not by
        # itself fatal, and r3c proved it: gputime 533 vs slotgpu 492, yet both nodes under test were
        # present exactly once per graph in BOTH logs. A uniform per-graph difference means some OTHER
        # node got absorbed into a fused group (it emits no row of its own) -- local and harmless,
        # because pairing is by (graph, name), not by row index. What IS fatal is a missing GRAPH
        # MARKER, which merges two passes into one and shifts every later graph. So the check that
        # matters is PER GRAPH, and that is what follows -- not a verdict on the totals.
        print("  note: totals differ; the per-graph check below is the one that decides.")
    bad = []
    for i in sel:
        x = sum(1 for _nm, p, _v, _f, _ne, _o in a[i] if p == "DST")
        y = sum(1 for _nm, p, _v, _f, _ne, _o in b[i] if p == "DST")
        if x != y:
            bad.append((i, x, y))
    if bad:
        print(f"  graphs whose DST row counts disagree: {bad[:10]}" + (" ..." if len(bad) > 10 else ""))
        print("  !! On THOSE graphs a name may be paired against a row from a different pass. Prefer the")
        print("  !! graphs NOT in that list, and confirm any node you rely on is present exactly once per")
        print("  !! graph in both logs -- that is a stronger guarantee than an equal total.")
    else:
        print(f"  per graph: all {len(sel)} compared graphs have equal DST row counts -- pairing is sound")
    print()

    if args.decode_only or args.select_t:
        print(f"=== stage (T = min ne in graph / {t_of(a)[1]} ne) -- this is what is being compared ===")
        print(f"{'graph':>6} {'A T':>5} {'B T':>5}")
        for i in sel:
            print(f"{i:>6} {ta.get(i, '?'):>5} {tb.get(i, '?'):>5}")
        print(f"  (full A sequence: " + ", ".join(f"g{i}:T{ta[i]}" for i in sorted(ta)) + ")")
        print()

    if args.windows:
        i = sel[0]
        print("=== window: which elements of the tensor the 32 words actually came from ===")
        print(f"  graph {i}. off=0 = the HEAD, off=ne-n = the TAIL. A ggml tensor is ne0-FASTEST, so a")
        print(f"  head window on a (ne0>32, T) output covers i1=0 ONLY -- token 0 of the pass -- whereas on")
        print(f"  `conv_input` (ne0 = K-1+T, always < 32) it covers EVERY token. Rows with different `ne`")
        print(f"  are NOT the same measurement and must not be read as SAME/DIFF against each other.")
        print(f"  {'node':36s} {'ne':>9} {'off':>9} {'n':>4}  {'window':>18}")
        seen = {}
        for name, p, vals, _f, ne, off in a[i]:
            if p == "DST":
                seen.setdefault(name, (ne, off, len(vals)))
        for name in sorted(seen, key=chain_rank):
            ne, off, n = seen[name]
            if off < 0:
                # off = -1 is the marker the instrument writes for a whole-tensor digest row.
                print(f"  {name:36s} {ne:>9} {'DIGEST':>9} {n:>4}  {'WHOLE TENSOR':>18}  "
                      f"<- [sum,xor,wsum,ne] of all {ne} elements, not a window")
                continue
            span = f"[{off}..{off + n})"
            note = ""
            if ne and ne > n:
                note = "  <- partial" if off == 0 else "  <- partial (tail)"
            print(f"  {name:36s} {ne:>9} {off:>9} {n:>4}  {span:>18}{note}")
        print()

    # ---- ids status per graph (the 09-15 instrumentation, re-derived in this very run) ----
    print("=== ids (what mul_mat_id consumed) ===")
    print(f"{'graph':>6} {'ids SAME':>9} {'ids DIFF':>9} {'DST SAME':>9} {'DST DIFF':>9}")
    for i in sel:
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
    # Ordered by the SOURCE CHAIN (see CHAIN above), not by emission order. `order` is SEEDED with
    # CHAIN and then extended with anything else that was captured, so a node that was REQUESTED and
    # never appeared still gets a row. It previously listed only names that were present somewhere,
    # which made the most dangerous failure invisible: measured 2026-09-16, a 23-name request overran
    # a 256-char filter and four names were dropped, and the report simply did not mention them --
    # a name that is absent from the table looks like a name nobody asked for.
    order = list(CHAIN)
    for i in sel:
        for name, _v in split(a[i])[0]:
            if name not in order:
                order.append(name)
    order.sort(key=chain_rank)

    print("=== localisation: first compared graph where each DST node DIVERGES, in the SOURCE chain ===")
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
        for i in sel:
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

    # Names in the CHAIN that appear in NEITHER log. Printed as a block of its own, because an absent
    # row in the table above is easy to skim past and reads like "nobody asked for this". NOTE: the
    # analyzer cannot tell "the run did not request it" from "the run requested it and the filter ate
    # it" -- the run's node list is in the log header, not parsed here -- so the cases are listed
    # together rather than guessed apart.
    never = []
    for n in CHAIN:
        if not any(n in dict(split(a[i])[0]) or n in dict(split(b[i])[0]) for i in sel):
            never.append(n)
    if never:
        print(f"  !! in CHAIN but NEVER captured in either log ({len(never)}): {', '.join(never)}")
        print(f"  !! None of these is 'identical'. Four causes, all of them silent:")
        print(f"  !!  1. the capture FILTER was truncated: CGC_TENSOR_CAPTURE is copied into a fixed")
        print(f"  !!     buffer at init, so a long list loses its tail, cut MID-NAME. Check the run log")
        print(f"  !!     for 'CGC_TENSOR_CAPTURE effective'; raise CGC_DST_FILTER_MAX or shorten the list.")
        print(f"  !!     Measured 2026-09-16: a 23-name list (312 chars) kept 19 and dropped 4.")
        print(f"  !!  2. the run never asked for it (not in that run's NODES).")
        print(f"  !!  3. the name belongs to a VIEW: cb() naming a reshape/transpose/view gives a node")
        print(f"  !!     with no kernel of its own, so no dispatcher ever sees that name.")
        print(f"  !!  4. the producing op is not hooked in this build.")
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
        print("=> every captured node agreed in every compared graph: this list exonerates all of them")
    print()

    # ---- per-graph detail for the first few, one line per captured node ----
    for i in sel[:max(0, args.graphs)]:
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
        stage = f"  (T={ta.get(i, '?')})" if args.decode_only else ""
        print(f"-- graph {i}{stage}: " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
