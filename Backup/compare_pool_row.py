#!/usr/bin/env python3
"""§9.18.6 r12 + §9.18.7: do the two arms read the SAME BYTES for the SAME slot -- and if not, WHY?

This reads the `path=POOL` rows (`Backup/run_ids_dst_capture.sh` with `POOL=1`), which carry, per
selected row: the id the consumer used, a device-side digest of the row's first `probe` bytes, and --
since §9.18.7 -- the EXPERT that slot holds on that arm (`own=`, snapshotted from the host
slot->expert map at the moment the consuming node was ENCODED, not at dump time).

THE DECISION RULE, and why the `own=` column is what turns this from a diff into a decision:

    id same  owner same  digest same    -> SAME. Same slot, same expert, same weights.
    id same  owner same  digest differs -> *** CONTENT ***  Same slot, same expert, different bytes.
                                           Neither index nor mapping can explain it. This is the
                                           fill/residency defect -- the last thing §9.18.4 could
                                           still have stood on, and what §9.18.7 was built to test.
    id same  owner differs              -> SLOT-REUSED. One slot index holds a different expert on
                                           each arm, so "same ids" is a coincidence, not agreement.
    id differs owner same  digest same  -> RELAYOUT. Different slots, same expert, same bytes. Benign
                                           by construction, and the STRONGEST form of "the pool is
                                           not the carrier": the weights behind a named expert match.
    id differs owner same  digest differs -> *** CONTENT *** too. The same expert's weights read
                                           byte-differently out of two different slots.
    id differs owner differs            -> ROUTING. The arms fetch different experts; the divergence
                                           is then UPSTREAM of the gather, in routing or in mapping.
    owner unknown on either side        -> UNCLASSIFIED, and the aggregate reports how many.

Without `own=`, four of those seven rows collapse into two observable outcomes, and SLOT-REUSED,
ROUTING and RELAYOUT all print as "ids differ, digests differ" -- three different next steps wearing
one word. That collapse is the whole reason §9.18.7 exists.

TWO WAYS THIS INSTRUMENT CAN LIE, both guarded here because both have already happened in this repo:
  1. rows=0 or every group n=0. A slot that was never written is filled with the 0x7fffffff sentinel,
     and a sentinel compares EQUAL to a sentinel. That is how a whole round was once read off
     all-sentinel rows (2026-09-17 00:30, the TAIL window bug). Such a row is reported as NOT-READ,
     never as SAME; at row level an unread side is UNREAD and is never counted as agreement.
  2. Nothing shared. If no (graph, name) pair exists on both sides -- e.g. the node name differs
     between arms, or the runs captured different layers -- then "no differences" means nothing was
     compared. Reported as NO-OVERLAP and the verdict is withheld.
A THIRD, new in §9.18.7: `own=none` on EVERY row means the owner channel never reached ggml-metal
(llama only registers it when CGC_POOL_CAPTURE is set on the SERVER side, not just on the sweep side).
The run is still a valid ids+bytes comparison, but every row is UNCLASSIFIED. That is printed loudly,
because an absent instrument and an instrument that found nothing look identical unless you say so --
the failure mode of §9.18.5, where a host-side probe was blind on the one arm it was built for.

SAME and RELAYOUT are strong evidence, not proof: the digest is (sum, xor, weighted-sum, nbytes) over
the row's bytes, so a compensating multiset change can collide. It is sensitive to a permuted row as
well as to a changed one.
"""
import argparse
import collections
import re
import sys
from pathlib import Path

# `path=(\w+)` deliberately matches only the first word: POOL rows carry `from=MV|MM` after the
# geometry fields, and the ids/dst rows carry `fuse=`/`ne=`/`off=`. One regex, three row shapes, and a
# row whose fields do not match at all is NOT silently dropped -- see the unmatched counter below.
RX_POOL = re.compile(
    r"CGC-IDS-CAP slot=(\d+) path=(\w+) n_ids=(\d+) name=(\S+) ids=\[([^\]]*)\]"
    r"(?: rows=(\d+) probe=(\d+) nsel=(\d+) from=(\w+))?"
    r"(?: own=(\S+))?"
    r"(?: exp=(\S+))?")
# Graph boundaries live in the IDS rows (see the note in ggml_metal_ops.cpp's dump): every stream is
# merged in submission order, so the marker is seen in order by this reader too.
GRAPH_START = "ffn_moe_gate-1"

SENTINEL = 0x7fffffff  # as int32


def as_i32(v):
    return v - (1 << 32) if v > 0x7FFFFFFF else v


def parse_own(tok):
    """`own=[1,2,-1]` -> [1, 2, -1]; `own=none` or absent -> None (NOT an empty list).

    Used for `exp=` too -- the two fields are both bracketed int lists.

    The distinction is the point: None is "this row was not annotated", [] would read as "no experts
    were involved". Returning [] for `none` would make an absent instrument look like a measurement.

    Both `[...]` and a bare `1,2,3` are accepted. The bare form is what the first build printed, and
    accepting it means logs captured before the delimiter was fixed still compare -- but note WHY the
    fix was needed: the reader had been written against the documented bracketed form, so a bare list
    parsed as "no owner" and the tool reported OWNER CHANNEL ABSENT for a run whose own banner said the
    channel was ACTIVE. Being liberal here is a compatibility convenience, not a licence to print
    either form; the dumper prints brackets.
    """
    if tok is None or tok == "none":
        return None
    t = tok.strip()
    if t.startswith("["):
        if not t.endswith("]"):
            return None          # a truncated list is not a list
        t = t[1:-1]
    if not t:
        return []
    try:
        return [as_i32(int(x)) for x in t.split(",")]
    except ValueError:
        return None              # unparsable => "not annotated", never a silent empty


def parse(path):
    """-> (rows, n_pool, n_other, n_unmatched); a row is

    (graph, name, n_groups, groups, rows, probe, nsel, origin, owns, exps)
    with groups = [(id, sum, xor, wsum, nbytes)], owns = [expert] | None (the expert the slot holds)
    and exps = [slot] | None (the slot the HOST expected at that position). Both annotations are
    optional and independent; None means "not annotated", never "empty".
    """
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
        _slot, pth, _n_ids, name, ids, rows, probe, nsel, origin, own_tok, exp_tok = m.groups()
        if pth != "POOL":
            n_other += 1
            # the graph marker is a ffn_moe_gate-1 IDS row: `name=ffn_moe_gate-1` with no suffix
            if name == GRAPH_START:
                graph += 1
            continue
        n_pool += 1
        owns = parse_own(own_tok)
        exps = parse_own(exp_tok)
        if rows is None:
            # a POOL row with no geometry is a parser/tooling mismatch, not a data point
            out.append((graph, name, -1, [], 0, 0, 0, "?", owns, exps))
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
        out.append((graph, name, n_groups, groups, rows, int(probe), int(nsel), origin, owns, exps))
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


def map_verdict(id_dev, exp):
    """One row of one arm: did the device use the slot the HOST expected at that position?

    OK     : the device read the slot this side's own table points at.
    LOST   : the device read a DIFFERENT slot -- the table the device read is not the one the host
             published, which is a machine defect and is invisible in every cross-arm comparison
             (a wrong-but-legal index produces a plausible expert, and `own` only says which one).
    NO-EXP : that position was not annotated, so nothing is claimed.
    """
    if exp is None or exp < 0:
        return "NO-EXP"
    if id_dev is None:
        return "NO-EXP"
    return "OK" if id_dev == exp else "LOST"


def classify(id_a, own_a, id_b, own_b, d_a, d_b):
    """One captured row -> a cause. `d_*` are the 4-word digests (sum,xor,wsum,nbytes).

    Order matters: an unread side is checked FIRST, because a sentinel digest compares equal to a
    sentinel digest and would otherwise reach 'SAME'.
    """
    if d_a[3] == 0 or d_b[3] == 0:
        return "UNREAD"           # one (or both) sides digested nothing: not a comparison
    same_digest = d_a == d_b
    if own_a is not None and own_b is not None:
        if id_a == id_b:
            if own_a == own_b:
                return "SAME" if same_digest else "CONTENT"
            return "SLOT-REUSED"
        if own_a == own_b:
            return "RELAYOUT" if same_digest else "CONTENT"
        return "ROUTING"
    # No owner on at least one side: fall back to what the bytes and the index alone can say, and say
    # only that. `UNCLASSIFIED-SAME-ID` is NOT a synonym for SAME -- it does not know which expert the
    # slot held, so it cannot rule out SLOT-REUSED.
    return "UNCLASSIFIED-SAME-ID" if id_a == id_b else "UNCLASSIFIED-DIFF-ID"


STARS = {"CONTENT"}


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

    n_own_rows = sum(1 for r in A + B if r[8] is not None)
    n_exp_rows = sum(1 for r in A + B if r[9] is not None)
    if n_own_rows and n_exp_rows == 0:
        print("  !! EXPECT CHANNEL ABSENT: rows carry `own=` but not `exp=`. The byte/content verdicts "
              "below are unaffected, but no row can say whether the DEVICE read the table this side "
              "published -- check that the build has ggml_metal_cgc_set_expect_fn and that the banner "
              "names both channels ACTIVE.")
    if n_own_rows == 0:
        print("  !! OWNER CHANNEL ABSENT: not one POOL row on either side carries `own=`. The ids and "
              "byte comparison below is still valid, but every row will be UNCLASSIFIED -- so "
              "'no CONTENT rows' must NOT be read as 'the content axis was cleared'. llama registers "
              "the callback only when CGC_POOL_CAPTURE is set on the SERVER side; check that first.")
    else:
        print(f"owner column: {n_own_rows} of {na + nb} POOL rows carry `own=`")

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

    n_same = n_diff = n_notread = n_partread = 0
    causes = {}
    mapc = collections.Counter()          # (side, OK|LOST|NO-EXP)
    lost_split = collections.Counter()    # for B's LOST rows, whether the two hosts agreed
    first_lost = None
    first = None
    first_star = None
    wanted = set(names)
    for k in shared:
        ra, rb = da[k], db[k]
        g, name = k
        if name not in wanted:
            continue
        _ga, _na_, nga, groups_a, rows_a, probe_a, nsel_a, org_a, owns_a, exps_a = ra
        _gb, _nb_, ngb, groups_b, rows_b, probe_b, nsel_b, org_b, owns_b, exps_b = rb

        nrows = min(rows_a, rows_b)
        # Rows beyond the ids operand DO NOT EXIST -- they are not unread rows. The kernel is asked for
        # `rows` rows but only writes the ones the operand has ids for, so at T=1 a `rows=12` capture
        # writes 8 groups and leaves 4 rows of pure sentinel. Counting those as a measurement gap would
        # manufacture a PART-READ verdict for a region that was never part of the question (measured:
        # 15 of 105 keys, on a run whose only real difference was none). `nsel` is the operand's id
        # count, so min(rows, nsel) is exactly "the rows that exist on both sides".
        if nsel_a > 0 and nsel_b > 0:
            nrows = min(nrows, nsel_a, nsel_b)
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
            causes["NOT-READ"] = causes.get("NOT-READ", 0) + 1
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

        # [§9.18.7] per-row cause, over every compared row of this key (not just the differing ones),
        # so the tally is a partition of what was compared rather than a description of the diff list
        key_causes = {}
        for r in range(nrows):
            ga = groups_a[r] if r < len(groups_a) else None
            gb = groups_b[r] if r < len(groups_b) else None
            oa_ = owns_a[r] if (owns_a is not None and r < len(owns_a)) else None
            ob_ = owns_b[r] if (owns_b is not None and r < len(owns_b)) else None
            if ga is None or gb is None:
                c = "ABSENT-ROW"
            else:
                c = classify(ga[0], oa_, gb[0], ob_, ga[1:], gb[1:])
            key_causes[c] = key_causes.get(c, 0) + 1
            causes[c] = causes.get(c, 0) + 1
            if c in STARS and first_star is None:
                first_star = (g, name, r, ga, gb, oa_, ob_)

        # [§9.18.8] The MAP axis: for each row, did the device use the slot the HOST expected at that
        # same position? This is a different question from every cause above -- those say WHAT changed
        # between arms, this says WHO disagrees with whom. `LOST` on an arm means that arm's device read
        # a slot its OWN host did not intend, which no cross-arm comparison can distinguish from a
        # benign relayout without it.
        for _i in range(nrows):
            _ida = groups_a[_i][0] if _i < len(groups_a) else None
            _idb = groups_b[_i][0] if _i < len(groups_b) else None
            _ea  = exps_a[_i] if (exps_a is not None and _i < len(exps_a)) else None
            _eb  = exps_b[_i] if (exps_b is not None and _i < len(exps_b)) else None
            _va  = map_verdict(_ida, _ea)
            _vb  = map_verdict(_idb, _eb)
            mapc[("A", _va)] += 1
            mapc[("B", _vb)] += 1
            if _vb == "LOST":
                # Split by whether the two HOSTS expected the same slot at that position. Comparing the
                # two expectations directly (rather than A's device id against B's expectation) keeps
                # the sentence true even if A is itself faulted -- A's device id equals A's expectation
                # only when A is healthy, and a faulted A is a case the reader must be able to separate.
                #   hosts agree  -> both published the same table, only B's DEVICE diverged  (machine)
                #   hosts differ -> the published tables differ; each device obeyed its own host
                #   A has no expectation -> the split is not answerable for that row, and says so.
                key = ("A not annotated" if _ea is None or _ea < 0
                       else ("hosts agree" if _ea == _eb else "hosts differ"))
                lost_split[key] += 1
                if first_lost is None:
                    first_lost = (g, name, _i, _ida, _ea, _idb, _eb)

        n_unread_rows = key_causes.get("UNREAD", 0)
        cs = " ".join(f"{c}={n}" for c, n in sorted(key_causes.items()))

        if not diffs:
            # A row where one side digested NOTHING is a gap in the measurement, not agreement. The
            # whole-key guard above catches the case where EVERY row is unread; this catches the one it
            # cannot -- a key where only some rows are, whose group tuples still compare equal because a
            # sentinel equals a sentinel. That is the same defect the whole-key guard exists for, one
            # level down, and it is why PART-READ is its own verdict rather than being folded into SAME.
            if n_unread_rows:
                n_partread += 1
                print(f"  g{g:<3} {name:<22} PART-READ rows={nrows} unread={n_unread_rows} "
                      f"read={read_a}/{read_b} probe={probe_a}/{probe_b}  <-- some rows digested "
                      f"nothing; equal sentinels are NOT agreement")
                print(f"  {'':<6} {'':<22} causes: {cs}")
                continue
            n_same += 1
            if not args.quiet_same:
                print(f"  g{g:<3} {name:<22} SAME      rows={nrows} probe={probe_a}/{probe_b} "
                      f"nsel={nsel_a}/{nsel_b} from={org_a}/{org_b}{tag}")
                print(f"  {'':<6} {'':<22} causes: {cs}")
        else:
            n_diff += 1
            r, ga, gb = diffs[0]
            oa_ = owns_a[r] if (owns_a is not None and r < len(owns_a)) else None
            ob_ = owns_b[r] if (owns_b is not None and r < len(owns_b)) else None
            print(f"  g{g:<3} {name:<22} DIFF      row={r}/{nrows}  A: {fmt_group(ga)} "
                  f"own={oa_}\n"
                  f"  {'':<6} {'':<22}           B: {fmt_group(gb)} own={ob_}{tag}")
            if len(diffs) > 1:
                print(f"  {'':<6} {'':<22}           +{len(diffs) - 1} more differing row(s)")
            print(f"  {'':<6} {'':<22} causes: {cs}")
            if first is None:
                first = (g, name, r, ga, gb, oa_, ob_)

    print()
    print(f"verdict rows: SAME={n_same} DIFF={n_diff} PART-READ={n_partread} NOT-READ={n_notread} "
          f"(graphs<= {maxg if maxg else 'all'}, names={len(names)})")
    print(f"causes (per compared row): " + " ".join(f"{c}={n}" for c, n in sorted(causes.items())))
    print()
    if sum(mapc.values()):
        print(f"map axis (device id vs the slot the HOST expected at that position; exp column: "
              f"{n_exp_rows} of {na + nb} POOL rows)")
        for side in ("A", "B"):
            ok, lost, nx = mapc[(side, "OK")], mapc[(side, "LOST")], mapc[(side, "NO-EXP")]
            print(f"  {side}: OK={ok} LOST={lost} n/a={nx}" +
                  ("   <-- this arm's device read a slot its own host did not intend" if lost else ""))
        if mapc[("A", "LOST")]:
            print("  !! INSTRUMENT FAULT: arm A's device disagreed with arm A's OWN host expectation on "
                  "some rows. A's graph consumes the very array the expectation is read from, so those "
                  "two are the same array by construction; a mismatch means the annotations are "
                  "misaligned (element order, or a stale snapshot) and NO verdict below is usable until "
                  "this is 0.")
        if mapc[("B", "LOST")]:
            g, name, i, ida, ea, idb, eb = first_lost
            print(f"  B-LOST first: graph {g}, {name}, row {i} -- A id={ida} (A host expected {ea}) | "
                  f"B id={idb} (B host expected {eb})")
            print(f"  B's LOST rows split by whether the two HOSTS agreed at that position: "
                  f"{dict(lost_split)}")
            if lost_split["hosts agree"] and not lost_split["hosts differ"] and not lost_split["A not annotated"]:
                print("  => the two hosts agree on the table and only B's DEVICE diverged from it. "
                      "The defect is in what the GPU read, not in what the host published.")
            elif lost_split["hosts differ"] and not lost_split["hosts agree"] and not lost_split["A not annotated"]:
                print("  => the two HOSTS' tables differ, and each device did what its own host said. "
                      "The defect is upstream, in the mapping state the host published.")
            else:
                print("  => BOTH shapes are present. They are different defects and must be separated "
                      "before either one is named.")
        elif n_exp_rows:
            print("  no B-LOST row: on every annotated position the S1 device read the slot its OWN "
                  "host expected, so the divergence is NOT in the device's table read.")
    else:
        print("map axis: not measured (`exp=` absent on every row) -- who disagrees with whom is NOT "
              "answerable from this run.")
    if n_same == 0 and n_diff == 0 and (n_notread + n_partread) > 0:
        print("NOTHING WAS COMPARED: every shared row is NOT-READ or PART-READ. Verdict WITHHELD -- a "
              "sentinel compares equal to a sentinel, and reading that as agreement is exactly how a "
              "round of this investigation was once spent on empty rows.")
        return 1

    n_content = causes.get("CONTENT", 0)
    if n_content:
        g, name, r, ga, gb, oa_, ob_ = first_star
        print(f"*** CONTENT ({n_content} row(s)): a slot delivered different bytes for the same "
              f"expert. FIRST: graph {g}, {name}, row {r}, own {oa_}/{ob_}.")
        print("    This is the fill/residency defect: the bytes behind a settled slot->expert "
              "assignment differ between arms. Look at the pool fill (what wrote that slot, and "
              "when) -- not at the routing.")
    else:
        print("no CONTENT row: the pool's BYTES are not the carrier for any compared row.")
        if n_own_rows == 0:
            print("  ...but the owner channel was ABSENT, so this is an ids+bytes statement only: "
                  "SLOT-REUSED / RELAYOUT / ROUTING were not separable, and a same-id row could "
                  "still have held a different expert. Do not upgrade it.")
        else:
            un = causes.get("UNCLASSIFIED-SAME-ID", 0) + causes.get("UNCLASSIFIED-DIFF-ID", 0)
            if un:
                print(f"  ({un} row(s) UNCLASSIFIED -- no owner on at least one side; those rows "
                      f"cannot rule out SLOT-REUSED.)")
    if first is not None:
        g, name, r, ga, gb, oa_, ob_ = first
        # The trailing sentence is driven by the CAUSE, not by whether the ids happened to match: with
        # the owner column available, "ids differ" no longer implies "the mapping is at fault" (it can
        # be a benign RELAYOUT) and "ids same" no longer implies "the bytes are at fault" (it can be
        # SLOT-REUSED). Phrasing it off the ids is how this line contradicted the CONTENT headline.
        c = classify(ga[0], oa_, gb[0], ob_, ga[1:], gb[1:]) if (ga and gb) else "ABSENT-ROW"
        note = {
            "CONTENT":       "Same expert, different bytes: the fill/residency path delivered "
                             "different content. This is a real defect.",
            "SLOT-REUSED":   "One slot index, two different experts: 'same ids' was a coincidence, "
                             "not agreement. A mapping/accounting question.",
            "RELAYOUT":      "Different slots, same expert, same bytes: benign relayout, not a defect.",
            "ROUTING":       "Different experts were fetched: look UPSTREAM of the gather, at routing.",
            "UNREAD":        "One side digested nothing: a measurement gap, not a difference.",
            "PART-READ":     "One side digested nothing on this row: a measurement gap.",
        }.get(c, "No owner on at least one side: ids+bytes only, and it cannot rule out SLOT-REUSED.")
        print(f"FIRST DIFF: graph {g}, {name}, row {r} -- ids "
              f"{'differ' if (ga and gb and ga[0] != gb[0]) else 'SAME'}, own {oa_}/{ob_} [{c}]. {note}")
    else:
        print("NO DIFFERENCE FOUND in any compared row.")
        print("  Read that as: the gather's ids AND the bytes behind them are identical on both arms "
              "in the compared graphs, so the divergence is not in what the gather read.")
    print()
    print("CONTROL (mandatory): run the SAME arm twice and diff the two POOL streams. If that reports "
          "anything other than SAME, the readout is the finding -- not the engine. The copy reads a "
          "buffer another kernel may still be writing, and the owner map is read unlocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
