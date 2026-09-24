#!/usr/bin/env python3
"""Price prescription A (speculative slot binding) by measuring h on a REAL id sequence.

Why this exists
---------------
`docs/GAP_FIX_WHITEPAPER_2026-09-24.md` §4 prices A at 15.99 t/s (+27%) and §6 sets a hard gate:
`h >= 0.65`, where h is the probability that a top-k hook call's ids were ALL predicted. Until now
h had no measured value anywhere -- §9.1 of the whitepaper lists it as "the only number that
separates +8% from +27%".

The trap this script is built to avoid: h is NOT a per-expert reuse rate. A hook call carries
n_tokens * top_k ids (32 at ntok=4) and A's payoff is all-or-nothing per call -- one unpredicted id
still forces the blocking fill and, worse, forces the pre-built graph to be thrown away. So the
quantity that prices A is P(uncovered == 0), which can be ~0 while the *coverage* looks healthy
(0.854). The whitepaper's own naive landing point `0.87**8 = 33%` uses the per-token exponent and
is therefore neither the right formula nor, on its own, conservative.

A second trap: coverage and h are related but one does not bound the other usefully in the
direction people assume. mean(uncovered) = 2.83 bounds h only as `h <= 1 - 2.83/x_max`, i.e.
`h <= 0.854` -- which does NOT clear the 0.65 gate. The distribution decides, so it has to be
measured, and this script measures it.

Input
-----
The `CGC_IDSEQ_DUMP=<path>` trace emitted by `llama_context::expert_cache_on_topk`. One line per
hook call:

    <pass> <ctx_is_mtp> <il> <n_tokens> <n_expert_used> <id0> <id1> ... <idN-1>

`pass` increments whenever the layer walk wraps (il goes back down), so consecutive lines with the
same (ctx, il) and increasing pass are consecutive calls of the same layer -- the only pairs this
script compares. Prefill passes are present in the file but are excluded by `n_tokens <= 8`
upstream and again by `--max-ntok` here.

Candidate sets priced
---------------------
* `prev_union`  -- the previous call's own ids (no widening). This is the cheapest possible
  predictor and the one A's "widen to top-16" note was written against.
* `freqK`       -- the per-layer frequency top-K taken over the WHOLE trace. This is an oracle-ish
  upper bound for "a static hot set" (it has seen the future), and it is exactly the set
  `PIN_PROFILE` tried to use. Reported so the two are never confused: prev_union is what A can
  actually have at submit time; freqK is what a resident/pinned pool can have.
* `prev_union|freqK` -- union of both, i.e. "yesterday's ids plus the pinned hot set".

Usage
-----
    python3 scripts/check/idseq_h_curve.py TRACE [--max-ntok N] [--ks 8,16,32,64,143]
    python3 scripts/check/idseq_h_curve.py --self-test
"""

import argparse
import collections
import sys

# The gate written down BEFORE the number was known (whitepaper §6 / §10).
H_GATE = 0.65


def parse_trace(text):
    """-> list of dicts, in file order. Malformed lines are skipped, not fatal."""
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rec = {
                "pass": int(parts[0]),
                "ctx": int(parts[1]),
                "il": int(parts[2]),
                "ntok": int(parts[3]),
                "k": int(parts[4]),
                "ids": [int(x) for x in parts[5:]],
                "line": ln,
            }
        except ValueError:
            continue
        if len(rec["ids"]) != rec["ntok"] * rec["k"]:
            # A truncated row would silently under-count the union -- the exact failure mode this
            # repo has already hit twice with id dumps. Refuse it loudly.
            raise SystemExit(
                "idseq_h_curve: line %d has %d ids but ntok*k = %d -- refusing to price a "
                "truncated row" % (ln, len(rec["ids"]), rec["ntok"] * rec["k"])
            )
        out.append(rec)
    return out


def freq_topk(recs, k):
    """Per-layer frequency ranking over the WHOLE trace (oracle-ish: sees the future)."""
    cnt = collections.defaultdict(collections.Counter)
    for r in recs:
        cnt[r["il"]].update(r["ids"])
    return {il: [e for e, _ in c.most_common(k)] for il, c in cnt.items()}


def price(recs, ks, max_ntok):
    """For every consecutive same-(ctx, il) pair, score each candidate set."""
    by_layer = collections.defaultdict(list)
    for r in recs:
        if r["ntok"] <= max_ntok:
            by_layer[(r["ctx"], r["il"])].append(r)

    freq_sets = {k: freq_topk(recs, k) for k in ks if k > 0}

    # Score buckets: key -> (n, sum_cov, n_hit, sum_union, sum_uncovered)
    acc = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0])

    for (ctx, il), seq in by_layer.items():
        for prev, cur in zip(seq, seq[1:]):
            true_set = set(cur["ids"])
            if not true_set:
                continue
            prev_set = set(prev["ids"])
            cand_sets = {"prev_union": prev_set}
            for k, tbl in freq_sets.items():
                s = set(tbl.get(il, []))
                cand_sets["freq%d" % k] = s
                cand_sets["prev|freq%d" % k] = prev_set | s

            for name, cand in cand_sets.items():
                covered = len(true_set & cand)
                uncovered = len(true_set) - covered
                key = (cur["ntok"], name)
                a = acc[key]
                a[0] += 1
                a[1] += covered / len(true_set)
                a[2] += 1 if uncovered == 0 else 0
                a[3] += len(true_set)
                a[4] += uncovered

    rows = []
    for (ntok, name), (n, cov_sum, hit, union_sum, unc_sum) in sorted(acc.items()):
        rows.append({
            "ntok": ntok,
            "set": name,
            "n": n,
            "cov": cov_sum / n if n else 0.0,
            "h": hit / n if n else 0.0,
            "union_avg": union_sum / n if n else 0.0,
            "uncovered_avg": unc_sum / n if n else 0.0,
        })
    return rows


def verdict(rows, ntok=None):
    """h for the `prev_union` set at the widest ntok present (the delivery shape)."""
    cand = [r for r in rows if r["set"] == "prev_union"]
    if not cand:
        return None
    if ntok is not None:
        cand = [r for r in cand if r["ntok"] == ntok]
        if not cand:
            return None
    return max(cand, key=lambda r: r["ntok"])


def render(rows, gate=H_GATE):
    print("%-6s %-14s %6s %8s %8s %10s %12s" %
          ("ntok", "candidate set", "n", "cov", "h", "union_avg", "uncov_avg"))
    print("-" * 70)
    for r in rows:
        print("%-6d %-14s %6d %8.4f %8.4f %10.2f %12.2f" %
              (r["ntok"], r["set"], r["n"], r["cov"], r["h"], r["union_avg"], r["uncovered_avg"]))
    print()
    v = verdict(rows)
    if v is None:
        print("no prev_union rows -- nothing to judge")
        return 1
    print("A's gate (whitepaper §6, written before the number existed): h >= %.2f" % gate)
    print("measured h (cheapest predictor, prev_union, ntok=%d): %.4f" % (v["ntok"], v["h"]))
    if v["h"] >= gate:
        print("VERDICT: PASS -- A is worth building")
    else:
        print("VERDICT: FAIL -- A would spend its budget on fallbacks (h below gate by %.2f)" %
              (gate - v["h"]))
    return 0


# --------------------------------------------------------------------------------------------
# Self-test: the failure mode guarded is a plausible-looking number, so the tests assert on the
# number, not on "it ran".
# --------------------------------------------------------------------------------------------

def _mk(pass_n, ctx, il, ntok, ids):
    k = len(ids) // ntok
    return "%d %d %d %d %d %s" % (pass_n, ctx, il, ntok, k, " ".join(str(i) for i in ids))


def self_test():
    ok = 0
    fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print("FAIL: %s" % name)

    # Case 1: perfectly repetitive routing -> h must be 1.0.
    t1 = "\n".join(_mk(p, 0, 0, 1, [1, 2, 3]) for p in range(4))
    r1 = price(parse_trace(t1), ks=[], max_ntok=8)
    check("repeat -> h=1", abs(verdict(r1)["h"] - 1.0) < 1e-9)

    # Case 2: every call introduces one new expert -> coverage stays high but h must be 0.
    # This is the whole point of the script: 8 ids, 1 new each time => cov 7/8 = 0.875, h = 0.
    seq = []
    for p in range(5):
        seq.append(_mk(p, 0, 0, 1, [10 + p, 2, 3, 4, 5, 6, 7, 8]))
    r2 = price(parse_trace("\n".join(seq)), ks=[], max_ntok=8)
    v2 = verdict(r2)
    check("one-new-each -> h=0", v2["h"] == 0.0)
    check("one-new-each -> cov~0.875", abs(v2["cov"] - 0.875) < 1e-9)

    # Case 3: the whitepaper's own numbers must REPRODUCE, not just be quoted.
    # cov 0.854 with union 19.36 => mean uncovered 2.83; assert the arithmetic is recoverable.
    check("rho arithmetic: 19.36*(1-0.8542)=2.83", abs(19.36 * (1 - 0.8542) - 2.83) < 0.01)

    # Case 4: coverage does NOT bound h downward -- two traces with the same coverage, h=0 and h=1.
    a = "\n".join(_mk(p, 0, 0, 1, [10 + p, 2, 3, 4, 5, 6, 7, 8]) for p in range(4))
    b = "\n".join(_mk(p, 0, 0, 1, [1, 2, 3, 4, 5, 6, 7, 8]) for p in range(4))
    va, vb = verdict(price(parse_trace(a), ks=[], max_ntok=8)), verdict(price(parse_trace(b), ks=[], max_ntok=8))
    check("same-ish coverage, different h", va["h"] != vb["h"])

    # Case 5: a truncated row must be refused, not silently priced.
    try:
        price(parse_trace("0 0 0 2 8 1 2 3"), ks=[], max_ntok=8)
        check("truncated row refused", False)
    except SystemExit:
        check("truncated row refused", True)

    # Case 6: freqK is oracle-ish -- with a static hot set it must reach h=1 where prev_union
    # cannot, so the two are never confused in a report.
    t6 = "\n".join(_mk(p, 0, 0, 1, [10 + p, 2, 3]) for p in range(4))
    r6 = price(parse_trace(t6), ks=[4], max_ntok=8)
    hz = {r["set"]: r["h"] for r in r6 if r["ntok"] == 1}
    check("freq4 beats prev_union", hz.get("freq4", 0) > hz.get("prev_union", 1))

    # Case 7: --max-ntok must actually exclude a wide pass (prefill rows).
    t7 = _mk(0, 0, 0, 512, [1] * 8 * 512) + "\n" + "\n".join(_mk(p, 0, 0, 1, [1, 2]) for p in range(3))
    r7 = price(parse_trace(t7), ks=[], max_ntok=8)
    check("max_ntok excludes prefill", all(r["ntok"] <= 8 for r in r7))

    print("selftest: %d/%d" % (ok, ok + fail))
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="CGC_IDSEQ_DUMP file")
    ap.add_argument("--max-ntok", type=int, default=8, help="skip calls wider than this (prefill)")
    ap.add_argument("--ks", default="16,32,64,143",
                    help="frequency top-K candidate sets to price (0 disables)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.trace:
        ap.error("trace required unless --self-test")

    with open(args.trace) as f:
        recs = parse_trace(f.read())
    if not recs:
        print("empty trace", file=sys.stderr)
        return 1
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    return render(price(recs, ks, args.max_ntok))


if __name__ == "__main__":
    sys.exit(main())
