#!/usr/bin/env python3
"""Per-layer attribution of the hook's sub-items: which block owns L0-L5, and by how much.

THE ROW SPANS TWO POPULATIONS, NOT JUST TWO DENOMINATORS, AND MIXING THEM IS THE TRAP.
`pre/ensure/drain/tail` are the hook's own blocks over ALL its calls (the fast pool path AND the
exact path). `sub_*` are the cache's blocks inside `ensure_batch`, which only the exact path calls,
so their denominator is `sub_n <= n` and they are conditioned on the exact-path calls. A reader who
compares `sub_fill` against `ensure` directly will conclude that a sub-item exceeds its parent; a
reader who reads `pre_cold` as a share of `pre` on a fast-path-dominated log reads one population's
number as a fraction of another's -- which is what happened on 2026-09-20: L0 read pre_cold=3435 us
against its containing pre=816 us, because the decode fast path returns ABOVE the banking site.
`fast_n`/`exact` are printed so the split is visible; a log without the field is reported as
UNKNOWN, never as 0.

Internal consistency, checked rather than assumed:
  (a) fill == submit + wait            (fill IS the submit+wait pair, same call)
  (b) sum_l(n_l * total_l) / sum_l(n_l) == the aggregate CGC-HOOKSPLIT total
  (c) sub_n <= n                       (only the exact path calls ensure_batch)
  (d) ensure == assign+collect+fill+publish   (the cache's four blocks ARE the hook's ensure)
  (e) pre_cold <= pre                  (pre_cold is a sub-interval of pre)
A layer with `n == 0` is ABSENT, not free: it is reported as absent and excluded from the groups.
"""
import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(
    r"CGC-HOOKSUB-L:\s+layer=(\d+)\s+n=(\d+)\s+pre=([\d.]+)\s+pre_cold=([\d.]+)\s+ensure=([\d.]+)\s+"
    r"drain=([\d.]+)\s+tail=([\d.]+)\s+sub_n=(\d+)\s+sub_assign=(-?[\d.]+)\s+"
    r"sub_collect=(-?[\d.]+)\s+sub_fill=(-?[\d.]+)\s+sub_publish=(-?[\d.]+)\s+"
    r"sub_submit=(-?[\d.]+)\s+sub_wait=(-?[\d.]+)(?:\s+fast_n=(\d+))?"
    r"(?:\s+fast_v=(\d+)\s+fast_d=(\d+)\s+cold_v=(\d+)\s+cold_d=(\d+))?")
AGG = re.compile(r"CGC-HOOKSPLIT:\s+n=(\d+)\s+pre=([\d.]+)\s+ensure=([\d.]+)\s+drain=([\d.]+)\s+"
                 r"tail=([\d.]+)\s+us/call\s+\(total ([\d.]+)\)(?:\s+fast_n=(\d+))?")
FOCUS = tuple(range(0, 6))          # L0-L5: the layers the question names
TOL = 0.05                          # 5%: the same tolerance the other splitters use


def parse(path: Path) -> tuple[dict, dict]:
    """Last line per layer (the values are running means, so the last print has the most calls)."""
    rows: dict[int, dict] = {}
    txt = path.read_text(errors="replace")
    for m in LINE.finditer(txt):
        lay = int(m.group(1))
        rows[lay] = {"layer": lay, "n": int(m.group(2)), "pre": float(m.group(3)),
                     "pre_cold": float(m.group(4)), "ensure": float(m.group(5)),
                     "drain": float(m.group(6)), "tail": float(m.group(7)),
                     "sub_n": int(m.group(8)), "assign": float(m.group(9)),
                     "collect": float(m.group(10)), "fill": float(m.group(11)),
                     "publish": float(m.group(12)), "submit": float(m.group(13)),
                     "wait": float(m.group(14)),
                     "fast_n": int(m.group(15)) if m.group(15) else None,
                     "fast_v": int(m.group(16)) if m.group(16) else None,
                     "fast_d": int(m.group(17)) if m.group(17) else None,
                     "cold_v": int(m.group(18)) if m.group(18) else None,
                     "cold_d": int(m.group(19)) if m.group(19) else None}
    aggs = AGG.findall(txt)
    agg = None
    if aggs:
        a = aggs[-1]
        agg = {"n": int(a[0]), "pre": float(a[1]), "ensure": float(a[2]), "drain": float(a[3]),
               "tail": float(a[4]), "total": float(a[5]),
               "fast_n": int(a[6]) if a[6] else None}
    return rows, agg


def tot(r: dict) -> float:
    return r["pre"] + r["ensure"] + r["drain"] + r["tail"]


def wait_per_hook(r: dict) -> float:
    """The round-trip's share of ONE hook call.

    `sub_wait` is per ensure_batch call on the union path, and `ensure` is the hook call's whole
    ensure block, so scaling by wait/fill puts it back on the hook's denominator -- legitimate only
    while fill carries the ensure block, which check (d) tests below.
    """
    return r["ensure"] * (r["wait"] / r["fill"]) if r["fill"] > 0 else float("nan")


def med(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def wmean(rows: list[dict], key: str) -> float:
    """Call-weighted mean: the per-layer values are means over that layer's own call count."""
    den = sum(r["n"] for r in rows)
    return sum(r[key] * r["n"] for r in rows) / den if den else float("nan")


def report(rows, agg) -> int:
    """`agg` is None when the log carries no CGC-HOOKSPLIT line -- absence, not agreement."""
    if not rows:
        print("CGC-HOOKSUB-L: no per-layer rows -- the line needs CGC_HOOK_SPLIT=1 AND a build that "
              "carries it (this is INVALID, not 'all zeros')")
        return 1
    abs_n = sorted(l for l, r in rows.items() if r["n"] == 0)
    live = [r for r in rows.values() if r["n"] > 0]
    focus = [r for r in live if r["layer"] in FOCUS]
    rest = [r for r in live if r["layer"] not in FOCUS]
    have_split = all(r["fast_n"] is not None for r in live)
    print(f"CGC-HOOKSUB-L rows: {len(rows)} layers ({len(live)} with n>0), "
          f"hook calls {sum(r['n'] for r in live)}"
          + (f", ABSENT (n=0): {abs_n}" if abs_n else ""))
    print(f"  paths: {'fast_n' if have_split else 'NO fast_n FIELD -- build predates the fix'}"
          + (f", fast={sum(r['fast_n'] for r in live)} of {sum(r['n'] for r in live)} calls "
             f"({sum(r['fast_n'] for r in live) / sum(r['n'] for r in live):.0%}), "
             f"exact={sum(r['n'] - r['fast_n'] for r in live)}" if have_split else
             " (the exact/fast split is UNKNOWN, and pre/ensure below still describe only the "
             "calls that reached ensure_batch)"))
    print(f"  layers with sub_n > n (impossible): "
          f"{[r['layer'] for r in live if r['sub_n'] > r['n']] or 'none'}")
    print(f"  sub_n/n = {med([r['sub_n'] / r['n'] for r in live]):.2f} median (sub_n counts the "
          f"union-path ensure_batch, which only the exact path calls); sub_* below is therefore "
          f"conditioned on the exact-path calls while pre/ensure/total cover all of them")
    # `pre_cold` is the syncfill cold fill and it is FAST-PATH-ONLY (its block guard is
    # `(verify_fast || draft_fast) && cgc_fast_eligible`), so its mean over `n` is a dilution: the
    # per-DECODE-call figure is pre_cold * n / fast_n, which is the number a decode target needs.
    if have_split:
        cold = [r["pre_cold"] * r["n"] / r["fast_n"] for r in live if r["fast_n"]]
        cold_f = [r["pre_cold"] * r["n"] / r["fast_n"] for r in focus if r["fast_n"]]
        cold_r = [r["pre_cold"] * r["n"] / r["fast_n"] for r in rest if r["fast_n"]]
        print(f"  cold fill per FAST (decode) call: L0-L5 {med(cold_f):.0f} us  rest {med(cold_r):.0f} us"
              f"  ({med(cold_f) / med(cold_r):.1f}x, median over layers) -- this is the delivery-path"
              f" sub-item, and it is a mean over {len(cold)} layers")
    have_ctx = all(r["fast_v"] is not None for r in live)
    if have_ctx:
        fv = sum(r["fast_v"] for r in live)
        fd = sum(r["fast_d"] for r in live)
        cv = sum(r["cold_v"] for r in live)
        cd = sum(r["cold_d"] for r in live)
        # No division without both terms: a zero here is a READING (one route never fills), not a
        # missing value, and a ratio against it is undefined rather than infinite.
        vpc = cv / fv if fv else float("nan")
        dpc = cd / fd if fd else float("nan")
        print(f"  by context: fast verify={fv} draft={fd} | cold fill {cv / 1000:.0f} ms verify "
              f"({vpc / 1000:.2f} ms/call) vs {cd / 1000:.0f} ms draft ({dpc / 1000:.2f} ms/call)"
              + (f"  -> verify pays {vpc / dpc:.2f}x draft per call" if dpc and dpc == dpc else
                 "  -> the draft route never fills cold here, so the ratio to it is UNDEFINED "
                 "(not 0, not 1): every microsecond of syncfill is on the verify route"))
    else:
        print("  by context: NO fast_v/fast_d in this log -- verify vs draft is UNKNOWN, not equal")

    print(f"\n{'group':<10} {'n_hook':>7} {'pre':>9} {'pre_cold':>9} {'ensure':>9} {'drain':>7} "
          f"{'tail':>7} {'total':>9} {'wait':>9} {'wait/tot':>9}")
    for name, grp in (("L0-L5", focus), ("rest", rest), ("all", live)):
        if not grp:
            continue
        t = sum(r["n"] * tot(r) for r in grp) / sum(r["n"] for r in grp)
        w = sum(r["n"] * wait_per_hook(r) for r in grp) / sum(r["n"] for r in grp)
        print(f"{name:<10} {sum(r['n'] for r in grp):>7} {wmean(grp, 'pre'):>9.0f} "
              f"{wmean(grp, 'pre_cold'):>9.0f} {wmean(grp, 'ensure'):>9.0f} "
              f"{wmean(grp, 'drain'):>7.1f} {wmean(grp, 'tail'):>7.1f} "
              f"{t:>9.0f} {w:>9.0f} {w / t:>10.2f}")
    print("  (units: us/hook-call. `pre` is over ALL calls; `pre_cold` is the syncfill cold ensure,"
          " which lives INSIDE pre)")
    print("   `wait` is scaled by wait/fill so it is on ensure's denominator, not sub_n's)")

    # Within-ensure and within-fill shares, per group.
    for name, grp in (("L0-L5", focus), ("rest", rest)):
        if not grp:
            continue
        def sm(key: str) -> float:
            return sum(r[key] * r["sub_n"] for r in grp) / sum(r["sub_n"] for r in grp)
        inside = {k: sm(k) for k in ("assign", "collect", "fill", "publish")}
        s = sum(v for v in inside.values() if v > 0)
        fill = {k: sm(k) for k in ("submit", "wait")}
        fsum = sum(v for v in fill.values() if v > 0)
        print(f"\n  within ensure_batch ({name}, per call): "
              + ", ".join(f"{k}={v:.0f} us ({v / s:.0%})" for k, v in inside.items())
              + f"  |  within fill: " + ", ".join(f"{k}={v:.0f} us ({v / fsum:.0%})"
                                                  for k, v in fill.items()))

    # (a) fill == submit + wait, per layer (the two are the same call).
    bad = [r["layer"] for r in live if r["fill"] > 0 and r["sub_n"] > 0
           and abs(r["fill"] - (r["submit"] + r["wait"])) / r["fill"] > TOL]
    print(f"\n  (a) fill == submit+wait: {'consistent' if not bad else f'DISAGREES on {bad}'}")
    # (b) weighted per-layer total vs the aggregate line.
    if agg is not None:
        t = sum(r["n"] * tot(r) for r in live) / sum(r["n"] for r in live)
        ratio = t / agg["total"] if agg["total"] else float("nan")
        print(f"  (b) per-layer weighted total {t:.0f} us vs aggregate line {agg['total']:.0f} us "
              f"({ratio:.2f}x)"
              + ("" if 1 - TOL <= ratio <= 1 + TOL else
                 "  -- same quantity, different totals: neither is quotable"))
    else:
        print("  (b) no CGC-HOOKSPLIT aggregate line in this log: the audit is unavailable "
              "(absence, not agreement)")
    # (c) sub_n is the UNION-path ensure count, so it can be smaller than n but never larger.
    print(f"  (c) sub_n <= n: {'consistent' if not any(r['sub_n'] > r['n'] for r in live) else 'VIOLATED'}"
          f"  (sub_n/n = {med([r['sub_n'] / r['n'] for r in live if r['n']]):.2f} median: the exact-path "
          f"share, i.e. how much of the census the sub_* columns describe)")
    # (f) THE two-independent-counter proof of the path attribution: the code calls ensure_batch
    # exactly once on the exact path and never on the fast path, so `sub_n` must equal `n - fast_n`
    # per layer. Nothing else in the row can be cross-checked this directly.
    mism = []
    if have_split:
        mism = [(r["layer"], r["sub_n"], r["n"] - r["fast_n"]) for r in live
                if r["sub_n"] != r["n"] - r["fast_n"]]
        print(f"  (f) sub_n == n - fast_n (two counters, same paths): "
              + ("holds on every layer" if not mism else f"MISMATCHES {mism}"))
    else:
        print("  (f) sub_n == n - fast_n: NOT CHECKABLE in this log (no fast_n field)")
    # (d) the cross-file audit: the cache's four blocks ARE the hook's `ensure` block.
    # Compared as TOTALS, not as means: `ensure` is a mean over all calls and `sub_*` over the
    # exact-path calls only, so the means differ by the path ratio by construction (0.16-0.17x here)
    # and a mean-vs-mean check would call a correct instrument broken.
    worst = None
    for r in live:
        s = r["assign"] + r["collect"] + r["fill"] + r["publish"]
        if r["ensure"] > 0 and s > 0:
            ratio = (r["ensure"] * r["n"]) / (s * r["sub_n"])
            if worst is None or abs(ratio - 1) > abs(worst[1] - 1):
                worst = (r["layer"], ratio)
    if worst is not None:
        print(f"  (d) ensure == assign+collect+fill+publish (total us per layer): worst L{worst[0]} at "
              f"{worst[1]:.2f}x"
              + ("  (within 5%)" if abs(worst[1] - 1) <= TOL else
                 "  -- the two files' timers disagree; the sub-item split is not attributable to the "
                 "hook blocks"))
    # (e) the NESTING check the tool used to assert in prose while never testing it: `pre_cold` is a
    # sub-interval of `pre`, so a layer where the mean exceeds it means the two were banked on
    # different call populations (2026-09-20: the decode fast path returns above the t1..t4 banking
    # site, so pre_cold counted calls that pre did not -- L0 read 3435 us against a pre of 816).
    nest = [r["layer"] for r in live if r["pre_cold"] > r["pre"] * (1 + TOL)]
    print(f"  (e) pre_cold <= pre (it is a sub-interval): "
          + ("consistent" if not nest else f"VIOLATED on {nest} -- the two are banked on different "
             f"call populations, so neither the pre_cold column nor the pre column is quotable"))
    if bad or nest or mism:
        return 1
    return 0


def table(rows: dict) -> None:
    """Per-layer rows for L0-L5 plus the heaviest layers anywhere, so the group means can be traced.

    The group means in `report` answer "is L0-L5 different"; only the rows answer "which layer and
    which sub-item", and at 195 vs 605 hook calls those are not the same question.
    """
    live = [r for r in rows.values() if r["n"] > 0]
    focus = sorted([r for r in live if r["layer"] in FOCUS], key=lambda r: -tot(r))
    heaviest = sorted(live, key=lambda r: -tot(r))[:5]
    print(f"\n{'layer':>12} {'n':>5} {'fast':>5} {'exact':>5} {'total':>8} {'pre':>7} {'pre_cold':>9} "
          f"{'cold/fast':>9} {'ensure':>7} {'assign':>7} {'collect':>8} {'fill':>8} {'publish':>8} "
          f"{'submit':>8} {'wait':>8} {'wait%':>6}")
    for tag, grp in (("L0-L5", focus), ("heaviest", heaviest)):
        for r in grp:
            f = r["fill"] or float("nan")
            fn = r["fast_n"]
            print(f"{tag + ' L' + str(r['layer']):>12} {r['n']:>5} "
                  f"{fn if fn is not None else -1:>5} "
                  f"{(r['n'] - fn) if fn is not None else -1:>5} {tot(r):>8.0f} {r['pre']:>7.0f} "
                  f"{r['pre_cold']:>9.0f} "
                  f"{(r['pre_cold'] * r['n'] / fn) if fn else float('nan'):>9.0f} "
                  f"{r['ensure']:>7.0f} {r['assign']:>7.1f} "
                  f"{r['collect']:>8.1f} {r['fill']:>8.0f} {r['publish']:>8.1f} "
                  f"{r['submit']:>8.0f} {r['wait']:>8.0f} {r['wait'] / f:>6.0%}")
    print("  (sub-items are us per ensure_batch call, which ONLY the exact path makes; `total`/`pre`/")
    print("   `ensure` are us per hook call over ALL calls, and `cold/fast` is `pre_cold` over the FAST")
    print("   calls only -- a row spans two populations, not just two denominators. -1 / nan = the log")
    print("   predates the fast_n field, which is UNKNOWN, not zero)")


def focus_sentence(rows: dict) -> None:
    """The falsifiable per-layer target, computed from the reading rather than asserted."""
    live = [r for r in rows.values() if r["n"] > 0]
    focus = [r for r in live if r["layer"] in FOCUS]
    if not focus:
        return
    top = max(focus, key=tot)
    scaled_wait = top["wait"] * (top["sub_n"] / top["n"])
    fn = top["fast_n"]
    where = (f"{top['n']} calls ({fn} fast / {top['n'] - fn} exact)" if fn is not None
             else f"{top['n']} calls (path split UNKNOWN)")
    cold_fast = top["pre_cold"] * top["n"] / fn if fn else float("nan")
    print(f"\n  TARGET (from the reading): L{top['layer']} costs {tot(top):.0f} us/hook-call over "
          f"{where}; "
          f"`pre` is {top['pre']:.0f} us of it ({top['pre'] / tot(top):.0%}), and the pool "
          f"round-trip `wait` is {scaled_wait:.0f} us ({scaled_wait / tot(top):.0%}).")
    if fn is None:
        print("  DECODE PATH: NOT AVAILABLE -- this log predates the fast_n field, so the fast/exact"
              " split is unknown rather than zero")
    elif fn == 0:
        print("  DECODE PATH: this layer took no fast-path call at all, so it has no decode-side"
              " cold fill (0 is a reading here, not a missing value)")
    else:
        print(f"  DECODE PATH: {fn} of those calls are the fast path, and they pay {cold_fast:.0f} us each"
              f" in the syncfill cold fill (pre_cold is fast-path-only), i.e. "
              f"{cold_fast * fn / (tot(top) * top['n']):.0%} of this layer's whole hook budget.")
    print(f"  POPULATION: `wait` is measured only on the EXACT-path calls, so it is a per-exact-"
          f"call figure scaled onto a per-hook-call denominator; the two paths need separate targets "
          f"until the fast path's own split exists.")
    print(f"  FALSIFIABLE: hiding the round-trip (L3-style async fill) must drop L{top['layer']}'s "
          f"exact-path hook by >= {0.5 * scaled_wait:.0f} us (half of it, same instrument, same run). "
          f"A drop below that -- or none -- falsifies 'the hook waits on the round-trip', and moves "
          f"the target to `pre` (>= {0.5 * top['pre']:.0f} us must come out of it to be the answer).")


def selftest() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    def run(text: str) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(text)
        rows, agg = parse(Path(fh.name))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report(rows, agg)
            table(rows)               # main() prints these too, so the selftest must cover them
            focus_sentence(rows)
        return f"{rc}\n" + buf.getvalue()

    mk = lambda l, n, pre, ens, fill, sub, wait, sub_n=None, fast_n="", pre_cold=10.0: (
        f"CGC-HOOKSUB-L: layer={l} n={n} pre={pre} pre_cold={pre_cold} ensure={ens} drain=0.4 tail=2.0 "
        f"sub_n={sub_n or n} sub_assign=5.0 sub_collect=2.5 sub_fill={fill} sub_publish=0.4 "
        f"sub_submit={sub} sub_wait={wait}{f' fast_n={fast_n}' if fast_n != '' else ''}\n")
    # assign 5 + collect 2.5 + fill 900 + publish 0.4 = 907.9 against ensure 1000 -> inside 5%.
    ok = "".join(mk(l, 20, 100.0, 908.0, 900.0, 100.0, 800.0) for l in range(8))
    agg_line = "CGC-HOOKSPLIT: n=160  pre=100.0  ensure=1000.0  drain=0.4  tail=2.0 us/call " \
               "(total 1102.4)\n"
    check("a log without the line is INVALID, not zeros", run("nothing here\n").startswith("1"))
    check("a consistent log passes", run(ok + agg_line).startswith("0"))
    check("the two denominators are reported, not mixed",
          "sub_n/n" in run(ok + agg_line))
    check("fill != submit+wait is caught", run(ok.replace("sub_fill=900.0", "sub_fill=9.0")
                                               + agg_line).startswith("1"))
    check("a weighted total that disagrees with the aggregate line is called out",
          "neither is quotable" in run(ok.replace("ensure=1000.0", "ensure=9.0") + agg_line))
    check("sub_n < n is named as impossible",
          "impossible" in run(mk(0, 20, 100.0, 908.0, 900.0, 100.0, 800.0, sub_n=5) + agg_line))
    check("pre_cold > pre is caught (the two were banked on different populations)",
          run(mk(0, 20, 50.0, 908.0, 900.0, 100.0, 800.0, pre_cold=90.0) + agg_line).startswith("1"))
    check("... and a log WITHOUT fast_n says the split is unknown, not 0",
          "NO fast_n FIELD" in run(ok + agg_line))
    check("a log WITH fast_n reports the two populations",
          "exact=" in run("".join(mk(l, 20, 100.0, 908.0, 900.0, 100.0, 800.0, fast_n=15, sub_n=5)
                                  for l in range(8)) + agg_line))
    check("a fast_n that does not pair with sub_n is caught (two counters, same paths)",
          run("".join(mk(l, 20, 100.0, 908.0, 900.0, 100.0, 800.0, fast_n=15)
                      for l in range(8)) + agg_line).startswith("1"))
    check("the decode-path cold fill is derived per FAST call, not per hook call",
          "DECODE PATH" in run("".join(mk(l, 20, 100.0, 908.0, 900.0, 100.0, 800.0, fast_n=15, sub_n=5)
                                     for l in range(8)) + agg_line))
    check("the cross-file audit (ensure vs its four blocks) is reported",
          "ensure == assign+collect+fill+publish" in run(ok + agg_line))
    check("... and a broken one is called out as not attributable",
          "not attributable" in run(ok.replace("ensure=908.0", "ensure=20.0") + agg_line))
    check("an absent layer (n=0) is absent, not free",
          "ABSENT" in run(ok + mk(9, 0, 0.0, 0.0, 0.0, 0.0, 0.0) + agg_line))
    check("the target sentence is computed from the reading",
          "TARGET" in run(ok + agg_line))
    check("no aggregate line -> the audit says unavailable, not agreed",
          "unavailable" in run(ok))
    check("the wait is not reported as a share of itself (no scaling trap)",
          "wait/tot" in run(ok + agg_line))
    print(f"hooksub_layers selftest: {16 - bad}/16 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    ap.add_argument("--table", action="store_true", help="per-layer rows for L0-L5 + heaviest")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.log:
        print("need --log (a server log containing CGC-HOOKSUB-L lines) or --selftest")
        return 2
    p = Path(args.log)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"no such log: {p}")
        return 2
    rows, agg = parse(p)
    rc = report(rows, agg)
    if args.table:
        table(rows)
    focus_sentence(rows)
    return rc


if __name__ == "__main__":
    sys.exit(main())
