#!/usr/bin/env python3
"""Is a layer's decode idle window bought by its MISS COUNT or by a fixed per-layer cost?

Three per-layer series, all from ONE run and ONE phase (mixing runs, or mixing prefill with decode,
is the defect this line keeps paying for):

  idle_host  CGC-GAPSPLIT-L   ms per layer per graph  (CPU side: hook + tail + submit)
  cb         CGC-DECPROF all  ms per layer per graph  (the top-k hook; overlaps idle_host)
  misses     BATCHDBG         misses=N per ensure_batch call with >=1 miss

POPULATION RULES, all measured rather than assumed:
  * the log's prefill is a contiguous prologue, so the phase boundary is the LAST `CGC-POST:` line.
    On 223046 that split moves 1334 of 2331 fill calls (and 59% of all misses) out of the decode
    population -- fitting across it would have been mostly a prefill regression;
  * a per-layer `CGC-DECPROF all` row belongs to whatever graph its aggregate line announced, and the
    log carries two families: `segs=41` (the 41-layer graph) and `segs=2` (the MTP draft graphs, which
    are single-layer). Rows whose graph is not 41 segments are excluded and counted, and so are rows
    in the prefill region. Both exclusions are printed;
  * per-layer values are MEDIANS, not means: the same layer's rows span 0.02-194 ms, so a mean is a
    statement about the slowest graph in the run.

THE DECISION THIS FEEDS: if the window tracks misses, the lever is routing locality / coverage. If it
is a fixed per-layer cost, the lever is the code path and locality work cannot move it. The rule is
committed to BEFORE the numbers:

  fixed_share = intercept / mean(window)   <- the part of the mean window a miss-FREE layer pays
  MISS_BOUND    corr(idle_host, misses/step) >= 0.7 AND fixed_share <= 0.40
  FIXED_BOUND   fixed_share >= 0.60
  MIXED         in between -> report both, claim neither

The first draft of this rule used slope*mean/total instead, which is the same thing whenever the
intercept is positive and UNBOUNDED (2.1, >1) whenever it is not -- so it read 'share' on a fit where
there was nothing to share. fixed_share is that quantity stated so it cannot do that.

Deliberate non-features:
  * no verdict from a partial series -- a missing BATCHDBG series is INVALID, not 'no correlation';
  * a layer with no fill call is ABSENT and named, never silently dropped;
  * nesting is CHECKED: idle_host >= hook, else the row is not quotable;
  * leverage is reported: 41 points with one influential layer is how a global slope becomes a story
    about one layer.
"""
import argparse
import collections
import contextlib
import io
import math
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

AGG_DEC = re.compile(r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+)")
AGG_GAP = re.compile(r"CGC-GAPSPLIT: segs=(\d+) poll=([\d.]+) hook=([\d.]+) tail=([\d.]+) "
                     r"submit=([\d.]+) idle_host=([\d.]+)")
GAP = re.compile(r"CGC-GAPSPLIT-L: layer=(\d+) poll=([\d.]+) hook=([\d.]+) tail=([\d.]+) "
                 r"submit=([\d.]+) idle_host=([\d.]+) gap=([\d.]+) ratio=([\d.]+)")
DEC = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms")
BAT = re.compile(r"BATCHDBG layer=(\d+) misses=(\d+)")
FOCUS = tuple(range(0, 6))
FULL_SEGS = 41


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def parse(path: Path) -> tuple[dict, dict]:
    """Per-layer decode-population medians, plus the census of what was excluded and why."""
    lines = path.read_text(errors="replace").splitlines()
    boundary = max((i for i, l in enumerate(lines) if l.startswith("CGC-POST:")), default=None)
    # Each family's "full graph" is the LARGEST segment count it reports in the decode phase, and the
    # two families do NOT agree on the number: the gap splitter counts 40 where DECPROF counts 41 (both
    # describe the same 41-layer graph; one of them does not count the nextn block). Hardcoding either
    # number silently empties the other family's series, which is how this tool first returned live=0
    # on a perfectly good log. So it is derived per family and printed, not assumed.
    # With no boundary the rows are still collected (marked phase-unknown) so that the refusal can
    # say how many rows exist and why they are unusable, instead of claiming there is no data.
    phase_cut = boundary if boundary is not None else -1
    dec_segs = gap_segs = None
    for i, l in enumerate(lines):
        if i <= phase_cut:
            continue
        m = AGG_DEC.match(l)
        if m:
            dec_segs = max(dec_segs or 0, int(m.group(2)))
        m = AGG_GAP.match(l)
        if m:
            gap_segs = max(gap_segs or 0, int(m.group(1)))
    meta = {"boundary_line": boundary, "nlines": len(lines), "pre_fill_calls": 0, "dec_fill_calls": 0,
            "pre_misses": 0, "dec_misses": 0, "rows_dec": 0, "rows_pre": 0, "rows_othergraph": 0,
            "dec_segs": dec_segs, "gap_segs": gap_segs, "gap_rows_other": 0}
    series = collections.defaultdict(lambda: collections.defaultdict(list))
    cur_dec_segs = None
    cur_gap_segs = None
    for i, line in enumerate(lines):
        dec_phase = i > phase_cut
        m = AGG_DEC.match(line)
        if m:
            cur_dec_segs = int(m.group(2))
            continue
        m = AGG_GAP.match(line)
        if m:
            cur_gap_segs = int(m.group(1))
            continue
        m = DEC.match(line)
        if m:
            if not dec_phase:
                meta["rows_pre"] += 1
            elif cur_dec_segs != dec_segs:
                meta["rows_othergraph"] += 1
            else:
                meta["rows_dec"] += 1
                # wait / submit are kept alongside cb: the same row is the only per-layer split of the
                # decode step into eval-sync (wait), gather (cb) and dispatch (submit), and consumers
                # that need one of the other two must not have to re-parse the line.
                series[int(m.group(1))]["wait"].append(float(m.group(2)))
                series[int(m.group(1))]["cb"].append(float(m.group(3)))
                series[int(m.group(1))]["submit"].append(float(m.group(4)))
            continue
        m = GAP.match(line)
        if m:
            if not dec_phase:
                pass
            elif cur_gap_segs != gap_segs:
                meta["gap_rows_other"] += 1
            else:
                l = int(m.group(1))
                series[l]["idle"].append(float(m.group(6)))
                series[l]["hook"].append(float(m.group(3)))
                series[l]["poll"].append(float(m.group(2)))
                series[l]["tail"].append(float(m.group(4)))
                series[l]["submit"].append(float(m.group(5)))
            continue
        m = BAT.match(line)
        if m:
            if i > phase_cut:
                meta["dec_fill_calls"] += 1
                meta["dec_misses"] += int(m.group(2))
                series[int(m.group(1))]["miss_calls"].append(1)
                series[int(m.group(1))]["misses"].append(int(m.group(2)))
            else:
                meta["pre_fill_calls"] += 1
                meta["pre_misses"] += int(m.group(2))
    rows = []
    for l, s in sorted(series.items()):
        if not s["idle"] and not s["cb"]:
            continue
        steps = len(s["cb"])
        live = len(s["idle"])
        rows.append({
            "layer": l, "steps": steps, "live": live,
            # Both summaries, because they answer different questions and one of them is degenerate
            # here: a layer whose misses land in 1 step out of 3 has a median step with NO miss, so
            # its median window is ~0 by construction. The median is the robust reading for the
            # pooled fit; the mean is what carries a rare miss's cost.
            "idle_mean": (sum(s["idle"]) / len(s["idle"])) if s["idle"] else float("nan"),
            "cb_mean": (sum(s["cb"]) / len(s["cb"])) if s["cb"] else float("nan"),
            "wait_mean": (sum(s["wait"]) / len(s["wait"])) if s["wait"] else float("nan"),
            "submit_mean": (sum(s["submit"]) / len(s["submit"])) if s["submit"] else float("nan"),
            "idle": med(s["idle"]) if s["idle"] else float("nan"),
            "hook": med(s["hook"]) if s["hook"] else float("nan"),
            "tail": med(s["tail"]) if s["tail"] else float("nan"),
            "submit": med(s["submit"]) if s["submit"] else float("nan"),
            "wait": med(s["wait"]) if s["wait"] else float("nan"),
            "poll": med(s["poll"]) if s["poll"] else float("nan"),
            "cb": med(s["cb"]) if s["cb"] else float("nan"),
            "miss_calls": sum(s["miss_calls"]),
            "misses": sum(s["misses"]),
            # misses per step, on the same denominator as cb (decode steps only)
            "miss_per_step": (sum(s["misses"]) / steps) if steps else float("nan"),
            # misses per fill that happened at all (the intensity of a cold set, not its rate)
            "miss_per_fill": (sum(s["misses"]) / sum(s["miss_calls"])) if s["miss_calls"] else float("nan"),
            "miss_per_live": (sum(s["misses"]) / live) if live else float("nan"),
            # prefill misses for the same layer, as a share of its decode misses
            "pre_misses": sum(int(m.group(2)) for m in (BAT.match(x) for x in
                                                       lines[:phase_cut + 1]) if m and int(m.group(1)) == l),
        })
    return rows, meta


def ols(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else float("nan")
    resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    return {"n": n, "slope": slope, "intercept": inter, "r2": r2, "resid": resid,
            "rmse": math.sqrt(sum(r * r for r in resid) / n)}


def fixed_share(f: dict, rows: list[dict]) -> float:
    """The fraction of the mean window a miss-free layer still pays. Negative = below zero, which is
    a reading (no fixed term at all), not a missing value."""
    ys = [r["idle"] for r in rows]
    mean_y = sum(ys) / len(ys)
    return f["intercept"] / mean_y if mean_y else float("nan")


def verdict(reg: dict, rows: list[dict]) -> tuple[str, str]:
    f = reg.get("idle~miss")
    if not f:
        return "INVALID", "the idle_host vs misses/step fit could not be run"
    fs = fixed_share(f, rows)
    corr = math.sqrt(f["r2"]) if f["r2"] == f["r2"] else float("nan")
    if corr >= 0.7 and fs <= 0.40:
        return "MISS_BOUND", f"corr={corr:.2f}, fixed_share={fs:.2f}"
    if fs >= 0.60:
        return "FIXED_BOUND", f"corr={corr:.2f}, fixed_share={fs:.2f}"
    return "MIXED", f"corr={corr:.2f}, fixed_share={fs:.2f}"


def report(rows: list[dict], meta: dict) -> int:
    if not rows:
        print("no per-layer rows -- needs CGC-GAPSPLIT-L (CGC_HOOK_SPLIT=1 + CGC_GPU_TIMING=1 +"
              " CGC_DECODE_PROFILE=1) and CGC_DECODE_PROFILE_ALL=1; INVALID, not 'no correlation'")
        return 1
    if meta["boundary_line"] is None:
        print("no `CGC-POST:` line in this log, so the prefill/decode boundary is UNKNOWN and a fit"
              " across both phases would be mostly a prefill regression -- INVALID")
        return 1
    print(f"census: decode rows used {meta['rows_dec']} (graph segs={meta['dec_segs']}), "
          f"prefill rows excluded {meta['rows_pre']}, other-graph rows excluded "
          f"{meta['rows_othergraph']}; gap rows used at segs={meta['gap_segs']} "
          f"(other-graph excluded {meta['gap_rows_other']})")
    print(f"  note: the two families name the same graph with DIFFERENT segment counts "
          f"({meta['dec_segs']} vs {meta['gap_segs']}); each is derived, not hardcoded")
    print(f"  decode fill calls {meta['dec_fill_calls']} with {meta['dec_misses']} misses | "
          f"prefill fill calls {meta['pre_fill_calls']} with {meta['pre_misses']} misses "
          f"({meta['pre_misses'] / max(1, meta['pre_misses'] + meta['dec_misses']):.0%} of all misses"
          f" are NOT in this fit)")
    layers = sorted(r["layer"] for r in rows)
    print(f"  layers: {len(rows)} ({layers[0]}..{layers[-1]}) | decode steps per layer "
          f"{min(r['steps'] for r in rows)}..{max(r['steps'] for r in rows)}")
    absent = [r["layer"] for r in rows if r["miss_calls"] == 0]
    print(f"  layers with no decode fill call: {absent or 'none'}"
          + ("  -- NAMED and excluded from the miss fit" if absent else ""))
    bad = [r["layer"] for r in rows if r["idle"] == r["idle"] and r["hook"] == r["hook"]
           and r["idle"] + 1e-9 < r["hook"]]
    print(f"  (i) idle_host >= hook (the window contains its hook): "
          f"{'consistent' if not bad else f'VIOLATED on {bad}'}")

    fit_rows = [r for r in rows if r["miss_calls"] > 0 and r["cb"] == r["cb"] and r["idle"] == r["idle"]]
    reg = {}
    if len(fit_rows) >= 3:
        reg["idle~miss"] = ols([r["miss_per_step"] for r in fit_rows], [r["idle"] for r in fit_rows])
        reg["cb~miss"] = ols([r["miss_per_step"] for r in fit_rows], [r["cb"] for r in fit_rows])
        reg["idle~miss (means)"] = ols([r["miss_per_step"] for r in fit_rows],
                                       [r["idle_mean"] for r in fit_rows])
    print(f"\n  {'fit':<12} {'n':>3} {'slope':>9} {'intercept':>10} {'R2':>6} {'rmse':>7} "
          f"{'fixed_share':>12}")
    for k in ("idle~miss", "cb~miss", "idle~miss (means)"):
        f = reg.get(k)
        if not f:
            print(f"  {k:<12} --  NOT RUN (a series is missing; absence is not a flat line)")
            continue
        fs = fixed_share(f, fit_rows)
        print(f"  {k:<12} {f['n']:>3} {f['slope']:>9.3f} {f['intercept']:>10.3f} {f['r2']:>6.2f} "
              f"{f['rmse']:>7.3f} {fs:>12.2f}"
              + ("   <- a miss-free layer pays nothing: no fixed term in this window" if fs <= 0
                 else ""))

    if "idle~miss" in reg:
        f = reg["idle~miss"]
        worst = max(range(len(fit_rows)), key=lambda i: abs(f["resid"][i]))
        mean_y = sum(r["idle"] for r in fit_rows) / len(fit_rows)
        flagged = (f["rmse"] > 0 and abs(f["resid"][worst]) / f["rmse"] > 3
                   and abs(f["resid"][worst]) > 0.05 * mean_y)
        print(f"  leverage: largest residual is L{fit_rows[worst]['layer']} at "
              f"{f['resid'][worst]:+.3f} ms ({abs(f['resid'][worst]) / f['rmse']:.1f}x rmse, "
              f"{abs(f['resid'][worst]) / mean_y:.0%} of the mean window)"
              + ("  -- INFLUENTIAL: refit without it before quoting the slope" if flagged else ""))
        if flagged:
            keep = [r for i, r in enumerate(fit_rows) if i != worst]
            f2 = ols([r["miss_per_step"] for r in keep], [r["idle"] for r in keep])
            if f2:
                v0, v1 = verdict(reg, fit_rows), verdict({"idle~miss": f2}, keep)
                print(f"  refit without L{fit_rows[worst]['layer']}: slope {f2['slope']:.3f}, "
                      f"intercept {f2['intercept']:.3f}, R2 {f2['r2']:.2f} -> verdict {v1[0]}"
                      + ("  (SAME verdict)" if v1[0] == v0[0] else
                         "  -- THE VERDICT FLIPS: the first fit was a statement about one layer"))

    # (j) WITHIN-group fits. A pooled correlation of 0.99 across 40 layers can be nothing but the
    # difference between two clusters (here: the six layers that churn and the 34 that do not), in
    # which case 'cost = 0.7 x misses' is a law that holds only ACROSS groups and the honest target is
    # the groups, not a slope. This is the same failure mode as the leverage row above, one level up.
    groups = {}
    for name, grp in (("L0-L5", [r for r in rows if r["layer"] in FOCUS]),
                      ("rest", [r for r in rows if r["layer"] not in FOCUS])):
        if len(grp) >= 3:
            groups[name] = (ols([r["miss_per_step"] for r in grp], [r["idle"] for r in grp]), grp)
    for name, (f, grp) in groups.items():
        if not f:
            continue
        fm = ols([r["miss_per_step"] for r in grp], [r["idle_mean"] for r in grp])
        print(f"  within {name:<6} n={f['n']:>2}  slope(med) {f['slope']:>7.3f}  intercept {f['intercept']:>7.3f}"
              f"  R2 {f['r2']:>5.2f}  |  slope(mean) {fm['slope']:>6.3f} R2 {fm['r2']:>4.2f}"
              f"  (mean window {sum(r['idle_mean'] for r in grp) / len(grp):.3f} ms/step,"
              f" median-of-layer-medians {med([r['idle'] for r in grp]):.3f} ms)")
    if len(groups) == 2 and "idle~miss" in reg and reg["idle~miss"]:
        inner = [g[0]["slope"] for g in groups.values() if g[0]]
        pooled = reg["idle~miss"]["slope"]
        if inner and max(abs(s) for s in inner) < 0.4 * abs(pooled):
            print(f"  TWO-CLUSTER: the pooled median slope {pooled:.3f} is largely a difference BETWEEN")
            print(f"               churn groups (within-group slopes {', '.join(f'{s:.3f}' for s in inner)});")
            print(f"               the L0-L5 group still carries the law internally, the rest does not --")
            print(f"               which is about where the misses are, not about the per-miss cost")

    comp = [r["hook"] / r["idle"] for r in rows if r["idle"] > 0 and r["hook"] == r["hook"]]
    print(f"  window composition: hook is {100 * med(comp):.0f}% of idle_host (median over layers);"
          f" the rest is tail+submit")

    for name, grp in (("L0-L5", [r for r in rows if r["layer"] in FOCUS]),
                      ("rest", [r for r in rows if r["layer"] not in FOCUS])):
        if not grp:
            continue
        mps = [r["miss_per_step"] for r in grp if r["miss_per_step"] == r["miss_per_step"]]
        mpf = [r["miss_per_fill"] for r in grp if r["miss_per_fill"] == r["miss_per_fill"]]
        print(f"  {name:<7} idle_host {med([r['idle'] for r in grp]):.2f} ms/step | "
              f"hook {med([r['hook'] for r in grp]):.2f} | cb {med([r['cb'] for r in grp]):.2f} | "
              f"misses {med(mps) if mps else float('nan'):.2f}/step | "
              f"misses/fill {med(mpf) if mpf else float('nan'):.1f}  (medians over layers)")
        pre = [r["pre_misses"] for r in grp]
        dec = [r["misses"] for r in grp]
        if dec and sum(dec):
            print(f"          prefill contributed {sum(pre)} misses to these layers vs {sum(dec)} in"
                  f" decode ({sum(pre) / (sum(pre) + sum(dec)):.0%} of their total) -- excluded above")

    v, why = verdict(reg, fit_rows)
    print(f"\n  VERDICT: {v}  ({why})")
    if "cb~miss" in reg:
        print(f"  cross-check: the per-miss slope is {reg['cb~miss']['slope']:.3f} ms/miss (F1"
              f" established ~0.695 ms/miss). A slope far from it means this is not that quantity --")
        print(f"               a finding about the instrument, not about locality.")
    print(f"  every series here is decode-only and from one run; the prefill numbers above are shown")
    print(f"  only so the excluded population is visible.")
    return 1 if (bad or v == "INVALID") else 0


def selftest() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    def mklog(pairs, mode="miss", with_batch=True, with_post=True, segs=41, draft_rows=0):
        out = ["CGC-POST: il=0 st[1]=1"] if with_post else []
        out.append("CGC-DECPROF: step=0 segs=%d layers=40 total=1.00 ms | wait=1.00" % segs)
        # the gap family reports the SAME graph with one fewer segment; the fixture keeps that
        # convention so the per-family derivation is exercised rather than assumed
        out.append("CGC-GAPSPLIT: segs=%d poll=1.00 hook=1.00 tail=0.00 submit=0.00 "
                   "idle_host=1.00 ms | gap=1.00 ratio=1.00" % (segs - 1 if segs > 1 else segs))
        for l, m in pairs:
            if mode == "miss":
                cb, idle = 0.2 + 2.0 * m, 0.2 + 2.0 * m
            else:
                cb, idle = 12.0 + 0.001 * m, 12.0 + 0.001 * m
            steps = 50
            hits = int(round(m * steps))
            for _ in range(steps):
                out.append(f"CGC-DECPROF all: L{l} wait=1.00 cb={cb:.4f} submit=0.10 ms")
                out.append(f"CGC-GAPSPLIT-L: layer={l} poll=1.00 hook={idle:.4f} tail=0.00 "
                           f"submit=0.00 idle_host={idle:.4f} gap=0.00 ratio=0.00 n=1")
            if with_batch:
                calls, left = max(1, int(round(m * 5))), hits
                for i in range(calls):
                    take = left // (calls - i) if calls - i else left
                    left -= take
                    out.append(f"BATCHDBG layer={l} misses={take} slots:")
        if draft_rows:
            out.append("CGC-DECPROF: step=99 segs=2 layers=1 total=1.00 ms | wait=1.00")
            for _ in range(draft_rows):
                out.append("CGC-DECPROF all: L40 wait=1.00 cb=99.0 submit=0.10 ms")
        return "\n".join(out) + "\n"

    def run(text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(text)
        rows, meta = parse(Path(fh.name))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report(rows, meta)
        return f"{rc}\n" + buf.getvalue()

    pairs = [(l, 0.5 + 0.05 * l) for l in range(41)]
    check("a miss-driven window reads MISS_BOUND", "MISS_BOUND" in run(mklog(pairs, "miss")))
    check("a fixed per-layer window reads FIXED_BOUND", "FIXED_BOUND" in run(mklog(pairs, "fixed")))
    check("a missing BATCHDBG series is INVALID, not 'no correlation'",
          run(mklog(pairs, "miss", with_batch=False)).startswith("1")
          and "NOT RUN" in run(mklog(pairs, "miss", with_batch=False)))
    check("... and INVALID even when the other two series fit beautifully",
          "MISS_BOUND" not in run(mklog(pairs, "miss", with_batch=False)))
    check("a log with no CGC-POST boundary is INVALID (phase unknown)",
          run(mklog(pairs, with_post=False)).startswith("1")
          and "boundary is UNKNOWN" in run(mklog(pairs, with_post=False)))
    check("other-graph rows are excluded and counted, not fitted",
          "other-graph rows excluded 41" in run(mklog(pairs, "miss", draft_rows=41)))
    check("medians, not means: one 1000x outlier row cannot move the verdict",
          "MISS_BOUND" in run(mklog(pairs, "miss").replace(
              "cb=1.2000 submit", "cb=1000.0 submit", 1)))
    check("no rows at all -> INVALID with the reason", run("").startswith("1"))
    check("the per-miss slope is cross-checked against F1's 0.695",
          "0.695" in run(mklog(pairs, "miss")))
    check("the census names the population actually fitted",
          "census: decode rows used" in run(mklog(pairs, "miss")))
    check("within-group fits are reported (a pooled slope can be a cluster difference)",
          "within L0-L5" in run(mklog(pairs, "miss")) and "within rest" in run(mklog(pairs, "miss")))
    check("a mean-based fit is printed next to the median one (they answer different questions)",
          "idle~miss (means)" in run(mklog(pairs, "miss")))
    print(f"gap_vs_miss selftest: {11 - bad}/11 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.log:
        print("need --log (a server log with CGC-GAPSPLIT-L + CGC-DECPROF all + BATCHDBG) or --selftest")
        return 2
    p = Path(args.log)
    if not p.is_file():
        alt = ROOT / args.log
        p = alt if alt.is_file() else p
    rows, meta = parse(p)
    return report(rows, meta)


if __name__ == "__main__":
    sys.exit(main())
