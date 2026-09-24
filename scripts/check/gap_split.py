#!/usr/bin/env python3
"""Read CGC-GAPSPLIT: is the decode step's GPU-idle window work or dependency?

THE QUESTION. The step's GPU clock reports a `gap` (idle from the previous segment's end to this
one's start). Two mutually exclusive readings demand opposite actions:

  host work      -> the dispatcher could hide it (line A's segment-split submit)
  dependency     -> it cannot be hidden at all: the ids the hook needs are the GPU's own argsort
                    output, so this window exists because the host is *waiting*, not working

THE INSTRUMENT. `CGC-GAPSPLIT` (add it with `CGC_HOOK_SPLIT=1`, and `CGC_GPU_TIMING=1` if you also
want `gap`; the line says `gap=unmeasured` when that is off, because absence is not zero):
    poll      host blocked on segment i's completion  == dependency
    hook      the top-k callback: ids + union + pool fill (see CGC-EBSPLIT / CGC-FSSPLIT)
    tail      hook bookkeeping after the callback
    submit    encode+commit of segment i+1 (the existing dp_last_submit_us timer, reused)
    idle_host hook + tail + submit  -- what the host executes while the GPU has nothing to do
    ratio     idle_host / gap

THE DECISION RULE (pre-registered here so the reading cannot be rationalised afterwards):
    ratio >= 0.80  -> the idle window is host work; L3 has a landing spot
    ratio <= 0.40  -> it is dependency/pipeline; L3 cannot reach the union floor
    in between     -> report the split, claim nothing
and `gap - idle_host` is cross-instrument evidence: negative means the two instruments disagree and
neither number may be quoted.

NOT a verdict: `poll` is reported but never counted as idle time -- it is the window in which the GPU
is producing the ids.
"""
import argparse
import contextlib
import io
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(
    r"CGC-GAPSPLIT:\s+segs=(\d+)\s+poll=([\d.]+)\s+hook=([\d.]+)\s+tail=([\d.]+)\s+submit=([\d.]+)\s+"
    r"idle_host=([\d.]+) ms \| gap=(unmeasured[^|]*|[\d.]+)(?:\s+ratio=([\d.]+))?")
LAYER = re.compile(
    r"CGC-GAPSPLIT-L:\s+layer=(\d+)\s+poll=([\d.]+)\s+hook=([\d.]+)\s+tail=([\d.]+)\s+"
    r"submit=([\d.]+)\s+idle_host=([\d.]+)\s+gap=([\d.]+)\s+ratio=([\d.]+)\s+n=(\d+)")
PARTIAL = re.compile(r"CGC-GAPSPLIT-L:\s+(\d+) of (\d+) layers shown")
WORK, DEP = 0.80, 0.40
# Concentration rule, pre-registered. A uniform step over 41 layers puts 8/41 = 19.5% of the idle
# window in the top 8 rows, so that is the baseline and not a threshold: a share near it means the
# problem is the per-segment round trip, a share well above it means a few layers own the window and
# the work is in those layers' hooks.
SPREAD_MAX, CONC_MIN = 0.25, 0.40


def parse(path: Path) -> list[dict]:
    out = []
    for m in LINE.finditer(path.read_text(errors="replace")):
        gap = None if m.group(7).startswith("unmeasured") else float(m.group(7))
        out.append({"segs": int(m.group(1)), "poll": float(m.group(2)), "hook": float(m.group(3)),
                    "tail": float(m.group(4)), "submit": float(m.group(5)),
                    "idle_host": float(m.group(6)), "gap": gap})
    return out


def parse_layers(path: Path) -> tuple[list[dict], str]:
    """Per-layer rows plus a note about whether the log holds ALL of them.

    The engine prints the top 8 by idle_host unless CGC_DECODE_PROFILE_ALL=1, so a log without that
    switch cannot support a share-of-total claim. The note is returned rather than a silent 41.
    """
    txt = path.read_text(errors="replace")
    rows = []
    for m in LAYER.finditer(txt):
        rows.append({"layer": int(m.group(1)), "poll": float(m.group(2)), "hook": float(m.group(3)),
                     "tail": float(m.group(4)), "submit": float(m.group(5)),
                     "idle_host": float(m.group(6)), "gap": float(m.group(7)),
                     "ratio": float(m.group(8)), "n": int(m.group(9))})
    partial = PARTIAL.search(txt)
    note = ("partial: only the top rows were printed, so the share below is a lower bound"
            if partial else "complete")
    return rows, note


def layer_report(rows: list[dict], note: str, agg_sum: float, agg_hook: float = 0.0) -> int:
    """Concentration verdict: does the idle window live in a few layers or in the round trip?"""
    if not rows:
        print("CGC-GAPSPLIT-L: no per-layer rows (the line needs CGC_DECODE_PROFILE=1) -> the "
              "spread/concentrated question is NOT answered")
        return 1
    by: dict[int, dict] = {}
    for r in rows:
        a = by.setdefault(r["layer"], {"idle_host": 0.0, "gap": 0.0, "poll": 0.0, "hook": 0.0,
                                          "n": 0})
        a["idle_host"] += r["idle_host"]
        a["gap"] += r["gap"]
        a["poll"] += r["poll"]
        a["hook"] += r["hook"]
        a["n"] += r["n"]
    # Same units on both sides: `total` sums every graph, and so does `top8`. The first version
    # divided by the AGGREGATE MEDIAN -- a single per-step number -- and reported a 112% share, which
    # is not a share of anything. The two views are still cross-checked, because a per-layer table
    # whose total disagrees with the aggregate line means the instrument, not the box, is talking.
    total = sum(a["idle_host"] for a in by.values())
    hook_l = sum(a["hook"] for a in by.values())
    gap_l = sum(a["gap"] for a in by.values())
    seen = (agg_sum / total) if (agg_sum > 0 and total > 0) else None
    seen_h = (agg_hook / hook_l) if (agg_hook > 0 and hook_l > 0) else None
    denom = total
    top = sorted(by.items(), key=lambda kv: -kv[1]["idle_host"])
    top8 = sum(a["idle_host"] for _, a in top[:8])
    share = top8 / denom if denom > 0 else 0.0
    print(f"  per layer: {len(by)} layers with rows, {note}; top-8 idle_host share = {share:.1%} "
          f"(uniform baseline for 41 layers = 19.5%)")
    print(f"  cross-check: per-layer total {total:.0f} ms vs the aggregate rows' total {agg_sum:.0f} ms"
          + (f"  ({seen:.2f}x)" if seen else "")
          + (f"; hook {hook_l:.0f} vs {agg_hook:.0f} ({seen_h:.2f}x)" if seen_h else "")
          + ("" if seen is None or 0.8 <= seen <= 1.2 else
             "  -- same quantity, different totals: the two views disagree, so neither is quotable"))
    if seen is not None and seen_h is not None and not (0.8 <= seen <= 1.2) and 0.98 <= seen_h <= 1.02:
        print("  ... the hook columns agree, so the difference is in the submit attribution"
              " (dp_lay_sub), not in the measured window.")
    # The GPU side per layer is a KNOWN gap: every row prints gap=0.00 while the aggregate has one.
    # Printing ratio 0.00 would be the absence-as-a-value mistake, so the ratio is named as absent.
    gaps_absent = gap_l == 0.0 and len(by) > 1
    if gaps_absent:
        print("  per-layer GPU side: NOT MEASURED (every row says gap=0.00 while the aggregate has a "
              "gap) -> the ratio column below is an absence, not a zero, and the per-layer split "
              "answers 'where is the host work' only.")
    for l, a in top[:10]:
        print(f"    layer {l:>3}: idle_host {a['idle_host']:8.2f}  gap {a['gap']:8.2f}  "
              f"ratio {(a['idle_host'] / a['gap']) if a['gap'] > 0 else '-':>5}  n {a['n']}")
    if note.startswith("partial"):
        print("VERDICT(spread): NOT COMPUTABLE from this log -- re-run with "
              "CGC_DECODE_PROFILE_ALL=1 so the denominator is all 41 layers")
        return 1
    if share >= CONC_MIN:
        print(f"VERDICT: CONCENTRATED ({share:.0%} of the idle window in 8 of {len(by)} layers) -> fix "
              f"those layers' hooks, not the round trip")
    elif share <= SPREAD_MAX:
        print(f"VERDICT: SPREAD ({share:.0%} in the top 8, near the 19.5% uniform baseline) -> the "
              f"window is per-segment structure; a per-layer hook change cannot remove it")
    else:
        print(f"VERDICT: MIXED ({share:.0%} in the top 8) -> report the table, claim neither")
    return 0


def report(rows: list[dict], layers=None, note: str = "complete") -> int:
    if not rows:
        print("INVALID: no CGC-GAPSPLIT line in this log -- nothing was measured. "
              "Arm CGC_HOOK_SPLIT=1 (the instrument prints nothing otherwise).")
        return 1
    med = lambda k: st.median([r[k] for r in rows if r[k] is not None])
    print(f"CGC-GAPSPLIT rows n={len(rows)}  (medians, ms)")
    for k in ("poll", "hook", "tail", "submit", "idle_host"):
        print(f"  {k:>9}: {med(k):8.2f}")
    measured = [r for r in rows if r["gap"] is not None]
    if not measured:
        print("  gap      : unmeasured (CGC_GPU_TIMING was off) -> the movable/not question cannot be "
              "answered from this log; re-run with CGC_GPU_TIMING=1")
        return 1
    gap = st.median([r["gap"] for r in measured])
    idle = med("idle_host")
    ratio = idle / gap if gap > 0 else 0.0
    resid = gap - idle
    print(f"  gap      : {gap:8.2f}   (rows with a gap: {len(measured)}/{len(rows)})")
    print(f"  ratio idle_host/gap = {ratio:.2f}   residual gap-idle_host = {resid:+.2f} ms")
    print()
    if resid < 0:
        print("VERDICT: INSTRUMENT DISAGREEMENT -- the host accounting exceeds the GPU-clock idle "
              "window. Neither number may be quoted until that is explained (a clock-domain or a "
              "segment-count mismatch).")
        return 1
    # "Where" is asked in every regime, not only in the host-work one: a MIXED verdict still needs to
    # know whether the window is spread over 41 segments or owned by a few layers.
    if layers is not None:
        layer_report(layers, note, sum(r["idle_host"] for r in rows),
                     sum(r["hook"] for r in rows))
    if ratio >= WORK:
        print(f"VERDICT: the idle window is HOST WORK ({ratio:.0%} of gap is hook+tail+submit). "
              f"L3 has a landing spot: the dispatcher can in principle hide it. Duration is bounded "
              f"below by union_sum, not by this ratio -- check CGC-DECPROF union_sum in the same log.")
    elif ratio <= DEP:
        print(f"VERDICT: the idle window is DEPENDENCY/PIPELINE ({ratio:.0%} host, "
              f"{resid:.2f} ms unexplained by host work). L3 cannot reach the union floor: the "
              f"window exists because the host is waiting for the GPU's own output.")
    else:
        print(f"VERDICT: MIXED ({ratio:.0%} host work, {resid:.2f} ms other). Report the split; the "
              f"decision needs a second regime (another pool size or MTP off) before claiming one.")
    return 0


def synth(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        gap = "unmeasured (CGC_GPU_TIMING off)" if r["gap"] is None else \
            f"{r['gap']:.2f} ratio={r['idle_host'] / r['gap']:.2f}"
        lines.append(f"CGC-GAPSPLIT: segs={r['segs']} poll={r['poll']:.2f} hook={r['hook']:.2f} "
                     f"tail={r['tail']:.2f} submit={r['submit']:.2f} idle_host={r['idle_host']:.2f} "
                     f"ms | gap={gap}")
    return "\n".join(lines) + "\n"


def selftest() -> int:
    import tempfile
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    def run(text: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report(parse(Path(fh.name)))
        return f"{rc}\n" + buf.getvalue()

    row = lambda **kw: {"segs": 41, "poll": 160.0, "hook": 74.0, "tail": 2.0, "submit": 10.0,
                        "idle_host": 86.0, "gap": 95.0} | kw
    check("a missing instrument is INVALID, not a pass", run("no lines here\n").startswith("1"))
    check("host-work gap is called host work",
          "HOST WORK" in run(synth([row()] * 3)))
    check("a dependency gap is called dependency",
          "DEPENDENCY" in run(synth([row(hook=2.0, tail=0.5, submit=2.0, idle_host=4.5)] * 3)))
    check("gap=unmeasured is refused, not read as zero",
          run(synth([row(gap=None)] * 3)).startswith("1"))
    check("host > gap is reported as instrument disagreement",
          "DISAGREEMENT" in run(synth([row(idle_host=120.0)] * 3)))
    check("the ratio is idle_host/gap", abs(86.0 / 95.0 - 0.905) < 0.002)

    def run_layers(rows_l: list[dict], partial: bool = False, agg_scale: float = 1.0) -> str:
        text = "".join(
            f"CGC-GAPSPLIT-L: layer={r['layer']} poll={r['poll']:.2f} hook={r['hook']:.2f} "
            f"tail={r['tail']:.2f} submit={r['submit']:.2f} idle_host={r['idle_host']:.2f} "
            f"gap={r['gap']:.2f} ratio={r['ratio']:.2f} n={r['n']}\n" for r in rows_l)
        if partial:
            text += "CGC-GAPSPLIT-L: 8 of 41 layers shown (top by idle_host)\n"
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(text)
        got, nt = parse_layers(Path(fh.name))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = layer_report(got, nt, agg_sum=agg_scale * sum(r["idle_host"] for r in rows_l),
                              agg_hook=sum(r["hook"] for r in rows_l))
        return f"{rc}\n" + buf.getvalue()

    # The denominator has to be the aggregate ROWS' total, same units as the table's own total. The
    # first version passed the per-step median (x 41) and printed a 112% share; this asserts the
    # guard that now catches it.
    unif = [{"layer": l, "poll": 1.0, "hook": 2.0, "tail": 0.2, "submit": 0.4,
             "idle_host": 2.6, "gap": 3.0, "ratio": 0.87, "n": 10} for l in range(41)]
    conc = [dict(unif[0], layer=l, idle_host=0.0, gap=0.0) for l in range(41)]
    for l in range(8):
        conc[l] = dict(conc[l], idle_host=8.0, gap=9.0, hook=7.0)
    check("a uniform table is called SPREAD, not concentrated", "SPREAD" in run_layers(unif))
    check("a top-heavy table is called CONCENTRATED", "CONCENTRATED" in run_layers(conc))
    check("an aggregate total that disagrees with the table is called out, not divided by",
          "the two views disagree" in run_layers(unif, agg_scale=3.0))
    zero_gap = [dict(r, gap=0.0) for r in unif]
    zg = run_layers(zero_gap)
    check("an unmeasured per-layer GPU side is printed as absent, not as ratio 0.00",
          "NOT MEASURED" in zg and "ratio     0.00" not in zg)
    check("a top-8-only log refuses the share verdict",
          run_layers(unif[:8], partial=True).startswith("1"))
    check("no per-layer rows is INVALID, not SPREAD", run_layers([]).startswith("1"))
    print(f"gap_split selftest: {10 - bad}/10 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.log:
        print("need --log (a server log containing CGC-GAPSPLIT lines) or --selftest")
        return 2
    p = Path(args.log)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"no such log: {p}")
        return 2
    layers, note = parse_layers(p)
    return report(parse(p), layers, note)


if __name__ == "__main__":
    sys.exit(main())
