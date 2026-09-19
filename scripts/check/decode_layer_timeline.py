#!/usr/bin/env python3
"""G3's reading: put the per-layer GPU intervals on one timeline and say whether they overlap.

The instrument (ggml-backend.cpp, `st=`/`en=` on each `CGC-DECPROF all:` line) reports every layer's
interval relative to the earliest segment start in that step. Durations alone cannot answer the
question G3 asks, because two 2.81 ms spans look identical whether they ran back-to-back or on top of
each other; intervals can.

WHAT IT COMPUTES, and what each number decides:

  * `span(all) / union_sum`  -- the parallelism actually achieved.
        ~1.00  => the 40 layers are strictly serial, so the gap is recoverable idle time and
                  G1 (hide the gap) is the whole story;
        <<1.00 => layers ALREADY overlap, so `union_sum` cannot be reduced by any schedule and G4
                  would have to come from making each layer faster.

  * the 39 inter-layer windows `st[i+1] - en[i]`, summed, vs the header's `gap_sum`.
        Their sum should reproduce `gap_sum` (both measure "GPU idle between segments"); if it does
        not, one of the two cuts is wrong and the difference is exactly what to investigate.

  * per pair (i, i+1): `en[i+1] - st[i]` vs `2 x union`. This is G3's gate metric measured WITHOUT
        a pair-merge arm: if the pair's combined elapsed span already exceeds 2 x a single layer,
        the excess IS the idle a merge could remove, and its size says whether merging is worth it.

Discipline: only `layers=40` headers; requests separated on prefill steps; per-request medians, never
a pooled one (one degraded request moves a pooled median for every layer at once).

Usage:
  python3 Backup/g3_timeline_20260919.py <server.log> [--drop-requests 1] [--csv out.csv]
"""
import argparse
import re
import statistics
import sys

LAYERS = 40
PREFILL_NTOK = 100
DECODE_NTOK = 4

RE_HDR = re.compile(r"^CGC-DECPROF: step=\d+ segs=(?P<segs>\d+) layers=(?P<layers>\d+)\b"
                    r".*\btotal=(?P<total>[\d.]+) ms.*?\bntok=(?P<ntok>\d+)")
# gpu= union= gap= sg= n= st= en=   (st/en added 2026-09-19; absent in older logs)
RE_ALL = re.compile(r"^CGC-DECPROF all: L(?P<L>\d+) wait=(?P<w>[\d.]+) cb=(?P<cb>[\d.]+) "
                    r"submit=(?P<s>[\d.]+) ms gpu=(?P<g>[\d.]+) union=(?P<u>[\d.]+) gap=(?P<gp>[\d.]+) "
                    r"sg=(?P<sg>\d+) n=(?P<n>\d+)(?: st=(?P<st>-?[\d.]+) en=(?P<en>-?[\d.]+))?")


def parse(path, a_decode_ntok=None, seen_ntok=None):
    """-> list of requests; each is a list of steps; each step is {hdr..., 'lay': {L: fields}}."""
    reqs, cur_req, cur_step = [], None, None
    for line in open(path, errors="replace"):
        if line.startswith("CGC-DECPROF all: "):
            m = RE_ALL.match(line)
            if m and cur_step is not None:
                cur_step["lay"][int(m.group("L"))] = {k: float(m.group(k)) for k in
                                                      ("w", "cb", "s", "g", "u", "gp", "sg", "n")
                                                      if m.group(k) is not None}
                if m.group("st") is not None:
                    cur_step["lay"][int(m.group("L"))]["st"] = float(m.group("st"))
                    cur_step["lay"][int(m.group("L"))]["en"] = float(m.group("en"))
            continue
        if not line.startswith("CGC-DECPROF: "):
            continue
        m = RE_HDR.match(line)
        if not m or int(m.group("layers")) != LAYERS:
            cur_step = None
            continue
        ntok = int(m.group("ntok"))
        # The decode graph WIDTH is a property of the run, not a constant: with MTP it is 1+n_max
        # (measured 4), but a run whose phase-split picks width 8 prints ntok=8 for decode -- and a
        # hard `ntok <= 4` then matches nothing at all and the tool reports "no steps" as if the log
        # were empty. Establish the width from the data: the modal non-prefill value.
        if ntok >= PREFILL_NTOK:
            cur_req = []
            reqs.append(cur_req)
        elif ntok < 100:
            seen_ntok[ntok] = seen_ntok.get(ntok, 0) + 1
            # Recompute every step: `decode_ntok = decode_ntok or modal` latches the FIRST value,
            # and the first value is a tie between the warm-up width and the real one (it locked
            # onto 2 and then declared every width-8 step "not a decode step"). The mode converges
            # as the histogram fills, so ask it each time.
            decode_ntok = a_decode_ntok or (
                max(seen_ntok, key=seen_ntok.get) if seen_ntok else DECODE_NTOK)
            # A request is normally opened by its prefill header. Some runs have NO header at
            # ntok>=100 at all (a profile without CGC_PREFILL_STREAM caps the prefill chunks, so the
            # widest graph is the decode width) -- and then every request stays unopened and the tool
            # reports "no decode steps" over a log full of them. Fall back to one request; the
            # per-request split is then unavailable and that is stated in the output, not implied.
            if cur_req is None:
                cur_req = []
                reqs.append(cur_req)
        if cur_req is not None and ntok == decode_ntok:
            cur_step = {"ntok": ntok, "total": float(m.group("total")), "lay": {}}
            cur_req.append(cur_step)
        else:
            cur_step = None
    return [r for r in reqs if r]


def step_stats(step):
    lay = {L: v for L, v in step["lay"].items() if "st" in v and v["n"] > 0}
    if len(lay) < 2:
        return None
    order = sorted(lay, key=lambda L: lay[L]["st"])
    union_sum = sum(lay[L]["u"] for L in order)
    span = max(lay[L]["en"] for L in order) - min(lay[L]["st"] for L in order)
    idle = 0.0
    max_idle, max_at = 0.0, None
    for a, b in zip(order, order[1:]):
        d = lay[b]["st"] - lay[a]["en"]
        idle += d
        if d > max_idle:
            max_idle, max_at = d, (a, b)
    # The decisive G1 question: is the idle really "GPU waiting for the next segment", or is it
    # the shadow of the CPU hook the next segment depends on? The hook for layer b must run AFTER
    # layer a's argsort is known, and its cost is a's `cb` (top-k + slot management, carried on the
    # hook that fires when a completes). So compare each inter-layer idle against the PREDECESSOR's
    # cb: idle ~ cb => the gap is that hook's shadow (a real dependency, not schedulable slack), and
    # the only correct lever is making the hook cheaper -- which is exactly what CGC_SLOT_TABLE_GPU
    # (S1) was for, and S1 measured no gain in this campaign's slotgpu arm.
    idle_cb = [(lay[b]["st"] - lay[a]["en"], lay[a]["cb"], a, b) for a, b in zip(order, order[1:])]
    # The G3 pair metric. Two definitions, both reported, because they answer different questions:
    #   gap           = combined - (union_a + union_b)  -> the RECOVERABLE idle. This is the one to
    #                   act on; a merge can remove exactly this and nothing more.
    #   vs_2x_single  = combined - 2*max(union_a, union_b) -> the gate's literal wording. It is
    #                   near ZERO whenever the two layers have similar spans (the two forms differ
    #                   only by |union_a - union_b|), so it reads as "no headroom" even when the gap
    #                   is real. Kept for continuity, not for decisions.
    pair_over = []
    for a, b in zip(order, order[1:]):
        combined = lay[b]["en"] - lay[a]["st"]
        gap = lay[b]["st"] - lay[a]["en"]
        pair_over.append((a, b, combined, 2.0 * max(lay[a]["u"], lay[b]["u"]),
                          gap, combined - 2.0 * max(lay[a]["u"], lay[b]["u"])))
    return {"order": order, "union_sum": union_sum, "span": span, "idle": idle, "idle_cb": idle_cb,
            "max_idle": max_idle, "max_at": max_at, "pair": pair_over,
            "total": step["total"], "ntok": step["ntok"], "lay": lay}


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--drop-requests", type=int, default=0)
    ap.add_argument("--csv")
    ap.add_argument("--decode-ntok", type=int, default=0,
                    help="graph width of a decode step. Default: the modal non-prefill ntok seen "
                         "in the log (4 with MTP drafting, 8 when the phase-split picks width 8).")
    a = ap.parse_args()

    seen_ntok = {}
    reqs = parse(a.log, a.decode_ntok, seen_ntok)
    if not reqs:
        print("no decode steps found in %s (is this the server log, not the launcher log?)" % a.log)
        return 2
    print("=" * 100)
    print("G3 TIMELINE  (%s)" % a.log)
    print("=" * 100)
    if seen_ntok:
        modal = max(seen_ntok, key=seen_ntok.get)
        if not any(True for n in seen_ntok if n >= PREFILL_NTOK):
            print("NOTE: no prefill header (ntok>=%d) in this log -- the request split is NOT "
                  "available; all decode steps are reported as one request." % PREFILL_NTOK)
        print("ntok histogram below 100 (decode width detected = %d): %s" % (
            a.decode_ntok or modal,
            ", ".join("n=%d:%d" % (k, v) for k, v in sorted(seen_ntok.items(), key=lambda kv: -kv[1])[:6])))
    allsteps = []
    for i, req in enumerate(reqs):
        if i < a.drop_requests:   # `<=` dropped request 0 even when nothing was asked to be
                                  # dropped, and an empty result then read as "the log is empty"
            continue
        st = [s for s in (step_stats(s) for s in req) if s]
        if not st:
            continue
        allsteps += st
        par = med([s["span"] / s["union_sum"] for s in st if s["union_sum"] > 0])
        print("request %d  steps=%d  parallelism span/union_sum=%.3f   idle=%.2f ms   "
              "span=%.2f  union_sum=%.2f  total=%.2f" % (
                  i, len(st), par, med([s["idle"] for s in st]),
                  med([s["span"] for s in st]), med([s["union_sum"] for s in st]),
                  med([s["total"] for s in st])))
    if not allsteps:
        if reqs and a.drop_requests >= len(reqs):
            print("every request was dropped: --drop-requests %d with only %d request(s) in the log."
                  % (a.drop_requests, len(reqs)))
            return 3
        print("no steps with timestamps -- was the binary built with the st=/en= instrument?")
        return 3   # "cannot answer" must not exit 0: this repo's recurring failure mode is a tool
                   # whose unable-to-measure case is indistinguishable from its pass case.
    print()
    par = med([s["span"] / s["union_sum"] for s in allsteps if s["union_sum"] > 0])
    print("ALL KEPT STEPS (n=%d)" % len(allsteps))
    print("  parallelism  span/union_sum = %.3f   (1.00 = strictly serial, <1 = already overlapping)"
          % par)
    print("  inter-layer idle: median %.2f ms/step, max-window median %.2f ms (worst pair L%d->L%d)"
          % (med([s["idle"] for s in allsteps]), med([s["max_idle"] for s in allsteps]),
             *([s["max_at"] for s in allsteps if s["max_at"]] and
               statistics.mode([tuple(x["max_at"]) for x in allsteps if x["max_at"]]) or (0, 0))))
    # G3 gate metric: the pair's combined span vs 2 x a single layer, over all pairs
    gaps = [p[4] for s in allsteps for p in s["pair"]]
    excess = [p[5] for s in allsteps for p in s["pair"]]
    print("  G3 pair metric (n=%d pairs):" % len(gaps))
    print("    RECOVERABLE gap = combined - (union_a+union_b): median %+.3f  p90 %+.3f  max %+.3f ms"
          % (med(gaps), sorted(gaps)[int(len(gaps) * 0.9)], max(gaps)))
    print("    literal 'combined - 2x single':              median %+.3f  p90 %+.3f  max %+.3f ms"
          % (med(excess), sorted(excess)[int(len(excess) * 0.9)], max(excess)))
    print("    (the two differ by |union_a-union_b|, so the literal form reads ~0 whenever the two")
    print("     layers have similar spans -- use the RECOVERABLE gap to decide.)")
    # idle vs the predecessor's CPU hook window
    pairs = [(d, cb) for s in allsteps for d, cb, _a, _b in s["idle_cb"]]
    if pairs:
        ds = [p[0] for p in pairs]; cs = [p[1] for p in pairs]
        md, mc = med(ds), med(cs)
        num = sum((d - md) * (c - mc) for d, c in pairs)
        den = (sum((d - md) ** 2 for d in ds) * sum((c - mc) ** 2 for c in cs)) ** 0.5
        r = num / den if den > 0 else float("nan")
        # The robust form is a REGRESSION, not a ratio of medians: cb_prev is bimodal
        # (a handful of loud layers at 0.8-2.9 ms, ~28 quiet ones at 0.01-0.06), so the pooled
        # median cb is ~0.02 ms against a median idle of ~0.45 ms and the ratio of medians is
        # meaningless (it prints ~20 while the slope is 1.0).
        sx = sum(cs) / len(cs)
        sy = sum(ds) / len(ds)
        sxx = sum((c - sx) ** 2 for c in cs)
        slope = (sum((c - sx) * (d - sy) for d, c in pairs) / sxx) if sxx > 0 else float("nan")
        icept = sy - slope * sx
        loud = [c for c in cs if c >= 0.4]
        print("  inter-layer idle vs PREDECESSOR's cb (the hook the next segment depends on):")
        print("    idle = %.2f x cb_prev + %.3f ms      r = %+.3f   (n=%d pairs)" % (
            slope, icept, r, len(pairs)))
        print("    cb_prev bimodality: %d/%d pairs with cb>=0.4 ms (median of those %.3f) ; "
              "median cb overall %.3f" % (
                  len(loud), len(cs), med(loud) if loud else float("nan"), mc))
        print("    a slope of 1.00 means the hook adds its WHOLE duration to the critical path --")
        print("    zero CPU/GPU overlap -- 39 times per step, and the intercept is the per-boundary")
        print("    launch/completion residue. So the idle has exactly two components and no third.")
        print("    The window is DEFINED as 'GPU idle while the CPU runs the hook', so hiding it")
        print("    behind GPU work is not available: the only levers are a cheaper hook or fewer")
        print("    boundaries. CGC_SUBMIT_AHEAD's x1.702 deletes the dependency itself and is racy,")
        print("    so it is no ceiling for any correct design.")
    print()
    print("  reading: if parallelism ~1.00 AND the excess is positive, the layers are serial with an")
    print("           idle between them -> merging a pair removes the idle (G3 met, G1 is the lever).")
    print("           if parallelism < 1.00, layers already overlap -> union cannot be cut by any")
    print("           schedule and G4 must come from making each layer faster.")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("step,ntok,total,span,union_sum,idle,layer,st,en,union\n")
            for si, s in enumerate(allsteps):
                for L in s["order"]:
                    f.write("%d,%d,%.3f,%.3f,%.3f,%.3f,L%d,%.3f,%.3f,%.3f\n" % (
                        si, s["ntok"], s["total"], s["span"], s["union_sum"], s["idle"], L,
                        s["lay"][L]["st"], s["lay"][L]["en"], s["lay"][L]["u"]))
        print("\n  wrote %s" % a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
