#!/usr/bin/env python3
"""Turn a `CGC_MISS_MASK_DBG=1` stderr log into the one number the step-3 gate prices against:
of this step's SELECTED experts, how many are placeholders (a device-side answer, no host).

WHY THIS FILE EXISTS
--------------------
`CGC_MISS_MASK` was rebuilt in commit 553424ec1 after it turned out it had never entered version
control. What came back was the INSTRUMENT, not a number: until something reads the log there is
exactly as much evidence about the miss rate as there was the day before that commit. This is the
reader. It is a file rather than a shell pipeline because the reading has three traps a one-liner
hits silently:

  1. steps vs decode steps. `cgc_mm_step` (llama-context.cpp) counts EVERY `graph_compute_async`,
     so prefill and warm-up occupy step numbers too, and an all-zero-miss decode compute prints
     `layers=0` -- indistinguishable from a prefill compute. Nothing here therefore derives
     anything from the step NUMBER; only ORDER among rows that exist is used, and the ambiguity is
     reported as a field rather than resolved by assumption.
  2. micro vs macro average. `sum(misses)/sum(nsel)` (micro) is "what fraction of the expert slots
     actually consumed were placeholders", which is what per-expert recompute is priced against.
     macro (mean of per-layer rates) answers a different question; both are printed, never
     silently substituted for one another.
  3. monotonic drift. The single-segment arm (`CGC_SEG_BATCH=1`) matters precisely because its hook
     never runs, so nothing ever fills the pool: the miss rate is not expected to jitter around a
     steady state, it is expected to RISE. One average hides the thing being measured, so the
     per-compute series and a head-vs-tail drift come out too.

USAGE
    python3 scripts/check/miss_rate_summary.py <stderr.log> [--json out.json] [--tail-series N]
    python3 scripts/check/miss_rate_summary.py --self-test

GRADING: the numbers below are M (measured) for what they are -- counts of printed bytes. Whether
they make per-expert recompute PAY is still a projection; the three branches are
docs/SPEED_ACCEPTANCE_GATE_2026-09-26.md §3. This file deliberately does not decide that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# MISSMASK il=3 step=190 nsel=8 misses=2 exps: 41 77
MISS_RE = re.compile(
    r"^MISSMASK\s+il=(\d+)\s+step=(\d+)\s+nsel=(\d+)\s+misses=(\d+)\s+exps:(.*)$")
# CGC-MISSMASK-STEP: step=190 misses=17 layers=38
STEP_RE = re.compile(
    r"^CGC-MISSMASK-STEP:\s+step=(\d+)\s+misses=(\d+)\s+layers=(\d+)\s*$")

NOTHING_WHY = (
    "no MISSMASK line was printed. Let EXIT-STATE choose among the causes rather than guessing:\n"
    "  A) CGC_MISS_MASK=1 never reached the engine -- scripts/run_server.sh forwards env through an\n"
    "     ALLOWLIST (commit 58e0f4020 added these two) and drops unlisted CGC_* silently.\n"
    "  B) CGC_MISS_MASK_DBG=1 without CGC_MISS_MASK=1 -- the engine prints one warning and no data.\n"
    "  C) every decode compute genuinely had zero misses.\n"
    "The count of `CGC-MISSMASK-STEP` lines separates them: with the instrument armed there is one\n"
    "per graph compute whether or not anything missed, so zero of those is A or B, some is C."
)


def parse(text: str) -> dict:
    per_layer, per_step = [], []
    for raw in text.splitlines():
        line = raw.strip()
        m = MISS_RE.match(line)
        if m:
            il, step, nsel, misses = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            exps = [int(x) for x in m[5].split()] if m[5].strip() else []
            per_layer.append(dict(il=il, step=step, nsel=nsel, misses=misses,
                                  rate=(misses / nsel) if nsel else None, exps=exps))
            continue
        m = STEP_RE.match(line)
        if m:
            per_step.append(dict(step=int(m[1]), misses=int(m[2]), layers=int(m[3])))
    return {"per_layer": per_layer, "per_step": per_step}


def summarize(text: str) -> dict:
    rows = parse(text)
    pl, ps = rows["per_layer"], rows["per_step"]

    out: dict = {
        "n_missmask_lines": len(pl),
        "n_step_lines": len(ps),
        "step_lines_layers_zero": sum(1 for e in ps if e["layers"] == 0),
        "step_lines_layers_pos": sum(1 for e in ps if e["layers"] > 0),
    }
    if not pl:
        out.update(verdict="NO_MASK_LINES", why=NOTHING_WHY,
                   engine_armed=(len(ps) > 0))
        return out

    tot_misses = sum(r["misses"] for r in pl)
    tot_nsel = sum(r["nsel"] for r in pl)
    rates = [r["rate"] for r in pl if r["rate"] is not None]

    out.update(
        sum_misses=tot_misses,
        sum_nsel=tot_nsel,
        miss_rate_micro=(tot_misses / tot_nsel) if tot_nsel else None,
        miss_rate_macro=(sum(rates) / len(rates)) if rates else None,
        layers_seen=sorted({r["il"] for r in pl}),
        nsel_values=sorted({r["nsel"] for r in pl}),
        distinct_computes=len({r["step"] for r in pl}),
        exps_len_matches=sum(1 for r in pl if len(r["exps"]) == r["misses"]),
        lines_with_zero_misses=sum(1 for r in pl if r["misses"] == 0),
    )

    by_step: dict[int, dict] = {}
    for r in pl:
        e = by_step.setdefault(r["step"], dict(step=r["step"], misses=0, nsel=0, layers=0))
        e["misses"] += r["misses"]
        e["nsel"] += r["nsel"]
        e["layers"] += 1
    series = sorted(by_step.values(), key=lambda e: e["step"])
    for e in series:
        e["rate"] = (e["misses"] / e["nsel"]) if e["nsel"] else None
    out["step_series"] = series

    # Drift is the question that decides whether per-expert recompute is even the right mechanism:
    # a rising rate means the pool is getting COLDER, i.e. nothing fills it, i.e. the premise that
    # "the missing term is ~0 and only needs recomputing" is only true because the whole layer is
    # wrong, not because one expert is late.
    # Needs at least three computes so that head and tail share no entry at k = max(1, n//3);
    # below that there is nothing to compare and printing a rate would invent a trend.
    n = len(series)
    if n >= 3:
        k = max(1, n // 3)
        head, tail = series[:k], series[-k:]
        hm, hn = sum(e["misses"] for e in head), sum(e["nsel"] for e in head)
        tm, tn = sum(e["misses"] for e in tail), sum(e["nsel"] for e in tail)
        out["drift"] = dict(
            head_rate=(hm / hn) if hn else None,
            tail_rate=(tm / tn) if tn else None,
            delta=((tm / tn) - (hm / hn)) if (hn and tn) else None,
            head_n=k, tail_n=k)
    return out


# --------------------------------------------------------------------------- self test
def self_test() -> int:
    fails = []

    def expect(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    empty = summarize("nothing here\n")
    expect("empty log -> NO_MASK_LINES", empty["verdict"], "NO_MASK_LINES")
    expect("empty log -> engine not armed", empty["engine_armed"], False)

    # zero misses but the instrument armed: proves cause C is distinguishable from A/B
    armed_zero = summarize(
        "load stuff\nCGC-MISSMASK-STEP: step=1 misses=0 layers=0\n"
        "CGC-MISSMASK-STEP: step=2 misses=0 layers=0\n")
    expect("armed + zero miss -> still NO_MASK_LINES", armed_zero["verdict"], "NO_MASK_LINES")
    expect("armed + zero miss -> engine_armed True", armed_zero["engine_armed"], True)

    # a real log: 2 computes x 2 layers, rising rate
    log = "\n".join([
        "MISSMASK il=10 step=5 nsel=8 misses=1 exps: 3",
        "MISSMASK il=11 step=5 nsel=8 misses=1 exps: 9",
        "CGC-MISSMASK-STEP: step=5 misses=2 layers=2",
        "MISSMASK il=10 step=6 nsel=8 misses=4 exps: 1 2 3 4",
        "MISSMASK il=11 step=6 nsel=8 misses=4 exps: 5 6 7 8",
        "CGC-MISSMASK-STEP: step=6 misses=8 layers=2",
        "MISSMASK il=10 step=7 nsel=8 misses=8 exps: 1 2 3 4 5 6 7 8",
        "MISSMASK il=11 step=7 nsel=8 misses=8 exps: 1 2 3 4 5 6 7 8",
        "CGC-MISSMASK-STEP: step=7 misses=16 layers=2",
    ]) + "\n"
    s = summarize(log)
    expect("3 mask lines of theirs -> n lines", s["n_missmask_lines"], 6)
    expect("sum misses", s["sum_misses"], 26)
    expect("sum nsel", s["sum_nsel"], 48)
    expect("micro rate", round(s["miss_rate_micro"], 6), round(26 / 48, 6))
    expect("macro rate (2 lines 1/8, 2 lines 4/8, 2 lines 8/8)",
           round(s["miss_rate_macro"], 6), round((1 / 8 * 2 + 4 / 8 * 2 + 8 / 8 * 2) / 6, 6))
    expect("micro == macro is a coincidence here, not an identity",
           s["miss_rate_micro"] == s["miss_rate_macro"], True)
    expect("exps list length consistent", s["exps_len_matches"], 6)
    expect("layers counted", s["layers_seen"], [10, 11])
    expect("distinct computes", s["distinct_computes"], 3)
    expect("step printed with layers>0", s["step_lines_layers_pos"], 3)
    expect("drift is detected as RISING", s["drift"]["delta"] > 0, True)
    expect("drift head is the low one",
           round(s["drift"]["head_rate"], 6), round(2 / 16, 6))
    expect("drift tail is the high one",
           round(s["drift"]["tail_rate"], 6), round(16 / 16, 6))

    # exps count must track misses -- the one internal consistency check the engine offers
    bad = summarize("MISSMASK il=1 step=1 nsel=8 misses=3 exps: 1 2\n")
    expect("exps shorter than misses is caught", bad["exps_len_matches"], 0)

    print("SELFTEST %s (%d)" % ("FAIL" if fails else "PASS", len(fails)))
    for f in fails:
        print("  " + f)
    return 1 if fails else 0


# --------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(
        description="miss rate from a CGC_MISS_MASK_DBG log (see module docstring)")
    ap.add_argument("log", nargs="?", help="path to a llama_bench_*.stderr.log")
    ap.add_argument("--json", dest="json_out", default="", help="write the summary here")
    ap.add_argument("--tail-series", type=int, default=0,
                    help="print the last N per-compute entries")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.log:
        ap.error("a log path is required (or --self-test)")

    s = summarize(Path(args.log).read_text(errors="replace"))

    if s.get("verdict"):
        print("verdict=%s" % s["verdict"])
        print(s["why"])
        print("CGC-MISSMASK-STEP lines seen: %d -> engine %s armed" % (
            s["n_step_lines"], "IS" if s["engine_armed"] else "is NOT"))
        return 1

    print("MISSMASK lines        : %d   (distinct computes: %d)" % (
        s["n_missmask_lines"], s["distinct_computes"]))
    print("layers seen           : %s" % s["layers_seen"])
    print("nsel values seen      : %s" % s["nsel_values"])
    print("sum misses / sum nsel : %d / %d" % (s["sum_misses"], s["sum_nsel"]))
    print("miss rate MICRO       : %.4f  (%.2f%%)  <- what per-expert recompute is priced against" % (
        s["miss_rate_micro"], 100.0 * s["miss_rate_micro"]))
    if s["miss_rate_macro"] is not None:
        print("miss rate MACRO       : %.4f  (%.2f%%)  <- mean of per-layer rates, a different "
              "question" % (s["miss_rate_macro"], 100.0 * s["miss_rate_macro"]))
    print("exps list length matches misses on %d/%d lines" % (
        s["exps_len_matches"], s["n_missmask_lines"]))
    if "drift" in s:
        d = s["drift"]
        print("drift head(%d)=%.4f  tail(%d)=%.4f  delta=%+.4f" % (
            d["head_n"], d["head_rate"], d["tail_n"], d["tail_rate"], d["delta"]))
    print("step lines: layers>0 = %d   layers=0 = %d   (layers=0 is AMBIGUOUS: prefill compute or "
          "a miss-free decode compute -- the counter counts every graph_compute_async)" % (
              s["step_lines_layers_pos"], s["step_lines_layers_zero"]))

    if args.tail_series:
        print("\nlast %d computes with mask rows:" % args.tail_series)
        for e in s["step_series"][-args.tail_series:]:
            print("  step=%-6d layers=%-3d nsel=%-8d misses=%-8d rate=%.4f" % (
                e["step"], e["layers"], e["nsel"], e["misses"], e["rate"]))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(s, indent=2) + "\n")
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
