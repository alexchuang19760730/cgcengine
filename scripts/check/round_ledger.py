#!/usr/bin/env python3
"""What is a round actually made of? Steps, not a mysterious post-layer remainder.

WHY THIS EXISTS

`docs/VERIFY_MARGINAL_2026-09-22.md` §5 ends with a "round remainder" of 108-138 ms at T=4
(70-102% of the marginal verify token) and calls it "post-layer per-round work". That number is
built by subtraction (`scripts/check/verify_marginal.py:245`):

    remainder = round_ms - sum_over_40_layers(wait + cb + submit) - draft_ms - begin_ms

The 40-layer sum is ONE STEP's layer time (each row is that layer's mean over sampled steps).
But one round is not one step: `CGC-DECPROF` shows a round contains several `segs=41` steps, and
the count per round grows with T (1.5-1.6 at T=2, 2.25-2.30 at T=4 in this log). So the
subtraction removes one step while the round spent more than one, and what is left over is
mostly "(n_step - 1) steps" -- which grows with T for free. That is why it looked like a
T-dependent post-layer term.

This tool recomputes the round from the step population instead:

    round_ms  ~=  (n_seg41_steps / calls_draft) * step_ms  +  draft_ms/round

and prints the closure error next to it, plus what a step is actually made of
(DECPROF: wait=eval-sync, cb=gather, submit=dispatch -- they sum to 100% of the step by
construction) and what the gather is made of (`CGC-HOOKSPLIT`: pre / ensure / drain / tail, where
`ensure` is `llama_expert_cache_ensure_batch`, llama-context.cpp:6735).

ZERO GPU. It reads logs that already exist; it runs nothing.

USAGE
    python3 scripts/check/round_ledger.py --dir /tmp/verify_layers/20260922_165215_prod25_decode-spec
    python3 scripts/check/round_ledger.py --arms /tmp/verify_layers/arms.jsonl
    python3 scripts/check/round_ledger.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

STEP = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \(([\d.]+)%\) cb=([\d.]+) \(([\d.]+)%\) submit=([\d.]+) \(([\d.]+)%\) ntok=(\d+)")
PERF = re.compile(
    r"CGC-MTP-PERF type=(\S+) calls_begin=(\d+) calls_draft=(\d+) calls_accept=(\d+) "
    r"gen_tokens=(\d+) acc_tokens=(\d+) t_begin_ms=([\d.]+) t_draft_ms=([\d.]+) t_accept_ms=([\d.]+)"
    r".*emit_tok_per_round=([\d.]+)")
HOOK = re.compile(
    r"CGC-HOOKSPLIT: n=(\d+)\s+pre=([\d.]+)\s+ensure=([\d.]+)\s+drain=([\d.]+)\s+tail=([\d.]+) us/call")


def newest_log(rdir: Path) -> Path | None:
    logs = sorted(rdir.glob("*.stderr.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse(text: str) -> dict:
    """Everything this tool needs from one run's stderr log."""
    perf = [m.groups() for m in PERF.finditer(text)]
    if not perf:
        return {"invalid": "no CGC-MTP-PERF line (needs CGC_MTP_PERF=1 and a spec arm)"}
    p = perf[-1]  # cumulative counters: the last line is the run total
    by_segs = collections.defaultdict(lambda: {"n": 0, "total": [], "wait": [], "cb": [], "submit": []})
    ntok = collections.Counter()
    for m in STEP.finditer(text):
        segs = int(m.group(2))
        g = by_segs[segs]
        g["n"] += 1
        g["total"].append(float(m.group(4)))
        g["wait"].append(float(m.group(5)))
        g["cb"].append(float(m.group(7)))
        g["submit"].append(float(m.group(9)))
        ntok[m.group(11)] += 1
    hooks = [m.groups() for m in HOOK.finditer(text)]
    rounds = int(p[2])
    if rounds <= 0:
        return {"invalid": f"calls_draft={rounds}: no rounds happened"}
    # group order: 0 type, 1 calls_begin, 2 calls_draft, 3 calls_accept, 4 gen_tokens,
    #              5 acc_tokens, 6 t_begin, 7 t_draft, 8 t_accept, 9 emit_tok_per_round
    # (acc_rate / *_per_round are skipped by the `.*` between t_accept and emit)
    return {"rounds": rounds, "emit_tok_per_round": float(p[9]),
            "t_begin_ms": float(p[6]), "t_draft_ms": float(p[7]), "t_accept_ms": float(p[8]),
            "segs": {k: dict(v) for k, v in by_segs.items()}, "ntok": dict(ntok),
            "hook": [{"n": int(h[0]), "pre": float(h[1]), "ensure": float(h[2]),
                      "drain": float(h[3]), "tail": float(h[4])} for h in hooks]}


def ledger(r: dict, round_ms: float, step_stat: str = "median") -> dict:
    """Close the round from the step population. Returns the terms and the closure error."""
    f = st.median if step_stat == "median" else st.mean
    seg = r["segs"].get(41)
    if not seg or not seg["total"]:
        return {"invalid": "no segs=41 DECPROF steps -- the verify population is absent"}
    n_per_round = seg["n"] / r["rounds"]
    step_ms = f(seg["total"])
    draft_per_round = r["t_draft_ms"] / r["rounds"]
    begin_per_round = r["t_begin_ms"] / r["rounds"]
    budget = n_per_round * step_ms + draft_per_round + begin_per_round
    return {"rounds": r["rounds"], "emit": r["emit_tok_per_round"],
            "n_seg41": seg["n"], "steps_per_round": n_per_round,
            "step_ms": step_ms, "step_mean": st.mean(seg["total"]),
            "draft_per_round": draft_per_round, "begin_per_round": begin_per_round,
            "budget": budget, "round_ms": round_ms,
            "closure_pct": 100.0 * (budget - round_ms) / round_ms if round_ms else float("nan"),
            "step_share": {"wait": f(seg["wait"]), "cb": f(seg["cb"]), "submit": f(seg["submit"])},
            "ntok": r["ntok"], "segs_seen": sorted(r["segs"])}


def report_one(rdir: Path, round_ms: float | None, label: str = "") -> int:
    log = newest_log(rdir)
    if log is None:
        print(f"INVALID: no *.stderr.log in {rdir}")
        return 1
    r = parse(log.read_text(errors="replace"))
    if "invalid" in r:
        print(f"INVALID: {r['invalid']}")
        return 1
    # round_ms comes from the arm record (t/s and emit/round); without it, derive from t/s is
    # impossible here, so the caller must pass it. None -> report the step population only.
    L = ledger(r, round_ms if round_ms else float("nan"))
    name = label or rdir.name
    print(f"  {name}")
    print(f"    rounds={L['rounds']}  emit/round={L['emit']:.3f}  "
          f"segs seen={L['segs_seen']}  ntok={L['ntok']}")
    print(f"    segs=41 steps n={L['n_seg41']}  -> {L['steps_per_round']:.2f} steps/round")
    print(f"    step ms: median={L['step_ms']:.1f}  mean={L['step_mean']:.1f}   "
          f"(median/mean = {L['step_ms'] / L['step_mean']:.2f} -> the mean is inflated by slow steps)")
    s = L["step_share"]
    tot = s["wait"] + s["cb"] + s["submit"]
    if tot:
        print(f"    step split: wait(eval-sync) {s['wait']:6.1f} ms {100 * s['wait'] / tot:3.0f}%   "
              f"cb(gather) {s['cb']:6.1f} ms {100 * s['cb'] / tot:3.0f}%   "
              f"submit(dispatch) {s['submit']:5.1f} ms {100 * s['submit'] / tot:3.0f}%")
    if r["hook"]:
        h = r["hook"][-1]  # cumulative average: the LAST line is the run-wide per-call average
        t = h["pre"] + h["ensure"] + h["drain"] + h["tail"]
        print(f"    hook/call (n={h['n']}): ensure={h['ensure']:.1f} us "
              f"({100 * h['ensure'] / t:.1f}% of {t:.1f})  pre={h['pre']:.1f} "
              f"drain={h['drain']:.1f} tail={h['tail']:.1f}")
    if round_ms:
        print(f"    closure: {L['steps_per_round']:.2f} x {L['step_ms']:.1f} + draft "
              f"{L['draft_per_round']:.1f} + begin {L['begin_per_round']:.3f} = {L['budget']:.1f} ms "
              f"vs round {L['round_ms']:.1f} ms  ({L['closure_pct']:+.1f}%)")
    return 0


def report_arms(jsonl: Path) -> int:
    recs = [json.loads(l) for l in open(jsonl)]
    print(f"round ledger from {jsonl}  ({len(recs)} arm run(s))")
    print("  round_ms is the arm's own (t/s, emit/round) reading; the rest is recomputed here\n")
    rc = 0
    for rec in sorted(recs, key=lambda x: (x.get("rep", 0), x.get("arm", ""))):
        rc |= report_one(Path(rec["dir"]), rec.get("round_ms"),
                         f"rep{rec.get('rep')}.{rec.get('arm')}  round_ms={rec.get('round_ms', 0):.1f}")
        print()
    return rc


def selftest() -> int:
    """Arithmetic, and the two ways this ledger could lie: a fake step count and a hidden term."""
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    check("no MTP-PERF line is INVALID, not a zero ledger",
          "invalid" in parse("nothing here\n"))
    r = parse("CGC-MTP-PERF type=draft-mtp calls_begin=4 calls_draft=0 calls_accept=4 "
              "gen_tokens=12 acc_tokens=8 t_begin_ms=1.0 t_draft_ms=2.0 t_accept_ms=0.1 "
              "acc_rate=0.6 gen_tok_per_round=3.0 acc_tok_per_round=2.0 emit_tok_per_round=3.0\n")
    check("calls_draft=0 is INVALID", "invalid" in r)

    text = ("CGC-MTP-PERF type=draft-mtp calls_begin=4 calls_draft=100 calls_accept=4 "
            "gen_tokens=300 acc_tokens=200 t_begin_ms=10.0 t_draft_ms=3000.0 t_accept_ms=20.0 "
            "acc_rate=0.66 gen_tok_per_round=3.0 acc_tok_per_round=2.0 emit_tok_per_round=3.0\n")
    for i in range(300):
        text += (f"CGC-DECPROF: step={i} segs=41 layers=40 total=100.0 ms | wait=40.0 (40%) "
                 f"cb=50.0 (50%) submit=10.0 (10%) ntok=1\n")
    r = parse(text)
    L = ledger(r, 300.0)
    check("steps/round from the step population", abs(L["steps_per_round"] - 3.0) < 1e-9)
    check("draft/round", abs(L["draft_per_round"] - 30.0) < 1e-9)
    check("budget = steps*step + draft + begin", abs(L["budget"] - (3 * 100.0 + 30.0 + 0.1)) < 1e-9)
    check("closure reports the budget's overshoot",
          abs(L["closure_pct"] - 100.0 * (330.1 - 300.0) / 300.0) < 1e-9)
    check("step shares sum to the step", abs(sum(L["step_share"].values()) - 100.0) < 1e-9)
    check("gather is the largest share at this synthetic 40/50/10",
          L["step_share"]["cb"] == max(L["step_share"].values()))

    # a mean inflated by one slow step must NOT be silently used as the step time
    r2 = parse(text + "CGC-DECPROF: step=999 segs=41 layers=40 total=10000.0 ms | wait=1.0 (0%) "
                      "cb=1.0 (0%) submit=1.0 (0%) ntok=1\n")
    L2 = ledger(r2, 300.0, step_stat="mean")
    check("mean step > median step when a slow step is added",
          L2["step_mean"] > ledger(r2, 300.0)["step_ms"])
    print(f"round_ledger selftest: {9 - bad}/9 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="one run dir (with *.stderr.log)")
    ap.add_argument("--round-ms", type=float, default=None,
                    help="the arm's round_ms (from its t/s); omitted -> no closure reported")
    ap.add_argument("--arms", help="arms.jsonl from verify_marginal / mtp_k_sweep")
    ap.add_argument("--step-stat", choices=("median", "mean"), default="median")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.arms:
        return report_arms(Path(args.arms))
    if args.dir:
        return report_one(Path(args.dir), args.round_ms)
    print("need --dir, --arms, or --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
