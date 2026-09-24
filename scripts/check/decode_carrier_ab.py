#!/usr/bin/env python3
"""decode_carrier_ab.py -- turn an ab_interleave run into a quotable decode-carrier verdict.

Why this exists rather than quoting the driver's own print: the protocol
(docs/ABBA_MEASUREMENT_PROTOCOL_2026-09-23.md) asks for a **paired per-rep ratio median** and for the
within-arm spread to be shown next to it, because this machine's launch-to-launch drift (+-20%,
measured) is larger than most of the effects being tested. A single pair, or a ratio of two medians,
reads that drift as an effect -- which is exactly the mistake this tool must not repeat. It also
prints `inconclusive` instead of a sign when the two arms' own spreads overlap the difference.

Usage:
    python3 scripts/check/decode_carrier_ab.py --json Backup/phase_decomp/ab_nospac_prodnew.json
    python3 scripts/check/decode_carrier_ab.py --json ... --write docs/DECODE_CARRIER_AB_2026-09-23
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import server_window as sw  # noqa: E402  (the shared probe + window taxonomy)
import io_symmetry as ios  # noqa: E402  (the I/O-structure rule; see its docstring)


def pairs(rows):
    """(rep, base_row, test_row) for reps where BOTH arms recorded. Arm 0 is the base by contract."""
    arms = []
    for r in rows:
        if r.get("arm") not in arms:
            arms.append(r["arm"])
    if len(arms) != 2:
        return arms, []
    base, test = arms
    by = {}
    for r in rows:
        by[(r["arm"], r.get("rep"))] = r
    out = []
    for rep in sorted({r.get("rep") for r in rows}):
        a, b = by.get((base, rep)), by.get((test, rep))
        if a and b:
            out.append((rep, a, b))
    return arms, out


def summarize(rows, io=None):
    arms, ps = pairs(rows)
    if len(arms) != 2 or not ps:
        return {"verdict": "not-ready", "why": f"arms={arms} pairs={len(ps)}"}
    base, test = arms
    ratios = [b["decode_tps_median"] / a["decode_tps_median"] for _, a, b in ps]
    # Spread comes from EVERY row of each arm, not from the paired subset: with one pair the paired
    # subset has one row per arm, so its "spread" is 0 -- and a rule that reads that as "no drift"
    # certifies the first pair as an effect. That is precisely how the 2026-09-23 21:14 pair would
    # have been quoted as '+9.7% for SpAc=0' while the same arm's next launch was 13.6% faster.
    a_ts = [r["decode_tps_median"] for r in rows if r.get("arm") == base]
    b_ts = [r["decode_tps_median"] for r in rows if r.get("arm") == test]
    spread_a = (max(a_ts) - min(a_ts)) / st.median(a_ts) * 100 if len(a_ts) > 1 else None
    spread_b = (max(b_ts) - min(b_ts)) / st.median(b_ts) * 100 if len(b_ts) > 1 else None
    med_ratio = st.median(ratios)
    effect = (med_ratio - 1.0) * 100
    if spread_a is None or spread_b is None:
        verdict, drift = "inconclusive", None
        why = "an arm has a single launch, so its own drift is UNKNOWN, not zero"
    else:
        drift = max(spread_a, spread_b)
        verdict = "inconclusive" if abs(effect) <= drift else ("faster" if effect > 0 else "slower")
        why = f"|{effect:+.1f}%| vs within-arm spread {drift:.1f}%"
    io = io or {}
    # THE HARD RULE. A paired ratio is only a shape effect when both arms did the same I/O work per
    # token; if they did not, the difference contains the I/O itself and must not be signed. Case that
    # forced it (2026-09-24): the segmented arm did 82 293 file reads, the single-submit arm 0 -- it
    # had no hook, so its fill path never ran -- and the 2.2x between them was quoted as a shape gain.
    if io.get("blocking") and verdict != "not-ready":
        verdict, why = "shape-confounded", (
            f"I/O structure asymmetric between the arms ({'; '.join(io.get('differing') or [])}) -- "
            f"the {med_ratio:.2f}x spans arms that did different I/O work, so it is not a shape gain")
    return {
        "base_arm": base, "test_arm": test, "pairs": len(ps),
        "per_rep_decodes": {f"{base}#r{r}": a["decode_tps_median"] for r, a, _ in ps}
        | {f"{test}#r{r}": b["decode_tps_median"] for r, _, b in ps},
        "paired_ratios": ratios,
        "paired_ratio_median": med_ratio,
        "effect_pct": effect,
        "within_arm_spread_pct": {"base": spread_a, "test": spread_b},
        "verdict": verdict,
        "verdict_why": why,
        "verdict_rule": "|paired median effect| > max(within-arm spread over ALL launches of each "
                        "arm); a missing partner makes the spread unknown, not zero",
        "n_launches_per_arm": {"base": len(a_ts), "test": len(b_ts)},
        "io_symmetry": io,
        "answer_md5": sorted({m for r in rows for m in (r.get("answer_md5_set") or [])}),
        "hit_rate_pct": {r["key"]: r.get("hit_rate_pct") for r in rows},
        "build": sorted({json.dumps(r.get("build"), sort_keys=True) for r in rows}),
        "per_arm_env": {r["key"]: r.get("env") for r in rows},
        "window": sw.provenance(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="an ab_interleave --json artifact")
    ap.add_argument("--write", default="", help="write <path>.json + <path>.md")
    ap.add_argument("--io-dir", default="",
                    help="directory holding the arms' logs (default: the artifact's own directory); "
                         "set to 'none' to skip the I/O-structure rule, which then says 'unchecked'")
    args = ap.parse_args()

    rows = json.load(open(args.json))
    if args.io_dir == "none":
        io = {"verdict": "unchecked", "blocking": False, "why": "rule skipped by --io-dir none"}
    else:
        d = args.io_dir or str(Path(args.json).resolve().parent)
        io = ios.audit_dir(d) or {"verdict": "unchecked", "blocking": False,
                                  "why": f"no abba_*.json (and so no arm logs) under {d}"}
    s = summarize(rows, io)
    print(json.dumps(s, indent=1, ensure_ascii=False))
    if s["io_symmetry"].get("verdict") == "unchecked":
        print(f"\nNOTE: I/O structure UNCHECKED -- {s['io_symmetry']['why']}. The ratio above is a speed\n"
              f"reading only; whether both arms did the same work per token is unverified.")
    else:
        print("\n" + ios.banner(s["io_symmetry"]))
    if s["verdict"] == "not-ready":
        return 1
    if s["verdict"] == "shape-confounded":
        return 2

    if args.write:
        out = Path(args.write)
        out.with_suffix(".json").write_text(json.dumps(
            {"product": "decode carrier A/B", "source": args.json, **s}, indent=1) + "\n")
        md = [              f"# decode carrier A/B — {s['base_arm']} vs {s['test_arm']}", "",
              f"**Verdict**: `{s['verdict']}` — paired ratio median **{s['paired_ratio_median']:.3f}** "
              f"({s['effect_pct']:+.1f}%) over {s['pairs']} interleaved pair(s); {s['verdict_why']}.", "",
              f"I/O structure: `{s['io_symmetry'].get('verdict')}` — {s['io_symmetry'].get('why')}", "",
              f"Within-arm launch spread (all launches of each arm): base "
              f"{s['within_arm_spread_pct']['base']}, test {s['within_arm_spread_pct']['test']}. "
              f"Rule: {s['verdict_rule']}.", "",
              "| arm#rep | decode t/s |", "|---|---:|"]
        for k, v in s["per_rep_decodes"].items():
            md.append(f"| {k} | {v:.2f} |")
        md += ["", f"Answer md5 set (must be identical for a speed-only claim): {s['answer_md5']}",
               "", f"Build: {s['build']}", "",
               f"Window at write time: `{s['window'].get('class')}` — {s['window'].get('why')}", ""]
        out.with_suffix(".md").write_text("\n".join(md))
        print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
