#!/usr/bin/env python3
"""drift_adjudicate.py -- L3: adjudicate a drift reading against the caliber contract.

WHY THIS EXISTS
---------------
`agent_harness/portal/targets.json` already states the rule in prose, in its
invariants, and it is the correct one:

    "M1/M2 = 884/884 on a LONG probe (short probes systematically read 'no
     difference': 9 steps gave M2 9/9 where 884 steps gave 25/884)"

That is a machine-checkable claim with no machine check. Three tools were
producing verdicts it contradicts, and every one of them was quotable:

  1. `m123_oracle_gate.py` reported `ok: true` on a 9-record run.
  2. `9/9` was quoted in charters and memory as "bit-identical" many times.
  3. The 2026-10-02 overlap batch reported `comparable: true, config_diffs: []`
     on runs whose M1 ranged 6/222 .. 222/222, because two of them were measured
     against a DIFFERENT reference file (fixed 2026-10-07, see
     `comparability_defects`).

So the drift readings this repo keeps producing are not untrustworthy by accident;
they are untrustworthy because no tool enforces the caliber they must be read at.

WHAT THIS TOOL DECIDES
-----------------------
Given one or more oracle comparison reports (jsonl pairs or oraclecmp_*.json), it
answers a single question per artefact:

    Is this drift reading quotable under the caliber contract?

and it grades three DISTINCT outcomes, because collapsing them is what produced
the errors above:

    CERTIFIABLE   prefix covers >= MIN_PREFIX consecutive bit-identical steps
                  AND no decision divergence. A change that breaks the first
                  steps is a real regression and this says so.
    DRIFT_ONLY    numeric divergence with every argmax unchanged. On this engine
                  that is reduction-order sensitivity propagating along the KV
                  chain (Metal 3 `simdgroup_multiply_accumulate` internals are
                  not programmable -- verified 2026-10-07). Not a behavioural
                  regression, and NOT quotable as bit-identical.
    UNQUOTABLE    a decision divergence, or a prefix too short to certify
                  anything, or the artefacts are not comparable to each other.

The point of grading rather than boolean-passing: a DRIFT_ONLY run must not be
reported as a failure (there is no bug to fix), and must not be reported as a
pass (the output is not bit-identical). Both of those misreports happened.

USAGE
    python3 scripts/check/drift_adjudicate.py Backup/m123_oracle_gate/oraclecmp_*.json
    python3 scripts/check/drift_adjudicate.py --pairs a.jsonl b.jsonl [...]
    python3 scripts/check/drift_adjudicate.py --json Backup/.../drift_verdict.json
    python3 scripts/check/drift_adjudicate.py --selftest

EXIT CODES
    0  every artefact is CERTIFIABLE or DRIFT_ONLY (nothing that claims equivalence is false)
    1  at least one artefact is UNQUOTABLE
    2  harness error (unreadable input, no artefacts)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "scripts" / "check" / "cgc_logits_oracle_compare.py"

# The caliber floor. 9 records is the SHORT probe (`gate_registry.py`: "enough to
# see that a knob MOVED the logits, NOT enough to say a choice survives"). The
# number that has actually been certified in this repo is 884 (targets.json G2).
# Anything under this floor cannot certify equivalence no matter how green it is,
# which is exactly the trap that made 9/9 quotable.
MIN_PREFIX = 9
CERTIFY_PREFIX = 100

GRADES = ("CERTIFIABLE", "DRIFT_ONLY", "UNQUOTABLE")


def _first_diff(report, field):
    """first_{numeric,decision,topk}_divergence -> int step, or None.

    Accepts both the 2026-10-07 `drift` block and the older shape where only
    `diff_examples` exists. A report with neither is NOT treated as clean: it is
    treated as unknown, because "no onset recorded" and "no divergence" are
    different claims and only one of them is a pass.
    """
    drift = report.get("drift")
    if isinstance(drift, dict):
        v = drift.get(field)
        if v is None:
            return None
        rec = v[0] if isinstance(v, (list, tuple)) and v else v
        return rec if isinstance(rec, int) else None
    # Fallback: reconstruct the onset from diff_examples (first 5 only -- so this
    # can only ever UNDER-report the onset, which is the safe direction).
    for ex in report.get("diff_examples") or []:
        key = ex.get("key")
        if key is None:
            continue
        field_name = {"first_numeric_divergence": "numeric_equal",
                      "first_decision_divergence": "decision_equal"}.get(field)
        if field_name is None:  # topk has no diff_examples column
            continue
        if not ex.get(field_name, True):
            return key[0] if isinstance(key[0], int) else None
    return None


def _prefix_of(report):
    drift = report.get("drift")
    if isinstance(drift, dict) and isinstance(drift.get("bit_identical_prefix"), int):
        return drift["bit_identical_prefix"]
    # Without the drift block the only lower bound we can state is "at least the
    # first diff example" -- i.e. the prefix is at least that many steps IF the
    # first example really is the onset. Not provable, so return None.
    return None


def _n_common(report):
    for k in ("n_common",):
        if isinstance(report.get(k), int):
            return report[k]
    m = report.get("metrics")
    if isinstance(m, dict) and isinstance(m.get("numeric_identity"), dict):
        return m["numeric_identity"].get("n")
    return None


def adjudicate(report, source):
    """Grade one comparison report. Pure -- no I/O -- so --selftest can drive it."""
    reasons = []
    n = _n_common(report)
    if n is None:
        return {"artifact": source, "grade": "UNQUOTABLE",
                "reasons": ["report carries no n_common / metrics.numeric_identity.n, so the "
                            "sample size is unknown and no ratio can be quoted"]}

    first_num = _first_diff(report, "first_numeric_divergence")
    first_dec = _first_diff(report, "first_decision_divergence")
    first_top = _first_diff(report, "first_topk_divergence")
    prefix = _prefix_of(report)
    has_drift_block = isinstance(report.get("drift"), dict)

    if not has_drift_block:
        reasons.append("no `drift` block (pre-2026-10-07 artefact): the onset is unrecorded, so "
                       "'no divergence' cannot be distinguished from 'divergence not measured'")

    # A cross-artefact mismatch in the reference is comparability, not drift, and it
    # was the 2026-10-02 root cause (a3fbfb0c vs 0cf5c593, M1 6/222 .. 222/222). Read here
    # only so the reason can be appended; the DEMOTION happens in the grade chain below.
    ref = report.get("ref_md5")

    if first_dec is not None:
        grade = "UNQUOTABLE"
        reasons.append(f"DECISION divergence at step {first_dec}: argmax changed, so this is a real "
                       "behavioural divergence, not numeric drift -- no equivalence may be quoted")
    elif report.get("ref_md5_mismatch"):
        # Comparability dominates every green ratio. This branch was MISSING on first
        # write: the mismatch was recorded as a reason but never demoted the grade, so a
        # 884/884 reading across two different reference files graded CERTIFIABLE --
        # exactly the 2026-10-02 failure this tool exists to prevent. selftest case 7
        # catches it (a reason without an effect is not a check).
        grade = "UNQUOTABLE"
        reasons.append(f"ref_md5 mismatch: {report['ref_md5_mismatch']} -- the two sides were measured "
                       "against DIFFERENT reference files, so no ratio between them answers any question")
    elif not has_drift_block:
        grade = "UNQUOTABLE"
    elif n < MIN_PREFIX:
        grade = "UNQUOTABLE"
        reasons.append(f"n={n} is below the caliber floor MIN_PREFIX={MIN_PREFIX}: too short to "
                       "certify anything (this is the 9-record short probe; gate_registry.py "
                       "already says it cannot show a choice survives)")
    elif first_num is None and prefix is not None and prefix >= CERTIFY_PREFIX:
        grade = "CERTIFIABLE"
        reasons.append(f"no divergence in {prefix} consecutive steps (n={n})")
    elif first_num is None:
        grade = "UNQUOTABLE"
        reasons.append(f"no divergence but only {prefix if prefix is not None else '<unrecorded>'} "
                       f"consecutive steps (n={n}), under CERTIFY_PREFIX={CERTIFY_PREFIX}")
    else:
        grade = "DRIFT_ONLY"
        reasons.append(f"numeric divergence onset at step {first_num}, every argmax identical "
                       f"(n={n}, prefix={prefix}): reduction-order sensitivity propagating along the "
                       "KV chain, NOT a behavioural regression and NOT bit-identical")
        if first_top is not None:
            reasons.append(f"top-k set also diverged at step {first_top} -- candidate set changed, so "
                           "even the 'numeric only' reading is only valid up to that step")

    return {"artifact": source, "grade": grade, "n_compared": n,
            "first_numeric_divergence": first_num, "first_decision_divergence": first_dec,
            "first_topk_divergence": first_top, "bit_identical_prefix": prefix,
            "ref_md5": ref, "reasons": reasons}


def _load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return None  # pair mode
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {"rows": d}


def _run_pair(a, b):
    """Delegate to the compare tool so the numbers come from ONE implementation.

    The compare tool only writes a report when asked for one (`--report`), and its rc is a
    VERDICT (1 = M1 or M2 did not match), not a harness error -- so this writes to an explicit
    path, reads it back, and only treats rc>=2 / a missing report as failure. Treating rc=1 as
    an error would reject every drift reading, i.e. exactly the artefact class this tool exists
    to adjudicate. Parsing stdout instead would be a second, weaker parser for the same numbers.
    """
    import tempfile
    out = Path(tempfile.gettempdir()) / f"drift_adjudicate_pair_{os.getpid()}.json"
    cmd = ["/opt/homebrew/bin/python3", str(COMPARE), "--a", a, "--b", b, "--report", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode >= 2 or not out.exists():
        raise RuntimeError(f"compare tool failed rc={r.returncode}: {r.stderr[-400:]}")
    try:
        return _load(out)
    finally:
        out.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="*", help="oraclecmp_*.json artefacts")
    ap.add_argument("--pairs", nargs="+", metavar="A:B",
                    help="adjudicate jsonl pairs, A:B or A B (colon form keeps them together)")
    ap.add_argument("--json", metavar="PATH", help="also write the verdict json here")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the grader separates the three grades (mutated fixtures, no repo data)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    verdicts, pairs = [], []
    for p in args.reports:
        if Path(p).suffix == ".jsonl":
            pairs.append((p, p))  # nonsense, caught below
            continue
        verdicts.append(adjudicate(_load(p), p))
    for spec in (args.pairs or []):
        if ":" in spec:
            a, b = spec.split(":", 1)
        else:
            pairs.append(None)
            a = b = spec
            continue
        try:
            verdicts.append(adjudicate(_run_pair(a, b), f"{a}|{b}"))
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if not verdicts:
        print("ERROR: no artefacts given. Pass oraclecmp_*.json paths or --pairs A:B.",
              file=sys.stderr)
        return 2

    unquotable = [v for v in verdicts if v["grade"] == "UNQUOTABLE"]
    width = max(len(v["artifact"]) for v in verdicts)
    for v in verdicts:
        print(f"{v['artifact']:<{width}}  {v['grade']}")
        for r in v["reasons"]:
            print(f"{'':<{width}}    - {r}")
    counts = {g: sum(1 for v in verdicts if v["grade"] == g) for g in GRADES}
    print("-" * 74)
    print("  " + "  ".join(f"{g}={counts[g]}" for g in GRADES))
    print("  DRIFT_ONLY is not a pass and not a bug: it is this engine's reduction-order")
    print("  sensitivity propagating along the KV chain. Quote it as such, never as identity.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"verdicts": verdicts, "counts": counts,
                       "caliber": {"min_prefix": MIN_PREFIX, "certify_prefix": CERTIFY_PREFIX},
                       "note": "a DRIFT_ONLY verdict is not quotable as bit-identical; "
                               "9/9 never was (targets.json: 9 steps gave M2 9/9 where 884 gave 25/884)"},
                      f, indent=2, ensure_ascii=False)
        print(f"  verdict -> {args.json}")

    return 1 if unquotable else 0


def _selftest():
    """Each case is a MUTATION of one fixture, so a grader that always returns the
    same grade fails here. Fixtures are hand-built from the real 2026-10-02 numbers."""
    base_drift = {"bit_identical_prefix": 884, "first_numeric_divergence": None,
                  "first_decision_divergence": None, "first_topk_divergence": None}
    cases = [
        ("CERTIFIABLE: 884 clean steps, no divergence",
         {"n_common": 884, "drift": dict(base_drift)}, "CERTIFIABLE"),
        ("DRIFT_ONLY: onset step 78, every argmax identical (the real fen1/fen2 shape)",
         {"n_common": 222, "drift": {"bit_identical_prefix": 78,
                                    "first_numeric_divergence": 78,
                                    "first_decision_divergence": None,
                                    "first_topk_divergence": 162}}, "DRIFT_ONLY"),
        ("UNQUOTABLE: argmax actually changed",
         {"n_common": 222, "drift": {"bit_identical_prefix": 78,
                                    "first_numeric_divergence": 78,
                                    "first_decision_divergence": 82,
                                    "first_topk_divergence": 82}}, "UNQUOTABLE"),
        ("UNQUOTABLE: 9-record short probe, perfectly clean (the 9/9 trap)",
         {"n_common": 9, "drift": {"bit_identical_prefix": 9,
                                   "first_numeric_divergence": None,
                                   "first_decision_divergence": None,
                                   "first_topk_divergence": None}}, "UNQUOTABLE"),
        ("UNQUOTABLE: pre-2026-10-07 artefact with no drift block",
         {"n_common": 222, "diff_examples": [{"key": [78, 0, "DEF"], "numeric_equal": False,
                                             "decision_equal": True}]}, "UNQUOTABLE"),
        ("UNQUOTABLE: sample size unknown",
         {"drift": dict(base_drift)}, "UNQUOTABLE"),
        ("UNQUOTABLE: ref bytes mismatch dominates any green ratio",
         {"n_common": 884, "ref_md5_mismatch": "a3fbfb0c vs 0cf5c593",
          "drift": dict(base_drift)}, "UNQUOTABLE"),
    ]
    ok = True
    for name, report, want in cases:
        got = adjudicate(report, name)["grade"]
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {name}\n         want={want} got={got}")
    print("-" * 74)
    print("SELFTEST " + ("PASS" if ok else "FAIL") +
          "  (a grader that cannot tell DRIFT_ONLY from CERTIFIABLE is not a grader)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())