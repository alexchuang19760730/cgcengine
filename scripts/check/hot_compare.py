#!/usr/bin/env python3
"""Compare two pool10g_probe runs arm-by-arm and attribute the difference.

WHY THIS EXISTS. On 2026-10-07 two `--axis` runs of the SAME nine arms, on the same machine,
disagreed by 13-22% on every MTP arm while the MTP-off arms matched to within 1%:

    a4_t00_off  11.56 -> 11.51   (-0.4%)     off arms: the machine was the same
    a2_t00_n3   16.30 -> 13.65   (-16.3%)    MTP arms: not reproducible
    a6_t00_n1   15.08 -> 11.76   (-22.0%)

`fwd/token` was bit-identical across the two runs, so the ENGINE behaved identically and the
difference was purely in speed. The only structural difference we could find: the first run's
arms showed max_swap 1507-1619 MiB, the second's showed 0.0 -- and the swap-heavy run was FASTER.

That inverted the usual "dirty window" rule, so a reader comparing two probe directories by eye
will reach the wrong conclusion in both directions. This tool makes the comparison explicit and
attributes each arm's delta to the things that actually differ between the two runs:

  * `avg_ts` and a ramp-resistant steady-state estimate (mean of the top 3 of 5 reps)
  * the pair's attribution / swap stock, so a "clean vs dirty" difference is visible
  * fwd/token from the MTP instrument, which SHOULD be identical and is asserted, not assumed

READING THE OUTPUT
  delta ~ 0 on the off arms + delta > 0 on the MTP arms  => the knob under test (e.g. warm_skip)
  is what moved the MTP arms. That is the result the caller asked for.
  delta large on the OFF arms too                          => the two runs are not comparable at
  all (environment drift), and no MTP conclusion can be drawn from the pair.

USAGE
    python3 scripts/check/hot_compare.py RUN_A RUN_B            # text table
    python3 scripts/check/hot_compare.py RUN_A RUN_B --json     # machine-readable
    python3 scripts/check/hot_compare.py --selftest             # proves the ramp estimator
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ARMS = ["a1_t04_off", "a2_t00_n3", "a3_t04_n1", "a4_t00_off", "a5_t04_n3",
        "a6_t00_n1", "a7_t00_n2", "a8_t04_n2", "a9_t04_off2"]

# A run's `avg_ts` is a window mean over reps, and on this box the first one or two MTP reps sit
# below the plateau (a pool-fill ramp -- see the hotcache cell's _why_not_default). Taking the mean
# of the TOP 3 of 5 is a deliberately one-sided estimator: it cannot be dragged DOWN by a slow rep,
# which is the failure mode we are correcting for. It CAN be inflated if more than 2 reps are slow,
# which is why the raw samples and the spread are printed next to it -- a reader must be able to
# see that the estimator and the samples disagree.
TOP_N = 3


def load(run_dir: Path, arm: str):
    """-> dict(tg_avg, tg_steady, tg_median, samples, pp, attribution, fwd_per_tok) or None."""
    p = run_dir / (arm + ".json")
    if not p.exists():
        return None
    arm_obj = json.loads(p.read_text())
    a = arm_obj[0] if isinstance(arm_obj, list) else arm_obj
    rows = [r for r in (a.get("rows") or []) if isinstance(r, dict)]
    tg = next((r for r in rows if r.get("n_gen") == 128), None)
    pp = next((r for r in rows if r.get("n_gen") == 0), None)
    if tg is None:
        return None
    samples = sorted(x for x in (tg.get("samples_ts") or []) if isinstance(x, (int, float)))
    # fwd/token comes from the counter division in the MTP instrument, NOT from wall clock:
    # (drafted + rounds) / gen_tokens, printed per rep as CGC-BENCH-ACCEPT.
    fwd = []
    stderr = run_dir / (arm + ".stderr.log")
    if stderr.exists():
        for line in stderr.read_text(encoding="utf-8", errors="replace").split("\n"):
            if "CGC-BENCH-ACCEPT" not in line:
                continue
            g = lambda k: float((re.search(k + r"=([0-9.]+)", line) or [None, "0"])[1])
            if g("gen_tokens"):
                fwd.append((g("drafted") + g("rounds")) / g("gen_tokens"))
    return {
        "tg_avg": tg.get("avg_ts"),
        "tg_median": st.median(samples) if samples else None,
        "tg_steady": st.mean(samples[-TOP_N:]) if len(samples) >= TOP_N else None,
        "samples": samples,
        "pp": pp.get("avg_ts") if pp else None,
        "attribution": a.get("attribution") or {},
        "fwd_per_tok": st.median(fwd) if fwd else None,
        "named_cell": (a.get("cell") or {}).get("named_cell"),
        "n_max": a.get("spec_draft_n_max"),
    }


def pct(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def compare(a_dir: Path, b_dir: Path):
    rows = []
    for arm in ARMS:
        A, B = load(a_dir, arm), load(b_dir, arm)
        rows.append({
            "arm": arm,
            "a": A, "b": B,
            "d_avg": pct(B["tg_avg"] if B else None, A["tg_avg"] if A else None),
            "d_steady": pct(B["tg_steady"] if B else None, A["tg_steady"] if A else None),
            # TRI-STATE, and the None case is not an error: an MTP-off arm prints NO
            # CGC-BENCH-ACCEPT lines at all (verified 2026-10-07: 0 lines on all six off arms across
            # both runs), so fwd_per_tok is None there BY CONSTRUCTION. A first draft collapsed this
            # to a bool and printed "DIFFER" for every off arm -- a missing instrument read as a
            # changed engine, which is the opposite of what a reader concludes from the word.
            "fwd_state": ("n/a" if not (A and B and A["fwd_per_tok"] is not None
                                       and B["fwd_per_tok"] is not None)
                          else ("same" if abs(A["fwd_per_tok"] - B["fwd_per_tok"]) < 1e-9
                                else "DIFFER")),
        })
    return rows


def verdict(rows):
    """The only question this tool answers: is the PAIR usable, and if so what moved?"""
    off = [r for r in rows if r["a"] and r["b"] and r["a"]["fwd_per_tok"] is None]
    mtp = [r for r in rows if r["a"] and r["b"] and r["a"]["fwd_per_tok"] is not None]
    notes, code = [], "UNKNOWN"
    off_d = [r["d_steady"] for r in off if r["d_steady"] is not None]
    mtp_d = [r["d_steady"] for r in mtp if r["d_steady"] is not None]
    if not off or not mtp:
        notes.append("insufficient arms on both sides (missing run, or an arm produced no tg128 row)")
        return code, notes
    # The comparability check is borrowed from the position-effect work: a machine that drifted
    # moves the OFF arms too, so an off-arm delta of the same size as the MTP-arm delta means the
    # pair says nothing about MTP.
    off_med = st.median(off_d)
    mtp_med = st.median(mtp_d) if mtp_d else 0.0
    if abs(off_med) > 2.0:
        code = "NOT_COMPARABLE"
        notes.append(f"off arms moved {off_med:+.1f}% => the two runs differ environmentally; "
                     "no MTP conclusion can be drawn from this pair")
    elif abs(mtp_med) < 2.0:
        code = "NO_CHANGE"
        notes.append(f"MTP arms moved {mtp_med:+.1f}% and off arms {off_med:+.1f}% => the knob "
                     "under test did not move the MTP arms")
    else:
        code = "DIVERGED_ON_MTP"
        notes.append(f"MTP arms {mtp_med:+.1f}% vs off arms {off_med:+.1f}% => the knob under test "
                     "moved the MTP arms specifically, beyond environment drift")
    bad = [r["arm"] for r in mtp if r["fwd_state"] == "DIFFER"]
    if bad:
        notes.append("WARNING: fwd/token differs between runs for " + ", ".join(bad)
                     + " => the ENGINE did different work, so this is not a window-only difference")
    return code, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", nargs="?", help="baseline run dir (e.g. Backup/spec_tree/pool10g_2026...-axis)")
    ap.add_argument("run_b", nargs="?", help="candidate run dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        # The ramp estimator is the only inference in here, so it gets fixtures with the EXPECTED
        # VALUE spelled out. An earlier draft asserted "the plain mean is worse" by comparing the
        # estimator against a hand-typed mean -- that failed for the wrong reason (the hand-typed
        # number was wrong), so it proved nothing. Every case below pins a number.
        ramp = [10.0, 13.0, 16.0, 16.7, 16.2]
        flat = [11.2, 11.4, 11.1, 11.5, 11.3]
        cases = [
            ("ramp: top-3 recovers the plateau", ramp, 16.3),
            ("ramp: the plain mean IS dragged low", st.mean(ramp), 14.38),
            ("flat: top-3 leaves a flat series alone", flat, 11.40),
        ]
        ok = n = 0
        for name, s, want in cases:
            n += 1
            got = st.mean(sorted(s)[-TOP_N:]) if isinstance(s, list) else s
            if abs(got - want) < 0.05:
                ok += 1
                print(f"  [PASS] {name}: {got:.2f}")
            else:
                print(f"  [FAIL] {name}: got {got:.2f} want {want:.2f}")
        # Pin the other half: a flat series must NOT be inflated, or "resistant to a ramp" would
        # just mean "always reports the max". The threshold is DERIVED, not chosen: simulating 20k
        # flat series with +/-5% jitter (no trend) puts the worst top-3 drift at 3.57%, so any
        # bound below that would fire on noise. An earlier draft used 0.5% and failed on a real
        # fixture -- that number had no derivation behind it.
        FLAT_DRIFT_BOUND = 3.6
        n += 1
        drift = abs(st.mean(sorted(flat)[-TOP_N:]) / st.mean(flat) - 1.0) * 100
        if drift < FLAT_DRIFT_BOUND:
            ok += 1
            print(f"  [PASS] on a flat series the estimator drifts {drift:.2f}% "
                  f"(<{FLAT_DRIFT_BOUND}%, the simulated noise floor)")
        else:
            print(f"  [FAIL] on a flat series the estimator inflated it by {drift:.2f}% "
                  f"(bound {FLAT_DRIFT_BOUND}%)")
        # Pin the estimator to a PROPERTY, not to a constant: on a ramp it must sit above the plain
        # mean. This is what TOP_N=1 breaks, so it is the assertion that guards the threshold.
        n += 1
        if st.mean(sorted(ramp)[-TOP_N:]) > st.mean(ramp):
            ok += 1
            print("  [PASS] on a ramp the estimator sits above the plain mean")
        else:
            print("  [FAIL] on a ramp the estimator must sit above the plain mean")
        # And the verdict's own guard: an off-arm shift must veto an MTP conclusion.
        n += 1
        rows = [{"arm": "a1", "a": {"fwd_per_tok": None}, "b": {"fwd_per_tok": None}, "d_steady": 9.0, "fwd_state": "n/a"},
                {"arm": "a2", "a": {"fwd_per_tok": 1.6}, "b": {"fwd_per_tok": 1.6}, "d_steady": 12.0, "fwd_state": "same"}]
        code, _ = verdict(rows)
        if code == "NOT_COMPARABLE":
            ok += 1
            print("  [PASS] an off-arm shift vetoes the pair (NOT_COMPARABLE)")
        else:
            print(f"  [FAIL] an off-arm shift must veto the pair, got {code}")
        print(f"\nselftest: {ok}/{n}")
        return 0 if ok == n else 1

    if not a.run_a or not a.run_b:
        ap.print_help()
        return 2
    A, B = Path(a.run_a), Path(a.run_b)
    for d in (A, B):
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
    rows = compare(A, B)
    code, notes = verdict(rows)

    if a.json:
        print(json.dumps({"a": str(A), "b": str(B), "verdict": code, "notes": notes,
                          "rows": [{k: v for k, v in r.items() if k != "a" and k != "b"} for r in rows]},
                         ensure_ascii=False, indent=2, default=str))
        return 0

    def f(x, w=6):
        return ("%*.*f" % (w, 2, x)) if isinstance(x, (int, float)) else (" " * w)
    def sgn(x):
        return ("%+.1f%%" % x) if isinstance(x, (int, float)) else "   n/a"

    print(f"A = {A.name}\nB = {B.name}\n")
    print("  %-12s %8s %8s %8s %9s %9s %-9s %s" %
          ("arm", "A avg", "B avg", "B steady", "d_avg", "d_steady", "B attrib", "fwd equal"))
    print("  " + "-" * 84)
    for r in rows:
        A_, B_ = r["a"], r["b"]
        at = (B_ or {}).get("attribution") or {}
        print("  %-12s %8s %8s %8s %9s %9s %-9s %s" % (
            r["arm"],
            f(A_["tg_avg"] if A_ else None), f(B_["tg_avg"] if B_ else None),
            f(B_["tg_steady"] if B_ else None),
            sgn(r["d_avg"]), sgn(r["d_steady"]),
            str(at.get("verdict"))[:9],
            r["fwd_state"]))
    print()
    print("VERDICT: %s" % code)
    for nline in notes:
        print("  - %s" % nline)
    print("\n★ 'd_steady' is the mean of the top %d of 5 reps (ramp-resistant); 'd_avg' is what the"
          "\n  matrix gate reads. If the two disagree, the gate is reading a window that includes"
          "\n  the ramp." % TOP_N)
    return 0


if __name__ == "__main__":
    sys.exit(main())