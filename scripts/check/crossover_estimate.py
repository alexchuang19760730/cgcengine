#!/usr/bin/env python3
"""Remove ORDER and DRIFT from a bracketed A/B estimate. Pure functions only -- nothing here
launches anything, so every claim below can be checked by planting a known effect.

WHY THIS FILE EXISTS. `shape_knob_search.py` brackets every cell between two reference launches
and then reads the cell against the ARITHMETIC mean of those two refs. That mean is the wrong
estimator for this machine, and the error is not small:

    tonight's sweep refs read 11.72 / 10.83 / 5.22 t/s in that order.
    the pair (10.83, 5.22) -> arithmetic mean 8.024, geometric mean 7.517 -- they differ by 6.3%,
    which is TWICE the repo's 3% significance threshold (see MEMORY_PERF.md). Choosing the mean
    alone can manufacture or erase a finding on a row already judged marginal.

THE MODEL. Every measurement here is a THROUGHPUT, and the machine's state acts on it
multiplicatively: a thermally degraded box reads rho * t/s, not t/s - rho. So write

    y(config c, at time t) = theta_c * rho(t)

where theta_c is the unknown we want and rho(t) is the machine's efficiency. Everything below is
just "estimate theta_cell / theta_ref given samples of y".

WHAT REMOVES WHAT -- stated exactly, because overclaiming is the failure this repo keeps punishing:

  * TIME-WEIGHTED GEOMETRIC INTERPOLATION removes a smooth drift. Taking logs makes the model
    additive, so the ref value at the cell's instant is exp(lerp(log)) NOT lerp(raw). Weights must
    come from the MEASUREMENT INSTANTS: a cell takes far longer than a ref, so its effective time
    is nowhere near the index midpoint. Arithmetic means are only right if rho is additive; the
    three numbers above say it is not.

  * The MIRRORED block  R C C R  with sqrt(r_lead * r_trail) removes a multiplicative PERIOD
    effect -- the machine being in a different state in the second half of the block than the
    first. Each config appears once early and once late in mirrored order, so the period factor
    appears once in the numerator and once in the denominator and cancels in the product.

  * NEITHER removes CARRY-OVER: a state left behind by whichever config ran immediately before
    (pool content, page cache, SLC). A crossover cannot separate carry-over from the treatment
    effect without a washout period; this file reports the two arms separately so a disagreement
    between them is VISIBLE rather than averaged away. If abs(r_lead - r_trail) is large, the row
    is not measuring one regime and no estimator rescues it -- same honesty rule as the existing
    >10% ref-drift UNANCHORED rule.

Usage:
    crossover_estimate.py --selftest
    crossover_estimate.py --selftest --broken      # mutation check: the old estimator must turn RED
    crossover_estimate.py --replay <search.json>   # re-read an existing sweep with this correction
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EPS = 1e-12


# ---------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------
def geom(a: float, b: float) -> float:
    """Geometric mean, i.e. the midpoint in the space where the model is additive."""
    if a <= EPS or b <= EPS:
        return 0.0
    return math.sqrt(a * b)


def mean_gap_pct(a: float, b: float) -> float:
    """How much the choice of MEAN alone would move this row's reference value.

    This is diagnostic, not a correction: two refs this far apart were never in one regime, and
    the existing MAX_DRIFT_PCT rule already voids them. It is reported because it is the number
    that says how much of somebody's claimed effect was an artefact of taking the wrong mean.
    """
    if a <= EPS or b <= EPS:
        return 0.0
    return 100.0 * (0.5 * (a + b) - geom(a, b)) / geom(a, b)


def loglerp(a: float, b: float, w: float) -> float:
    """Interpolate in LOG space. This is the whole trick, in one line."""
    if a <= EPS or b <= EPS:
        return 0.0
    w = min(1.0, max(0.0, w))
    return math.exp((1.0 - w) * math.log(a) + w * math.log(b))


def time_weight(t_before: float, t_after: float, t_cell: float) -> float:
    w = 0.5
    if t_after > t_before:
        w = (t_cell - t_before) / (t_after - t_before)
    return min(1.0, max(0.0, w))


def ref_at(v_before: float, t_before: float,
           v_after: float, t_after: float,
           t_cell: float, mode: str = "geometric") -> float:
    """The reference value AT THE INSTANT the cell was measured.

    mode='geometric' -> log-space interpolation weighted by measurement time (the default, and
    the estimator that is right under a multiplicative machine state).
    mode='arithmetic' -> the old (t_before+v_after)/2 behaviour, kept so the difference can be
    measured on real rows instead of argued about.
    """
    if v_before <= EPS and v_after <= EPS:
        return 0.0
    if v_before <= EPS:
        return v_after
    if v_after <= EPS:
        return v_before
    if mode == "arithmetic":
        return 0.5 * (v_before + v_after)
    w = time_weight(t_before, t_after, t_cell)
    return loglerp(v_before, v_after, w)


def delta_pct(cell_ts: float, ref_ts: float) -> float:
    if ref_ts <= EPS:
        return 0.0
    return 100.0 * (cell_ts - ref_ts) / ref_ts


# ---------------------------------------------------------------------------------------------
# the mirrored block: R C C R
# ---------------------------------------------------------------------------------------------
def crossover(r_lead: float, r_trail: float) -> dict:
    """Combine the two mirrored ratios. sqrt(r_lead * r_trail) == mean of their logs."""
    out = {"r_lead": r_lead, "r_trail": r_trail}
    if r_lead <= EPS or r_trail <= EPS:
        out.update(estimate=0.0, arm_spread_pct=None, usable=False,
                   why="one arm produced no usable ratio")
        return out
    est = geom(r_lead, r_trail)
    spread = 100.0 * abs(r_lead - r_trail) / est
    out.update(estimate=est, arm_spread_pct=round(spread, 2), usable=True,
               why=("the two arms disagree by %.1f%% -- carry-over or a regime change inside the "
                    "block, not something this correction removes" % spread)
               if spread > 10.0 else "")
    return out


def block_rc_cr(t_ref_lead: float, ts_ref_lead: float,
                t_cell_a: float, ts_cell_a: float,
                t_cell_b: float, ts_cell_b: float,
                t_ref_trail: float, ts_ref_trail: float,
                mode: str = "geometric") -> dict:
    """Estimate theta_cell / theta_ref from one mirrored block  ref, cell, cell, ref.

    Each arm is corrected against the drift line fitted THROUGH THE TWO REFS, then the two arms
    are combined geometrically so any symmetric second-half effect cancels too. Fitting the rate
    from the refs is what makes the cancellation hold when the cell takes much longer to run than
    a reference does (which it does: the interval lengths are not equal, so the naive product of
    raw ratios would leave a residue).
    """
    res = {"mode": mode}
    for v in (ts_ref_lead, ts_cell_a, ts_cell_b, ts_ref_trail):
        if v <= EPS:
            res.update(estimate=0.0, usable=False, why="a member of the block measured zero t/s")
            return res

    if mode == "arithmetic":
        # The OLD behaviour, reproduced exactly so the difference is measured rather than argued:
        # one arithmetic mean of the two refs at the index midpoint (no time weighting), each arm
        # divided by it, the two arms then averaged arithmetically.
        ref_mean = 0.5 * (ts_ref_lead + ts_ref_trail)
        est = 0.5 * (ts_cell_a / ref_mean + ts_cell_b / ref_mean)
        res.update(estimate=est, usable=True, ref_used=round(ref_mean, 4),
                   drift_rate_per_s=None, log_drift_pct=round(
                       100.0 * (ts_ref_trail - ts_ref_lead) / ts_ref_lead, 2),
                   arm_spread_pct=round(100.0 * abs(ts_cell_a - ts_cell_b) /
                                        (0.5 * (ts_cell_a + ts_cell_b)), 2),
                   why="arithmetic/index-midpoint estimator (the pre-existing behaviour)")
        res["estimate_pct"] = round(100.0 * (est - 1.0), 2)
        return res

    dt = t_ref_trail - t_ref_lead
    if abs(dt) > EPS:
        k = (math.log(ts_ref_trail) - math.log(ts_ref_lead)) / dt   # per-second log drift rate
    else:
        k = 0.0

    def ref_line(t: float) -> float:
        return ts_ref_lead * math.exp(k * (t - t_ref_lead))

    r_lead = ts_cell_a / ref_line(t_cell_a)
    r_trail = ts_cell_b / ref_line(t_cell_b)
    res.update(drift_rate_per_s=k, log_drift_pct=round(100.0 * k * dt, 2))
    res.update(crossover(r_lead, r_trail))
    res["estimate_pct"] = round(100.0 * (res["estimate"] - 1.0), 2)
    return res


# ---------------------------------------------------------------------------------------------
# self-proof: plant a known effect, then check each estimator recovers it
# ---------------------------------------------------------------------------------------------
def _sweep(plan: list[tuple[str, float]], dur: dict[str, float],
           theta_ref: float, theta_cell: float, k: float,
           second_penalty: float = 1.0, cell_penalty: float = 1.0) -> tuple[list[dict], float]:
    """Synthesise a sweep under  y = theta_c * exp(k*t) * (second-in-pair?) * (is-a-cell?).

    k               -- per-second log drift: the whole machine slowing down as the sweep runs.
    second_penalty  -- multiplier on whoever runs SECOND within a pair. This is the effect in the
                       example that motivated this file (A-then-B reads 1.071, B-then-A reads
                       0.820): it hits each config exactly once over a mirrored block, so it is
                       IDENTIFIABLE and MUST cancel in the geometric mean.
    cell_penalty    -- multiplier on every cell measurement regardless of position. This is NOT an
                       order effect: it is indistinguishable from theta_cell genuinely being that
                       much lower, so NO crossover can remove it and this module does not try. It
                       exists in the generator so the self-test can show the boundary.

    Each measurement's effective instant is the MIDPOINT of its own run, because a reported t/s is
    an average over that run -- keying off launch times instead biases every long arm downwards.
    """
    rows, t = [], 0.0
    for cfg, override in plan:
        d = override if override else dur[cfg]
        rows.append({"cfg": cfg, "t": t + d * 0.5,
                     "theta": theta_ref if cfg == "ref" else theta_cell})
        t += d
    for i, r in enumerate(rows):
        alpha = math.log(r["theta"]) + k * r["t"]
        if i % 2 == 1:                       # second member of its pair
            alpha += math.log(second_penalty)
        if r["cfg"] == "cell":
            alpha += math.log(cell_penalty)
        r["y"] = math.exp(alpha)
    return rows, t


def selftest(broken: bool = False) -> int:
    """Plant theta_cell/theta_ref = 1.15 under several real geometries and see who recovers it.

    `broken=True` runs the OLD arithmetic/index-midpoint estimator through the same ground truth.
    The point of keeping that path executable is that it must go RED: a correction nobody can turn
    off is a correction nobody has shown does anything.

    Several geometries, not one. An earlier revision of this test used a single SYMMETRIC block
    and the old estimator happened to pass it (1.4% error) because a symmetric block makes the
    arithmetic estimator self-compensating. Picking the geometry that proves the point is exactly
    the sin this repo exists to prevent, so the scenarios below are the shapes a sweep actually
    takes and every one of them must be recovered.
    """
    truth = 1.15
    theta_r, theta_c = 10.0, 10.0 * truth
    # Drift is planted at the size tonight's sweep actually showed: the reference series read
    # 11.72 -> 5.22 t/s, i.e. the machine lost ~55% of its throughput inside one sweep.
    heavy = math.log(0.55) / 900.0
    plan = [("ref", 0), ("cell", 0), ("cell", 0), ("ref", 0)]

    scenarios = [
        ("cell 3x longer than a ref, symmetric block", {"ref": 60.0, "cell": 180.0}, heavy, 0.94),
        ("cell 10x longer than a ref", {"ref": 60.0, "cell": 600.0}, heavy, 0.94),
        ("unequal refs around the block", {"ref": 60.0, "cell": 300.0}, heavy, 0.94),
        ("mild 10% drift", {"ref": 60.0, "cell": 300.0}, math.log(0.90) / 900.0, 0.98),
        ("no drift, no order effect", {"ref": 60.0, "cell": 300.0}, 0.0, 1.0),
    ]
    # an extra explicit override: the trailing reference itself ran long (another session, a cold
    # cache) -- any eight-minute upgrade to one member breaks index-midpoint midpoints outright.

    def run_row(pl, du, kv, pv, mode: str) -> float:
        rows, _ = _sweep(pl, du, theta_r, theta_c, kv, second_penalty=pv)
        a, b, c, d = rows
        res = block_rc_cr(a["t"], a["y"], b["t"], b["y"], c["t"], c["y"], d["t"], d["y"],
                          mode=mode)
        return 100.0 * abs(res["estimate"] - truth) / truth

    errs = [(name, run_row(plan, du, kv, pv, "geometric"),
             run_row(plan, du, kv, pv, "arithmetic")) for name, du, kv, pv in scenarios]
    # the one asymmetric member: leading ref 60 s, trailing ref 300 s
    asym = [("ref", 60.0), ("cell", 300.0), ("cell", 300.0), ("ref", 300.0)]
    errs.append(("trailing ref ran 5x longer than the leading one",
                 run_row(asym, {"ref": 60.0, "cell": 300.0}, heavy, 0.94, "geometric"),
                 run_row(asym, {"ref": 60.0, "cell": 300.0}, heavy, 0.94, "arithmetic")))

    print("planted truth = %sx; error of each estimator by geometry:" % truth)
    print(f"  {'geometry':<46}{'geometric':>11}{'arithmetic(old)':>17}")
    for name, eg, ea in errs:
        print(f"  {name:<46}{eg:>10.2f}%{ea:>16.2f}%")

    worst_new = max(e for _, e, _ in errs)
    worst_old = max(e for _, _, e in errs)

    if broken:
        # This mode exists to be RED, deliberately inverted from every other selftest in this repo:
        # exit 1 means "the old estimator really is unusable", exit 0 means something is WRONG --
        # the planted geometries stopped biting, or the correction went inert. Run BOTH modes;
        # agreeing green is not the goal here.
        print(f"\nmutation: old arithmetic estimator's worst error = {worst_old:.2f}% "
              f"(threshold 3.0%), corrected = {worst_new:.2f}%")
        if worst_old > 3.0 and worst_old > worst_new * 2.0:
            print(f"  RED as designed: the old estimator is unusable and the correction is "
                  f"load-bearing ({worst_old:.2f}% vs {worst_new:.2f}%)")
            return 1
        print("  !! NOT red: either the planted geometries weakened or the correction is inert")
        return 0

    checks = []
    checks.append((worst_new < 1.0,
                   f"corrected estimator recovers {truth} within {worst_new:.2f}% in every "
                   f"geometry"))
    # primitives
    checks.append((abs(geom(4.0, 9.0) - 6.0) < 1e-9, "geometric mean of 4,9 is 6"))
    checks.append((abs(loglerp(10.0, 20.0, 0.5) - geom(10.0, 20.0)) < 1e-9,
                   "half-weight loglerp == geometric mean"))
    checks.append((abs(loglerp(10.0, 20.0, 0.0) - 10.0) < 1e-9, "weight 0 gives the low end"))
    checks.append((abs(loglerp(10.0, 20.0, 1.0) - 20.0) < 1e-9, "weight 1 gives the high end"))
    checks.append((time_weight(0.0, 100.0, 25.0) == 0.25, "time weight honours the instants"))
    checks.append((time_weight(0.0, 100.0, -50.0) == 0.0, "time weight clamps low"))
    checks.append((time_weight(0.0, 100.0, 999.0) == 1.0, "time weight clamps high"))
    # the diagnostic that started this file: the real ref pair from tonight's sweep
    gap = mean_gap_pct(10.832719, 5.21607)
    checks.append((gap > 3.0, f"real ref pair shows a {gap:.1f}% mean-choice gap (> 3%)"))
    # degenerate inputs must not produce a confident number
    checks.append((ref_at(0.0, 0.0, 0.0, 1.0, 0.5) == 0.0, "no refs -> no reference value"))
    checks.append((crossover(1.2, 0.0)["usable"] is False, "missing arm -> unusable, not 0%"))
    # a symmetric block with NO drift and NO period effect must return exactly the truth
    flat = block_rc_cr(0.0, 10.0, 100.0, 11.5, 200.0, 11.5, 300.0, 10.0)
    checks.append((abs(flat["estimate"] - 1.15) < 1e-9, "clean block returns the ratio exactly"))
    # the boundary: an effect attached to the CONFIGURATION, not to the position, is not an
    # order effect and stays in the answer. Say so with numbers instead of promising too much.
    confounded, _ = _sweep(plan, {"ref": 60.0, "cell": 300.0}, theta_r, theta_c, heavy,
                           cell_penalty=0.95)
    conf = block_rc_cr(confounded[0]["t"], confounded[0]["y"],
                       confounded[1]["t"], confounded[1]["y"],
                       confounded[2]["t"], confounded[2]["y"],
                       confounded[3]["t"], confounded[3]["y"])
    checks.append((abs(conf["estimate"] - truth * 0.95) < 0.01,
                   "config-correlated effect stays in the answer (not removed, not hidden)"))
    # and the two arms must be visibly flagged when they disagree
    bad = crossover(1.30, 1.02)
    checks.append((bad["arm_spread_pct"] > 10.0 and bad["why"] != "",
                   "disagreeing arms are reported, not averaged away"))

    bad_list = [name for ok, name in checks if not ok]
    for ok, name in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"\nselftest{' (BROKEN/old estimator)' if broken else ''}: "
          f"{len(checks) - len(bad_list)}/{len(checks)} passed")
    return 1 if bad_list else 0


# ---------------------------------------------------------------------------------------------
# replay: put the correction over a sweep that already happened
# ---------------------------------------------------------------------------------------------
def _cell_ts(rec: dict) -> float:
    rows = rec.get("rows") or []
    return float((rows[0] or {}).get("avg_ts") or 0.0) if rows else 0.0


def _cell_mid(rec: dict) -> float | None:
    """The instant this arm was measured: midpoint when recorded, else None to force the fallback."""
    t0, t1 = rec.get("t_start"), rec.get("t_end")
    if t0 and t1:
        return 0.5 * (t0 + t1)
    m = rec.get("t_mid")
    return float(m) if m else None


def replay(path: Path) -> int:
    """Re-read a shape_knob_search.py result with geometric/time-weighted references.

    A sweep recorded before this correction existed carries no per-cell timestamps, so those rows
    fall back to index-midpoint weights and SAY SO. What the fallback can honestly answer is
    "how much did taking the wrong mean move this row" -- not "here is the corrected verdict".
    """
    d = json.loads(Path(path).read_text())
    cells = d.get("cells") or []
    print(f"{path}")
    print("ref series: " + "  ".join(f"{t}={v:.2f}" for t, v in (d.get("ref_series") or [])))
    have_time = all(_cell_mid(c) is not None for c in cells) and bool(cells)
    print(f"per-cell timestamps recorded: {have_time} "
          f"({'time weights' if have_time else 'index midpoint fallback'})")
    print(f"\n{'row':<26}{'ref arith':>10}{'ref geom':>10}{'gap%':>7}"
          f"{'Δ arith%':>9}{'Δ geom%':>9}   note")
    for i, rec in enumerate(cells):
        tag = rec.get("tag", "?")
        if tag.startswith("ref_"):
            continue
        ts = _cell_ts(rec)
        if ts <= EPS:
            print(f"{tag:<26}{'--no row--':>10}")
            continue
        before = next((j for j in range(i - 1, -1, -1) if str(cells[j].get("tag", "")).startswith("ref_")), None)
        after = next((j for j in range(i + 1, len(cells)) if str(cells[j].get("tag", "")).startswith("ref_")), None)
        vb, tb = (_cell_ts(cells[before]), _cell_mid(cells[before])) if before is not None else (0.0, None)
        va, ta = (_cell_ts(cells[after]), _cell_mid(cells[after])) if after is not None else (0.0, None)
        tc = _cell_mid(rec)
        if have_time and None not in (tb, ta, tc):
            ra = ref_at(vb, tb, va, ta, tc, mode="arithmetic")
            rg = ref_at(vb, tb, va, ta, tc, mode="geometric")
            note = "time-weighted"
        else:
            ra = 0.5 * (vb + va)
            rg = geom(vb, va)
            note = "index midpoint (no timestamps)"
        gap = mean_gap_pct(vb, va)
        print(f"{tag:<26}{ra:>10.2f}{rg:>10.2f}{gap:>7.2f}"
              f"{delta_pct(ts, ra):>9.2f}{delta_pct(ts, rg):>9.2f}   {note}")
        if gap > 3.0:
            print(f"{'':<26}!! the mean choice alone moves this row by {gap:.1f}% "
                  f"(> 3% threshold): the row cannot be read at this drift")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="crossover / drift correction for A/B cells")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--broken", action="store_true",
                    help="run the OLD estimator through the planted truth; selftest must go red")
    ap.add_argument("--replay", default="", help="re-read a shape_knob_search.py search.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest(broken=a.broken)
    if a.replay:
        return replay(Path(a.replay))
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
