#!/usr/bin/env python3
"""bracketed_ab.py -- a protocol for 3%-class arm comparisons on a box whose throughput drifts.

THE PROBLEM, MEASURED
    Launches of ONE configuration on this machine spanned 5.84-12.83 t/s (2.2x) with bit-identical
    outputs. Enforcing thermal Nominal at launch cut the k=2 arm's spread from 18.5% to 4.8%, but
    the k=3 arm still span 1.43x and the paired difference (5 rounds, ABBA-rotated) had sd 1.83 t/s
    -- a single-pair 2-sigma resolution of ~3.5 t/s, i.e. 25%. Treating launches as independent and
    averaging is hopeless at this scale: resolving 3% of 12 t/s (+-0.36) that way needs n ~ 98
    pairs. So the covariate has to be REMOVED, not averaged.

THE COVARIATE
    "Which launch in this sequence" is measurable, and it is not noise: within a sequence the
    readings decay with the launch index (the pre-gate 6-arm null sequence: 10.46, 5.84, 8.79, 8.93,
    8.05, 7.69) while the reference arm is textually identical every time. Other per-launch
    covariates are recorded too (gap since the previous arm, swap in use, thermal level, and the
    pool partition, which this line measured to differ across launches at 33 vs 36 segments).

THREE ESTIMATORS, AND WHAT EACH CAN SURVIVE
    1. paired()      -- ABBA rounds, first minus second. Survives nothing: it assumes the pairs are
                        drawable from one distribution, which is exactly what drift breaks.
    2. adjusted()    -- OLS of log(tps) on [arm, order] (plus any extra covariates). Removes a
                        LINEAR-in-index drift; the residual sd is what the CI is built from.
    3. bracketed()   -- the protocol this tool exists for. Pattern R T R T R T R: every test arm is
                        BRACKETED by the reference arm, and each block's contrast is
                        T - (R_before + R_after)/2. A linear drift cancels EXACTLY inside the
                        bracket, without estimating it, so the estimate is immune to the level and
                        slope of the drift. This is the one to run when the answer matters.

    Which one wins depends on the drift's SHAPE, which is why this file carries a simulator: it will
    tell you the effect a protocol can actually resolve on this box rather than assuming.

RECORDED FORMAT (one CSV, one row per launch; `--analyze`)
    arm,order,tps[,gap_s,swap_mb,thermal,partition]
        arm       arm label (e.g. k2 / k3, or R / T for the bracket protocol)
        order     launch index within the window, 1-based
        tps       the reading (decode t/s)
        others    optional covariates, used by `--covariates`

    bracketed_ab.py --selftest
    bracketed_ab.py --simulate --effect 0.03 --drift geometric --drift-pct 8 --noise 0.05
    bracketed_ab.py --analyze arms.csv --ref R --test T
"""
from __future__ import annotations

import argparse
import random
import statistics as st
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

Z80 = 2.802          # 2-sided 80% power: (z_{1-a/2} + z_{0.8}) = 1.96 + 0.842


@dataclass
class Row:
    arm: str
    order: int
    tps: float
    cov: dict = field(default_factory=dict)


# ----------------------------------------------------------------- estimators

import math


def summarize_log(logs):
    """Per-block log-ratios -> the effect as a fraction, with its CI, in the SAME units every
    estimator here reports (a relative effect), because an absolute t/s difference is meaningless
    across a window whose level moves by 2x."""
    if not logs:
        return {"n": 0, "effect": float("nan"), "sd_log": float("nan"), "se_log": float("nan"),
                "ci95_frac": float("nan"), "resolvable_80pct_frac": float("nan")}
    m = st.mean(logs)
    sd = st.stdev(logs) if len(logs) > 1 else float("nan")
    se = sd / len(logs) ** 0.5 if len(logs) > 1 else float("nan")
    return {"n": len(logs), "effect": math.exp(m) - 1, "sd_log": sd, "se_log": se,
            "ci95_frac": 1.96 * se, "resolvable_80pct_frac": Z80 * se, "mean_log": m}


def paired(rows, ref, test):
    """The naive one: consecutive (ref, test) pairs, each as a log-ratio.

    Survives nothing -- it assumes the two arms of a pair are drawable from one distribution, which
    is the assumption drift breaks. Kept as the baseline the other two have to beat.
    """
    seq = sorted(rows, key=lambda r: r.order)
    logs, i = [], 0
    while i < len(seq) - 1:
        a, b = seq[i], seq[i + 1]
        if {a.arm, b.arm} == {ref, test}:
            logs.append(math.log((b.tps if b.arm == test else a.tps) /
                                 (a.tps if a.arm == ref else b.tps)))
            i += 2
        else:
            i += 1
    return summarize_log(logs)


def bracketed(rows, ref, test, mid="geometric"):
    """R T R brackets, as a RATIO against the reference value AT the test's position.

    `mid="geometric"` (sqrt of the two references) is exact for a drift that is geometric in launch
    index, which is the shape this box shows; `mid="arithmetic"` (the mean) is exact for a drift that
    is linear in the index. Neither is exact for the other shape, and the residue is second order:
    under geometric drift the arithmetic mid leaves log(2r/(1+r^2)) (r=1.08 -> -0.30%), so the two
    forms are BOTH reported: their disagreement is this protocol's own model error, and it has to be
    smaller than the effect being claimed (see the doc). Using a difference instead of a ratio is
    the trap this avoids: a difference grows with the drift LEVEL, so averaging differences across
    a window whose level moves is averaging different quantities.
    """
    seq = sorted(rows, key=lambda r: r.order)
    logs = []
    for i, r in enumerate(seq):
        if r.arm != test or i == 0 or i + 1 >= len(seq):
            continue
        before, after = seq[i - 1], seq[i + 1]
        if before.arm == ref and after.arm == ref:
            m = (math.sqrt(before.tps * after.tps) if mid == "geometric"
                 else 0.5 * (before.tps + after.tps))
            logs.append(math.log(r.tps / m))
    return summarize_log(logs)


def _solve(a, b):
    """Small dense solve (Gaussian elimination with partial pivoting); no numpy dependency."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def adjusted(rows, ref, test, covariates=()):
    """OLS of log(tps) on [1, arm, order, covariates...]. Returns the arm contrast (log ratio),
    its SE, and how much of the residual variance the covariates removed.

    log because drift here looks multiplicative (a 2.2x span at a fixed configuration), which makes
    a linear-in-index fit a model of a constant *rate* of decay rather than a decelerating one.
    """
    covs = list(covariates)
    y = [math.log(r.tps) for r in rows]
    cols = [[1.0] * len(rows), [1.0 if r.arm == test else 0.0 for r in rows],
            [float(r.order) for r in rows]]
    dropped = []
    for c in covs:
        try:
            vals = [float(r.cov.get(c, "")) for r in rows]
        except (TypeError, ValueError):
            # A label (e.g. thermal=NOMINAL) is NOT a zero: excluding it and saying so is the only
            # honest option, because silently coding it as 0 puts a category into a regression as a
            # magnitude. (This crashed before it could do either.)
            dropped.append(c)
            continue
        mu, sd = st.mean(vals), (st.pstdev(vals) or 1.0)
        cols.append([(v - mu) / sd for v in vals])
    p = len(cols)
    xtx = [[sum(cols[i][k] * cols[j][k] for k in range(len(rows))) for j in range(p)] for i in range(p)]
    xty = [sum(cols[i][k] * y[k] for k in range(len(rows))) for i in range(p)]
    try:
        beta = _solve(xtx, xty)
    except ValueError:
        return {"n": len(rows), "log_effect": float("nan"), "se": float("nan"), "t": float("nan")}
    resid = [y[k] - sum(beta[i] * cols[i][k] for i in range(p)) for k in range(len(rows))]
    dof = len(rows) - p
    s2 = sum(r * r for r in resid) / dof if dof > 0 else float("nan")
    try:
        inv_diag = _inv_diag(xtx)
    except ValueError:
        return {"n": len(rows), "log_effect": beta[1], "se": float("nan"), "t": float("nan")}
    se = (s2 * inv_diag[1]) ** 0.5
    # variance removed by the drift term alone: fit without `order` for the comparison
    resid_no = _resid_without_order(rows, ref, test)
    var_all, var_no = _var(resid), _var(resid_no)
    if var_no in (0, None) or var_no != var_no:
        removed = 0.0            # no spread to remove (a fixture case, not a nan to be read as evidence)
    else:
        removed = 1 - var_all / var_no
    return {"n": len(rows), "log_effect": beta[1], "se": se, "t": beta[1] / se if se else float("nan"),
            "resid_sd": s2 ** 0.5 if s2 == s2 else float("nan"),
            "drift_slope_per_launch": beta[2], "dropped_covariates": dropped,
            "var_removed_by_order": removed}


def _var(xs):
    return st.pvariance(xs) if len(xs) > 1 else float("nan")


def _resid_without_order(rows, ref, test):
    yt = [math.log(r.tps) for r in rows if r.arm == test]
    yr = [math.log(r.tps) for r in rows if r.arm == ref]
    if not yt or not yr:
        return []
    mt, mr = st.mean(yt), st.mean(yr)
    return [v - mt for v in yt] + [v - mr for v in yr]


def _spearman(xs, ys):
    """Spearman rho with AVERAGE ranks for ties.

    Ties matter here and must not be left to sort stability: in the certified window two of the five
    k=2 readings are the same value (12.20, 12.20), and assigning those two ranks 2 and 3 in input
    order rather than 2.5 and 2.5 moved rho from -0.15 to -0.10 -- a sign-flipping difference on a
    number that was being quoted as the evidence for "the covariate is arm-specific".
    """
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out
    a, b = ranks(xs), ranks(ys)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    den = (sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b)) ** 0.5
    return num / den if den else float("nan")


# At n=5 a rank correlation needs |rho| >= 0.90 to be worth anything at p<0.05, so the per-arm rho
# below is a description of five launches, not a significance claim about the covariate.
RHO_N_FLOOR = {4: 1.00, 5: 0.90, 6: 0.83, 7: 0.71, 8: 0.64, 9: 0.60, 10: 0.56}


def _inv_diag(xtx):
    """Diagonal of the inverse, via solving X'X z = e_i for each i (small p)."""
    n = len(xtx)
    out = []
    for i in range(n):
        e = [1.0 if j == i else 0.0 for j in range(n)]
        out.append(_solve(xtx, e)[i])
    return out


# ----------------------------------------------------------------- simulator

def simulate(n_blocks, effect, drift, drift_pct, noise, contention, rng, pattern="RTR"):
    """One simulated window. Returns rows with the TRUTH known.

    drift shapes:
      none        no drift (the sanity cell: every estimator must be unbiased here)
      linear      tps *= (1 + drift_pct * (order-1)/100)
      geometric   tps *= (1 + drift_pct/100) ** (order-1)      <- what this box looks like
    contention:  with probability `contention`, multiply by 1+U(0, c) -- the one-sided half (a
                 neighbour on the GPU only ever slows a reading down)
    """
    rows, order = [], 1
    if pattern != "RTR":
        raise ValueError("only the RTR pattern is implemented")
    # R T R T ... R: `n_blocks` test arms, each bracketed by a reference on both sides, so the
    # sequence starts and ends with R and has odd length. (The first version indexed the string by
    # i%3 -- "RTRRTR" -- which produced R T R R T R: every third test arm was NOT bracketed, and
    # the check that was supposed to prove drift cancellation was measuring the drift instead.)
    arms = ["R" if i % 2 == 0 else "T" for i in range(2 * n_blocks + 1)]
    for a in arms:
        base = 12.0
        if drift == "linear":
            base *= 1 + drift_pct * (order - 1) / 100.0
        elif drift == "geometric":
            base *= (1 + drift_pct / 100.0) ** (order - 1)
        elif drift != "none":
            raise ValueError(drift)
        v = base * (1 + effect if a == "T" else 1.0)
        v *= 1 + rng.gauss(0, noise)
        if contention and rng.random() < contention:
            v *= 1 + rng.uniform(0, 0.25)
        rows.append(Row(a, order, v))
        order += 1
    return rows


def run_simulation(args):
    """Power table: for each estimator, what it recovers and what it could resolve.

    Everything is in FRACTIONS of the reference level, because that is the only unit a comparison
    across a drifting window can be stated in.
    """
    rng = random.Random(args.seed)
    truth = args.effect
    print("simulated window: %d blocks (R T R T .. R), effect %+.1f%% when true, drift %s "
          "%.0f%%/launch, noise %.1f%%, one-sided contention on %.0f%% of arms"
          % (args.blocks, 100 * truth, args.drift, args.drift_pct, 100 * args.noise,
             100 * args.contention))
    ests = {
        "paired": ([], 2, lambda r: paired(r, "R", "T")),
        "adjusted": ([], 1, lambda r: _as_stat(adjusted(r, "R", "T"))),
        "bracketed": ([], 3, lambda r: bracketed(r, "R", "T")),
        "bracket-arith": ([], 3, lambda r: bracketed(r, "R", "T", mid="arithmetic")),
    }
    for _ in range(args.reps):
        rows = simulate(args.blocks, truth, args.drift, args.drift_pct, args.noise,
                        args.contention, rng)
        for name, (acc, _cost, fn) in ests.items():
            s = fn(rows)
            if s["effect"] == s["effect"]:
                acc.append(s)
    print("\n%-14s %8s %9s %9s %11s %13s %11s"
          % ("estimator", "recovers", "bias(pp)", "sd/block", "resolves@80", "CI95 of mean", "launches/3%"))
    for name, (acc, cost, _fn) in ests.items():
        if not acc:
            print("%-14s no estimate" % name)
            continue
        eff = [s["effect"] for s in acc]
        bias = 100 * (st.mean(eff) - truth)
        sds = [s["sd_log"] for s in acc if s["sd_log"] == s["sd_log"]]
        ses = [s["se_log"] for s in acc]
        sd_block = st.mean(sds) if sds else float("nan")
        print("%-14s %+7.2f%% %+9.2f %9.4f %10.2f%% %12.2f%% %11.0f"
              % (name, 100 * st.mean(eff), bias, sd_block, 100 * Z80 * st.mean(ses),
                 100 * 1.96 * st.mean(ses),
                 (Z80 * sd_block / 0.03) ** 2 * cost if sd_block == sd_block else float("nan")))
    print("\nsd/block is the per-contrast scatter in log space; `launches/3%` is the number of LAUNCHES")
    print("needed to detect a true 3% effect at 80% power (x2 for paired, x3 for a bracket, x1 for")
    print("the regression). `bracket-arith` is the same protocol read with the arithmetic mid: its")
    print("bias is this protocol's own model error, and it must stay well under the claimed effect.")


def _as_stat(d):
    """Adapt the regression result to the same shape as the other estimators (fraction + log SE)."""
    return {"n": d.get("n", 0), "effect": math.exp(d.get("log_effect", float("nan"))) - 1,
            "se_log": d.get("se", float("nan")), "sd_log": d.get("resid_sd", float("nan"))}


# ----------------------------------------------------------------- real data

def load(path: Path) -> list:
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("arm,"):
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append(Row(parts[0], int(parts[1]), float(parts[2]),
                        {k: parts[3 + j] for j, k in enumerate(("gap_s", "swap_mb", "thermal", "partition"))
                         if 3 + j < len(parts)}))
    return rows


def run_analysis(path, ref, test, covariates):
    rows = load(path)
    print("rows: %d arms (%s x%d, %s x%d), order %d..%d"
          % (len(rows), ref, sum(1 for r in rows if r.arm == ref),
             test, sum(1 for r in rows if r.arm == test),
             min(r.order for r in rows), max(r.order for r in rows)))
    # The covariate's own description, per arm: this is the table the protocol is justified by, so it
    # comes from the tool rather than from an ad-hoc script (which is how the tie-handling bug above
    # survived into a document).
    for arm in (ref, test):
        vs = [r.tps for r in rows if r.arm == arm]
        os_ = [r.order for r in rows if r.arm == arm]
        if len(vs) < 3:
            continue
        m, sd = st.mean(vs), st.stdev(vs)
        rho = _spearman(os_, vs)
        floor = RHO_N_FLOOR.get(len(vs))
        print("  %-13s n=%d mean %.2f sd %.2f (%.1f%%) spread %.2f-%.2f  rho(order,tps) %+.2f%s"
              % (arm, len(vs), m, sd, 100 * sd / m, min(vs), max(vs), rho,
                 "" if floor is None or abs(rho) >= floor else "  (below the n=%d floor %.2f)"
                 % (len(vs), floor)))
    print()
    named = (("paired", paired(rows, ref, test)),
             ("bracketed", bracketed(rows, ref, test)),
             ("bracket-arith", bracketed(rows, ref, test, mid="arithmetic")))
    for name, s in named:
        if s["n"]:
            print("  %-13s n=%-3d effect %+7.2f%%  per-block sd %.4f  CI95 +-%5.2fpp"
                  % (name, s["n"], 100 * s["effect"], s["sd_log"], 100 * s["ci95_frac"]))
        else:
            print("  %-13s no contrast (the pattern does not bracket a test arm with references)"
                  % name)
    a = adjusted(rows, ref, test, covariates)
    if a.get("log_effect") == a.get("log_effect"):
        print("  %-13s n=%-3d effect %+7.2f%%  t %6.2f  drift %+8.5f/launch  "
              "order removed %.0f%% of the arm-only residual variance"
              % ("adjusted", a["n"], 100 * (math.exp(a["log_effect"]) - 1), a["t"],
                 a["drift_slope_per_launch"], 100 * a["var_removed_by_order"]))
        if a.get("dropped_covariates"):
            print("  (non-numeric covariate(s) excluded rather than coded as 0: %s)"
                  % ", ".join(a["dropped_covariates"]))
        if named[1][1]["n"] and named[2][1]["n"]:
            gap = 100 * abs(named[1][1]["effect"] - named[2][1]["effect"])
            print("  protocol's own model error (the two bracket forms disagree by): %.2fpp%s"
                  % (gap, "  <-- larger than a 3% claim: do not quote the effect" if gap > 1.5 else ""))
    return 0


# ----------------------------------------------------------------- selftest

def selftest():
    bad = 0

    def chk(name, got, want, tol=1e-6):
        nonlocal bad
        ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
        if not ok:
            print("FAIL %s: got %r want %r" % (name, got, want))
            bad += 1

    rng = random.Random(1)
    r = 1.20
    # (1) a geometric drift is cancelled EXACTLY by the geometric mid, with no effect present
    rows = simulate(4, 0.0, "geometric", 100 * (r - 1), 0.0, 0.0, rng)
    chk("bracket cancels geometric drift", bracketed(rows, "R", "T")["effect"], 0.0, 1e-12)
    # (2) the arithmetic mid does not, and its residue is the closed form log(2r/(1+r^2))
    chk("arithmetic mid leaves exactly 2r/(1+r^2) - 1",
        bracketed(rows, "R", "T", mid="arithmetic")["effect"], 2 * r / (1 + r * r) - 1, 1e-12)
    # ... and that residue is small compared with the effects this protocol targets
    chk("that residue is under 2pp at 20%/launch",
        abs(2 * r / (1 + r * r) - 1) < 0.02, True)
    # (3) the naive pairing keeps the drift instead of cancelling it
    chk("pairing keeps the drift", abs(paired(rows, "R", "T")["effect"]) > 0.10, True)

    # (4) with a real 10% effect, the bracket and the regression both recover it exactly
    rows = simulate(6, 0.10, "geometric", 8.0, 0.0, 0.0, rng)
    chk("bracket recovers a 10% effect", bracketed(rows, "R", "T")["effect"], 0.10, 1e-12)
    chk("regression recovers a 10% effect", math.exp(adjusted(rows, "R", "T")["log_effect"]) - 1,
        0.10, 1e-9)
    # and the two bracket forms agree to well under a 3% claim at this drift
    gap = abs(bracketed(rows, "R", "T")["effect"]
              - bracketed(rows, "R", "T", mid="arithmetic")["effect"])
    chk("the two bracket forms agree under 1pp at 8%/launch", gap < 0.01, True)

    # (5) a null window: every estimator must read zero, in every drift shape
    for shape in ("none", "linear", "geometric"):
        rows = simulate(6, 0.0, shape, 0.0, 0.0, 0.0, rng)
        chk("null (%s): bracket" % shape, abs(bracketed(rows, "R", "T")["effect"]) < 1e-12, True)
        chk("null (%s): adjusted" % shape, abs(adjusted(rows, "R", "T")["log_effect"]) < 1e-12, True)

    # (6) the order covariate's reported removal: nothing to remove without drift, most of it with
    a = adjusted(simulate(4, 0.0, "none", 0.0, 0.0, 0.0, rng), "R", "T")
    chk("no drift -> order removes nothing", a["var_removed_by_order"], 0.0, 1e-9)
    a = adjusted(simulate(6, 0.10, "geometric", 12.0, 0.01, 0.0, rng), "R", "T")
    chk("real drift -> order removes most of the arm-only variance",
        a["var_removed_by_order"] > 0.9, True)

    # (7) an unmatched test arm yields no contrast (rather than a zero)
    chk("unclosed bracket gives nothing", bracketed([Row("R", 1, 12.0), Row("T", 2, 12.5)],
                                                    "R", "T")["n"], 0)
    # (8) power arithmetic: 3% of 12 t/s at sd 0.3 t/s per contrast needs (2.802*0.3/0.36)^2 = 5.45
    chk("power arithmetic", (Z80 * 0.3 / (0.03 * 12.0)) ** 2, 5.45, 0.01)

    # (9) rho handles TIES by average rank, and the two readings it must differ on are real: the
    # certified window's k=2 arm has two identical values (12.20, 12.20).
    chk("rho, tied x: [1,1,2] vs [1,2,3]", _spearman([1, 1, 2], [1, 2, 3]), 0.866, 1e-3)
    chk("rho of a reversal is -1", _spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, 1e-12)
    chk("rho, tied x: [1,1,2,3] vs [4,3,2,1]", _spearman([1, 1, 2, 3], [4, 3, 2, 1]), -0.9487, 1e-3)
    # ... and that is NOT -1: ranking a tie by input order instead of average rank is exactly how the
    # quoted k=2 number drifted in the document this tool replaces.
    chk("the tied answer is not the naive -1", abs(_spearman([1, 1, 2, 3], [4, 3, 2, 1]) + 1) > 0.05,
        True)

    print("selftest: %d/%d passed" % (22 - bad, 22))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--ref", default="R")
    ap.add_argument("--test", default="T")
    ap.add_argument("--covariates", default="", help="extra columns, comma separated")
    ap.add_argument("--effect", type=float, default=0.03)
    ap.add_argument("--drift", choices=("none", "linear", "geometric"), default="geometric")
    ap.add_argument("--drift-pct", type=float, default=8.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--contention", type=float, default=0.0)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.analyze:
        covs = [c for c in args.covariates.split(",") if c]
        return run_analysis(Path(args.analyze), args.ref, args.test, covs)
    run_simulation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
