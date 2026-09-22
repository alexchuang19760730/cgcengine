#!/usr/bin/env python3
"""Split a repeated-arm experiment's spread into WITHIN-launch and LAUNCH-level parts.

WHY THIS EXISTS
    `docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md` ends on an unresolved question: identical
    work (bit-identical output, identical engine counters) reads 8.59-12.25 t/s on the
    k=3 arm and 11.21-12.83 on k=2. Two candidate causes were already killed there (the
    gate wait, the pool-event counts), leaving "something about the rate".

    "The rate swings" is not one hypothesis, it is two very different ones:

      (a) every STEP is noisy                 -> nothing about the launch matters;
                                                 more requests per launch is the only fix.
      (b) each LAUNCH is assigned a different baseline, and the steps inside it are
          relatively tight                    -> the launch is the unit that varies, so
                                                 pairing, warmup and repetition behave
                                                 completely differently.

    Those two are separable from data already on disk, because one arm = one launch =
    several requests. This script does that separation. It runs no GPU.

THE ESTIMATOR
    One-way random effects, per group (k=2 / k=3), with m requests in each of n arms:

        MS_within  = SS_within  / (n * (m - 1))     expected  = sigma_within^2
        MS_between = SS_between / (n - 1)           expected  = sigma_within^2 + m * sigma_launch^2

        sigma_launch^2 = max(MS_between - MS_within, 0) / m

    The max(...,0) is a deliberate floor, not a convenience: MS_between < MS_within
    happens by chance and means "no launch-level term was detected", never "a negative
    variance". Reporting 0 is the answer; reporting a signed number invites someone to
    subtract it later.

    F = MS_between / MS_within, df = (n-1, n*(m-1)). This script reports F and refuses to
    call anything a finding below F_CRIT, which is passed in rather than tuned per run.

WHAT IT DOES NOT DO
    It cannot say WHAT the launch-level term is. It only says whether one exists, how
    large, and in which group. Attributing it needs a different instrument (see the doc).

USAGE
    python3 scripts/check/k_swing_decompose.py --selftest
    python3 scripts/check/k_swing_decompose.py --dir Backup/k_abba_2026-09-23/abba \
            --dir Backup/k_abba_2026-09-23/gated --groups k2,k3
"""

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

# df = (4, 10) for the 5-arm / 3-request design this was written for. Passed in, not
# derived per run, so that "is this a finding" cannot be quietly re-tuned after seeing
# the data -- the same discipline as nsg_decide's 5% null-cell gate.
F_CRIT_DEFAULT = 3.48  # F(0.05; 4, 10)


def arm_rows(path):
    """(group, [per-request decode t/s]) or None. Requests with no t/s are dropped, and an
    arm with fewer than two usable requests cannot contribute a within-arm variance, so it
    is dropped rather than silently counted as one."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    if isinstance(d, list):
        d = d[0] if d else None
    if not isinstance(d, dict):
        return None
    reqs = [r.get("decode_tps") for r in (d.get("requests") or [])]
    reqs = [float(x) for x in reqs if isinstance(x, (int, float)) and x > 0]
    if len(reqs) < 2:
        return None
    return reqs


def group_of(path, groups):
    base = os.path.basename(path).lower()
    hits = [g for g in groups if g.lower() in base]
    if len(hits) != 1:
        return None        # ambiguous or absent: not guessable, and guessing would
    return hits[0]         # silently move an arm between the two cells being compared


def _betacf(a, b, x):
    """Lentz continued fraction for the incomplete beta (Numerical Recipes form)."""
    tiny, eps, itmax = 1e-30, 3e-12, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def f_cdf(x, d1, d2):
    """P(F <= x) for the F distribution, via the regularized incomplete beta."""
    if x <= 0.0:
        return 0.0
    a, b = d1 / 2.0, d2 / 2.0
    z = d1 * x / (d1 * x + d2)                   # F -> beta transform
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(z) + b * math.log1p(-z))
    if z < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, z) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - z) / b


def f_ppf(p, d1, d2):
    """Inverse of f_cdf by bisection. Pure stdlib on purpose: this file is a check script."""
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f_cdf(mid, d1, d2) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def launch_sd_upper(d, conf=0.95):
    """Largest launch-level sd consistent with the observed F at `conf`.

    WHY THIS EXISTS: `max(MS_b - MS_w, 0)` reports zero whenever MS_b <= MS_w, and
    "zero" reads like a finding. It is not - it is the point estimate at the edge of
    its own range. Under the model, F_obs / (1 + m*lambda) ~ F(df1, df2) with
    lambda = var_launch / var_within, so inverting the lower conf tail of F gives
    the largest lambda the data cannot rule out. A group whose launch term prints
    0.00% can still be bounded at, say, 10% - and that bound is what prices the arm
    count, because n is computed from the arm-mean scatter which contains it.
    """
    if d.get("F") is None or d.get("m_requests", 0) < 2 or d.get("n_arms", 0) < 2:
        return None
    d1, d2 = d["df"]
    if d1 <= 0 or d2 <= 0 or not math.isfinite(d["F"]):
        return None
    f_lo = f_ppf(1.0 - conf, d1, d2)
    lam = (d["F"] / f_lo - 1.0) / d["m_requests"]
    w = d.get("within_sd_pct")
    if lam <= 0 or w is None:
        return 0.0
    return w * math.sqrt(lam)


def t_two_sided_p(t, df):
    """P(|T| > |t|) for Student t, via the identity |T| > t  <=>  F(1,df) > t^2.

    Reuses f_cdf so this file carries one distribution implementation, not two.
    """
    if df <= 0:
        return None
    return 1.0 - f_cdf(t * t, 1, df)


def t_crit(df, alpha=0.05):
    """Two-sided critical value, by bisection on t_two_sided_p."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_diffs(by_rep):
    """by_rep: {rep: {"k2": [arm means], "k3": [arm means]}} -> one row per rep.

    A pair is only formed when BOTH sides are present: an unpaired rep would silently
    drop out of the mean and make n look larger than the evidence supports.
    """
    rows = []
    for r in sorted(by_rep):
        a, b = by_rep[r].get("k2"), by_rep[r].get("k3")
        if not a or not b:
            continue
        ma, mb = st.mean(a), st.mean(b)
        rows.append({"rep": r, "a": ma, "b": mb, "diff": ma - mb})
    return rows


def paired_verdict(rows, planned_n, alpha=0.05):
    """Judge a paired difference -- but ONLY at the pre-registered n.

    WHY planned_n IS REQUIRED: the temptation this blocks is the one that already
    happened. Five pairs gave t=2.08 (p=0.11); 'just two more' is exactly the move
    that turns a fixed-n test into optional stopping, and optional stopping does not
    keep the alpha it advertises. So below planned_n this returns a refusal naming
    how many are still owed, and above it says so too -- extra pairs after the test
    are a new experiment, not a better version of this one.
    """
    n = len(rows)
    out = {"n": n, "planned_n": planned_n, "alpha": alpha}
    if n < 2:
        out["class"] = "UNRESOLVED"
        out["why"] = "fewer than two pairs: no variance, no test"
        return out
    diffs = [r["diff"] for r in rows]
    mean_d = st.mean(diffs)
    sd = st.stdev(diffs)
    base = st.mean([r["b"] for r in rows])
    out.update({"mean_diff": mean_d, "sd": sd, "se": sd / (n ** 0.5),
                "mean_diff_pct": 100.0 * mean_d / base if base else None,
                "t": (mean_d / (sd / (n ** 0.5))) if sd > 0 else None,
                "slower_pairs": sum(1 for d in diffs if d > 0),
                "df": n - 1})
    out["crit"] = t_crit(n - 1, alpha)
    if n < planned_n:
        out["class"] = "NOT YET"
        out["why"] = "n=%d < planned %d: testing now would be optional stopping" % (n, planned_n)
        return out
    if n > planned_n:
        out["class"] = "OVERRUN"
        out["why"] = ("n=%d > planned %d: the pre-registered test happened at %d; this is a "
                      "new experiment" % (n, planned_n, planned_n))
        return out
    t = out["t"]
    if t is None:
        out["class"] = "UNRESOLVED"
        out["why"] = "zero spread across pairs"
        return out
    out["class"] = "CERTIFIED" if abs(t) >= out["crit"] else "NOT SEPARATED"
    return out


def decompose(arms):
    """arms: list of per-arm lists. Returns the random-effects split."""
    n = len(arms)
    m = min(len(a) for a in arms)
    arms = [a[:m] for a in arms]                  # equal m keeps the algebra honest
    flat = [x for a in arms for x in a]
    grand = st.mean(flat)
    ss_w = sum(sum((x - st.mean(a)) ** 2 for x in a) for a in arms)
    ss_b = sum(m * (st.mean(a) - grand) ** 2 for a in arms)
    ms_w = ss_w / (n * (m - 1)) if n * (m - 1) > 0 else 0.0
    ms_b = ss_b / (n - 1) if n > 1 else 0.0
    var_launch = max(ms_b - ms_w, 0.0) / m
    f = (ms_b / ms_w) if ms_w > 0 else float("inf")
    out = {
        "n_arms": n, "m_requests": m, "grand_mean": grand,
        "ms_within": ms_w, "ms_between": ms_b,
        "within_sd_pct": 100.0 * (ms_w ** 0.5) / grand if grand else None,
        "launch_sd_pct": 100.0 * (var_launch ** 0.5) / grand if grand else None,
        # what an arm MEAN would scatter by if every arm drew the same baseline
        "expected_arm_mean_sd_pct": 100.0 * ((ms_w / m) ** 0.5) / grand if grand else None,
        "total_arm_mean_sd_pct": 100.0 * ((var_launch + ms_w / m) ** 0.5) / grand if grand else None,
        "F": f, "df": (n - 1, n * (m - 1)),
        "arms": arms,
    }
    out["declines"] = sum(1 for a in arms if all(a[i] > a[i + 1] for i in range(len(a) - 1)))
    out["first_to_last_pct"] = [100.0 * (a[0] - a[-1]) / a[0] for a in arms if a[0]]
    return out


def verdict(d, f_crit):
    """The one sentence this tool exists to produce. Three answers, not two: a group with
    too few arms to resolve anything must say so instead of printing 'no launch term'."""
    if d["n_arms"] < 2 or d["m_requests"] < 2:
        return "UNRESOLVED (need >=2 arms and >=2 requests per arm)"
    if d["F"] >= f_crit:
        return "LAUNCH-LEVEL TERM PRESENT"
    return "no launch-level term detected (arms differ no more than requests do)"


def collect(dirs, groups):
    by = {g: [] for g in groups}
    skipped = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            g = group_of(p, groups)
            if g is None:
                skipped.append(os.path.basename(p))
                continue
            r = arm_rows(p)
            if r is None:
                skipped.append(os.path.basename(p))
                continue
            by[g].append(r)
    return by, skipped


def selftest():
    tot = bad = 0

    def check(name, cond):
        nonlocal tot, bad
        tot += 1
        if not cond:
            bad += 1
            print("FAIL: %s" % name)

    # (1) pure within noise, no launch offset: MS_between should land near MS_within and
    # the launch term must floor at zero rather than go negative.
    pure = [[10.0, 11.0, 9.0], [9.5, 10.5, 11.0], [10.0, 10.0, 10.5],
            [11.0, 9.5, 10.0], [9.8, 10.2, 10.0]]
    d0 = decompose(pure)
    check("identical baselines give no launch-level term", d0["launch_sd_pct"] == 0.0)
    check("...and F below 1, since arms differ less than requests do", d0["F"] < 1.0)
    check("...and the within term is what remains", d0["within_sd_pct"] > 0.0)

    # (2) the same within noise plus a per-launch offset of +-20%: the estimator must
    # recover roughly that offset and must clear the F gate.
    import copy
    off = copy.deepcopy(pure)
    for i, a in enumerate(off):
        k = 1.0 + (0.20, -0.20, 0.20, -0.20, 0.20)[i]
        off[i] = [x * k for x in a]
    d1 = decompose(off)
    check("a per-launch offset is detected", d1["launch_sd_pct"] > 15.0)
    check("...and it clears the gate", d1["F"] >= F_CRIT_DEFAULT)
    check("...while the within term stays where it was",
          abs(d1["within_sd_pct"] - d0["within_sd_pct"]) < 1.0)

    # (3) the two are distinguishable by the number this tool prints, which is the whole
    # point: same request-level noise, different launch-level noise.
    check("the two groups differ in launch term but not in within term",
          d1["launch_sd_pct"] - d0["launch_sd_pct"] > 15.0
          and abs(d1["within_sd_pct"] - d0["within_sd_pct"]) < 1.0)

    # (4) an arm mean's total scatter is the quadrature sum -- this is the number that
    # prices "how many launches to certify 3%".
    check("total arm-mean sd is the quadrature sum of launch and within/m",
          abs(d1["total_arm_mean_sd_pct"]
              - ((d1["launch_sd_pct"] ** 2 + d1["expected_arm_mean_sd_pct"] ** 2) ** 0.5)) < 0.01)

    # (5) the floor: MS_between < MS_within must never print a negative variance.
    d2 = decompose([[10.0, 10.1, 9.9], [10.0, 9.9, 10.1]])
    check("a negative MS_between - MS_within floors at zero, never reports negative",
          d2["launch_sd_pct"] == 0.0)

    # (6) refusal paths.
    check("one arm cannot resolve a launch term",
          verdict(decompose([[1.0, 2.0, 3.0]]), F_CRIT_DEFAULT).startswith("UNRESOLVED"))
    check("one request per arm cannot either",
          verdict(decompose([[1.0], [2.0], [3.0]]), F_CRIT_DEFAULT).startswith("UNRESOLVED"))
    check("a detected term is named", verdict(d1, F_CRIT_DEFAULT) == "LAUNCH-LEVEL TERM PRESENT")
    check("an undetected one is named too, and says what it did not find",
          verdict(d0, F_CRIT_DEFAULT).startswith("no launch-level term"))

    # (7) monotone decline counting, used for the "the arm heats itself" read.
    d3 = decompose([[10.0, 9.0, 8.0], [10.0, 9.5, 9.0], [8.0, 9.0, 10.0]])
    check("monotone declines are counted, rises are not", d3["declines"] == 2)
    check("first-to-last drop is signed (a rise prints negative)",
          abs(d3["first_to_last_pct"][2] + 25.0) < 0.01)

    # (8) group assignment must not guess: a file matching no group is skipped, and one
    # matching two is skipped too.
    check("an arm matching no group is skipped, not assigned",
          group_of("/x/r1_a_k9.json", ["k2", "k3"]) is None)
    check("an arm matching two groups is skipped, not assigned to the first",
          group_of("/x/r1_k2_k3.json", ["k2", "k3"]) is None)
    check("an unambiguous arm is assigned", group_of("/x/r1_b_k3.json", ["k2", "k3"]) == "k3")

    # The F quantile is pinned against the two values this whole file already leans
    # on: F_CRIT_DEFAULT is the 95% point of F(4,10), and the k=2 F of 0.62 sits just
    # above the 5% point -- which is exactly why its bound comes out near zero.
    check("F(4,10) 95%% point reproduces the F_CRIT this file already uses",
          abs(f_ppf(0.95, 4, 10) - F_CRIT_DEFAULT) < 0.01)
    # 0.1677 is pinned two ways: by this file's own f_cdf, and by 1/F_0.95(10,4).
    # A first pass had 0.6126 here, from a Simpson integration of the F pdf that was
    # simply wrong -- and it made k=2's bound look like ~0.7% instead of ~10%. The
    # check exists so the number cannot silently drift again.
    check("F(4,10) 5%% point is the reciprocal-side tail used for the bound",
          abs(f_ppf(0.05, 4, 10) - 0.1677) < 0.0005
          and abs(f_ppf(0.05, 4, 10) - 1.0 / f_ppf(0.95, 10, 4)) < 1e-9)
    check("f_cdf is monotone and lands on the quantile it inverted",
          abs(f_cdf(f_ppf(0.95, 4, 10), 4, 10) - 0.95) < 1e-6)

    # A group with arms identical to each other must bound to zero: there is no
    # between-arm variance to be generous about.
    check("no between-arm spread => the upper bound is zero, not a positive guess",
          launch_sd_upper(decompose([[10.0, 10.1, 9.9], [10.0, 9.9, 10.1]])) == 0.0)
    # ...and a group WITH one must bound strictly above its own point estimate,
    # because a point estimate at the edge of the range is not the range.
    dhi = decompose([[10.0, 10.1, 9.9], [13.0, 12.9, 13.1], [10.0, 9.9, 10.1],
                     [13.0, 13.2, 12.8], [11.5, 11.4, 11.6]])
    check("a detected launch term bounds strictly ABOVE its point estimate",
          launch_sd_upper(dhi) > (dhi["launch_sd_pct"] or 0.0))
    check("...and above zero, so '0.00%%' can never be read as 'proven absent'",
          launch_sd_upper(dhi) > 0.0)
    check("too few arms to invert gives None rather than a flattering zero",
          launch_sd_upper({"F": 1.0, "df": (0, 2), "m_requests": 3,
                           "within_sd_pct": 5.0}) is None)

    # t critical values, pinned against the standard table -- an independent anchor,
    # like F_0.95(4,10)=3.478 was for the F side.
    check("t_crit(4) reproduces the two-sided 5% point 2.776", abs(t_crit(4) - 2.776) < 0.002)
    check("t_crit(6) reproduces 2.447", abs(t_crit(6) - 2.447) < 0.002)
    check("t_crit(10) reproduces 2.228", abs(t_crit(10) - 2.228) < 0.002)
    check("the t and F tails agree: |T|>t is F(1,df)>t^2",
          abs(t_two_sided_p(2.776, 4) - 0.05) < 0.001)

    # Optional stopping is the failure mode this mode exists to block, so the refusal
    # is the part under test -- not the arithmetic.
    pr = [{"rep": "r%d" % i, "a": 12.0, "b": 10.3, "diff": 1.7} for i in range(1, 5)]
    check("5 pairs against a planned 7 is a refusal, not a result",
          paired_verdict(pr, 7)["class"] == "NOT YET")
    check("...and it names how many are still owed", "7" in paired_verdict(pr, 7)["why"])
    # spread is deliberately small but NON-zero: exactly zero spread makes the t
    # statistic degenerate, and degenerate is a refusal here, not a perfect result.
    eight = pr + [dict(rep="r%d" % i, a=12.0, b=10.3, diff=1.7 + (i - 6) * 0.01)
                  for i in range(5, 9)]
    check("8 pairs against a planned 8 is judged, neither refused nor called an overrun",
          paired_verdict(eight, 8)["class"] in ("CERTIFIED", "NOT SEPARATED"))
    check("9 pairs against a planned 8 is an overrun, not a better 8",
          paired_verdict(eight + [dict(rep="r9", a=12.0, b=10.3, diff=1.7)], 8)["class"]
          == "OVERRUN")
    check("a steady 1.7 t/s gap over 8 pairs with sd 0 certifies",
          paired_verdict(eight, 8)["class"] == "CERTIFIED")
    check("an unpaired rep cannot inflate n",
          len(paired_diffs({"r1": {"k2": [12.0]}, "r2": {"k2": [12.0], "k3": [10.0]}})) == 1)
    check("one pair is unresolved rather than a test with df=0",
          paired_verdict([{"rep": "r1", "a": 12.0, "b": 10.0, "diff": 2.0}], 1)["class"]
          == "UNRESOLVED")

    print("selftest: %d/%d passed" % (tot - bad, tot))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", action="append", default=[],
                    help="directory of arm JSONs (repeatable)")
    ap.add_argument("--groups", default="k2,k3",
                    help="comma-separated tokens matched against the filename")
    ap.add_argument("--f-crit", type=float, default=F_CRIT_DEFAULT,
                    help="F gate for declaring a launch-level term (default %.2f = F(0.05;4,10))"
                         % F_CRIT_DEFAULT)
    ap.add_argument("--paired", action="store_true",
                    help="judge the k2-vs-k3 EFFECT (paired by repetition) instead of only "
                         "the noise split. --planned-n is REQUIRED for a verdict.")
    ap.add_argument("--planned-n", type=int, default=0,
                    help="the n fixed BEFORE the run. Below it the verdict refuses; above it "
                         "the verdict calls it a new experiment. 0 = report the numbers, no test.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-pct", type=float, default=3.0,
                    help="effect size to price, for the arms-per-group estimate")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.dir:
        print("nothing to do: pass --dir (or --selftest)")
        return 2

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    by, skipped = collect(args.dir, groups)
    if skipped:
        print("skipped (group not identifiable or <2 usable requests): %s"
              % ", ".join(sorted(set(skipped))))

    print("=" * 96)
    for g in groups:
        arms = by.get(g) or []
        if len(arms) < 2:
            print("%-4s  %s" % (g, verdict({"n_arms": len(arms), "m_requests": 0}, args.f_crit)))
            continue
        d = decompose(arms)
        print("%-4s  n=%d arms x %d requests   grand mean %.2f t/s"
              % (g, d["n_arms"], d["m_requests"], d["grand_mean"]))
        print("      within-arm   sd %5.2f%%   (one request)" % (d["within_sd_pct"] or 0.0))
        ub = launch_sd_upper(d)
        print("      launch-level sd %5.2f%%   F=%.2f df=%s%s"
              % (d["launch_sd_pct"] or 0.0, d["F"], d["df"],
                 "" if ub is None else "   (95%% upper bound %.2f%%)" % ub))
        print("      an arm MEAN scatters %5.2f%%  (%.2f%% of that is just request noise)"
              % (d["total_arm_mean_sd_pct"] or 0.0, d["expected_arm_mean_sd_pct"] or 0.0))
        tot = d["total_arm_mean_sd_pct"] or 0.0
        if tot > 0:
            # TWO different numbers, and mixing them is how "4 arms" gets believed:
            # n_se drives the STANDARD ERROR down to target; n_ci drives the 95%
            # CONFIDENCE INTERVAL half-width down to target. They differ by 1.96^2.
            n_se = (tot / args.target_pct) ** 2
            n_ci = (1.96 * tot / args.target_pct) ** 2
            print("      => %.0f arms/group for standard error %.1f%%; %.0f for a 95%% CI "
                  "half-width %.1f%%" % (n_se, args.target_pct, n_ci, args.target_pct))
            if ub is not None and ub > 0:
                tot_hi = ((ub ** 2) + (d["expected_arm_mean_sd_pct"] or 0.0) ** 2) ** 0.5
                print("      => worst case at that bound: %.0f (SE) / %.0f (CI)"
                      % ((tot_hi / args.target_pct) ** 2,
                         (1.96 * tot_hi / args.target_pct) ** 2))
        print("      first->last within arm: %s   (monotone decline %d/%d)"
              % (" ".join("%+.0f%%" % x for x in d["first_to_last_pct"]),
                 d["declines"], d["n_arms"]))
        print("      VERDICT: %s" % verdict(d, args.f_crit))
        print("-" * 96)

    if args.paired:
        # Same loader the noise split uses (arm_rows), so the two modes can never
        # disagree about what one arm's number is.
        by_rep = {}
        for d in args.dir:
            for pp in sorted(glob.glob(os.path.join(d, "*.json"))):
                g = group_of(pp, groups)
                if g is None:
                    continue
                rr = arm_rows(pp)
                if not rr:
                    continue
                rep = os.path.basename(pp)[:-5].split("_")[0]
                by_rep.setdefault(rep, {}).setdefault(g, []).append(st.mean(rr))
        rows = paired_diffs(by_rep)
        v = paired_verdict(rows, args.planned_n, args.alpha)
        print("=" * 96)
        print("PAIRED EFFECT  (each row = one ABBA repetition, arm means)")
        for r in rows:
            print("  %-8s  %s %7.2f   %s %7.2f   diff %+6.2f t/s"
                  % (r["rep"], groups[0], r["a"], groups[1], r["b"], r["diff"]))
        if rows:
            print("-" * 96)
            print("  n=%d pairs   mean diff %+.2f t/s (%+.1f%%)   sd %.2f   SE %.2f   t=%.2f  "
                  "crit=%.2f (df=%d, alpha=%.2f)"
                  % (v["n"], v["mean_diff"], v["mean_diff_pct"], v["sd"], v["se"],
                     v["t"], v["crit"], v["df"], v["alpha"]))
            print("  %s slower in %d/%d pairs"
                  % (groups[1], v["slower_pairs"], v["n"]))
            print("  VERDICT: %s%s" % (v["class"], "" if "why" not in v else "  -- " + v["why"]))
            if args.planned_n:
                print("  (pre-registered n=%d; the honest number of MORE pairs to run is "
                      "%d, and the test happens once, at the end)"
                      % (args.planned_n, max(0, args.planned_n - v["n"])))
        else:
            print("  no complete pairs found")
        print("=" * 96)

    # The comparison the doc could not make by eye: which group carries a launch term.
    got = {}
    for g in groups:
        arms = by.get(g) or []
        got[g] = decompose(arms)["launch_sd_pct"] if len(arms) >= 2 else None
    if all(v is not None for v in got.values()) and len(groups) >= 2:
        hi = max(groups, key=lambda g: got[g] or 0.0)
        lo = min(groups, key=lambda g: got[g] or 0.0)
        print("asymmetry: %s carries a launch-level term of %.2f%%, %s carries %.2f%%"
              % (hi, got[hi] or 0.0, lo, got[lo] or 0.0))
        print("  => pairing (AB/BA) can only cancel a term BOTH groups share, so it cannot"
              " remove this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
