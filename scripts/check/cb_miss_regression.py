#!/usr/bin/env python3
"""F1: is the decode hook's `cb` a per-MISS quantity (bandwidth) or a per-ROUND-TRIP one (latency)?

Why this tool exists
--------------------
`cb` -- the host-side top-k hook, timed in ggml-backend.cpp as `st2 - st1` around
`callback_eval(ttopk)` -- is 29.9% of a delivery decode step (74.18 ms of 247.98 ms at ntok=4,
MTP on, n=107 work rows). `CGC_HOOK_SPLIT` says 99.7% of that hook is `ensure`, i.e. the union's
demand fill. So the lever is whatever sets `ensure`'s cost, and two models predict different shapes
in the layer's miss count `m`:

    bandwidth : cost ∝ bytes          -> cb_layer linear in m, no plateau
    round-trip: cost = ceil(3m/8) x one pread latency  -> cb_layer FLAT for m=1..2, then steps

The device numbers make round-trip the prior (`jobs=6150, 0.37 MiB/job, us/job=1647, 235 MiB/s` on
an 823 MB/s disk: a 0.37 MiB job should cost ~0.45 ms, so ~1.2 ms/job is fixed overhead). But that
was inferred from run totals. This pairs per-layer `cb` with per-layer miss counts and decides.

Inputs it needs in ONE server log:
  * `CGC_DECODE_PROFILE=1` + `CGC_DECODE_PROFILE_ALL=1`  -> `CGC-DECPROF all: L<n> ... cb=<ms>` and
    the `CGC-DECPROF: step=<n> ... ntok=<k>` step boundary
  * `LLAMA_EXPERT_CACHE_BATCH_DBG=1`                     -> `BATCHDBG layer=<l> misses=<m>`
    (this one was NOT in the launcher's env allowlist until 2026-09-20, so before that it could not
    be turned on at all through run_server.sh -- see the note in the allowlist block)

Pairing rule (and why it is not a guess)
----------------------------------------
The step line is emitted at the END of its step, `BATCHDBG` lines are emitted DURING it, and the
per-layer `all:` lines are emitted after it. So for step N: misses = the `BATCHDBG` lines seen since
the previous step line; cb = the `all:` lines seen until the next step line. A layer with no
`BATCHDBG` line has m=0 -- which is itself a testable claim (see the null check).

Honest limits
-------------
* `BATCHDBG` lives in `ensure_batch` (the worker-pool fill). A layer served by the MTP fast path's
  `ensure_slot` (3 threads per expert, serial per expert) prints no `BATCHDBG` line, so a layer with
  `cb >= 1 ms` and no line is counted in the PATH check rather than silently read as m=0.
* Absolute `cb` on a run carrying `CGC_HOOK_SPLIT` is instrumented (four clock reads per hook);
  the verdict rests on the SHAPE across m, not on the absolute level.
* This says what sets `cb`. It says nothing about whether a fix is worth doing.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STEP_RX = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \([\d.]+%\) cb=([\d.]+) \([\d.]+%\) submit=([\d.]+) \([\d.]+%\) ntok=(\d+)")
ALL_RX = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms")
# `gap` is what separates the two step-row populations. With CGC_GPU_TIMING=1 every turn emits TWO
# step rows for the SAME step: the work row (real GPU work, gap>0) and a shadow row (gap_sum==0,
# total ~1-2 ms). Pooling them produced a 62 ms "median" that described neither -- see
# docs/DECODE_STEP_ROW_POPULATIONS_2026-09-20.md. The split has to be available to every consumer of
# this parser, not just the one that found the bug.
GAP_RX = re.compile(r"CGC-DECPROF all: L(\d+) .*?\sgap=([\d.]+)")
BATCH_RX = re.compile(r"BATCHDBG layer=(\d+) misses=(\d+)")

# The decision thresholds are FIXED IN ADVANCE (Backup/phase_decomp/F1_CB_MISS_REGRESSION_20260920
# .conditions.md). They are constants here so that a reader can see the rule the number was judged
# by, and so that changing the rule is a visible edit rather than a memory.
PLATEAU_SAME_12 = 1.25   # med(m=2)/med(m=1) must be within 25%
PLATEAU_RATIO_18 = 2.0   # med(m=6..8)/med(m=1) must be >= 2.0
LINEAR_R12 = 1.6
LINEAR_R24 = 1.6
MIN_STEPS = 8
NO_MISS_CB = 0.05        # ms; below this a layer must have had no misses at all


def parse_log(path: Path) -> dict:
    """Return {'steps': [...]} where each step carries its ntok, its per-layer cb, and per-layer m."""
    steps: list[dict] = []
    pending_m: dict[int, int] = {}
    cur: dict | None = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            b = BATCH_RX.search(line)
            if b:
                lay = int(b.group(1))
                # a layer can be filled more than once per step in principle; count the misses,
                # and keep the last call's number as the layer's m for the pairing.
                pending_m[lay] = pending_m.get(lay, 0) + int(b.group(2))
                continue
            s = STEP_RX.search(line)
            if s:
                cur = {
                    "step": int(s.group(1)),
                    "ntok": int(s.group(8)),
                    "total": float(s.group(4)),
                    "cb_total": float(s.group(6)),
                    "misses": dict(pending_m),
                    "cb": {},
                    "gap_sum": 0.0,
                    "per_layer_rows": 0,
                    "has_gap_field": False,
                }
                steps.append(cur)
                pending_m = {}
                continue
            a = ALL_RX.search(line)
            if a and cur is not None:
                cur["cb"][int(a.group(1))] = float(a.group(3))
                cur["per_layer_rows"] += 1
                # BOTH fields live on this same line, so the gap must be read HERE. Pulling it into a
                # separate search below `continue` is how it silently stopped being parsed at all.
                g = GAP_RX.search(line)
                if g:
                    cur["gap_sum"] += float(g.group(2))
                    cur["has_gap_field"] = True
    for s in steps:
        # A row is a WORK row iff some per-layer line reports gap > 0. A log without the gap field at
        # all is "no instrument", NOT "shadow" -- conflating those two is how a rule that keys on
        # gap==0 ends up calling an uninstrumented run a shadow run.
        if s["per_layer_rows"] == 0:
            s["row_kind"] = "NO-ROWS"
        elif not s["has_gap_field"]:
            s["row_kind"] = "UNSPLIT"
        elif s["gap_sum"] > 0:
            s["row_kind"] = "work"
        else:
            s["row_kind"] = "shadow"
    return {"steps": steps, "has_gap_instrument": any(s["has_gap_field"] for s in steps)}


def bucket(m: int, workers: int = 8) -> int:
    """The round-trip model's own predictor: number of pread rounds a layer needs."""
    if m <= 0:
        return 0
    return -(-(3 * m) // workers)  # ceil(3m/workers)


def _verdict_from_medians(by_m, med) -> tuple[str, str, dict]:
    r12 = r24 = r18 = None
    m1, m2 = med(by_m.get(1, [])), med(by_m.get(2, []))
    m4 = med(by_m.get(4, []))
    hi = [c for m in (6, 7, 8) for c in by_m.get(m, [])]
    hi_med = med(hi)
    if m1:
        r12 = m2 / m1 if m2 else None
        r18 = hi_med / m1 if hi_med else None
    if m2 and m4:
        r24 = m4 / m2
    ratios = {"r12": r12, "r24": r24, "r18": r18,
              "m1": m1, "m2": m2, "m4": m4, "hi": hi_med}
    if r12 is not None and r18 is not None and r12 <= PLATEAU_SAME_12 and r18 >= PLATEAU_RATIO_18:
        return "PLATEAU", (f"med(m=1)={m1:.2f} med(m=2)={m2:.2f} (ratio {r12:.2f} <= {PLATEAU_SAME_12}) and "
                           f"med(m=6..8)={hi_med:.2f} (ratio to m=1 {r18:.2f} >= {PLATEAU_RATIO_18}) "
                           f"=> round-trip / latency bound"), ratios
    if r12 is not None and r24 is not None and r12 >= LINEAR_R12 and r24 >= LINEAR_R24:
        return "LINEAR", (f"med(m=2)/med(m=1)={r12:.2f} >= {LINEAR_R12} and med(m=4)/med(m=2)={r24:.2f} "
                           f">= {LINEAR_R24} => bandwidth bound"), ratios
    return "INCONCLUSIVE", (f"med(m=1)={m1} med(m=2)={m2} med(m=4)={m4} med(m=6..8)={hi_med}: "
                            f"neither rule fired (r12={r12}, r24={r24}, r18={r18})"), ratios


def _fit(groups: dict[int, list[float]], pred) -> dict:
    """Least-squares y = a + b*x on GROUP MEDIANS (robust to the heavy right tail), plus R^2.

    Post-hoc by construction: neither model's coefficients were pre-registered. It is reported
    separately from the pre-registered verdict so the two cannot be conflated.
    """
    pts = [(x, st.median(v)) for x, v in sorted(groups.items()) if v]
    if len(pts) < 3:
        return {"n_groups": len(pts)}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"n_groups": len(pts)}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"n_groups": len(pts), "intercept_ms": a, "slope_ms_per_unit": b,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
            "aug_per_pred": {int(x): round(y, 3) for x, y in pts},
            "pred": {int(pred(x)): round(a + b * pred(x), 3) for x in xs}}


def analyse(parsed: dict, ntok_filter=None) -> dict:
    steps = parsed["steps"]
    if ntok_filter is not None:
        steps = [s for s in steps if s["ntok"] in ntok_filter]

    pairs: list[tuple[int, float, int]] = []   # (m, cb, ntok)
    null_violations: list[tuple[int, int, float]] = []
    path_suspects: list[tuple[int, int, float]] = []
    evaluable = 0
    for s in steps:
        if not s["cb"]:
            continue
        evaluable += 1
        for lay, cb in s["cb"].items():
            m = s["misses"].get(lay, 0)
            if cb < NO_MISS_CB and m > 0:
                null_violations.append((s["step"], lay, cb))
            if cb >= 1.0 and m == 0:
                path_suspects.append((s["step"], lay, cb))
            pairs.append((m, cb, s["ntok"]))

    by_m: dict[int, list[float]] = {}
    for m, cb, _ in pairs:
        by_m.setdefault(m, []).append(cb)
    by_b: dict[int, list[float]] = {}
    for m, cb, _ in pairs:
        by_b.setdefault(bucket(m), []).append(cb)

    # the same grouping on the subset that passes the null check (see the report's deviation note)
    flagged = {(s, l) for s, l, _ in null_violations}
    by_m_clean: dict[int, list[float]] = {}
    for s in steps:
        if not s["cb"]:
            continue
        for lay, cb in s["cb"].items():
            if (s["step"], lay) in flagged:
                continue
            by_m_clean.setdefault(s["misses"].get(lay, 0), []).append(cb)

    def med(v):
        return st.median(v) if v else None

    if evaluable < MIN_STEPS:
        verdict, reason = "INCONCLUSIVE", f"only {evaluable} step(s) with per-layer rows (<{MIN_STEPS})"
        ratios = {}
    else:
        verdict, reason, ratios = _verdict_from_medians(by_m, med)

    verdict_clean, reason_clean, _ = _verdict_from_medians(by_m_clean, med)

    # Where does cb actually come from? The path check above can only see cb >= 1 ms with no miss
    # line; this says the share outright, which is what retires (or keeps) the hypothesis that the
    # MTP fast path's per-expert `ensure_slot` carries part of cb.
    sum_miss = sum(cb for m, cb, _ in pairs if m > 0)
    sum_nomiss = sum(cb for m, cb, _ in pairs if m == 0)
    tot = sum_miss + sum_nomiss
    cb_share = {
        "total_ms": tot,
        "from_layers_with_miss_ms": sum_miss,
        "from_layers_without_miss_ms": sum_nomiss,
        "pct_from_layers_with_miss": (100.0 * sum_miss / tot) if tot else None,
        "no_miss_layers_above_noise": sum(1 for m, cb, _ in pairs if m == 0 and cb >= NO_MISS_CB),
    }

    # ── F5: the first-miss premium, read off the same two arms ────────────────────────────────────
    # The premium is a per-LAYER quantity (`cb(m=1)` costs more than the per-miss trend predicts), so
    # the lever is not the cost per miss but HOW MANY LAYERS carry one at all. L is that count per
    # step, on the same pairing rule as `pairs`. A fix that removes layers from L is moving the
    # mechanism; one that leaves L alone and shaves the premium is not.
    #   F5 supported : the treated arm lowers L AND leaves median(cb|m=1) ~unchanged
    #   F5 refuted   : L is unchanged
    # Both readings are reported unconditionally so the comparison cannot be made post-hoc.
    layers_with_miss = []
    for s in steps:
        if not s["cb"]:
            continue
        layers_with_miss.append(sum(1 for lay in s["cb"] if s["misses"].get(lay, 0) > 0))
    # L is bimodal: a step is either fully warm (no layer has a miss -> cb ~0.4 ms) or broadly cold
    # (most layers have one). The mean alone hides that, and the two regimes call for different
    # levers, so the warm share and the conditional mean are reported alongside it.
    #
    # L is computed PER ROW POPULATION. Pooling the work rows with the shadow artifacts is the same
    # defect that produced the 62 ms phantom median, and it would inflate the warm share here (a
    # shadow row has almost no cb and therefore almost no miss-bearing layers). Both are reported
    # so no reader has to guess which one a number came from.
    def l_stats(rows: list[dict]) -> dict:
        L = [sum(1 for lay in s["cb"] if s["misses"].get(lay, 0) > 0) for s in rows if s["cb"]]
        warm = [x for x in L if x == 0]
        cold = [x for x in L if x > 0]
        return {
            "n_steps": len(L),
            "L_median": med(L), "L_mean": st.mean(L) if L else None,
            "L_max": max(L) if L else None,
            "warm_steps": len(warm),
            "warm_share": (len(warm) / len(L)) if L else None,
            "L_mean_when_cold": st.mean(cold) if cold else None,
        }

    work_rows = [s for s in steps if s.get("row_kind") == "work"]
    f5 = l_stats(steps)
    f5["populations"] = {"all": l_stats(steps), "work": l_stats(work_rows) if work_rows else None}
    # The arm-level `median_cb_ms` is taken over work rows, so the work row is the population that
    # belongs beside it. `confirm_population` names which one this report leads with.
    f5["confirm_population"] = "work" if work_rows else "all"
    if work_rows:
        f5.update({f"work_{k}": v for k, v in f5["populations"]["work"].items()})
    f5["cb_at_m1_median"] = med(by_m.get(1, []))
    f5["cb_at_m1_n"] = len(by_m.get(1, []))
    f5["cb_at_m0_median"] = med(by_m.get(0, []))

    # POST-HOC model comparison (not pre-registered -- see render()): the two 1-parameter shapes,
    # fitted on group medians. `bucket` is the round-trip model's own predictor ceil(3m/workers).
    models = {
        "per_miss_linear": _fit(by_m, lambda m: m),
        "per_round_trip": _fit(by_b, lambda b: b),
    }

    return {
        "evaluable_steps": evaluable,
        "pairs": len(pairs),
        "by_m": {str(k): {"n": len(v), "median": med(v), "mean": st.mean(v)} for k, v in sorted(by_m.items())},
        "by_m_clean": {str(k): {"n": len(v), "median": med(v)} for k, v in sorted(by_m_clean.items())},
        "by_bucket": {str(k): {"n": len(v), "median": med(v)} for k, v in sorted(by_b.items())},
        "verdict": verdict,
        "reason": reason,
        "ratios": ratios,
        "verdict_excl_null_violations": verdict_clean,
        "reason_excl_null_violations": reason_clean,
        "null_violations": null_violations,
        "path_suspects": path_suspects,
        "cb_share": cb_share,
        "f5": f5,
        "posthoc_models": models,
    }


# F4's bound: with the union resident a layer has no miss and the hook must cost ~nothing. Two
# independent measurements agree -- 0.40/0.33/0.36 ms per STEP (09-19 steps 9/10/11, 40/40 layers
# <0.05 ms) and cb(m=0)=0.01 ms over n=1251 pairs (2026-09-20). This is the control a fix has to
# approach: a change that lowers cb without moving it toward this bound has not removed the
# mechanism. FIXED IN ADVANCE, so it cannot be re-tuned to fit a result.
F4_WARM_UNION_MS = 0.05


def f4_check(rep: dict) -> tuple[bool | None, str]:
    g = rep.get("by_m", {}).get("0")
    if not g or not g.get("n"):
        return None, "no m=0 samples: nothing is resident in this log, so F4 cannot be read"
    med = g["median"]
    ok = med <= F4_WARM_UNION_MS
    return ok, (f"median(cb | m=0) = {med:.3f} ms over n={g['n']} (bound <= {F4_WARM_UNION_MS} ms) "
                + ("PASS" if ok else "FAIL"))


def render(rep: dict, logpath: str) -> str:
    out = [f"F1  cb vs per-layer miss count", f"log: {logpath}",
           f"evaluable steps: {rep['evaluable_steps']}   paired (step,layer) samples: {rep['pairs']}",
           ""]
    out.append("per-m (m = that layer's cache misses that step):")
    out.append(f"  {'m':>4} {'n':>6} {'cb_med':>9} {'cb_mean':>9}")
    for k, v in rep["by_m"].items():
        out.append(f"  {k:>4} {v['n']:>6} {v['median']:>9.2f} {v['mean']:>9.2f}")
    out.append("")
    out.append("per round-trip bucket (bucket = ceil(3m/8); the round-trip model predicts cb ∝ bucket):")
    out.append(f"  {'bkt':>4} {'n':>6} {'cb_med':>9}")
    base = None
    for k, v in rep["by_bucket"].items():
        if int(k) == 1:
            base = v["median"]
        ratio = f"   ({v['median']/base:.2f}x bucket1)" if base and int(k) > 1 else ""
        out.append(f"  {k:>4} {v['n']:>6} {v['median']:>9.2f}{ratio}")
    out.append("")
    out.append(f"VERDICT (pre-registered rule, full set): {rep['verdict']}")
    out.append(f"  {rep['reason']}")
    out.append(f"VERDICT (same rule, {len(rep['null_violations'])} null-check samples excluded): "
               f"{rep['verdict_excl_null_violations']}")
    out.append(f"  {rep['reason_excl_null_violations']}")
    out.append("")
    out.append("POST-HOC model comparison (NOT pre-registered; coefficients were not fixed in advance,")
    out.append("so this may not be read as the pre-registered verdict):")
    for name, mt in rep.get("posthoc_models", {}).items():
        if mt.get("r2") is None:
            out.append(f"  {name:<18} n_groups={mt.get('n_groups')} (too few to fit)")
            continue
        out.append(f"  {name:<18} R2={mt['r2']:.4f}  y = {mt['intercept_ms']:.2f} + "
                   f"{mt['slope_ms_per_unit']:.3f} * x")
        out.append(f"    {'x':>4} {'observed_med':>13} {'predicted':>10}")
        for x, y in mt["aug_per_pred"].items():
            out.append(f"    {x:>4} {y:>13.2f} {mt['pred'].get(x, float('nan')):>10.2f}")
    out.append("")
    out.append("secondary checks:")
    nv = rep["null_violations"]
    out.append(f"  null check   : layers with cb<{NO_MISS_CB} but m>0 -> {len(nv)}"
               + ("  OK" if not nv else f"  VIOLATIONS (first 5: {nv[:5]})"))
    ps = rep["path_suspects"]
    out.append(f"  path check   : (step,layer) with cb>=1ms and no BATCHDBG line -> {len(ps)}"
               "  (these ran outside the worker-pool fill: MTP fast path ensure_slot, or m=0)")
    sh = rep.get("cb_share")
    if sh and sh.get("total_ms"):
        out.append(f"  cb share     : {sh['pct_from_layers_with_miss']:.1f}% of all cb ({sh['total_ms']:.0f} ms) "
                   f"comes from layers WITH a miss line; no-miss layers contribute "
                   f"{sh['from_layers_without_miss_ms']:.1f} ms, of which "
                   f"{sh['no_miss_layers_above_noise']} sample(s) reach {NO_MISS_CB} ms")
    out.append(f"  steps needed : >= {MIN_STEPS} -> {rep['evaluable_steps']}"
               + ("  OK" if rep["evaluable_steps"] >= MIN_STEPS else "  TOO FEW"))
    ok, msg = f4_check(rep)
    out.append(f"  F4 warm union: {msg}")
    f5 = rep.get("f5") or {}
    if f5.get("n_steps"):
        out.append(f"  F5 first-miss : L (layers/step carrying >=1 miss) median={f5['L_median']:.1f} "
                   f"mean={f5['L_mean']:.1f} max={f5['L_max']} over {f5['n_steps']} steps  |  "
                   f"median(cb|m=1)={f5['cb_at_m1_median']:.2f} ms (n={f5['cb_at_m1_n']})  |  "
                   f"median(cb|m=0)={f5['cb_at_m0_median']:.3f} ms")
        pops = f5.get("populations") or {}
        for pname in ("all", "work"):
            p = pops.get(pname)
            if not p or not p.get("n_steps"):
                continue
            lead = "  <-- population the arm's median_cb_ms uses" if pname == f5.get("confirm_population") else ""
            out.append(f"                  [{pname:4s}] {p['n_steps']:4d} steps: L median={p['L_median']:.1f} "
                       f"mean={p['L_mean']:.1f} max={p['L_max']}  |  bimodal: "
                       f"{p['warm_steps']}/{p['n_steps']} ({100 * p['warm_share']:.0f}%) FULLY WARM "
                       f"(L=0); when cold at all, mean L={p['L_mean_when_cold']:.1f}{lead}")
        out.append("                  F5 reads off TWO arms: supported if the treated arm lowers L and "
                   "leaves median(cb|m=1) ~unchanged; refuted if L is unchanged.")
    return "\n".join(out)


# ── selftest ────────────────────────────────────────────────────────────────────────────────────

def _synth_log(rows) -> str:
    """rows: list of (ntok, {layer: (misses, cb)}) -- emits a log that exercises the pairing rule."""
    out = []
    for i, (ntok, per) in enumerate(rows, start=1):
        for lay, (m, cb) in sorted(per.items()):
            if m:
                out.append(f"BATCHDBG layer={lay} misses={m} slots: e1->s0")
        out.append(f"CGC-DECPROF: step={i} segs=41 layers=40 total=2.0 ms | "
                   f"wait=1.0 (50%) cb={sum(c for _, c in per.values()):.2f} (10%) submit=0.1 (5%) ntok={ntok}")
        for lay, (m, cb) in sorted(per.items()):
            out.append(f"CGC-DECPROF all: L{lay} wait=1.0 cb={cb:.2f} submit=0.1 ms gpu=1.0 union=1.0 gap=1.0 sg=1 n=1")
    return "\n".join(out) + "\n"


def selftest() -> int:
    import tempfile
    bad = 0

    def check(name, cond, detail=""):
        nonlocal bad
        if not cond:
            bad += 1
            print(f"  FAIL {name} {detail}")
        else:
            print(f"  ok   {name}")

    # 1. PLATEAU fixture: cb = 1.65 * ceil(3m/8) + noise
    rows = []
    for step in range(12):
        per = {}
        per = {}
        for lay in range(32):
            m = (lay % 8) + 1                  # 1..8
            per[lay] = (m, 1.65 * bucket(m))
        for lay in range(32, 40):              # a resident sub-population, for the F4 gate
            per[lay] = (0, 0.01)
        rows.append((4, per))
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(_synth_log(rows))
        p = Path(fh.name)
    r = analyse(parse_log(p))
    check("plateau fixture -> PLATEAU", r["verdict"] == "PLATEAU", r["reason"])
    check("plateau fixture -> null check clean", not r["null_violations"])
    check("plateau fixture -> no path suspects", not r["path_suspects"])

    # 1b. the F4 gate must fire BOTH ways, or it is decoration.
    check("F4 gate passes on a resident-union fixture", f4_check(r)[0] is True, f4_check(r)[1])
    r_bad = {**r, "by_m": {**r["by_m"], "0": {"n": 100, "median": 0.9, "mean": 0.9}}}
    check("F4 gate FAILS when a 'no miss' layer still costs 0.9 ms", f4_check(r_bad)[0] is False)
    check("F4 gate says None when there are no m=0 samples",
          f4_check({**r, "by_m": {"1": {"n": 5, "median": 1.0, "mean": 1.0}}})[0] is None)

    # F5's two readings, on the SAME fixture whose geometry is known by construction: the plateau
    # fixture has 32 layers with m>=1 and 8 resident ones, so L must be exactly 32 -- not 40 (that
    # would be counting layers rather than layers-with-a-miss) and not 8. cb|m=1 is set to 1.65 in
    # that fixture, so the second reading is pinned too. And the F5 block must be ABSENT-safe: a
    # log with no per-layer rows must not crash the renderer.
    check("F5 L counts layers with a miss (32 of 40), not all layers",
          r["f5"]["L_median"] == 32 and r["f5"]["L_mean"] == 32 and r["f5"]["L_max"] == 32,
          str(r["f5"]))
    check("F5 median(cb|m=1) reads the m=1 group, not the pooled median",
          abs(r["f5"]["cb_at_m1_median"] - 1.65) < 1e-9, str(r["f5"]["cb_at_m1_median"]))
    check("F5 renders the empty case without raising",
          isinstance(render({**r, "f5": {"n_steps": 0}}, p), str))

    # 6. The work/shadow split must exist IN THE PARSER, because the inflation it prevents is silent.
    #    Each turn emits a work row (gap>0) and a shadow row (gap_sum=0, tiny total). A parser that
    #    pools them reports L and cb over a population that describes neither.
    def rows(step, gap, cb):
        o = [f"CGC-DECPROF: step={step} segs=41 layers=40 total={cb + 1:.2f} ms | "
             f"wait=1.0 (50%) cb={cb:.2f} (10%) submit=0.1 (5%) ntok=4"]
        for lay in range(40):
            o.append(f"CGC-DECPROF all: L{lay} wait=1.0 cb={cb / 40:.4f} submit=0.1 ms "
                     f"gpu=1.0 union=1.0 gap={gap:.2f} sg=1 n=1")
        return o

    body = []
    for stp in range(1, 13):
        body.append("BATCHDBG layer=0 misses=3 slots: e1->s0")   # only layer 0 misses -> L=1
        body += rows(stp, gap=2.0, cb=5.0)                        # the work row
        body += rows(stp, gap=0.0, cb=0.02)                       # the shadow row
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh6:
        fh6.write("\n".join(body) + "\n")
        p6 = Path(fh6.name)
    r6 = analyse(parse_log(p6))
    kinds = [s["row_kind"] for s in parse_log(p6)["steps"]]
    check("split labels work vs shadow rows",
          kinds.count("work") == 12 and kinds.count("shadow") == 12, str({k: kinds.count(k) for k in set(kinds)}))
    w6 = r6["f5"]["populations"]["work"]
    a6 = r6["f5"]["populations"]["all"]
    check("work population L is the real one (1 layer), not the shadow-diluted pooled value",
          w6["L_median"] == 1 and w6["warm_share"] == 0.0, str(w6))
    check("pooled population is visibly different (shadow rows dilute L toward 0)",
          a6["warm_share"] > 0 or a6["L_median"] != w6["L_median"], str(a6))
    check("confirm_population names the work rows", r6["f5"]["confirm_population"] == "work")
    # and a log with no gap field must be UNSPLIT, never silently called shadow
    nur = "\n".join(["CGC-DECPROF: step=1 segs=41 layers=40 total=2.0 ms | wait=1.0 (50%) cb=1.0 (10%) submit=0.1 (5%) ntok=4",
                     "CGC-DECPROF all: L0 wait=1.0 cb=0.50 submit=0.1 ms gpu=1.0 union=1.0 sg=1 n=1"]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh7:
        fh7.write(nur)
        p7 = Path(fh7.name)
    check("a log without the gap field is UNSPLIT, not shadow",
          parse_log(p7)["steps"][0]["row_kind"] == "UNSPLIT",
          str(parse_log(p7)["steps"][0]["row_kind"]))

    # 2. LINEAR fixture: cb = 1.65 * m
    rows = []
    for step in range(12):
        rows.append((4, {lay: ((lay % 8) + 1, 1.65 * ((lay % 8) + 1)) for lay in range(40)}))
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh2:
        fh2.write(_synth_log(rows))
        p2 = Path(fh2.name)
    r2 = analyse(parse_log(p2))
    check("linear fixture -> LINEAR", r2["verdict"] == "LINEAR", r2["reason"])

    # 3. too few steps
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh3:
        fh3.write(_synth_log([(4, {0: (3, 5.0)})] * 3))
        p3 = Path(fh3.name)
    r3 = analyse(parse_log(p3))
    check("3 steps -> INCONCLUSIVE", r3["verdict"] == "INCONCLUSIVE", r3["reason"])

    # 4. the pairing rule itself: misses printed BEFORE the step line must attach to that step,
    #    and a layer with no line must read as m=0.
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh4:
        fh4.write(
            "BATCHDBG layer=7 misses=6 slots: e1->s0\n"
            "CGC-DECPROF: step=1 segs=41 layers=40 total=2.0 ms | wait=1.0 (50%) cb=9.90 (10%) submit=0.1 (5%) ntok=4\n"
            "CGC-DECPROF all: L7 wait=1.0 cb=4.95 submit=0.1 ms gpu=1.0 union=1.0 gap=1.0 sg=1 n=1\n"
            "CGC-DECPROF all: L8 wait=1.0 cb=0.01 submit=0.1 ms gpu=1.0 union=1.0 gap=1.0 sg=1 n=1\n")
        p4 = Path(fh4.name)
    s4 = parse_log(p4)["steps"][0]
    check("pairing attaches BATCHDBG to its own step", s4["misses"].get(7) == 6, str(s4["misses"]))
    check("layer without a line reads m=0", 8 not in s4["misses"] and s4["cb"].get(8) == 0.01)
    check("bucket() = ceil(3m/8)", [bucket(x) for x in (0, 1, 2, 3, 5, 6, 8, 9)] == [0, 1, 1, 2, 2, 3, 3, 4],
          str([bucket(x) for x in (0, 1, 2, 3, 5, 6, 8, 9)]))

    # 5. the null check must FAIL loudly when it should: 14 pairs, cb tiny but misses claimed.
    rows = [(4, {lay: ((lay % 8) + 1, 0.01) for lay in range(40)}) for _ in range(12)]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh5:
        fh5.write(_synth_log(rows))   # m printed (nonzero) but cb<0.05 -> violations
        p5 = Path(fh5.name)
    r5 = analyse(parse_log(p5))
    check("null check fires when it should", len(r5["null_violations"]) == 480, str(len(r5["null_violations"])))

    # 6. path suspects: cb>=1ms with no BATCHDBG at all
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh6:
        body = "".join(
            f"CGC-DECPROF: step={i} segs=41 layers=40 total=2.0 ms | wait=1.0 (50%) cb=9.90 (10%) submit=0.1 (5%) ntok=4\n"
            f"CGC-DECPROF all: L0 wait=1.0 cb=2.00 submit=0.1 ms gpu=1.0 union=1.0 gap=1.0 sg=1 n=1\n"
            for i in range(1, 13))
        fh6.write(body)
        p6 = Path(fh6.name)
    r6 = analyse(parse_log(p6))
    check("path check counts cb>=1ms with no BATCHDBG", len(r6["path_suspects"]) == 12, str(len(r6["path_suspects"])))

    for f in (p, p2, p3, p4, p5, p6):
        f.unlink(missing_ok=True)
    print(f"\nselftest: {'PASS' if bad == 0 else f'{bad} FAILED'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="", help="a server log carrying DECPROF all + BATCHDBG rows")
    ap.add_argument("--ntok", default="", help="comma list of ntok values to include (default: >=2)")
    ap.add_argument("--json-out", default="", help="write the report dict here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.log:
        ap.error("--log or --selftest is required")
    flt = None
    if args.ntok:
        flt = {int(x) for x in args.ntok.split(",") if x.strip()}
    rep = analyse(parse_log(Path(args.log)), ntok_filter=flt)
    print(render(rep, args.log))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
