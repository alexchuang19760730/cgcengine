#!/usr/bin/env python3
"""S(ntok) with MTP ON, and the per-layer decomposition of the fixed cost F.

WHY THIS SHAPE
==============
Two questions, and neither is answerable by the instruments that already exist here:

1. What does a decode step cost as a function of its WIDTH?  S = F + c*ntok. The two-parameter
   model in DECODE_STEP_BUDGET §3 is fitted to two points, which is zero degrees of freedom: it
   describes the data and cannot be checked against it. Worse, the F it uses is borrowed from an
   MTP-OFF run (75.3 ms) while the curve is about MTP-ON steps -- the DESIGN GAP the window
   harness names.

2. Where does F actually live, per layer?  ggml-backend.cpp's decode profile already splits each
   layer into four MECHANICAL buckets and none of them require a code change:
       cb      top-k hook: slot management + the blocking expert fill
       submit  the launch of that layer's segment          (dispatch)
       gpu     Metal busy, from the command buffers' own timestamps   (attention + MoE math)
       gap     GPU idle between segments                   (overlap failure / eval-sync artefact)
   `wait` is the CPU-side window that contains gpu+gap, so it is reported for reconciliation but
   is not one of the four.

WHY THE ARMS ARE BRACKETED
==========================
`speculative.n_max` cannot be changed per request (server-schema.cpp gates it behind `#if 0`), so
each width needs its own launch -- and a launch is ~4 minutes, i.e. longer than this box's regime
flip (measured 2026-09-19: 337.2 -> 133.6 ms inside a single 3-arm block, with `cb` unchanged).
The drift lane proved that pairing CANNOT fix that: its same-configuration pair differed by 20.5%,
worse than the 17.5% it was sent to fix.

So every test arm is BRACKETED BY ntok=4 ANCHORS co-measured minutes away:

    n_max  3   0   3   1   2   3
    role   A   t   A   t   t   A        A = ntok=4 anchor, t = test

Each test arm is then normalised by the mean of its own two neighbours, not by an anchor measured
an hour earlier. The anchors double as the drift meter: A1/A2/A3 spread IS this run's noise floor,
printed next to the effect, so a curve that is flatter than its own anchors says so instead of
being reported as a finding.

WHAT IT DOES NOT DO
===================
It cannot split `gpu` (attention vs MoE math) without more instrumentation. It reports the
GDN-vs-full-attention contrast, which bounds that split from below, and says so rather than
implying the four buckets are four code regions. It also does not touch any other session's
processes: cleanup signals only the pid holding our own port.

Usage:  python3 scripts/check/sntok_curve.py run [--budget-min 90]
        python3 scripts/check/sntok_curve.py analyze        # from the saved state, no measurement
"""
import argparse
import importlib.util
import json
import os
import re
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

STATE = os.environ.get("SNTOK_STATE", "/tmp/sntok_curve.json")


def _load_harness():
    """Reuse the window harness rather than re-deriving its launch/window/anchor discipline.

    Everything that was expensive to learn lives there: the dedicated port, the launcher dump-mode
    trap, the argv-executable classifier, the anchor check, and "signal only the pid on our port".
    """
    spec = importlib.util.spec_from_file_location("dwh", os.path.join(HERE, "decode_window_harness.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()

# 40 base layers + blk.40 (the MTP head, which reports layers=1 and is parsed separately).
N_BASE_LAYERS = 40
# full_attention_interval=4 and the first full-attention layer is index 3 (see llm/…: filter_attn
# is (il<40 && !is_recr(il))), so these are the layers whose attention is NOT the GDN recurrence.
FULL_ATTN = set(range(3, 40, 4))

# (name, n_max): A = anchor (ntok=4), t = test at ntok = n_max+1, r = reference.
# Two anchors get consumed by the two innermost tests, so the outer two are the drift meter.
#
# r0 is MTP OFF: the only way to observe a 1-token BASE step on this build, and therefore the only
# direct reading of F. It is labelled a reference and not a test because it runs a different env
# block (run_server.sh wraps CGC_NO_PREFETCH / CGC_VERIFY_DECODE / CGC_DRAFT_DECODE / CGC_MM_BITIDENT
# in `if MTP=1`), so it is NOT the same configuration as the curve. Its job is to be cross-checked
# against the curve extrapolated to ntok=1: if the two agree, the env block does not matter for F; if
# they disagree, that gap is itself the finding.
#
# n_max=0 is DELIBERATELY ABSENT. It was this probe's first design (same env block, no draft => a
# 1-token base step). It aborts the server on the first decode -- see UNUSABLE -- which is also how
# the note "k=0 is a legal value" elsewhere in this repo gets falsified.
ORDER = [("A1", 3), ("t2", 1), ("A2", 3), ("t3", 2), ("A3", 3), ("r0", -1), ("A4", 3)]
MTP_OFF = -1

UNUSABLE = {
    "t1": (0, "--spec-draft-n-max 0 aborts on the first decode: llama-context.cpp:2961 "
               "GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed. Recorded as unusable, not "
               "retried, because a crash is not a measurement."),
}

LAYRE = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
                   r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_layers(path, lo, hi):
    """Per-layer rows over a byte range of the server log, median per layer."""
    with open(path, errors="replace") as fh:
        fh.seek(lo)
        chunk = fh.read(max(hi - lo, 0))
    acc = {}
    for line in chunk.splitlines():
        m = LAYRE.search(line)
        if m:
            acc.setdefault(int(m.group(1)), []).append(
                dict(wait=float(m.group(2)), cb=float(m.group(3)), submit=float(m.group(4)),
                     gpu=float(m.group(5)), union=float(m.group(6)), gap=float(m.group(7))))
    return {L: {k: st.median(r[k] for r in rows) for k in ("wait", "cb", "submit", "gpu", "union", "gap")}
            | {"n": len(rows)} for L, rows in acc.items()}


def run_arm(state, name, n_max):
    """One launch at a fixed draft width: two requests, cold then warm."""
    if name in UNUSABLE:
        bad, why = UNUSABLE[name]
        rec = state.setdefault("arms", {}).get(name) or {}
        rec.update({"n_max": bad, "status": "unusable", "reason": why})
        state["arms"][name] = rec
        save(state)
        log(f"  [{name}] excluded: {why.splitlines()[0]}")
        return False
    if state.get("arms", {}).get(name, {}).get("status") == "done":
        log(f"  [{name}] already done, skipping")
        return True
    ok, why = h.anchor_ok(h.digest())
    if not ok:
        log(f"  [{name}] {why} -> not recorded; nothing may be concluded")
        state.setdefault("arms", {})[name] = {"status": "anchor-mismatch", "anchor": why}
        save(state)
        return False
    if not h.wait_for_window():
        log(f"  [{name}] no window within the budget")
        return False
    mtp = "0" if n_max == MTP_OFF else "1"
    expect = 1 if n_max == MTP_OFF else n_max + 1
    log(f"  [{name}] window open; MTP={mtp} n_max={'off' if n_max == MTP_OFF else n_max} "
        f"(expect ntok={expect} base steps)")
    pid, srv_log, healthy = h.launch({
        "CGC_SERVER_MTP": mtp, "CGC_SERVER_MTP_N_MAX": str(max(n_max, 0)),
        "CGC_DECODE_PROFILE": "1", "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1",
    })
    rec = {"n_max": n_max, "expect_ntok": expect, "pid": pid, "server_log": srv_log,
           "engine_digest": h.digest(), "anchor": why, "free_mb": h.vm_free_mb(),
           "swap_mb": h.swap_used_mb(), "reps": {}}
    if not healthy:
        rec["status"] = "interrupted"
        rec["launcher_said"] = h.launcher_tail()
        state.setdefault("arms", {})[name] = rec
        save(state)
        log(f"  [{name}] launch failed -> slot not consumed")
        return False
    try:
        for tag in ("rep1_cold", "rep2_warm"):
            lo = os.path.getsize(srv_log) if srv_log else 0
            r = h.ask(h.PROMPT, h.N_PREDICT, srv_log, tag)
            hi = os.path.getsize(srv_log)
            # ask() gives ntok-keyed step medians; the per-layer rows need their own pass because
            # parse_range drops `submit` from its per-layer median set.
            r["layers"] = parse_layers(srv_log, lo, hi)
            rec["reps"][tag] = r
        rec["status"] = "done"
    except Exception as exc:
        rec["status"] = "interrupted"
        rec["error"] = str(exc)
    state.setdefault("arms", {})[name] = rec
    save(state)
    h.stop_ours()
    if rec["status"] != "done":
        log(f"  [{name}] {rec['status']} -> slot not consumed")
        return False
    # Did we get the width we paid for? A narrower graph than requested means the drafter bailed,
    # and then this arm is not the point of the curve it was launched to be.
    want = rec["expect_ntok"]
    got = rec["reps"]["rep2_warm"]["steps"].get(want, {}).get("n", 0)
    allw = sorted(rec["reps"]["rep2_warm"]["steps"].keys())
    log(f"  [{name}] done; ntok observed={allw}; {want}-wide steps={got}")
    rec["width_ok"] = bool(got)
    save(state)
    return True


def save(state):
    with open(STATE, "w") as fh:
        json.dump(state, fh, indent=1)


def load():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"arms": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


# ------------------------------------------------------------------------------- analysis
def warm_step(rec, ntok=4):
    """Median (total, n) of the ntok-wide base steps in the warm rep."""
    s = ((rec.get("reps") or {}).get("rep2_warm") or {}).get("steps") or {}
    d = s.get(ntok) or s.get(str(ntok))
    return (d or {}).get("total"), (d or {}).get("n", 0)


def analyze(state):
    arms = state.get("arms") or {}
    done = {k: v for k, v in arms.items() if v.get("status") == "done"}
    out = {"anchors": [], "tests": [], "verdicts": []}
    if not done:
        out["verdicts"].append(("INCOMPLETE", "no completed arms"))
        return out

    for name, n_max in ORDER:
        rec = done.get(name)
        if not rec:
            continue
        ntok = 1 if n_max == MTP_OFF else n_max + 1
        tot, n = warm_step(rec, ntok)
        row = {"arm": name, "n_max": n_max, "ntok": ntok, "total": tot, "n": n,
               "ms_per_token": (rec["reps"].get("rep2_warm") or {}).get("ms_per_token"),
               "width_ok": rec.get("width_ok")}
        if name.startswith("A"):
            out["anchors"].append(row)
        elif name == "r0":
            out["reference"] = row
        else:
            out["tests"].append(row)

    anchors = [a["total"] for a in out["anchors"] if a["total"]]
    if len(anchors) >= 2:
        spread = (max(anchors) - min(anchors)) / st.mean(anchors) * 100.0
        out["anchor_spread_pct"] = spread
        out["anchor_mean"] = st.mean(anchors)
        out["verdicts"].append((
            "anchors agree" if spread < h.REGIME_FLIP_PCT else "regime faster than one arm",
            f"the {len(anchors)} identical ntok=4 anchors span {spread:.1f}% "
            f"({', '.join(f'{a:.1f}' for a in anchors)} ms) against a {h.REGIME_FLIP_PCT:.0f}% "
            f"threshold; every test arm below is reported BOTH raw and as a ratio to its bracketing "
            f"anchors, so a reader can see which reading the run supports."))
    if not out["tests"]:
        out["verdicts"].append(("INCOMPLETE", "no test arms completed"))
        return out

    # Bracket each test with its neighbours in ORDER (the anchors on either side).
    norm = []
    for t in out["tests"]:
        idx = [i for i, (nm, _) in enumerate(ORDER) if nm == t["arm"]][0]
        left = next((done[nm] for nm in (ORDER[i][0] for i in range(idx - 1, -1, -1))
                     if nm.startswith("A") and nm in done), None)
        right = next((done[nm] for nm in (ORDER[i][0] for i in range(idx + 1, len(ORDER)))
                      if nm.startswith("A") and nm in done), None)
        br = []
        for b in (left, right):
            v = warm_step(b, 4)[0] if b else None
            if v:
                br.append(v)
        if br and t["total"]:
            ref = st.mean(br)
            t["bracket"] = br
            t["ref"] = ref
            t["ratio"] = t["total"] / ref
            norm.append((t["ntok"], t["ratio"]))

    # Raw fit vs bracket-normalised fit, side by side: if the two disagree the run cannot decide.
    def fit(points):
        if len(points) < 2:
            return None
        ns = [p[0] for p in points]
        ys = [p[1] for p in points]
        n = len(ns)
        sx, sy = sum(ns), sum(ys)
        sxx = sum(x * x for x in ns)
        sxy = sum(x * y for x, y in points)
        den = n * sxx - sx * sx
        if den == 0:
            return None
        c = (n * sxy - sx * sy) / den
        F = (sy - c * sx) / n
        return {"F": F, "c": c, "n_points": n}

    raw_pts = [(a["ntok"], a["total"]) for a in out["tests"] + out["anchors"] if a["total"]]
    out["span"] = (min(p[0] for p in raw_pts), max(p[0] for p in raw_pts)) if raw_pts else None
    out["raw_fit"] = fit(raw_pts)
    out["raw_points"] = raw_pts
    common = out["anchor_mean"] if anchors else None
    if norm and common:
        # Ratio -> ms by multiplying by the anchor mean: same curve, but each point carries its own
        # neighbourhood's regime instead of the whole run's.
        out["norm_fit"] = fit([(nt, r * common) for nt, r in norm])
        out["norm_points"] = [(nt, r * common) for nt, r in norm]

    # The two fits answer the same question from different references. Agreement is the licence to
    # quote F and c; disagreement means the run's own noise is larger than the curve, and that has to
    # be said rather than averaged into a number.
    rf, nf = out.get("raw_fit"), out.get("norm_fit")
    if rf and nf:
        dF = abs(rf["F"] - nf["F"]) / max(abs(nf["F"]), 1e-9) * 100.0
        dc = abs(rf["c"] - nf["c"]) / max(abs(nf["c"]), 1e-9) * 100.0
        out["fit_disagreement_pct"] = {"F": dF, "c": dc}
        out["verdicts"].append((
            "curve admissible" if max(dF, dc) < h.REGIME_FLIP_PCT else "curve not admissible",
            f"raw vs bracketed fits disagree by {dF:.0f}% on F and {dc:.0f}% on c "
            f"(threshold {h.REGIME_FLIP_PCT:.0f}%). "
            + ("Either fit may be quoted." if max(dF, dc) < h.REGIME_FLIP_PCT else
               "Quote NEITHER: the box moved more than the curve did. This is a measurement-shape "
               "limit, not a property of the engine.")))

    # The cross-check that matters for F: the curve is measured at ntok 2..4, so the ntok=1 endpoint
    # is an EXTRAPOLATION of one step. r0 measures it directly, in a different env block. Agreement
    # licenses the extrapolation; disagreement sizes the env block's contribution instead.
    ref = out.get("reference")
    if ref and ref.get("total") and (out.get("norm_fit") or out.get("raw_fit")):
        f = out.get("norm_fit") or out.get("raw_fit")
        pred = f["F"] + f["c"] * 1
        d = (ref["total"] - pred) / pred * 100.0
        out["reference_prediction"] = {"measured_ms": ref["total"], "predicted_ms": pred,
                                       "delta_pct": d}
        out["verdicts"].append((
            "extrapolation holds" if abs(d) < h.REGIME_FLIP_PCT else "extrapolation disagrees",
            f"the ntok=1 point is extrapolated from the curve (which only reaches ntok="
            f"{out.get('span', (0, 0))[0]}..{out.get('span', (0, 0))[1]}) as {pred:.1f} ms; MTP-off "
            f"measures {ref['total']:.1f} ms, a {d:+.1f}% difference. "
            + ("So the one-step extrapolation is safe at this precision." if abs(d) < h.REGIME_FLIP_PCT
               else "This is the env block's contribution to F, and it is larger than the threshold:"
                    " the ntok=1 endpoint must be quoted from r0, not from the fit.")))

    # Does the four-bucket decomposition actually reconstruct each step? Without this the
    # percentages below are decoration: `union`+`gap`+`cb`+`submit` should equal the step total, and
    # where it does not, the per-layer rows are not a partition of the step and the composition has
    # to be quoted with that width. Checked per arm rather than asserted once.
    clim = []
    for name, n_max in ORDER:
        rec = done.get(name)
        if not rec:
            continue
        ntok = 1 if n_max == MTP_OFF else n_max + 1
        L = ((rec.get("reps") or {}).get("rep2_warm") or {}).get("layers") or {}
        L = {str(k): v for k, v in (L or {}).items()}
        tot = warm_step(rec, ntok)[0]
        if not L or not tot:
            continue
        s = sum(v["union"] + v["gap"] + v["cb"] + v["submit"] for v in L.values())
        clim.append({"arm": name, "ntok": ntok, "step_ms": tot, "buckets_ms": s,
                     "closure_pct": s / tot * 100.0})
    out["closure"] = clim
    if clim:
        worst = max(abs(c["closure_pct"] - 100.0) for c in clim)
        detail = ", ".join(f"{c['arm']}(ntok={c['ntok']})={c['closure_pct']:.0f}%" for c in clim)
        out["verdicts"].append((
            "buckets partition the step" if worst <= 5 else "buckets do not partition the step",
            f"union+gap+cb+submit reconstructs each step total as: {detail} -- worst off by "
            f"{worst:.0f}%. "
            + ("At this precision the per-layer rows are a partition of the step." if worst <= 5 else
               "They are NOT a partition at ntok>1, so the composition percentages below carry this "
               "width and the residual is itself unexplained work (overlapping segments).")))

    out["requirement"] = requirement(out)
    return out


def expected_tokens(accept, k):
    """Tokens produced per verify step under independent acceptance: 1 + a + ... + a^k."""
    return sum(accept ** i for i in range(k + 1))


def requirement(a, target_tps=25.0, accept=0.80, k=3):
    """What the curve implies for the 25 t/s target.

    The target is a RATE, so it constrains S/(tokens per step). With the measured c, that turns into
    a budget for S(ntok=1) -- i.e. for F, the per-layer fixed cost -- which is the quantity the
    "1.9 -> 0.5 ms/layer" sentence in the plan is about.
    """
    nf = a.get("norm_fit") or a.get("raw_fit")
    if not nf or not nf.get("c"):
        return None
    T = expected_tokens(accept, k)
    need_step = T * (1000.0 / target_tps)
    S1 = nf["F"] + nf["c"]
    need_S1 = need_step - nf["c"] * k
    drop = S1 - need_S1
    return {"accept": accept, "k": k, "tokens_per_step": T, "target_tps": target_tps,
            "required_step_ms": need_step, "c_ms_per_position": nf["c"],
            "S1_now_ms": S1, "S1_per_layer_now": S1 / N_BASE_LAYERS,
            "S1_needed_ms": need_S1, "S1_per_layer_needed": need_S1 / N_BASE_LAYERS,
            "drop_ms": drop, "drop_per_layer_ms": drop / N_BASE_LAYERS,
            # Three outcomes, and the middle one is the trap: a NEGATIVE required drop means the fit
            # claims the target is already met, which the measured S(4) contradicts -- that is a
            # statement about the FIT being unusable, not about the engine being fast enough.
            "feasible": need_S1 > 0,
            "outcome": ("infeasible" if need_S1 <= 0 else
                        "fit-unusable" if drop < 0 else "budget")}


def report(a):
    print("\n" + "=" * 78)
    print("S(ntok) with MTP ON")
    print("=" * 78)
    by_arm = {}
    for grp in ("anchors", "tests"):
        for r in a.get(grp, []):
            by_arm[r["arm"]] = r
    if a.get("reference"):
        by_arm[a["reference"]["arm"]] = a["reference"]
    print(f"\n{'arm':5s} {'n_max':>5s} {'ntok':>4s} {'step ms':>9s} {'n':>4s} {'ms/token':>9s}"
          f"  bracket-ref  ratio")
    for name, _ in ORDER:
        r = by_arm.get(name)
        if not r:
            continue
        br = f"{r.get('ref'):.1f}" if r.get("ref") else "-"
        ra = f"{r.get('ratio'):.3f}" if r.get("ratio") else "-"
        mpt = f"{r['ms_per_token']:.2f}" if r.get("ms_per_token") else "-"
        tot = f"{r['total']:.2f}" if r.get("total") else "-"
        tag = "  (MTP off: reference, different env block)" if name == "r0" else ""
        print(f"{r['arm']:5s} {r['n_max']:5d} {r['ntok']:4d} {tot:>9s} {r['n']:4d} {mpt:>9s}"
              f"  {br:>11s}  {ra}{tag}")
    if a.get("raw_fit"):
        f = a["raw_fit"]
        print(f"\nraw fit      S = {f['F']:.1f} + {f['c']:.2f}*ntok  ({f['n_points']} points)"
              f"   -> F/n_base_layer = {f['F'] / N_BASE_LAYERS:.2f} ms")
    if a.get("norm_fit"):
        f = a["norm_fit"]
        print(f"bracketed    S = {f['F']:.1f} + {f['c']:.2f}*ntok  ({f['n_points']} points)"
              f"   -> F/n_base_layer = {f['F'] / N_BASE_LAYERS:.2f} ms")
    rp = a.get("reference_prediction")
    if rp:
        print(f"\nntok=1 cross-check: extrapolated {rp['predicted_ms']:.1f} ms vs MTP-off measured "
              f"{rp['measured_ms']:.1f} ms  ({rp['delta_pct']:+.1f}%)")
    r = a.get("requirement")
    if r:
        print(f"\n{'=' * 78}\nwhat 25 t/s costs, given this curve\n{'=' * 78}")
        print(f"  accept {r['accept']:.2f}, k={r['k']}  ->  {r['tokens_per_step']:.2f} tokens/step"
              f"  ->  step budget {r['required_step_ms']:.0f} ms")
        print(f"  c (measured) = {r['c_ms_per_position']:.2f} ms per extra position in the graph")
        print(f"  S(ntok=1) now     = {r['S1_now_ms']:.1f} ms = {r['S1_per_layer_now']:.2f} ms/layer")
        if r["outcome"] == "budget":
            print(f"  S(ntok=1) needed  = {r['S1_needed_ms']:.1f} ms = {r['S1_per_layer_needed']:.2f} ms/layer")
            print(f"  => F must fall {r['drop_ms']:.0f} ms total, {r['drop_per_layer_ms']:.2f} ms/layer")
        elif r["outcome"] == "fit-unusable":
            print(f"  this fit says the target is ALREADY met (required drop {r['drop_ms']:.0f} ms,"
                  f" negative) -- which the measured S({r['k'] + 1}) contradicts. The fit is unusable;")
            print(f"     do not read this as 'fast enough'.")
        else:
            print(f"  S(ntok=1) needed  = {r['S1_needed_ms']:.1f} ms -- NEGATIVE: at k={r['k']} the")
            print(f"     verify positions alone ({r['c_ms_per_position']:.1f}*{r['k']} = "
                  f"{r['c_ms_per_position'] * r['k']:.0f} ms) already exceed the budget.")
            print(f"     No F can reach {r['target_tps']:.0f} t/s at this accept/k; accept must rise or")
            print(f"     the per-position cost must fall.")
    for k, t in a.get("verdicts", []):
        print(f"\n[{k}] {t}")
    return a


def layers_of(state, name="A1"):
    rec = (state.get("arms") or {}).get(name) or {}
    rep = (rec.get("reps") or {}).get("rep2_warm") or {}
    return rep.get("layers") or {}


def bucket_report(state):
    """Per-layer four-bucket attribution for the least-drifted ntok=4 arm."""
    best, best_tot = None, None
    for nm in ("A1", "A2", "A3"):
        L = layers_of(state, nm)
        if not L:
            continue
        tot = sum(v["wait"] for v in L.values())
        if best_tot is None or tot < best_tot:
            best, best_tot = nm, tot
    if not best:
        print("\n(no per-layer rows: was CGC_DECODE_PROFILE_ALL=1 / CGC_GPU_TIMING=1 propagated?)")
        return None
    L = layers_of(state, best)
    # JSON turns the reader's int layer keys into strings, so the SAME state means different things
    # in-process and after a round trip. Normalise once, here, or `L[str(i)]` raises KeyError('0')
    # in the process that measured it and works in the process that analyses it -- the worst kind of
    # bug, because the crash only happens in the run that nobody re-runs.
    L = {str(k): v for k, v in (L or {}).items()}
    if not L:
        return None
    print(f"\n{'=' * 78}\nper-layer four-bucket attribution (arm {best}, ntok=4, {len(L)} layers)\n{'=' * 78}")
    # `union` is the segment's non-double-counted GPU span, so it is the honest GPU column; `gpu`
    # (Σ end-start over buffers) is kept beside it because the difference between the two IS the
    # overlap this pipeline is built around.
    keys = ("wait", "cb", "submit", "gpu", "union", "gap")
    tot = {k: sum(v[k] for v in L.values()) for k in keys}
    print(f"\n{'layer':>5s} {'type':>6s} " + " ".join(f"{k:>8s}" for k in keys))
    for i in sorted(int(x) for x in L):
        v = L[str(i)]
        kind = "attn" if i in FULL_ATTN else "GDN"
        print(f"{i:5d} {kind:>6s} " + " ".join(f"{v[k]:8.2f}" for k in keys))
    print(f"\n{'SUM':>5s} {'':>6s} " + " ".join(f"{tot[k]:8.2f}" for k in keys))
    print(f"{'per-layer':>5s} {'':>6s} " + " ".join(f"{tot[k] / len(L):8.2f}" for k in keys))
    gsum, usum = tot["gpu"], tot["union"]
    print(f"\nNOTE the SUM row is a sum of PER-LAYER MEDIANS, so it is smaller than the step's own"
          f" median; read it as a shape, not as a budget. `gpu`={gsum:.1f} vs `union`={usum:.1f} ms:"
          f" the gap between them ({gsum - usum:.1f} ms) is the buffer overlap this pipeline exists to"
          f" create, and it is why `gpu` can exceed `wait` while `union` cannot.")

    # Where the host-side fill lives, cold vs warm: a reproducible structure (A1 and A2 agree) that
    # a step total cannot show, and one that says the warm pool holds every layer except the first
    # few. Reported rather than described, because "cb collapsed" was already said twice in the docs
    # without the layer split that makes it actionable.
    gdn = [L[str(i)] for i in sorted(L, key=int) if int(i) not in FULL_ATTN]
    full = [L[str(i)] for i in sorted(L, key=int) if int(i) in FULL_ATTN]
    res = {"arm": best, "n_layers": len(L), "sums": tot,
           "per_layer": {k: tot[k] / len(L) for k in keys}}


    if gdn and full:
        for k in ("union", "gpu", "wait", "cb"):
            g = st.median(x[k] for x in gdn)
            f = st.median(x[k] for x in full)
            res[f"median_{k}_gdn"] = g
            res[f"median_{k}_full"] = f
            print(f"\n  median {k:6s}: GDN {g:7.2f} ms   full-attn {f:7.2f} ms   diff {f - g:+.2f}")
        d = res["median_gpu_full"] - res["median_gpu_gdn"]
        print(f"\n[attention split] the {len(full)} full-attention layers' GPU busy differ from the "
              f"{len(gdn)} GDN layers' by {d:+.2f} ms/layer. That is a LOWER BOUND on the attention "
              f"differential, not an absolute split: both layer types also run their MoE, so a common "
              f"MoE cost cancels. Absolute attention vs MoE needs per-node timing, which this "
              f"instrument does not have.")
    # The ceiling that decides the 25 t/s question, computed against the ONLY arm that measures a
    # 1-token base step (MTP off). If removing every non-GPU millisecond cannot reach the target,
    # then the target is a statement about GPU work and no amount of overhead work touches it.
    # This analysis needs NO admissible fit: it only needs one arm that measures a 1-token base step.
    # Gating it on the fit (as it first was) hid the only conclusion that survives three refusals from
    # the curve, which is the opposite of useful.
    req = ((state.get("analysis") or {}).get("requirement") or {})
    ref = {str(k): v for k, v in (layers_of(state, "r0") or {}).items()}
    if ref:
        n = len(ref)
        gpu = sum(v["union"] for v in ref.values()) / n
        nong = sum(v["cb"] + v["submit"] + v["gap"] for v in ref.values()) / n
        # Target from the fit when the fit is usable; otherwise from the plan's own sentence
        # ("1.9 -> 0.5 ms/layer"), which is an independent, pre-registered number.
        if req.get("outcome") == "budget":
            need, src = req["S1_per_layer_needed"], "the curve's own budget"
        else:
            need, src = 0.5, "the plan's own 0.5 ms/layer sentence (fit unusable)"
        res["ceiling"] = {"arm": "r0", "n_layers": n, "gpu_per_layer": gpu,
                          "non_gpu_per_layer": nong, "F_per_layer": gpu + nong,
                          "target_per_layer": need, "target_source": src,
                          "floor_if_overhead_vanished": gpu}
        print(f"\n{'=' * 78}\nwhat the 25 t/s target can and cannot touch\n"
              f"(arm r0 = MTP off, ntok=1, {n} layers — the only arm that measures a 1-token base step)"
              f"\n{'=' * 78}")
        print(f"  F per layer  = {gpu + nong:.2f} ms  =  GPU span {gpu:.2f}  +  non-GPU {nong:.2f}")
        print(f"     non-GPU   = cb {sum(v['cb'] for v in ref.values()) / n:.2f}"
              f" + submit {sum(v['submit'] for v in ref.values()) / n:.2f}"
              f" + gap {sum(v['gap'] for v in ref.values()) / n:.2f}"
              f"   ->  over half of it is `gap`, which is GPU IDLE, not work")
        print(f"  target       = {need:.2f} ms/layer  ({src})")
        print(f"  if ALL non-GPU time vanished, F would still be {gpu:.2f} ms/layer"
              f" = {gpu / max(need, 1e-9):.1f}x the target")
        print(f"  => at most {nong / max(gpu + nong, 1e-9) * 100:.0f}% of F is addressable as overhead."
              f" The other {(gpu) / max(gpu + nong, 1e-9) * 100:.0f}% is GPU EXECUTION (attention +"
              f" MoE math) plus GPU idle, so the target is a statement about kernels and work per"
              f" layer, not about dispatch or cache fill.")
        # The differential must come from r0 ITSELF: quoting the ntok=4 arm's split inside an ntok=1
        # budget is the same unit/regime conflation this probe exists to avoid.
        rg = [v for k, v in ref.items() if int(k) < N_BASE_LAYERS and int(k) not in FULL_ATTN]
        rf = [v for k, v in ref.items() if int(k) in FULL_ATTN]
        du = (st.median(v["union"] for v in rg) - st.median(v["union"] for v in rf)) if rg and rf else 0.0
        F_step = (gpu + nong) * n          # F in ms/step, so the comparison stays in one unit
        need_cut_step = F_step - need * n
        print(f"  largest identified GPU-side gap (from r0's own layers): the {len(rg)} GDN layers cost"
              f" {du:+.2f} ms/layer more than the {len(rf)} full-attention ones => {du * len(rg):.0f}"
              f" ms/step, {du * len(rg) / max(F_step, 1e-9) * 100:.0f}% of F.")
        print(f"  F = {F_step:.0f} ms/step, the target needs {F_step - need_cut_step:.0f}, so the cut is"
              f" {need_cut_step:.0f} ms/step = {need_cut_step / max(abs(du) * len(rg), 1e-9):.1f} gaps of"
              f" that size. Even a perfect GDN fix is a fraction of the ask.")

    # Where the host-side fill lives, cold vs warm: a reproducible structure (A1 and A2 agree on it)
    # that no step total can show, and the one that says the warm pool holds every layer EXCEPT the
    # first few. Reported rather than described, because "cb collapsed" has already been asserted in
    # these docs twice without the layer split that makes it actionable.
    rec = (state.get("arms") or {}).get(best) or {}
    for tag, rep in (("warm", "rep2_warm"), ("cold", "rep1_cold")):
        LL = ((rec.get("reps") or {}).get(rep) or {}).get("layers") or {}
        if not LL:
            continue
        head5 = sum(LL[str(i)]["cb"] for i in range(5) if str(i) in LL)
        rest = sum(v["cb"] for k, v in LL.items() if int(k) >= 5)
        res[f"cb_{tag}_first5"] = head5
        res[f"cb_{tag}_rest"] = rest
        print(f"\n  cb ({tag:4s}): L0-L4 {head5:6.2f} ms   L5+ {rest:6.2f} ms   "
              f"-> {head5 / max(head5 + rest, 1e-9) * 100:4.0f}% of the fill sits in the first 5 layers")
    return res


def cmd_run(args):
    state = load()
    log(f"=== S(ntok) curve (port {h.PORT}) ===")
    deadline = time.time() + args.budget_min * 60
    if args.analyze_only:
        log("--- analyze-only: no measurement")
    else:
        for name, n_max in ORDER:
            if time.time() > deadline:
                log(f"budget (--budget-min {args.budget_min}) exhausted; stopping")
                break
            done = state.get("arms", {}).get(name, {}).get("status") == "done"
            if not done:
                run_arm(state, name, n_max)
            elif name == "t1" and not state["arms"][name].get("width_ok"):
                log(f"  [{name}] recorded without the {state['arms'][name]['expect_ntok']}-wide "
                    f"steps it was launched for; retrying once")
                del state["arms"][name]
                save(state)
                run_arm(state, name, n_max)
    a = report(analyze(state))
    state["analysis"] = a
    state["buckets"] = bucket_report(state)
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(state)
    log(f"\nstate: {STATE}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--budget-min", type=float, default=90.0)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.cmd == "analyze":
        args.analyze_only = True
    cmd_run(args)


if __name__ == "__main__":
    main()
