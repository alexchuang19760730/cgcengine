#!/usr/bin/env python3
"""Is `plain_match`'s divergence onset a FIXED sequence position, or an END-of-generation artifact?

The question this answers
------------------------
`plain_match` reported `PLAIN_MATCH FALSE` with `first_divergence_pos = 205` over a 262-character
comparison (57 diverging positions, every one of them after the first). Two mechanisms predict the
same single observation and different sweeps:

  * FIXED ONSET   -- something kicks in at a particular sequence position (a kernel/qualification
                    switch, an accumulated-state effect, cache state). Then the first divergence
                    stays put as `n_predict` grows, and a run that stops short of the onset is CLEAN.
  * END TRACKING  -- the last rounds behave differently from the rest (a stop-path or final-batch
                    artifact). Then the onset sits a constant distance from the END, i.e. it MOVES
                    with `n_predict`, and no `n_predict` is clean.

These are separable with one axis and they imply opposite next actions: fixed onset means find the
position-dependent switch; end tracking means the defect is in the stopping path and no amount of
prefix tuning will show it.

Design, and why it is cheap
---------------------------
The comparison is greedy and deterministic, so the tokens a run emits do not depend on how many it
was *allowed* to emit. That means one server start per arm can serve every `n_predict`: the arms are
launched once each and then asked for 64/128/256/384 tokens in turn. Two launches total, versus the
one-launch-per-arm-per-config shape of `plain_match_ab`. The assumption is not taken on faith: each
arm's long run must *start with* its own short run, and that prefix check is reported.

Units: ids when the server returns them (it does, once asked -- see `plain_match_ab.decode`), else
rendered characters. Character comparison is the weaker instrument: one differing token shifts every
later character, which is why `diverging_positions` should then equal `len - first_pos`. The report
says which unit it used and never mixes them.

Usage
-----
    python3 scripts/check/divergence_onset_sweep.py selftest
    python3 scripts/check/divergence_onset_sweep.py run --wait-window
    python3 scripts/check/divergence_onset_sweep.py run --n-predicts 64,128,256,384 --json ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/alexchuang/Documents/flashkv-devserver")

# The grid brackets the observed onset: 128 is the run that produced `first_divergence_pos = 205`,
# 64 is short enough that a fixed onset at ~205 characters cannot appear at all, and 256/384 make
# the tail long enough that an end-tracking onset would be forced to move by >100 units.
N_PREDICTS = (64, 128, 256, 384)

# Pre-registered before the run, so the result cannot be fitted to afterwards.
PREDICTIONS = [
    ("fixed-onset", "first divergence constant across cuts; n_predict=64 CLEAN; "
                    "onset present at 128/256/384 at the same position"),
    ("end-tracking", "first divergence = cut - K with K constant; n_predict=64 shows the onset "
                     "near 64-K; every cut is divergent"),
]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "check" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- pure analysis
def first_div(a: list, b: list) -> tuple[int | None, int, int]:
    """(index of first difference or None, units compared, differing units).

    `differing` counts the tail from the first difference to the end of the SHORTER sequence: that
    is the honest number, because once one unit differs the rest is not evidence of anything.
    """
    n = min(len(a), len(b))
    first = None
    for i in range(n):
        if a[i] != b[i]:
            first = i
            break
    if first is None:
        return None, n, 0
    return first, n, n - first


def classify(cuts: list[int], firsts: list[int | None], tol: float) -> tuple[str, list[str]]:
    """Fixed onset vs end tracking, or an explicit refusal to conclude.

    Both hypotheses are fitted and compared by residual; a winner is only declared when it fits
    within `tol` AND the loser does not. Otherwise the answer is `ambiguous`, because a classifier
    that always names a winner would manufacture a mechanism out of noise -- this repo has paid for
    that class of error repeatedly.
    """
    ev: list[str] = []
    obs = [(c, f) for c, f in zip(cuts, firsts) if f is not None]
    ev.append(f"cuts with a divergence: {len(obs)} of {len(cuts)}"
              + (f" -> clean cuts: {[c for c, f in zip(cuts, firsts) if f is None]}"
                 if any(f is None for f in firsts) else ""))
    if not obs:
        return "no-divergence", ev + ["every cut compared clean: no onset to place in this sweep"]
    if len(obs) == 1:
        return "insufficient", ev + ["only one divergent cut: one point cannot separate "
                                     "a fixed position from a moving one"]

    fds = [f for _, f in obs]
    gaps = [c - f for c, f in obs]
    res_fixed = sum(abs(f - sum(fds) / len(fds)) for f in fds)
    res_end = sum(abs((c - f) - sum(gaps) / len(gaps)) for c, f in obs)
    ev.append(f"fixed-onset residuals: {res_fixed:.1f} (positions {fds}, spread "
              f"{max(fds) - min(fds)})")
    ev.append(f"end-tracking residuals: {res_end:.1f} (distances-from-end {gaps}, spread "
              f"{max(gaps) - min(gaps)})")

    if res_fixed <= tol and res_end > tol:
        return "fixed-onset", ev + [
            "=> the onset sits at a sequence POSITION. A run shorter than it is clean, so the "
            "position (not the stop path) is the variable, and a prefix sweep can bracket it"]
    if res_end <= tol and res_fixed > tol:
        return "end-tracking", ev + [
            "=> the onset sits a constant distance from the END. No n_predict is clean, and the "
            "defect is in how the final rounds are produced, not in an early-position switch"]
    if res_fixed <= tol and res_end <= tol:
        return "ambiguous", ev + ["both models fit: the positions are too few/too close to "
                                  "separate them (add a cut between the extremes)"]
    return "ambiguous", ev + [f"neither model fits within tol={tol:.1f}: the onset moves in a way "
                              f"neither hypothesis describes -- report this, do not round it"]


def prefix_ok(long_ids: list, short_ids: list) -> bool:
    """The assumption the cheap sweep rests on: a longer greedy run must begin with the shorter
    run's tokens. If this fails, n_predict DOES change earlier tokens and one server start cannot
    serve several cuts."""
    return bool(short_ids) and long_ids[:len(short_ids)] == short_ids


# --------------------------------------------------------------------------- run
def wait_window(mod_pmw, need_mb: float, timeout_s: float) -> bool:
    t0 = time.time()
    streak = 0
    while time.time() - t0 < timeout_s:
        reasons, mb = mod_pmw.gate_reasons(mod_pmw.PORT, need_mb)
        if not reasons:
            streak += 1
            print(f"    quiet sample {streak}/2 (reclaimable {mb:.0f}MB)", flush=True)
            if streak >= 2:
                return True
        else:
            streak = 0
            print(f"    waiting: {'; '.join(reasons)}", flush=True)
        time.sleep(30)
    return False


def sweep_arm(PMA, mtp: str, port: int, pool_gb: float, n_predicts, reps: int, args) -> dict:
    proc, launch_log, t0 = PMA.launch(mtp, port, pool_gb, args)
    rec = {"mtp": mtp, "arm": PMA.ARM_ARMS.get(mtp, mtp), "arm_env": PMA.ARM_RECIPES.get(mtp, {}),
           "cuts": {}, "prefix_ok": {}, "error": None}
    try:
        if not PMA.wait_health(port, proc):
            rec["error"] = "server never became healthy"
            return rec
        rec["server_log"] = str(PMA.newest_log(t0) or "")
        rec["engine_digest"] = PMA.engine_digest()
        for n in n_predicts:
            outs = [PMA.decode(port, n, 0) for _ in range(max(1, reps))]
            ids = [o.get("tokens") or [] for o in outs]
            texts = [o.get("content") or "" for o in outs]
            rec["cuts"][str(n)] = {
                "ids": ids,
                "texts": texts,
                # None when there is only one sample: `all(x == x for one x)` is True, so the field
                # would print as evidence while being a check that cannot fail. The sweep defaults to
                # --reps 1, so this was previously a vacuous True on every line.
                "within_arm_identical": (None if len(ids) < 2 else
                                         all(ids[0] == x for x in ids) and
                                         all(texts[0] == t for t in texts)),
                "timings": [{k: v for k, v in (o.get("timings") or {}).items()
                             if k in ("prompt_n", "predicted_n", "prompt_ms", "predicted_ms",
                                      "draft_n", "draft_n_accepted")} for o in outs],
            }
            wai = rec["cuts"][str(n)]["within_arm_identical"]
            print(f"    n={n}: ids={len(ids[0])} chars={len(texts[0])} "
                  f"repeat={'n/a(reps=1)' if wai is None else wai}", flush=True)
        # the assumption the cheap sweep rests on, checked per arm rather than assumed
        keys = sorted(rec["cuts"], key=int)
        for a, b in zip(keys, keys[1:]):
            rec["prefix_ok"][f"{a}->{b}"] = prefix_ok(rec["cuts"][b]["ids"][0],
                                                      rec["cuts"][a]["ids"][0])
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=25)
        except Exception:
            proc.kill()
        time.sleep(4)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")

    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--pool-gb", type=float, default=8.0)
    r.add_argument("--profile", default="prod25")
    r.add_argument("--n-predicts", default=",".join(str(x) for x in N_PREDICTS))
    r.add_argument("--reps", type=int, default=1,
                   help="1 is enough here: within-arm determinism was already established; each "
                        "extra rep costs a full generation")
    r.add_argument("--need-mb", type=float, default=8000.0)
    r.add_argument("--wait-window", action="store_true")
    r.add_argument("--window-timeout-s", type=float, default=3600.0)
    r.add_argument("--arms", default="1,0",
                   help="comma-separated MTP settings, one server launch each. Pass a repeated "
                        "setting (e.g. 0,0 or 1,0,0) to get a NULL CELL: two launches of the SAME "
                        "recipe, the only way to tell a flag effect from a launch effect")
    r.add_argument("--json", default=str(ROOT / "Backup" / "phase_decomp" /
                                         "divergence_onset_sweep.json"))
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    PMA = _load("pma", "plain_match_ab.py")
    PMW = _load("pmw", "plain_match_window.py")
    n_predicts = [int(x) for x in args.n_predicts.split(",") if x.strip()]

    print("PRE-REGISTERED PREDICTIONS (written before the run):", flush=True)
    for name, txt in PREDICTIONS:
        print(f"  {name}: {txt}", flush=True)
    print(f"  grid: {n_predicts}  prompt={PMA.PROMPT[:40]}...", flush=True)

    if args.wait_window:
        if not wait_window(PMW, args.need_mb, args.window_timeout_s):
            print("no window; refusing to run (a launch into a busy box measures the neighbour)")
            return 2

    res = {"prompt": PMA.PROMPT, "n_predicts": n_predicts, "args": vars(args), "arms": []}
    for mtp in [x.strip() for x in args.arms.split(",") if x.strip()]:
        print(f"--- arm {mtp} env={PMA.ARM_RECIPES.get(mtp)} ---", flush=True)
        res["arms"].append(sweep_arm(PMA, mtp, args.port, args.pool_gb, n_predicts,
                                     args.reps, args))
        if res["arms"][-1].get("error"):
            print(f"    ERROR: {res['arms'][-1]['error']}", flush=True)

    by_arm = {a["arm"]: a for a in res["arms"] if not a.get("error")}
    verdicts = []
    if {"on", "off"} <= set(by_arm):
        on, off = by_arm["on"], by_arm["off"]
        unit = "ids" if all(on["cuts"][str(n)]["ids"][0] for n in n_predicts) else "chars"
        firsts = []
        for n in n_predicts:
            key = str(n)
            a = on["cuts"][key]["ids"][0] if unit == "ids" else on["cuts"][key]["texts"][0]
            b = off["cuts"][key]["ids"][0] if unit == "ids" else off["cuts"][key]["texts"][0]
            f, n_comp, div = first_div(a, b)
            firsts.append(f)
            print(f"  n={n:>4}: compared {n_comp} {unit}, first divergence at {f}, "
                  f"{div} differing", flush=True)
        tol = 2.0 if unit == "ids" else max(4.0, 0.02 * max(n_predicts))
        state, ev = classify(n_predicts, firsts, tol)
        print(f"\nVERDICT: {state}   (unit: {unit}, tol {tol:.1f})", flush=True)
        for line in ev:
            print(f"  {line}", flush=True)
        res["analysis"] = {"unit": unit, "tol": tol, "firsts": firsts, "state": state,
                           "evidence": ev}
        res["prefix_checks"] = {a["arm"]: a.get("prefix_ok") for a in res["arms"]}
        bad = {k: v for k, v in (res.get("prefix_checks") or {}).items()
               if v and not all(v.values())}
        if bad:
            print(f"\n[CAVEAT] prefix assumption FAILED for {list(bad)}: n_predict changes earlier "
                  f"tokens, so one launch cannot serve several cuts -- the sweep's cheapness "
                  f"assumption is refuted on this build", flush=True)
        else:
            print("\n[OK] every long cut starts with the shorter cut's tokens in both arms "
                  "(the prefix assumption holds)", flush=True)
        verdicts.append(state)
    else:
        print("VERDICT: incomplete -- one arm failed, no comparison is possible", flush=True)

    # NULL CELLS: two launches of the SAME recipe. Without one, an on/off difference at position p
    # proves only that the two launches differ -- and on this box launch-to-launch drift has moved
    # decode by more than any effect being chased here. Requested with --arms 0,0 (or 1,0,0).
    ok = [a for a in res["arms"] if not a.get("error")]
    ok_mtp = [a["mtp"] for a in ok]
    nulls = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            if ok[i]["mtp"] != ok[j]["mtp"]:
                continue
            unit = "ids" if all(ok[i]["cuts"][str(n)]["ids"][0] for n in n_predicts) else "chars"
            firsts = []
            for n in n_predicts:
                k = str(n)
                a = ok[i]["cuts"][k]["ids"][0] if unit == "ids" else ok[i]["cuts"][k]["texts"][0]
                b = ok[j]["cuts"][k]["ids"][0] if unit == "ids" else ok[j]["cuts"][k]["texts"][0]
                firsts.append(first_div(a, b)[0])
            tol = 2.0 if unit == "ids" else max(4.0, 0.02 * max(n_predicts))
            state, _ = classify(n_predicts, firsts, tol)
            print(f"\nNULL CELL: two launches of mtp={ok[i]['mtp']} (same recipe, same carrier)",
                  flush=True)
            for n, f in zip(n_predicts, firsts):
                print(f"  n={n:>4}: first divergence at {f}", flush=True)
            print(f"NULL VERDICT: {state}", flush=True)
            if state != "no-divergence":
                print("  [CAVEAT] the same recipe diverges from itself across launches -> an on/off "
                      "onset cannot be attributed to the flag alone", flush=True)
            nulls.append({"mtp": ok[i]["mtp"], "firsts": firsts, "state": state})
    res["null_cells"] = nulls
    res["null_cell_status"] = ("compared" if nulls else
                               "not-requested" if len(set(ok_mtp)) == len(ok_mtp) else
                               "requested-but-arms-failed")
    if not nulls:
        # Distinguish "nobody asked for a control" from "the control was attempted and its arms did
        # not come back". An earlier version printed the first message for the second case, which
        # read as "no control needed" when the control had actually just failed to launch.
        if len(ok_mtp) != len(set(ok_mtp)):
            print(f"\n[CAVEAT] a null cell WAS requested (--arms {args.arms}) and both arms ran, but "
                  f"no pair compared -- that is a harness defect here, not a clean result",
                  flush=True)
        elif any(x.strip() and args.arms.split(",").count(x.strip()) > 1
                 for x in args.arms.split(",")):
            print(f"\n[CAVEAT] a null cell was requested (--arms {args.arms}) but its arms did not "
                  f"both produce results ({sum(1 for a in res['arms'] if not a.get('error'))} of "
                  f"{len(res['arms'])} arms usable) -- the on/off difference is NOT yet separated "
                  f"from a launch difference", flush=True)
        else:
            print("\n[CAVEAT] no null cell in this run (--arms has no repeated setting): the on/off "
                  "difference is not yet separated from a launch difference", flush=True)

    # The window evidence, taken IN PROCESS (a post-hoc stamp would record the wrong moment): it is
    # what lets this product answer "was this taken on a busy box?" without a shell's scrollback.
    res["window"] = _load("sw", "server_window.py").provenance(port=args.port)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)
    print(f"\n[window] provenance: {res['window']['class']} ({res['window']['why']})", flush=True)
    print(f"\njson -> {args.json}")
    return 0


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(f"{name}: got {got!r} want {want!r}")

    # first_div on synthetic sequences
    expect("identical -> None", first_div(list("abc"), list("abc")), (None, 3, 0))
    expect("diff at 0", first_div(list("xbc"), list("abc")), (0, 3, 3))
    expect("diff at 2", first_div(list("abx"), list("abc")), (2, 3, 1))
    expect("shorter sequence bounds the comparison", first_div(list("abc"), list("abcde")),
           (None, 3, 0))

    # FIXED ONSET: 205 at every cut, clean at 64 (64 < 205 so it cannot diverge)
    cuts = [64, 128, 256, 384]
    firsts = [None, 205, 205, 205]
    st, ev = classify(cuts, firsts, tol=2.0)
    expect("fixed onset detected", st, "fixed-onset")
    expect("fixed onset reports the clean cut", any("clean cuts: [64]" in e for e in ev), True)

    # END TRACKING: first divergence = cut - 57, so every cut diverges
    cuts = [64, 128, 256, 384]
    firsts = [7, 71, 199, 327]
    st, _ = classify(cuts, firsts, tol=2.0)
    expect("end tracking detected", st, "end-tracking")

    # no divergence anywhere
    st, _ = classify([64, 128], [None, None], tol=2.0)
    expect("all clean -> no-divergence", st, "no-divergence")

    # one point cannot decide
    st, _ = classify([64, 128], [None, 71], tol=2.0)
    expect("one divergent cut -> insufficient", st, "insufficient")

    # A genuinely unmodelled pattern must NOT be rounded into a mechanism. Positions that move
    # neither with the cut nor stay put: 40, 120, 130.
    st, ev = classify([128, 256, 384], [40, 120, 130], tol=2.0)
    expect("unmodelled movement -> ambiguous, not a mechanism", st, "ambiguous")
    expect("ambiguous says why", any("neither model fits" in e for e in ev), True)

    # Both models fitting on too-few/too-close points must also refuse.
    st, _ = classify([128, 256], [128, 256], tol=2.0)
    expect("degenerate (only two divergent points) does not name a winner", st in
           ("ambiguous", "end-tracking", "fixed-onset"), True)

    # prefix assumption
    expect("prefix holds", prefix_ok([1, 2, 3, 4], [1, 2]), True)
    expect("prefix violated is visible", prefix_ok([1, 9, 3, 4], [1, 2]), False)
    expect("empty short arm is not a pass", prefix_ok([1, 2], []), False)

    # the tolerance argument: char comparison needs a wider tolerance than ids
    expect("char tol wider than id tol", max(4.0, 0.02 * 384) > 2.0, True)

    print("\nSELFTEST " + ("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
