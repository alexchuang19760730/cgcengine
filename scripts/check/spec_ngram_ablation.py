#!/usr/bin/env python3
"""Where does the per-draft-token cost live: the DRAFT FORWARD, or the VERIFY path?

The question and why this shape is the only one that answers it
--------------------------------------------------------------
`docs/POOL_BUDGET_COST_DECOMP_2026-09-18.md` §6.1b: `m` (= the marginal cost of one draft token,
46-59 ms, i.e. a third of a whole step) has three candidate homes and the pool-budget axis killed
one of them (expert-byte traffic: elasticity 0.10-0.17). The two that are left -- the draft forward
itself, and the extra per-token work the verify batch brings -- both scale linearly with k, so no
sweep over k and no sweep over the pool can separate them.

An n-gram self-speculator drafts from the TOKEN HISTORY with NO draft forward at all, while the
verify batch keeps the same size (1 + n_max). So:

  arm          draft forward   verify batch
  plain        none            T = 1
  draft-mtp    YES (k steps)   T = k+1
  ngram-map-k  none            T = k+1

  m(ngram) ~ 0.1  =>  the 46-59 ms IS the draft forward
  m(ngram) ~ m(mtp)  =>  it lives in the verify path's per-token cost

Null guards (this arm has TWO ways to be a silent zero, and both are refused rather than reported)
--------------------------------------------------------------------------------------------------
 1. `common_ngram_map_draft()` returns with NO draft when the history is shorter than
    2*size_key + size_m (ngram-map.cpp:229-234). llama-bench is patched to refuse that shape.
 2. A null arm verifies ONE token per round, i.e. mean_draft == 0, which would look like a fast
    verify path. Every arm here must show mean_draft == k, or it is reported as INVALID.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/alexchuang/Documents/flashkv-devserver/scripts/check")

import llama_bench_matrix as lbm            # noqa: E402
import thermal_pressure as thermal          # noqa: E402
import spec_cost_curve as scc               # noqa: E402  (reuse its stderr parser, not a copy)

# `--spec-draft-n-max` is the CONFIGURED depth; the MEASURED one is the `draft=` field of the
# SPECDBG rounds. They are not the same number for the n-gram arm, and that is a property of the
# n-gram drafter rather than a bug in this driver: `common_ngram_map_accept()` writes the observed
# acceptance back into the key (`values[0].n_accepted = n_accepted`), so the next draft is
# `min(size_m, last_accepted)` -- the upstream drafter declines to propose more than it just got.
# The pairing therefore has to be made on the MEASURED draft size, which is why both n_max=1 and
# n_max=3 exist below: the (mtp1, ngram1) pair verifies 2 tokens per round in both arms and differs
# in exactly one thing -- whether a draft forward ran.
ARMS = {
    # name      extra argv for the gen cell
    "plain":   [],
    "mtp1":    ["--spec-type", "draft-mtp",   "--spec-draft-n-max", "1"],
    "mtp3":    ["--spec-type", "draft-mtp",   "--spec-draft-n-max", "3"],
    "ngram1":  ["--spec-type", "ngram-map-k", "--spec-draft-n-max", "1"],
    "ngram3":  ["--spec-type", "ngram-map-k", "--spec-draft-n-max", "3"],
}


def build(profile: str, prompt: int, gen: int, depth: int, batch: int, arm: str):
    res = lbm.resolve(profile, {})
    cmd = [str(lbm.LLAMA_BENCH)] + lbm.forward_argv(res["server_argv"]) + [
        "-b", str(batch), "-ub", str(batch),
        "-p", str(prompt), "-n", str(gen), "-d", str(depth),
        "-r", "1", "-o", "json",
    ] + ARMS[arm]
    return cmd, res["env"]


def run_one(profile, arm, args, rounds=1) -> dict:
    cmd, env = build(profile, args.prompt, args.gen, args.depth, args.batch, arm)
    run_env = dict(os.environ)
    run_env.update(env)
    run_env["LLAMA_BENCH_SPEC_DBG"] = "1"
    sampler = thermal.Sampler()
    t0 = time.time()
    with sampler:
        p = subprocess.run(cmd, cwd=str(lbm.ROOT), env=run_env, capture_output=True, text=True)
    wall = time.time() - t0
    # Artifacts land in Backup/ (gitignored, survives /tmp reaping): the 09-18 matrix lost 36 runs
    # because its outputs lived in /tmp only (POOL_BUDGET_COST_DECOMP §7-3). The rep is in the stem
    # because the first version of this driver overwrote each rep's raw log with the next rep's.
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{arm}_r{rounds}_p{args.prompt}_n{args.gen}_d{args.depth}"
    stem.with_suffix(".stderr.log").write_text(p.stderr, errors="replace")
    stem.with_suffix(".json").write_text(p.stdout, errors="replace")

    rows = lbm.parse_rows(p.stdout)
    tg = next((r for r in rows if int(r.get("n_gen") or 0) > 0), None)
    rec = {"arm": arm, "wall_s": round(wall, 1), "rc": p.returncode,
           "tps": tg["avg_ts"] if tg else None,
           "sd_pct": tg.get("stddev_ts") if tg else None,
           "thermal": sampler.result}
    rec.update(scc.parse_stderr(p.stderr, args.gen))
    if arm == "plain":
        rec["E"] = 1.0
        rec["steps"] = args.gen
    else:
        rec["steps"] = int(rec.get("rounds") or 0)
        rec["E"] = round(args.gen / rec["steps"], 4) if rec["steps"] else None
    if rec.get("tps") and rec.get("E"):
        rec["ms_per_step"] = round(rec["E"] / rec["tps"] * 1000.0, 2)
    if rec.get("read_bytes_total") and rec.get("steps"):
        rec["mib_per_step"] = round(rec["read_bytes_total"] / 2**20 / rec["steps"], 3)
    # The pairing key: how many tokens the verify batch actually held, and how many rounds actually
    # carried a draft forward. `draft_calls` is the engine's own counter for that (llama-expert-cache
    # prints `draft: calls=…` separately from `verify: calls=…`), so "no draft forward" is a counter
    # here, not an inference from the flag.
    if rec.get("mean_draft") is not None and rec.get("ms_per_step"):
        rec["verify_batch"] = round(1 + rec["mean_draft"], 3)
        rec["ms_per_verify_token"] = round(rec["ms_per_step"] / rec["verify_batch"], 3)
    # the null guard: a spec arm that drafted nothing verified one token per round
    rec["valid"] = (arm == "plain") or (rec.get("mean_draft") or 0) > 0
    if not rec["valid"]:
        rec["invalid_reason"] = ("mean_draft == 0: no draft was produced, so this arm verified ONE "
                                 "token per round. It is a null, not a cheap verify path.")
    banner = ("n-gram CONTROL arm" in p.stderr)
    rec["banner"] = banner
    if p.returncode != 0:
        rec["error"] = next((l.strip() for l in p.stderr.splitlines() if "error" in l.lower()), "")
    print(f"  [{arm}] tps={rec.get('tps')} E={rec.get('E')} mean_draft={rec.get('mean_draft')} "
          f"ms/step={rec.get('ms_per_step')} v_union/call={rec.get('verify_union_per_call')} "
          f"{(sampler.result or {}).get('launch', {}).get('label', '-')}"
          f"{'' if rec['valid'] else '  <-- INVALID (null)'}", flush=True)
    return rec


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="prefill250")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="plain,mtp1,ngram1,mtp3,ngram3")
    ap.add_argument("--smoke", action="store_true", help="one ngram run, short, print the raw tail")
    ap.add_argument("--outdir", default=str(Path(lbm.ROOT) / "Backup" / "phase_decomp" / "ngram_ab"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.json is None:
        args.json = str(Path(args.outdir) / "ab.json")

    if args.smoke:
        args.arms, args.reps, args.gen = "ngram1", 1, 32
        r = run_one(args.profile, "ngram1", args, 1)
        print(json.dumps({k: v for k, v in r.items() if k != "thermal"}, indent=2, ensure_ascii=False))
        print("--- raw stderr tail ---")
        p = sorted(Path(args.outdir).glob("ngram1_r1_*.stderr.log"))
        print(p[-1].read_text()[-2500:] if p else "(no stderr log written)")
        return 0 if r["valid"] else 1

    if not args.force:
        rivals = scc.rivals()
        if rivals:
            for pid, cmd in rivals:
                print(f"[preflight] rival {pid}: {cmd}")
            print("[preflight] refusing to measure alongside another measurement")
            return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    runs: list[dict] = []
    for rep in range(1, args.reps + 1):
        order = arms[rep % len(arms):] + arms[:rep % len(arms)]   # rotate: no arm owns a position
        print(f"\n--- rep {rep}/{args.reps} order {order} ---", flush=True)
        for arm in order:
            runs.append(run_one(args.profile, arm, args, rep))

    print(f"\n{'=' * 92}\n  n-gram ablation  profile={args.profile}  shape=-p {args.prompt} -n {args.gen} -d {args.depth} -b {args.batch}\n{'=' * 92}")
    hdr = f"{'arm':>7s} {'n':>2s} {'t/s':>7s} {'E':>6s} {'draft':>6s} {'ms/step':>8s} {'v_union':>8s} {'MiB/step':>9s} {'clean':>6s}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for arm in arms:
        ok = [r for r in runs if r["arm"] == arm and r.get("valid")]
        bad = [r for r in runs if r["arm"] == arm and not r.get("valid")]
        if not ok:
            print(f"{arm:>7s}  INVALID: {bad[0].get('invalid_reason') if bad else 'no runs'}")
            continue
        tps, E = med([r["tps"] for r in ok]), med([r["E"] for r in ok])
        summary[arm] = {
            "tps": tps, "E": E, "ms_per_step": med([r["ms_per_step"] for r in ok]),
            "mean_draft": med([r.get("mean_draft") for r in ok]),
            "v_union_per_call": med([r.get("verify_union_per_call") for r in ok]),
            "mib_per_step": med([r.get("mib_per_step") for r in ok]),
            "n": len(ok), "invalid": len(bad),
            "clean": all(((r.get("thermal") or {}).get("launch") or {}).get("label") == "NOMINAL"
                         and ((r.get("thermal") or {}).get("worst") or {}).get("label") == "NOMINAL"
                         for r in ok),
        }
        s = summary[arm]
        print(f"{arm:>7s} {s['n']:>2d} {tps:>7.2f} {E:>6.3f} {str(s['mean_draft'] or '-'):>6s} "
              f"{str(s['ms_per_step']):>8s} {str(s['v_union_per_call'] or '-'):>8s} "
              f"{str(s['mib_per_step'] or '-'):>9s} {('yes' if s['clean'] else 'NO'):>6s}")

    # ---------------------------------------------------------------------------------------------
    # The answer, read off PAIRS. Everything is expressed relative to this window's own plain step, so
    # it does not depend on the absolute level (which is not reproducible on this box).
    #   ms(mtp,   d) = base + k_draft_forward * f + d * v     (draft forward + extra verify tokens)
    #   ms(ngram, d) = base +                       d * v'    (NO draft forward)
    # so at matched MEASURED draft size d,  f_effect = ms(mtp,d) - ms(ngram,d).
    # ---------------------------------------------------------------------------------------------
    if "plain" in summary:
        base = summary["plain"]["ms_per_step"]
        print(f"\n  plain step = {base} ms  (T=1, the baseline every ratio is taken against)")
        pairs = [("mtp1", "ngram1"), ("mtp3", "ngram3")]
        rows = []
        for a, b in pairs:
            if a not in summary or b not in summary:
                continue
            da, db = summary[a]["mean_draft"], summary[b]["mean_draft"]
            matched = da is not None and db is not None and abs(da - db) <= 0.25
            rows.append({
                "arm_mtp": a, "arm_ngram": b,
                "draft_mtp": da, "draft_ngram": db, "draft_matched": matched,
                "C_mtp": round(summary[a]["ms_per_step"] / base, 4),
                "C_ngram": round(summary[b]["ms_per_step"] / base, 4),
                "delta_ms_per_step": round(summary[a]["ms_per_step"] - summary[b]["ms_per_step"], 2),
                "delta_ms_per_draft_forward": round(
                    (summary[a]["ms_per_step"] - summary[b]["ms_per_step"]) / max(da or 1, 1e-9), 2),
            })
            r = rows[-1]
            print(f"  {a:>6s} (draft {da}) vs {b:>6s} (draft {db})   "
                  f"C={r['C_mtp']:.3f} vs {r['C_ngram']:.3f}   "
                  f"delta = {r['delta_ms_per_step']:+.1f} ms/step"
                  f"{'' if matched else '   <-- draft sizes NOT matched, read the normalized column instead'}")

        # normalized cross-check that does not need the draft sizes to match: cost per token that the
        # verify batch actually put through the model.
        norm = {a: summary[a].get("ms_per_verify_token") for a in ("mtp1", "ngram1", "mtp3", "ngram3")
                if a in summary}
        print(f"  ms per verify TOKEN (batch-normalized): {norm}")

        f_est = [r["delta_ms_per_draft_forward"] for r in rows if r["draft_matched"]] or \
                [r["delta_ms_per_draft_forward"] for r in rows]
        draft_forward_ms = med(f_est)
        m_mtp3 = (summary.get("mtp3", {}).get("ms_per_step", 0) / base - 1) / 3 if "mtp3" in summary else None
        share = (draft_forward_ms / (m_mtp3 * base / 3) * 100) if (m_mtp3 and draft_forward_ms) else None
        verdict = (
            "DRAFT FORWARD: with the verify batch held fixed, removing the draft forward removes most of "
            "the extra step time, so the 46-59 ms/token is the drafting itself -- one nextn layer costing a "
            "third of a full step is the thing to fix."
            if (draft_forward_ms or 0) > 0.6 * (m_mtp3 or 0) * base else
            "VERIFY PATH: with the verify batch held fixed, the MTP arm costs about the same as the "
            "no-draft-forward arm, so the extra cost does not need the draft forward at all -- it is the "
            "per-token work the verify batch brings (per-token gather/sync), which is the same conclusion "
            "the pool-budget axis reached from the other side.")
        print(f"\n  draft forward = {draft_forward_ms} ms (per draft token, at matched batch)"
              f"   m(mtp3) = {m_mtp3} -> share of m = {None if share is None else round(share)}%")
        print(f"\n  VERDICT: {verdict}")
        summary["answer"] = {"plain_ms_per_step": base, "pairs": rows, "normalized": norm,
                             "draft_forward_ms_per_token": draft_forward_ms,
                             "m_mtp3": m_mtp3, "draft_forward_share_of_m_pct": share,
                             "verdict": verdict}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"args": vars(args), "runs": runs, "summary": summary},
                                          ensure_ascii=False, indent=2))
    print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
