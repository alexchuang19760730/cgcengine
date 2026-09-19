#!/usr/bin/env python3
"""Measure the speculative-decoding cost curve cost(k) = 1 + m*k on the decode axis.

WHY THIS EXISTS
---------------
The MoE-MTP plan (`docs/MOE_MTP_FEASIBILITY_2026-09-18.md`) puts 2-3x on raising the accept
rate. That is only half the model: the achievable speedup is

    S(k) = E(k) / cost(k),      cost(k) = 1 + m*k      (unit = one plain decode step)

where `k` is the number of draft tokens per round and `E` is the expected number of tokens
committed per round. When `m` is large there is NO amortisation: a verify batch of k+1 tokens
costs as much as k+1 separate steps, so `S <= (k+1)/(k+1) = 1` and no accept rate can rescue it.
`m` is therefore the number that decides whether training a better draft head is worth anything,
and it has never been measured -- the 0.21 in the feasibility doc was back-solved from ONE k
(one point cannot separate `E` from `cost`).

HOW IT IS MEASURED (no two-parameter fit needed)
------------------------------------------------
`llama-bench` in this fork runs the speculative path with `--spec-type draft-mtp
--spec-draft-n-max k`. Two things make `E` and `cost` directly observable instead of fitted:

  * `LLAMA_BENCH_SPEC_DBG=1` prints one `SPECDBG round: n_done=.. draft=..` line per verify
    round (llama-bench.cpp:2631). Counting them gives the number of steps actually taken to
    emit `n_gen` tokens, so `E = n_gen / rounds` and the *effective* draft depth is the mean
    `draft=` (not the requested `--spec-draft-n-max`, which the MTP module may cap).
    The warmup generation run uses the NON-speculative path (`test_gen(ctx, 1, ..)`,
    llama-bench.cpp:2926), so the counted rounds are the timed ones and nothing else.
  * The expert cache prints its MTP fast-path counters at teardown
    (llama-expert-cache.cpp:2529), split into verify vs draft. `verify union / verify calls`
    is the size of the expert union the engine actually fetches for one verify batch -- i.e.
    whether the draft tokens' experts really are batched (server path: 18.69 experts/call) or
    whether the instrument silently fell back to one-token-at-a-time (union/calls == 8.00,
    which is exactly what the 09-17 note reports for the bench side).

Then `cost(k) = E(k) / S(k)` with `S(k) = tps(k) / tps(k=0)`, both measured, and
`m = (cost - 1) / k_effective`.

DESIGN (drift, not explanation)
-------------------------------
Single-arm decode noise on this machine is about +/-1.9 t/s, larger than several of the effects
being chased, and no recorded variable (thermal, memory, engine version) orders the arms -- so
this does NOT try to explain the drift, it cancels it:

  * one process per k (llama-bench resolves `--spec-draft-n-max` per invocation, so a single
    call cannot sweep k), and the ks are run in an interleaved ABBA order per round,
  * every ratio is taken **within a round** against that round's own k=0 baseline,
  * the reported figure is the **median over rounds** of those within-round ratios.

Reading rules that apply to every number this prints:
  * llama-bench, not HTTP, and both axes are recorded (the pp row comes free in the same
    invocation and is a control: it must NOT move with k, because k only touches generation).
  * A number is citable only if the thermal sampler recorded NOMINAL at launch and worst.

USAGE
-----
    # the curve, 3 paired rounds, decode axis (-n 128) at depth 0
    python3 scripts/check/spec_cost_curve.py --profile prefill250 --ks 0,1,2,3,5,7 --rounds 3 \
        --json Backup/phase_decomp/spec_cost_curve_<tag>.json

    # see the exact command for one k without running it
    python3 scripts/check/spec_cost_curve.py --ks 3 --rounds 1 --dry-run

    # arithmetic + parsing self-test, no GPU
    python3 scripts/check/spec_cost_curve.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llama_bench_matrix as lbm  # noqa: E402  (single source of truth for env/argv/batch)
import thermal_pressure as thermal  # noqa: E402

ROUND_RE = re.compile(r"SPECDBG round: n_done=(\d+) n_past=(\d+) draft=(\d+)")
VERIFY_RE = re.compile(r"verify: calls=(\d+) union=(\d+) cold=(\d+)")
DRAFT_RE = re.compile(r"draft: calls=(\d+) union=(\d+) cold=(\d+)")
# Cache teardown counters (printed unconditionally at exit by llama-expert-cache.cpp).
# These are the ONLY place the residency story lives: how many bytes this run re-read,
# how often the pool answered, and how many layers blew through their slot quota.
POOL_READ_RE = re.compile(r"read shape: jobs=(\d+) bytes=(\d+)")
POOL_HIT_RE = re.compile(r"decode/pool \(ensure_slot\+batch\) hits=(\d+)/(\d+) \(([0-9.]+)%\)")
MISS_ATTR_RE = re.compile(r"miss attribution: compulsory=(\d+) capacity=(\d+)")
LAYER_OVER_RE = re.compile(r"layers_distinct_over_slots=(\d+)\s+worst=layer (\d+) distinct=(\d+) slots=(\d+)")

# Anything that looks like somebody else's measurement on this machine. Deliberately does NOT
# match on `llama` alone: the 09-18 competitor was `http_duo.py`, whose argv contains no
# `llama` substring at all, so a name-based check gave a false all-clear.
RIVAL_PATTERNS = (
    r"llama-bench",
    r"llama-server",
    r"run_server\.sh",
    r"http_duo\.py",
    r"profile_duo\.py",
    r"decode_sweep\.py",
    r"llama_bench_matrix\.py",
    r"prod_matrix\.py",
)


# ---------------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------------
def rivals() -> list[tuple[int, str]]:
    """Other people's measurement processes. Our own line is filtered by OUR script name."""
    out: list[tuple[int, str]] = []
    for pat in RIVAL_PATTERNS:
        p = subprocess.run(["pgrep", "-fl", pat], capture_output=True, text=True)
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_s, _, cmd = line.partition(" ")
            if "spec_cost_curve" in cmd:
                continue  # ourselves (our own argv names this file)
            try:
                out.append((int(pid_s), cmd[:110]))
            except ValueError:
                continue
    # de-dup by pid, keep order
    seen: set[int] = set()
    uniq = [(pid, cmd) for pid, cmd in out if not (pid in seen or seen.add(pid))]
    return uniq


def preflight(force: bool) -> None:
    r = rivals()
    if r and not force:
        sys.stderr.write(
            "[preflight] other measurement processes are running; a cost curve measured "
            "alongside them is not attributable:\n")
        for pid, cmd in r:
            sys.stderr.write(f"  [preflight]   {pid}  {cmd}\n")
        sys.stderr.write("[preflight] aborting. Re-run when the machine is free, or pass "
                         "--force to accept the contamination.\n")
        raise SystemExit(2)
    if r:
        sys.stderr.write(f"[preflight] WARNING: {len(r)} rival process(es) present, --force given\n")


# ---------------------------------------------------------------------------------------------
# one measurement
# ---------------------------------------------------------------------------------------------
def build_cmd(k: int, args, extra_env: dict[str, str]):
    res = lbm.resolve(args.profile, extra_env)
    env, argv, scalars = res["env"], res["server_argv"], res["scalars"]
    fwd = lbm.forward_argv(argv)
    if args.batch and args.ubatch:
        b, ub, why = args.batch, args.ubatch, "cli"
    else:
        b, ub, why = lbm.default_batch(env, scalars)
    cmd = [str(lbm.LLAMA_BENCH)] + fwd + [
        "-b", str(b), "-ub", str(ub),
        "-p", str(args.prompt), "-n", str(args.gen), "-d", str(args.depth),
        "-r", "1", "-o", "json",
    ]
    # `-r 1` on purpose: the SPECDBG round lines are not tagged per rep, so one rep per process
    # is what makes `rounds` attributable to a single measured generation.
    if k > 0:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)]
    return cmd, env, scalars, (b, ub, why)


def parse_stderr(text: str, n_gen: int) -> dict:
    rounds = ROUND_RE.findall(text)
    n_rounds = len(rounds)
    drafts = [int(d) for _, _, d in rounds]
    out = {
        "rounds": n_rounds,
        "mean_draft": round(statistics.fmean(drafts), 3) if drafts else None,
        "max_draft": max(drafts) if drafts else None,
        "last_n_done": int(rounds[-1][0]) if rounds else None,
        "n_gen": n_gen,
    }
    # E = tokens committed per verify round. Only meaningful when the round count is real.
    out["E"] = round(n_gen / n_rounds, 4) if n_rounds else None
    m = VERIFY_RE.search(text)
    if m:
        calls, union, cold = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out["verify_calls"] = calls
        out["verify_union_per_call"] = round(union / calls, 3) if calls else None
        out["verify_cold_pct"] = round(100.0 * cold / union, 2) if union else None
    d = DRAFT_RE.search(text)
    if d:
        calls, union = int(d.group(1)), int(d.group(2))
        out["draft_calls"] = calls
        out["draft_union_per_call"] = round(union / calls, 3) if calls else None
    rd = POOL_READ_RE.search(text)
    if rd:
        out["read_jobs"] = int(rd.group(1))
        out["read_bytes_total"] = int(rd.group(2))
    h = POOL_HIT_RE.search(text)
    if h:
        out["pool_hits"] = int(h.group(1))
        out["pool_lookups"] = int(h.group(2))
        out["pool_hit_pct"] = float(h.group(3))
    ma = MISS_ATTR_RE.search(text)
    if ma:
        out["miss_compulsory"] = int(ma.group(1))
        out["miss_capacity"] = int(ma.group(2))
    lo = LAYER_OVER_RE.search(text)
    if lo:
        out["layers_over_slots"] = int(lo.group(1))
        out["worst_layer_distinct"] = int(lo.group(3))
        out["per_layer_slots"] = int(lo.group(4))  # the quota itself, per layer
    return out


def run_one(k: int, rnd: int, args, extra_env: dict[str, str],
            budget_label: str | None = None, budget_bytes: int | None = None) -> dict:
    env_extra = dict(extra_env)
    if budget_bytes is not None:
        # The profile pins BUDGET itself (`[ -z "${CGC_SERVER_EXPERT_CACHE_BYTES+x}" ]`
        # in run_server.sh), so the ONLY way to actually move the pool is to set this
        # BEFORE the resolve -- it has to be folded into the CGC_DUMP_ENV call.
        env_extra["CGC_SERVER_EXPERT_CACHE_BYTES"] = str(budget_bytes)
    cmd, env, scalars, (b, ub, why) = build_cmd(k, args, env_extra)
    tag = f"k{k}" if budget_label is None else f"{budget_label}k{k}"
    print(f"  [{tag} r{rnd}] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {"k": k, "round": rnd, "cmd": cmd, "dry_run": True}

    run_env = dict(os.environ)
    run_env.update(env)
    run_env["LLAMA_BENCH_SPEC_DBG"] = "1"

    sampler = thermal.Sampler()
    t0 = time.time()
    with sampler:
        proc = subprocess.run(cmd, cwd=str(lbm.ROOT), env=run_env,
                              capture_output=True, text=True)
    wall = time.time() - t0

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = workdir / f"spec_cost_{tag}_r{rnd}_p{args.prompt}_n{args.gen}_d{args.depth}"
    # stderr is the ONLY place the round lines and the cache counters land; stdout is the JSON.
    stem.with_suffix(".stderr.log").write_text(proc.stderr, errors="replace")
    stem.with_suffix(".json").write_text(proc.stdout, errors="replace")

    rows = lbm.parse_rows(proc.stdout)
    tg = next((r for r in rows if int(r.get("n_gen") or 0) > 0), None)
    pp = next((r for r in rows if int(r.get("n_gen") or 0) == 0), None)

    rec = {
        "k": k, "round": rnd, "wall_s": round(wall, 1),
        "batch": b, "ubatch": ub, "batch_why": why,
        "tps_tg": tg["avg_ts"] if tg else None,
        "sd_tg": tg["stddev_ts"] if tg else None,
        "tps_pp": pp["avg_ts"] if pp else None,
        "thermal": sampler.result,
        "rc": proc.returncode,
    }
    rec.update(parse_stderr(proc.stderr, int(args.gen)))
    rec["budget_label"] = budget_label
    rec["budget_bytes"] = budget_bytes
    # k=0 emits no SPECDBG lines at all (there is no verify round to report), so E is 1 by
    # construction and a step is one generated token. Everything else reads its step count
    # off the round lines -- rounds==0 there is a MISSING measurement, not E=1.
    if k == 0:
        rec["E"] = 1.0
        rec["steps"] = int(args.gen)
        rec["E_source"] = "plain-gen (no verify round; 1 token per step)"
    else:
        rec["steps"] = int(rec.get("rounds") or 0)
        rec["E_source"] = "SPECDBG rounds" if rec.get("rounds") else "MISSING"
    if rec.get("read_bytes_total") and rec.get("steps"):
        rec["read_mib_per_step"] = round(rec["read_bytes_total"] / 2**20 / rec["steps"], 3)
    if rec.get("tps_tg") and rec.get("E"):
        rec["ms_per_step"] = round(rec["E"] / rec["tps_tg"] * 1000.0, 2)
    # rc < 0 = killed by a signal. On this machine that is USUALLY another session's cleanup
    # (observed 2026-09-18 20:17: the plain-gen baseline died rc=-9 mid-generation while a
    # neighbouring session was clearing `llama-server|llama-bench` before its own D5 run).
    # A killed run is not a slow run: its JSON is truncated and its round count is short, so
    # folding it into a median would silently bias the curve. Marked, retried, and if every
    # attempt dies, excluded.
    rec["signal_killed"] = proc.returncode < 0
    if proc.returncode != 0:
        sigs = ("failed to decode", "res = -", "error:", "GGML_ASSERT", "abort", "SIGSEGV",
                "SIGABRT", "out of memory", "Unable to")
        lines = [l.strip() for l in proc.stderr.splitlines() if l.strip()]
        rec["error"] = next((l for l in lines if any(s in l for s in sigs)),
                            lines[-1] if lines else "")
        print(f"  !! [{tag} r{rnd}] rc={proc.returncode}: {rec['error']}", flush=True)
    print(f"  [{tag} r{rnd}] tg={rec['tps_tg']} pp={rec['tps_pp']} "
          f"rounds={rec['rounds']} E={rec['E']} mean_draft={rec['mean_draft']} "
          f"v_union/call={rec.get('verify_union_per_call')} "
          f"{(sampler.result or {}).get('launch', {}).get('label', '-')}", flush=True)
    return rec


# ---------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------
def solve_a(E: float, k: float) -> float | None:
    """Per-draft-token accept probability from E = 1 + a + a^2 + ... + a^k."""
    if k <= 0 or E <= 1.0:
        return None
    if E >= k + 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        s = (1.0 - mid ** (k + 1)) / (1.0 - mid) if mid < 1.0 else k + 1
        if s < E:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def fit_m(points: list[tuple[float, float]]) -> dict:
    """Least squares through the origin on (k_eff, cost-1) => m."""
    if not points:
        return {}
    sxy = sum(x * y for x, y in points)
    sxx = sum(x * x for x, y in points)
    m = sxy / sxx if sxx else None
    ys = [y for _, y in points]
    ybar = statistics.fmean(ys)
    ss_res = sum((y - (m * x if m else 0.0)) ** 2 for x, y in points)
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return {"m_fit": round(m, 4) if m is not None else None,
            "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot else None,
            "n_points": len(points)}


def analyse(runs: list[dict], ks: list[int], args, budgets: list[tuple[str, int]] | None = None) -> dict:
    killed = [r for r in runs if r.get("signal_killed")]
    labels = [lbl for lbl, _ in (budgets or [])] or [""]
    per_budget: dict[str, dict] = {}
    for lbl in labels:
        blk = _stats_for_label(runs, ks, lbl)
        if blk and blk["per_k"]:
            per_budget[lbl] = blk
    # Back-compat: with no budget axis everything still collapses onto the flat fields,
    # and because these are the SAME dict objects the per-k loop below fills in,
    # the budget view stays in sync.
    lead = labels[0] if labels[0] in per_budget else next(iter(per_budget), None)
    _blk = per_budget.get(lead) if lead else None
    per_k = _blk["per_k"] if _blk else {}
    base = _blk.get("baseline") if _blk else None
    paired = _blk["paired"] if _blk else {}


    points: list[tuple[float, float]] = []
    for k in ks:
        if k == 0 or k not in per_k or not base:
            continue
        pk = per_k[k]
        Ss = paired.get(k) or []
        if not Ss:
            continue
        S = statistics.median(Ss)
        pk["S_paired_median"] = round(S, 4)
        pk["S_min"] = round(min(Ss), 4)
        pk["S_max"] = round(max(Ss), 4)
        pk["accept_a"] = solve_a(pk["E"], pk["mean_draft"]) if pk["E"] else None
        if pk["E"] and S > 0:
            cost = pk["E"] / S
            pk["cost"] = round(cost, 4)
            k_eff = pk["mean_draft"] or k
            pk["m"] = round((cost - 1.0) / k_eff, 4)
            points.append((k_eff, cost - 1.0))

    fit = fit_m(points)
    out = {"per_k": per_k, "fit": fit, "killed": killed}
    if base:
        out["baseline_tps"] = base["tps_tg"]
    if (budgets or []) and len(labels) > 1:
        out["per_budget"] = {lbl: {"per_k": b["per_k"], "fit": b["fit"],
                                   "baseline_tps": (b.get("baseline") or {}).get("tps_tg"),
                                   "per_layer_slots": b.get("per_layer_slots")}
                             for lbl, b in per_budget.items()}
        out["bytes_regression"] = bytes_regression(runs)

    # The verdict. `m` is what decides whether a better draft head can pay for itself.
    m = fit.get("m_fit")
    if m is None:
        out["verdict"] = "NO DATA"
    elif m >= 0.6:
        out["verdict"] = ("m >= 0.6: verify barely amortises. Raising the accept rate cannot "
                          "deliver 2x -- S <= (k+1)/(1+m*k) stays near 1. Training a draft head "
                          "is not the lever here; the batch path is.")
    elif m <= 0.25:
        out["verdict"] = ("m <= 0.25: verify amortises well. The accept rate IS the lever, so "
                          "the plan's training route is worth its cost.")
    else:
        out["verdict"] = ("m in (0.25, 0.6): both levers matter. Accept rate alone tops out "
                          "around 2x; the rest has to come from pushing m down.")
    return out


# ---------------------------------------------------------------------------------------------
# budget axis helpers
# ---------------------------------------------------------------------------------------------
def _budget_label(tok: str) -> str:
    """GiB token -> short tag used in log/experiment names (8 -> p8g, 6.5 -> p6p5g)."""
    return "p" + str(tok).strip().replace(".", "p") + "g"


def _med(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def _stats_for_label(runs: list[dict], ks: list[int], label: str) -> dict:
    """All paired statistics for one pool budget. Mirrors what analyse() used to do inline."""
    by_round: dict[int, dict[int, dict]] = {}
    for r in runs:
        if r.get("signal_killed"):
            continue  # truncated JSON, short round count
        if (r.get("budget_label") or "") != label:
            continue
        by_round.setdefault(r["round"], {})[r["k"]] = r

    per_k: dict[int, dict] = {}
    for k in ks:
        recs = [by_round[rd][k] for rd in sorted(by_round) if k in by_round[rd]]
        ok = [r for r in recs if r.get("tps_tg")]
        if not ok:
            continue
        # Baseline E is 1 by construction now (set in run_one), so it can be averaged safely.
        per_k[k] = {
            "n": len(ok),
            "tps_tg": round(statistics.median([r["tps_tg"] for r in ok]), 3),
            "tps_pp": round(statistics.median([r["tps_pp"] for r in ok]), 3) if ok[0].get("tps_pp") else None,
            "E": round(statistics.median([r["E"] for r in ok if r.get("E")]), 4) if any(r.get("E") for r in ok) else None,
            "mean_draft": round(statistics.fmean([r["mean_draft"] for r in ok if r.get("mean_draft")]), 3)
                          if any(r.get("mean_draft") for r in ok) else None,
            "verify_union_per_call": round(statistics.fmean(
                [r["verify_union_per_call"] for r in ok if r.get("verify_union_per_call")]), 3)
                if any(r.get("verify_union_per_call") for r in ok) else None,
            # residency read-outs -- these are what the budget axis is FOR
            "read_mib_per_step": _med([r.get("read_mib_per_step") for r in ok]),
            "ms_per_step": _med([r.get("ms_per_step") for r in ok]),
            "pool_hit_pct": _med([r.get("pool_hit_pct") for r in ok]),
            "layers_over_slots": _med([r.get("layers_over_slots") for r in ok]),
            "worst_layer_distinct": _med([r.get("worst_layer_distinct") for r in ok]),
            "clean": all(
                ((r.get("thermal") or {}).get("launch", {}) or {}).get("label") == "NOMINAL"
                and ((r.get("thermal") or {}).get("worst", {}) or {}).get("label") == "NOMINAL"
                for r in ok),
        }
    per_layer_slots = next((r.get("per_layer_slots") for r in runs
                            if (r.get("budget_label") or "") == label and r.get("per_layer_slots")), None)

    base = per_k.get(0)
    paired: dict[int, list[float]] = {}
    for rd in sorted(by_round):
        b = by_round[rd].get(0)
        if not b or not b.get("tps_tg"):
            continue
        for k in ks:
            if k == 0:
                continue
            r = by_round[rd].get(k)
            if not r or not r.get("tps_tg") or not r.get("E"):
                continue
            paired.setdefault(k, []).append(r["tps_tg"] / b["tps_tg"])
    return {"per_k": per_k, "paired": paired, "baseline": base,
            "per_layer_slots": per_layer_slots}


def bytes_regression(runs: list[dict]) -> dict:
    """ms/step vs MiB/step across BOTH budgets and all k.

    With one budget these two are collinear with k, so the slope is uninterpretable. Two budgets
    put differing working-set sizes at the same k, which is what makes the slope attributable.
    Residual-by-k is the other half of the answer: if the residual grows with k at equal bytes,
    then something OTHER than re-read traffic scales with the draft depth.
    """
    pts = [(r["read_mib_per_step"], r["ms_per_step"], r["k"], (r.get("budget_label") or ""))
           for r in runs
           if not r.get("signal_killed") and not r.get("dry_run")
           and r.get("read_mib_per_step") and r.get("ms_per_step")]
    if len(pts) < 4:
        return {"n_points": len(pts), "note": "not enough points to fit (<4)"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None
    ic = my - (slope or 0.0) * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (ic + slope * x)) ** 2 for x, y in zip(xs, ys)) if slope else ss_tot
    resid_k: dict[int, float] = {}
    for x, y, k, _lbl in pts:
        resid_k.setdefault(k, []).append(y - (ic + slope * x))
    resid_b: dict[str, float] = {}
    for x, y, _k, lbl in pts:
        resid_b.setdefault(lbl, []).append(y - (ic + slope * x))
    return {
        "n_points": n,
        "slope_ms_per_mib": round(slope, 3) if slope else None,
        "intercept_ms": round(ic, 1),
        "r2": round(1.0 - ss_res / ss_tot, 3) if ss_tot else None,
        "implied_rate_mib_s": round(1000.0 / slope, 0) if slope else None,
        "mean_resid_ms_by_k": {str(k): round(statistics.fmean(v), 1) for k, v in sorted(resid_k.items())},
        "mean_resid_ms_by_budget": {k: round(statistics.fmean(v), 1) for k, v in sorted(resid_b.items())},
        "bytes_range_mib": [round(min(xs), 1), round(max(xs), 1)],
        "k_collinear_note": "if every k appears at only one byte level this fit is still confounded",
    }


def report_budget_axis(runs: list[dict], analysis: dict, args) -> None:
    pb = analysis.get("per_budget") or {}
    if not pb:
        return
    print(f"\n{'=' * 100}\n  budget axis (expert pool)\n{'=' * 100}")
    hdr = (f"{'pool':>6s} {'slots':>6s} {'k':>3s} {'n':>2s} {'E':>6s} {'S':>6s} {'cost':>6s} "
           f"{'m':>7s} {'MiB/step':>9s} {'ms/step':>8s} {'hit%':>6s} {'over':>5s} {'worst':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for lbl in sorted(pb):
        blk = pb[lbl]
        slots = blk.get("per_layer_slots")
        for k in sorted(blk["per_k"]):
            p = blk["per_k"][k]
            print(f"{lbl:>6s} {str(slots or '-'):>6s} {k:>3d} {p['n']:>2d} "
                  f"{str(p.get('E') if p.get('E') is not None else '-'):>6s} "
                  f"{('%6.3f' % p['S_paired_median']) if p.get('S_paired_median') else '     -':>6s} "
                  f"{str(p.get('cost') or '-'):>6s} {str(p.get('m') or '-'):>7s} "
                  f"{str(p.get('read_mib_per_step') or '-'):>9s} "
                  f"{str(p.get('ms_per_step') or '-'):>8s} "
                  f"{str(p.get('pool_hit_pct') or '-'):>6s} "
                  f"{str(p.get('layers_over_slots') or '-'):>5s} "
                  f"{str(p.get('worst_layer_distinct') or '-'):>6s}")
        f = blk.get("fit") or {}
        print(f"       -> baseline t/s {blk.get('baseline_tps')}   m_fit {f.get('m_fit')} "
              f"r2 {f.get('r2')}")
    br = analysis.get("bytes_regression") or {}
    if br.get("n_points", 0) >= 4:
        print("\n  ms/step vs MiB/step across all pools and k:")
        print(f"    n = {br['n_points']}   slope = {br['slope_ms_per_mib']} ms/MiB   "
              f"intercept = {br['intercept_ms']} ms   r2 = {br['r2']}")
        print(f"    implied read rate = {br['implied_rate_mib_s']} MiB/s   "
              f"bytes range = {br['bytes_range_mib']} MiB/step")
        print(f"    mean residual by k      = {br['mean_resid_ms_by_k']}")
        print(f"    mean residual by budget = {br['mean_resid_ms_by_budget']}")
        print("    read: a residual that grows with k at EQUAL bytes means some of m is not")
        print("    re-read traffic (it is the k draft forwards / per-step sync).")
    else:
        print(f"\n  bytes regression skipped: {br.get('note') or br}")

def report(runs: list[dict], analysis: dict, args) -> None:
    print(f"\n{'=' * 100}\n  spec cost curve -- {args.profile}  "
          f"p{args.prompt}/n{args.gen}/d{args.depth}  rounds={args.rounds}\n{'=' * 100}")
    hdr = (f"{'k':>3s} {'n':>2s} {'draft':>6s} {'E':>6s} {'a':>6s} {'t/s tg':>8s} {'t/s pp':>8s} "
           f"{'S':>6s} {'S range':>14s} {'cost':>6s} {'m':>7s} {'v_union':>8s} {'clean':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(analysis["per_k"]):
        p = analysis["per_k"][k]
        rng = f"{p['S_min']:.3f}..{p['S_max']:.3f}" if p.get("S_min") is not None else "-"
        print(f"{k:>3d} {p['n']:>2d} {str(p['mean_draft'] or '-'):>6s} "
              f"{str(p['E'] if p['E'] is not None else '-'):>6s} "
              f"{str(p.get('accept_a') or '-'):>6s} {p['tps_tg']:>8.2f} "
              f"{('%8.2f' % p['tps_pp']) if p['tps_pp'] else '       -':>8s} "
              f"{('%6.3f' % p['S_paired_median']) if p.get('S_paired_median') else '     -':>6s} "
              f"{rng:>14s} {str(p.get('cost') or '-'):>6s} {str(p.get('m') or '-'):>7s} "
              f"{str(p.get('verify_union_per_call') or '-'):>8s} "
              f"{('yes' if p.get('thermal_clean', p.get('clean')) else 'NO'):>6s}")
    f = analysis.get("fit") or {}
    if analysis.get("killed"):
        print(f"\n  !! {len(analysis['killed'])} run(s) were killed by a signal and were EXCLUDED "
              f"(truncated JSON / short round count):")
        for r in analysis["killed"]:
            print(f"  !!   k={r['k']} round={r['round']} rc={r['rc']} "
                  f"{(r.get('error') or '')[:80]}")
    print(f"\n  baseline (k=0) t/s: {analysis.get('baseline_tps')}")
    print(f"  fit cost-1 = m * k_eff :  m = {f.get('m_fit')}  r2 = {f.get('r2')}  "
          f"n = {f.get('n_points')}")
    print(f"  VERDICT: {analysis.get('verdict')}")
    print("\n  k=0 row is the plain gen cell (no --spec-type): S and cost are 1 by construction,")
    print("  and its pp row is the control -- k only touches generation, so pp must not move.")


# ---------------------------------------------------------------------------------------------
# self-test (no GPU)
# ---------------------------------------------------------------------------------------------
def self_test() -> int:
    checks: list[tuple[str, bool]] = []

    def chk(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    # 1. round parsing + E
    err = ("SPECDBG round: n_done=1 n_past=0 draft=3\n"
           "SPECDBG round: n_done=5 n_past=4 draft=3\n"
           "SPECDBG round: n_done=8 n_past=7 draft=3\n")
    p = parse_stderr(err, 128)
    chk("rounds counted", p["rounds"] == 3)
    chk("E = n_gen/rounds", abs(p["E"] - round(128 / 3, 4)) < 1e-9)  # E is stored at 4 dp
    chk("mean_draft", p["mean_draft"] == 3.0)

    # 2. warmup must not be counted: the warmup gen is non-spec, so a run with no round lines
    #    must report rounds == 0 rather than silently defaulting E to 1.
    chk("no round lines -> rounds 0, E None", parse_stderr("nothing here\n", 128)["E"] is None)

    # 3. verify/draft union counters
    err2 = ("llama_expert_cache: MTP fast path: calls=900 union=9900 cold(ZERO)=10 (0.1%)   "
            "verify: calls=400 union=7476 cold=8 (0.1%)   draft: calls=500 union=4000 cold=2 (0.1%)\n")
    q = parse_stderr(err2, 128)
    chk("verify union/call", abs(q["verify_union_per_call"] - 18.69) < 0.01)
    chk("draft union/call", abs(q["draft_union_per_call"] - 8.0) < 1e-6)

    # 4. accept probability inversion: a=0.6, k=3 -> E = 1+0.6+0.36+0.216 = 2.176
    a = solve_a(2.176, 3.0)
    chk("solve_a round trip", a is not None and abs(a - 0.6) < 0.002)
    chk("solve_a clamps at E = k+1", solve_a(4.0, 3.0) == 1.0)
    chk("solve_a: E<=1 is not an accept-rate question", solve_a(1.0, 3.0) is None)

    # 5. the amortisation theorem: m=1 (no amortisation) => S <= 1 for ANY accept rate
    worst = max((k + 1) / (1 + 1.0 * k) for k in range(1, 16))  # E is at most k+1
    chk("m=1 caps speedup at 1.0", abs(worst - 1.0) < 1e-9)

    # 6. fit through the origin
    fit = fit_m([(1.0, 0.2), (2.0, 0.4), (3.0, 0.6), (5.0, 1.0)])
    chk("m_fit on exact line", abs(fit["m_fit"] - 0.2) < 1e-6)
    chk("r2 == 1 on exact line", fit["r2"] is not None and abs(fit["r2"] - 1.0) < 1e-6)

    # 7. cost/m arithmetic end to end: E=2.12, S=1.31, k=3 -> cost=1.618, m=0.206
    cost = 2.12 / 1.31
    chk("cost(3) matches the doc back-solve", abs(cost - 1.6183) < 0.001)
    chk("m from that point", abs((cost - 1) / 3 - 0.2061) < 0.001)

    # 8. rival filter must exclude our own argv
    chk("rival filter skips self", "spec_cost_curve" in __file__)

    # 9. cache teardown counters (the residency read-out the budget axis depends on)
    err3 = ("llama_expert_cache: miss attribution: compulsory=5601 capacity=873 (86.5% / 13.5% of 6474)  evictions=6474  layers_distinct_over_slots=15  worst=layer 0 distinct=211 slots=143\n"
            "llama_expert_cache: read shape: jobs=46347 bytes=4305625088 (0.09 MiB/job as one contiguous run)\n"
            "llama_expert_cache: decode/pool (ensure_slot+batch) hits=6156/12630 (48.7%)  gather (ensure) hits=0/0\n")
    q3 = parse_stderr(err3, 128)
    chk("pool read bytes", q3.get("read_bytes_total") == 4305625088)
    chk("pool hit pct", abs((q3.get("pool_hit_pct") or 0) - 48.7) < 1e-6)
    chk("miss compulsory", q3.get("miss_compulsory") == 5601)
    chk("layers over slots", q3.get("layers_over_slots") == 15)
    chk("worst layer distinct", q3.get("worst_layer_distinct") == 211)
    chk("per-layer slots", q3.get("per_layer_slots") == 143)
    chk("no round lines -> no derived per-step", "E" in q3 and q3["rounds"] == 0)

    # 10. budget axis plumbing
    chk("budget label tokens", _budget_label("8") == "p8g" and _budget_label("6.5") == "p6p5g")
    flat = bytes_regression([{"read_mib_per_step": 10.0, "ms_per_step": 100.0, "k": 0}])
    chk("too few points -> skip regression", flat.get("note") is not None)
    line = [{"read_mib_per_step": float(x), "ms_per_step": 50.0 + 2.0 * x, "k": x,
             "budget_label": "p8g" if x % 2 else "p6g"} for x in (10, 20, 30, 40)]
    fit2 = bytes_regression(line)
    chk("regression recovers the slope", abs((fit2["slope_ms_per_mib"] or 0) - 2.0) < 1e-6)
    chk("regression recovers r2=1", abs((fit2["r2"] or 0) - 1.0) < 1e-6)

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in checks if not ok]
    print(f"\nself-test: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="prefill250",
                    help="run_server.sh profile the env/argv are resolved from")
    ap.add_argument("--extra", default="", help="extra env as K=V;K=V folded into the resolve")
    ap.add_argument("--ks", default="0,1,2,3,5,7",
                    help="draft depths. 0 = plain gen cell (the paired baseline, MANDATORY)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prompt", default="512")
    ap.add_argument("--gen", default=128, type=int)
    ap.add_argument("--depth", default="0")
    ap.add_argument("--batch")
    ap.add_argument("--ubatch")
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--json")
    ap.add_argument("--retry", type=int, default=2,
                    help="re-attempt a run that was killed by a SIGNAL (default 2). Only signal "
                         "deaths are retried; a real failure (assert/OOM) is a result.")
    ap.add_argument("--budgets", default="",
                    help="expert pool budgets in GiB, comma separated (e.g. 8,6). Each one becomes a\nsecond axis: with a single budget MiB/step and k are collinear, so the \nms/step-vs-bytes slope is not attributable. Two budgets fix that.")
    ap.add_argument("--force", action="store_true", help="run even if rivals are present")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.batch and not args.ubatch:
        args.ubatch = args.batch

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if 0 not in ks:
        raise SystemExit("--ks must include 0: every ratio is taken against that round's own "
                         "plain-gen baseline, so a curve without it cannot be paired.")
    extra_env = dict(kv.split("=", 1) for kv in args.extra.split(";") if "=" in kv)
    budgets: list[tuple[str, int]] = []
    for tok in args.budgets.split(","):
        tok = tok.strip()
        if not tok:
            continue
        budgets.append((_budget_label(tok), int(float(tok) * 1024**3)))

    if not args.dry_run:
        preflight(args.force)

    runs: list[dict] = []
    for rnd in range(1, args.rounds + 1):
        # ABBA: reverse on odd rounds so a monotone drift hits the ks in the opposite order.
        order = ks if rnd % 2 == 1 else list(reversed(ks))
        print(f"\n--- round {rnd}/{args.rounds} (order {order}) ---", flush=True)
        # k is the outer loop and budget the inner one: that keeps the two budgets for the
        # SAME k adjacent, so their difference is not a drift difference.
        plan = [(k, lbl, by) for k in order for (lbl, by) in budgets] \
            if budgets else [(k, None, None) for k in order]
        for k, lbl, by in plan:
            # Retry on signal death only. A crash with a real diagnostic (assert/OOM) is a
            # RESULT about that k and must not be retried away -- retrying would quietly turn
            # "k=7 does not fit" into "k=7 is fine on the second try".
            for attempt in range(1, args.retry + 2):
                rec = run_one(k, rnd, args, extra_env, lbl, by)
                if not rec.get("signal_killed"):
                    break
                print(f"  !! [{rec['k']} r{rnd}] killed by signal (rc={rec['rc']}); "
                      f"attempt {attempt}/{args.retry + 1}", flush=True)
            runs.append(rec)

    if args.dry_run:
        for r in runs:
            print(r.get("cmd"))
        return 0

    analysis = analyse(runs, ks, args, budgets)
    report(runs, analysis, args)
    report_budget_axis(runs, analysis, args)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"args": {k: v for k, v in vars(args).items()},
             "runs": runs, "analysis": analysis}, ensure_ascii=False, indent=2))
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
