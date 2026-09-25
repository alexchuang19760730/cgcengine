#!/usr/bin/env python3
"""Separate one decode step into "caused by expert-cache miss" vs "structural", and price
the two ceilings that separation implies.

WHY THIS EXISTS
---------------
The segmented dispatcher emits, per step:

    CGC-DECPROF: step=.. segs=.. layers=.. total=T ms | wait=W cb=C submit=S ntok=N
                 [ | layer gpu_sum=.. union_sum=.. gap_sum=.. ms ]

and `wait + cb + submit == total`. The question the ABBA cannot answer is *why* W and C are
large: arm B (single submit) removes BOTH the hook (`cb`) and the inter-segment GPU holes
(`gap`) at once, so its 2.2x is one number for two different causes. Those causes have
completely different price tags -- killing misses is a cache/pool problem, killing the
per-segment structure is a graph-submission problem -- so they must not be pooled.

This script resolves it by regressing the per-layer terms on that layer's miss count:

    y_l = b0 + b1 * m_l          y in {cb, wait, gap, submit},  m_l from BATCHDBG

`b0` is what a layer costs when it misses nothing (the structural floor), `b1 * m` is what
the misses add. That single fit answers all three questions the run was commissioned for:

  Q1  Is the structural floor ~0, i.e. are `cb` and the window almost entirely miss-driven?
      (2026-09-20 server-channel answer: yes -- cb m=0 was 0.01 ms vs 0.695 ms/miss. This is
      the transfer check of that verdict onto today's llama-bench / prod-new carrier.)
  Q2  Ceiling if cb + inter-segment holes are removed:      step - cb - gap
  Q3  Ceiling if cache misses are removed:                  step - b1_cb*M - b1_wait*M

Q2 and Q3 are the SAME NUMBER if and only if Q1 holds, because Q2 deletes the whole term
while Q3 deletes only its miss-attributable part. Their difference is therefore not noise --
it is the measured size of "normal operation", and it is the number that decides whether
single-submit is worth implementing at all.

INSTRUMENTS (all env-gated, all already in the shipped binary -- 0 rebuild)
  CGC_DECODE_PROFILE=1        the step row + per-layer rows
  CGC_DECODE_PROFILE_ALL=1    print ALL layers, not just the top-8 by (wait+cb)
  CGC_GPU_TIMING=1            gpu/union/gap become non-zero (otherwise "NO TIMESTAMPS")
  CGC_HOOK_SPLIT=1            CGC-HOOKSPLIT: splits `cb` into pre/ensure/drain/tail;
                              `ensure` is the demand fill, i.e. miss cost measured directly
  LLAMA_EXPERT_CACHE_BATCH_DBG=1   BATCHDBG layer=<l> misses=<m> -- the regressor

  Do NOT set CGC_VERIFY_OP_TIMING: with it on, CGC-SEG / CGC-DECPROF / CGC-GPUTIME all emit
  zero rows (measured 2026-09-22, recorded in verify_marginal.py). Do NOT set CGC_SEG_BATCH:
  this arm must keep the 41-segment structure intact, since that structure is what it prices.

This is a diagnostic, not a delivery number: an instrumented run is slower than a clean one,
so its t/s is never quotable. Only the fitted slopes/intercepts and the derived ceilings are.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "check" / "llama_bench_matrix.py"

# ---------------------------------------------------------------------------------------
# Decision thresholds. FIXED IN ADVANCE, before any GPU run -- see module docstring. They are
# module constants so the rule a number was judged by is visible in the same file as the
# number, and so changing the rule is a visible edit rather than a hindsight adjustment.
#
# Anchors: the 2026-09-20 server-channel fit (docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md)
# measured cb(m=0) = 0.01 ms and 0.695 ms/miss. These thresholds are deliberately looser than
# that, because this run is a DIFFERENT carrier (llama-bench, MTP off, later binary) and the
# transfer is the thing under test -- a threshold copied from the old carrier would test the
# copy, not the transfer.
# ---------------------------------------------------------------------------------------
B0_CB_MAX_MS = 0.10        # structural floor of `cb`: PASS if intercept is below this
SLOPE_CB_MIN_MS = 0.10     # marginal cost of one miss inside `cb`
R2_MIN = 0.30              # the fit has to actually explain the layer-to-layer spread
SHARE_MISS_MIN_CB = 0.85   # fraction of mean `cb` attributable to the miss term
SHARE_MISS_MIN_GAP = 0.60  # ...and of `gap` (looser: gap is also shaped by submit order)
UB_RTOL = 0.05             # ceiling-2 vs ceiling-3 agreement tolerance, as frac of baseline

NTOK_DECODE_MAX = 8        # a step row with more tokens than this is prefill, not decode


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------
STEP_RX = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \([\d.]+%\) cb=([\d.]+) \([\d.]+%\) submit=([\d.]+) \([\d.]+%\) ntok=(\d+)"
    r"(?:\s*\|\s*layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms)?")
# Two per-layer forms with different punctuation; a pattern demanding the colon silently
# drops the top-8 rows. Same trap as decode_step_profile.py.
LAYER_RX = re.compile(
    r"CGC-DECPROF (?:all|top\d+): L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
    r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")
# Only printed when miss_exps is NON-EMPTY -- so a miss-free layer emits no line at all, and
# "no BATCHDBG for layer L" is evidence of m=0, not of a parse failure.
BATCH_RX = re.compile(r"BATCHDBG layer=(\d+) misses=(\d+)")
HOOK_RX = re.compile(
    r"CGC-HOOKSPLIT: n=(\d+)\s+pre=([\d.]+)\s+ensure=([\d.]+)\s+drain=([\d.]+)\s+tail=([\d.]+)"
    r"\s+us/call \(total ([\d.]+)\)")


def parse_log(text: str) -> dict:
    """Group the flat stderr stream into per-step records with per-layer rows.

    Ordering matters and is not obvious: BATCHDBG is printed DURING the step (from the hook)
    while CGC-DECPROF is printed at the END of it. So the BATCHDBG lines seen so far belong
    to the step announced by the NEXT step header. Prefill emits its own step row and its own
    BATCHDBG lines; those are dropped on the ntok test rather than merged into the first
    decode step, which would otherwise poison that step's regressor.
    """
    steps: list[dict] = []
    pending: dict[int, int] = {}
    cur: dict | None = None
    hook_rows: list[dict] = []

    for line in text.splitlines():
        m = BATCH_RX.search(line)
        if m:
            layer, misses = int(m.group(1)), int(m.group(2))
            pending[layer] = pending.get(layer, 0) + misses
            continue

        m = HOOK_RX.search(line)
        if m:
            hook_rows.append({"n": int(m.group(1)), "pre": float(m.group(2)),
                              "ensure": float(m.group(3)), "drain": float(m.group(4)),
                              "tail": float(m.group(5)), "total": float(m.group(6))})
            continue

        m = STEP_RX.search(line)
        if m:
            ntok = int(m.group(8))
            if ntok > NTOK_DECODE_MAX:
                pending = {}          # prefill: its misses are not this decode step's
                cur = None
                continue
            cur = {"step": int(m.group(1)), "segs": int(m.group(2)), "layers": int(m.group(3)),
                   "total_ms": float(m.group(4)), "wait_ms": float(m.group(5)),
                   "cb_ms": float(m.group(6)), "submit_ms": float(m.group(7)), "ntok": ntok,
                   "gpu_ms": float(m.group(9)) if m.group(9) else None,
                   "union_ms": float(m.group(10)) if m.group(10) else None,
                   "gap_ms": float(m.group(11)) if m.group(11) else None,
                   "misses": pending, "layer_rows": []}
            pending = {}
            steps.append(cur)
            continue

        m = LAYER_RX.search(line)
        if m and cur is not None:
            cur["layer_rows"].append({
                "L": int(m.group(1)), "wait": float(m.group(2)), "cb": float(m.group(3)),
                "submit": float(m.group(4)), "gpu": float(m.group(5)),
                "union": float(m.group(6)), "gap": float(m.group(7))})
    return {"steps": steps, "hook_split": hook_rows}


def layer_samples(steps: list[dict]) -> list[dict]:
    """Flatten to one row per (step, layer), attaching that layer's miss count.

    A layer present in the DECPROF rows but absent from `misses` is a miss-free layer -- that
    is the m=0 population, and it is what pins the intercept. Dropping it (easy to do, since
    BATCHDBG never prints it) would leave the fit with no leverage at the origin.
    """
    out = []
    for st in steps:
        miss = st["misses"]
        for r in st["layer_rows"]:
            row = dict(r)
            row["m"] = miss.get(r["L"], 0)
            row["step"] = st["step"]
            row["ntok"] = st["ntok"]
            out.append(row)
    return out


# ---------------------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------------------
def ols(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares with the regression's own r^2. Plain-Python on purpose."""
    n = len(xs)
    if n < 3:
        return {"n": n, "b0": None, "b1": None, "r2": None}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return {"n": n, "b0": my, "b1": None, "r2": None}   # every layer has the same m
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    syy = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (b0 + b1 * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / syy if syy > 0 else None
    return {"n": n, "b0": b0, "b1": b1, "r2": r2}


def median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------------------
# The three verdicts
# ---------------------------------------------------------------------------------------
def analyse(parsed: dict) -> dict:
    steps = parsed["steps"]
    res: dict = {"n_steps": len(steps)}
    if not steps:
        return {"error": "no decode step rows (is CGC_DECODE_PROFILE=1 set, and is this the "
                         "41-segment arm?)", "n_steps": 0}

    samples = layer_samples(steps)
    res["n_samples"] = len(samples)
    if len(samples) < 30:
        return {"error": f"only {len(samples)} (step,layer) samples; need >= 30 for a fit",
                "n_steps": len(steps)}

    ms = [s["m"] for s in samples]
    mean_m = sum(ms) / len(ms)

    # ---- Q1: per-term fits -------------------------------------------------------------
    fits = {}
    for key in ("cb", "wait", "gap", "submit", "union"):
        ys = [s[key] for s in samples]
        f = ols(ms, ys)
        f["mean_y"] = sum(ys) / len(ys)
        f["med_y"] = median(ys)
        # Fraction of the term's mean that the miss term accounts for. Clipped at 0: a
        # negative slope means the fit is noise and must not be read as "negative cost".
        if f["b1"] is None or f["mean_y"] <= 0:
            f["miss_share"] = None            # no regressor spread, or a silent term
        else:
            f["miss_share"] = max(f["b1"], 0.0) * mean_m / f["mean_y"]
        fits[key] = f
    res["fits"] = fits
    res["mean_misses_per_layer"] = mean_m

    cb, wait, gap = fits["cb"], fits["wait"], fits["gap"]

    # ---- Q1 verdict --------------------------------------------------------------------
    checks = {
        "cb_intercept_small": (cb["b0"] is not None and cb["b0"] <= B0_CB_MAX_MS),
        "cb_slope_positive": (cb["b1"] is not None and cb["b1"] >= SLOPE_CB_MIN_MS),
        "cb_fit_explains": (cb["r2"] is not None and cb["r2"] >= R2_MIN),
        "cb_miss_share": (cb["miss_share"] is not None and cb["miss_share"] >= SHARE_MISS_MIN_CB),
        "gap_miss_share": (gap["miss_share"] is not None
                           and gap["miss_share"] >= SHARE_MISS_MIN_GAP),
    }
    res["q1_checks"] = checks
    res["q1_verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["q1_detail"] = {
        "cb_floor_ms": cb["b0"], "cb_per_miss_ms": cb["b1"], "cb_r2": cb["r2"],
        "cb_miss_share": cb["miss_share"],
        "gap_floor_ms": gap["b0"], "gap_per_miss_ms": gap["b1"], "gap_r2": gap["r2"],
        "gap_miss_share": gap["miss_share"],
        "wait_per_miss_ms": wait["b1"], "wait_r2": wait["r2"]}

    # ---- baseline / ceilings -----------------------------------------------------------
    # Step-level medians, not means: a single stalled step would drag a mean off the plateau.
    step_ms = median([s["total_ms"] for s in steps])
    cb_ms = median([s["cb_ms"] for s in steps])
    gap_ms = median([s["gap_ms"] for s in steps if s["gap_ms"] is not None]) or 0.0
    ntok = median([s["ntok"] for s in steps])
    res["step_terms_ms"] = {"total": step_ms, "cb": cb_ms, "gap": gap_ms, "ntok": ntok}

    def tps(ms: float) -> float:
        return 1000.0 * ntok / ms if ms > 0 else float("inf")

    base_ms = step_ms
    # Q2: delete the hook entirely AND the inter-segment GPU holes.
    ub2_ms = max(base_ms - cb_ms - gap_ms, 1e-6)
    # Q3: delete only the miss-attributable part of cb and of wait, keeping both floors.
    # mean_m is per-LAYER, so the per-step miss budget is mean_m * layers_per_step; the slopes
    # are also per-layer, which is why the product below is the step-level miss cost.
    layers_per_step = sum(len(s["layer_rows"]) for s in steps) / len(steps)
    miss_term = ((cb["b1"] or 0.0) + (wait["b1"] or 0.0)) * mean_m * layers_per_step
    ub3_ms = max(base_ms - miss_term, 1e-6)

    res["ceilings"] = {
        "baseline_ms": base_ms, "baseline_tps": tps(base_ms),
        "ub2_no_cb_no_gap_ms": ub2_ms, "ub2_tps": tps(ub2_ms),
        "ub2_gain_pct": 100.0 * (base_ms - ub2_ms) / base_ms,
        "ub3_no_miss_ms": ub3_ms, "ub3_tps": tps(ub3_ms),
        "ub3_gain_pct": 100.0 * (base_ms - ub3_ms) / base_ms,
        "miss_term_ms": miss_term}
    delta = abs(tps(ub2_ms) - tps(ub3_ms)) / tps(base_ms)
    res["q23_agreement"] = {"rel_delta": delta, "tol": UB_RTOL,
                            "verdict": "CONSISTENT" if delta <= UB_RTOL else "DIVERGENT"}
    # The gap between the two ceilings IS the structural (non-miss) price of the segmented
    # dispatcher -- the part single-submit would recover even if every miss stayed.
    res["structural_ms"] = ub3_ms - ub2_ms

    # ---- hook split: an independent read of the same quantity --------------------------
    if parsed["hook_split"]:
        h = parsed["hook_split"][-1]        # running mean; the last row has the most calls
        res["hook_split_us_per_call"] = {k: h[k] for k in ("pre", "ensure", "drain", "tail",
                                                           "total", "n")}
        # `ensure` is the demand fill. Cross-check: it should be the bulk of cb.
        res["hook_split_check"] = {
            # Units: h[*] are MICROseconds PER CALL, cb_ms is milliseconds PER STEP.
            # Comparable only after x calls-per-step (= h["n"] / n_steps). Dividing them
            # directly used to print a meaningless 0.03.
            "calls_per_step": round(h["n"] / float(len(parsed["steps"]) or 1), 1),
            "ensure_ms_per_step": round(h["ensure"] * (h["n"] / float(len(parsed["steps"]) or 1))
                                        / 1000.0, 3),
            "cb_ms_per_step": round(cb_ms, 3),
            "ensure_over_cb": ((h["ensure"] * (h["n"] / float(len(parsed["steps"]) or 1)) / 1000.0)
                               / cb_ms) if cb_ms > 0 else None,
            "note": "ensure = demand fill (the miss cost measured directly, no regression)"}
    return res


def _f(v, spec=".3f"):
    """Format a possibly-None float (n<3 samples leave the slope/r2 undefined)."""
    return ("n/a" if v is None else format(v, spec))


def verdict_text(res: dict) -> str:
    if res.get("error"):
        return f"NO VERDICT: {res['error']}"
    L = []
    d = res["q1_detail"]
    L.append(f"Q1  structural floor ~ 0 ?            {res['q1_verdict']}")
    L.append(f"    cb   : floor {_f(d['cb_floor_ms'])} ms  per-miss {_f(d['cb_per_miss_ms'])} ms  "
             f"r2 {_f(d['cb_r2'], '.2f')}  miss-share {_f(d['cb_miss_share'], '.2f')}")
    L.append(f"    gap  : floor {_f(d['gap_floor_ms'])} ms  per-miss {_f(d['gap_per_miss_ms'])} ms  "
             f"r2 {_f(d['gap_r2'], '.2f')}  miss-share {_f(d['gap_miss_share'], '.2f')}")
    L.append(f"    fail : {[k for k, v in res['q1_checks'].items() if not v] or '-'}")
    c = res["ceilings"]
    t = res["step_terms_ms"]
    L.append("")
    L.append(f"baseline  {c['baseline_ms']:.2f} ms/step -> {c['baseline_tps']:.2f} t/s   "
             f"(cb {t['cb']:.2f} ms, gap {t['gap']:.2f} ms, ntok {t['ntok']:.0f})")
    L.append(f"Q2  -cb -gap   {c['ub2_no_cb_no_gap_ms']:.2f} ms -> {c['ub2_tps']:.2f} t/s   "
             f"(+{c['ub2_gain_pct']:.1f}%)")
    L.append(f"Q3  miss -> 0  {c['ub3_no_miss_ms']:.2f} ms -> {c['ub3_tps']:.2f} t/s   "
             f"(+{c['ub3_gain_pct']:.1f}%)")
    a = res["q23_agreement"]
    L.append(f"    Q2 vs Q3   {a['verdict']}  (rel delta {_f(a['rel_delta'])}, tol {a['tol']})")
    L.append(f"    structural (non-miss) part of the segmented dispatcher: "
             f"{res['structural_ms']:.2f} ms/step")
    if res.get("hook_split_us_per_call"):
        h = res["hook_split_us_per_call"]
        L.append(f"    hook split us/call: pre {h['pre']:.1f} ensure {h['ensure']:.1f} "
                 f"drain {h['drain']:.1f} tail {h['tail']:.1f} (n={h['n']})")
        hc = res.get("hook_split_check") or {}
        if hc.get("ensure_over_cb") is not None:
            L.append(f"    ensure {hc['ensure_ms_per_step']:.2f} ms/step vs "
                     f"cb {hc['cb_ms_per_step']:.2f} ms/step  "
                     f"({hc['calls_per_step']:.0f} calls/step) -> ratio {hc['ensure_over_cb']:.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------
# Presence-gated switches: off means ABSENT, never "=0". CGC_VERIFY_OP_TIMING is deliberately
# popped as well -- it does not merely add rows, it silences the three we need.
INSTRUMENT_ENV = {
    "CGC_DECODE_PROFILE": "1",
    "CGC_DECODE_PROFILE_ALL": "1",
    "CGC_GPU_TIMING": "1",
    "CGC_HOOK_SPLIT": "1",
    "LLAMA_EXPERT_CACHE_BATCH_DBG": "1",
}

# P0: skip-load expert 不 read_raw -> 省 ~10.9 GiB 匿名駐留。
# 沒帶它會讓 prefill 在 llama_context::synchronize 撞
# kIOGPUCommandBufferCallbackErrorOutOfMemory（recommendedMaxWorkingSetSize ~11.45 GB）。
P0_ENV = {"CGC_EXPERT_SKIP_READRAW": "1"}

# [2026-09-24 定讞] 沒有這條，expert cache 整個不會啟用 —— 跟 -expert-cache 給多少無關。
#   llama-model-loader.cpp:1113  compute_l4_pool_capacity():
#       if ((no_gather && no_gather[0]) || !getenv("LLAMA_EXPERT_CACHE_ALLOW_NGL")) return;
#   llama.cpp:402  l4_path = n_gpu_layers > 0 && getenv("LLAMA_EXPERT_CACHE_ALLOW_NGL") && ...
# 缺它 => pool_cap_slots=0 / routable=0 => 13.0 GB 模型整包上 Metal（上限 11.45 GB）
#        => prefill 收尾 ggml_metal_synchronize 撞 kIOGPUCommandBufferCallbackErrorOutOfMemory。
# 帶它 => pool_cap_slots=143 / routable=143（對上 514 份歷史 log），OOM 消失。
# ⚠ presence-gated（getenv 非空即可），但寫 "1" 最不容易被誤讀。
# ⚠ 全 repo 只有 deploy-harmonyos/* 會 export 它；scripts/check/ 下沒任何 runner 有設。
REQUIRED_ENV = {"LLAMA_EXPERT_CACHE_ALLOW_NGL": "1"}
FORBIDDEN_VARS = ("CGC_VERIFY_OP_TIMING", "CGC_SEG_BATCH", "CGC_B_SCHEME",
                  "CGC_SLOT_TABLE_GPU")


def build_env(extra: dict[str, str] | None = None, p0: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    for k in FORBIDDEN_VARS:
        env.pop(k, None)
    env.update(INSTRUMENT_ENV)
    env.update(REQUIRED_ENV)
    if p0:
        env.update(P0_ENV)
    else:
        env.pop("CGC_EXPERT_SKIP_READRAW", None)
    if extra:
        env.update(extra)
    return env


def run(args) -> dict:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stderr_log = workdir / "miss_axis_stderr.log"
    cmd = [sys.executable, str(HARNESS),
           "--arms", args.profile,
           "--reps", str(args.reps),
           "--prompt", str(args.prompt),
           "--gen", str(args.gen),
           "--depths", str(args.depth),
           "--ctx-size", str(args.ctx_size),
           "--workdir", str(workdir),
           "--json", str(workdir / "miss_axis_res.json")]
    if args.ubatch:
        cmd += ["--batch", str(args.ubatch), "--ubatch", str(args.ubatch)]
    # [2026-09-24] 交付 cell 的兩個約定，原本漏傳 =>
    #   沒有 --spec-type  => 無 MTP/spec，ntok=1（已知 off 9.82 vs on 12.62，×1.28）
    #   沒有 --warm-skip  => 計時視窗內含冷池 fill
    # 兩者都缺時 tg 讀到 10.14 t/s，那是「關掉 MTP 的 cell」應有的值，不是引擎退化。
    if args.spec_type:
        cmd += ["--spec-type", args.spec_type]
        cmd += ["--spec-draft-n-max", str(args.spec_draft_n_max)]
    if args.warm_skip:
        cmd += ["--warm-skip", str(args.warm_skip)]
    env = build_env(p0=bool(args.p0))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    wall = time.time() - t0
    stderr_log.write_text(proc.stderr, errors="replace")
    (workdir / "miss_axis_stdout.log").write_text(proc.stdout, errors="replace")

    # [2026-09-24 修] CGC-DECPROF / CGC-HOOK 行在 **子行程 llama-bench** 的 stderr 裡。
    # llama_bench_matrix.py 用 capture_output=True 把它吃掉，只寫進
    #   {workdir}/llama_bench_{tag}_p{prompt}_n{gen}_d{depth}_r{reps}.stderr.log
    # 而它自己印到 stderr 的只有一行失敗摘要。parse 父行程的 stderr 會永遠得到 steps=0。
    child = sorted(workdir.glob(f"llama_bench_*_p{args.prompt}_n{args.gen}_d{args.depth}"
                                f"_r{args.reps}.stderr.log"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    child_log = child[0] if child else None
    body = child_log.read_text(errors="replace") if child_log else proc.stderr
    rec = {"rc": proc.returncode, "wall_s": round(wall, 1), "stderr_log": str(stderr_log),
           "child_log": str(child_log) if child_log else None,
           "cmd": " ".join(cmd)}
    parsed = parse_log(body)
    rec["n_steps"] = len(parsed["steps"])
    rec["n_hook_rows"] = len(parsed["hook_split"])
    rec["result"] = analyse(parsed)
    return rec


# ---------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------
def _synth(n_steps=60, n_layers=40, cb_floor=0.01, cb_slope=0.70,
           gap_floor=0.0, gap_slope=0.30, sync_floor=0.10, union=1.0,
           max_m=6, prefill=True):
    """Build a synthetic CGC-DECPROF stream with known coefficients.

    `wait` is derived as gap + union + sync_floor rather than being an independent knob,
    because that containment is a property of the real instrument: the CPU polls the segment
    to completion, so the segment's own GPU hole is inside the poll window, not beside it.
    An earlier version of this generator made wait and gap independent slopes, and every
    ceiling comparison it produced was nonsense -- wait's miss term exceeded gap's entirety,
    so "delete the misses" appeared to buy more than "delete cb and gap" did.
    """
    out = []
    if prefill:                              # a prefill step + its own BATCHDBG noise
        out.append("BATCHDBG layer=3 misses=5 slots: e12->s3")
        out.append("CGC-DECPROF: step=0 segs=41 layers=40 total=5791.00 ms | "
                   "wait=100.00 (2%) cb=5600.00 (97%) submit=91.00 (2%) ntok=512")
    sid = 0
    for st in range(1, n_steps + 1):
        misses = {}
        for l in range(n_layers):
            m = (l * 7 + st * 3) % (max_m + 1) if ((l + st) % 4 == 0) else 0
            if m:
                misses[l] = m
                out.append(f"BATCHDBG layer={l} misses={m} slots: e1->s2")
        w = c = s = g = 0.0
        gpu = u = 0.0
        rows = []
        for l in range(n_layers):
            m = misses.get(l, 0)
            lg = gap_floor + gap_slope * m
            lw = lg + union + sync_floor          # wait CONTAINS the hole, see docstring
            lc = cb_floor + cb_slope * m
            ls = 0.05
            lgpu, luni = 1.2, union
            w += lw; c += lc; s += ls; g += lg; gpu += lgpu; u += luni
            rows.append(f"CGC-DECPROF all: L{l} wait={lw:.2f} cb={lc:.2f} submit={ls:.2f} ms "
                        f"gpu={lgpu:.2f} union={luni:.2f} gap={lg:.2f} sg=3 n=1 "
                        f"st=0.000 en=0.000")
        tot = w + c + s
        # Real emission order: BATCHDBG during the step, then the step header at its END,
        # then the per-layer rows. Getting this backwards silently drops the first step's
        # layers, which is exactly the bug this self-test exists to catch.
        out.append(f"CGC-DECPROF: step={st} segs=41 layers={n_layers} total={tot:.2f} ms | "
                   f"wait={w:.2f} ({100*w/tot:.0f}%) cb={c:.2f} ({100*c/tot:.0f}%) "
                   f"submit={s:.2f} ({100*s/tot:.0f}%) ntok=1 | "
                   f"layer gpu_sum={gpu:.2f} union_sum={u:.2f} gap_sum={g:.2f} ms")
        out.extend(rows)
        sid += 1
        if st % 4 == 0:
            out.append(f"CGC-HOOKSPLIT: n={st*40}  pre=20.0  ensure={cb_slope*1000*0.25:.1f}  "
                       f"drain=5.0  tail=8.0 us/call (total 100.0)")
    return "\n".join(out)


def self_test() -> int:
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name} {extra}")

    print("--- parser ---")
    txt = _synth()
    p = parse_log(txt)
    chk("60 decode steps parsed (prefill step dropped)", len(p["steps"]) == 60,
        f"got {len(p['steps'])}")
    chk("hook rows parsed", len(p["hook_split"]) == 15, f"got {len(p['hook_split'])}")
    chk("first step is not the prefill row", p["steps"][0]["ntok"] == 1)
    chk("prefill BATCHDBG did not leak into step 1",
        p["steps"][0]["misses"].get(3, 0) != 5, f"got {p['steps'][0]['misses'].get(3)}")
    chk("40 layer rows per step", len(p["steps"][0]["layer_rows"]) == 40)
    samp = layer_samples(p["steps"])
    chk("samples = 60*40", len(samp) == 2400, f"got {len(samp)}")
    zero = [s for s in samp if s["m"] == 0]
    chk("m=0 population present (pins the intercept)", len(zero) > 500, f"got {len(zero)}")
    chk("m=0 layers have a nonzero cb (the floor)",
        all(abs(s["cb"] - 0.01) < 1e-6 for s in zero[:50]))

    print("--- fit recovers known coefficients ---")
    r = analyse(p)
    chk("no error", not r.get("error"), str(r.get("error")))
    f = r["fits"]["cb"]
    chk("cb intercept ~ 0.01", abs(f["b0"] - 0.01) < 1e-6, f"got {f['b0']}")
    chk("cb slope ~ 0.70", abs(f["b1"] - 0.70) < 1e-6, f"got {f['b1']}")
    chk("cb r2 ~ 1", f["r2"] > 0.999, f"got {f['r2']}")
    g = r["fits"]["gap"]
    chk("gap slope ~ 0.30", abs(g["b1"] - 0.30) < 1e-6, f"got {g['b1']}")

    print("--- Q1 verdict on a miss-driven world ---")
    chk("Q1 PASS", r["q1_verdict"] == "PASS", str(r["q1_checks"]))

    print("--- Q2 vs Q3 agree when the floor is ~0 ---")
    a = r["q23_agreement"]
    chk("CONSISTENT", a["verdict"] == "CONSISTENT",
        f"rel_delta={a['rel_delta']:.4f} ub2={r['ceilings']['ub2_tps']:.2f} "
        f"ub3={r['ceilings']['ub3_tps']:.2f}")
    chk("structural term is small", abs(r["structural_ms"]) < 1.0,
        f"got {r['structural_ms']:.3f}")

    print("--- Q2 vs Q3 diverge when a real fixed cost exists ---")
    # Same world, but every layer pays 0.5 ms of structure regardless of misses.
    txt2 = _synth(cb_floor=0.50, gap_floor=0.80)
    r2 = analyse(parse_log(txt2))
    chk("Q1 FAILS on a structural floor", r2["q1_verdict"] == "FAIL", str(r2["q1_checks"]))
    chk("structural term now positive", r2["structural_ms"] > 1.0,
        f"got {r2['structural_ms']:.3f}")
    chk("Q2 overshoots Q3 (deleting cb+gap claims more than misses explain)",
        r2["ceilings"]["ub2_tps"] > r2["ceilings"]["ub3_tps"],
        f"ub2={r2['ceilings']['ub2_tps']:.2f} ub3={r2['ceilings']['ub3_tps']:.2f}")

    print("--- degenerate inputs ---")
    chk("empty log -> error not crash", "error" in analyse(parse_log("")))
    chk("prefill-only log -> error",
        "error" in analyse(parse_log(
            "CGC-DECPROF: step=0 segs=41 layers=40 total=5791.00 ms | wait=100.00 (2%) "
            "cb=5600.00 (97%) submit=91.00 (2%) ntok=512")))
    chk("too few samples -> error",
        "error" in analyse(parse_log(_synth(n_steps=1, n_layers=10))))
    r3 = analyse(parse_log(_synth(n_steps=60).replace("BATCHDBG", "XDBG")))
    chk("all-zero regressor still produces a verdict (no crash, r2 None)",
        r3.get("error") is None and r3["fits"]["cb"]["b1"] is None)

    print("--- env construction ---")
    os.environ["CGC_VERIFY_OP_TIMING"] = "1"      # must be stripped
    os.environ["CGC_SEG_BATCH"] = "1"             # must be stripped: this is the A arm
    e = build_env()
    del os.environ["CGC_VERIFY_OP_TIMING"], os.environ["CGC_SEG_BATCH"]
    chk("forbidden vars stripped", all(k not in e for k in FORBIDDEN_VARS))
    chk("instruments present", all(e.get(k) == v for k, v in INSTRUMENT_ENV.items()))
    chk("ALLOW_NGL present (else pool_cap_slots=0 -> Metal OOM)",
        e.get("LLAMA_EXPERT_CACHE_ALLOW_NGL") == "1")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", type=int, default=1, choices=(0, 1),
                    help="CGC_EXPERT_SKIP_READRAW (default 1). MUST stay 1: without it the cell OOMs in prefill.")
    ap.add_argument("--self-test", action="store_true", help="run the built-in checks")
    ap.add_argument("--log", type=Path, help="analyse an existing stderr log (no GPU)")
    ap.add_argument("--json", type=Path, help="write the verdict as JSON")
    ap.add_argument("--run", action="store_true",
                    help="run one instrumented 41-segment arm (needs an idle GPU)")
    ap.add_argument("--profile", default="prod-new")
    # 交付 cell 約定（`MEMORY_PERF.md`：--batch 512 / --ctx-size 4096 / --warm-skip 64 /
    # --spec-type draft-mtp）。這裡 --batch 用 profile 值 5632、--ctx-size 預設 0
    # （8192 會 Metal OOM），但 spec 與 warm-skip 必須帶，否則讀到的是 MTP-off 的數字。
    ap.add_argument("--spec-type", default="draft-mtp",
                    help="llama-bench --spec-type (default draft-mtp = the delivery cell). "
                         "Pass '' to disable and get ntok=1 decode.")
    ap.add_argument("--spec-draft-n-max", type=int, default=3)
    ap.add_argument("--warm-skip", type=int, default=64,
                    help="tokens generated per rep before the clock starts (delivery cell = 64)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depth", type=int, default=512)
    # [2026-09-24 定讞] 預設 0（= 不傳 --ctx-size，llama-bench 用 p+n 自己推）。
    # 傳 8192 會讓 llama-bench 配置 8192-token KV/compute buffer，把 Metal 推過
    # recommendedMaxWorkingSetSize 11453 MB ⇒ prefill/decode 收尾
    # kIOGPUCommandBufferCallbackErrorOutOfMemory（A/B 實測：只差這一個 flag，
    # 無 ctx-size 存活 6 steps、--ctx-size 8192 當場 rc=-6）。
    # 標準測試卡 docs/PROD_NEW_TEST_CARD_2026-09-24.md §2 的指令也沒有 --ctx-size，§7 寫 --ctx-size 0。
    ap.add_argument("--ctx-size", type=int, default=0)
    ap.add_argument("--ubatch", type=int, default=5632)
    ap.add_argument("--workdir", default="Backup/miss_axis")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.log:
        parsed = parse_log(args.log.read_text(errors="replace"))
        res = analyse(parsed)
        print(verdict_text(res))
        if args.json:
            args.json.write_text(json.dumps(res, indent=2))
        return 0 if not res.get("error") else 2

    if args.run:
        rec = run(args)
        print(f"rc={rec['rc']}  wall={rec['wall_s']}s  steps={rec['n_steps']}  "
              f"hook_rows={rec['n_hook_rows']}")
        print(f"stderr log: {rec['stderr_log']}")
        print()
        print(verdict_text(rec["result"]))
        if args.json:
            args.json.write_text(json.dumps(rec, indent=2))
        return 0 if not rec["result"].get("error") else 2

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
