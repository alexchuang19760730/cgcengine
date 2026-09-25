#!/usr/bin/env python3
"""Price the two ways of handling the misses that single-submit (S1) cannot avoid.

WHY THIS EXISTS
---------------
`docs/SHAPE_GAP_TO_TARGETS_2026-09-24.md` §2.5 leaves the single open gap as "miss
handling", and prices only ONE of the two possible handlers:

    whole-layer recompute:  wrong_layers x per_layer_ms   = ~8 x 1.6-2.5 = 12.8-20 ms
    vs. what single-submit saves                          = 13.6 ms
    => net +1% .. -8%  (i.e. not worth doing)

The layer is the wrong unit. A MoE layer's output is

    y = sum_i g_i * down_i(up_i(x))          i over the top-k experts

so an expert that was served by a placeholder slot contributes ~0 and can be repaired by
re-running ONLY that term and ADDING it back -- the other k-1 terms are already correct and
must not be recomputed. The unit is the expert, not the layer, and the price differs by
roughly a factor of k (=8).

This script answers, from an already-recorded trace (zero GPU):

  Q0  Do the misses decay?  If they were only cold-start, single-segment could wait them
      out and no handler would be needed at all. (Measured 2026-09-24: they do NOT decay.)
  Q1  whole-layer recompute:            wrong_layers x per_layer_ms
  Q2  per-expert recompute:             misses_per_step x per_expert_ms  (+ launch overhead)
  Q3  net of each against what S1 saves, and the resulting t/s band.

INSTRUMENTS needed in the trace (all env-gated, already shipped -- 0 rebuild):
  CGC_DECODE_PROFILE=1 / _ALL=1   -> "CGC-DECPROF: step=.. total=T ms | .. ntok=N"
  CGC_GPU_TIMING=1                -> "CGC-GPUTIME: step=.. gpu_union=U ms .."
  LLAMA_EXPERT_CACHE_BATCH_DBG=1  -> "BATCHDBG layer=<l> misses=<m>"

This is offline arithmetic on a trace, not a measurement: it can only be as good as the
per-layer split fed into --moe-share (default 0.6-0.8, sensitivity band).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

# --- knobs --------------------------------------------------------------------------------
TOP_K = 8                  # experts routed per layer (Qwen3.6-35B-A3B; 40 layers x 8 = 320 req/step
                           # matches the 47000 requests / 147 decode steps seen in the clean arm)
MOE_SHARE_LO = 0.60        # fraction of one layer's GPU time spent in the MoE block
MOE_SHARE_HI = 0.80
LAUNCH_US = 13.0           # per graph node, from §3 (RMS_NORM node / CB fixed cost)
NODES_PER_EXPERT = 3       # up / gate / down GEMV
S1_SAVES_MS = 13.6         # what single-submit removes (SHAPE_GAP §2.5 headline number)
S1_SAVES_MS_HI = 17.04     # same thing measured as cb+gap by miss_axis Q2 (2026-09-24)

RE_STEP = re.compile(r"CGC-DECPROF: step=(\d+).*?total=([\d.]+).*?ntok=(\d+)")
RE_GPU = re.compile(r"CGC-GPUTIME: step=(\d+).*?gpu_union=([\d.]+)")
RE_BATCHDBG = re.compile(r"BATCHDBG layer=(\d+) misses=(\d+)")


# --- parse --------------------------------------------------------------------------------
def parse(text: str) -> dict:
    steps: list[dict] = []          # decode-step rows in file order
    unions: dict[int, float] = {}
    cur: dict | None = None
    for line in text.splitlines():
        m = RE_STEP.search(line)
        if m:
            cur = {"step": int(m.group(1)), "total": float(m.group(2)),
                   "ntok": int(m.group(3)), "m": {}}
            steps.append(cur)
            continue
        m = RE_GPU.search(line)
        if m:
            unions[int(m.group(1))] = float(m.group(2))
            continue
        m = RE_BATCHDBG.search(line)
        if m and cur is not None:
            cur["m"][int(m.group(1))] = int(m.group(2))
    dec = [s for s in steps if s["ntok"] <= 8]          # drop the prefill rows
    return {"steps": dec, "unions": unions, "n_all": len(steps)}


# --- analyse ------------------------------------------------------------------------------
def analyse(p: dict, top_k: int = TOP_K,
            share_lo: float = MOE_SHARE_LO, share_hi: float = MOE_SHARE_HI,
            saves_lo: float = S1_SAVES_MS, saves_hi: float = S1_SAVES_MS_HI,
            wrong_layers: float | None = None) -> dict:
    steps = p["steps"]
    if len(steps) < 3:
        return {"error": f"only {len(steps)} decode steps (need >= 3)"}
    n_layers = 40
    miss_layers = [len([v for v in s["m"].values() if v > 0]) for s in steps]
    miss_total = [sum(s["m"].values()) for s in steps]

    # Q0 -- decay: compare thirds. A cold-start-only population collapses toward 0.
    third = len(steps) // 3
    thirds = [st.mean(miss_layers[i * third:(i + 1) * third]) for i in range(3)]
    decay = thirds[2] < 0.5 * thirds[0] if thirds[0] > 0 else False

    # per-layer GPU cost, from the union (the honest GPU-side measure; per-layer DECPROF
    # rows are on the `cb`-like partial caliber and under-read by ~9x -- see MISS_AXIS §3).
    u = [p["unions"][s["step"]] for s in steps if s["step"] in p["unions"]]
    union_ms = st.median(u) if u else st.median([s["total"] for s in steps])
    layer_ms = union_ms / n_layers
    step_total = st.median([s["total"] for s in steps])

    expert_lo = layer_ms * share_lo / top_k
    expert_hi = layer_ms * share_hi / top_k

    m_step = st.mean(miss_total)                     # misses per step
    wl = wrong_layers if wrong_layers is not None else st.mean(miss_layers)

    # Q1 whole-layer
    q1_lo, q1_hi = wl * layer_ms * 0.87, wl * layer_ms * 1.49   # 1.6..2.5 ms/ layer band
    # Q2 per-expert: GEMV time + launch overhead of up/gate/down per repaired expert
    launch = m_step * NODES_PER_EXPERT * LAUNCH_US / 1000.0
    q2_lo, q2_hi = m_step * expert_lo + launch, m_step * expert_hi + launch

    def net(cost: float) -> float:
        return saves_lo - cost                       # conservative: smallest saving

    return {
        "n_steps": len(steps),
        "miss_layers_per_step_mean": round(st.mean(miss_layers), 2),
        "miss_layers_per_step_median": round(st.median(miss_layers), 2),
        "misses_per_step_mean": round(m_step, 2),
        "thirds_miss_layers": [round(t, 2) for t in thirds],
        "decays": decay,
        "step_total_ms": round(step_total, 2),
        "union_ms": round(union_ms, 2),
        "layer_ms": round(layer_ms, 3),
        "expert_ms_band": [round(expert_lo, 3), round(expert_hi, 3)],
        "q1_whole_layer_ms": [round(q1_lo, 2), round(q1_hi, 2)],
        "q1_net_ms": [round(net(q1_hi), 2), round(net(q1_lo), 2)],
        "q2_per_expert_ms": [round(q2_lo, 2), round(q2_hi, 2)],
        "q2_launch_ms": round(launch, 2),
        "q2_net_ms": [round(net(q2_hi), 2), round(net(q2_lo), 2)],
        "wrong_layers_used": round(wl, 2),
        "saves_ms": [saves_lo, saves_hi],
        "verdict_q1": "PASS" if net(q1_hi) > 3 else "FAIL",
        "verdict_q2": "PASS" if net(q2_hi) > 3 else "FAIL",
        "ts_now": round(1000.0 / step_total, 2),
        "ts_band_q2": [round(1000.0 / (step_total - net(q2_lo)), 2),
                       round(1000.0 / (step_total - net(q2_hi)), 2)],
        "ts_band_q2_from_saves_hi": [round(1000.0 / (step_total - (saves_hi - q2_lo)), 2),
                                     round(1000.0 / (step_total - (saves_hi - q2_hi)), 2)],
    }


def verdict_text(r: dict) -> str:
    if r.get("error"):
        return f"ERROR: {r['error']}"
    L = []
    L.append(f"trace: {r['n_steps']} decode steps, union {r['union_ms']} ms/step, "
             f"total {r['step_total_ms']} ms/step -> {r['ts_now']} t/s (median caliber)")
    L.append("")
    L.append(f"Q0  do misses decay?            {'YES' if r['decays'] else 'NO'}")
    L.append(f"    miss-layers/step by third:  {r['thirds_miss_layers']}  "
             f"(mean {r['miss_layers_per_step_mean']}, misses/step {r['misses_per_step_mean']})")
    if not r["decays"]:
        L.append("    => misses are STEADY-STATE, not cold start: a handler is mandatory.")
    L.append("")
    L.append(f"    per-layer GPU   {r['layer_ms']} ms   per-expert {r['expert_ms_band']} ms "
             f"(MoE share {MOE_SHARE_LO}-{MOE_SHARE_HI}, top-{TOP_K})")
    L.append("")
    L.append(f"Q1  whole-layer recompute  {r['wrong_layers_used']} layers x 1.6-2.5 ms "
             f"= {r['q1_whole_layer_ms']} ms   net {r['q1_net_ms']} ms   -> {r['verdict_q1']}")
    L.append(f"Q2  per-expert recompute   {r['misses_per_step_mean']} misses x "
             f"{r['expert_ms_band']} + {r['q2_launch_ms']} launch = {r['q2_per_expert_ms']} ms"
             f"   net {r['q2_net_ms']} ms   -> {r['verdict_q2']}")
    L.append("")
    L.append(f"    t/s if Q2 lands: {r['ts_band_q2']} (vs S1 saves {r['saves_ms'][0]} ms) "
             f"/ {r['ts_band_q2_from_saves_hi']} (vs {r['saves_ms'][1]} ms)")
    return "\n".join(L)


# --- self-test ----------------------------------------------------------------------------
def self_test() -> int:
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}")

    n_layers, k = 40, 8
    # synthetic steady stream: ~18 miss layers/step, ~22 misses/step, 40 layers, 1.68 ms/layer
    lines = []
    for i in range(1, 31):
        sid = i * 4
        for l in range(n_layers):
            if (l + i) % 2 == 0:
                lines.append(f"BATCHDBG layer={l} misses={1 + (l % 2)}")
        lines.append(f"CGC-DECPROF: step={sid} segs=41 layers=40 total=73.40 ms | "
                     f"wait=60.00 cb=4.00 submit=3.40 ntok=1")
        lines.append(f"CGC-GPUTIME: step={sid} segs=40 bufs=360 skipped=0 wait=60.00 "
                     f"gpu_busy_sum=88.00 (146%) gpu_union=67.20 (104%) gap=14.00 (22%) ms")
    p = parse("\n".join(lines))
    chk("parses 30 decode steps", len(p["steps"]) == 30)
    chk("unions captured", len(p["unions"]) == 30)
    chk("prefill row (ntok=512) filtered out",
        all(s["ntok"] <= 8 for s in p["steps"]))

    r = analyse(p)
    chk("no error", "error" not in r)
    chk("layer_ms = union/40", abs(r["layer_ms"] - 67.2 / 40) < 1e-6)
    chk("steady stream does not decay", r["decays"] is False)
    chk("miss-layers/step ~ 20", 18 <= r["miss_layers_per_step_mean"] <= 22)
    chk("per-expert cheaper than per-layer",
        r["expert_ms_band"][1] < r["layer_ms"])
    chk("Q2 cheaper than Q1", r["q2_per_expert_ms"][1] < r["q1_whole_layer_ms"][0])
    chk("Q2 PASSes where Q1 FAILs",
        r["verdict_q2"] == "PASS" and r["verdict_q1"] == "FAIL")
    chk("q2 net positive", r["q2_net_ms"][0] > 0)

    # decay detection: misses only in the first third
    lines2 = []
    for i in range(1, 31):
        sid = i * 4
        if i <= 10:
            for l in range(n_layers):
                lines2.append(f"BATCHDBG layer={l} misses=2")
        lines2.append(f"CGC-DECPROF: step={sid} segs=41 layers=40 total=73.40 ms | "
                      f"wait=60.00 cb=4.00 submit=3.40 ntok=1")
        lines2.append(f"CGC-GPUTIME: step={sid} segs=40 bufs=360 skipped=0 wait=60.00 "
                      f"gpu_busy_sum=88.00 (146%) gpu_union=67.20 (104%) gap=14.00 (22%) ms")
    chk("decay detected when it exists", analyse(parse("\n".join(lines2)))["decays"] is True)

    # too few steps -> explicit error, not a crash
    chk("short trace errors cleanly",
        "error" in analyse(parse("CGC-DECPROF: step=4 segs=41 layers=40 total=73.4 ms | "
                                 "wait=60 cb=4 submit=3 ntok=1")))
    # wrong_layers override is respected
    r2 = analyse(p, wrong_layers=8.0)
    chk("--wrong-layers override used", r2["wrong_layers_used"] == 8.0)
    # 8 wrong layers x 1.68 ms/layer = 13.4 ms ~= the 13.6 ms S1 saves: even the OPTIMISTIC
    # end of the band leaves <3 ms, which is below the resolution we can afford to measure.
    chk("Q1 with 8 wrong layers nets <3 ms even at best", r2["q1_net_ms"][1] <= 3.0)
    chk("Q1 with 8 wrong layers FAILs", r2["verdict_q1"] == "FAIL")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--log", type=Path, help="an existing stderr log (offline, no GPU)")
    ap.add_argument("--wrong-layers", type=float, default=None,
                    help="override wrong-layers/step for Q1 (default: measured miss-layers)")
    ap.add_argument("--saves", type=float, default=S1_SAVES_MS,
                    help="ms/step that single-submit saves (default 13.6)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.log:
        ap.print_help()
        return 0

    res = analyse(parse(args.log.read_text(errors="replace")),
                  wrong_layers=args.wrong_layers, saves_lo=args.saves)
    print(verdict_text(res))
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    return 0 if "error" not in res else 2


if __name__ == "__main__":
    sys.exit(main())
