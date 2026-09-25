#!/usr/bin/env python3
"""Shape-1 wait budget: is the per-layer wait of the segmented arm miss-bound or round-trip-bound?

WHY THIS EXISTS (the question shape 1 was blocked on)
----------------------------------------------------
"Shape 1" = single-submit S1 with a *synchronous* wait for whatever the layer needs.  The cheap
argument for it was `48.2 ms (S1) + 20.8 ms (fill) = 69 ms`, i.e. +28% over the 88.5 ms segmented
step, with no async fill and no bit-exact recompute.  That argument is only valid if the cost of
waiting is proportional to the MISSES it waits for.  If instead the wait is dominated by the
per-segment round trip (wait for the whole segment to complete -> fire hook -> submit next), then
reducing misses buys nothing and only overlap (shape 2) can pay.

The two are separable with instruments that already exist, on the arm that already carries them:
  * `CGC-DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1` prints, per step, one `CGC-DECPROF all:` line
    per layer (wait / cb / submit, and with `CGC_GPU_TIMING=1` also gpu / union / gap).  This block
    lives AFTER the 41-segment loop, so it is only reachable on the SEGMENTED arm -- under
    `CGC_SEG_BATCH=1` the function returns before the loop (ggml-backend.cpp:1781) and the whole
    instrument disappears (measured 2026-09-25: DECPROF=0, HOOK=0, miss dump file never created).
  * `LLAMA_EXPERT_CACHE_BATCH_DBG=1` prints `BATCHDBG layer=<L> misses=<n> slots: ...` once per
    layer per decode step, and ONLY for layers that had >=1 miss (llama-expert-cache.cpp:1434).
    That is the miss axis that belongs with the DECPROF block: both describe decode steps, layer by
    layer.
  * `LLAMA_EXPERT_CACHE_MISS_DUMP=<path>` writes `"<layer> <expert>"` lines for the whole run
    (llama-expert-cache.cpp:1344). It is the fallback only: it includes the prefill's expert reads,
    so joining it to decode-step waits is a cross-regime join.

ALIGNMENT: the two producers do not carry the same step index. DECPROF prints on a cadence
(`dp_step % 8`, step 1, or any batched graph) and resets each step; BATCHDBG prints only for
miss-bearing layers, so a step cannot be located by layer wrap alone (and the run ends mid-step).
So neither "the last block" nor "the last step" is a safe pairing. Instead both sides are
aggregated over every COMPLETE step: mean misses per step per layer, median wait per step per
layer. Both are then steady-state per-layer distributions, which is what the rule below is about.

DECISION RULE -- FROZEN BEFORE THE RUN (2026-09-25, before the arm was launched)
-------------------------------------------------------------------------------
Let `W_miss` = sum of per-layer median wait over layers whose mean misses/step > 0,
`W_clean` = the same over layers with zero mean misses, and rho = Spearman rank correlation
between per-layer mean misses/step and per-layer median wait.

  R1 HOLDS (wait is miss-bound; shape 3 -- reduce misses -- directly pays):
      W_miss / (W_miss + W_clean) >= 0.60  AND  rho >= 0.50
  R2 FAILS (wait is round-trip-bound; only overlap/shape 2 pays):
      W_miss / (W_miss + W_clean) <  0.60  OR   rho <  0.50
  INVALID (no verdict): fewer than 3 complete miss steps, or no CGC-DECPROF block, or no layer in
      common.  A run without a usable miss axis cannot answer the question -- reporting R1 there
      would be the absence-as-value bug.

Both quantities are inflated together by swap pressure, so the *split* is the signal and the
absolute ms/step is not (this is a diagnostic arm, never a t/s number).

Selftest: `python3 scripts/check/s1_wait_budget.py --selftest` (7 fixtures, including one where the
verdict must be INVALID and two where malformed lines must be counted rather than ignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HDR_RE = re.compile(
    r"CGC-DECPROF: step=(?P<step>-?\d+) segs=(?P<segs>\d+) layers=(?P<layers>\d+) "
    r"total=(?P<total>[-\d.]+) ms \| wait=(?P<wait>[-\d.]+) \((?P<waitpc>\d+)%\) "
    r"cb=(?P<cb>[-\d.]+) \((?P<cbpc>\d+)%\) submit=(?P<sub>[-\d.]+) \([-\d.]+%\) "
    r"ntok=(?P<ntok>-?\d+)")
ALL_RE = re.compile(
    r"CGC-DECPROF all: L(?P<layer>\d+) wait=(?P<wait>[-\d.]+) cb=(?P<cb>[-\d.]+) "
    r"submit=(?P<submit>[-\d.]+) ms gpu=(?P<gpu>[-\d.]+) union=(?P<union>[-\d.]+) "
    r"gap=(?P<gap>[-\d.]+) sg=(?P<sg>\d+) n=(?P<n>\d+)")
FINAL_RE = re.compile(
    r"final stats: runtime requests=(?P<req>\d+) hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"\(hit rate (?P<rate>[-\d.]+)%\)")
BATCH_RE = re.compile(r"BATCHDBG layer=(?P<layer>\d+) misses=(?P<misses>\d+) slots:")

W_MISS_MIN = 0.60       # frozen threshold R1
RHO_MIN = 0.50          # frozen threshold R1
MIN_STEP_LAYERS = 5     # a "step" of the miss axis with fewer rows than this is a fragment
MIN_STEPS = 3           # needed for a mean worth regressing on


def parse_blocks(stderr: str) -> tuple[list[dict], int]:
    """Return (complete blocks, n_unparsed_all_lines).

    A block is a header followed by its `CGC-DECPROF all:` rows.  An `all:` row that does not match
    the pattern is counted, never dropped quietly -- a silently-shrunk table would shrink the
    denominator of every share below.
    """
    blocks: list[dict] = []
    unparsed = 0
    cur: dict | None = None
    for line in stderr.splitlines():
        if (m := HDR_RE.search(line)):
            cur = {"header": m.groupdict(), "layers": {}}
            blocks.append(cur)
            continue
        if "CGC-DECPROF all:" in line:
            if cur is None:
                unparsed += 1
                continue
            if (m := ALL_RE.search(line)):
                cur["layers"][int(m.group("layer"))] = {
                    **{k: float(m.group(k)) for k in ("wait", "cb", "submit", "gpu", "union", "gap")},
                    "n": int(m.group("n"))}
            else:
                unparsed += 1
    return [b for b in blocks if b["layers"]], unparsed


def parse_misses(text: str) -> tuple[dict[int, int], int, int]:
    """`<layer> <expert>` per line -> (per-layer counts, total, n_malformed)."""
    per: dict[int, int] = {}
    total = 0
    bad = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            if line.strip():
                bad += 1
            continue
        try:
            layer = int(parts[0])
        except ValueError:
            bad += 1
            continue
        per[layer] = per.get(layer, 0) + 1
        total += 1
    return per, total, bad


def parse_batchdbg(text: str) -> tuple[list[dict[int, int]], int]:
    """Per-layer misses grouped into steps, plus the number of BATCHDBG lines that matched nothing.

    A step boundary is the layer index going BACKWARDS.  Because the producer prints only for
    miss-bearing layers, a group can be short; it can never be *wrong* about a layer it does print,
    and a step that printed nothing simply has no misses.
    """
    steps: list[dict[int, int]] = []
    bad = 0
    cur: dict[int, int] | None = None
    prev = -1
    for line in text.splitlines():
        if "BATCHDBG" not in line:
            continue
        m = BATCH_RE.search(line)
        if not m:
            bad += 1
            continue
        layer = int(m.group("layer"))
        if cur is None or layer <= prev:
            cur = {}
            steps.append(cur)
        cur[layer] = int(m.group("misses"))
        prev = layer
    return steps, bad


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation with average ranks for ties; None when undefined (n<3 or no variance)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def apply_rule(miss_per_step: dict[int, float], wait_ms: dict[int, float], note: str) -> dict:
    """The frozen rule, applied to two per-layer vectors (misses/step, median wait)."""
    layers = sorted(set(miss_per_step) | set(wait_ms))
    if len(layers) < 3:
        return {"verdict": "INVALID", "why": f"only {len(layers)} layers in common"}
    if sum(1 for v in miss_per_step.values() if v > 0) == 0:
        return {"verdict": "INVALID",
                "why": "no layer has a miss: a wait split cannot be attributed (absence != 0)"}
    w_miss = sum(wait_ms.get(l, 0.0) for l in layers if miss_per_step.get(l, 0.0) > 0)
    w_clean = sum(wait_ms.get(l, 0.0) for l in layers if miss_per_step.get(l, 0.0) <= 0)
    denom = w_miss + w_clean
    share = (w_miss / denom) if denom > 0 else None
    rho = spearman([miss_per_step.get(l, 0.0) for l in layers],
                   [wait_ms.get(l, 0.0) for l in layers])
    holds = (share is not None and share >= W_MISS_MIN and rho is not None and rho >= RHO_MIN)
    return {
        "verdict": "R1_HOLDS_miss_bound" if holds else "R2_FAILS_roundtrip_bound",
        "why": (f"{note} wait share on miss-bearing layers="
                f"{None if share is None else round(share, 3)} (>= {W_MISS_MIN}), "
                f"rho={None if rho is None else round(rho, 3)} (>= {RHO_MIN})"),
        "w_miss_ms": round(w_miss, 2),
        "w_clean_ms": round(w_clean, 2),
        "wait_share_miss_layers": None if share is None else round(share, 3),
        "rho_wait_vs_misses": None if rho is None else round(rho, 3),
        "n_miss_layers": sum(1 for l in layers if miss_per_step.get(l, 0.0) > 0),
        "n_clean_layers": sum(1 for l in layers if miss_per_step.get(l, 0.0) <= 0),
    }


def analyse(stderr: str, miss_text: str = "", column: str = "wait") -> dict:
    """`column` is the dependent variable: 'wait' (pre-registered) or 'cb' (the hook, which is
    where the fill itself is spent -- reported alongside, never substituted for the frozen one)."""
    blocks, unparsed = parse_blocks(stderr)
    per_layer_miss, total_miss, bad_miss = parse_misses(miss_text)
    steps = [s for s in (st for st in parse_batchdbg(stderr)[0]) if len(s) >= MIN_STEP_LAYERS]
    bad_batch = parse_batchdbg(stderr)[1]
    fin = FINAL_RE.search(stderr)
    out: dict = {
        "n_blocks": len(blocks),
        "n_unparsed_decprof_lines": unparsed,
        "n_malformed_batchdbg_lines": bad_batch,
        "n_miss_steps": len(steps),
        "miss_total_run": total_miss,
        "n_malformed_miss_lines": bad_miss,
        "miss_source": None,
        "pool": ({k: (int(v) if k in ("req", "hits", "misses") else float(v))
                  for k, v in fin.groupdict().items()} if fin else None),
    }
    if not blocks:
        out["verdict"] = {"verdict": "INVALID", "why": "no CGC-DECPROF block in this stderr"}
        return out

    # per-layer cost: median across the printed steps (each block is one step, accumulators reset)
    wait_med: dict[int, float] = {}
    for l in {l for b in blocks for l in b["layers"]}:
        wait_med[l] = _median([b["layers"][l][column] for b in blocks if l in b["layers"]])

    if len(steps) >= MIN_STEPS:
        miss_per_step = {l: _mean([float(s.get(l, 0)) for s in steps])
                         for l in miss_steps_layers(steps)}
        out["miss_source"] = (f"BATCHDBG: mean over {len(steps)} complete steps "
                              f"(run total {total_miss} misses)")
        out["verdict"] = apply_rule(miss_per_step, wait_med, out["miss_source"])
    elif per_layer_miss:
        miss_per_step = {l: float(v) for l, v in per_layer_miss.items()}
        out["miss_source"] = (f"run-total dump FALLBACK ({len(steps)} usable steps < {MIN_STEPS}; "
                              f"includes prefill reads)")
        out["verdict"] = apply_rule(miss_per_step, wait_med, out["miss_source"])
    else:
        out["verdict"] = {"verdict": "INVALID",
                          "why": f"no usable miss axis: {len(steps)} complete steps, empty dump"}
        return out

    out["column"] = column
    out["per_layer"] = [
        {"layer": l,
         "misses_per_step": round(miss_per_step.get(l, 0.0), 4),
         f"{column}_med_ms": round(wait_med.get(l, 0.0), 3),
         "cost_share": round(wait_med.get(l, 0.0) / (sum(wait_med.values()) or 1), 4),
         "miss_share": round(miss_per_step.get(l, 0.0) / (sum(miss_per_step.values()) or 1), 4)}
        for l in sorted(wait_med)]
    return out


def miss_steps_layers(steps: list[dict[int, int]]) -> set[int]:
    return {l for s in steps for l in s}


# ── fixtures: the rule must be able to say both things, and to refuse ────────────────

def _blk(rows: list[tuple[int, float, float]], step: int = 16) -> str:
    """rows = (layer, wait_ms, cb_ms)."""
    head = (f"CGC-DECPROF: step={step} segs=41 layers=40 total=100.00 ms | wait=50.00 (50%) "
            "cb=40.00 (40%) submit=10.00 (10%) ntok=1 | layer gpu_sum=1.00 union_sum=1.00 "
            "gap_sum=1.00 ms\n")
    body = "".join(
        f"CGC-DECPROF all: L{l} wait={w} cb={c} submit=0.10 ms gpu=0.5 union=0.5 gap=0.5 "
        f"sg=1 n=19 st=0.000 en=0.500\n" for l, w, c in rows)
    return head + body + ("llama_expert_cache: final stats: runtime requests=5720 hits=5600 "
                          "misses=120 (hit rate 97.9%)  prewarm req=0 hit=0 miss=0  resident=1 MiB "
                          "file_reads=100 pread_usec=1 fill_batch_usec=1 fill_wait_us=1 prefetch=0/0\n")


def _steps(n_steps: int, misses: dict[int, int]) -> str:
    return "".join(
        "".join(f"BATCHDBG layer={l} misses={m} slots: e1->s0\n" for l, m in misses.items())
        for _ in range(n_steps))


def selftest() -> int:
    fails = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra and not cond else ''}")
        if not cond:
            fails += 1

    high_wait = [(l, 20.0 if l < 24 else 1.0, 5.0) for l in range(40)]   # 24 layers own the wait

    # (a) miss-bound: the same layers that own the wait are the ones with misses
    a = analyse(_blk(high_wait) + _steps(4, {l: 2 for l in range(24)}))
    check("miss-bound fixture -> R1", a["verdict"]["verdict"] == "R1_HOLDS_miss_bound",
          str(a["verdict"]))
    check("  mean misses/step is a mean over steps, not a sum",
          abs(a["per_layer"][0]["misses_per_step"] - 2.0) < 1e-9, str(a["per_layer"][0]))

    # (b) round-trip-bound: wait is uniform, misses sit in a few layers -> R2
    b = analyse(_blk([(l, 5.0, 1.0) for l in range(40)]) + _steps(4, {l: 2 for l in range(8)}))
    check("uniform-wait fixture -> R2", b["verdict"]["verdict"] == "R2_FAILS_roundtrip_bound",
          str(b["verdict"]))

    # (c) no misses at all: unanswerable -> INVALID, never a pass
    c = analyse(_blk(high_wait))
    check("no miss axis at all -> INVALID (absence is not a value)",
          c["verdict"]["verdict"] == "INVALID", str(c["verdict"]))

    # (c2) only 2 usable steps: not enough for a mean -> must fall back or refuse, never claim R1
    c2 = analyse(_blk(high_wait) + _steps(2, {l: 2 for l in range(24)}))
    check("2 steps and no dump -> INVALID, never R1 on a thin mean",
          c2["verdict"]["verdict"] == "INVALID", str(c2["verdict"]))
    c3 = analyse(_blk(high_wait) + _steps(2, {l: 2 for l in range(24)}),
                 "\n".join(f"{l} 3" for l in range(24)))
    check("2 steps + dump -> FALLBACK path, labelled as such",
          (c3["miss_source"] or "").startswith("run-total dump FALLBACK"), str(c3["miss_source"]))

    # (d) malformed producers must be counted, not ignored
    d_bad = _blk([(l, 5.0, 1.0) for l in range(6)]).replace("CGC-DECPROF all: L3",
                                                            "CGC-DECPROF all: L3 (TRUNCATED)")
    d = analyse(d_bad + _steps(4, {l: 1 for l in range(6)}) + "BATCHDBG layer=x misses=oops\n",
                "0 1\ngarbage\n")
    check("truncated DECPROF line counted", d["n_unparsed_decprof_lines"] == 1,
          str(d["n_unparsed_decprof_lines"]))
    check("malformed BATCHDBG line counted", d["n_malformed_batchdbg_lines"] == 1,
          str(d["n_malformed_batchdbg_lines"]))
    check("malformed miss-dump line counted", d["n_malformed_miss_lines"] == 1,
          str(d["n_malformed_miss_lines"]))

    # (e) a trailing fragment must not become a step
    e = analyse(_blk(high_wait) + _steps(4, {l: 2 for l in range(24)}) +
                "".join(f"BATCHDBG layer={l} misses=99 slots: e1->s0\n" for l in range(3)))
    check("trailing fragment is not a step", e["n_miss_steps"] == 4, str(e["n_miss_steps"]))
    check("  and it does not leak into the mean", abs(e["per_layer"][0]["misses_per_step"] - 2.0) < 1e-9,
          str(e["per_layer"][0]))

    print(f"\nselftest {'OK' if fails == 0 else f'FAILED ({fails})'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stderr", help="llama-bench stderr log (CGC-DECPROF + BATCHDBG)")
    ap.add_argument("--miss-dump", default="", help="LLAMA_EXPERT_CACHE_MISS_DUMP file (fallback)")
    ap.add_argument("--json", help="write the analysis here")
    ap.add_argument("--column", default="wait", choices=("wait", "cb", "submit", "gpu", "union",
                                                        "gap"),
                    help="dependent variable; 'wait' is the pre-registered one, 'cb' carries the "
                         "hook/fill cost")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.stderr:
        ap.error("--stderr is required (or --selftest)")
    miss_text = Path(args.miss_dump).read_text(errors="replace") if args.miss_dump else ""
    res = analyse(Path(args.stderr).read_text(errors="replace"), miss_text, args.column)
    v = res["verdict"]
    print(f"column={res['column']}  blocks={res['n_blocks']} unparsed={res['n_unparsed_decprof_lines']}"
          f"  miss steps={res['n_miss_steps']} (run misses={res['miss_total_run']})")
    print(f"miss source: {res['miss_source']}\npool: {res['pool']}")
    print(f"\nVERDICT: {v['verdict']}\n  {v['why']}")
    if "w_miss_ms" in v:
        print(f"  median {res['column']} summed: miss-bearing layers = {v['w_miss_ms']} ms, clean = "
              f"{v['w_clean_ms']} ms  ({v['n_miss_layers']} vs {v['n_clean_layers']} layers)")
        if v["n_clean_layers"] == 0:
            print("  ⚠ every layer misses: the miss-bearing/clean split is degenerate, so the "
                  "verdict rests on rho alone")
    print(f"\npre-registered rule: R1 iff wait-share(miss layers) >= {W_MISS_MIN} and rho >= "
          f"{RHO_MIN}; no usable miss axis => INVALID")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
