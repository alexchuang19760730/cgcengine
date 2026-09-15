#!/usr/bin/env python3
"""Is prefill 250 tok/s a spec, or a lottery?

THE QUESTION
------------
The 250 target has been *reached* repeatedly: 254.74 and 264.78 tok/s in `prefill250`
(`Backup/cgc_logs/llama_server_20260915_011146.log`), peak 276.59. And it has also been
missed by the same command: `pp2048 @ -ub 6144` over four independent launches gave

    276.59 / 198.84 / 176.18 / 122.68  t/s   ->  2.25x spread

while the three samples *inside* each launch agreed to 2.08-15.41 stddev. So the spread is
NOT measurement noise: it is state that differs between processes. A number you can hit
2.25x of the time is not a spec, it is a draw. Nothing downstream can be built on it.

THE HYPOTHESIS THIS TESTS
-------------------------
The candidate mechanism is memory state at launch: 16 GB unified memory has to hold
13.66 GiB of model + an 8 GiB expert pool + the compute buffer for a 6144-wide ubatch, and
`run_server.sh:427-434` already documents that +2 GiB of pool is enough to push the 6144
compute buffer out of the GPU entirely ("command buffer failed with status 5 / Insufficient
Memory ... 2026-09-15 用 CGC_SERVER_PROFILE=prefill250 實跑就是這樣炸的"). So the shape is
over-subscribed, and what the draw actually selects is *how much of the working set is
resident* -- i.e. how much of the run is served from page cache vs. from the 1404 MiB/s cold
device.

If that is the mechanism, throughput should track free memory at launch **monotonically**.
If it does, the fix is to make free memory a precondition (a gate) and 250 becomes a spec.
If it does not, the draw is something else and we should stop guessing.

WHAT IT MEASURES
----------------
N independent processes, same resolved command every time (the env is resolved by
`run_server.sh CGC_DUMP_ENV=1` through `llama_bench_matrix.py`, never re-typed here), with
the launch-time memory state recorded for each. Reported per run: free%, pp t/s, stddev, wall.

VERDICT
-------
    spread < 5%            -> certifiable; the number is a spec
    5% <= spread, monotone -> gated spec; ship the precondition, not the number
    spread >= 5%, not monotone -> still a draw; the mechanism is not launch memory

Usage:
    /opt/homebrew/bin/python3 scripts/check/prefill_certifiability.py --runs 5
    /opt/homebrew/bin/python3 scripts/check/prefill_certifiability.py --runs 5 --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "scripts" / "check" / "llama_bench_matrix.py"
PY = "/opt/homebrew/bin/python3"
TOTAL_BYTES = 16 * 1024**3


def mem_state() -> dict:
    """Launch-time memory state.

    NOTE ON WHICH VARIABLE MATTERS -- this is a correction, not a detail.

    The first version of this script recorded `free + inactive` as "usable" and called that the
    candidate mechanism. Five runs falsified that in the *inverted* direction (rho = -0.70): the
    run with the LEAST free memory was the ONLY one to clear 250 t/s. The reason is that free
    memory is not a proxy for the thing the engine actually depends on. The engine streams a
    13.66 GiB model off a 1404 MiB/s cold device; what matters is how much of that file is RESIDENT
    in the page cache (7721 MiB/s hot). And `Pages free` cannot distinguish "small cache" from
    "another process holds it" -- it moves with both. Measured right now on this box:
    free = 5.94 GiB but File-backed (the actual cache) = only 3.33 GiB, with 3.82 GiB anonymous.
    Low free can mean a LARGE cache (MacOS fills spare RAM with cache) or a LARGE app footprint.
    Those predict opposite throughputs, and the metric averaged them.

    So this records the page cache explicitly. `cached_gib` is the leading candidate; free/inactive
    are kept so the two can be compared on the same runs.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    d = {k.strip(): int(v) for k, v in re.findall(r"([A-Za-z\- ]+):\s+(\d+)\.", out)}
    free = d.get("Pages free", 0) * page
    inactive = d.get("Pages inactive", 0) * page
    speculative = d.get("Pages speculative", 0) * page
    cached = d.get("File-backed pages", 0) * page
    anon = d.get("Anonymous pages", 0) * page
    swap = 0
    try:
        so = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", so)
        if m:
            swap = float(m.group(1)) * 1024**2
    except Exception:
        pass
    return {
        "free_gib": free / 1024**3,
        "inactive_gib": inactive / 1024**3,
        "speculative_gib": speculative / 1024**3,
        "usable_gib": (free + inactive + speculative) / 1024**3,
        "usable_pct": 100.0 * (free + inactive + speculative) / TOTAL_BYTES,
        "cached_gib": cached / 1024**3,      # File-backed pages: the page cache. Leading candidate.
        "anon_gib": anon / 1024**3,          # Anonymous pages: what other processes hold.
        "swap_used_mib": swap / 1024**2,
    }


SEG_PATTERNS = {
    # from `llama_expert_cache: final stats: ...`
    "resident_mib":    r"resident=([\d.]+) MiB",
    "file_reads":      r"file_reads=(\d+)",
    "pread_usec":      r"pread_usec=(\d+)",
    "fill_batch_usec": r"fill_batch_usec=(\d+)",
    "fill_wait_us":    r"fill_wait_us=(\d+)",
    "req_hit_rate":    r"hit rate ([\d.]+)%",
    # from `slab fills:` -- the M2 exit condition, and the first thing to check
    "slab_pool_mib":   r"slab fills: pool=([\d.]+) MiB",
    "slab_disk_mib":   r"disk=([\d.]+) MiB",
    "nonresident_pct": r"non-resident share ([\d.]+)%",
    # from `read shape:`
    "read_jobs":       r"read shape: jobs=(\d+)",
    "read_bytes":      r"read shape: jobs=\d+ bytes=(\d+)",
    "us_per_job":      r"us/job=([\d.]+)",
    # from `pool integrity:` and the two failure gates
    "owner_slots":     r"owner-set slots=(\d+)",
    "zero_regions":    r"zero-regions=(\d+)",
    "verify_refused":  r"verify-strict: refused=(\d+)",
    "zero_mapped":     r"zero_mapped_selected=(\d+)",
    # from `miss attribution:`
    "miss_compulsory": r"compulsory=(\d+)",
    "miss_capacity":   r"capacity=(\d+)",
    "evictions":       r"evictions=(\d+)",
}


def harvest_segments(stderr_text: str) -> dict:
    """Mine the always-on teardown counters out of one launch's stderr.

    These lines are emitted by the engine at every `~llama_expert_cache`, with no env gate, so
    the numbers are directly comparable to a run that was profiled with no instrumentation at
    all -- which is the only reason a high run and a low run can be differenced. `CGC-SEG` is
    summed here as (wait, cb, submit) totals over its per-segment lines.

    Two fields on these lines are NOT trustworthy as an I/O measure, and are harvested anyway so
    the defect stays visible rather than being silently quoted:
      - `read_bytes` / `us_per_job`: `n_read_bytes` is incremented on the pool path only
        (llama-expert-cache.cpp:2657,2671), never in `fill_job` (:41), so a run whose fills all
        take the concurrent-segment path reports bytes=0 and an `effective_rate` of ~1 MiB/s
        against 178 GiB actually read. Use `pread_usec` + `file_reads` instead.
      - `pread_usec` is an AGGREGATE over worker threads (llama-expert-cache.h:463,473) and can
        exceed wall time by the worker count; only `fill_wait_us` is on the calling thread.
    The whole-layer slab `pread` (llama-expert-cache.cpp:3351) increments NOTHING, so
    `file_reads`/`pread_usec` under-count prefill I/O by an amount that is not currently known.
    """
    seg: dict = {}
    for key, pat in SEG_PATTERNS.items():
        m = None
        for m in re.finditer(pat, stderr_text):
            pass
        if m is not None:
            seg[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    # CGC-SEG prints CUMULATIVE AVERAGES (`w_us/n`), so the last line times n is the total and
    # summing the printed values is meaningless -- doing that inflated a 30 s wait to 0.54 s on
    # the first version of this harvester. Take the last line only.
    m = None
    for m in re.finditer(r"CGC-SEG: wait ([\d.]+) cb ([\d.]+) submit ([\d.]+) us \((\d+)\)",
                         stderr_text):
        pass
    if m is not None:
        w, c, s, n_seg = (float(m.group(1)), float(m.group(2)), float(m.group(3)),
                          int(m.group(4)))
        seg.update({"seg_n": n_seg,
                    "seg_wait_s": w * n_seg / 1e6,      # GPU-bound: spin until the cmd buffers land
                    "seg_cb_s": c * n_seg / 1e6,        # CPU: encode
                    "seg_submit_s": s * n_seg / 1e6,    # CPU: commit
                    "seg_wait_avg_ms": w / 1e3})
    seg["prefill_stream_fills"] = len(re.findall(r"CGC-PREFILL-STREAM: il=\d+ kind=\d+ ntok=",
                                                 stderr_text))
    return seg


def harvest_gputime(stderr_text: str) -> dict:
    """Mine `CGC-GPUTIME` (needs CGC_GPU_TIMING=1; gate: ggml-backend.cpp:1844).

    One line == ONE graph_compute (the accumulators reset per graph, :2143). `wait` is wall time
    the CPU spent spinning for that graph's command buffers; `gpu_union` is the union of the GPU's
    own start/end intervals -- how much of that wait the GPU was genuinely busy. The engine states
    its own decision rule at :2115-2122: union/wait >= 70% means the wait IS GPU execution (batch
    the per-expert GEMVs); <= 40% means launch/completion latency (remove the GPU->CPU->GPU round
    trip at the segment boundary). `skipped` must stay 0 or Metal reported no timestamps and the
    whole line means nothing.
    """
    rows = []
    for m in re.finditer(
            r"CGC-GPUTIME: step=(\d+) segs=(\d+) bufs=(\d+) skipped=(\d+) "
            r"wait=([\d.]+) gpu_busy_sum=([\d.]+) \(([\d.]+)%\) "
            r"gpu_union=([\d.]+) \(([\d.]+)%\) gap=([\d.]+) \(([\d.]+)%\) ms", stderr_text):
        rows.append(dict(step=int(m.group(1)), segs=int(m.group(2)), bufs=int(m.group(3)),
                         skipped=int(m.group(4)), wait_ms=float(m.group(5)),
                         busy_ms=float(m.group(6)), union_ms=float(m.group(8)),
                         union_pct=float(m.group(9)), gap_ms=float(m.group(10))))
    if not rows:
        return {}
    big = max(rows, key=lambda r: r["wait_ms"])        # the prefill graph is the big one
    tw = sum(r["wait_ms"] for r in rows)
    tu = sum(r["union_ms"] for r in rows)
    return {"gpu_lines": len(rows),
            "gpu_skipped": sum(r["skipped"] for r in rows),
            "gpu_wait_ms": tw, "gpu_union_ms": tu,
            "gpu_union_pct": 100.0 * tu / tw if tw else 0.0,
            "gpu_gap_ms": sum(r["gap_ms"] for r in rows),
            "ppg_wait_ms": big["wait_ms"], "ppg_union_ms": big["union_ms"],
            "ppg_union_pct": big["union_pct"], "ppg_gap_ms": big["gap_ms"],
            "ppg_segs": big["segs"], "ppg_bufs": big["bufs"]}


DECPROF_SUM_RE = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \((\d+)%\) cb=([\d.]+) \((\d+)%\) submit=([\d.]+) \((\d+)%\)"
    r"(?: ntok=(\d+))?")
DECPROF_LAY_RE = re.compile(
    r"CGC-DECPROF (top\d+|all): L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms n=(\d+)")


def harvest_decprof(stderr_text: str) -> dict:
    """Mine `CGC-DECPROF` -- the per-layer wait/cb/submit split. Needs `CGC_DECODE_PROFILE=1`
    AND the segmented dispatcher (`CGC_OA_ASYNC != 0`), i.e. `arm prefill250-decprof`.

    Why this harvester exists at all: until 2026-09-16 the instrument could not see a prefill.
    Its print gate was `(dp_step % 8) == 0` (`ggml-backend.cpp:2061`) and a 2048-token prefill at
    `-ub 6144` is a single `graph_compute`, so `dp_step` was 1 and the per-layer accumulators were
    zeroed long before step 8 was reached. 95 preserved logs, every one of them with a minimum
    printed `step` of 8, is the fingerprint of that (lesson `eng-src-0011`). The gate now also
    fires on the first graph of the process and on any graph whose top-k tensor reports
    `n_tokens > 1`, and every line carries `ntok=` -- so **this harvester keys off `ntok`, never off
    `step`**: a graph is a prefill because it says it is, not because it came first.

    What the three components mean, per layer:
      - `wait_ms`  -- the CPU spinning until that layer's command buffers complete. This is the
                      bucket that dominates prefill, and it is the same quantity `CGC-GPUTIME`
                      reports as `gpu_union/wait` for the whole graph.
      - `cb_ms`    -- the top-k hook: slot management plus any *blocking* fill, charged to the
                      layer whose segment was just consumed. This is where a cold expert fill
                      lands, and it is why the first prefill graph costs more than the second.
      - `submit_ms`-- committing that layer's segment to Metal.
    Uniformity of `wait_ms` across layers is the discriminator the report asks for: a tight spread
    means every layer slowed by the same factor (a clock/power ceiling), a single dominant layer
    means one implementation is the cost.

    One parsing rule that is load-bearing: `top<N>:` and `all:` lines describe the SAME layers --
    `top8` is a subset of `all`. Both are collected into one dict keyed by layer index, never a
    list, because appending them double-counts every layer (40 `all` + 8 `top` = 48 "layers" on a
    40-layer model) and inflates `max` and the coefficient of variation, which is exactly the
    quantity the uniformity verdict is computed from. Measured 2026-09-16, before the fix.
    """
    graphs: list[dict] = []
    cur: dict | None = None
    for line in stderr_text.splitlines():
        m = DECPROF_SUM_RE.search(line)
        if m is not None:
            cur = {"step": int(m.group(1)), "segs": int(m.group(2)), "layers": int(m.group(3)),
                   "total_ms": float(m.group(4)), "wait_ms": float(m.group(5)),
                   "cb_ms": float(m.group(7)), "submit_ms": float(m.group(9)),
                   "ntok": int(m.group(11)) if m.group(11) is not None else None,
                   "lay": {}}
            graphs.append(cur)
            continue
        m = DECPROF_LAY_RE.search(line)
        if m is not None and cur is not None:
            # keyed by layer, so a `topN` line for a layer the `all:` pass already reported just
            # rewrites it with identical numbers instead of counting it twice
            cur["lay"][int(m.group(2))] = [float(m.group(3)), float(m.group(4)),
                                           float(m.group(5)), int(m.group(6))]
    for g in graphs:
        g["lay"] = [[l, *g["lay"][l]] for l in sorted(g["lay"])]
    if not graphs:
        return {}

    pre = [g for g in graphs if (g["ntok"] or 0) > 1]
    dec = [g for g in graphs if (g["ntok"] or 0) <= 1]
    out: dict = {"dp_graphs": len(graphs), "dp_prefill_graphs": len(pre),
                 "dp_decode_graphs": len(dec),
                 # the per-graph records themselves: the per-layer wait series is what says whether
                 # a slowdown was uniform or localised, and it cannot be reconstructed from the
                 # summary keys alone
                 "dp_series": graphs,
                 "dp_ntok_max": max((g["ntok"] or 0) for g in graphs),
                 "dp_prefill_s": round(sum(g["total_ms"] for g in pre) / 1000.0, 3),
                 "dp_decode_s": round(sum(g["total_ms"] for g in dec) / 1000.0, 3),
                 "dp_ntok_none": sum(1 for g in graphs if g["ntok"] is None)}

    # The cold/hot pair the report needs: the FIRST prefill graph of the process (which also pays
    # the cold expert fill) and the SECOND (same shape, fill already warm). Also flatten each, so a
    # diff between two launches is a key-by-key comparison rather than a re-parse.
    for q, g in (("dpq1", pre[0] if pre else None), ("dpq2", pre[1] if len(pre) > 1 else None)):
        if g is None:
            continue
        ws = [v[1] for v in g["lay"]]
        cbs = [v[2] for v in g["lay"]]
        n = len(ws)
        mean = sum(ws) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in ws) / n if n else 0.0
        srt = sorted(ws)
        med = (srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])) if n else 0.0
        p25 = srt[n // 4] if n else 0.0
        p75 = srt[(3 * n) // 4] if n else 0.0
        within = sum(1 for x in ws if med and abs(x - med) <= 0.25 * med)
        top = max(g["lay"], key=lambda v: v[1]) if g["lay"] else None
        out.update({
            f"{q}_step": g["step"], f"{q}_ntok": g["ntok"], f"{q}_segs": g["segs"],
            f"{q}_layers_seen": n, f"{q}_total_ms": g["total_ms"],
            f"{q}_wait_ms": g["wait_ms"], f"{q}_cb_ms": g["cb_ms"],
            f"{q}_submit_ms": g["submit_ms"],
            f"{q}_lay_w_mean_ms": round(mean, 3),
            f"{q}_lay_w_median_ms": round(med, 3),
            f"{q}_lay_w_min_ms": round(min(ws), 3) if ws else 0.0,
            f"{q}_lay_w_max_ms": round(max(ws), 3) if ws else 0.0,
            f"{q}_lay_w_cv_pct": round(100.0 * (var ** 0.5) / mean, 2) if mean else 0.0,
            # Robust uniformity, because the raw spread is dominated by a PIPELINE RAMP, not by a
            # slow layer: with `submit_ahead`, layer i's wait is what layer i's segment had to wait
            # for, so the first handful of layers have less GPU backlog to wait on and report less.
            # In the fast 285 t/s probe L0 was ALSO the minimum (112 ms against a 155 ms plateau), so
            # the ramp is structural and says nothing about the state. `peak/mean` conflates the two
            # and mislabels a uniformly-scaled graph as localised; IQR/median and the fraction of
            # layers within +/-25% of the median do not.
            f"{q}_lay_w_iqr_over_median": round((p75 - p25) / med, 3) if med else 0.0,
            f"{q}_lay_w_frac_within_25pct": round(within / n, 3) if n else 0.0,
            f"{q}_lay_w_argmax_layer": top[0] if top else None,
            f"{q}_lay_w_sum_ms": round(sum(ws), 3),
            f"{q}_lay_cb_sum_ms": round(sum(cbs), 3),
            f"{q}_lay_w_top_layer": top[0] if top else None,
            f"{q}_lay_w_top_ms": top[1] if top else None,
        })
    if out.get("dpq1_lay_w_max_ms") and out.get("dpq1_lay_w_mean_ms"):
        # ratio of the slowest layer to the mean. < ~1.3 => the graph slowed as a whole.
        out["dpq1_lay_w_peak_over_mean"] = round(
            out["dpq1_lay_w_max_ms"] / out["dpq1_lay_w_mean_ms"], 3)
    if out.get("dpq2_lay_w_mean_ms") and out.get("dpq1_lay_w_mean_ms"):
        out["dpq1_over_dpq2_lay_w"] = round(
            out["dpq1_lay_w_mean_ms"] / out["dpq2_lay_w_mean_ms"], 3)
    return out


def pp_from(json_path: Path) -> list[dict]:
    """Pull the pp row(s) out of the matrix harness summary json."""
    if not json_path.exists():
        return []
    try:
        res = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return []
    rows = []
    for arm in res:
        for r in arm.get("rows", []):
            if r.get("n_prompt", 0) > 0:
                rows.append({"avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"],
                             "n_prompt": r["n_prompt"], "n_depth": r["n_depth"],
                             "n_batch": r.get("n_batch"), "incomplete": arm.get("incomplete"),
                             "error": arm.get("error")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5, help="independent processes (default 5)")
    ap.add_argument("--arm", default="prefill250")
    ap.add_argument("--prompt", default="2048", help="llama-bench -p")
    ap.add_argument("--gen", default="16", help="llama-bench -n (pp rows are read; tg is incidental)")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench -r (within-process samples)")
    ap.add_argument("--depths", default="0")
    ap.add_argument("--min-usable-pct", type=float, default=30.0,
                    help="refuse to launch below this free+inactive+speculative floor")
    ap.add_argument("--warm-runs", default="",
                    help="comma list of run indices to PRE-READ the model file before "
                         "(forces the page cache up). This is what breaks the order confound: "
                         "plain runs move cache and free together, warmed runs move cache up "
                         "while free falls.")
    ap.add_argument("--warm-file",
                    default=str(ROOT / "models" / "gguf"
                                / "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"))
    ap.add_argument("--logdir",
                    default=str(ROOT / "Backup" / "llama_bench" / "cert_runs"),
                    help="one subdirectory per launch, so every launch keeps its own "
                         "llama_bench_<arm>.stderr.log (the matrix names the file after the ARM)")
    ap.add_argument("--idle-before", type=int, default=0,
                    help="sleep this many seconds before the FIRST launch, to test whether the "
                         "fast state is the cold state (first launch after an idle period)")
    ap.add_argument("--json")
    args = ap.parse_args()

    # A relative `--logdir` breaks the reporting step: the per-launch log path is then relative
    # while ROOT is absolute, and `segpath.relative_to(ROOT)` raises ValueError -- which crashed the
    # harness *after* the measurement but *before* the summary, so the run's numbers were lost to a
    # cosmetic lookup (measured 2026-09-16: a 285.58 t/s probe died on exactly this line). Resolve
    # both paths once, here, so every later use is absolute by construction.
    logdir = Path(args.logdir)
    if not logdir.is_absolute():
        logdir = ROOT / logdir
    args.logdir = str(logdir)
    if args.json:
        jp = Path(args.json)
        args.json = str(jp if jp.is_absolute() else ROOT / jp)

    warm_set = {int(x) for x in args.warm_runs.split(",") if x.strip()}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.idle_before:
        print(f"  [idle] sleeping {args.idle_before}s before the first launch "
              f"(cold-state probe)", flush=True)
        time.sleep(args.idle_before)
    runs: list[dict] = []
    for i in range(1, args.runs + 1):
        warm_s = None
        if i in warm_set:
            t = time.time()
            try:
                with open(args.warm_file, "rb") as f:
                    n = 0
                    while True:
                        b = f.read(4 << 20)
                        if not b:
                            break
                        n += len(b)
                warm_s = round(time.time() - t, 1)
                print(f"\n  [warm] read {n/1024**3:.2f} GiB of {Path(args.warm_file).name} "
                      f"in {warm_s}s to force the page cache up", flush=True)
            except OSError as e:
                print(f"  [warm] FAILED: {e}", flush=True)
        # Recorded AFTER warming: the post-warm state IS the launch state.
        pre = mem_state()
        print(f"\n{'='*90}\n  run {i}/{args.runs}   launch state: "
              f"free={pre['free_gib']:.2f} GiB inactive={pre['inactive_gib']:.2f} "
              f"usable={pre['usable_pct']:.1f}%  swap={pre['swap_used_mib']:.0f} MiB\n{'='*90}",
              flush=True)
        if pre["usable_pct"] < args.min_usable_pct:
            print(f"  SKIP: usable {pre['usable_pct']:.1f}% < floor {args.min_usable_pct}%; "
                  f"launching here would measure the OOM cliff, not the shape.", flush=True)
            runs.append({"run": i, "skipped": True, "pre": pre})
            continue

        # ONE WORKDIR PER RUN. `llama_bench_matrix.py` names its stderr capture after the ARM
        # (`llama_bench_<arm>.stderr.log`), so N launches of one arm overwrite each other and the
        # only surviving log belongs to the last launch. That is not a cosmetic loss: it is how
        # the 286.67 run's segment counters were destroyed on 2026-09-16, which is exactly the
        # evidence this experiment needs. The tag itself cannot vary (it selects the arm), so the
        # directory has to.
        rdir = Path(args.logdir) / f"{stamp}_run{i:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        jpath = rdir / "summary.json"
        cmd = [PY, str(MATRIX), "--arms", args.arm, "--prompt", args.prompt,
               "--gen", args.gen, "--depths", args.depths, "--reps", str(args.reps),
               "--workdir", str(rdir), "--json", str(jpath)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        wall = time.time() - t0
        sys.stdout.write(proc.stdout[-2500:])
        if proc.returncode != 0:
            sys.stdout.write(proc.stderr[-1200:])
        rows = pp_from(jpath)

        # Harvest the engine's own teardown counters for THIS launch from its preserved log.
        segpath = rdir / f"llama_bench_{args.arm}.stderr.log"
        raw = segpath.read_text(errors="replace") if segpath.exists() else None
        seg = harvest_segments(raw) if raw is not None else {}
        if raw is not None:
            seg.update(harvest_gputime(raw))
            seg.update(harvest_decprof(raw))
        if seg.get("slab_disk_mib"):
            print(f"  -> seg: slab pool={seg.get('slab_pool_mib')} MiB "
                  f"disk={seg.get('slab_disk_mib')} MiB ({seg.get('nonresident_pct')}% non-resident)"
                  f"  file_reads={seg.get('file_reads')} pread_usec={seg.get('pread_usec')}"
                  f"  fill_wait_us={seg.get('fill_wait_us')}", flush=True)
        # The instrument this whole arm exists for: is the prefill slowdown uniform across layers
        # (=> clock/power) or carried by one layer (=> implementation)?
        if seg.get("dp_prefill_graphs"):
            print(f"  -> decprof: {seg['dp_prefill_graphs']} prefill graph(s) ntok={seg.get('dp_ntok_max')}"
                  f" totalling {seg.get('dp_prefill_s')} s, "
                  f"{seg.get('dp_decode_graphs')} decode graph(s) totalling {seg.get('dp_decode_s')} s",
                  flush=True)
            print(f"     1st prefill graph: total={seg.get('dpq1_total_ms')} ms "
                  f"wait={seg.get('dpq1_wait_ms')} cb={seg.get('dpq1_cb_ms')} "
                  f"submit={seg.get('dpq1_submit_ms')}", flush=True)
            if seg.get("dpq1_lay_w_median_ms"):
                med = seg["dpq1_lay_w_median_ms"]
                iqr = seg.get("dpq1_lay_w_iqr_over_median") or 0.0
                frac = seg.get("dpq1_lay_w_frac_within_25pct") or 0.0
                uniform = frac >= 0.80 and iqr <= 0.35
                verdict = ("UNIFORM across layers => the graph scaled as a whole "
                           "(clock / power ceiling)" if uniform else
                           "NOT uniform => a subrange dominates; read the series, not the summary")
                print(f"     layer wait over {seg.get('dpq1_layers_seen')} layers: "
                      f"median={med} ms iqr/median={iqr} within+-25%={frac:.0%} "
                      f"min={seg.get('dpq1_lay_w_min_ms')} max={seg.get('dpq1_lay_w_max_ms')} "
                      f"(argmax=L{seg.get('dpq1_lay_w_argmax_layer')}) -> {verdict}", flush=True)
                print("     judged on median/IQR: the first few layers read LOW in both states "
                      "(pipeline ramp), so peak/mean mislabels a uniformly slow graph.", flush=True)
            if seg.get("dpq2_total_ms"):
                cb1 = seg.get("dpq1_cb_ms") or 0.0
                cb2 = seg.get("dpq2_cb_ms") or 0.0
                print(f"     2nd prefill graph: total={seg.get('dpq2_total_ms')} ms "
                      f"wait={seg.get('dpq2_wait_ms')} cb={seg.get('dpq2_cb_ms')} "
                      f"submit={seg.get('dpq2_submit_ms')}   "
                      f"cb 1st/2nd = {(cb1 / cb2):.2f}x, layer-wait 1st/2nd = "
                      f"{seg.get('dpq1_over_dpq2_lay_w')}x", flush=True)
        try:
            log_disp = segpath.resolve().relative_to(ROOT.resolve())
        except ValueError:
            log_disp = segpath          # --logdir may legitimately point outside the repo
        print(f"  -> log: {log_disp if segpath.exists() else '(missing)'}", flush=True)
        for r in rows:
            print(f"  -> pp{r['n_prompt']} d={r['n_depth']} = {r['avg_ts']:.2f} ± "
                  f"{r['stddev_ts']:.2f} t/s  (b={r['n_batch']}, {wall:.0f}s wall, rc={proc.returncode})",
                  flush=True)
        runs.append({"run": i, "skipped": False, "pre": pre, "wall_s": round(wall, 1),
                     "warm_s": warm_s, "rc": proc.returncode, "pp": rows, "seg": seg,
                     "log": str(segpath),
                     "err": None if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1:]})
        time.sleep(5)  # let Metal release the GPU buffers before the next process

    # ---- verdict ---------------------------------------------------------------------------
    got = [r for r in runs if not r["skipped"] and r["pp"] and not r["pp"][0]["incomplete"]]
    print(f"\n{'='*90}\n  SUMMARY -- {args.arm}  pp{args.prompt} @ -ub 6144\n{'='*90}")
    print(f"{'run':>4s} {'usable%':>8s} {'free GiB':>9s} {'cache GiB':>10s} {'anon GiB':>9s} "
          f"{'swap MiB':>9s} {'pp t/s':>9s} {'±':>7s} {'wall s':>7s}")
    for r in runs:
        if r["skipped"]:
            print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
                  f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
                  f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
                  f"{r['pre']['swap_used_mib']:>9.0f}   (skipped: below floor)")
            continue
        if not r["pp"]:
            print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
                  f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
                  f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
                  f"{r['pre']['swap_used_mib']:>9.0f}   FAILED rc={r['rc']}")
            continue
        p = r["pp"][0]
        mark = "  [INCOMPLETE]" if p["incomplete"] else ""
        print(f"{r['run']:>4d} {r['pre']['usable_pct']:>8.1f} {r['pre']['free_gib']:>9.2f} "
              f"{r['pre'].get('cached_gib', float('nan')):>10.2f} "
              f"{r['pre'].get('anon_gib', float('nan')):>9.2f} "
              f"{r['pre']['swap_used_mib']:>9.0f} {p['avg_ts']:>9.2f} {p['stddev_ts']:>7.2f} "
              f"{r['wall_s']:>7.0f}{mark}")

    if len(got) < 2:
        print(f"\nVERDICT: UNDECIDED -- only {len(got)} usable run(s); need >= 2 to talk about spread.")
        return 2

    vals = [r["pp"][0]["avg_ts"] for r in got]
    lo, hi = min(vals), max(vals)
    spread = 100.0 * (hi - lo) / lo
    within = max(r["pp"][0]["stddev_ts"] / r["pp"][0]["avg_ts"] for r in got) * 100.0

    # Monotonicity -- Spearman rho, no scipy needed: rank both, Pearson on ranks.
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        rk = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    def spearman(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else float("nan")

    ts = [r["pp"][0]["avg_ts"] for r in got]
    rho_free = spearman([r["pre"]["usable_pct"] for r in got], ts)
    rho_cache = spearman([r["pre"].get("cached_gib", 0.0) for r in got], ts)
    rho_anon = spearman([r["pre"].get("anon_gib", 0.0) for r in got], ts)
    # The verdict keys off whichever variable actually explains the spread. rho_cache is the
    # leading candidate; rho_free is kept because the first version of this script guessed it.
    rho = rho_cache if abs(rho_cache) >= abs(rho_free) else rho_free
    which = "page cache (File-backed)" if rho is rho_cache else "free+inactive"

    print(f"\n  runs usable      : {len(got)}/{args.runs}")
    print(f"  range            : {lo:.2f} .. {hi:.2f} t/s   (spread {spread:.1f}%)")
    print(f"  worst within-run : {within:.1f}%  (llama-bench -r {args.reps}, same process)")
    print(f"  spearman rho(page-cache GiB, t/s) : {rho_cache:+.2f}   <- leading candidate")
    print(f"  spearman rho(usable%, t/s)        : {rho_free:+.2f}")
    print(f"  spearman rho(anon GiB, t/s)       : {rho_anon:+.2f}")
    print(f"  CONFOUND: n={len(got)} and run order == the order these variables moved. A perfect")
    print(f"            rank agreement cannot separate 'this variable causes it' from 'state")
    print(f"            accumulates with run order'. Treat |rho| > 0.7 here as a HYPOTHESIS.")
    print(f"  target           : 250 t/s")
    for r in got:
        print(f"    run {r['run']}: {r['pp'][0]['avg_ts']:7.2f} t/s  -> "
              f"{'AT/ABOVE' if r['pp'][0]['avg_ts'] >= 250 else 'below':>9s} target")
    print()
    # The useful output is not "the spread" but "what is the REPEATABLE BAND". With one outlier in
    # five, the honest reading is the cluster -- the range just says an outlier exists.
    rest = sorted(vals)[:-1]
    rest_spread = 100.0 * (rest[-1] - rest[0]) / rest[0] if len(rest) > 1 else float("nan")
    n_hit = sum(1 for v in vals if v >= 250.0)
    print(f"  repeatable band  : {rest[0]:.2f} .. {rest[-1]:.2f} t/s  ({len(rest)} runs, excluding "
          f"the single max; spread {rest_spread:.1f}%)")
    print(f"  cleared 250      : {n_hit}/{len(vals)} runs")
    print()
    if spread < 5.0:
        print("VERDICT: CERTIFIABLE -- spread < 5%. The number is a spec; use these runs as the "
              "reference and stop calling it a draw.")
    elif n_hit == 0:
        print(f"VERDICT: DRAW, TARGET UNREACHED AT THIS SHAPE -- 0/{len(vals)} runs cleared 250. "
              f"Quote the repeatable band ({rest[0]:.0f}-{rest[-1]:.0f} t/s), not the target.")
    else:
        print(f"VERDICT: DRAW -- {n_hit}/{len(vals)} runs cleared 250, spread {spread:.1f}%. "
              f"250 is an outlier, not the operating point; the repeatable band is "
              f"{rest[0]:.0f}-{rest[-1]:.0f} t/s ({rest_spread:.1f}% across {len(rest)} runs).")
    if len(got) >= 4 and abs(rho) >= 0.7:
        print(f"  leading explanation: {which} tracks t/s (rho={rho:+.2f}), BUT n={len(got)} with "
              f"run order confounded -- this is a hypothesis to test by forcing the two states "
              f"explicitly (pre-read the model to force cache up; run a hog to force anon up), "
              f"not a measured mechanism.")
    elif len(got) >= 4:
        print(f"  no candidate variable tracked t/s (|rho| < 0.7 for page cache, free memory and "
              f"anonymous pages). The draw is driven by something not measured here; instrument "
              f"the slot/fill path instead of the machine's memory counters.")

    # ---- segment decomposition: fastest vs slowest launch --------------------------------
    # The machine-level counters are all dead, so the difference has to live inside the engine.
    # These are the engine's own teardown numbers, emitted with no env gate, so the two launches
    # being differenced ran byte-identical instrumentation. Whatever moves here is a segment; a
    # segment that does NOT move cannot be the cause.
    srt = sorted(got, key=lambda r: r["pp"][0]["avg_ts"])
    slow, fast = srt[0], srt[-1]
    if fast["seg"] and slow["seg"]:
        print(f"\n{'='*90}\n  SEGMENT DECOMPOSITION -- run {fast['run']} ({fast['pp'][0]['avg_ts']:.2f} t/s)"
              f"  vs  run {slow['run']} ({slow['pp'][0]['avg_ts']:.2f} t/s)"
              f"   ratio {fast['pp'][0]['avg_ts']/slow['pp'][0]['avg_ts']:.2f}x\n{'='*90}")
        keys = [k for k in SEG_PATTERNS if k in fast["seg"] or k in slow["seg"]]
        print(f"{'segment':>18s} {'FAST':>16s} {'SLOW':>16s} {'fast/slow':>10s}  read")
        for k in keys:
            fv, sv = fast["seg"].get(k), slow["seg"].get(k)
            if fv is None or sv is None:
                continue
            ratio = (fv / sv) if sv else float("inf")
            # A segment can only explain the gap if it moves; flag the ones that do not.
            note = "" if abs(ratio - 1.0) > 0.02 else "  (unchanged)"
            print(f"{k:>18s} {fv:>16.2f} {sv:>16.2f} {ratio:>10.2f}{note}")
        # Wall time implied by each candidate, per launch: does the movement ACCOUNT for the gap?
        for k, unit in (("slab_disk_mib", "MiB"), ("pread_usec", "us aggr"),
                        ("fill_wait_us", "us"), ("seg_wait_s", "s"), ("seg_cb_s", "s"),
                        ("gpu_wait_ms", "ms"), ("gpu_union_ms", "ms"), ("gpu_gap_ms", "ms"),
                        ("ppg_wait_ms", "ms"), ("ppg_union_ms", "ms")):
            if k in fast["seg"] and k in slow["seg"]:
                print(f"  {k}: fast={fast['seg'][k]:.0f} {unit}  slow={slow['seg'][k]:.0f} {unit}  "
                      f"delta={fast['seg'][k]-slow['seg'][k]:+.0f} {unit}")
        if "nonresident_pct" in fast["seg"]:
            print(f"  non-resident share: fast={fast['seg']['nonresident_pct']:.1f}%  "
                  f"slow={slow['seg']['nonresident_pct']:.1f}%   <- M2 exit condition")
        # The two numbers that decide whether the gap is fixable in the engine at all.
        print(f"\n  >>> throughput ratio (fast/slow)          : "
              f"{fast['pp'][0]['avg_ts']/slow['pp'][0]['avg_ts']:.2f}x")
        for k in ("seg_wait_s", "seg_cb_s", "gpu_union_ms"):
            if k in fast["seg"] and slow["seg"] and fast["seg"][k]:
                print(f"  >>> {k:14s} ratio (slow/fast)     : {slow['seg'][k]/fast['seg'][k]:.2f}x")
        if "gpu_union_pct" in fast["seg"]:
            print(f"  >>> GPU busy / wait  fast={fast['seg']['gpu_union_pct']:.0f}%  "
                  f"slow={slow['seg']['gpu_union_pct']:.0f}%   "
                  f"(engine rule: >=70% = real GPU execution, <=40% = launch latency)")
        if "gpu_skipped" in fast["seg"]:
            print(f"  >>> GPU timestamps skipped: fast={fast['seg']['gpu_skipped']} "
                  f"slow={slow['seg']['gpu_skipped']}  (must be 0 or the GPU numbers are void)")
        print(f"  logs: fast {Path(fast['log']).relative_to(ROOT)}")
        print(f"        slow {Path(slow['log']).relative_to(ROOT)}")
        print(f"  NOTE  fill_wait_us is the only I/O term on the CALLING thread (comparable to wall);")
        print(f"        pread_usec is an aggregate over worker threads and CANNOT be compared to it.")
        print(f"        The whole-layer slab pread increments NO counter, so slab I/O is invisible here.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arm": args.arm, "prompt": args.prompt, "runs": runs,
             "spread_pct": spread, "within_run_pct": within,
             "repeatable_band_ts": [rest[0], rest[-1]], "repeatable_band_spread_pct": rest_spread,
             "n_cleared_250": n_hit,
             "spearman_rho_cache": rho_cache, "spearman_rho_usable": rho_free,
             "spearman_rho_anon": rho_anon},
            ensure_ascii=False, indent=2))
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
