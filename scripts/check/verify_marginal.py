#!/usr/bin/env python3
"""Where does the marginal verify token go: misses, matmul compute, or the host path?

WHY THIS EXISTS

The k-sweep (`docs/MTP_K_SWEEP_2026-09-22.md`) measured the marginal cost of one more verify
token at c_tok = 69.6 ms and split it with the engine's own counters into ~12% MTP-head forward
and ~88% "residual (target verify + host)" -- a 61 ms/token black box. Three mechanisms are
distinguishable *a priori* and lead to three different pieces of work:

  MISS-DRIVEN     each extra verify token brings its own top-k ids, so the layer's distinct-expert
                  set grows => more expert fills.          lever = coverage / routing locality
  COMPUTE-DRIVEN  the verify batch's mul_mat_id work.        lever = the verify op path
  HOST-DRIVEN     per-token host work around the batch (dispatch, slot management, submission).
                                                             lever = batch the host path

THE DESIGN: two arms whose ONLY difference is the verify width, k=1 (T=2) vs k=3 (T=4), and three
independently measured axes, each expressed PER ROUND and then differenced / dT:

  marginal_total   = (round_ms(T=4) - round_ms(T=2)) / 2        CGC-MTP-PERF + the arm's t/s
  marginal_miss    = (miss/round(T=4) - miss/round(T=2)) * PER_MISS_MS / 2    BATCHDBG
  marginal_compute = (vop_us/round(T=4) - vop_us/round(T=2)) / 2 / 1000       CGC-VERIFY-OP
  marginal_host    = marginal_total - marginal_miss - marginal_compute        (a residual)

`PER_MISS_MS = 0.695` is F1's per-miss law (`docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md`), applied
as an EXTERNAL constant with its measured range reported -- a law borrowed from another experiment
is an assumption, not a measurement.

Per-round denominators come from the engine's own counters (`calls_draft` = rounds, measured), not
from an assumed round count. Misses are counted only AFTER the last `CGC-POST:` line: on this line
of work prefill was measured to carry 59% of a run's misses, so a miss series that does not respect
the phase boundary is mostly a prefill regression.

AN INSTRUMENT TRAP THIS TOOL EXISTS TO DOCUMENT (measured 2026-09-22)

`CGC_VERIFY_OP_TIMING=1` and the DECPROF/GAPSPLIT per-layer family are **mutually exclusive**.
VERIFY_OP_TIMING makes the expert-cache callback answer `ask=true` for the verify MUL_MAT_ID nodes,
so those nodes are evaluated one at a time and the segmented async dispatcher never runs -- measured
in the same run: CGC-VERIFY-OP 4096 rows, BATCHDBG 5647 rows, and CGC-SEG / CGC-DECPROF /
CGC-GAPSPLIT-L / CGC-GPUTIME **all 0**. run_server.sh's own note ("only produces output under
CGC_OA_ASYNC=1") is necessary but not sufficient; arming both looks exactly like "the per-layer
instrument has no effect". So: this tool uses CGC-VERIFY-OP + BATCHDBG, and reads the per-layer cb
table only when it is present (a run that was armed with DECPROF and not VERIFY_OP_TIMING).

PREDECLARED RULE (committed before the numbers)

    share_x = marginal_x / marginal_total
    >= 0.60 on one axis -> that axis owns it (MISS / COMPUTE / HOST)
    otherwise           -> NO SINGLE LEVER; report the three and the concentration reading

THE SECOND READING (added 2026-09-23): the same marginal, split by STEP CHANNEL

The round-level axes above need a borrowed constant and a second instrument (CGC-VERIFY-OP, which
suppresses the segmented dispatcher -- see the trap). The DECPROF step line carries the same
question without either, because one step's own channels are measured directly:

    total = wait + cb + submit      (wait = the CPU window enclosing the GPU work,
                                     cb = the expert-cache top-k hook,
                                     submit = dispatching that step's segments)
    and, with CGC_GPU_TIMING=1, a GPU-clock tail: gpu_sum / union_sum / gap_sum.

`union_sum` is the device-span wall time of the step's segments, i.e. the verify graph; `cb` is the
host bookkeeping; `submit` is the dispatch. So the three buckets are measured, not derived, and no
constant is borrowed. Two estimators are reported because neither design is clean on its own:

  WITHIN-ARM SLOPE   verify width is not assigned. Inside one arm the widths vary (2/3/4 on a k=3
                     arm) and the slope over them is drift-free, but the narrow steps are the ones
                     adjacent to a rejection, so their pool state is not the wide steps'. When
                     CGC_PHASE_DBG=1 labels every graph with its phase, the family is restricted to
                     phase=VERIFY and the labels are reported with the count.
  ARM CONTRAST       k=1 (T=2) vs k=3 (T=4): a true experimental contrast, but the two arms differ
                     in rounds-per-token, so their miss regimes differ. The tool prints both and, if
                     they disagree by more than 2x, says so instead of picking one.

TWO MORE READS, ADDED 2026-09-23 (both fail closed, both stand on measured columns)

`--node-attr LOG` asks which OP owns the span, from the per-step op table (CGC-GPUOPS), and refuses the
slope when the table cannot carry it: the table is printed on a 1-in-8 sample whose width mix has to
match the family's, and every width the regression uses needs >=4 samples. Census over the 45 logs on
disk that carry the table: 25 refused for the mix, 17 for a one-to-three-sample cell, 3 have no table in
the family, 0 pass -- so the span's per-op split is not obtainable from any product we have. The capture
that answers it is CGC_GPU_NODES_MATRIX=1 (one line per command buffer, unsampled, identified within a
single width); `gdn_split.py` already builds its design matrix.

`--k-econ FILE` turns the round into arithmetic: `rest = F + m*T` with rest and draft both measured, then
prints the amortisation share, the break-even emitted length against the smallest-k arm, the ceiling at
perfect accept, and the budget a target (default 40 ms/token = 25 t/s) leaves for F + m*T. It prints its
own dof, and with two arms that is 0 -- describe, not test -- so m is quoted next to the independent
within-arm slope.

USAGE
    python3 scripts/check/verify_marginal.py --selftest
    python3 scripts/check/verify_marginal.py --logdir /tmp/verify_marginal --arm-t1 k1 --arm-t4 k3
    python3 scripts/check/verify_marginal.py --logdir /tmp/verify_layers --arm-t1 k1 --arm-t4 k3 \\
            --steps          # add the step-channel split (needs CGC_DECODE_PROFILE=1 + CGC_GPU_TIMING)
    python3 scripts/check/verify_marginal.py --node-attr <arm>.stderr.log
    python3 scripts/check/verify_marginal.py --k-econ /tmp/verify_layers/arms.jsonl [--target-ms 40]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gap_vs_miss as gvm  # noqa: E402  (per-layer DECPROF/GAPSPLIT/BATCHDBG series + phase rules)
import gdn_split as gds    # noqa: E402  (the CGC-NSM per-command-buffer design matrix + its solver)

K_OF = {"k1": 1, "k2": 2, "k3": 3, "k4": 4}
PER_MISS_MS = 0.695
PER_MISS_RANGE = (0.59, 0.85)
MTP = re.compile(r"CGC-MTP-PERF\s+type=(\S+)\s+calls_begin=(\d+)\s+calls_draft=(\d+)\s+"
                 r"calls_accept=(\d+)\s+gen_tokens=(\d+)\s+acc_tokens=(\d+)\s+"
                 r"t_begin_ms=([\d.]+)\s+t_draft_ms=([\d.]+)\s+t_accept_ms=([\d.]+)\s+"
                 r"acc_rate=([\d.]+)\s+gen_tok_per_round=([\d.]+)\s+acc_tok_per_round=([\d.]+)\s+"
                 r"emit_tok_per_round=([\d.]+)\s+ms_per_round=([\d.]+)")
VOP = re.compile(r"CGC-VERIFY-OP: kind=(\S+) us=(\d+) ntok=(\d+)")
BAT = re.compile(r"BATCHDBG layer=(\d+) misses=(\d+)")
# The step line (`ggml-backend.cpp`, the CGC-DECPROF block) and its optional GPU-clock tail. Both
# live in the same step, so the split costs no extra instrument and no borrowed constant.
STEP = re.compile(r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
                  r"wait=([\d.]+) \(\d+%\) cb=([\d.]+) \(\d+%\) submit=([\d.]+) \(\d+%\) ntok=(\d+)")
GPTAIL = re.compile(r"layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms")
PHASE = re.compile(r"CGC-PHASE-DBG: il=(\d+) phase=(\S+) n_past=(-?\d+) n_tokens=(\d+) ")

# The three buckets, in the order a reader should meet them: the enclosing window first, then the
# device span, the host hook and the dispatch. `gpu`/`gap` are reported beside `union` because
# gpu_sum sums per-buffer busy time and can exceed the wait window (overlap double counting), so a
# share computed on it would be a share of a different quantity.
CHANNELS = (("total", "step total"), ("wait", "eval-sync window"), ("union", "device span (graph)"),
            ("cb", "host hook"), ("submit", "dispatch"), ("gpu", "device busy"), ("gap", "device idle"))


def newest_log(rdir: Path) -> Path | None:
    logs = sorted(rdir.glob("*.stderr.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def last_mtp_perf(text: str) -> dict | None:
    """Cumulative counters: the last line is the run total (common/speculative.cpp:2986)."""
    hits = [m for m in (MTP.search(l) for l in text.splitlines()) if m]
    if not hits:
        return None
    g = hits[-1].groups()
    return {"type": g[0], "rounds": int(g[2]), "gen_tokens": int(g[4]), "acc_tokens": int(g[5]),
            "t_begin_ms": float(g[6]), "t_draft_ms": float(g[7]),
            "emit_tok_per_round": float(g[12]), "ms_per_round": float(g[13])}


def arm_axes(run: dict) -> dict:
    """Per-ROUND axes for one arm, plus the per-layer miss series for the decode phase."""
    rdir = Path(run.get("dir") or "")
    log = newest_log(rdir)
    if log is None:
        return {"invalid": f"no *.stderr.log in {rdir}"}
    text = log.read_text(errors="replace")
    lines = text.splitlines()
    boundary = max((i for i, l in enumerate(lines) if l.startswith("CGC-POST:")), default=None)
    if boundary is None:
        return {"invalid": "no CGC-POST: boundary -- the decode miss population cannot be separated "
                           "from prefill (prefill carried 59% of misses on this line of work)"}
    perf = last_mtp_perf(text)
    if perf is None or perf["rounds"] <= 0:
        return {"invalid": "no usable CGC-MTP-PERF line (needs CGC_MTP_PERF=1, a spec arm)"}

    misses_dec, misses_by_layer = 0, collections.defaultdict(int)
    for l in lines[boundary + 1:]:
        if (m := BAT.search(l)):
            misses_dec += int(m.group(2))
            misses_by_layer[int(m.group(1))] += int(m.group(2))
    if not misses_dec:
        return {"invalid": "no BATCHDBG rows after the boundary -> no miss axis; arm "
                           "LLAMA_EXPERT_CACHE_BATCH_DBG=1 (run_server.sh forwards it now)"}
    vop = collections.defaultdict(lambda: {"us": 0, "tok": 0, "rows": 0})
    for l in lines[boundary + 1:]:
        if (m := VOP.search(l)):
            v = vop[m.group(1)]
            v["us"] += int(m.group(2))
            v["tok"] += int(m.group(3))
            v["rows"] += 1
    rounds = perf["rounds"]
    return {"log": str(log), "rounds": rounds, "emit_tok_per_round": perf["emit_tok_per_round"],
            "emitted": perf["acc_tokens"] + rounds, "misses_dec": misses_dec,
            "misses_by_layer": dict(misses_by_layer),
            "miss_per_round": misses_dec / rounds,
            "vop_us_per_round": (sum(v["us"] for v in vop.values()) / 1000.0 / rounds) if vop else None,
            "vop": {k: dict(v, us_per_tok=(v["us"] / v["tok"]) if v["tok"] else None)
                    for k, v in vop.items()}}


def pair_rep(run1: dict, run3: dict, dt: int, per_miss: float) -> dict:
    a, b = arm_axes(run1), arm_axes(run3)
    for r in (a, b):
        if "invalid" in r:
            return {"invalid": r["invalid"]}
    if a["rounds"] == b["rounds"]:
        return {"invalid": f"both arms have {a['rounds']} rounds -- the two arms were not different"}
    marg_total = (run3["round_ms"] - run1["round_ms"]) / dt
    marg_miss = (b["miss_per_round"] - a["miss_per_round"]) * per_miss / dt
    marg_comp = ((b["vop_us_per_round"] - a["vop_us_per_round"]) / dt
                 if (a["vop_us_per_round"] is not None and b["vop_us_per_round"] is not None)
                 else float("nan"))
    marg_host = marg_total - marg_miss - (0.0 if marg_comp != marg_comp else marg_comp)
    return {
        "invalid": None, "marginal_total": marg_total, "marginal_miss": marg_miss,
        "marginal_compute": marg_comp, "marginal_host": marg_host,
        "round1": run1["round_ms"], "round3": run3["round_ms"],
        "miss_per_round1": a["miss_per_round"], "miss_per_round3": b["miss_per_round"],
        "rounds1": a["rounds"], "rounds3": b["rounds"],
        "vop_us_round1": a["vop_us_per_round"], "vop_us_round3": b["vop_us_per_round"],
        "misses_by_layer1": a["misses_by_layer"], "misses_by_layer3": b["misses_by_layer"],
        "log1": a["log"], "log3": b["log"],
    }


# The DECPROF row is the only per-layer split of a decode step: wait = eval sync (CPU blocked on the
# GPU), cb = gather (the expert-cache top-k hook), submit = dispatch. Named here once so the three
# axes and the report cannot drift apart.
AXES = (("wait", "eval-sync"), ("cb", "gather"), ("submit", "dispatch"))


def layer_table(run1: dict, run3: dict, dt: int, per_miss: float, stat: str = "mean") -> tuple:
    """Per-layer marginal of each step term, for a pair armed with DECPROF.

    `stat` defaults to the MEAN, and that is a deliberate choice a reader must be able to see: the
    marginal cost of one more verify token is carried by the expensive steps, and the same layer's
    per-step values span 0.02-194 ms, so the median step of a layer that occasionally pays is ~0 by
    construction (`gap_vs_miss` says so in its own header). Measured on a real pair: the median
    version sums to 14% of the round-level marginal, i.e. a decomposition of a different quantity.

    Returns (rows, reason). The per-layer series come from `gap_vs_miss.parse`, so the phase
    boundary and the per-family segment-count rules are that module's, not this file's.
    """
    suffix = {"mean": "_mean", "median": ""}.get(stat)
    if suffix is None:
        return None, f"unknown stat {stat!r}"
    key = lambda k: k + suffix  # noqa: E731  (one name for the axis accessor, used below)
    l1, l3 = newest_log(Path(run1.get("dir") or "")), newest_log(Path(run3.get("dir") or ""))
    if l1 is None or l3 is None:
        return None, "no stderr log"
    rows1, _ = gvm.parse(l1)
    rows3, _ = gvm.parse(l3)
    if not any(r["cb"] == r["cb"] for r in rows1) or not any(r["cb"] == r["cb"] for r in rows3):
        return None, ("no per-layer cb rows -- either DECPROF was not armed, or it was armed "
                      "together with CGC_VERIFY_OP_TIMING, which suppresses the segmented path")
    a, b = {r["layer"]: r for r in rows1}, {r["layer"]: r for r in rows3}
    out = []
    for l in sorted(set(a) & set(b)):
        if not all(a[l].get(key(k)) == a[l].get(key(k)) and b[l].get(key(k)) == b[l].get(key(k))
                   for k, _ in AXES):
            continue
        row = {"layer": l,
               "d_miss": ((b[l]["miss_per_step"] - a[l]["miss_per_step"]) / dt
                          if (a[l]["miss_per_step"] == a[l]["miss_per_step"]
                              and b[l]["miss_per_step"] == b[l]["miss_per_step"]) else float("nan"))}
        row["miss_ms"] = row["d_miss"] * per_miss
        for k, _ in AXES:
            row[k] = (b[l][key(k)] - a[l][key(k)]) / dt
            row[k + "_1"] = a[l][key(k)]
            row[k + "_3"] = b[l][key(k)]
        row["marg"] = sum(row[k] for k, _ in AXES)
        row["resid"] = row["marg"] - row["miss_ms"]
        out.append(row)
    return (out, None) if out else (None, "no layer had a finite row in both arms")


# ── the step-channel split ─────────────────────────────────────────────────────────────────────
# One step, four measured terms (wait/cb/submit/total) plus the GPU-clock tail. No constant, no
# second instrument, and the graph/host/dispatch question is answered inside a single launch.


def step_rows(log: Path) -> list:
    """Every DECPROF step line in `log`, with its GPU tail and its phase label when one exists.

    The labels come two-per-graph (il=0,1) and are printed while layers 0-1 are encoded, i.e. BEFORE
    that graph's own step line (measured on a real log, 2026-09-23). So each pair is claimed by the
    next step line whose `ntok` matches; a step that cannot claim one stays unlabeled rather than
    guessing, and the caller reports how many were labeled (an unlabeled graph and a graph whose
    labels were dropped look alike, and the count is what tells them apart).
    """
    lines = log.read_text(errors="replace").splitlines()
    steps, pending = [], []
    i = 0
    while i < len(lines):
        l = lines[i]
        m = PHASE.search(l)
        if m and int(m.group(1)) == 0:
            nxt = PHASE.search(lines[i + 1]) if i + 1 < len(lines) else None
            pending.append((m.group(2), int(m.group(4))))
            i += 2 if (nxt and int(nxt.group(1)) == 1) else 1
            continue
        s = STEP.search(l)
        if s:
            tail = GPTAIL.search(l) or (GPTAIL.search(lines[i + 1]) if i + 1 < len(lines) else None)
            ntok = int(s.group(8))
            phase = None
            for k in range(len(pending) - 1, -1, -1):  # last pair wins: the nearest before this step
                if pending[k][1] == ntok:
                    phase = pending[k][0]
                    break
            steps.append({"layers": int(s.group(3)), "ntok": ntok,
                          "total": float(s.group(4)), "wait": float(s.group(5)),
                          "cb": float(s.group(6)), "submit": float(s.group(7)),
                          "gpu": float(tail.group(1)) if tail else None,
                          "union": float(tail.group(2)) if tail else None,
                          "gap": float(tail.group(3)) if tail else None,
                          "phase": phase})
            pending = []
        i += 1
    return steps


def _median(vals):
    v = sorted(vals)
    if not v:
        return float("nan")
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def _ols(rows: list, ch: str) -> float:
    """Slope of a channel against the step's token count, over the rows that have it."""
    pts = [(r["ntok"], r[ch]) for r in rows if r.get(ch) is not None]
    if len(pts) < 3:
        return float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else float("nan")


class Steps(list):
    """A step family that can carry the widths it excluded, so nothing is dropped silently."""

    excluded: list = []


def step_family(log: Path, phase: str | None = None, lo: int = 2, hi: int = 4,
                rows: list | None = None) -> tuple:
    """The trunk's verify steps: widths [lo,hi], optionally restricted to one phase.

    `hi` defaults to 4 because a verify step is 1 + (drafts) and these arms draft at most 3. It is
    NOT 8: this cell's prompt chunks are also 8 tokens wide, and a prefill chunk in the same
    regression moves the slope by 2x (measured: +45.0 with them, ... they are 64 of 289 steps). So
    the excluded widths are reported with their counts rather than silently dropped, and when the
    arms carry CGC_PHASE_DBG the phase filter is the stronger exclusion.

    The trunk is the LARGEST `layers` value, not the modal one and not a literal: the MTP head is the
    1-2 layer family, it can outnumber the trunk (measured: 4400 head steps to 231 trunk steps in the
    L3 dense arm), and a literal 40 would silently become wrong the day the layer count moves. The
    4x separation rule keeps "largest" from picking a second trunk-like family; if it does not hold,
    the tool refuses instead of regressing on the wrong graph.
    """
    # `rows=` lets a caller that has already parsed the log pass ITS rows in, so the family and the
    # caller's own step ids describe the same objects. Without it the family is built from a second
    # parse whose dicts are equal but not identical -- which is how a caller that maps steps by
    # identity ends up with an empty family next to a log full of rows (measured 2026-09-23).
    rows = step_rows(log) if rows is None else rows
    if not rows:
        return None, "no CGC-DECPROF step rows (arm CGC_DECODE_PROFILE=1 for this split)"
    fams = collections.Counter(r["layers"] for r in rows)
    mode = max(fams)
    others = sorted((l for l in fams if l != mode), reverse=True)
    if mode <= 2:
        return None, f"the largest graph has {mode} layer(s) -- no trunk family in this log"
    if others and mode < 4 * others[0]:
        return None, (f"layer families {sorted(fams)} are not separated by 4x -- refusing to guess "
                      f"which one is the trunk")
    trunk = [r for r in rows if r["layers"] == mode]
    fam = [r for r in trunk if lo <= r["ntok"] <= hi and (phase is None or r["phase"] == phase)]
    if not fam:
        return None, (f"no trunk step in widths [{lo},{hi}]"
                      + (f" with phase={phase}" if phase else ""))
    if all(r.get("union") is None for r in fam):
        return None, "no GPU tail on any step -- arm CGC_GPU_TIMING=1 for the device side"
    fam = Steps(fam)
    fam.excluded = sorted(collections.Counter(r["ntok"] for r in trunk if r["ntok"] > hi).items())
    return fam, None


def device_gate(fam: list) -> tuple:
    """Is this family's GPU-clock tail in the unit the line claims (ms)?

    The tail is a device SPAN, and a span cannot exceed the CPU window that encloses it by more than
    the segment overlap the dispatcher actually allows. Measured 2026-09-22: 289/290 steps at
    union/wait <= 1.1, i.e. the rule is tight in a healthy build. Measured 2026-09-23 on a LATER
    build of the same file: 326/362 steps off by 10^5-10^7, first violation at step 1 -- a unit
    regression in the instrument, invisible to every consumer that only reads the host channels.

    So this refuses the DEVICE channels rather than a whole arm: the host channels share neither the
    accumulation nor the unit, and they are the ones that answer the dispatch question.
    """
    ratios = sorted((r["union"] / r["wait"]) for r in fam if r["wait"])
    bad = sum(1 for x in ratios if x > 1.10)
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    # The threshold is 5%. Measured with the family restricted to verify widths: 0.3-2.4% (the 09-22
    # pair), 0.9% (the 09-22 L3 dense arm) and 93% (a 09-23 build of the same file). The 5% floor is
    # what keeps an arm sitting at 10-90% from being quoted at all -- those steps are what produced the
    # +47.8 ms/token plateau that the clean arms show as +31.1 on the same widths.
    if ratios and med <= 2.0 and bad <= 0.05 * len(ratios):
        return {"ok": True, "median_ratio": med, "implausible": bad, "n": len(ratios)}, None
    return {"ok": False, "median_ratio": med, "implausible": bad, "n": len(ratios)}, (
        f"the GPU-clock tail is not usable: median(union/wait)={med:.2f}, {bad}/{len(ratios)} steps "
        f"> 1.10x (the clean arms measure 0.3-2.4%, this one {100 * bad / max(len(ratios), 1):.0f}%) "
        f"-- refusing the device channels, host channels still reported")


def _arm_steps(run: dict, phase: str | None) -> tuple:
    """One arm's step family with its per-channel slope and per-width medians, or a refusal."""
    log = newest_log(Path(run.get("dir") or ""))
    if log is None:
        return None, f"no stderr log in {run.get('dir')}"
    fam, why = step_family(log, phase)
    if why:
        return None, why
    gaps = [r for r in fam if r.get("union") is None]
    if gaps:
        return None, (f"{len(gaps)} of {len(fam)} steps have no GPU tail -- a mixed population "
                      f"cannot be split by channel")
    gate, warn = device_gate(fam)
    devices = ["union", "gpu", "gap"] if gate["ok"] else []
    if not gate["ok"]:
        fam = [dict(r, union=None, gpu=None, gap=None) for r in fam]
    widths = sorted({r["ntok"] for r in fam})
    t_cfg = (run.get("k") + 1) if isinstance(run.get("k"), int) else None
    return {"log": str(log), "n": len(fam), "widths": widths, "gate": gate, "device_warning": warn,
            "excluded": getattr(fam, "excluded", []), "t_cfg": t_cfg,
            "labeled": sum(1 for r in fam if r["phase"]),
            "counts": {w: sum(1 for r in fam if r["ntok"] == w) for w in widths},
            "slope": {c: _ols(fam, c) for c, _ in CHANNELS},
            "device": devices,
            "med": {w: {c: _median([r[c] for r in fam if r["ntok"] == w and r[c] is not None])
                         if any(r[c] is not None for r in fam if r["ntok"] == w) else float("nan")
                         for c, _ in CHANNELS} for w in widths}}, None


def channel_split(run1: dict, run3: dict, dt: int, phase: str | None = None) -> dict:
    """Per-channel marginal verify token: within-arm slope + arm contrast, on the same channels.

    `phase` restricts the family when the arms were armed with CGC_PHASE_DBG. When it is None the
    widths alone define the family, which is what the header says is weaker.
    """
    out = {}
    for tag, run in (("T2", run1), ("T4", run3)):
        out[tag], why = _arm_steps(run, phase)
        if why:
            return {"invalid": f"{tag}: {why}"}
    # The contrast is taken at each arm's CONFIGURED verify width (k+1 from its own record). Taking
    # max/min of the observed widths instead picks up whichever width the arm happens to span --
    # and when that set includes a prefill chunk, the "marginal token" comes out 2x too large.
    w2 = out["T2"]["t_cfg"] or min(out["T2"]["widths"])
    w4 = out["T4"]["t_cfg"] or max(out["T4"]["widths"])
    for arm, w in ((out["T2"], w2), (out["T4"], w4)):
        if w not in arm["med"]:
            return {"invalid": f"the configured width T={w} is not in this arm's observed widths "
                               f"{arm['widths']}"}
    return {
        "invalid": None, "phase": phase, "T2": out["T2"], "T4": out["T4"],
        "contrast": {c: (out["T4"]["med"][w4][c] - out["T2"]["med"][w2][c]) / dt
                     for c, _ in CHANNELS},
        "contrast_widths": (w2, w4),
        "slope": out["T4"]["slope"],  # the primary estimator: one launch, several widths
        "primary": "within-arm slope" if len(out["T4"]["widths"]) >= 2 else "arm contrast",
    }


def one_arm_split(run: dict, phase: str | None) -> dict:
    """The same split from a SINGLE arm: the within-arm slope needs no second launch.

    That matters because the arm contrast is the weaker of the two designs whenever the two arms'
    miss regimes differ, and because it is the only reading available from a fresh cell whose pair
    was never launched.
    """
    arm, why = _arm_steps(run, phase)
    if why:
        return {"invalid": why}
    return {"invalid": None, "phase": phase, "T4": arm, "T2": None, "slope": arm["slope"]}


def channel_verdict(marg: dict, ref: float) -> str:
    """The predeclared rule, applied to the device/host/dispatch buckets only.

    `wait` is not a bucket here: it ENCLOSES the device work, so a share computed against it would
    count the same time twice. `total` is the reference for the shares, and it is printed so the
    three buckets' closure (total = wait + cb + submit) can be audited on the same screen.
    """
    sh = {b: (marg[b] / ref if ref else float("nan")) for b in ("union", "cb", "submit")}
    if not ref:
        return "NO VERDICT (no finite marginal)"
    if sh["union"] != sh["union"]:
        # the device side was refused by the unit gate, so a device share would be a share of nothing
        top, val = max(((k, v) for k, v in sh.items() if v == v), key=lambda kv: abs(kv[1]))
        return (f"DEVICE SIDE NOT MEASURABLE (unit gate) -- among the measured channels "
                f"{top.upper()}-LED, {100 * val:.0f}% of the marginal token")
    top, val = max(sh.items(), key=lambda kv: abs(kv[1]))
    if abs(val) < 0.60:
        return "NO SINGLE LEVER -- report all three"
    return f"{top.upper()}-LED -- {100 * val:.0f}% of the marginal token"


def shares(p: dict) -> dict:
    tot = p["marginal_total"]
    out = {"miss": None, "compute": None, "host": None}
    if tot:
        out["miss"] = p["marginal_miss"] / tot
        if p["marginal_compute"] == p["marginal_compute"]:
            out["compute"] = p["marginal_compute"] / tot
            out["host"] = p["marginal_host"] / tot
    return out


def verdict(sh: dict) -> str:
    if sh["miss"] is None:
        return "NO VERDICT (no marginal detected)"
    if sh["miss"] >= 0.60:
        return "MISS-BOUND -- lever is coverage / routing locality"
    if sh["compute"] is not None and sh["compute"] >= 0.60:
        return "COMPUTE-BOUND -- lever is the verify op path"
    if sh["host"] is not None and sh["host"] >= 0.60:
        return "HOST-BOUND -- lever is the per-token host path (dispatch/slot/submit), NOT misses " \
               "and NOT the matmul"
    return "NO SINGLE LEVER -- report all three"


def round_budget(lt: list, run1: dict, run3: dict, dt: int) -> list:
    """The round's terms per arm, and as a per-token marginal.

    The remainder is what NEITHER instrument covers -- named rather than spread over the layers,
    because "the marginal is not in these terms" and "the marginal is zero" must not look alike.
    """
    s1 = sum(r["wait_1"] + r["cb_1"] + r["submit_1"] for r in lt)
    s3 = sum(r["wait_3"] + r["cb_3"] + r["submit_3"] for r in lt)
    terms = [(f"per-layer terms ({len(lt)})", s1, s3),
             ("draft head", run1.get("draft_ms"), run3.get("draft_ms")),
             ("begin", run1.get("begin_ms"), run3.get("begin_ms"))]
    known1 = sum(t[1] for t in terms if t[1] is not None)
    known3 = sum(t[2] for t in terms if t[2] is not None)
    terms.append(("remainder (not in either instrument)", run1["round_ms"] - known1,
                  run3["round_ms"] - known3))
    terms.append(("ROUND TOTAL", run1["round_ms"], run3["round_ms"]))
    return [(n, v1, v3, (v3 - v1) / dt) for n, v1, v3 in terms]


def print_budget(rows: list, ref: float, label: str = "") -> None:
    print(f"    {label}{'term':<36} {'T=2':>9} {'T=4':>9} {'marginal/tok':>13} {'share':>7}")
    for name, v1, v3, m in rows:
        share = (m / ref) if ref else float("nan")
        print(f"      {name:<36} {v1:>9.1f} {v3:>9.1f} {m:>+13.2f} {100 * share:>6.0f}%")


def print_step_channels(rows: dict, dt: int, phase: str | None) -> None:
    """The same marginal token, split by channel: one step's own terms, two estimators."""
    print(f"\n  step-channel split (DECPROF step line + its GPU tail; CGC_DECODE_PROFILE=1 + "
          f"CGC_GPU_TIMING=1)"
          + (f"  [phase={phase} only]" if phase else "  [no phase filter: widths alone define the family]"))
    for tag in ("T2", "T4"):
        a = rows.get(tag)
        if a is None:
            continue
        print(f"    {tag}: {a['n']} steps, widths {a['widths']}, labeled {a['labeled']}/{a['n']}"
              + (f", excluded widths > {max(a['widths'])}: {a['excluded']}" if a.get("excluded") else ""))
        print(f"      {'T':>3} {'n':>5} " + " ".join(f"{c:>9}" for c, _ in CHANNELS))
        for w in a["widths"]:
            print(f"      {w:>3} {a['counts'][w]:>5} "
                  + " ".join((f"{a['med'][w][c]:>9.2f}" if a['med'][w][c] == a['med'][w][c]
                              else f"{'n/a':>9}") for c, _ in CHANNELS))
    warn = (rows["T4"] or rows.get("T2") or {}).get("device_warning")
    if warn:
        print(f"    !! {warn}")
    print(f"    within-arm slope (primary when the arm has >=2 widths), ms per token:")
    print("      " + "  ".join((f"{name} {rows['slope'][c]:+.2f}" if rows['slope'][c] == rows['slope'][c]
                                 else f"{name} n/a") for c, name in CHANNELS))
    ref = rows["slope"]["total"]
    if rows.get("contrast"):
        cw = rows.get("contrast_widths")
        print(f"    arm contrast (T={cw[1]} median minus T={cw[0]} median, /dT={dt}), ms per token:"
              if cw else
              f"    arm contrast (median at each arm's configured width, /dT={dt}), ms per token:")
        print("      " + "  ".join(f"{name} {rows['contrast'][c]:+.2f}" for c, name in CHANNELS))
        refc = rows["contrast"]["total"]
        print(f"    closure, within-arm: total {ref:+.2f} vs wait+cb+submit "
              f"{rows['slope']['wait'] + rows['slope']['cb'] + rows['slope']['submit']:+.2f}"
              f"   | contrast: {refc:+.2f} vs "
              f"{rows['contrast']['wait'] + rows['contrast']['cb'] + rows['contrast']['submit']:+.2f}")
        if ref and refc and abs(ref / refc) > 2:
            print(f"    !! the two estimators disagree by {abs(ref / refc):.1f}x -- the width mix is not "
                  f"assigned (narrow steps sit next to a rejection), so only the ORDER of the buckets "
                  f"is quotable from this cell")
    else:
        print(f"    closure: total {ref:+.2f} vs wait+cb+submit "
              f"{rows['slope']['wait'] + rows['slope']['cb'] + rows['slope']['submit']:+.2f}"
              f"  (single arm: no contrast to cross-check it)")
    print(f"    shares of the within-arm total: "
          + "  ".join((f"{b} {100 * rows['slope'][b] / ref:.0f}%"
                       if (ref and rows['slope'][b] == rows['slope'][b]) else f"{b} n/a")
                      for b in ("union", "cb", "submit")))
    print(f"    RULE -> {channel_verdict(rows['slope'], ref)}")
    print(f"    (device span is `union`; `gpu` sums per-buffer busy time and can exceed the wait "
          f"window, so it is not used for a share)")


# ── the node-level table: what the device span is made of, when the file on disk can say ─────────
#
# `CGC-GPUOPS` is the per-step op table (one row per ggml op: node count, the count-weighted share,
# the work-weighted share `wcntw`, the upper bound, and `uni`). It is the obvious attribution for the
# span and it comes with two conditions, both MEASURED rather than assumed:
#   * it is printed on a 1-in-8 sample of the DECPROF cadence (ggml-backend.cpp: `ns_print_n++ % 8`),
#     and the sampled widths are the spec loop's own, so the decimation can entangle with a periodic
#     accept pattern -- measured 83% width-4 in a sample of a 65% width-4 family;
#   * `wcntw` splits a buffer's duration over its work nodes BY COUNT ("justified by the encoder, not
#     by a model"), so it is a partition of buffer busy time, not a measurement of one op's cost.
OPSHDR = re.compile(r"CGC-GPUOPS: step=(\d+) total=([\d.]+) ms nodes_all=(\d+) nodes_work=(\d+) "
                    r"nodes_named=(\d+) nop=(\d+)")
OPSROW = re.compile(r"CGC-GPUOPS:\s+(\S+)\s+nd=\s*(\d+) (work|NOOP) wcntw=\s*([\d.]+)\s+[\d.]+%\s+"
                    r"cntw=\s*([\d.]+)\s+[\d.]+%\s+ub=\s*([\d.]+)\s+[\d.]+%\s+uni=\s*([-\d.]+)")
STEPID = re.compile(r"CGC-DECPROF: step=(\d+) segs=")
OP_WIDTHS = (2, 3, 4)


def op_tables(log: Path) -> dict:
    """Every `CGC-GPUOPS` table in `log`, keyed by step, with a per-table parse-integrity flag.

    The emitter ranks rows by `ub` and ZEROES each entry as it prints it, so the printed order must be
    non-increasing. A break means the row regex matched something else, and a misparse would look
    exactly like "the op you are looking for is absent" -- the one failure this table exists to avoid.
    """
    tables, cur = {}, None
    for line in log.read_text(errors="replace").splitlines():
        m = OPSHDR.match(line)
        if m:
            cur = {"total": float(m.group(2)), "rows": [], "ranked": True}
            tables[int(m.group(1))] = cur
            continue
        m = OPSROW.match(line)
        if m and cur is not None:
            row = {"op": m.group(1), "nd": int(m.group(2)), "work": m.group(3) == "work",
                   "wcntw": float(m.group(4)), "ub": float(m.group(6)), "uni": float(m.group(7))}
            if cur["rows"] and cur["rows"][-1]["ub"] < row["ub"]:
                cur["ranked"] = False
            cur["rows"].append(row)
    return tables


def node_attribution(log: Path, mix_tol: float = 10.0, cell_min: int = 4) -> tuple:
    """The per-op table over the trunk's verify steps, with the two gates that decide if it can speak.

    G1 sampling   the sample's width mix must match the family's (the sample is a decimation of a
                  sequence whose widths the spec loop chooses, so the two can be entangled)
    G2 cells      every width carrying >=10% of the family needs >= cell_min samples: a slope needs two
                  populated cells and the low one is exactly where a decimation leaves holes
    and, always reported rather than gated, the two quantities that qualify the numbers themselves:
    the table's denominator (`total` = buffers that held >=1 node) against the step's `gpu_sum` (all
    buffers), and the residual `total - sum(wcntw)` that is NOT redistributed.

    When either gate fails the composition is still printed (a per-step partition needs no
    regression) and the slope is refused, with the capture that would answer it named.
    """
    text = log.read_text(errors="replace")
    rows = step_rows(log)
    ids = [int(m.group(1)) for m in (STEPID.match(l) for l in text.splitlines()) if m]
    if not rows or len(ids) != len(rows):
        return None, f"cannot pair step ids with step rows ({len(ids)} ids, {len(rows)} rows)"
    fam, why = step_family(log, rows=rows)
    if fam is None:
        return None, why
    famset = {id(r) for r in fam}
    by_id = {ids[i]: rows[i] for i in range(len(rows))}
    fam_ids = [s for s in by_id if id(by_id[s]) in famset]
    tables = op_tables(log)
    samp = {s: tables[s] for s in fam_ids if s in tables}
    if not samp:
        return None, "no CGC-GPUOPS table falls inside the trunk family (arm CGC_GPU_OPS=1)"

    def mix(counter: collections.Counter) -> dict:
        n = max(sum(counter.values()), 1)
        return {w: 100.0 * counter.get(w, 0) / n for w in OP_WIDTHS}

    pop_n = collections.Counter(by_id[s]["ntok"] for s in fam_ids)
    smp_n = collections.Counter(by_id[s]["ntok"] for s in samp)
    pop_mix, sample_mix = mix(pop_n), mix(smp_n)
    mix_dev = {w: abs(pop_mix[w] - sample_mix[w]) for w in OP_WIDTHS}
    # Two ways a cell can be too thin to carry a slope, and both were measured: a width the family
    # mainly lives in can be missing from the sample entirely (width 3 here), and a width with a
    # single sample stands in for that whole width -- one step's value becomes the low end of the
    # regression (the 0919 server log: width 2 has one sample and the table-total slope comes out
    # NEGATIVE against the family's own +50.2 ms/token).
    thin = sorted({w for w in OP_WIDTHS if smp_n.get(w, 0) and smp_n[w] < cell_min} |
                  {w for w in OP_WIDTHS if pop_mix[w] >= 10.0 and smp_n.get(w, 0) < cell_min})
    bad_rank = sorted(s for s, t in samp.items() if not t["ranked"])
    gates = (max(mix_dev.values(), default=0.0) <= mix_tol) and not thin and not bad_rank

    comp, present, n_at = {}, {}, {}
    for w in OP_WIDTHS:
        ts = [t for s, t in samp.items() if by_id[s]["ntok"] == w]
        if not ts:
            continue
        n_at[w] = len(ts)
        comp[w] = {op: _median([r["wcntw"] for t in ts for r in t["rows"] if r["op"] == op])
                   for op in {r["op"] for t in ts for r in t["rows"]}}
        present[w] = {op: sum(1 for t in ts if any(r["op"] == op and r["wcntw"] > 0
                                                  for r in t["rows"]))
                      for op in comp[w]}

    def slope(pts: list, key: str) -> float:
        return _ols([{"ntok": x, key: y} for x, y in pts], key)

    out = {"gates": gates, "n_steps": len(rows), "n_tables": len(tables), "family": len(fam),
           "sampled": len(samp), "pop_mix": pop_mix, "sample_mix": sample_mix, "mix_dev": mix_dev,
           "mix_tol": mix_tol, "thin": thin, "cell_min": cell_min, "bad_rank": bad_rank,
           "comp": comp, "present": present, "n_at": n_at,
           "denominator_pct": _median([100.0 * (by_id[s]["gpu"] - t["total"]) / by_id[s]["gpu"]
                                       for s, t in samp.items() if by_id[s].get("gpu")]),
           "residual_pct": _median([100.0 * (t["total"] - sum(r["wcntw"] for r in t["rows"]))
                                    / t["total"] for s, t in samp.items() if t["total"]])}
    if gates:
        per_op = collections.defaultdict(list)
        for s, t in samp.items():
            for r in t["rows"]:
                per_op[r["op"]].append((by_id[s]["ntok"], r["wcntw"]))
        out["op_slope"] = {op: slope(pts, "wcntw") for op, pts in per_op.items()}
        out["op_slope"]["(residual)"] = slope([(by_id[s]["ntok"],
                                                t["total"] - sum(r["wcntw"] for r in t["rows"]))
                                               for s, t in samp.items()], "wcntw")
        out["total_slope"] = slope([(by_id[s]["ntok"], t["total"]) for s, t in samp.items()], "wcntw")
        vals = [v for v in out["op_slope"].values() if v == v]
        out["sum_slope"] = sum(vals) if vals else float("nan")
        out["n_op_slopes"] = len(vals)
        out["fam_gpu_slope"] = _ols(fam, "gpu")
    return out, None


def print_node_attribution(res: dict, log: Path) -> None:
    def fmt(m: dict) -> str:
        return "{" + ",".join(f"{w}:{m[w]:.0f}" for w in OP_WIDTHS) + "}%"

    print(f"\n  node-level attribution attempt (CGC-GPUOPS: the per-step op table, printed on a "
          f"1-in-8 DECPROF sample)")
    print(f"    {log.name}: {res['n_steps']} steps, {res['n_tables']} tables; trunk family (widths "
          f"2-4) has {res['family']} steps, {res['sampled']} of them sampled")
    dev = max(res["mix_dev"].values(), default=0.0)
    print(f"    GATE G1 sampling mix: population {fmt(res['pop_mix'])} vs sample "
          f"{fmt(res['sample_mix'])}  max|d|={dev:.1f}pp (tol {res['mix_tol']:.0f})"
          + ("  -> REFUSED" if dev > res["mix_tol"] else "  -> ok"))
    if res["thin"]:
        print(f"    GATE G2 cells: width(s) {res['thin']} are used by the sample with "
              f"< {res['cell_min']} samples each (or carry >=10% of the family and are missing) "
              f"-> REFUSED")
    if res["bad_rank"]:
        print(f"    GATE parse: {len(res['bad_rank'])} sample table(s) are not ub-descending "
              f"(first: {res['bad_rank'][0]}) -> REFUSED")
    print(f"    denominators (printed, never redistributed): the table's `total` is "
          f"{res['denominator_pct']:.1f}% below the step's own gpu_sum (median; `total` counts only "
          f"buffers that held >=1 node) and the residual total-sum(wcntw) is "
          f"{res['residual_pct']:.1f}% of `total`")
    w_main = max(res["comp"], key=lambda w: res["n_at"][w])
    ops = sorted(res["comp"][w_main], key=lambda o: -(res["comp"][w_main][o] or 0))[:12]
    print(f"    per-op composition (LEVEL, not a slope: medians of wcntw in ms over the sampled "
          f"steps of each width, an op absent from a step counted as 0)")
    print("      width  n  " + " ".join(f"{o:>9}" for o in ops))
    for w in sorted(res["comp"]):
        print(f"      {w:>5} {res['n_at'][w]:>2}  "
              + " ".join(f"{res['comp'][w].get(o, float('nan')):>9.2f}" for o in ops))
    if not res["gates"]:
        print("    -> the SLOPE (which ms/token lives in which op) is NOT computable from this "
              "file: gates above.")
        print(f"       The capture that answers it is CGC_GPU_NODES_MATRIX=1: one line per command "
              f"buffer, no sampling, and its model (dur = sum_k cnt_k * t_k) is identified WITHIN a "
              f"single width because buffers differ in node composition -- the design matrix "
              f"`gdn_split.py` already builds.")
    else:
        top = sorted(res["op_slope"].items(), key=lambda x: -x[1])[:12]
        print(f"    per-op slope (wcntw over width), ms/token: "
              + "  ".join(f"{o} {v:+.2f}" for o, v in top))
        print(f"    closure: sum over {res['n_op_slopes']} row(s) (ops + residual) {res['sum_slope']:+.2f} vs the "
              f"table's own total {res['total_slope']:+.2f} vs the family's gpu_sum slope "
              f"{res['fam_gpu_slope']:+.2f} ms/token -- a slope claim needs these to agree (the table "
              f"total and the family's gpu_sum differ by the node-less buffers plus the residual), and "
              f"rows that do not carry the signal are unattributed, not zero")


# ── the design matrix: per-KIND cost from one line per command buffer (CGC-NSM) ──────────────────
#
# Two things separate this from the op table above, and both are properties of the emitter rather
# than of the analysis: NSM prints one line PER COMMAND BUFFER (no 1-in-8 sampling), and its model
# `dur_i = sum_k cnt_ik * t_k` is identified from the buffers' own composition, so unlike a slope
# over widths it does not need a width spread at all -- which is exactly what the sampled table
# could not supply. The parser and the least-squares solve are `gdn_split.py`'s, not a third copy.
#
# What it does NOT cover, and has to be printed rather than assumed away: `ns_total` (the step's
# `gpu_sum`) sums EVERY buffer, while the NSM lines cover only the buffers that held >=1 node. So a
# fit over NSM is a partition of the node-bearing busy time, and the coverage ratio is the honest
# qualifier on every number below.
NSMMIN_STEPS = 8
# A buffer cannot be longer than the step that contains it. Two buffers per step fail that on the
# 09-23 build (measured 72 of 11806): `GPUEndTime - GPUStartTime` comes back as an absolute uptime
# stamp (~7.7e13 ns) because the start time arrived as a small POSITIVE value, which passes the
# emitter's `s > 0.0 && e > s` guard in ggml-metal-context.m. The bound is applied here, counted and
# reported rather than smoothed -- and because the same defect inflates the step-level `gpu_sum`
# (5.4e8 ms/step on that build, the unit regression of docs/VERIFY_TOKEN_CHANNELS_2026-09-23.md §7),
# the closure below is stated against the step's HOST wall time, which is healthy in every build.
NSM_GROUP_RULES = (("moe", r"moe|expert"), ("attn", r"attn|^[qkv]cur$|qkv|^l_out$|final_output"),
                   ("gdn", r"gdn|conv|dnqkv|dnbeta|dn_normg|alpha|beta|softplus|predelta|"
                           r"linear_attn|^z-$"),
                   ("ffn_dense", r"^ffn_(gate|up|_)$"), ("norm", r"norm"),
                   ("glue", r"^(node|cache|\(other\))$"))
NSM_GROUPS = tuple(g for g, _ in NSM_GROUP_RULES) + ("other",)


def kind_group(kind: str) -> str:
    """The group a kind belongs to, FIRST rule wins. Printed with the report: an attribution whose
    grouping is not on the page is a guess wearing a table."""
    for g, rx in NSM_GROUP_RULES:
        if re.search(rx, kind):
            return g
    return "other"


def _count_rank(bufs: list) -> tuple:
    """Rank of the count matrix, by pivot elimination on its Gram with the same relative tolerance
    gdn_split's solver uses. Rank r of k kinds means k-r kinds are linear combinations of the others
    in EVERY buffer of this graph, so their individual costs are not recoverable from it."""
    kinds = sorted({k for b in bufs for k in b["cnt"]})
    idx = {k: i for i, k in enumerate(kinds)}
    n = len(kinds)
    gram = [[0.0] * n for _ in range(n)]
    for b in bufs:
        cols = [(idx[k], float(c)) for k, c in b["cnt"].items()]
        for i, (ci, vi) in enumerate(cols):
            for cj, vj in cols[i:]:
                gram[ci][cj] += vi * vj
                if ci != cj:
                    gram[cj][ci] += vi * vj
    dmax = max((gram[i][i] for i in range(n)), default=0.0) or 1.0
    M = [row[:] for row in gram]
    piv = 0
    for c in range(n):
        best = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[best][c]) < 1e-8 * dmax:
            continue
        M[c], M[best] = M[best], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            fq = M[r][c] / pv
            if fq:
                for q in range(c, n):
                    M[r][q] -= fq * M[c][q]
        piv += 1
    return piv, n


NSM_MIN_PATTERNS = 3   # a kind seen in fewer distinct co-occurrence signatures than this keeps a
                       # cost it does not own; the fit returns a number either way (see `patterns`)


def _nsm_arm(log: Path, phase: str | None = None) -> tuple:
    """One arm's family steps paired with their NSM buffers: [(step_dict, row_dict)], or (None, why).

    The pairing is by IDENTITY of the caller's step-row objects, not by value: the family is built
    from `rows` so the two describe the same objects. Pairing by value is what produced an empty
    family next to a log full of rows (measured 2026-09-23); the mismatched counts are returned as a
    refusal instead.
    """
    text = log.read_text(errors="replace")
    rows = step_rows(log)
    if not rows:
        return None, f"{log.name}: no CGC-DECPROF step rows (arm CGC_DECODE_PROFILE=1)"
    ids = [int(m.group(1)) for m in (STEPID.match(l) for l in text.splitlines()) if m]
    if len(ids) != len(rows):
        return None, (f"{log.name}: cannot pair step ids with step rows "
                      f"({len(ids)} ids, {len(rows)} rows)")
    fam, why = step_family(log, phase=phase, rows=rows)
    if fam is None:
        return None, f"{log.name}: {why}"
    gsteps, _ = gds.parse_steps(text)
    by_id = {s["step"]: s for s in gsteps}
    if set(ids) != set(by_id):
        return None, (f"{log.name}: the two parsers disagree about the step set: "
                      f"{len(set(ids) - set(by_id))} id(s) only in the step reader, "
                      f"{len(set(by_id) - set(ids))} only in the NSM reader")
    famset = {id(r) for r in fam}
    return [(by_id[ids[i]], r) for i, r in enumerate(rows) if id(r) in famset], None


def nsm_attribution(log, cell_min: int = 4, phase: str | None = None) -> tuple:
    """What the per-command-buffer design matrix can say about the verify token's span, MEASURED.

    `dur_i = sum_k cnt_ik * t_k` plus one launch offset per log after the first, so a buffer's
    duration is SPLIT among its kinds by the counts instead of being dumped into one bucket.

    Why a solve and not the co-occurrence partition this read started as (measured 2026-09-23):
    every buffer of this graph mixes MoE, attention and GDN kinds, so assigning each buffer to "the
    one group it contains" put 20.30 of the 20.36 ms/token into `(mixed)` and left every real group
    at 0.00 -- a table that reads like an attribution and identifies nothing.

    `logs` may hold several logs and they become ONE design matrix: the marginal per-token split
    needs buffer compositions that only exist at different widths, and the widths live in different
    arms (k=1 -> T=2 ... k=3 -> T=4). A launch is a parameter here, not noise -- one indicator per
    log after the first absorbs pool warmth and carried swap, so t_k stays a within-launch cost.

    What it still cannot say, printed rather than assumed: the count matrix is rank-deficient
    (measured 20 of 44 kinds), so a kind seen in fewer than NSM_MIN_PATTERNS distinct co-occurrence
    signatures keeps a cost it does not own and is marked, never quoted; and NSM prints only the
    buffers that held >= 1 node, so the fit partitions the node-bearing device time and its coverage
    against the step's host window qualifies every number below.
    """
    logs = [Path(p) for p in ([log] if isinstance(log, (str, Path)) else log)]
    per_log, fam_all = [], []
    for ai, lg in enumerate(logs):
        pairs, why = _nsm_arm(lg, phase)
        if pairs is None:
            return None, why
        per_log.append({"arm": ai, "log": lg.name, "pairs": pairs})
        fam_all += [r for _, r in pairs]
    n_fam = len(fam_all)

    kept, dropped, dropped_ms = [], 0, 0.0
    with_nsm = []
    for lg in per_log:
        for step, row in lg["pairs"]:
            if not step["has_nsm"]:
                continue
            wall = row["total"] or 0.0
            bufs = []
            for b in step["nsm"]:
                d = b["dur_ns"] / 1e6
                if wall and d > wall:        # cannot be a duration; see NSM_GROUP_RULES' comment
                    dropped += 1
                    dropped_ms += d
                    continue
                eb = {"arm": lg["arm"], "cnt": b["cnt"], "dur_ns": b["dur_ns"]}
                bufs.append(eb)
                kept.append(eb)
            with_nsm.append({"arm": lg["arm"], "w": row["ntok"], "wall": wall,
                             "wait": row["wait"], "bufs": bufs})
    if len(with_nsm) < NSMMIN_STEPS:
        return None, (f"only {len(with_nsm)} step(s) carry CGC-NSM across {len(logs)} log(s) "
                      f"(need >= {NSMMIN_STEPS}; arm CGC_GPU_NODES=1 + CGC_GPU_NODES_MATRIX=1)")
    if not kept:
        return None, "every NSM buffer is longer than its own step -- the duration field is not one"
    n_at = collections.Counter(s["w"] for s in with_nsm)
    thin = sorted(w for w, n in n_at.items() if n < cell_min)

    # ── the solve ─────────────────────────────────────────────────────
    kinds = sorted({k for b in kept for k in b["cnt"]})
    idx = {k: i for i, k in enumerate(kinds)}
    arms_used = sorted({b["arm"] for b in kept})
    col_of = {a: len(kinds) + i - 1 for i, a in enumerate(arms_used) if i}   # one offset per launch
    A, yv = [], []
    for b in kept:
        row = [0.0] * (len(kinds) + len(col_of))
        for k, c in b["cnt"].items():
            row[idx[k]] = float(c)
        if b["arm"] in col_of:
            row[col_of[b["arm"]]] = 1.0
        A.append(row)
        yv.append(float(b["dur_ns"]))
    sol, r2, cond = gds.lstsq(A, yv)
    t_ns = {k: sol[idx[k]] for k in kinds}
    off_ns = {a: (sol[col_of[a]] if a in col_of else 0.0) for a in arms_used}
    rank, nkinds = _count_rank(kept)
    # A kind's cost is its own only if it is seen with more than one set of neighbours: with a single
    # signature its column is a combination of its neighbours' columns, and the fit still returns a
    # number -- `patterns`, not R2, is what tells the two apart.
    patterns = {k: len({frozenset(b["cnt"]) for b in kept if k in b["cnt"]}) for k in kinds}

    def _fitted(b):                      # ns, the same unit as the durations it is compared against
        return sum(t_ns[k] * c for k, c in b["cnt"].items()) + off_ns[b["arm"]]

    def _per_step_ms(fn, s):
        return sum(fn(b) for b in s["bufs"]) / 1e6

    meas_slope = _ols([{"ntok": s["w"], "v": _per_step_ms(lambda b: b["dur_ns"], s)}
                       for s in with_nsm], "v")
    fit_slope = _ols([{"ntok": s["w"], "v": _per_step_ms(_fitted, s)} for s in with_nsm], "v")
    # Per-kind marginal: t_k times the kind's OWN count growth per extra verify token. The counts come
    # from the kept buffers only, so the rows sum to the fitted slope by construction. Widths with
    # fewer than `cell_min` steps are not used for a marginal -- a slope through one sample is what
    # made the sampled op table print -31.4 ms/token against a measured +50.2 (docs §8).
    wide = sorted(w for w, n in n_at.items() if n >= cell_min)
    marg = None
    if len(wide) >= 2:
        msteps = [s for s in with_nsm if s["w"] in wide]
        # A row's slope is t_k times the kind's OWN count growth per extra token, both from the same
        # OLS over the same steps -- so the rows sum to the fitted slope exactly (OLS is linear in the
        # response), and the closure below is arithmetic rather than a tolerance.
        beta = {k: _ols([{"ntok": s["w"], "v": sum(b["cnt"].get(k, 0) for b in s["bufs"])}
                         for s in msteps], "v") for k in kinds}
        rows_k, unident = {}, []
        for k in kinds:
            if patterns[k] < NSM_MIN_PATTERNS or t_ns[k] != t_ns[k] or beta[k] != beta[k]:
                unident.append(k)
                continue
            rows_k[k] = t_ns[k] / 1e6 * beta[k]
        grp = collections.defaultdict(float)
        for k, v in rows_k.items():
            grp[kind_group(k)] += v
        # The fit's own offset column is per BUFFER row, and the number of command buffers per step
        # grows with the width -- so that term can carry real marginal cost and has to be a row, not
        # a hidden constant. Measured on the fixture below: +0.0025 ms/token from a 1 us/buffer
        # offset, with t_k unmoved. It belongs to no op, and is named rather than left inside
        # `unattributed`.
        off_row = sum(off_ns[a] * _ols([{"ntok": s["w"],
                                        "v": sum(1 for b in s["bufs"] if b["arm"] == a)}
                                       for s in msteps], "v") for a in arms_used) / 1e6
        marg = {"wide": wide, "rows": rows_k, "unident": unident, "sum": sum(rows_k.values()),
                "group": dict(grp), "offset": off_row,
                "fit_slope": _ols([{"ntok": s["w"], "v": _per_step_ms(_fitted, s)}
                                   for s in msteps], "v"),
                "meas_slope": _ols([{"ntok": s["w"], "v": _per_step_ms(lambda b: b["dur_ns"], s)}
                                    for s in msteps], "v")}
    groups = collections.defaultdict(list)
    for b in kept:
        for k in b["cnt"]:
            if k not in groups[kind_group(k)]:      # the group is the key, the kind the entry
                groups[kind_group(k)].append(k)
    return {"n_fam": n_fam, "n_nsm": len(with_nsm), "n_at": dict(n_at), "thin": thin,
            "rank": rank, "nkinds": nkinds, "buffers": len(kept), "dropped": dropped,
            "dropped_ms": dropped_ms, "groups": {k: sorted(v) for k, v in groups.items()},
            "t_ns": t_ns, "patterns": patterns, "r2": r2, "cond_ok": cond, "marg": marg,
            "logs": [l["log"] for l in per_log],
            "meas_slope": meas_slope, "fit_slope": fit_slope,
            "meas_ms": [(s["w"], _per_step_ms(lambda b: b["dur_ns"], s)) for s in with_nsm],
            "fit_ms": [(s["w"], _per_step_ms(_fitted, s)) for s in with_nsm],
            "wall_ms": [(s["w"], s["wall"]) for s in with_nsm],
            "wait_ms": [(s["w"], s["wait"]) for s in with_nsm],
            "kept_share": 100.0 * sum(b["dur_ns"] for b in kept) / 1e6
                          / sum(s["wall"] for s in with_nsm),
            "fam_wall_slope": _ols(fam_all, "total"), "fam_wait_slope": _ols(fam_all, "wait"),
            "fam_union_slope": _ols(fam_all, "union"), "device_gate": device_gate(fam_all)}, None


def print_nsm_attribution(res: dict, logs) -> None:
    """The design matrix, solved. Each branch prints what it cannot say beside what it can."""
    names = ", ".join(Path(p).name for p in ([logs] if isinstance(logs, (str, Path)) else logs))
    print(f"\n  the design matrix, SOLVED (CGC-NSM: one line per command buffer, no sampling)")
    print(f"    {names}")
    print(f"    trunk family {res['n_fam']} step(s), {res['n_nsm']} with NSM, widths {res['n_at']}"
          + (f"  !! cells thinner than 4: {res['thin']}" if res["thin"] else ""))
    print(f"    filter: {res['dropped']} of {res['buffers'] + res['dropped']} buffers "
          f"({100.0 * res['dropped'] / (res['buffers'] + res['dropped']):.2f}%) report a duration "
          f"longer than their own step and are dropped, not smoothed (an absolute uptime stamp in "
          f"the field: see NSM_GROUP_RULES' comment)")
    gate, gwhy = res["device_gate"]
    print(f"    fit: {res['buffers']} buffer rows over {res['nkinds']} kinds, R2={res['r2']:.4f}, "
          f"count-matrix rank {res['rank']}"
          + ("" if res["cond_ok"] else "  !! a pivot is negligible: part of this split is arbitrary"))
    weak = sorted(k for k, p in res["patterns"].items() if p < NSM_MIN_PATTERNS)
    print(f"    {len(weak)} of {res['nkinds']} kind(s) sit in fewer than {NSM_MIN_PATTERNS} distinct "
          f"co-occurrence signatures, so they keep a cost they do not own: not quoted"
          + (f" ({', '.join(weak[:6])}{', ...' if len(weak) > 6 else ''})" if weak else ""))
    print(f"    kept buffers carry {res['kept_share']:.1f}% of the family's host wall "
          f"(the rest are buffers with no nodes, which NSM does not print)")

    def byw(pairs):
        d = collections.defaultdict(list)
        for w, v in pairs:
            d[w].append(v)
        return {w: sum(v) / len(v) for w, v in d.items()}

    mw, fw, ww = byw(res["meas_ms"]), byw(res["fit_ms"]), byw(res["wall_ms"])
    print("    per step, ms: measured(NSM sum) / fitted / host wall")
    for w in sorted(mw):
        print(f"      T={w}  n={res['n_at'][w]:<4d} {mw[w]:>9.2f} {fw[w]:>9.2f} {ww[w]:>9.2f}")

    m = res["marg"]
    if m is None:
        print(f"    marginal per verify token: NOT AVAILABLE -- a per-kind slope needs >= 2 widths "
              f"with >= 4 steps each inside one matrix; this one has {res['n_at']} "
              f"(k1 gives T=2, k3 gives T=4: run both into one --nsm list)")
    else:
        print(f"    marginal per verify token, ms, widths {m['wide']} "
              f"(rows sum to the fitted slope by construction):")
        for g in NSM_GROUPS:
            if g in m["group"]:
                print(f"      {g:<12s} {m['group'][g]:>+8.2f}   "
                      f"[{len(res['groups'].get(g, []))} kind(s) in group]")
        print(f"      {'(unident.)':<12s} {m['fit_slope'] - m['sum'] - m['offset']:>+8.2f}   "
              f"[{len(m['unident'])} kind(s) under {NSM_MIN_PATTERNS} signatures]")
        print(f"      {'(per-buffer)':<12s} {m['offset']:>+8.2f}   [the fit's offset column times the "
              f"buffer-count slope: real cost, no op owns it]")
        print(f"      {'(all rows)':<12s} {m['fit_slope']:>+8.2f}   = the fitted slope, exactly"
              f" (rows + unident + offset) and {m['meas_slope']:+.2f} measured")
        print(f"    measured minus fitted {m['meas_slope'] - m['fit_slope']:+.2f} ms/token: a gap here is "
              f"composition the linear model does not carry, not a missing row")
    print(f"    across the whole family: NSM-duration slope {res['meas_slope']:+.2f}, host wall "
          f"{res['fam_wall_slope']:+.2f}, wait {res['fam_wait_slope']:+.2f}, device-clock union "
          f"{res['fam_union_slope']:+.2f} ms/token")
    print(f"    closure target: " + (f"the GPU-clock tail is usable (median union/wait "
          f"{gate['median_ratio']:.2f}) -- the rows above are a partition of the node-bearing "
          f"device time, {res['kept_share']:.1f}% of the host wall" if gate["ok"] else
          f"not the union slope -- {gwhy}; the rows partition the node-bearing device time that NSM "
          f"itself measured, which is {res['kept_share']:.1f}% of the host wall"))
    print(f"    (group rules: " + "; ".join(f"{g}={rx}" for g, rx in NSM_GROUP_RULES) + ")")


# ── the round's budget: rest = F + m*T, and what amortising F actually buys ───────────────────────
#
# A spec round pays a draft chain (k forward passes) plus one verify step. The verify step is
# `rest_ms` here (round minus draft, both measured), and the model that has been quoted all along is
# exactly `rest = F + m*T` with T = k+1 the verify width. With two arms and two parameters the fit is
# EXACTLY IDENTIFIED: it describes the data and cannot test it. So the tool prints the dof, and the
# fit is only quotable next to a second instrument for m (the within-arm slope, which uses three
# widths from three launches of one arm and no cross-arm assumption at all).
K_ECON_KEYS = ("k", "round_ms", "mean_len", "draft_ms")


def k_econ_rows(path: Path) -> tuple:
    """One row per arm for the round-budget model, from either product this line writes.

    jsonl     the k-sweep's `arms.jsonl`: k, round_ms, mean_len, draft_ms all measured per arm-rep
    k_sweep   a `k_sweep.json` product: k and tps are measured, mean_len/draft_ms come from the arm's
              own CGC-MTP-PERF line, and round_ms is DERIVED as 1000*mean_len/tps -- labelled, and
              refused if the product's own per_k step_ms disagrees with the derivation by >2%.
    """
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(r.get(x) is not None for x in K_ECON_KEYS):
                rows.append(r)
        if not rows:
            return None, f"no row in {path} carries all of {K_ECON_KEYS}"
        return rows, "jsonl"

    doc = json.loads(path.read_text())
    per_k = (doc.get("analysis") or {}).get("per_k") or {}
    rows, skipped = [], []
    for arm in doc.get("arms") or []:
        if arm.get("k") is None or not arm.get("metric") or not arm.get("dir"):
            continue
        log = newest_log(Path(arm["dir"]))
        if log is None:
            return None, f"no *.stderr.log in {arm['dir']}"
        perf = last_mtp_perf(log.read_text(errors="replace"))
        if perf is None or perf["rounds"] <= 0:
            # The sweep's k=0 baseline is a plain (non-spec) cell: it has no verify round, so it is
            # not part of this model rather than a refusal.
            skipped.append(f"k={arm['k']} ({Path(arm['dir']).name}: no CGC-MTP-PERF)")
            continue
        e, tps = perf["emit_tok_per_round"], float(arm["metric"])
        rows.append({"arm": f"k{int(arm['k'])}", "k": int(arm["k"]), "rep": arm.get("pair", 0),
                     "round_ms": 1000.0 * e / tps, "mean_len": e,
                     "draft_ms": perf["t_draft_ms"] / perf["rounds"], "tps": tps,
                     "derived": "round_ms = 1000*mean_len/tps"})
    if not rows:
        return None, (f"no usable arm in {path} (need k, metric and a directory per arm); skipped: "
                      + ", ".join(skipped))
    # `per_k` is that sweep's own per-k aggregate, and it is NOT the median of this derivation (it
    # uses another timing convention: k=3 aggregates 317.4 ms against these arms' {255.3, 330.1,
    # 431.7}). What must hold is that the aggregate lies inside its own arms' range -- a check that
    # passes on both sweeps here and would fail on a wrong derivation.
    for k, pk in per_k.items():
        mine = [r["round_ms"] for r in rows if r["k"] == int(k)]
        if pk.get("step_ms") and mine and not (min(mine) <= pk["step_ms"] <= max(mine)):
            return None, (f"k={k}: the product's own per_k step_ms is {pk['step_ms']:.1f} ms but the "
                          f"derived rounds are {min(mine):.1f}-{max(mine):.1f} ms -- an aggregate "
                          f"outside its own arms' range means one of the two is not the round")
    return rows, "k_sweep" + (f"  [skipped {', '.join(skipped)}]" if skipped else "")


def k_econ(rows: list, target_ms: float = 40.0) -> dict:
    """Per-arm budget, the fit, and the three consequences a k decision needs.

    The three consequences (all arithmetic on measured columns, no extra assumption):
      * the round's fixed work as a share of the delivered cost -- why k is a lever at all;
      * the break-even emitted length: arm k vs the smallest-k arm, E > (draft+rest)/delivered_base;
      * the ceiling at perfect accept (E = T), which is the number that says whether an accept-only
        programme can ever reach the target.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("arm") or f"k{r['k']}"].append(r)
    arms = []
    for a, rs in by.items():
        med = lambda f: _median([f(r) for r in rs])  # noqa: E731  (a local median over this arm's reps)
        k = int(min(r["k"] for r in rs))
        e, round_ms, draft = med(lambda r: r["mean_len"]), med(lambda r: r["round_ms"]), med(lambda r: r["draft_ms"])
        arms.append({"arm": a, "k": k, "n": len(rs), "T": k + 1, "E": e, "round_ms": round_ms,
                     "draft_ms": draft, "d_per_draft": draft / max(k, 1), "rest_ms": round_ms - draft,
                     "delivered": round_ms / e if e else float("nan"), "tps": med(lambda r: r.get("tps") or 0.0),
                     # per arm, from ITS OWN reps: reading this off the loop variable left over from
                     # the grouping pass made every arm carry the last row's digest (caught by the
                     # selftest's two-build case, which then fitted across builds)
                     "digests": {tuple(sorted((x.get("engine_digest") or {}).items())) for x in rs} - {()}})
    arms.sort(key=lambda x: x["k"])
    have = [d for a in arms for d in a["digests"]]
    res = {"arms": arms, "mixed_build": len(set(have)) > 1,
           "partial_digest": any(not a["digests"] for a in arms), "target_ms": target_ms}
    if res["mixed_build"] or len(arms) < 2:
        res["fit"] = None
        return res
    # A two-point fit is exactly identified, and `_ols` refuses below three points on purpose (it is
    # used for the step channels, where a slope has to be earned). So the line is fitted here.
    pts = [(a["T"], a["rest_ms"]) for a in arms]
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        res["fit"] = None
        return res
    m = sum((x - mx) * (y - my) for x, y in pts) / den
    F = my - m * mx
    base = arms[0]
    for a in arms[1:]:
        a["break_even_E"] = (a["draft_ms"] + a["rest_ms"]) / base["delivered"]
        a["margin_pct"] = 100.0 * (a["E"] / a["break_even_E"] - 1.0) if a["break_even_E"] else float("nan")
    res.update({"fit": {"m": m, "F": F, "dof": len(arms) - 2},
                "intervals": [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)],
                "ceiling": {a["arm"]: (a["draft_ms"] + a["rest_ms"]) / a["T"] for a in arms},
                "budget": {a["arm"]: a["E"] * target_ms - a["draft_ms"] for a in arms},
                "base": base["arm"]})
    return res


K_ECON_KMAX = 8
K_ECON_ALPHAS = tuple(x / 100.0 for x in range(35, 101, 5))


def k_econ_accept(res: dict, kmax: int = K_ECON_KMAX, alphas=K_ECON_ALPHAS) -> tuple:
    """The k that minimises delivered ms/token, and how that choice moves with the accept rate.

    Two of the three terms are measured -- the round budget fitted as `rest = F + m*T`, and the draft
    chain `D(k)` through the arms' own draft_ms. The third has to be modelled, and it is the accept
    rate:

        E(k, a)         = 1 + a + ... + a^k          a verify token survives only if its predecessors
        delivered(k, a) = (D(k) + F + m*(k+1)) / E(k, a)     did, which is the geometric sum

    The arms' own E is deliberately NOT taken as `a`: on llama-bench it sits at its ceiling (the k=1
    arm emits E = 2.000, i.e. a = 1.0) because the synthetic cell repeats tokens, so that alpha is
    not the server's. The answer is the curve over `a`, the sensitivity is where its argmin moves,
    and a k that is never optimal is reported as a hole rather than interpolated over.
    """
    fit = res.get("fit")
    if fit is None:
        return None, "no fit (needs >= 2 arms on one build)"
    pts = [(a["k"], a["draft_ms"]) for a in res["arms"] if a["k"] >= 1]
    if len(pts) < 2:
        return None, "the draft chain needs >= 2 arms carrying draft_ms"
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return None, "every arm has the same k -- the draft chain has no slope"
    d1 = sum((x - mx) * (y - my) for x, y in pts) / den
    d0 = my - d1 * mx
    F, m = fit["F"], fit["m"]
    ks = list(range(1, kmax + 1))

    def deliv(k, a):
        return (d0 + d1 * k + F + m * (k + 1)) / sum(a ** i for i in range(k + 1))

    def best(a):
        ms, k = min((deliv(k, a), k) for k in ks)
        return k, ms

    curve = []
    for a in alphas:
        k, ms = best(a)
        curve.append({"alpha": a, "k": k, "ms": ms, "tps": 1000.0 / ms,
                      "ms_at": {kk: deliv(kk, a) for kk in ks}})
    # The argmin as a function of a, on a grid fine enough to locate the transitions rather than
    # bracket them, compressed into runs so "which k, from which accept rate" is directly readable.
    fine = [i / 1000.0 for i in range(1001)]
    runs = []
    for a in fine:
        k = best(a)[0]
        if runs and runs[-1]["k"] == k:
            runs[-1]["hi"] = a
        else:
            runs.append({"k": k, "lo": a, "hi": a})
    best_tps = {a: 1000.0 / best(a)[1] for a in fine}
    targets = {}
    for t in (12.0, 15.0, 20.0, 25.0):
        hit = [a for a in fine if best_tps[a] >= t]
        targets[t] = hit[0] if hit else None
    return {"fit": fit, "draft": {"d0": d0, "d1": d1, "n": len(pts)}, "kmax": kmax,
            "curve": curve, "runs": runs, "targets": targets,
            "tps_at": best_tps, "perfect_accept": dict(zip(("k", "ms"), best(1.0)))}, None


def print_k_econ(res: dict, path: Path, schema: str) -> None:
    print(f"\n  k economics: the round's fixed work, and what amortising it buys")
    print(f"    {path}  schema={schema}" + (
        f"  [derived: {res['arms'][0].get('derived')}]" if res["arms"] and res["arms"][0].get("derived") else ""))
    if res["mixed_build"]:
        print(f"    !! arms span more than one engine digest -- no fit across builds; per-arm rows only")
    if res.get("partial_digest"):
        print("    note: not every arm carries an engine digest, so 'one build' is not established "
              "for this fit")
    print(f"    {'k':>2} {'T':>2} {'reps':>4} {'E':>7} {'round':>8} {'draft':>8} {'d/draft':>8} "
          f"{'rest=F+mT':>10} {'delivered':>10} {'tps':>7}")
    for a in res["arms"]:
        print(f"    {a['k']:>2} {a['T']:>2} {a['n']:>4} {a['E']:>7.3f} {a['round_ms']:>8.1f} {a['draft_ms']:>8.1f} "
              f"{a['d_per_draft']:>8.2f} {a['rest_ms']:>10.1f} {a['delivered']:>10.1f} {a['tps']:>7.2f}")
    fit = res["fit"]
    if fit is None:
        print(f"    no fit (needs >=2 arms on one build)")
        return
    print(f"    fit rest = F + m*T  ->  m = {fit['m']:+.2f} ms per verify token, "
          f"F = {fit['F']:.1f} ms per round   ({len(res['arms'])} arms, dof {fit['dof']})")
    if fit["dof"] == 0:
        print(f"      !! dof 0: two arms and two parameters describe these numbers and cannot test them. "
              f"Quote m next to the within-arm slope (+51.0 on the 09-22 pair), which is another "
              f"instrument; add a k=2 arm to spend the missing degree of freedom.")
    for (x0, y0), (x1, y1) in res["intervals"]:
        print(f"    interval T={x0}->{x1}: rest {y0:.1f} -> {y1:.1f}  (+{y1 - y0:.1f}, "
              f"{(y1 - y0) / (x1 - x0):+.2f}/token)")
    base = res["base"]
    b = next(a for a in res["arms"] if a["arm"] == base)
    print(f"    amortisation: the round's fixed work is {100 * (fit['F'] / b['E']) / b['delivered']:.0f}% "
          f"of the {base} delivered cost ({fit['F'] / b['E']:.1f} of {b['delivered']:.1f} ms/token, "
          f"because F = {fit['F']:.1f} per round is paid once for E = {b['E']:.3f} tokens) -- that is "
          f"the whole reason k is a lever")
    for a in res["arms"][1:]:
        print(f"    break-even: {a['arm']} beats {base} while E > {a['break_even_E']:.3f} "
              f"(measured {a['E']:.3f} -> margin {a['margin_pct']:+.1f}%)")
    print(f"    ceiling at perfect accept (E = T): "
          + "  ".join(f"{a} {res['ceiling'][a]:.1f} ms/token = {1000 / res['ceiling'][a]:.1f} t/s"
                      for a in sorted(res["ceiling"])))
    print(f"    target {res['target_ms']:.1f} ms/token ({1000 / res['target_ms']:.0f} t/s) needs "
          f"F + m*T <= E*target - draft at the measured E: "
          + "  ".join(f"{a} budget {res['budget'][a]:.1f} vs measured {x['rest_ms']:.1f} "
                      f"(gap {x['rest_ms'] - res['budget'][a]:.1f})"
                      for a, x in ((a['arm'], a) for a in res["arms"])))
    a0, a1 = res["arms"][0], res["arms"][-1]
    print(f"    the two anchors, measured: inside a round an extra verify token costs m = {fit['m']:+.1f} "
          f"ms, while the DELIVERED contrast {a1['arm']} vs {a0['arm']} is only "
          f"{a1['delivered'] - a0['delivered']:+.1f} ms/token -- the difference is F, spread over "
          f"{a0['E']:.2f} -> {a1['E']:.2f} tokens")
    acc, why = k_econ_accept(res)
    if acc is None:
        print(f"    the k decision vs accept rate: not available -- {why}")
        return
    print(f"\n    the k decision as a function of the ACCEPT RATE (the two measured terms are the fit "
          f"above and the arms' draft chain, D(k) = {acc['draft']['d0']:.1f} + {acc['draft']['d1']:.2f}*k "
          f"from {acc['draft']['n']} arms; E(k,a) = 1+a+...+a^k):")
    shown = (1, 2, 3)
    print(f"      {'alpha':>6} {'k*':>3} {'delivered':>10} {'t/s':>7}   "
          + "  ".join(f"k={k}" for k in shown))
    for c in acc["curve"]:
        print(f"      {c['alpha']:>6.2f} {c['k']:>3} {c['ms']:>10.1f} {c['tps']:>7.2f}   "
              + "  ".join(f"{c['ms_at'][k]:>4.0f}" for k in shown))
    holes = [k for k in range(1, acc["kmax"] + 1)
             if not any(r["k"] == k for r in acc["runs"])]
    print(f"    sensitivity: " + "; ".join(
        f"k*={r['k']} for a in [{r['lo']:.3f}, {r['hi']:.3f}]" for r in acc["runs"])
        + (f"; never optimal: {holes} (the model walks past them, so no measurement of them can "
           f"move the choice)" if holes else ""))
    print(f"    what accept buys: " + "  ".join(
        (f"{t:.0f} t/s from a >= {acc['targets'][t]:.2f}" if acc["targets"][t] is not None
         else f"{t:.0f} t/s unreachable at any a") for t in (15.0, 20.0, 25.0)))
    print(f"    ceiling at perfect accept: k={acc['perfect_accept']['k']} "
          f"{acc['perfect_accept']['ms']:.1f} ms/token = "
          f"{1000 / acc['perfect_accept']['ms']:.1f} t/s -- an accept-only programme stops here")


def report(pairs: dict, per_miss: float, layer_pairs: dict | None = None, dt: int = 2,
           step_pairs: dict | None = None, step_phase: str | None = None) -> int:
    ok = {r: p for r, p in pairs.items() if not p.get("invalid")}
    for rep, p in sorted(pairs.items()):
        if p.get("invalid"):
            print(f"  rep {rep + 1}: refused -- {p['invalid']}")
    if not ok:
        print("\nno usable rep pair -- nothing to report")
        return 1

    print(f"\nmarginal verify token, per round, differenced over dT and split three ways"
          f"\n  borrowed constant PER_MISS_MS={per_miss} ms/miss (F1; range {PER_MISS_RANGE[0]}-"
          f"{PER_MISS_RANGE[1]})")
    print(f"\n  {'rep':>3} {'round T=2':>10} {'round T=4':>10} {'marginal/tok':>13} "
          f"{'miss':>8} {'compute':>8} {'host':>8}")
    for rep, p in sorted(ok.items()):
        print(f"  {rep + 1:>3} {p['round1']:>10.1f} {p['round3']:>10.1f} {p['marginal_total']:>+13.2f} "
              f"{p['marginal_miss']:>+8.2f} "
              f"{(p['marginal_compute'] if p['marginal_compute'] == p['marginal_compute'] else float('nan')):>+8.2f} "
              f"{p['marginal_host']:>+8.2f}")
    print(f"  (ms/token. misses: {ok[sorted(ok)[0]]['miss_per_round1']:.1f} -> "
          f"{ok[sorted(ok)[0]]['miss_per_round3']:.1f} per round in rep {sorted(ok)[0] + 1})")

    sh = [shares(p) for p in ok.values()]
    med = {k: sorted(x[k] for x in sh if x[k] is not None) for k in ("miss", "compute", "host")}
    med = {k: (v[len(v) // 2] if v else None) for k, v in med.items()}

    def pc(x):
        return f"{100 * x:.1f}%" if x is not None else "n/a"

    print(f"\n  MEDIAN shares: miss {pc(med['miss'])}  compute {pc(med['compute'])}  "
          f"host {pc(med['host'])}")
    print(f"  RULE -> {verdict(med)}")

    # Concentration: the miss axis is per-layer, so say who owns it -- actionable only on that side.
    p0 = ok[sorted(ok)[0]]
    d = {l: p0["misses_by_layer3"].get(l, 0) - p0["misses_by_layer1"].get(l, 0)
         for l in set(p0["misses_by_layer1"]) | set(p0["misses_by_layer3"])}
    tot = sum(abs(v) for v in d.values()) or 1
    top = sorted(d.items(), key=lambda x: -x[1])[:8]
    print(f"\n  miss axis by layer (rep {sorted(ok)[0] + 1}): total delta {sum(d.values()):+d} over "
          f"{len(d)} layers")
    print("    " + "  ".join(f"L{l}:{v:+d}" for l, v in top))
    print(f"    the largest 8 of {len(d)} layers hold {100 * sum(v for _, v in top) / tot:.0f}% of "
          f"the |miss delta|")

    if p0["vop_us_round1"] is not None and p0["vop_us_round3"] is not None:
        print(f"\n  verify matmul per round: {p0['vop_us_round1']:.1f} -> {p0['vop_us_round3']:.1f} ms"
              f"   (us/layer-token by kind, from CGC-VERIFY-OP)")
        print("    (the per-layer cb table needs a DECPROF-armed pair; this pair was armed with "
              "CGC_VERIFY_OP_TIMING, which is mutually exclusive with it)")

    # Per-layer cb axis, when the pair carried DECPROF. Reported WITH its own audit against the
    # round-level marginal, because the per-layer rows are 1-in-8 sampled medians and will not sum
    # to the step difference exactly -- a reader has to be able to see by how much.
    if layer_pairs:
        runs = layer_pairs[min(layer_pairs)]
        lt, why = layer_table(runs[0], runs[1], dt, per_miss)
        if lt is None:
            print(f"\n  per-layer table not available: {why}")
        else:
            tot_marg = sum(r["marg"] for r in lt)
            tot_miss = sum(r["miss_ms"] for r in lt if r["miss_ms"] == r["miss_ms"])
            axis_tot = {k: sum(r[k] for r in lt) for k, _ in AXES}
            print(f"\n  per-layer marginal (DECPROF, per token: /dT={dt}), {len(lt)} layers, "
                  f"per-layer MEAN (a median step does not carry a marginal cost)")
            print("    step terms summed over layers: "
                  + "  ".join(f"{name} {axis_tot[k]:+.2f}" for k, name in AXES)
                  + f"  => total {tot_marg:+.2f} ms/token")
            print(f"    miss axis from BATCHDBG: {tot_miss:+.2f} ms/token (borrowed {per_miss} ms/miss)")
            # The audit that makes this table quotable: the per-layer rows are medians of a sampled
            # population, so their sum is NOT the round-level marginal by construction. Showing the
            # gap is the difference between "the decomposition covers it" and "the decomposition is
            # of a different quantity".
            ref = ok[sorted(ok)[0]]["marginal_total"]
            print(f"    audit: round-level marginal in rep {sorted(ok)[0] + 1} is {ref:+.2f} ms/token; "
                  f"per-layer sum is {tot_marg:+.2f} ({100 * tot_marg / ref if ref else float('nan'):.0f}%)"
                  if ref else "")
            print("    top 8 layers by |eval-sync marginal|, with all three axes:")
            print(f"      {'layer':>6} {'wait':>8} {'gather':>8} {'dispatch':>9} {'sum':>8} "
                  f"{'miss':>8} {'resid':>8}")
            for r in sorted(lt, key=lambda x: -abs(x["wait"]))[:8]:
                print(f"      L{r['layer']:<5} {r['wait']:+8.3f} {r['cb']:+8.3f} {r['submit']:+9.3f} "
                      f"{r['marg']:+8.3f} {r['miss_ms']:+8.3f} {r['resid']:+8.3f}")
            # Which axis owns the top: the three have different levers, so the answer decides the work.
            dom = max(((k, abs(v)) for k, v in axis_tot.items()), key=lambda x: x[1])
            print(f"    dominant axis: {dict(AXES)[dom[0]]} ({dom[1]:.2f} ms/token of "
                  f"{sum(abs(v) for v in axis_tot.values()):.2f} moved)")

            # ROUND BUDGET CLOSURE. The per-layer table above answers "which term moves", and it can
            # come out near zero while the round moves a lot -- in which case the honest reading is
            # "the marginal is not in these terms at all", and the reader needs to see where it IS.
            # Every term here is measured (MTP-PERF + DECPROF); the remainder is what neither
            # instrument covers, named rather than spread over the layers.
            print()
            print_budget(round_budget(lt, runs[0], runs[1], dt), ref)

    # ... and the closure for EVERY usable rep, so the remainder's share is a reading, not one
    # lucky pair. The per-layer detail above stays single-rep because it is 8 rows x 3 axes.
    if layer_pairs:
        print("\n  round budget closure, every usable rep")
        for rep in sorted(ok):
            lt_r, why = layer_table(*layer_pairs[rep], dt, per_miss)
            if lt_r is None:
                print(f"    rep {rep + 1}: {why}")
                continue
            print_budget(round_budget(lt_r, layer_pairs[rep][0], layer_pairs[rep][1], dt),
                         ok[rep]["marginal_total"], label=f"rep {rep + 1}: ")
    # The step-channel split, when asked for. Reported even for pairs the round-level axes refused,
    # because it needs neither the miss axis nor a round_ms: a reader asking "where does the extra
    # verify token go" should not be blocked by a BATCHDBG knob they did not arm.
    if step_pairs:
        rep = sorted(step_pairs)[0]
        rows = channel_split(*step_pairs[rep], dt, step_phase)
        if rows.get("invalid"):
            print(f"\n  step-channel split not available: {rows['invalid']}")
        else:
            print_step_channels(rows, dt, step_phase)
    return 0


def selftest() -> int:
    bad = 0

    def check(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + ("" if ok else f"  got {got!r} want {want!r}"))

    import tempfile

    def arm_dir(tmp, name, rounds, emit, dec_misses, pre_miss, vop_round_ms, boundary=True):
        """A synthetic arm in its own directory: `rounds` rounds, `dec_misses` misses in decode."""
        d = tmp / name
        d.mkdir()
        lines = []
        if pre_miss:
            # Before the boundary, so it IS prefill: the phase rule must drop it.
            lines.append(f"BATCHDBG layer=0 misses={pre_miss} slots: e1->s2\n")
        if boundary:
            lines.append("CGC-POST: done\n")
        # Counters are integers in the engine's own line, so the fixture's must be integers too:
        # `acc_tokens=192.0` does not match and the fixture would test the parser's failure mode.
        lines.append(f"CGC-MTP-PERF type=draft-mtp calls_begin=1 calls_draft={rounds} "
                     f"calls_accept=1 gen_tokens={rounds} acc_tokens={int(rounds * (emit - 1))} "
                     f"t_begin_ms=0.0 t_draft_ms=0.0 t_accept_ms=0.0 acc_rate=1.0 "
                     f"gen_tok_per_round=1.0 acc_tok_per_round=1.0 "
                     f"emit_tok_per_round={emit} ms_per_round=0.0\n")
        for l, n in sorted(dec_misses.items()):
            lines.append(f"BATCHDBG layer={l} misses={n} slots: e1->s2\n")
        # one row per round, 8 tokens per row, so us/round = vop_round_ms * 1000
        lines.append(f"CGC-VERIFY-OP: kind=down us={int(vop_round_ms * rounds * 1000)} "
                     f"ntok={8 * rounds}\n")
        (d / "x.stderr.log").write_text("".join(lines))
        return d

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # T=2 arm: 192 rounds, 10 misses/round decode, 2.0 ms/round verify compute
        d1 = arm_dir(tmp, "t2", 192, 2.0, {0: 960, 5: 960}, pre_miss=99, vop_round_ms=2.0)
        # T=4 arm:  96 rounds, 20 misses/round decode, 4.0 ms/round verify compute
        d3 = arm_dir(tmp, "t4", 96, 4.0, {0: 1920, 5: 0}, pre_miss=0, vop_round_ms=4.0)
        a1 = arm_axes({"dir": str(d1)})
        check("prefill miss excluded from the decode count", a1["misses_dec"], 1920)
        check("misses per round", round(a1["miss_per_round"], 4), 10.0)
        check("rounds come from calls_draft", a1["rounds"], 192)
        check("emitted tokens", a1["emitted"], 2 * 192)
        check("verify compute per round (ms)", round(a1["vop_us_per_round"], 4), 2.0)
        check("verify compute us per layer-token", round(a1["vop"]["down"]["us_per_tok"], 3), 250.0)

        # end to end: marginal_total = (500-300)/2 = 100
        #   miss    = (20-10) * 0.695 / 2        = 3.475
        #   compute = (4.0-2.0)/2                = 1.0
        #   host    = 100 - 3.475 - 1.0          = 95.525
        p = pair_rep({"dir": str(d1), "round_ms": 300.0}, {"dir": str(d3), "round_ms": 500.0},
                     2, PER_MISS_MS)
        check("pair: no refusal", p.get("invalid"), None)
        check("marginal total", round(p["marginal_total"], 4), 100.0)
        check("marginal miss", round(p["marginal_miss"], 4), 3.475)
        check("marginal compute", round(p["marginal_compute"], 4), 1.0)
        check("marginal host", round(p["marginal_host"], 4), 95.525)
        s = shares(p)
        check("share: miss", round(s["miss"], 5), 0.03475)
        check("share: host", round(s["host"], 5), 0.95525)
        check("verdict HOST-BOUND", verdict(s),
              "HOST-BOUND -- lever is the per-token host path (dispatch/slot/submit), NOT misses "
              "and NOT the matmul")
        check("verdict MISS-BOUND at 0.60", verdict({"miss": 0.60, "compute": 0.2, "host": 0.2}),
              "MISS-BOUND -- lever is coverage / routing locality")
        check("verdict COMPUTE-BOUND at 0.70", verdict({"miss": 0.1, "compute": 0.7, "host": 0.2}),
              "COMPUTE-BOUND -- lever is the verify op path")
        check("verdict MIXED", verdict({"miss": 0.3, "compute": 0.3, "host": 0.4}),
              "NO SINGLE LEVER -- report all three")
        check("verdict none", verdict({"miss": None, "compute": None, "host": None}),
              "NO VERDICT (no marginal detected)")

        # refusals
        nb = arm_dir(tmp, "nobound", 10, 2.0, {0: 5}, pre_miss=0, vop_round_ms=1.0, boundary=False)
        check("refuse: no boundary", "CGC-POST" in arm_axes({"dir": str(nb)})["invalid"], True)
        nbat = tmp / "nobat"
        nbat.mkdir()
        (nbat / "y.stderr.log").write_text(
            "CGC-POST: d\nCGC-MTP-PERF type=draft-mtp calls_begin=1 calls_draft=10 calls_accept=1 "
            "gen_tokens=10 acc_tokens=9 t_begin_ms=0.0 t_draft_ms=0.0 t_accept_ms=0.0 acc_rate=1.0 "
            "gen_tok_per_round=1.0 acc_tok_per_round=1.0 emit_tok_per_round=1.9 ms_per_round=0.0\n")
        check("refuse: no miss axis", "BATCHDBG" in arm_axes({"dir": str(nbat)})["invalid"], True)
        check("refuse: empty dir",
              "no *.stderr.log" in arm_axes({"dir": str(tmp / "nope")})["invalid"], True)
        check("refuse: same round count",
              "not different" in pair_rep({"dir": str(d1), "round_ms": 300.0},
                                          {"dir": str(d1), "round_ms": 500.0},
                                          2, PER_MISS_MS)["invalid"], True)

        # The per-layer decomposition. A DECPROF row only counts if its aggregate announced the
        # 41-segment graph, so the fixture writes that line too -- otherwise the fixture would pass
        # by exercising the exclusion path instead of the parser.
        def dec_dir(tmp, name, steps: dict) -> Path:
            d = tmp / name
            d.mkdir()
            body = ["CGC-POST: d\n", "CGC-DECPROF: step=0 segs=41 layers=41\n"]
            for l, (w, c, s) in sorted(steps.items()):
                body.append(f"CGC-DECPROF all: L{l} wait={w} cb={c} submit={s} ms\n")
            (d / "x.stderr.log").write_text("".join(body))
            return d

        e1 = dec_dir(tmp, "dec2", {0: (2.0, 1.0, 0.1), 1: (1.0, 0.5, 0.05)})
        e3 = dec_dir(tmp, "dec4", {0: (4.0, 3.0, 0.3), 1: (2.0, 0.5, 0.05)})
        lt, why = layer_table({"dir": str(e1)}, {"dir": str(e3)}, 2, PER_MISS_MS)
        check("layer_table: no refusal", why, None)
        rows_by = {r["layer"]: r for r in (lt or [])}
        check("layer_table: wait marginal", round(rows_by[0]["wait"], 6), 1.0)
        check("layer_table: gather marginal", round(rows_by[0]["cb"], 6), 1.0)
        check("layer_table: dispatch marginal", round(rows_by[0]["submit"], 6), 0.1)
        check("layer_table: row marginal is the three summed", round(rows_by[0]["marg"], 6), 2.1)
        check("layer_table: unchanged layer is flat", round(rows_by[1]["marg"], 6), 0.5)
        check("layer_table: all layers present", sorted(rows_by), [0, 1])
        # an arm with no DECPROF at all must refuse by name, not silently return an empty table
        check("refuse: no per-layer rows",
              "no per-layer cb rows" in layer_table({"dir": str(d1)}, {"dir": str(e3)}, 2,
                                                    PER_MISS_MS)[1], True)

        # ── the step-channel split. The fixture writes the two lines the engine writes together
        # (the step line and its GPU tail) plus the label pair, and the expected slope is arithmetic:
        # union 10 ms per token, cb 1 ms, submit 0.1 ms.
        def step_dir(tmp, name, widths: dict, phases: dict | None = None, tail: bool = True) -> Path:
            d = tmp / name
            d.mkdir()
            body = []
            stepno = 0
            for w, n in sorted(widths.items()):
                for _ in range(n):
                    stepno += 1
                    # ENGINE ORDER, not the convenient one: the phase labels are emitted while layers
                    # 0-1 are encoded, so they precede the step line they describe. A fixture that
                    # wrote them after would pass while testing an alignment the engine never emits.
                    if phases:
                        ph = phases[w]
                        for il in (0, 1):
                            body.append(f"CGC-PHASE-DBG: il={il} phase={ph} n_past=500 n_tokens={w} "
                                        f"warm_gate=0 verify_fast=1 draft_fast=1 fast_eligible=1\n")
                    # total is the SUM of the three terms by construction, so the closure check in the
                    # tool is exercised rather than assumed.
                    body.append(f"CGC-DECPROF: step={stepno} segs=41 layers=40 total={20 + 11.1 * w:.2f} "
                                f"ms | wait={15 + 10 * w:.2f} (70%) cb={5 + 1 * w:.2f} (20%) "
                                f"submit={0.1 * w:.2f} (1%) ntok={w}")
                    if tail:
                        body.append(f" | layer gpu_sum={12 + 12 * w:.2f} union_sum={10 * w:.2f} "
                                    f"gap_sum={1.0 * w:.2f} ms")
                    body.append("\n")
            (d / "x.stderr.log").write_text("".join(body))
            return d

        s2 = step_dir(tmp, "st2", {2: 30}, phases={2: "VERIFY"})
        s4 = step_dir(tmp, "st4", {2: 10, 3: 10, 4: 30}, phases={2: "VERIFY", 3: "VERIFY", 4: "VERIFY"})
        ch = channel_split({"dir": str(s2)}, {"dir": str(s4)}, 2, "VERIFY")
        check("step split: no refusal", ch.get("invalid"), None)
        check("step split: primary estimator is the within-arm slope", ch["primary"],
              "within-arm slope")
        check("step split: device span slope", round(ch["slope"]["union"], 6), 10.0)
        check("step split: host hook slope", round(ch["slope"]["cb"], 6), 1.0)
        check("step split: dispatch slope", round(ch["slope"]["submit"], 6), 0.1)
        check("step split: wait slope", round(ch["slope"]["wait"], 6), 10.0)
        check("step split: total slope", round(ch["slope"]["total"], 6), 11.1)
        check("step split: closure total = wait+cb+submit",
              round(ch["slope"]["total"] - ch["slope"]["wait"] - ch["slope"]["cb"]
                    - ch["slope"]["submit"], 6), 0.0)
        check("step split: per-width medians", round(ch["T4"]["med"][4]["union"], 6), 40.0)
        check("step split: width counts", ch["T4"]["counts"], {2: 10, 3: 10, 4: 30})
        # the head can OUTNUMBER the trunk (measured in the L3 dense arm: 4400 vs 231), so "modal" is
        # the wrong rule and the fixture must contain that shape rather than the convenient one.
        hd = tmp / "headheavy"
        hd.mkdir()
        body = []
        for i in range(1, 61):
            body.append(f"CGC-DECPROF: step={i} segs=2 layers=1 total=1.0 ms | wait=1.0 (90%) cb=0.0 "
                        f"(1%) submit=0.1 (9%) ntok=4 | layer gpu_sum=1.0 union_sum=0.5 gap_sum=0.0 ms\n")
        body.append("".join(
            f"CGC-DECPROF: step={100 + i} segs=41 layers=40 total={20 + 11.1 * w:.2f} ms | "
            f"wait={15 + 10 * w:.2f} (70%) cb={5 + 1 * w:.2f} (20%) submit={0.1 * w:.2f} (1%) ntok={w} | "
            f"layer gpu_sum={12 + 12 * w:.2f} union_sum={10 * w:.2f} gap_sum={1.0 * w:.2f} ms\n"
            for w, n in ((2, 10), (4, 20)) for i in range(n)))
        (hd / "h.stderr.log").write_text("".join(body))
        hfam, hwhy = step_family(hd / "h.stderr.log")
        check("step split: the trunk is the largest family, not the modal one",
              (hwhy, len(hfam) if hfam else None, hfam[0]["layers"] if hfam else None), (None, 30, 40))
        amb = tmp / "ambiguous"
        amb.mkdir()
        (amb / "a.stderr.log").write_text(
            "".join(f"CGC-DECPROF: step={i} segs=41 layers={L} total=20.0 ms | wait=15.0 (70%) cb=5.0 "
                    f"(20%) submit=0.5 (2%) ntok=4 | layer gpu_sum=12.0 union_sum=10.0 gap_sum=1.0 ms\n"
                    for L in (20, 40) for i in range(10)))
        check("refuse: two unseparated layer families",
              "not separated by 4x" in step_family(amb / "a.stderr.log")[1], True)
        check("step split: labels claimed", ch["T4"]["labeled"], 50)
        check("step split: verdict is device-led",
              channel_verdict(ch["slope"], ch["slope"]["total"]), "UNION-LED -- 90% of the marginal token")
        # the phase filter must be able to EMPTY the family: DRAFT labels on a VERIFY request
        # a graph with the same ntok as its predecessor must not inherit the predecessor's label
        mis = tmp / "misalign"
        mis.mkdir()
        (mis / "m.stderr.log").write_text(
            "CGC-PHASE-DBG: il=0 phase=VERIFY n_past=1 n_tokens=4 warm_gate=0 verify_fast=1 draft_fast=1 "
            "fast_eligible=1\n"
            "CGC-PHASE-DBG: il=1 phase=VERIFY n_past=1 n_tokens=4 warm_gate=0 verify_fast=1 draft_fast=1 "
            "fast_eligible=1\n"
            "CGC-DECPROF: step=1 segs=41 layers=40 total=31.10 ms | wait=25.00 (80%) cb=6.00 (19%) "
            "submit=0.10 (0%) ntok=4 | layer gpu_sum=52.00 union_sum=40.00 gap_sum=4.00 ms\n"
            # no labels for the second graph, and the third is NORMAL_DECODE
            "CGC-DECPROF: step=2 segs=41 layers=40 total=31.10 ms | wait=25.00 (80%) cb=6.00 (19%) "
            "submit=0.10 (0%) ntok=4 | layer gpu_sum=52.00 union_sum=40.00 gap_sum=4.00 ms\n"
            "CGC-PHASE-DBG: il=0 phase=NORMAL_DECODE n_past=2 n_tokens=4 warm_gate=0 verify_fast=0 "
            "draft_fast=0 fast_eligible=0\n"
            "CGC-PHASE-DBG: il=1 phase=NORMAL_DECODE n_past=2 n_tokens=4 warm_gate=0 verify_fast=0 "
            "draft_fast=0 fast_eligible=0\n"
            "CGC-DECPROF: step=3 segs=41 layers=40 total=31.10 ms | wait=25.00 (80%) cb=6.00 (19%) "
            "submit=0.10 (0%) ntok=4 | layer gpu_sum=52.00 union_sum=40.00 gap_sum=4.00 ms\n")
        lbl = [r["phase"] for r in step_rows(mis / "m.stderr.log")]
        check("phase labels land on their own step, in engine order", lbl,
              ["VERIFY", None, "NORMAL_DECODE"])
        check("phase filter keeps only the labeled VERIFY step",
              len(step_family(mis / "m.stderr.log", "VERIFY")[0]), 1)
        sd = step_dir(tmp, "std", {4: 30}, phases={4: "DRAFT"})
        check("refuse: phase filter has no member",
              "no trunk step" in channel_split({"dir": str(s2)}, {"dir": str(sd)}, 2, "VERIFY")["invalid"],
              True)
        # a single-width arm has no slope, so the tool must fall back and SAY so
        ch3 = channel_split({"dir": str(s2)}, {"dir": str(s2)}, 2, "VERIFY")
        check("step split: single-width T4 falls back to the contrast", ch3["primary"], "arm contrast")
        # no GPU tail -> the device side is absent, and a channel share would be a share of nothing
        snt = step_dir(tmp, "stnt", {2: 30, 4: 30}, tail=False)
        check("refuse: no GPU tail",
              "no GPU tail" in channel_split({"dir": str(snt)}, {"dir": str(s4)}, 2)["invalid"], True)
        # no step lines at all (a VERIFY-OP pair): refuse by name, do not return an empty split
        check("refuse: no step rows",
              "no CGC-DECPROF step rows" in channel_split({"dir": str(d1)}, {"dir": str(d3)}, 2)["invalid"],
              True)
        # the modal-layers rule: a log whose only family is the 1-layer head has no trunk
        head = tmp / "head"
        head.mkdir()
        (head / "h.stderr.log").write_text(
            "".join(f"CGC-DECPROF: step={i} segs=2 layers=1 total=1.0 ms | wait=1.0 (90%) cb=0.0 (1%) "
                    f"submit=0.1 (9%) ntok=4 | layer gpu_sum=1.0 union_sum=0.5 gap_sum=0.0 ms\n"
                    for i in range(1, 20)))
        check("refuse: no trunk family",
              "no trunk family" in channel_split({"dir": str(head)}, {"dir": str(s4)}, 2)["invalid"], True)
        # ── node attribution: one fixture whose gates pass, and three that must be refused for the
        #    three ways the REAL logs fail (a decimated sample, a one-step wide cell, a misparse).
        def ops_log(name, widths, sampled, reversed_rows=False):
            d = tmp / name
            d.mkdir()
            out = []
            for i, w in enumerate(widths, start=1):
                out.append(f"CGC-DECPROF: step={i} segs=41 layers=40 total={25 * w + 10:.1f} ms | "
                           f"wait=90.0 (90%) cb=8.0 (8%) submit=2.0 (2%) ntok={w} | "
                           f"layer gpu_sum=80.0 union_sum=70.0 gap_sum=5.0 ms\n")
                if i in sampled:
                    rows = [(20.0 * w, 20.0, "MUL_MAT"), (5.0 * w, 10.0, "ADD")]
                    if reversed_rows:
                        rows.reverse()
                    out.append(f"CGC-GPUOPS: step={i} total={25 * w + 10:.1f} ms nodes_all=20 "
                               f"nodes_work=20 nodes_named=20 nop=2\n")
                    for wc, ub, op in rows:
                        out.append(f"CGC-GPUOPS:   {op:<16s} nd=    10 work wcntw={wc:8.2f}   0.0%  "
                                   f"cntw={wc:8.2f}   0.0%  ub={ub:8.2f}   0.0%  uni=  -1.000\n")
            (d / f"{name}.stderr.log").write_text("".join(out))
            return d

        every = set(range(1, 19))
        balanced = ops_log("ops_bal", [2] * 6 + [3] * 6 + [4] * 6, every)
        na, why = node_attribution(balanced / "ops_bal.stderr.log")
        check("node-attr: balanced sample passes both gates", (na["gates"], why), (True, None))
        check("node-attr: per-op slope is the fixture's", round(na["sum_slope"], 2), 25.0)
        check("node-attr: every printed row carries a slope (2 ops + residual)", na["n_op_slopes"], 3)
        check("node-attr: table total slope matches the op sum", round(na["total_slope"], 2), 25.0)
        # the fixture's residual is 10 ms against totals of 25w+10, so the median PERCENT is what has
        # to come back (10/85 = 11.8% on the middle width), not the 10 ms itself
        check("node-attr: residual of the fixture", round(na["residual_pct"], 0), 12.0)

        alias = ops_log("ops_alias", [2] * 6 + [3] * 6 + [4] * 18, set(range(7, 25)))
        na2, _ = node_attribution(alias / "ops_alias.stderr.log")
        check("node-attr: decimated sample is refused", na2["gates"], False)
        check("node-attr: refusal names the mix deviation",
              max(na2["mix_dev"].values()) > na2["mix_tol"], True)

        thin = ops_log("ops_thin", [2] * 6 + [4] * 30, {1} | set(range(7, 17)))
        na3, _ = node_attribution(thin / "ops_thin.stderr.log")
        check("node-attr: a one-sample wide cell is refused", (na3["gates"], na3["thin"]), (False, [2]))

        unord = ops_log("ops_unord", [4] * 12, set(range(1, 13)), reversed_rows=True)
        na4, _ = node_attribution(unord / "ops_unord.stderr.log")
        check("node-attr: a non-ub-descending table is refused", (na4["gates"], len(na4["bad_rank"])), (False, 12))

        # ── k economics: rest = F + m*T with a deliberately convex term, so the third arm spends the
        #    degree of freedom and the interval slopes must show the deviation rather than average it.
        econ_rows = [dict(arm=f"k{k}", k=k, rep=0, mean_len=e, draft_ms=10.0,
                          round_ms=rest + 10.0, tps=1000.0 * e / (rest + 10.0))
                     for k, e, rest in ((1, 2.0, 200.0), (2, 2.5, 258.0), (3, 3.0, 332.0))]
        ec = k_econ(econ_rows, 40.0)
        check("k-econ: three arms spend the degree of freedom", ec["fit"]["dof"], 1)
        check("k-econ: intervals expose the convexity",
              [round((y1 - y0) / (x1 - x0), 1) for (x0, y0), (x1, y1) in ec["intervals"]], [58.0, 74.0])
        check("k-econ: break-even against the base arm's delivered cost",
              round(ec["arms"][1]["break_even_E"], 3), round((10.0 + 258.0) / ((200.0 + 10.0) / 2.0), 3))
        check("k-econ: the ceiling is the perfect-accept cost per token",
              round(ec["ceiling"]["k3"], 2), round((10.0 + 332.0) / 4, 2))
        mixed = [dict(r, engine_digest={"libllama": "a" if r["k"] == 1 else "b"}) for r in econ_rows]
        check("k-econ: no fit across two builds", k_econ(mixed)["fit"], None)

        # the accept model. Its two invariants hold for ANY fit, so they are worth asserting where the
        # numbers are not: the argmin k cannot fall as accept rises, and the runs must tile [0,1].
        ka, kwhy = k_econ_accept(ec)
        check("k-econ: the optimal k is non-decreasing in accept",
              (kwhy, all(x["k"] <= y["k"] for x, y in zip(ka["curve"], ka["curve"][1:]))), (None, True))
        check("k-econ: the optimal-k runs tile the whole accept range",
              (ka["runs"][0]["lo"], ka["runs"][-1]["hi"]), (0.0, 1.0))
        # two hand-built budgets where the answer is arithmetic: a marginal token 50x the fixed work
        # cannot pay for itself at a=0.4, and a nearly-free one always can at a=0.9.
        dear = dict(fit={"F": 10.0, "m": 500.0}, arms=[{"k": 1, "draft_ms": 10.0},
                                                         {"k": 3, "draft_ms": 30.0}])
        cheap = dict(fit={"F": 200.0, "m": 1.0}, arms=[{"k": 1, "draft_ms": 10.0},
                                                         {"k": 3, "draft_ms": 30.0}])
        check("k-econ: an expensive marginal token wants k=1 at low accept",
              k_econ_accept(dear, alphas=(0.4,))[0]["curve"][0]["k"], 1)
        check("k-econ: a cheap one wants wide verify at high accept",
              k_econ_accept(cheap, alphas=(0.9,))[0]["curve"][0]["k"], 8)
        check("k-econ: no fit => the accept model refuses instead of inventing F and m",
              k_econ_accept({"fit": None, "arms": []}), (None, "no fit (needs >= 2 arms on one build)"))

        # ── NSM: the design matrix has to recover costs it was not told, and the rows have to sum
        #    back to the span the same log reports (that closure is the point of the read).
        def nsm_log(name, costs, specs, bogus=0, off=0):
            d = tmp / name
            d.mkdir()
            out = []
            for i, (w, bufs) in enumerate(specs, start=1):
                gpu = 0
                for cnt in bufs:
                    dur = sum(costs[k] * c for k, c in cnt.items()) + off
                    gpu += dur
                    out.append(f"CGC-NSM a=0 b={len(cnt)} dur_ns={dur} nk={sum(cnt.values())} "
                               + " ".join(f"{k}:{c}" for k, c in cnt.items()) + "\n")
                if bogus and i == 1:
                    # the 09-23 defect in miniature: an uptime stamp in the duration field, which the
                    # emitter's own `s > 0 && e > s` guard lets through
                    out.append("CGC-NSM a=0 b=1 dur_ns=77100000000000 nk=1 moe:1\n")
                out.append(f"CGC-DECPROF: step={i} segs=41 layers=40 total={gpu / 1e6 + 1:.2f} ms | "
                           f"wait=1.00 (1%) cb=2.00 (2%) submit=3.00 (3%) ntok={w} | "
                           f"layer gpu_sum={gpu / 1e6:.2f} union_sum={gpu / 2e6:.2f} gap_sum=0.10 ms\n")
            (d / f"{name}.stderr.log").write_text("".join(out))
            return d

        costs = {"moe": 10_000, "gdn": 5_000, "norm": 30_000}   # ns per unit of count
        # Composition that changes with the WIDTH (that is what makes a marginal exist) and also
        # between steps (a kind with one co-occurrence signature keeps a cost it does not own, so the
        # fixture has to offer several signatures or the checks would be testing the weak branch).
        def nsm_specs(w):
            return [(w, [{"moe": w, "gdn": i % 3 + 1}, {"moe": 1, "norm": 1, "gdn": 1},
                         {"norm": i % 2 + 1}, {"norm": 2, "gdn": 1}, {"moe": 1, "norm": 1}])
                    for i in range(1, 17)]

        t1 = nsm_log("nsm_t2", costs, nsm_specs(2))
        t4 = nsm_log("nsm_t4", costs, nsm_specs(4))
        na, why = nsm_attribution([t1 / "nsm_t2.stderr.log", t4 / "nsm_t4.stderr.log"])
        check("nsm: the solve recovers costs it was never told (ns per unit)",
              (why, round(na["t_ns"]["moe"]), round(na["t_ns"]["norm"]), round(na["t_ns"]["gdn"])),
              (None, 10_000, 30_000, 5_000))
        check("nsm: a mixed buffer is split by its counts, not dumped in a bucket (moe group)",
              round(na["marg"]["group"].get("moe", 0.0), 6), 0.01)
        check("nsm: the rows sum to the fitted slope arithmetically",
              round(na["marg"]["sum"] - na["marg"]["fit_slope"], 9), 0.0)
        check("nsm: the fitted slope is the one the fixture generated",
              (round(na["marg"]["fit_slope"], 6), round(na["marg"]["meas_slope"], 6)), (0.01, 0.01))
        check("nsm: two logs are one matrix, not two fits", len(na["logs"]), 2)
        check("nsm: no kind in the fixture is held back", na["marg"]["unident"], [])
        _old, sys.stdout = sys.stdout, io.StringIO()
        try:
            print_nsm_attribution(na, [t1 / "nsm_t2.stderr.log", t4 / "nsm_t4.stderr.log"])
        finally:
            printed, sys.stdout = sys.stdout.getvalue(), _old
        check("nsm: the report prints (a missing key would raise, not warn)",
              "SOLVED" in printed, True)

        # the launch is a parameter, not noise: the second log's durations are all shifted and t_k
        # must not move. The offset itself DOES reach the marginal -- it is per buffer row and the
        # buffer count grows with the width -- so it has to be its own row (0.0025 ms/token from
        # 1 us/buffer here) rather than a constant folded into someone else's cost.
        t4b = nsm_log("nsm_t4b", costs, nsm_specs(4), off=1000)
        nsh, _ = nsm_attribution([t1 / "nsm_t2.stderr.log", t4b / "nsm_t4b.stderr.log"])
        check("nsm: a launch offset leaves t_k untouched",
              round(nsh["t_ns"]["moe"]), 10_000)
        check("nsm: ...and is named as its own (per-buffer) row, not folded into an op",
              (round(nsh["marg"]["offset"], 6), round(nsh["marg"]["meas_slope"], 6)),
              (0.0025, 0.0125))
        check("nsm: the decomposition is still complete with it",
              round(nsh["marg"]["sum"] + nsh["marg"]["offset"] - nsh["marg"]["fit_slope"], 9), 0.0)

        # a duration longer than its own step is dropped and counted -- and must not move the solve
        bog = nsm_log("nsm_bogus", costs, nsm_specs(4), bogus=1)
        nb, whyb = nsm_attribution(bog / "nsm_bogus.stderr.log")
        check("nsm: an impossible buffer is dropped and counted",
              (whyb, nb["dropped"], round(nb["t_ns"]["moe"])), (None, 1, 10_000))

        # one width cannot give a per-token split at all -- refused by name, not extrapolated
        check("nsm: a single-width capture refuses the marginal and says why",
              nb["marg"], None)

        coll = nsm_log("nsm_coll", {"alpha": 1000, "beta": 2000},
                       [(2, [{"alpha": 2, "beta": 4}, {"alpha": 1, "beta": 2}])] * 8
                       + [(4, [{"alpha": 4, "beta": 8}, {"alpha": 1, "beta": 2}])] * 8)
        nc, whyc = nsm_attribution(coll / "nsm_coll.stderr.log")
        check("nsm: collinear kinds are reported as unidentifiable, not averaged away",
              (whyc, nc["rank"], nc["nkinds"]), (None, 1, 2))
        check("nsm: and their costs are held out of the marginal",
              sorted(nc["marg"]["unident"]), ["alpha", "beta"])

        few = nsm_log("nsm_few", costs, [(4, [{"moe": 3, "gdn": 1}]) for _ in range(6)])
        nf, whyf = nsm_attribution(few / "nsm_few.stderr.log")
        check("nsm: too few NSM steps is refused, and names the flag",
              (nf, "CGC_GPU_NODES=1" in (whyf or "")), (None, True))
    print(f"selftest: {'PASS' if bad == 0 else f'{bad} FAILED'}")
    return 0 if bad == 0 else 1


def arms_from_jsonl(jsonl: Path) -> dict:
    by: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    seen: dict[tuple, str] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "invalid" in r or not r.get("dir") or r.get("round_ms") is None:
            continue
        key = (r["arm"], r["rep"])
        if seen.get(key, "") <= r.get("ts", ""):
            seen[key] = r.get("ts", "")
            by[r["arm"]][r["rep"]] = r
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/tmp/verify_marginal")
    ap.add_argument("--arm-t1", default="k1")
    ap.add_argument("--arm-t4", default="k3")
    ap.add_argument("--per-miss-ms", type=float, default=PER_MISS_MS)
    ap.add_argument("--steps", action="store_true",
                    help="add the step-channel split (device span / host hook / dispatch); needs "
                         "CGC_DECODE_PROFILE=1 (+ CGC_GPU_TIMING=1 for the device side)")
    ap.add_argument("--phase", default="",
                    help="restrict the step family to this phase (e.g. VERIFY); needs "
                         "CGC_PHASE_DBG=1 in the arms, else every step is unlabeled")
    ap.add_argument("--arm-dir", default="",
                    help="read the step-channel split from ONE prod_matrix run directory, which needs "
                         "no k1/k3 pair and none of the round-level axes")
    ap.add_argument("--node-attr", default="",
                    help="attribute the device span to ops from one log's CGC-GPUOPS tables; needs "
                         "CGC_GPU_OPS=1 (+ CGC_DECODE_PROFILE=1 for the family and CGC_GPU_TIMING=1 "
                         "for the span it is meant to explain)")
    ap.add_argument("--k-econ", default="",
                    help="the round budget (rest = F + m*T) from an arms.jsonl or a k_sweep.json")
    ap.add_argument("--nsm", default="",
                    help="per-op attribution of the device span from the CGC-NSM design matrix; "
                         "comma-separated logs become ONE matrix, which is what a per-token split "
                         "needs (k1 log + k3 log). Needs CGC_GPU_NODES=1 + CGC_GPU_NODES_MATRIX=1 + "
                         "CGC_DECODE_PROFILE=1 (+ CGC_GPU_TIMING=1 for the span itself)")
    ap.add_argument("--target-ms", type=float, default=40.0,
                    help="the delivered ms/token the budget is checked against (default 40 = 25 t/s)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    if args.node_attr:
        res, why = node_attribution(Path(args.node_attr))
        if res is None:
            print(f"node attribution not available: {why}")
            return 1
        print_node_attribution(res, Path(args.node_attr))
        return 0

    if args.nsm:
        nsm_logs = [Path(p) for p in args.nsm.split(",") if p.strip()]
        res, why = nsm_attribution(nsm_logs)
        if res is None:
            print(f"NSM attribution not available: {why}")
            return 1
        print_nsm_attribution(res, nsm_logs)
        return 0

    if args.k_econ:
        rows, schema = k_econ_rows(Path(args.k_econ))
        if rows is None:
            print(f"k economics not available: {schema}")
            return 1
        print_k_econ(k_econ(rows, args.target_ms), Path(args.k_econ), schema)
        return 0

    if args.arm_dir:
        rows = one_arm_split({"dir": args.arm_dir}, args.phase or None)
        print(f"single arm {args.arm_dir}")
        if rows.get("invalid"):
            print(f"  step-channel split not available: {rows['invalid']}")
            return 1
        print_step_channels(rows, 0, args.phase or None)
        print("  (the round-level axes and the arm contrast need a k1/k3 pair; this reading needs "
              "neither, which is why it is the one a fresh cell can pay for)")
        return 0

    jsonl = Path(args.logdir) / "arms.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"no {jsonl} -- run mtp_k_sweep.py on this logdir first")
    arms = arms_from_jsonl(jsonl)
    for a in (args.arm_t1, args.arm_t4):
        if not arms.get(a):
            raise SystemExit(f"arm {a!r} has no usable record in {jsonl} (present: {sorted(arms)})")
    dt = K_OF.get(args.arm_t4, 0) - K_OF.get(args.arm_t1, 0)
    if dt <= 0:
        raise SystemExit(f"arms {args.arm_t1}/{args.arm_t4} do not differ in T (dT={dt})")
    reps = sorted(set(arms[args.arm_t1]) & set(arms[args.arm_t4]))
    if not reps:
        raise SystemExit("the two arms share no rep")
    print(f"arms {args.arm_t1} (T={K_OF[args.arm_t1] + 1}) vs {args.arm_t4} "
          f"(T={K_OF[args.arm_t4] + 1}), dT={dt}, {len(reps)} paired rep(s)")

    pairs = {rep: pair_rep(arms[args.arm_t1][rep], arms[args.arm_t4][rep], dt, args.per_miss_ms)
             for rep in reps}
    layer_pairs = {rep: (arms[args.arm_t1][rep], arms[args.arm_t4][rep]) for rep in reps
                   if not pairs[rep].get("invalid")}
    if args.steps:
        step_pairs = {rep: (arms[args.arm_t1][rep], arms[args.arm_t4][rep]) for rep in reps}
    else:
        step_pairs = None
    return report(pairs, args.per_miss_ms, layer_pairs, dt, step_pairs, args.phase or None)


if __name__ == "__main__":
    raise SystemExit(main())
