#!/usr/bin/env python3
"""gdn_split.py — WHERE does the GDN layers' extra per-layer GPU cost live?

The question
------------
`docs/DECODE_STEP_BUDGET_2026-09-19.md` §16 measures, per layer, the GPU union of the 30 GDN
(gated-delta-net / recurrent) layers against the 10 full-attention layers in the SAME step and
reports them ~0.48 ms/layer apart. That delta is a *difference of differences*: the shared MoE cost
cancels, but nothing inside the GDN layer is attributed. This probe asks which part of the GDN
layer carries it:

    conv preprocessing  (conv_states/conv_input/conv_state_last/conv_state_update + the conv kernel)
    the fused scan      (gdn_state / gdn_out = ggml_gated_delta_net, K=1 autoregressive at T=1)
    gates + projections (alpha / beta / a_softplus / z- / q_conv / k_conv / v_conv / qkv_mixed / ...)

Why it can be answered at all
-----------------------------
The delta-net builder NAMES its nodes (cb() is ggml_format_name), and the engine already has a
node-level GPU instrument (`CGC_GPU_NODES`): each command buffer's GPUStartTime/GPUEndTime is
attributed across the node range that buffer encoded. So no new C++ is needed — but the instrument
alone is not enough, for the reason its own comment records: n_main = MAX(64, 0.1*n_nodes) puts
>=64 nodes into the main buffer, so a count-weighted split dilutes whatever lives there. Hence
three independent readings, cross-checked rather than averaged:

  1. CGC-GPUNODE  per-KIND table: wcntw (work-weighted), plus lb (solo-buffer lower bound) and
                  ub (sum of buffers containing the kind) — the truth is in [lb, ub].
  2. CGC-NSM      one line per command buffer (duration + per-kind node counts) = the design
                  matrix for dur_i = sum_k cnt_ik * t_k, solved here by least squares with a
                  hand-rolled normal-equation solve (no numpy dependency).
  3. CGC-GPUOPS   op-keyed table; the decisive column is `uni` (exact us/node from buffers holding
                  exclusively that op) — a positive control the fit can be checked against.

Exclusivity is EVIDENCE, not assumption
---------------------------------------
A kind is only usable for a GDN-vs-attention split if it comes from one layer type only. The step
contains 40 base layers: 30 GDN + 10 full attention (full_attention_interval=4, first full-attention
layer index 3 => FULL_ATTN = {3,7,...,39}). So per-step node counts are the mechanical test:

    c % 40 == 0  -> SHARED      (exists in both layer types)
    c % 30 == 0  -> GDN-only
    c % 10 == 0  -> ATTN-only
    otherwise    -> UNKNOWN     (never bucketed)

and any count that satisfies more than one of these is flagged AMBIG rather than bucketed. The
classification is printed with its count on every run, so a wrong label is visible in the output
instead of hidden inside a total. It is additionally cross-checked against the builder source
(delta-net-base.cpp names conv_*/gdn_*; llama-graph.cpp names Qcur/Kcur/Vcur/kqv_out).

What this probe does NOT do
---------------------------
- It does not attribute PER (layer, kind). The node->layer mapping in index space is not dumped for
  a decode-shaped graph (CGC_GRPH_DBG only fires on the first 6 graph computes, which are prefill),
  so per-layer resolution stays where it already was: the per-layer union delta from CGC-DECPROF.
  What is new here is the kind-level split of the attention-side cost, layer-normalized by the
  layer count each kind provably belongs to.
- It does not compare across launches. `raw` and `bracketed` both come from steps of the one warm
  request, so the regime flapping that ate lane A / the drift lane (an anchor's own spread of 34.5%
  and 97.5%) cannot enter a *within-step* comparison.
- It cannot make the fit identifiable where kinds never appear apart. `unident` is reported per
  kind as the number of distinct co-occurrence patterns, and the fit's residual is printed before
  any per-kind number is quoted.

usage:
    python3 scripts/check/gdn_split.py run          # launch, measure, save raw, analyze
    python3 scripts/check/gdn_split.py analyze      # re-analyze the saved raw data only
    python3 scripts/check/gdn_split.py selftest     # the tests that must go red when fed wrong input
    python3 scripts/check/gdn_split.py widthscan <server_log> [...]
                                                   # GDN/attn ratio per decode width; the ratio is
                                                   # the scale-free form (see `stability`)
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

RAW = os.environ.get("GDN_RAW", "/tmp/gdn_split_raw.json")
ANALYSIS = os.environ.get("GDN_ANALYSIS", "/tmp/gdn_split_analysis.json")

N_BASE_LAYERS = 40
# full_attention_interval=4 and the first full-attention layer is index 3 (filter_attn is
# (il<40 && !is_recr(il)) in models/qwen35moe.cpp), so layer 3, 7, ... 39 are NOT GDN.
FULL_ATTN = sorted(range(3, N_BASE_LAYERS, 4))
N_ATTN = len(FULL_ATTN)                 # 10
N_GDN = N_BASE_LAYERS - N_ATTN          # 30

# The buckets this probe exists to separate. Membership is from the builder source, not from the
# instrument: src/llama.cpp/src/models/delta-net-base.cpp (conv_*, gdn_state, gdn_out) and the
# delta-net gate/state names recorded in ggml-backend.cpp's vocabulary comment (§EN-149 round 2).
BUCKETS = {
    "conv": ["conv", "q_conv", "k_conv", "v_conv"],
    "scan+state": ["gdn_state", "gdn_out", "state_predelta"],
    "gdn gates+proj": ["alpha", "beta", "a_softplus", "z-", "qkv_mixed", "dnqkv_proj", "dnbeta_proj",
                       "dn_normg_mul"],
    "attn core": ["Qcur", "Kcur", "Vcur", "kqv_out", "attn_", "attn_q", "attn_k", "attn_v",
                  "attn_output", "attn_norm", "attn_post_norm", "attn_residual", "attn_inp_k_rot",
                  "attn_inp_v_rot", "attn_inp_kq_mask", "rope", "soft_max"],
    "moe": ["ffn_moe_", "ffn_moe_add", "ffn_moe_argsort", "ffn_moe_logits", "ffn_moe_probs",
            "ffn_moe_slots", "ffn_moe_topk", "ffn_moe_gate_up", "ffn_moe_gate", "ffn_moe_up",
            "ffn_moe_down", "ffn_", "top_k", "shared_expert_gate"],
}
# Which side each bucket may be used for. `moe` is present in every layer (both types) so it is
# only ever used as a same-step control, never as part of the difference.
BUCKET_SIDE = {"conv": "gdn", "scan+state": "gdn", "gdn gates+proj": "gdn",
               "attn core": "attn", "moe": "shared"}

# The op-level reading, which is the one that does NOT need a divisibility guess: the instrument
# prints `nd` (node count) per op, and for these ops that count IS the layer population.
#   GATED_DELTA_NET / SSM_CONV  = 1 per GDN layer   -> 30
#   ROPE = 2, SOFT_MAX = 4, SET_ROWS = 2 per full-attention layer -> 20 / 40 / 20
# The two populations sum to the layer count (30 + 10 = 40), which is a consistency argument the
# name-based count test cannot make.
OP_SIDE = {
    "GATED_DELTA_NET": ("gdn", "scan+state (fused: the whole recurrence incl. state update)"),
    "SSM_CONV": ("gdn", "conv (depthwise conv over the mixed qkv)"),
    "FLASH_ATTN": ("attn", "attention core"),
    "SOFT_MAX": ("attn", "attention core"),
    "ROPE": ("attn", "attention core"),
    "SET_ROWS": ("attn", "KV write"),
    "MUL_MAT_ID": ("shared", "MoE expert matmul"),
    "GLU": ("shared", "MoE"),
    "ARGSORT": ("shared", "MoE selection"),
    "TOP_K": ("shared", "MoE selection"),
}
# Layer populations the graph actually has (measured below, not assumed): 40 base layers, of which
# 30 GDN + 10 full attention, and 39 for the MoE sub-block (one layer has none).
LAYER_POPS = ((N_BASE_LAYERS, "per-layer"), (N_GDN, "per-GDN-layer"), (N_ATTN, "per-attn-layer"),
              (N_BASE_LAYERS - 1, "per-layer(minus one)"))


def op_populations(nd: int) -> list[dict]:
    """Every whole per-layer reading of an op's node count, with its multiplicity."""
    out = []
    for n, label in LAYER_POPS:
        if nd % n == 0 and 1 <= nd // n <= 8:
            out.append(dict(pop=n, per_layer=nd // n, label=label))
    return out

HDR = re.compile(r"CGC-GPUNODE: step=(\d+) seg_busy=([\d.]+) ms \| layer gpu_sum=([\d.]+) ms \| "
                 r"delta=([\-\d.]+)% \| other=([\d.]+) ms \| nkind=(\d+)")
KIND = re.compile(r"CGC-GPUNODE:\s\s+(\S+)\s+wcntw=\s*([\d.]+)\s+([\d.]+)% \| cntw=\s*([\d.]+)\s+"
                  r"([\d.]+)% \| lb=\s*([\d.]+)\s+([\d.]+)% \| ub=\s*([\d.]+)\s+([\d.]+)%")
WORK = re.compile(r"CGC-GPUNODE: work-attributed ([\d.]+) of ([\d.]+) ms \(([\d.]+)%\) \| "
                  r"named_work_nodes=(\d+) \| residual=([\d.]+) ms \(([\d.]+)%\)")
NSM = re.compile(r"CGC-NSM a=(\d+) b=(\d+) dur_ns=(\d+) nk=(\d+)(.*)")
NSMKV = re.compile(r"\s+(\S+):(\d+)")
OPSHDR = re.compile(r"CGC-GPUOPS: step=(\d+) total=([\d.]+) ms nodes_all=(\d+) nodes_work=(\d+) "
                    r"nodes_named=(\d+) nop=(\d+)")
OPROW = re.compile(r"CGC-GPUOPS:\s\s+(\S+)\s+nd=\s*(\d+) (work|NOOP) wcntw=\s*([\d.]+)\s+([\d.]+)%\s+"
                   r"cntw=\s*([\d.]+)\s+([\d.]+)%\s+ub=\s*([\d.]+)\s+([\d.]+)%\s+uni=\s*([\-\d.]+)")
TRACE_HOT = re.compile(r"CGC-GPUNODE: hot#(\d) dur=([\d.]+) ms range=(\d+) nodes:(.*)")
TRACE_RNG = re.compile(r"CGC-GPUNODE: range sizes 1=(\d+) 2=(\d+) 3-7=(\d+) 8\+=(\d+)")
ALLRE = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
                   r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")
# The per-STEP delimiter. CGC-GPUNODE's header is printed only on a 1-in-8 step sample, so using
# it to group would merge eight steps of NSM into one record -- the first revision did exactly
# that and every per-step count came out ~8x too large. `CGC-DECPROF: step=` fires on EVERY step
# and carries ntok, which is also what separates decode (ntok=1) from prefill chunks.
STEPHDR = re.compile(r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
                     r"wait=([\d.]+) \(\d+%\) cb=([\d.]+) \(\d+%\) submit=([\d.]+) \(\d+%\) "
                     r"ntok=(\d+) \| layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load_harness():
    """Reuse the window harness: its launch/window/port discipline was expensive to learn."""
    spec = importlib.util.spec_from_file_location("dwh", os.path.join(HERE, "decode_window_harness.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------- parsing
def _blank(step=None):
    return {"step": step, "seg_busy": None, "gpu_sum": None, "delta_pct": None, "other": None,
            "nkind": None, "kinds": {}, "nsm": [], "ops": {}, "hot": [], "layers": {},
            "work": None}


def parse_steps(chunk: str) -> tuple[list[dict], dict]:
    """Group a log chunk into per-STEP records, keyed on `CGC-DECPROF: step=`.

    The print order inside one step is fixed, and the two halves sit on opposite sides of the
    delimiter:

        [CGC-NSM ...]                     <- this step's command buffers (arrive BEFORE the header)
        CGC-DECPROF: step=N ... ntok=X    <- the delimiter
        [CGC-DECPROF all: L.. rows]       <- this step
        [CGC-GPUNODE table, 1 in 8 steps] <- this step

    So NSM goes into `pending` and is flushed by the following header; everything else attaches to
    the record already open. The GPU table's own `step=` is then checked against the delimiter's:
    if the two ever disagree, the grouping is wrong and the numbers mean nothing -- which is worth a
    counter rather than a comment, because an off-by-one grouping produces numbers that look fine.
    """
    steps, cur, pending = [], None, None
    diag = dict(steps=0, gpunode_matched=0, gpunode_mismatch=0, nsm_orphans=0, ntok={})
    for line in chunk.splitlines():
        m = STEPHDR.search(line)
        if m:
            if cur is not None:
                steps.append(cur)
            cur = _blank(int(m.group(1)))
            cur.update(segs=int(m.group(2)), layers_n=int(m.group(3)), total=float(m.group(4)),
                       wait=float(m.group(5)), cb=float(m.group(6)), submit=float(m.group(7)),
                       ntok=int(m.group(8)), gpu_sum=float(m.group(9)), union_sum=float(m.group(10)),
                       gap_sum=float(m.group(11)), nsm=(pending["nsm"] if pending else []))
            pending = None
            continue
        m = NSM.search(line)
        if m:
            if pending is None:
                pending = _blank()
            pending["nsm"].append(
                dict(a=int(m.group(1)), b=int(m.group(2)), dur_ns=int(m.group(3)), nk=int(m.group(4)),
                     cnt={k: int(v) for k, v in NSMKV.findall(m.group(5))}))
            continue
        if cur is None:
            # NSM before the very first delimiter in the range is the first step's own matrix only
            # if a delimiter follows; if the chunk starts mid-step it belongs to a step we cannot
            # see. Counted, not silently absorbed into a step it may not belong to.
            if pending is not None and pending["nsm"]:
                diag["nsm_orphans"] += len(pending["nsm"])
                pending = None
            continue
        m = HDR.search(line)
        if m:
            if int(m.group(1)) == cur["step"]:
                diag["gpunode_matched"] += 1
                cur.update(seg_busy=float(m.group(2)), gp_gpu_sum=float(m.group(3)),
                           delta_pct=float(m.group(4)), other=float(m.group(5)),
                           nkind=int(m.group(6)))
            else:
                diag["gpunode_mismatch"] += 1
                cur["hdr_mismatch"] = (cur["step"], int(m.group(1)))
            continue
        m = ALLRE.search(line)
        if m:
            cur["layers"][int(m.group(1))] = dict(
                wait=float(m.group(2)), cb=float(m.group(3)), submit=float(m.group(4)),
                gpu=float(m.group(5)), union=float(m.group(6)), gap=float(m.group(7)))
            continue
        m = KIND.search(line)
        if m:
            cur["kinds"][m.group(1)] = dict(wcntw=float(m.group(2)), cntw=float(m.group(4)),
                                            lb=float(m.group(6)), ub=float(m.group(8)))
            continue
        m = WORK.search(line)
        if m:
            cur["work"] = dict(attr=float(m.group(1)), total=float(m.group(2)), attr_pct=float(m.group(3)),
                               nodes=int(m.group(4)), residual=float(m.group(5)),
                               residual_pct=float(m.group(6)))
            continue
        m = OPSHDR.search(line)
        if m:
            cur["ops"] = {"total": float(m.group(2)), "nodes_all": int(m.group(3)),
                          "nodes_work": int(m.group(4)), "nodes_named": int(m.group(5)),
                          "rows": cur["ops"].get("rows", {})}
            continue
        m = OPROW.search(line)
        if m:
            cur["ops"].setdefault("rows", {})[m.group(1)] = dict(
                nd=int(m.group(2)), emits=m.group(3) == "work", wcntw=float(m.group(4)),
                cntw=float(m.group(6)), ub=float(m.group(8)), uni=float(m.group(10)))
            continue
        m = TRACE_HOT.search(line)
        if m:
            cur["hot"].append(dict(rank=int(m.group(1)), dur=float(m.group(2)),
                                   range=int(m.group(3)), names=m.group(4).split()))
            continue
        m = TRACE_RNG.search(line)
        if m:
            cur["rng"] = [int(m.group(i)) for i in (1, 2, 3, 4)]
            continue
    if cur is not None:
        steps.append(cur)
    diag["steps"] = len(steps)
    diag["nsm_orphans"] += len(pending["nsm"]) if pending else 0
    for s in steps:
        # A step without NSM lines carries no design matrix; keep it for the LAYER table only, so it
        # still counts toward the per-layer medians while never entering the fit.
        s["has_nsm"] = bool(s["nsm"])
        s["has_gputable"] = s.get("nkind") is not None
        diag["ntok"][s["ntok"]] = diag["ntok"].get(s["ntok"], 0) + 1
    return steps, diag


# ------------------------------------------------------------------ exclusivity from evidence
PLAUSIBLE_MAX = 8      # per-layer multiplicity above which a division stops being a reading


def classify(count: int) -> dict:
    """Which layer populations a per-step node count is CONSISTENT with.

    Divisibility is a weak test and this function does not pretend otherwise: a count of 30 is
    simultaneously "1 per GDN layer x 30" and "3 per attention layer x 10", and no arithmetic on a
    single step can separate those. So it returns EVERY reading that is still plausible, and the
    caller compares them against the side the builder source assigns. The only decisive outcomes are
    the ones where a single reading survives.
    """
    if count <= 0:
        return dict(sides=[], why="0 nodes")
    cand = [(N_BASE_LAYERS, "SHARED"), (N_GDN, "GDN-only"), (N_ATTN, "ATTN-only")]
    sides = [s for n, s in cand if count % n == 0 and 1 <= count // n <= PLAUSIBLE_MAX]
    if not sides:
        return dict(sides=[], why=f"{count} matches no whole per-layer pattern")
    why = " | ".join(f"{count} = {count // n} per layer x {n} -> {s}" for n, s in cand
                     if count % n == 0 and 1 <= count // n <= PLAUSIBLE_MAX)
    return dict(sides=sides, why=why)


def median_counts(steps: list[dict]) -> dict:
    """Per-step node count per kind, as the median over steps (a median, not a mean: one step with
    a different graph shape must not move the divisor)."""
    acc = {}
    for s in steps:
        if not s["has_nsm"]:
            continue
        tot = {}
        for buf in s["nsm"]:
            for k, c in buf["cnt"].items():
                tot[k] = tot.get(k, 0) + c
        for k, c in tot.items():
            acc.setdefault(k, []).append(c)
    return {k: st.median(v) for k, v in acc.items()}


def steps_with_kind_conflict(a: dict) -> list[str]:
    """Kinds whose count evidence CONTRADICTS the side the builder source assigns them.

    Only a decisive single reading can contradict. A count that admits both "1 per GDN layer x 30"
    and "3 per attention layer x 10" contradicts nothing -- it is inconclusive, and calling that a
    conflict is how a weak test starts to look like a strong one (it did, on the first real run:
    23 kinds were flagged CONFLICT while almost none of them could decide anything).
    """
    return [k for k, r in a.get("kind_counts", {}).items()
            if r.get("agrees_with_source") is False and len(r.get("consistent_sides") or []) == 1]


def kind_inconclusive(a: dict) -> list[str]:
    """Kinds the count test could not decide (a real, reported absence of evidence)."""
    return [k for k, r in a.get("kind_counts", {}).items()
            if r.get("agrees_with_source") is False and len(r.get("consistent_sides") or []) > 1]


# ---------------------------------------------------------------------------- least squares
def _cv(x: list[float]):
    """Coefficient of variation, or None for a degenerate sample."""
    m = st.mean(x) if x else 0.0
    return (st.pstdev(x) / abs(m)) if m else None


def _corr(x: list[float], y: list[float]):
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return (num / den) if den else None


def stability(gdn_u: list[float], attn_u: list[float], usum: list[float]) -> dict:
    """Is the GDN-vs-attention gap an ADDITIVE per-layer amount, or a RATIO of the two?

    Both forms describe the same steps, so the question is which one stays put when the box
    changes -- and only the one that stays put can be spent as a budget line. Measured over the
    r16/r18 pair (25 ntok=1 decode steps, same binary): the additive delta had CV 61% and
    correlated +0.60 with the step's own union_sum, i.e. it mostly reports the regime it was
    measured in; the ratio had CV 18% and correlated +0.15. So the ratio is the quotable form and
    the ms/layer delta is quoted only together with the regime it came from.

    This exists because the number in docs §16 ("+0.770 ms/layer") was quoted in the unstable form.
    """
    n = min(len(gdn_u), len(attn_u))
    dl, rl, ul = [], [], []
    for i in range(n):
        if not attn_u[i]:
            continue
        dl.append(gdn_u[i] - attn_u[i])
        rl.append(gdn_u[i] / attn_u[i])
        ul.append(usum[i] if i < len(usum) and usum[i] else None)
    paired = [(d, r, u) for d, r, u in zip(dl, rl, ul) if u]
    return dict(
        n=len(dl),
        delta_mean=st.mean(dl) if dl else None, delta_cv=_cv(dl),
        ratio_mean=st.mean(rl) if rl else None, ratio_cv=_cv(rl),
        n_paired=len(paired),
        corr_delta_usum=_corr([p[0] for p in paired], [p[2] for p in paired]),
        corr_ratio_usum=_corr([p[1] for p in paired], [p[2] for p in paired]),
        usum_min=min((p[2] for p in paired), default=None),
        usum_max=max((p[2] for p in paired), default=None),
    )


def width_scan_steps(step_lists: list[list[dict]]) -> dict:
    """Per-width GDN/attn ratio, pooled over any number of logs.

    Why a separate axis: each decode width needs its own launch, and launches on this box differ by
    up to 2x in absolute per-layer span (measured: 0.90..2.48 ms for the SAME attention layers). So
    across widths only a scale-free statistic is comparable. Measured: the ratio is 1.52..1.55 for
    ntok = 1,2,3,4 and 1.51..1.54 per launch (CV 4..16%), while the additive delta grows 0.49 ->
    1.15 ms. A ratio that is invariant in width and regime, and in position within the repeating
    (3 GDN + 1 attn) group (1.55/1.56/1.56), is a critical-path property -- not an amount of work.
    """
    by = {}
    for steps in step_lists:
        for s in steps:
            lay = {int(k): v for k, v in (s.get("layers") or {}).items()}
            if len(lay) != N_BASE_LAYERS:
                continue
            g = [lay[L]["union"] for L in lay if L not in FULL_ATTN]
            a = [lay[L]["union"] for L in lay if L in FULL_ATTN]
            if not g or not a:
                continue
            gm, am = st.median(g), st.median(a)
            if not am:
                continue
            by.setdefault(s["ntok"], []).append((gm, am, gm - am, gm / am))
    out = {}
    for nt, rs in by.items():
        rt = [r[3] for r in rs]
        out[nt] = dict(n=len(rs), gdn=st.median([r[0] for r in rs]),
                       attn=st.median([r[1] for r in rs]),
                       delta=st.median([r[2] for r in rs]), ratio=st.median(rt),
                       ratio_cv=_cv(rt), ratio_p25=st.quantiles(rt, n=4)[0],
                       ratio_p75=st.quantiles(rt, n=4)[2])
    return out


def lstsq(A: list[list[float]], b: list[float], ridge: float = 1e-9):
    """Normal equations with a ridge term and partial pivoting. No numpy: this runs on system
    python3, and a 48x48 solve is not worth a dependency.

    Returns (x, residual_r2, cond_ok). cond_ok is False when a pivot is negligible RELATIVE to the
    largest diagonal of AtA. That is the case that matters here: two kinds that always appear in a
    fixed ratio are not separately identifiable, the fit still returns a valid but arbitrary split,
    and R2 stays 1.0 -- so R2 cannot detect it and the pivot can. The first revision compared
    pivots to an absolute 1e-12, which every collinear system passes.
    """
    n = len(A[0]) if A else 0
    if n == 0 or not A:
        return [], 0.0, False
    scale = max(1.0, max(abs(v) for row in A for v in row))
    AtA = [[sum(A[i][p] * A[i][q] for i in range(len(A))) + (ridge * scale if p == q else 0.0)
            for q in range(n)] for p in range(n)]
    Atb = [sum(A[i][p] * b[i] for i in range(len(A))) for p in range(n)]
    max_diag = max(abs(AtA[i][i]) for i in range(n)) or 1.0
    M = [row[:] + [Atb[i]] for i, row in enumerate(AtA)]
    cond_ok = True
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-8 * max_diag:
            cond_ok = False
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / pv
            if f:
                for q in range(c, n + 1):
                    M[r][q] -= f * M[c][q]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        if abs(M[r][r]) < 1e-12 * scale:
            continue
        s = M[r][n] - sum(M[r][q] * x[q] for q in range(r + 1, n))
        x[r] = s / M[r][r]
    pred = [sum(row[j] * x[j] for j in range(n)) for row in A]
    ss_res = sum((b[i] - pred[i]) ** 2 for i in range(len(b)))
    mu = sum(b) / len(b)
    ss_tot = sum((v - mu) ** 2 for v in b)
    return x, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0), cond_ok


def fit_kinds(steps: list[dict], min_seen: int = 3):
    """Fit dur_i = sum_k cnt_ik * t_k over the NSM buffers of the sampled steps.

    Also returns, per kind, `patterns` = how many distinct co-occurrence signatures the kind has.
    A kind that always appears inside the same set of neighbours has few patterns and its per-node
    cost is a combination, not a measurement of that kind alone -- reported so the caller can refuse
    to quote it.
    """
    kinds: list[str] = []
    for s in steps:
        if not s["has_nsm"]:
            continue
        for buf in s["nsm"]:
            for k in buf["cnt"]:
                if k not in kinds:
                    kinds.append(k)
    if not kinds:
        return {}, {}
    idx = {k: i for i, k in enumerate(kinds)}
    A, b = [], []
    patterns: dict[str, set] = {k: set() for k in kinds}
    for s in steps:
        if not s["has_nsm"]:
            continue
        for buf in s["nsm"]:
            row = [0.0] * len(kinds)
            for k, c in buf["cnt"].items():
                row[idx[k]] = float(c)
            A.append(row)
            b.append(float(buf["dur_ns"]))
            sig = frozenset(buf["cnt"])
            for k in buf["cnt"]:
                patterns[k].add(sig)
    x, r2, cond = lstsq(A, b)
    seen = {k: patterns[k] for k in kinds}
    t = {}
    for k in kinds:
        t[k] = dict(t_ns=x[idx[k]], t_us=x[idx[k]] / 1e3,
                    patterns=len(seen[k]),
                    ident="ok" if len(seen[k]) >= min_seen else "weak")
    return t, dict(r2=r2, cond_ok=cond, n=len(A), nk=len(kinds))


# ------------------------------------------------------------------------------- analysis
def analyze(raw: dict) -> dict:
    arm = raw["arms"].get("rep2_warm") or next(iter(raw["arms"].values()))
    steps = arm["steps"]
    diag = arm.get("diag") or {}
    # ONLY ntok=1 steps. A prefill chunk is a different graph (chunked scan instead of the
    # autoregressive one), so mixing the two would fit one shape and quote it for the other.
    dec = [s for s in steps if s.get("ntok") == 1]
    # `.get` on every step field: these dicts also arrive from JSON (or from a hand-built fixture),
    # where the parser's guaranteed keys are not guaranteed at all.
    with_nsm = [s for s in dec if s.get("has_nsm") or s.get("nsm")]
    dec_table = [s for s in dec if s.get("has_gputable")]
    pre_table = [s for s in steps if s.get("has_gputable") and s.get("ntok") != 1]
    out = {"arm": arm.get("tag"), "steps_all": len(steps), "steps_decode": len(dec),
           "steps_with_nsm": len(with_nsm), "steps_with_gputable_dec": len(dec_table),
           "steps_with_gputable_prefill": len(pre_table), "diag": diag,
           "ntok_hist": {str(k): v for k, v in sorted((diag.get("ntok") or {}).items())},
           "engine_digest": raw.get("engine_digest"), "anchor": raw.get("anchor"),
           "free_mb_at_launch": raw.get("free_mb"), "swap_mb_at_launch": raw.get("swap_mb")}
    if diag.get("gpunode_mismatch"):
        out["verdict"] = (f"INCOMPLETE: {diag['gpunode_mismatch']} GPU-table header(s) disagreed with "
                          f"the step delimiter -> the grouping is wrong and no number below is usable")
        return out
    if not dec:
        out["verdict"] = (f"INCOMPLETE: no ntok=1 step in the range (saw ntok={out['ntok_hist']}); "
                          f"only decode steps are comparable")
        return out
    if not with_nsm:
        out["verdict"] = "INCOMPLETE: no step carried CGC-NSM (is CGC_GPU_NODES_MATRIX=1 set?)"
        return out
    if not dec_table:
        # B8: say which input would have been needed and was not there, rather than letting the
        # absence of a cross-check read as agreement.
        out["crosscheck_absent"] = (f"the per-kind wcntw/lb/ub table is printed 1 in 8 steps and none of "
                                    f"the {len(pre_table)} sampled step(s) in range was ntok=1 -> the "
                                    f"split rests on the least-squares fit + per-step counts alone")
    else:
        out["crosscheck_steps"] = [s["step"] for s in dec_table]

    # ---- 0. the instrument's own self-check, quoted before any number it gates.
    # The GPU table is printed 1 in 8 steps, so on a run whose sampled steps are all prefill there
    # is no `delta_pct` at ANY decode step. That is a missing input, not a passing check: it is
    # reported as absent (`abs(None)` used to crash the whole analysis here).
    deltas = [s["delta_pct"] for s in with_nsm if s.get("delta_pct") is not None]
    segs = [s["seg_busy"] for s in with_nsm if s.get("seg_busy") is not None]
    if deltas:
        out["selfcheck_delta_pct"] = dict(median=st.median(deltas), worst=max(abs(d) for d in deltas),
                                          n=len(deltas), steps=[s["step"] for s in with_nsm
                                                                if s.get("delta_pct") is not None])
    else:
        out["selfcheck_delta_pct"] = None
        out["selfcheck_absent"] = ("no decode step carried the CGC-GPUNODE table, so its "
                                    "seg_busy-vs-gpu_sum self-check could not be read on a comparable "
                                    "step; the fit's own R2/cond_ok is the only instrument check here")
    out["seg_busy_ms"] = dict(median=st.median(segs), n=len(segs)) if segs else None
    wa = [s["work"] for s in with_nsm if s.get("work")]
    if wa:
        out["work_attributed_pct"] = dict(median=st.median(w["attr_pct"] for w in wa),
                                          residual_pct=st.median(w["residual_pct"] for w in wa))

    # ---- 1. per-layer union, GDN vs attention, within the same steps
    seen_any = any(s.get("layers") for s in dec)
    seen_attn = any(any(L in FULL_ATTN for L in s.get("layers", {})) for s in dec)
    if seen_any and not seen_attn:
        out["verdict"] = ("INCOMPLETE: layer rows parsed but not ONE of the 10 full-attention layer "
                          "indices matched -> the render side differs from the layer-index side "
                          "(a JSON round trip stringifies keys); the GDN/attn split would be wrong "
                          "in one direction and must not be quoted")
        return out
    gdn_u, attn_u, gdn_w, attn_w, gdn_gap, attn_gap, usum = [], [], [], [], [], [], []
    for s in dec:
        lay = s.get("layers") or {}
        usum.append(s.get("union_sum"))
        gu = [lay[L]["union"] for L in lay if L not in FULL_ATTN]
        au = [lay[L]["union"] for L in FULL_ATTN if L in lay]
        if gu:
            gdn_u.append(st.median(gu))
        if au:
            attn_u.append(st.median(au))
        gw = [lay[L]["wait"] for L in lay if L not in FULL_ATTN]
        aw = [lay[L]["wait"] for L in FULL_ATTN if L in lay]
        if gw:
            gdn_w.append(st.median(gw))
        if aw:
            attn_w.append(st.median(aw))
        gg = [lay[L]["gap"] for L in lay if L not in FULL_ATTN]
        ag = [lay[L]["gap"] for L in FULL_ATTN if L in lay]
        if gg:
            gdn_gap.append(st.median(gg))
        if ag:
            attn_gap.append(st.median(ag))
    out["per_layer"] = {
        "gdn_union_ms": st.median(gdn_u), "attn_union_ms": st.median(attn_u),
        "union_delta_ms": st.median(gdn_u) - st.median(attn_u),
        "gdn_wait_ms": st.median(gdn_w), "attn_wait_ms": st.median(attn_w),
        "gdn_gap_ms": st.median(gdn_gap), "attn_gap_ms": st.median(attn_gap),
        "n_gdn_layers": N_GDN, "n_attn_layers": N_ATTN, "steps": len(gdn_u),
        "stability": stability(gdn_u, attn_u, usum),
    }

    # ---- 2. exclusivity from per-step counts, read against the side the builder source assigns
    counts = median_counts(with_nsm)
    source_side = {k: BUCKET_SIDE[b] for b, ms in BUCKETS.items() for k in ms}
    cls = {k: classify(int(c)) for k, c in counts.items()}
    out["kind_counts"] = {}
    for k in sorted(counts, key=lambda k: -counts[k]):
        src = source_side.get(k, "unbucketed")
        sides = cls[k]["sides"]
        want = {"gdn": "GDN-only", "attn": "ATTN-only", "shared": "SHARED"}.get(src)
        out["kind_counts"][k] = dict(count=int(counts[k]), source_side=src,
                                     consistent_sides=sides,
                                     agrees_with_source=(want in sides) if want else None,
                                     decisive=len(sides) == 1, why=cls[k]["why"])

    # ---- 3. three readings of the same per-kind cost
    t, fitinfo = fit_kinds(with_nsm)
    out["fit"] = fitinfo
    kinds_ms = {}
    for k, kn in counts.items():
        rec = {"count": int(kn), "source_side": source_side.get(k, "unbucketed"),
               "consistent_sides": cls[k]["sides"], "why": cls[k]["why"]}
        for name, key in (("wcntw", "wcntw"), ("lb", "lb"), ("ub", "ub")):
            vals = [s["kinds"][k][key] for s in with_nsm if k in s["kinds"]]
            rec[name] = st.median(vals) if vals else None
        # complement: cntw is the value the instrument itself warns is diluted; kept for contrast
        vals = [s["kinds"][k]["cntw"] for s in with_nsm if k in s["kinds"]]
        rec["cntw"] = st.median(vals) if vals else None
        if k in t:
            rec["ls_us_per_node"] = t[k]["t_us"]
            rec["ls_patterns"] = t[k]["patterns"]
            rec["ls_ident"] = t[k]["ident"]
            if rec["count"]:
                rec["ls_step_ms"] = t[k]["t_ns"] * rec["count"] / 1e6
        kinds_ms[k] = rec
    out["kinds_ms"] = kinds_ms

    # ---- 3b. the OP table: per-op cost with the instrument's own work/NOOP label, and the op's node
    # count as the layer population. This is the reading that does not need the divisibility guess:
    # an op whose count is 30 with a `work` label and no counterpart on the other layer type IS the
    # per-layer cost of that op, divided by 30.
    op_acc = {}
    for s in with_nsm:
        for op, r in (s.get("ops") or {}).get("rows", {}).items():
            a = op_acc.setdefault(op, dict(nd=[], wcntw=[], ub=[], emits=r["emits"], uni=[]))
            a["nd"].append(r["nd"])
            a["wcntw"].append(r["wcntw"])
            a["ub"].append(r["ub"])
            if r["uni"] is not None and r["uni"] >= 0:
                a["uni"].append(r["uni"])
    ops_ms = {}
    for op, a in sorted(op_acc.items(), key=lambda kv: -st.median(kv[1]["wcntw"])):
        nd = int(st.median(a["nd"]))
        reads = op_populations(nd)
        side, note = OP_SIDE.get(op, ("unclassified", ""))
        per_layer = None
        if side == "gdn" and nd % N_GDN == 0:
            per_layer = nd // N_GDN
        elif side == "attn" and nd % N_ATTN == 0:
            per_layer = nd // N_ATTN
        # The divisor is the LAYER COUNT (30 or 10), never the nodes-per-layer: an op with 2 nodes per
        # attention layer has 20 nodes per step, and its per-layer cost is total/10. Dividing by 2
        # instead (the first revision) inflates the per-layer cost by exactly the multiplicity.
        pop = N_GDN if side == "gdn" else N_ATTN if side == "attn" else None
        ops_ms[op] = dict(nd=nd, emits=a["emits"], side=side, note=note,
                          wcntw=st.median(a["wcntw"]), ub=st.median(a["ub"]),
                          uni=st.median(a["uni"]) if a["uni"] else None,
                          populations=reads, per_layer_nodes=per_layer,
                          n_layers=pop if per_layer else None,
                          wcntw_per_layer=(st.median(a["wcntw"]) / pop) if (per_layer and pop) else None,
                          ub_per_layer=(st.median(a["ub"]) / pop) if (per_layer and pop) else None)
    out["ops_ms"] = ops_ms

    # ---- 4. buckets, layer-normalized by the layer count the kind provably belongs to
    buckets = {}
    for bname, members in BUCKETS.items():
        acc = dict(wcntw=0.0, lb=0.0, ub=0.0, ls=0.0, kinds=[], clipped=[])
        okay = True
        for k in members:
            if k not in kinds_ms:
                continue
            r = kinds_ms[k]
            acc["kinds"].append(k)
            if r["wcntw"] is not None:
                acc["wcntw"] += r["wcntw"]
            if r["lb"] is not None:
                acc["lb"] += r["lb"]
            if r["ub"] is not None:
                acc["ub"] += r["ub"]
            if "ls_step_ms" in r:
                v = r["ls_step_ms"]
                if v < 0:
                    # A negative per-node cost is the fit telling us the kind does not emit work
                    # (VIEW/RESHAPE). Clipped to 0 for the sum, and NAMED, because clipping a kind
                    # this probe actually cares about would change the answer.
                    acc["clipped"].append(k)
                    v = 0.0
                    okay = okay and BUCKET_SIDE[bname] != "gdn"
                acc["ls"] += v
        side = BUCKET_SIDE[bname]
        div = N_GDN if side == "gdn" else (N_ATTN if side == "attn" else N_BASE_LAYERS)
        buckets[bname] = {**{k: v for k, v in acc.items() if k not in ("kinds", "clipped")},
                          "kinds": acc["kinds"], "clipped_negative": acc["clipped"],
                          "layers": div,
                          "wcntw_per_layer": acc["wcntw"] / div if div else None,
                          "lb_per_layer": acc["lb"] / div if div else None,
                          "ub_per_layer": acc["ub"] / div if div else None,
                          "ls_per_layer": acc["ls"] / div if div else None}
    out["buckets"] = buckets

    # ---- 5. reconciliation: does the split account for the measured per-layer delta?
    gdn_side = sum(buckets[b]["wcntw_per_layer"] or 0.0 for b in buckets if BUCKET_SIDE[b] == "gdn")
    attn_side = sum(buckets[b]["wcntw_per_layer"] or 0.0 for b in buckets if BUCKET_SIDE[b] == "attn")
    delta_meas = out["per_layer"]["union_delta_ms"]
    out["reconcile"] = dict(
        gdn_attention_side_per_layer=gdn_side, attn_attention_side_per_layer=attn_side,
        exclusive_delta_ms=gdn_side - attn_side, measured_delta_ms=delta_meas,
        unexplained_ms=(gdn_side - attn_side) - delta_meas)
    return out


# ------------------------------------------------------------------------------- running
def run():
    h = _load_harness()
    ok, why = h.anchor_ok(h.digest())
    if not ok:
        log(f"anchor: {why} -> nothing may be concluded; refusing to measure")
        return 2
    log(f"anchor OK: {why}")
    if not h.wait_for_window():
        log("no window within the budget; nothing measured")
        return 3
    log("window open; launching MTP-off with the node instrument armed")
    pid, srv_log, healthy = h.launch({
        "CGC_SERVER_MTP": "0",
        # `off` (not the harness's prefill250) so the decode graph is free of prefill-regime knobs.
        "CGC_SERVER_PROFILE": "off",
        "CGC_DECODE_PROFILE": "1", "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1",
        "CGC_GPU_NODES": "1", "CGC_GPU_NODES_MATRIX": "1", "CGC_GPU_OPS": "1",
        "CGC_GPU_NODES_TRACE": "1",
    })
    raw = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": pid, "server_log": srv_log,
           "engine_digest": h.digest(), "anchor": why, "free_mb": h.vm_free_mb(),
           "swap_mb": h.swap_used_mb(), "arms": {}}
    if not healthy:
        raw["status"] = "interrupted"
        raw["launcher_said"] = h.launcher_tail()
        json.dump(raw, open(RAW, "w"), indent=1)
        log("launch failed; wrote the launcher's own words to the raw file")
        return 4
    try:
        for tag in ("rep1_cold", "rep2_warm"):
            lo = os.path.getsize(srv_log) if srv_log else 0
            r = h.ask(h.PROMPT, h.N_PREDICT, srv_log, tag)
            hi = os.path.getsize(srv_log)
            with open(srv_log, errors="replace") as fh:
                fh.seek(lo)
                chunk = fh.read(hi - lo)
            steps, diag = parse_steps(chunk)
            raw["arms"][tag] = {"tag": tag, "tokens": r.get("tokens"),
                                "ms_per_token": r.get("ms_per_token"), "steps": steps,
                                "diag": diag,
                                "nsm_lines": sum(len(s["nsm"]) for s in steps)}
            log(f"  {tag}: {r.get('tokens')} tok, nsm buffers={raw['arms'][tag]['nsm_lines']}, "
                f"steps={diag.get('steps')} ntok={diag.get('ntok')} "
                f"gpunode_table={diag.get('gpunode_matched')} (mismatch {diag.get('gpunode_mismatch')}), "
                f"orphan NSM={diag.get('nsm_orphans')}")
    finally:
        h.stop_ours()
    json.dump(raw, open(RAW, "w"), indent=1)
    log(f"raw written to {RAW}")
    return 0


def normalize_raw(raw: dict) -> dict:
    """Restore int layer keys after a JSON round trip.

    JSON turns every dict key into a string, so a reloaded raw has layers {"3": ...} while
    FULL_ATTN is {3, 7, ...}. The split then silently counts ALL 40 layers as GDN and ZERO as
    attention -- a one-directional error that inflates the GDN side and empties the other, and the
    only reason it was caught is that the empty side raised instead of producing a number. Both the
    normalisation and the guard in analyze() exist because of that.
    """
    for arm in (raw.get("arms") or {}).values():
        for s in arm.get("steps") or []:
            if s.get("layers"):
                s["layers"] = {int(k): v for k, v in s["layers"].items()}
    return raw


def split_requests(steps: list[dict]) -> list[list[dict]]:
    """Split a server log's step sequence into per-request segments.

    A new request starts with a prefill, i.e. a run of steps with ntok>1 turns up after decode
    steps. The step counter is global across requests, so this is the only in-band boundary marker;
    the segments let the caller use the last (warm) request without needing a launch.
    """
    segs, cur, prev = [], [], None
    for s in steps:
        if cur and s.get("ntok", 1) > 1 and prev == 1:
            segs.append(cur)
            cur = []
        cur.append(s)
        prev = s.get("ntok", 1)
    if cur:
        segs.append(cur)
    return segs


def seg_diag(seg: list[dict]) -> dict:
    nt = {}
    for s in seg:
        nt[s["ntok"]] = nt.get(s["ntok"], 0) + 1
    return dict(steps=len(seg), ntok=nt,
                nsm_orphans=0, gpunode_matched=sum(1 for s in seg if s.get("has_gputable")),
                gpunode_mismatch=0)


def cmd_reparse():
    """Re-parse the ALREADY-COLLECTED server log with the current parser.

    Exists because the first collection wrote a raw file whose step grouping was wrong (keyed on the
    1-in-8 GPU table). Re-launching to fix a parser bug is the expensive mistake; the log is the
    primary record and stays valid.
    """
    raw = normalize_raw(json.load(open(RAW)))
    path = raw.get("server_log")
    if not path or not os.path.exists(path):
        log(f"no server log recorded in {RAW} (or it is gone): {path}")
        return 5
    with open(path, errors="replace") as fh:
        steps, diag = parse_steps(fh.read())
    segs = split_requests(steps)
    raw["reparsed"] = {"server_log": path,
                       "segments": [dict(seg_diag(s), first=s[0]["step"], last=s[-1]["step"]) for s in segs]}
    log(f"reparsed {path}: {diag['steps']} steps, ntok={diag['ntok']}, "
        f"gpunode table {diag['gpunode_matched']} (mismatch {diag['gpunode_mismatch']}), "
        f"orphan NSM={diag['nsm_orphans']}")
    for i, sg in enumerate(segs):
        d = seg_diag(sg)
        log(f"  segment {i}: steps {sg[0]['step']}..{sg[-1]['step']} ntok={d['ntok']}")
    # The last segment with decode steps is the warm request: same machine state, same page cache.
    warm = next((s for s in reversed(segs) if any(x["ntok"] == 1 for x in s)), segs[-1])
    raw["arms"] = {"rep2_warm": {"tag": f"rep2_warm(segment of {len(warm)} steps)",
                                  "tokens": None, "ms_per_token": None, "steps": warm,
                                  "diag": seg_diag(warm),
                                  "nsm_lines": sum(len(x["nsm"]) for x in warm)}}
    json.dump(raw, open(RAW, "w"), indent=1)
    log(f"warm segment: {len(warm)} steps, NSM buffers={raw['arms']['rep2_warm']['nsm_lines']}")
    return cmd_analyze()


def cmd_widthscan(paths: list[str]):
    lists = []
    for p in paths:
        if not os.path.exists(p):
            log(f"skip (missing): {p}")
            continue
        steps, _ = parse_steps(open(p, errors="replace").read())
        lists.append(steps)
        log(f"parsed {p}: {len(steps)} steps")
    if not lists:
        return 5
    tab = width_scan_steps(lists)
    print(f"\n{'ntok':>5s} {'n':>6s} {'gdn_med':>8s} {'attn_med':>9s} {'delta_med':>10s} "
          f"{'ratio_med':>10s} {'p25':>6s} {'p75':>6s} {'ratio_cv':>9s}")
    for nt in sorted(tab):
        r = tab[nt]
        print(f"{nt:5d} {r['n']:6d} {r['gdn']:8.2f} {r['attn']:9.2f} {r['delta']:10.2f} "
              f"{r['ratio']:10.2f} {r['ratio_p25']:6.2f} {r['ratio_p75']:6.2f} "
              f"{100 * r['ratio_cv']:8.0f}%")
    if len(tab) > 1:
        ds = [tab[nt]["delta"] for nt in sorted(tab)]
        rs = [tab[nt]["ratio"] for nt in sorted(tab)]
        print(f"\n  across widths: delta {min(ds):.2f}..{max(ds):.2f} ms (grows with the batch), "
              f"ratio {min(rs):.2f}..{max(rs):.2f}x (flat) -> the gap is scale-free")
    return 0


def cmd_analyze():
    raw = normalize_raw(json.load(open(RAW)))
    a = analyze(raw)
    json.dump(a, open(ANALYSIS, "w"), indent=1)
    print_report(a)
    return 0


def print_report(a):
    if "verdict" in a:
        print(a["verdict"])
        return
    p = a["per_layer"]
    print(f"\n=== per-layer GPU union (same steps, {p['steps']} steps) ===")
    print(f"  GDN layers (n={p['n_gdn_layers']}):      {p['gdn_union_ms']:.3f} ms  "
          f"(wait {p['gdn_wait_ms']:.3f}, gap {p['gdn_gap_ms']:.3f})")
    print(f"  full-attn layers (n={p['n_attn_layers']}): {p['attn_union_ms']:.3f} ms  "
          f"(wait {p['attn_wait_ms']:.3f}, gap {p['attn_gap_ms']:.3f})")
    print(f"  delta to explain: {p['union_delta_ms']:+.3f} ms/layer")
    sb = (p.get("stability") or {})
    if sb.get("delta_cv") is not None and sb.get("ratio_cv") is not None:
        print(f"  FORM of the claim (which one survives a regime change, n={sb['n']} steps):")
        print(f"    additive delta : {sb['delta_mean']:+.3f} ms/layer  CV {100 * sb['delta_cv']:.0f}%  "
              f"corr(delta, step union_sum)={sb['corr_delta_usum']:+.2f}" if sb.get("corr_delta_usum") is not None else "")
        print(f"    ratio GDN/attn : {sb['ratio_mean']:.3f}x          CV {100 * sb['ratio_cv']:.0f}%  "
              f"corr(ratio, step union_sum)={sb['corr_ratio_usum']:+.2f}" if sb.get("corr_ratio_usum") is not None else "")
        if sb.get("usum_min"):
            print(f"    (step union_sum ranged {sb['usum_min']:.0f}..{sb['usum_max']:.0f} ms in this sample: "
                  f"that spread is what the additive delta was tracking)")
    if a.get("selfcheck_delta_pct"):
        sc = a["selfcheck_delta_pct"]
        print(f"  instrument self-check: delta={sc['median']:.2f}% (worst |delta| {sc['worst']:.2f}%, "
              f"n={sc['n']} steps), seg_busy={a['seg_busy_ms']['median']:.2f} ms")
    else:
        print(f"  instrument self-check: ABSENT -- {a.get('selfcheck_absent')}")
    print(f"  steps: {a['steps_decode']} decode (ntok=1) of {a['steps_all']}; "
          f"design rows={a['steps_with_nsm']} steps; GPU-table steps(at ntok=1)={a['steps_with_gputable_dec']}, "
          f"(at other ntok)={a['steps_with_gputable_prefill']}")
    if a.get("crosscheck_absent"):
        print(f"  CROSS-CHECK ABSENT: {a['crosscheck_absent']}")
    elif a.get("crosscheck_steps"):
        print(f"  per-kind table cross-check on decode steps {a['crosscheck_steps']}")
    if "fit" in a:
        f = a["fit"]
        print(f"  least-squares: {f['n']} buffers, {f['nk']} kinds, R2={f['r2']:.4f}, "
              f"cond_ok={f['cond_ok']}")
    print("\n=== buckets (ms per layer) ===")
    print(f"  {'bucket':16s} {'side':7s} {'wcntw':>8s} {'lb':>8s} {'ub':>8s} {'ls':>8s}  kinds")
    for b, v in a["buckets"].items():
        print(f"  {b:16s} {BUCKET_SIDE[b]:7s} {v['wcntw_per_layer']:8.3f} {v['lb_per_layer']:8.3f} "
              f"{v['ub_per_layer']:8.3f} {v['ls_per_layer']:8.3f}  {','.join(v['kinds'])}")
        if v["clipped_negative"]:
            print(f"                   (negative per-node cost clipped to 0: {','.join(v['clipped_negative'])})")
    fb = a.get("fit") or {}
    if fb.get("cond_ok") is False:
        print("  WARNING: the least-squares system is rank deficient AND its R2 is "
              f"{fb.get('r2', float('nan')):.3f} -> the `ls` column is deliberately left blank; a fit")
        print("           that does not describe the data must not be quoted per kind.")
    ops = a.get("ops_ms") or {}
    if ops:
        print("\n=== op table (per-layer normalized by the op's OWN node count) ===")
        print(f"  {'op':16s} {'nd':>5s} {'work':4s} {'side':7s} {'wcntw':>8s} {'ub':>8s} {'uni':>8s} "
              f"{'nodes/L':>7s} {'ms/L':>7s}  population readings")
        for op, r in ops.items():
            if r["wcntw"] is None:
                continue
            pops = ",".join(f"{p['per_layer']}x{p['pop']}" for p in r["populations"]) or "none"
            msp = f"{r['wcntw_per_layer']:.3f}" if r["wcntw_per_layer"] is not None else "-"
            uni_s = f"{r['uni']:.3f}" if r["uni"] is not None else "-"
            print(f"  {op:16s} {r['nd']:5d} {'work' if r['emits'] else 'NOOP':4s} {r['side']:7s} "
                  f"{r['wcntw']:8.3f} {r['ub']:8.3f} {uni_s:>8s} "
                  f"{str(r['per_layer_nodes'] or '-'):>7s} {msp:>7s}  {pops}")
        gdn_ops = [(op, r) for op, r in ops.items()
                   if r["side"] == "gdn" and r["wcntw_per_layer"] is not None]
        if gdn_ops:
            print("\n  GDN-side op buckets per layer: " + ", ".join(
                f"{op}={r['wcntw_per_layer']:.3f}" for op, r in gdn_ops))
    print("\n=== verdict ===")
    for line in verdict(a):
        print(f"  {line}")
    r = a["reconcile"]
    print("\n=== exclusivity evidence (source side vs per-step node count) ===")
    for k, v in a["kind_counts"].items():
        if v["agrees_with_source"] is not False:
            flag = ""
        elif len(v["consistent_sides"]) == 1:
            flag = "  <== CONFLICT"
        else:
            flag = "  (inconclusive: count admits both readings)"
        print(f"  {k:18s} n={v['count']:5d}  source={v['source_side']:10s} "
              f"count-reads-as={','.join(v['consistent_sides']) or '(none)':22s}{flag}")
    conflict = steps_with_kind_conflict(a)
    inconc = kind_inconclusive(a)
    print(f"  count test: {len(conflict)} contradiction(s), {len(inconc)} inconclusive. "
          f"Neither is used to build the split - the op table's node counts do that.")
    print("\n=== reconciliation ===")
    print(f"  GDN attention-side (exclusive kinds) : {r['gdn_attention_side_per_layer']:.3f} ms/layer")
    print(f"  attn attention-side (exclusive kinds): {r['attn_attention_side_per_layer']:.3f} ms/layer")
    print(f"  exclusive delta                      : {r['exclusive_delta_ms']:+.3f} ms/layer")
    print(f"  measured union delta                 : {r['measured_delta_ms']:+.3f} ms/layer")
    print(f"  unexplained                          : {r['unexplained_ms']:+.3f} ms/layer")


# ------------------------------------------------------------------------------- selftest
def selftest():
    """Feed the parser and the classifier wrong inputs. Each must go RED, not quiet."""
    fails = []

    def expect(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: {got!r}")

    # 1. exclusivity classifier. Divisibility is weak by construction, so the test asserts the
    #    reading SET, not a single side -- a kind whose count admits two readings must say so.
    expect("count 10 -> only ATTN-only", classify(10)["sides"], ["ATTN-only"])
    # 40 admits "1 per layer x 40" AND "4 per attn layer x 10": both are whole numbers, so a count
    # alone cannot pick one -- the test asserts it says both rather than picking silently.
    expect("count 40 -> admits SHARED", "SHARED" in classify(40)["sides"], True)
    expect("count 30 -> admits GDN-only", "GDN-only" in classify(30)["sides"], True)
    expect("count 30 -> and says so (not decisive)", len(classify(30)["sides"]) > 1, True)
    expect("count 120 -> 12/attn-layer rejected as implausible", classify(120)["sides"],
           ["SHARED", "GDN-only"])
    expect("count 7 -> no reading", classify(7)["sides"], [])
    expect("count 0 -> no reading", classify(0)["sides"], [])

    # 2. the fit must recover a known per-node cost vector. The design has to be IDENTIFIABLE: an
    #    earlier version of this test used one mixture only, which makes conv and gdn_state
    #    collinear, and then "the fit is wrong" was really "the test asked an unanswerable
    #    question" -- the same defect the classifier above refuses to make.
    truth = {"gdn_state": 40_000.0, "conv": 5_000.0, "moe": 9_000.0}   # ns per node
    def dur(cnt):
        return int(sum(truth[k] * c for k, c in cnt.items()))
    synth = ([dict(a=0, b=1, dur_ns=dur({"gdn_state": 1, "conv": 2, "moe": 1}), nk=4,
                   cnt={"gdn_state": 1, "conv": 2, "moe": 1}) for _ in range(30)] +
             [dict(a=0, b=1, dur_ns=dur({"moe": 3}), nk=3, cnt={"moe": 3}) for _ in range(10)] +
             [dict(a=0, b=1, dur_ns=dur({"gdn_state": 4, "moe": 2}), nk=6,
                   cnt={"gdn_state": 4, "moe": 2}) for _ in range(20)])
    t, info = fit_kinds([{"has_nsm": True, "nsm": synth}])
    expect("identifiable design -> cond_ok", info["cond_ok"], True)
    expect("synthetic fit R2 == 1", round(info["r2"], 6), 1.0)
    if t:
        expect("recovers conv us/node", round(t["conv"]["t_ns"] / 1000.0, 1), 5.0)
        expect("recovers gdn_state us/node", round(t["gdn_state"]["t_ns"] / 1000.0, 1), 40.0)

    # 3. an unidentifiable kind must NOT come back as a confident number: two kinds in a fixed ratio
    #    leave the split arbitrary while R2 stays 1.0, so only the pivot test can catch it.
    degener = [dict(a=0, b=1, dur_ns=1000 * (i + 1), nk=2, cnt={"alpha": i + 1, "beta": i + 1})
               for i in range(6)]
    _, info2 = fit_kinds([{"has_nsm": True, "nsm": degener}])
    expect("collinear kinds -> R2 still 1.0 (so R2 is not the check)", round(info2["r2"], 6), 1.0)
    expect("collinear kinds -> cond_ok False", info2["cond_ok"], False)

    # 4. a kind with too few co-occurrence patterns must be labelled weak, not quoted as measured
    weak = [dict(a=0, b=1, dur_ns=1000, nk=1, cnt={"gdn_state": 1}) for _ in range(5)]
    t3, _ = fit_kinds([{"has_nsm": True, "nsm": weak}])
    expect("single-pattern kind -> weak", t3["gdn_state"]["ident"], "weak")

    # 5. grouping. NSM precedes its own delimiter and must land in THAT step; the GPU table's
    #    `step=` must equal the delimiter's. The first revision keyed on the GPU table (printed
    #    only 1 in 8 steps) and merged eight steps of NSM into one record -- every count came out
    #    ~8x high, which is a number that still looks plausible.
    def hdr(step, ntok):
        return (f"CGC-DECPROF: step={step} segs=41 layers=40 total=100.00 ms | wait=1.00 (1%) "
                f"cb=2.00 (2%) submit=3.00 (3%) ntok={ntok} | layer gpu_sum=4.00 union_sum=5.00 "
                f"gap_sum=6.00 ms")

    def layer(L, union):
        return (f"CGC-DECPROF all: L{L} wait=1.00 cb=0.10 submit=0.05 ms gpu=1.20 union={union} "
                f"gap=0.20 sg=1 n=1")

    def gpu(step):
        return (f"CGC-GPUNODE: step={step} seg_busy=10.00 ms | layer gpu_sum=9.50 ms | "
                f"delta=5.26% | other=1.00 ms | nkind=2")

    chunk = "\n".join([
        "CGC-NSM a=0 b=3 dur_ns=100 nk=3 gdn_state:1 conv:2",       # step 7, arrives before its hdr
        hdr(7, 1),
        layer(0, 1.30),
        gpu(7),
        "CGC-GPUNODE:   gdn_state               wcntw=   1.00   10.0% | cntw=   1.00   10.0% | "
        "lb=   0.50    5.0% | ub=   2.00   20.0%",
        "CGC-GPUNODE: work-attributed 8.00 of 10.00 ms (80.0%) | named_work_nodes=40 | "
        "residual=2.00 ms (20.0%)",
        "CGC-NSM a=3 b=6 dur_ns=200 nk=2 moe:2",                    # step 8
        hdr(8, 1),
        layer(1, 2.30),
    ])
    steps, diag = parse_steps(chunk)
    expect("two steps parsed", [s["step"] for s in steps], [7, 8])
    expect("nsm lands in ITS OWN step, not the next", steps[0]["nsm"][0]["dur_ns"], 100)
    expect("step 7 nsm buffer count", len(steps[0]["nsm"]), 1)
    expect("step 7 layer row", steps[0]["layers"][0]["union"], 1.3)
    expect("step 7 kind row", sorted(steps[0]["kinds"]), ["gdn_state"])
    expect("step 7 ntok", steps[0]["ntok"], 1)
    expect("step 8 got its own nsm", steps[1]["nsm"][0]["dur_ns"], 200)
    expect("step 8 has no kinds of its own", steps[1]["kinds"], {})
    expect("step 8 has no GPU table", steps[1]["has_gputable"], False)
    expect("gpunode header matched its step", diag["gpunode_matched"], 1)
    expect("no mismatch in a well-formed chunk", diag["gpunode_mismatch"], 0)
    expect("ntok histogram", diag["ntok"], {1: 2})

    # 6. buckets must not silently absorb an undeclared kind: membership is the source vocabulary.
    declared = {k for ms in BUCKETS.values() for k in ms}
    expect("moe is declared but shared, not gdn", BUCKET_SIDE["moe"], "shared")
    expect("no bucket member is left out of the vocabulary", "gdn_state" in declared, True)

    # 7. a header that disagrees with the delimiter must stop the analysis, not be averaged in.
    bad = "\n".join([hdr(7, 1), layer(0, 1.3), gpu(9)])
    _, diag_bad = parse_steps(bad)
    expect("disagreeing header counted", diag_bad["gpunode_mismatch"], 1)
    a_bad = analyze({"arms": {"rep2_warm": {"tag": "rep2_warm", "steps": parse_steps(bad)[0],
                                             "diag": diag_bad, "tokens": 1, "ms_per_token": 1.0}}})
    expect("analysis refuses a mis-grouped run", a_bad["verdict"].startswith("INCOMPLETE"), True)

    # 8. only ntok=1 is admissible: a prefill chunk is a different graph.
    pre = "\n".join([hdr(1, 4), layer(0, 1.3), hdr(2, 8), layer(0, 2.3)])
    steps_pre, diag_pre = parse_steps(pre)
    a_pre = analyze({"arms": {"rep2_warm": {"tag": "rep2_warm", "steps": steps_pre,
                                             "diag": diag_pre, "tokens": 1, "ms_per_token": 1.0}}})
    expect("prefill-only run is refused", a_pre["verdict"].startswith("INCOMPLETE"), True)
    expect("and says what it saw", "ntok={" in a_pre["verdict"], True)

    # 9. when the per-kind table never landed on a decode step, the absence must be STATED.
    dec_only = "\n".join([hdr(7, 1), layer(0, 1.3), layer(3, 1.0),
                          "CGC-NSM a=0 b=1 dur_ns=1000 nk=1 gdn_state:1",
                          hdr(8, 1), layer(0, 1.4), layer(3, 1.1)])
    steps_d, diag_d = parse_steps(dec_only)
    a_d = analyze({"arms": {"rep2_warm": {"tag": "rep2_warm", "steps": steps_d,
                                           "diag": diag_d, "tokens": 1, "ms_per_token": 1.0}}})
    expect("decode-only run is admissible", "verdict" not in a_d, True)
    expect("missing cross-check is stated, not implied", "crosscheck_absent" in a_d, True)
    expect("per-layer medians came from both decode steps", a_d["per_layer"]["steps"], 2)
    expect("absent self-check reported as absent, not as 0", a_d["selfcheck_delta_pct"], None)
    expect("and the reason travels with it", "selfcheck_absent" in a_d, True)

    # 10. a JSON round trip must not be able to turn all 40 layers into "GDN". Both halves are
    #     asserted, plus the guard that refuses to quote a split when only one side matched.
    def json_raw(nsm=True):
        step = {"step": 7, "ntok": 1, "kinds": {}, "ops": {}, "hot": [],
                "nsm": ([{"a": 0, "b": 1, "dur_ns": 1000, "nk": 1, "cnt": {"gdn_state": 1}}]
                        if nsm else []),
                "layers": {str(L): {"union": 2.0, "wait": 1.0, "gap": 0.1, "cb": 0.1,
                                    "submit": 0.1, "gpu": 1.5} for L in range(40)}}
        step["has_nsm"] = nsm
        inner = {"tag": "t", "steps": [step], "diag": {"steps": 1, "ntok": {1: 1}}}
        return json.loads(json.dumps({"arms": {"rep2_warm": inner}}))

    nr = normalize_raw(json_raw())
    expect("normalisation restores int layer keys",
           all(isinstance(L, int) for L in nr["arms"]["rep2_warm"]["steps"][0]["layers"]), True)
    a10 = analyze(nr)
    expect("JSON round trip keeps 30 GDN layers", a10["per_layer"]["n_gdn_layers"], 30)
    expect("JSON round trip keeps 10 attention layers", a10["per_layer"]["n_attn_layers"], 10)
    expect("and the attn side is not empty", a10["per_layer"]["attn_union_ms"], 2.0)
    # ... and with the keys LEFT as strings the guard must refuse instead of reporting a delta.
    a11 = analyze(json_raw())          # deliberately NOT normalised: this is the bad input
    expect("string layer keys -> refused, not split", a11["verdict"].startswith("INCOMPLETE"), True)
    expect("and it names the cause", "layer-index" in a11["verdict"], True)

    # 11. the FORM gate, asserted in BOTH directions (a gate that always fires is as wrong as one
    #     that never does). In the drift case the ratio is constant and the additive delta grows with
    #     the step's own union_sum -- exactly the shape measured on the r16/r18 pair, where the
    #     delta had CV 61% and correlated +0.60 with it.
    def form_raw(pairs):
        steps = []
        for i, (g, a, us) in enumerate(pairs):
            steps.append({"step": i, "ntok": 1, "kinds": {}, "ops": {}, "hot": [], "has_nsm": True,
                          "nsm": [{"a": 0, "b": 1, "dur_ns": 1000, "nk": 1,
                                   "cnt": {"gdn_state": 1}}],
                          "union_sum": us,
                          "layers": {L: {"union": (a if L in FULL_ATTN else g), "wait": 1.0,
                                         "gap": 0.1, "cb": 0.1, "submit": 0.1, "gpu": 1.0}
                                     for L in range(40)}})
        return {"arms": {"rep2_warm": {"tag": "t", "steps": steps, "tokens": 1,
                                       "ms_per_token": 1.0,
                                       "diag": {"steps": len(steps), "ntok": {1: len(steps)}}}}}

    drift = [(1.0, 0.5, 50), (2.0, 1.0, 100), (3.0, 1.5, 150), (4.0, 2.0, 200), (5.0, 2.5, 250)]
    a12 = analyze(form_raw(drift))
    v12 = " ".join(verdict(a12))
    expect("drift-shaped delta -> FORM refuses the ms/layer as an amount",
           "quote the RATIO" in v12, True)
    expect("and it offers the ratio instead", "GDN/attn = 2.000x" in v12, True)
    expect("and names what the delta was tracking", "correlates +1.00 with the step's own union_sum" in v12, True)
    expect("the stability block reports a constant ratio", round(a12["per_layer"]["stability"]["ratio_cv"], 6), 0.0)
    expect("a constant ratio's correlation is printed as undefined, not as a number",
           "n/a (constant)" in v12, True)

    flat = [(1.0, 0.5, 100), (1.02, 0.52, 100), (0.98, 0.48, 100), (1.0, 0.5, 100)]
    a13 = analyze(form_raw(flat))
    v13 = " ".join(verdict(a13))
    expect("flat delta -> quotable as an amount", "may be quoted as an amount" in v13, True)
    expect("and it does NOT raise the ratio complaint", "quote the RATIO" in v13, False)

    # 12. the width axis: the ratio must be flat across widths while the additive delta grows. Fed
    #     two synthetic widths whose attention layers scale 1x -> 4x with the GDN layers kept at a
    #     constant 2x ratio, the scan must recover 2.00x for both and a delta that grew.
    def width_steps(ntok, n, scale):
        out = []
        for i in range(n):
            out.append({"step": i, "ntok": ntok, "kinds": {}, "ops": {}, "hot": [],
                        "has_nsm": True,
                        "nsm": [{"a": 0, "b": 1, "dur_ns": 1000, "nk": 1, "cnt": {"gdn_state": 1}}],
                        "layers": {L: {"union": (4.0 * scale if L in FULL_ATTN else 8.0 * scale),
                                       "wait": 1.0, "gap": 0.1, "cb": 0.1, "submit": 0.1,
                                       "gpu": 1.0} for L in range(40)}})
        return out

    tab = width_scan_steps([width_steps(1, 3, 1.0), width_steps(4, 3, 4.0)])
    expect("width scan keeps the ratio flat at 2.00x",
           (round(tab[1]["ratio"], 2), round(tab[4]["ratio"], 2)), (2.0, 2.0))
    expect("and the additive delta grows with the width", tab[4]["delta"] > tab[1]["delta"], True)
    expect("and a width with no layer rows is absent, not zero", 2 in tab, False)

    print("\nSELFTEST " + ("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


# ------------------------------------------------------------------------- reporting
VERDICT_MIN_STEPS = 8      # below this, a per-layer median of 30 vs 10 layers is not worth quoting


def verdict(a: dict) -> list[str]:
    """The falsifiable part: what the GDN layers' extra per-layer cost is made of, and what would
    contradict it. Every number is quoted with the reading it came from and the bound that carries
    it, and the numbers the instrument cannot support are named as such."""
    out = []
    if "verdict" in a:
        return [a["verdict"]]
    p = a["per_layer"]
    ops = a.get("ops_ms") or {}
    if p["steps"] < VERDICT_MIN_STEPS:
        out.append(f"only {p['steps']} decode steps -> medians reported but no verdict "
                   f"(need >= {VERDICT_MIN_STEPS})")

    # (a) the layer-population closure, which is what makes the op table decisive: the GDN-side op
    #     population and the attention-side one must add up to the layer count.
    gdn_pop = {ops[o]["nd"] // ops[o]["per_layer_nodes"] for o in ("GATED_DELTA_NET", "SSM_CONV")
               if o in ops and ops[o]["per_layer_nodes"]}
    attn_pop = {(ops[o]["nd"] // ops[o]["per_layer_nodes"]) for o in ("ROPE", "SOFT_MAX", "SET_ROWS")
                if o in ops and ops[o]["per_layer_nodes"]}
    if gdn_pop and attn_pop:
        closes = (gdn_pop | attn_pop) == {N_GDN, N_ATTN} and sum(gdn_pop) + sum(attn_pop) == N_BASE_LAYERS
        out.append(f"populations read off the op table: GDN-side {sorted(gdn_pop)} node(s) per layer, "
                   f"attention-side {sorted(attn_pop)} -> "
                   f"{'30 + 10 = 40 closes with the layer count' if closes else 'DOES NOT CLOSE'}")

    # (b) the split itself, at the granularity the fused build actually allows. The two columns have
    #     very different standing and are quoted separately: `wcntw` is work-weighted and usable,
    #     while `ub` is "every buffer that contained this op" -- for SSM_CONV that is ~40 ms/step of
    #     buffers dominated by other work, i.e. a bound so loose it would swamp the delta (quoting the
    #     MAX of the two bounds produced a "174% of the delta" line, which is the kind of
    #     self-contradicting number that should never be printed).
    gdn_ops = (("GATED_DELTA_NET", "scan + state update (FUSED into one node, so scan and state update "
                              "are not separable in this build)"),
               ("SSM_CONV", "conv (op SSM_CONV)"))
    fused_ub = 0.0
    conv_ub_usable = None
    for o, label in gdn_ops:
        r = ops.get(o)
        if r and r["wcntw_per_layer"] is not None:
            ratio = (r["ub_per_layer"] / r["wcntw_per_layer"]) if r["wcntw_per_layer"] else 0
            note = "bound usable" if ratio <= 4 else f"bound DILUTED ({ratio:.0f}x) -> not quotable"
            out.append(f"{label}: {r['wcntw_per_layer']:.3f} ms/GDN-layer point (work-weighted), "
                       f"ub {r['ub_per_layer']:.3f} ({note})")
            if o == "GATED_DELTA_NET":
                fused_ub = r["ub_per_layer"] or 0.0
            else:
                conv_ub_usable = ratio <= 4
    point = sum((ops[o]["wcntw_per_layer"] or 0) for o in ("GATED_DELTA_NET", "SSM_CONV") if o in ops)
    # Bracket: point estimate, and the worst case that uses only bounds that survive the dilution test.
    worst = point
    if fused_ub:
        worst += fused_ub - (ops["GATED_DELTA_NET"]["wcntw_per_layer"] or 0)

    # (c) the falsification test of the premise: does conv+scan+state account for the measured delta?
    delta = p["union_delta_ms"]
    if delta > 0:
        out.append(f"FALSIFIABLE: conv+scan+state = {point:.3f} ms/GDN-layer against the measured "
                   f"{delta:+.3f} ms/layer delta = {100 * point / delta:.0f}% of it "
                   f"({100 * worst / delta:.0f}% if every defensible bound is spent). The test that "
                   f"would falsify this: an instrument showing the fused node above its own ub, "
                   f"{fused_ub:.3f} ms/GDN-layer.")
        b = a["buckets"]
        gdn_attn = sum(v["wcntw_per_layer"] or 0 for n, v in b.items() if BUCKET_SIDE[n] == "gdn")
        attn_attn = sum(v["wcntw_per_layer"] or 0 for n, v in b.items() if BUCKET_SIDE[n] == "attn")
        if gdn_attn < attn_attn:
            out.append(f"and the GDN layers' own attention side is NOT more expensive than the "
                       f"attention layers': {gdn_attn:.3f} vs {attn_attn:.3f} ms/layer (name table) "
                       f"-> the +{delta:.3f} ms/layer is not GDN attention work; the live candidates "
                       f"are the shared kinds' per-layer cost and the delta's own stability")

    # (c2) the FORM of the claim. An additive ms/layer gap is only spendable if it does not move
    #      with the regime: measured over 25 steps the delta tracked the step's own union_sum at
    #      r=+0.60 (CV 61%) while the ratio did not (r=+0.15, CV 18%). Quoting the delta without
    #      this line is how §16 produced a number that only exists at its own regime.
    sb = p.get("stability") or {}
    # `is not None`, not truthiness: a perfectly flat delta has CV exactly 0.0, and treating that as
    # "absent" would silently drop the quote-the-amount branch (the self-test's flat case caught it).
    if sb.get("delta_cv") is not None and sb.get("ratio_cv") is not None:
        # A correlation is undefined for a constant sample (zero variance), which is exactly what a
        # truly proportional gap looks like, so it must print as such rather than crash.
        f2 = lambda v: (f"{v:+.2f}" if v is not None else "n/a (constant)")  # noqa: E731
        if sb["delta_cv"] > 0.25 and sb["delta_cv"] > 2 * sb["ratio_cv"]:
            out.append(f"FORM: quote the RATIO, not the ms/layer. additive delta "
                       f"{sb['delta_mean']:+.3f} ms/layer has CV {100 * sb['delta_cv']:.0f}% and "
                       f"correlates {f2(sb['corr_delta_usum'])} with the step's own union_sum "
                       f"(which ranged {sb['usum_min']:.0f}..{sb['usum_max']:.0f} ms in this sample), "
                       f"while GDN/attn = {sb['ratio_mean']:.3f}x has CV {100 * sb['ratio_cv']:.0f}% "
                       f"and correlates {f2(sb['corr_ratio_usum'])} -> the gap scales with the "
                       f"regime, so it is not an amount of work to harvest")
        else:
            out.append(f"FORM: the additive delta is stable across the sample "
                       f"(CV {100 * sb['delta_cv']:.0f}% vs ratio CV {100 * sb['ratio_cv']:.0f}%), so "
                       f"{sb['delta_mean']:+.3f} ms/layer may be quoted as an amount")

    # (d) what the 25 t/s target would need, in the same units.
    need = 1.81 - 0.54     # docs §16: F must fall from 1.81 to 0.54 ms/layer
    out.append(f"target check: F must fall {need:.2f} ms/layer for 25 t/s. Zeroing the ENTIRE GDN"
               f" attention side buys {point:.3f} ms/layer ({100 * point / need:.0f}% of it) -- so the"
               f" GDN layer is not the lever; the remaining {need - point:.2f} ms/layer sits in the "
               f"shared per-layer cost.")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    # widthscan takes log paths, not the standard commands
    if cmd == "run":
        return run() or cmd_analyze()
    if cmd == "analyze":
        return cmd_analyze()
    if cmd == "widthscan":
        return cmd_widthscan(sys.argv[2:])
    if cmd == "reparse":
        return cmd_reparse()
    if cmd == "selftest":
        return selftest()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
