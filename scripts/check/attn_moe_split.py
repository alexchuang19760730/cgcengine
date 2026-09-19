#!/usr/bin/env python3
"""attn_moe_split.py — how much of a decode step's GPU time is ATTENTION vs MoE?

The question
------------
`docs/DECODE_STEP_BUDGET_2026-09-19.md` §16 decomposes F into union/gap/cb/submit and finds
`union` = 70%, i.e. the step is GPU-execution bound. It then bounds the attention differential
only as a LOWER BOUND (GDN layers run +0.48 ms/layer union over the 10 full-attention layers)
because "absolute attention vs MoE needs per-node timing, which this instrument does not have".

`scripts/check/gdn_split.py` (another line, 2026-09-19 18:36) answers a DIFFERENT question — where
inside a GDN layer its extra cost lives — and it explicitly refuses to use its `moe` bucket for the
difference ("`moe` is present in every layer (both types) so it is only ever used as a same-step
control, never as part of the difference").

So the attention-vs-MoE split is still open. It does not need new C++: the engine's node-level
instrument already prints a per-KIND table for EVERY kind, including `ffn_moe_*`
(`ggml-backend.cpp`, `CGC_GPU_NODES=1`). This tool turns that table into the split, with bounds.

Why bounds and not one number
-----------------------------
Inside a command buffer the node->kind split is a GUESS: an all-VIEW buffer was once reported at
592 us/node, which is impossible for a buffer that encodes nothing. So the engine prints three
columns and this tool keeps all three:
    wcntw = buffer duration split over the nodes that actually ENCODE (the ranking column)
    lb    = sum of the durations of buffers where the kind is ALONE  (each is exactly that kind)
    ub    = sum of the durations of every buffer that CONTAINS the kind
The truth is in [sum(lb), sum(ub)] per bucket. A bucket whose lb already dominates has passed the
test without finer machinery; one whose ub is small has failed it.

Instrument self-check (the reason this tool can refuse)
------------------------------------------------------
`CGC-GPUNODE` prints `seg_busy` and `layer gpu_sum`, which are two sums of the SAME per-segment
Metal busy time; `delta` is their relative difference. Non-zero delta means the node ranges or the
buffer/node mapping are wrong and every row below it means nothing. This tool FAILs on that instead
of reporting the rows. It also prints the engine's own `residual` (segment busy that no named work
node accounted for), because that residual GROWS with the command-buffer count and is therefore the
thing that forbids quoting an absolute seg_busy.

Usage
-----
    python3 scripts/check/attn_moe_split.py run                 # launch, measure, analyze
    python3 scripts/check/attn_moe_split.py analyze <server.log>
    python3 scripts/check/attn_moe_split.py selftest            # must go red on wrong input
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---------------------------------------------------------------- the engine's lines

HDR = re.compile(
    r"CGC-GPUNODE: step=(\d+) seg_busy=([\d.]+) ms \| layer gpu_sum=([\d.]+) ms "
    r"\| delta=([-\d.]+)% \| other=([\d.]+) ms \| nkind=(\d+)")
ROW = re.compile(
    r"CGC-GPUNODE:\s\s+(\S+)\s+wcntw=\s*([\d.]+)\s+([\d.]+)% \| cntw=\s*([\d.]+)\s+([\d.]+)% "
    r"\| lb=\s*([\d.]+)\s+([\d.]+)% \| ub=\s*([\d.]+)\s+([\d.]+)%")
WORK = re.compile(
    r"CGC-GPUNODE: work-attributed ([\d.]+) of ([\d.]+) ms \(([\d.]+)%\) "
    r"\| named_work_nodes=(\d+) \| residual=([\d.]+) ms \(([\d.]+)%\)")

# ---------------------------------------------------------------- buckets
#
# Membership is by NAME PREFIX, and every row that matches nothing is reported as `other` rather
# than absorbed -- the same rule gdn_split.py uses for its AMBIG flag. The four write buckets are
# the ones the question is about; `ffn_dense` is kept separate because a bare `ffn_` prefix would
# otherwise silently pull norm/scale nodes into MoE and inflate it.

def bucket_of(kind: str) -> str:
    k = kind
    if k.startswith("ffn_moe") or k in ("top_k", "ffn_moe_add", "shared_expert_gate"):
        return "moe"
    if k.startswith("ffn_") or k in ("FFN", "ffn"):
        return "ffn_dense"          # shared expert / dense FFN -- not the routed MoE gather
    if k.startswith("attn") or k in ("Qcur", "Kcur", "Vcur", "kqv_out", "rope", "soft_max",
                                     "q_cur", "k_cur", "v_cur"):
        return "attention"
    if k.startswith("gdn") or k.startswith("conv") or k.startswith("dn") or \
       k in ("alpha", "beta", "a_softplus", "z-", "qkv_mixed"):
        return "gdn"
    return "other"


BUCKETS = ("attention", "moe", "gdn", "ffn_dense", "other")


# ---------------------------------------------------------------- parsing

def parse(text: str):
    """Return a list of per-step records. A step record is the header + its kind rows + the
    work-attributed line, in the order the engine printed them."""
    lines = text.splitlines()
    steps, cur = [], None
    for ln in lines:
        m = HDR.search(ln)
        if m:
            if cur is not None:
                steps.append(cur)
            cur = {
                "step": int(m.group(1)), "seg_busy": float(m.group(2)),
                "gpu_sum": float(m.group(3)), "delta_pct": float(m.group(4)),
                "other_ms": float(m.group(5)), "nkind": int(m.group(6)), "rows": [],
            }
            continue
        if cur is None:
            continue
        m = ROW.search(ln)
        if m:
            cur["rows"].append({
                "kind": m.group(1),
                "wcntw": float(m.group(2)), "cntw": float(m.group(4)),
                "lb": float(m.group(6)), "ub": float(m.group(8)),
            })
            continue
        m = WORK.search(ln)
        if m:
            cur["work_attr"] = float(m.group(1))
            cur["work_attr_pct"] = float(m.group(3))
            cur["residual"] = float(m.group(5))
            cur["residual_pct"] = float(m.group(6))
            steps.append(cur)
            cur = None
    if cur is not None:
        steps.append(cur)
    return [s for s in steps if s["rows"]]


def aggregate(steps):
    """Per-step bucket sums, then medians across steps. Percentages are of that step's seg_busy."""
    per_step = []
    for s in steps:
        acc = {b: {"wcntw": 0.0, "lb": 0.0, "ub": 0.0} for b in BUCKETS}
        for r in s["rows"]:
            b = bucket_of(r["kind"])
            acc[b]["wcntw"] += r["wcntw"]
            acc[b]["lb"] += r["lb"]
            acc[b]["ub"] += r["ub"]
        per_step.append({
            "step": s["step"], "seg_busy": s["seg_busy"], "delta_pct": s["delta_pct"],
            "work_attr_pct": s.get("work_attr_pct"), "residual_pct": s.get("residual_pct"),
            "buckets": acc,
        })

    def med(sel):
        v = [sel(p) for p in per_step if sel(p) is not None]
        return st.median(v) if v else None

    out = {
        "n_steps": len(per_step),
        "seg_busy_med": med(lambda p: p["seg_busy"]),
        "delta_pct_med": med(lambda p: abs(p["delta_pct"])),
        "delta_pct_worst": max((abs(p["delta_pct"]) for p in per_step), default=None),
        "work_attr_pct_med": med(lambda p: p["work_attr_pct"]),
        "residual_pct_med": med(lambda p: p["residual_pct"]),
        "buckets": {},
        "per_step": per_step,
    }
    for b in BUCKETS:
        out["buckets"][b] = {
            col: med(lambda p, b=b, col=col: 100.0 * p["buckets"][b][col] / p["seg_busy"]
                     if p["seg_busy"] else None)
            for col in ("wcntw", "lb", "ub")
        }
        out["buckets"][b]["ms_wcntw"] = med(lambda p, b=b: p["buckets"][b]["wcntw"])
        out["buckets"][b]["ms_lb"] = med(lambda p, b=b: p["buckets"][b]["lb"])
        out["buckets"][b]["ms_ub"] = med(lambda p, b=b: p["buckets"][b]["ub"])
    return out


# ---------------------------------------------------------------- step shapes (MUST be split)

def group_key(step):
    """Tables come from DIFFERENT step shapes and must never be averaged together.

    The engine prints ONE ROW PER KIND PRESENT IN THAT STEP, so the row count IS the shape
    signature -- no name heuristic is needed, and none works: the first revision tried "does it
    contain ffn_moe/attn/gdn/conv" and every table matched, because the MTP layer is itself a GDN
    layer and carries conv/gdn_* rows. MEASURED on a real 2-request log (`-n 128`, prod25, MTP on):
    38 tables = 17 with 44 rows (all 40 trunk layers) + 21 with 7 rows (the single MTP layer).
    Averaging them divides every trunk kind by ~2.2 and is how a first read reported
    `attention 0.0% / moe 2.5%` while the raw table plainly showed `ffn_moe_ 8.6%`.

    The correction is by shape, and the shape here is measured, not assumed.
    """
    return "kinds=%d" % len(step["rows"])


def aggregate_by_group(steps):
    """[(label, n_steps, agg), ...] ordered by descending n, so the dominant shape prints first."""
    groups = {}
    for s in steps:
        groups.setdefault(group_key(s), []).append(s)
    return [(lab, len(v), aggregate(v))
            for lab, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]


# ---------------------------------------------------------------- verdict

DELTA_FAIL = 1.0        # % : seg_busy vs layer gpu_sum. The engine's own self-check.
RESIDUAL_WARN = 25.0    # % : named-work residual. Above this, shares are a direction, not a number.


def verdict(a):
    if a["n_steps"] == 0:
        return "INCOMPLETE: no CGC-GPUNODE block -- is CGC_DECODE_PROFILE=1 CGC_GPU_NODES=1 set?"
    if a["delta_pct_worst"] is not None and a["delta_pct_worst"] > DELTA_FAIL:
        return (f"INSTRUMENT FAIL: |delta| up to {a['delta_pct_worst']:.2f}% > {DELTA_FAIL}% -- "
                f"seg_busy != layer gpu_sum means the node->buffer mapping is wrong; rows are void.")
    sep = ""
    b = a["buckets"]
    if b["attention"]["lb"] is not None and b["moe"]["ub"] is not None:
        if b["attention"]["lb"] > b["moe"]["ub"]:
            sep = "ATTENTION DOMINATES (lb_attn > ub_moe)"
        elif b["moe"]["lb"] > b["attention"]["ub"]:
            sep = "MOE DOMINATES (lb_moe > ub_attn)"
        else:
            sep = "INTERVALS OVERLAP -- this step cannot separate them; the split is unidentifiable here"
    if a["residual_pct_med"] is not None and a["residual_pct_med"] > RESIDUAL_WARN:
        sep += f" | WARN residual {a['residual_pct_med']:.1f}% > {RESIDUAL_WARN}%: quote bounds, not points"
    return sep or "OK"


# ---------------------------------------------------------------- ggml-OP level (CGC-GPUOPS)

OPSHDR = re.compile(
    r"CGC-GPUOPS: step=(\d+) total=([\d.]+) ms nodes_all=(\d+) nodes_work=(\d+) "
    r"nodes_named=(\d+) nop=(\d+)")
OPSROW = re.compile(
    r"CGC-GPUOPS:\s+(\S+)\s+nd=\s*(\d+) (work|NOOP) wcntw=\s*([\d.]+)\s+([\d.]+)%")
NSM = re.compile(r"CGC-NSM a=(\d+) b=(\d+) dur_ns=(\d+) nk=(\d+)")
# [CGC 2026-09-19] `CGC-GPUOPK` -- the (kind, op) split of the kind table's wcntw column.
# A kind NAME is a label; the OP is the work. This is what IDENTIFIES `node` (ggml auto-names
# UNNAMED nodes `node_<i>`, ggml.c:7192) instead of inferring it from two separate tables.
OPK = re.compile(r"CGC-GPUOPK:\s+(\S+)\s+(\S+)\s+([\d.]+) ms\s+([\d.]+)% of kind\s+\|\s+([\d.]+)% of step")
NSMKV = re.compile(r"(\S+):(\d+)")


def parse_ops(text):
    steps, cur = [], None
    for ln in text.splitlines():
        if not ln.startswith("CGC-GPUOPS"):
            continue
        m = OPSHDR.search(ln)
        if m:
            if cur:
                steps.append(cur)
            cur = {"step": int(m.group(1)), "total": float(m.group(2)),
                   "nodes_all": int(m.group(3)), "nodes_work": int(m.group(4)), "rows": []}
            continue
        if cur is None:
            continue
        m = OPSROW.search(ln)
        if m:
            cur["rows"].append({"op": m.group(1), "nd": int(m.group(2)),
                                "noop": m.group(3) == "NOOP", "wcntw_pct": float(m.group(5))})
    if cur:
        steps.append(cur)
    return steps


def aggregate_ops(steps):
    """Group by `nodes_all`.

    The op table prints ALL ops in EVERY shape, so the ROW COUNT is NOT the shape signature here --
    the kind table's lesson does not transfer. The graph size is. MEASURED on a real 2-request log:
    {45: 21, 4064: 1, 4068: 1, 4070: 1, 4072: 3, 4076: 11} -- the 21 x 45 are the single MTP layer,
    the ~4076 are the 40 trunk layers. Reading the mixed median reported CPY at 466% of the nodes.
    """
    g = {}
    for s in steps:
        g.setdefault(s["nodes_all"], []).append(s)
    out = []
    for na, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
        rows = {}
        for s in v:
            for r in s["rows"]:
                rows.setdefault(r["op"], []).append(r)
        agg = [{"op": op, "noop": all(r["noop"] for r in rs),
                "wcntw_pct": st.median([r["wcntw_pct"] for r in rs]),
                "nd": st.median([r["nd"] for r in rs])} for op, rs in rows.items()]
        agg.sort(key=lambda r: -r["wcntw_pct"])
        out.append({"nodes_all": na, "n_tables": len(v),
                    "nodes_work": st.median([s["nodes_work"] for s in v]),
                    "total_ms": st.median([s["total"] for s in v]), "rows": agg})
    return out


def parse_nsm(text):
    bufs = []
    for ln in text.splitlines():
        m = NSM.search(ln)
        if m:
            bufs.append({"dur_ns": int(m.group(3)), "nk": int(m.group(4)),
                         "kinds": [k for k, _ in NSMKV.findall(ln.split("nk=" + m.group(4), 1)[-1])]})
    return bufs


def nsm_report(bufs):
    """Whether `lb` can EVER rank anything.

    `lb` is the sum over buffers where a kind is ALONE. If no buffer is ever solo, lb is 0 for every
    kind BY CONSTRUCTION and [lb, ub] cannot separate two kinds -- a mechanism, not bad luck. This is
    the engine's own documented failure mode, and the NSM lines let it be checked instead of assumed.
    """
    if not bufs:
        return {"n": 0, "verdict": "INCOMPLETE: no CGC-NSM lines (needs CGC_GPU_NODES_MATRIX=1)"}
    solo = [b for b in bufs if b["nk"] == 1]
    hist = {}
    for b in bufs:
        hist[b["nk"]] = hist.get(b["nk"], 0) + 1
    v = (("BOUNDS ARE VOID: 0 of %d buffers are solo => lb == 0 for EVERY kind by construction; "
          "[lb, ub] cannot separate anything. Quote `wcntw` only, and call it a model.") % len(bufs)
         if not solo else
         "%d solo buffers => lb is informative for %d kind(s)" %
         (len(solo), len({k for b in solo for k in b["kinds"]})))
    return {"n": len(bufs), "total_ms": sum(b["dur_ns"] for b in bufs) / 1e6,
            "range_hist": dict(sorted(hist.items())), "solo_total": len(solo), "verdict": v}


ROWCOUNT = re.compile(r"CGC-GPUNODE:\s+(?!\*bywork)(\S+)\s+wcntw=")


def parse_opk(text):
    """Tag every CGC-GPUOPK row with the SHAPE of the CGC-GPUNODE block it sits inside.

    Third instance of the SAME trap today, and this time I introduced it: the op table prints every
    op in every shape, the kind table prints one row per kind PRESENT, and CGC-GPUOPK inherits
    whichever block encloses it. Aggregating without that shape mixed the 40-trunk-layer step with
    the single-MTP-layer step and reported `node 75.3% of step` for a kind that is 13.1%.
    The shape key is the number of kind rows already printed in the enclosing block -- measured,
    not assumed, and it needs no change to the engine: the rows precede this block.
    """
    rows, nkind = [], 0
    for ln in text.splitlines():
        if ln.startswith("CGC-GPUNODE: step="):
            nkind = 0
            continue
        if ROWCOUNT.search(ln):
            nkind += 1
            continue
        m = OPK.search(ln)
        if m:
            rows.append({"shape": nkind, "kind": m.group(1), "op": m.group(2),
                         "ms": float(m.group(3)), "pct_kind": float(m.group(4)),
                         "pct_step": float(m.group(5))})
    return rows


def opk_report(rows):
    """{shape: [kind rows]} -- split by SHAPE first, then per kind its op mix. None when absent.

    None, not an empty report: a log without these lines means the instrument was not built or
    CGC_GPU_OPS was not set -- which must not read as "every kind has no ops".
    """
    if not rows:
        return None
    shapes = {}
    for r in rows:
        byk = shapes.setdefault(r["shape"], {})
        byk.setdefault(r["kind"], {}).setdefault(r["op"], []).append(r)
    out = {}
    for shape, byk in shapes.items():
        ks = []
        for kind, ops in byk.items():
            agg = [{"op": o, "pct_kind": st.median([x["pct_kind"] for x in v]),
                    "pct_step": st.median([x["pct_step"] for x in v])} for o, v in ops.items()]
            agg.sort(key=lambda r: -r["pct_step"])
            ks.append({"kind": kind, "ops": agg, "pct_step_total": sum(r["pct_step"] for r in agg)})
        ks.sort(key=lambda k: -k["pct_step_total"])
        out[shape] = ks
    return out


def cmd_ops(args):
    text = open(args.log, errors="replace").read()
    groups = aggregate_ops(parse_ops(text))
    print("=" * 78)
    print("ggml-OP level (CGC-GPUOPS) + the solo check that decides whether [lb,ub] can rank")
    print("=" * 78)
    print(f"log: {args.log}")
    if not groups:
        print("INCOMPLETE: no CGC-GPUOPS block -- is CGC_GPU_OPS=1 set (needs CGC_GPU_NODES=1)?")
        return 0
    print("shapes by nodes_all: " + ", ".join(f"{g['nodes_all']}(n={g['n_tables']})" for g in groups))
    print("   The op table prints ALL ops in every shape, so its signature is nodes_all, not the row")
    print("   count -- the kind table's lesson does NOT transfer.")
    for g in groups:
        if g["n_tables"] < 2:
            continue
        print()
        print(f"--- nodes_all={g['nodes_all']}  (tables: {g['n_tables']}) ---")
        print(f"total {g['total_ms']:.1f} ms | nodes_all {g['nodes_all']} | nodes_work "
              f"{g['nodes_work']:.0f} ({100*(1-g['nodes_work']/g['nodes_all']):.0f}% of nodes emit NO work)")
        print(f"{'op':<16}{'wcntw%':>9}{'nodes':>8}{'nodes%':>9}  flags")
        print("-" * 78)
        for r in g["rows"]:
            print(f"{r['op']:<16}{r['wcntw_pct']:>9.2f}{r['nd']:>8.0f}"
                  f"{100*r['nd']/g['nodes_all']:>8.1f}%  {'NOOP' if r['noop'] else ''}")
    opk = opk_report(parse_opk(text))
    if opk:
        print()
        print("--- KIND x OP (CGC-GPUOPK): what a kind is actually made of ---")
        print("   A kind NAME is a label; the OP is the work. A kind's rows sum to its wcntw entry.")
        print("   Split by shape first -- grouping without it read `node` at 75% of a step.")
        for shape, ks in sorted(opk.items(), key=lambda kv: -len(kv[1])):
            print()
            print("  [shape: %d kind rows in the enclosing CGC-GPUNODE block]" % shape)
            for k in ks[:10]:
                ops = k["ops"][:6]
                mix = ", ".join("%s %.0f%%" % (r["op"], r["pct_kind"]) for r in ops)
                print("  %-18s %5.1f%% of step   %s" % (k["kind"], k["pct_step_total"], mix))
    else:
        print()
        print("  (no CGC-GPUOPK lines: needs the 2026-09-19 KIND x OP instrument AND "
              "CGC_GPU_OPS=1; the block prints on a 1-in-8 step cadence)")
    bufs = parse_nsm(text)
    nr = nsm_report(bufs)
    print()
    print(f"NSM (one line per command buffer): {nr['n']} buffers, {nr.get('total_ms', 0):.1f} ms total")
    if nr.get("range_hist"):
        print(f"  nodes-per-buffer histogram: {nr['range_hist']}")
    print(f"  VERDICT: {nr['verdict']}")
    return 0


# ---------------------------------------------------------------- reporting

def _report_one(a, label, n):
    npad = 20 - len(label)
    print(f"--- step shape: {label}{' ' * max(npad, 1)}(tables: {n}) ---")
    print(f"seg_busy (median)  : {a['seg_busy_med']:.2f} ms" if a["seg_busy_med"] else "seg_busy: n/a")
    print(f"self-check delta   : median {a['delta_pct_med']:.3f}%  worst {a['delta_pct_worst']:.3f}% "
          f"(threshold {DELTA_FAIL}%)" if a["delta_pct_med"] is not None else "self-check: n/a")
    if a["work_attr_pct_med"] is not None:
        print(f"named-work mass    : {a['work_attr_pct_med']:.1f}% of seg_busy  "
              f"(residual {a['residual_pct_med']:.1f}% is UNATTRIBUTED -- it grows with n_cb)")
    print()
    print(f"{'bucket':<12}{'wcntw%':>9}{'lb%':>9}{'ub%':>9}   {'ms(wcntw)':>10}{'ms(lb)':>9}{'ms(ub)':>9}")
    print("-" * 78)
    for b in BUCKETS:
        d = a["buckets"][b]
        if d["wcntw"] is None:
            continue
        print(f"{b:<12}{d['wcntw']:>9.1f}{d['lb']:>9.1f}{d['ub']:>9.1f}   "
              f"{d['ms_wcntw']:>10.2f}{d['ms_lb']:>9.2f}{d['ms_ub']:>9.2f}")
    print()
    print(f"VERDICT: {verdict(a)}")
    print()


def report(groups, log_path):
    print("=" * 78)
    print("ATTENTION vs MoE  (per-step GPU kind table, split by STEP SHAPE first)")
    print("=" * 78)
    print(f"log                : {log_path}")
    if not groups:
        print("VERDICT: INCOMPLETE: no CGC-GPUNODE block -- is "
              "CGC_DECODE_PROFILE=1 CGC_GPU_NODES=1 set? (Refusing: an empty sample is not a zero.)")
        return
    print(f"step shapes present: {', '.join(f'{lab}(n={n})' for lab, n, _ in groups)}")
    print("   Shape == number of kind rows in that step, because the engine prints one row per kind")
    print("   PRESENT. Measured on a 2-request prod25 log: 17 tables x 44 rows (40 trunk layers)")
    print("   + 21 tables x 7 rows (the single MTP layer). Averaging the two dilutes every trunk")
    print("   kind by ~2.2x -- that mistake read attention 0.0% while the raw table said 6-8%.")
    print()
    for label, n, a in groups:
        _report_one(a, label, n)
    print("Read it this way: wcntw ranks, [lb, ub] bounds. A bucket whose lb already exceeds the")
    print("other's ub is separated WITHOUT needing a better split weight; where the intervals")
    print("overlap, this step cannot answer the question and no amount of averaging will.")


# ---------------------------------------------------------------- run

MEASURE_ENV = ("CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256;"
               "CGC_DECODE_PROFILE=1;CGC_DECODE_PROFILE_ALL=1;CGC_GPU_TIMING=1;"
               "CGC_GPU_NODES=1;CGC_GPU_OPS=1;CGC_GPU_NODES_MATRIX=1;CGC_GPU_NODES_TRACE=1")


def newest_server_log():
    fs = sorted(glob.glob(os.path.join(ROOT, "Backup", "cgc_logs", "llama_server_*.log")),
                key=os.path.getmtime)
    return fs[-1] if fs else None


def cmd_run(args):
    before = newest_server_log()
    before_t = os.path.getmtime(before) if before else 0.0
    out = args.out or os.path.join(ROOT, "Backup", "phase_decomp", "attn_moe_split")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "http_duo.py"),
           "--profile", args.profile, "--reps", str(args.reps),
           "--decode-predict", str(args.decode_predict), "--port", args.port,
           "--extra-env", MEASURE_ENV,
           "--logdir", out,
           "--json", os.path.join(out, "result.json")]
    print("$ " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"[http_duo rc={rc} in {time.time()-t0:.0f}s]")
    log = newest_server_log()
    if log is None or (before and log == before and os.path.getmtime(log) <= before_t):
        print("!! the server log did not move: nothing was measured. Refusing to analyze a stale log.")
        return 2
    text = open(log, errors="replace").read()
    steps = parse(text)
    groups = aggregate_by_group(steps)
    report(groups, log)
    jp = os.path.join(out, "attn_moe_split.json")
    # Identify the binary at write time. Without it this product cannot be joined to a generation
    # on the timeline (0 of 186 capability products were placeable because of exactly this gap) --
    # and a decomposition that does not say which build it describes cannot be compared with an
    # oracle reading. See scripts/check/engine_identity.py.
    _spec = importlib.util.spec_from_file_location(
        "engine_identity", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "engine_identity.py"))
    _ei = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ei)
    doc = _ei.stamp({"log": log, "n_tables": len(steps),
                     "groups": {lab: {"n_tables": n, "agg": a} for lab, n, a in groups}})
    with open(jp, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nsaved: {jp}")
    return 0


def cmd_analyze(args):
    text = open(args.log, errors="replace").read()
    steps = parse(text)
    report(aggregate_by_group(steps), args.log)
    return 0


# ---------------------------------------------------------------- selftest

def _synthetic(step, seg, gpu, delta, rows, work=None):
    s = [f"CGC-GPUNODE: step={step} seg_busy={seg} ms | layer gpu_sum={gpu} ms | "
         f"delta={delta}% | other=0.00 ms | nkind={len(rows)}"]
    for name, w, lb, ub in rows:
        s.append(f"CGC-GPUNODE:   {name:<24} wcntw={w:7.2f} {w/seg*100:5.1f}% | "
                 f"cntw={w:7.2f} {w/seg*100:5.1f}% | lb={lb:7.2f} {lb/seg*100:5.1f}% | "
                 f"ub={ub:7.2f} {ub/seg*100:5.1f}%")
    if work is not None:
        s.append(f"CGC-GPUNODE: work-attributed {work:.2f} of {seg:.2f} ms ({work/seg*100:.1f}%) | "
                 f"named_work_nodes=10 | residual={seg-work:.2f} ms ({(seg-work)/seg*100:.1f}%)")
    return "\n".join(s)


def cmd_selftest(_args):
    fails = []

    def ok(name, cond):
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            fails.append(name)

    # 1. a clean step: attention 60, moe 20, other 20 -> bucketed correctly
    t = _synthetic(1, 100.0, 100.0, 0.0,
                   [("attn_q", 60.0, 60.0, 60.0), ("ffn_moe_gate", 20.0, 20.0, 20.0),
                    ("norm", 20.0, 20.0, 20.0)], work=100.0)
    a = aggregate(parse(t))
    b = a["buckets"]
    ok("clean step: n_steps == 1", a["n_steps"] == 1)
    ok("clean step: attention == 60%", abs(b["attention"]["wcntw"] - 60.0) < 0.01)
    ok("clean step: moe == 20%", abs(b["moe"]["wcntw"] - 20.0) < 0.01)
    ok("clean step: other == 20% (never silently absorbed)", abs(b["other"]["wcntw"] - 20.0) < 0.01)
    ok("clean step: verdict names attention", "ATTENTION DOMINATES" in verdict(a))

    # 2. the engine's own self-check must be able to VOID the table
    t = _synthetic(1, 130.0, 100.0, 30.0, [("attn_q", 130.0, 130.0, 130.0)], work=130.0)
    a = aggregate(parse(t))
    ok("delta 30%: verdict is INSTRUMENT FAIL", verdict(a).startswith("INSTRUMENT FAIL"))

    # 3. no table at all -> INCOMPLETE, never a zero
    a = aggregate(parse("nothing here\n"))
    ok("empty log: INCOMPLETE", verdict(a).startswith("INCOMPLETE"))

    # 4. overlapping intervals must be refused, not resolved
    t = _synthetic(1, 100.0, 100.0, 0.0,
                   [("attn_q", 40.0, 20.0, 70.0), ("ffn_moe_gate", 40.0, 25.0, 80.0)], work=100.0)
    a = aggregate(parse(t))
    ok("overlap: refused as unidentifiable", "OVERLAP" in verdict(a))

    # 5. a bare `ffn_` prefix must NOT be counted as routed MoE
    t = _synthetic(1, 100.0, 100.0, 0.0,
                   [("ffn_norm", 50.0, 50.0, 50.0), ("ffn_moe_gate", 50.0, 50.0, 50.0)], work=100.0)
    a = aggregate(parse(t))
    ok("ffn_norm is NOT moe", abs(a["buckets"]["moe"]["wcntw"] - 50.0) < 0.01)
    ok("ffn_norm lands in ffn_dense", abs(a["buckets"]["ffn_dense"]["wcntw"] - 50.0) < 0.01)

    # 6. residual must be reported, and a big one must warn
    t = _synthetic(1, 100.0, 100.0, 0.0, [("attn_q", 40.0, 40.0, 40.0)], work=40.0)
    a = aggregate(parse(t))
    ok("residual is surfaced", a["residual_pct_med"] is not None and a["residual_pct_med"] > 50)
    ok("big residual warns", "WARN residual" in verdict(a))

    # 7. two step shapes in ONE log must be SPLIT, not averaged. This is the defect this tool
    #    shipped with for its first measurement: 21 tiny draft tables outvoted 17 real verify
    #    tables and `attention` read 0.0% while the raw table said ffn_moe_ 7.7%.
    full = _synthetic(1, 100.0, 100.0, 0.0,
                      [("attn_q", 60.0, 60.0, 60.0), ("ffn_moe_gate", 40.0, 40.0, 40.0)], work=100.0)
    tiny = _synthetic(9, 10.0, 10.0, 0.0, [("node", 10.0, 10.0, 10.0)], work=10.0)
    groups = aggregate_by_group(parse(full + "\n" + tiny))
    ok("two shapes -> two groups", len(groups) == 2 and groups[0][0] != groups[1][0])
    g_two = [g for g in groups if g[0] == "kinds=2"][0][2]
    ok("split keeps attention at 60% in the 2-kind group",
       abs(g_two["buckets"]["attention"]["wcntw"] - 60.0) < 0.01)
    mixed = aggregate(parse(full + "\n" + tiny))     # the WRONG way, kept as a negative control
    ok("negative control: the un-split way really does dilute (<40%)",
       mixed["buckets"]["attention"]["wcntw"] < 40.0)

    # 8. the op table's shape signature is nodes_all, NOT the row count (a real log reported CPY
    #    at 466% of the nodes when the two shapes were mixed)
    op_t = ("CGC-GPUOPS: step=1 total=100.00 ms nodes_all=100 nodes_work=60 nodes_named=100 nop=2\n"
            "CGC-GPUOPS:   MUL              nd=   60 work wcntw=   30.00  30.0%  cntw=  1.0  1.0%  ub= 1.0 1.0%\n"
            "CGC-GPUOPS:   VIEW             nd=   40 NOOP wcntw=    0.00   0.0%  cntw=  1.0  1.0%  ub= 1.0 1.0%\n"
            "CGC-GPUOPS: step=2 total=10.00 ms nodes_all=10 nodes_work=6 nodes_named=10 nop=2\n"
            "CGC-GPUOPS:   MUL              nd=    6 work wcntw=    3.00  30.0%  cntw=  1.0  1.0%  ub= 1.0 1.0%\n"
            "CGC-GPUOPS:   VIEW             nd=    4 NOOP wcntw=    0.00   0.0%  cntw=  1.0  1.0%  ub= 1.0 1.0%")
    og = aggregate_ops(parse_ops(op_t))
    ok("op table: two shapes split by nodes_all", len(og) == 2 and og[0]["nodes_all"] == 100)
    ok("op table: NOOP flagged", [r for r in og[0]["rows"] if r["op"] == "VIEW"][0]["noop"])
    ok("op table: a per-op share is a share, never >100%",
       all(0 <= r["wcntw_pct"] <= 100 for g2 in og for r in g2["rows"]))

    # 9. the solo check must REFUSE the bounds when nothing is ever solo, and say so
    nsm_none = "CGC-NSM a=1 b=7 dur_ns=1000 nk=6 cache:1 norm:1 node:1 attn_norm:1 ffn_moe_ :1 ffn_:1 leaf:1"
    nr = nsm_report(parse_nsm(nsm_none))
    ok("nsm: no solo buffers -> BOUNDS ARE VOID", "BOUNDS ARE VOID" in nr["verdict"])
    nsm_solo = "CGC-NSM a=1 b=2 dur_ns=1000 nk=1 ffn_moe_gate:1"
    ok("nsm: a solo buffer -> lb becomes informative",
       "informative" in nsm_report(parse_nsm(nsm_solo))["verdict"])
    ok("nsm: empty sample -> INCOMPLETE, not a zero",
       nsm_report([])["verdict"].startswith("INCOMPLETE"))

    # 10. the (kind, op) table groups by KIND, a kind's mix sums to 100%, and absence is None
    opk_t = ("CGC-GPUOPK: node                   ADD                8.00 ms  57.1% of kind |   6.9% of step\n"
             "CGC-GPUOPK: node                   MUL                6.00 ms  42.9% of kind |   5.2% of step\n"
             "CGC-GPUOPK: cache                  CPY                4.00 ms 100.0% of kind |   3.5% of step")
    orr = opk_report(parse_opk(opk_t))
    ok("kind x op: grouped by kind", [k["kind"] for k in orr[0]] == ["node", "cache"])
    ok("kind x op: a kind's mix sums to ~100%",
       abs(sum(r["pct_kind"] for r in orr[0][0]["ops"]) - 100.0) < 0.01)
    ok("kind x op: absent -> None, not a zero", opk_report(parse_opk("nothing")) is None)
    # 11. the SAME trap a third time: a GPUOPK row must inherit its enclosing block's shape
    shape_t = ("CGC-GPUNODE: step=1 seg_busy=100.00 ms | delta=0.00% | nkind=2\n"
               "CGC-GPUNODE:   node                   wcntw=  10.00  50.0% | cntw= 1.0 1.0% | lb= 0.0 0.0% | ub= 1.0 1.0%\n"
               "CGC-GPUNODE:   cache                  wcntw=  10.00  50.0% | cntw= 1.0 1.0% | lb= 0.0 0.0% | ub= 1.0 1.0%\n"
               "CGC-GPUOPK: node                   ADD                8.00 ms  57.1% of kind |  6.9% of step\n"
               "CGC-GPUNODE: step=9 seg_busy=10.00 ms | delta=0.00% | nkind=2\n"
               "CGC-GPUNODE:   node                   wcntw=   5.00  50.0% | cntw= 1.0 1.0% | lb= 0.0 0.0% | ub= 1.0 1.0%\n"
               "CGC-GPUOPK: node                   MUL                4.00 ms  50.0% of kind | 40.0% of step")
    shapes = opk_report(parse_opk(shape_t))
    ok("kind x op: two shapes kept apart", sorted(shapes) == [1, 2])
    # the shape key is the kind-row count of the ENCLOSING block: step=1 printed two rows (=> 2),
    # step=9 printed one (=> 1). The ADD row belongs to the two-row block; MUL to the one-row block.
    ok("kind x op: shape 2 holds the ADD row, shape 1 holds MUL",
       shapes[2][0]["ops"][0]["op"] == "ADD" and shapes[1][0]["ops"][0]["op"] == "MUL")

    # 12. per-layer: the engine states gap lives inside the PREVIOUS segment's hook+submit
    #     window, so gap == cb+submit must read back as a ratio of 1.
    def lay(il, wait, cb, sub, gpu, uni, gap):
        return ("CGC-DECPROF all: L%d wait=%.2f cb=%.2f submit=%.2f ms gpu=%.2f union=%.2f gap=%.2f sg=1 n=1\n" % (il, wait, cb, sub, gpu, uni, gap))
    body = "".join(lay(i, 3.0, 0.5, 0.5, 4.0, 3.5, 1.0) for i in range(40))
    rep = layer_report(body)
    ok("layers: table found", rep is not None and rep["n_steps"] == 1)
    ok("layers: 30 GDN / 10 full-attn",
       rep["families"]["gdn"]["n_layers"] == 30 and rep["families"]["full_attn"]["n_layers"] == 10)
    ok("layers: gap/hook == 1 by construction", abs(rep["ratios"]["gap_over_hook"] - 1.0) < 1e-9)
    ok("layers: wall == wait+cb+submit",
       abs(rep["sums"]["wall"] - (3.0 + 0.5 + 0.5) * 40) < 0.5)
    # negative control: no idle at all -> the verdict must not claim an idle story
    noidle = "".join(lay(i, 3.0, 0.5, 0.5, 4.0, 3.5, 0.0) for i in range(40))
    ok("layers: zero gap reads as zero", layer_report(noidle)["ratios"]["gap_over_hook"] == 0.0)
    # a one-row run is the MTP head, not a step: it must be dropped, not averaged in
    tiny = lay(40, 1.0, 0.1, 0.1, 1.0, 1.0, 9.9)
    ok("layers: single-row table excluded", layer_report(tiny) is None)
    ok("layers: absent -> None, not a zero", layer_report("nothing here") is None)

    print()
    print(f"{len(fails)} failed" if fails else "all selftest cases behaved")
    return 1 if fails else 0


# ------------------------------------------------------- per-layer DECPROF (GPU busy vs idle)
#
# Closes the question the kind/op tables cannot: for a given layer, was the GPU WORKING or WAITING?
#
# Field meanings, read off ggml-backend.cpp (the accumulation points are dp_lay_* / sg_*):
#   wait   = CPU blocked waiting for the previous segment's GPU work to finish
#   cb     = the top-k hook: expert slot management + BLOCKING fill
#   submit = dispatch of this segment
#   ---- total = wait + cb + submit. Verified against the step line, which prints all three and a
#        total that they reproduce to 0.1 ms. This is the ONLY additive decomposition here.
#   gpu    = sum over the segment's buffers of their GPU busy time -- OVERLAPPING buffers are
#            counted twice, so it may exceed `union`. Never add it to anything.
#   union  = the GPU SPAN of this segment (first start .. last end). Idle INSIDE the span is
#            included in it, so union is a span, not a busy time.
#   gap    = GPU idle between the previous segment's GPU end and this segment's GPU start.
#            ggml-backend.cpp:2130 states where that window lives: "This window sits inside the
#            previous segment's hook+submit (CPU) time, so gap vs (cb+submit) is a built-in
#            cross-check on both instruments."  =>  gap/(cb+submit) ~ 1 means the hook window is
#            time the GPU spends doing nothing.
#
# The family split below is not cosmetic: the qwen35 trunk interleaves full attention every 4th
# layer from 3, and the OTHER 30 carry the GDN mixer, which is the family that owns `cache_r_l*`
# (build_rwkv_token_shift_store, llama-graph.cpp:4077). So "GDN vs full-attn" IS the
# "does the cache_r copy show up as idle" question, at the granularity the engine can answer.
FULL_ATTN = frozenset(range(3, 40, 4))

LAYRE = re.compile(r"CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
                   r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")
STEPRE = re.compile(r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
                    r"wait=([\d.]+) \(\d+%\) cb=([\d.]+) \(\d+%\) submit=([\d.]+) \(\d+%\) "
                    r"ntok=(\d+) \| layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+)")


def parse_layer_tables(text):
    """Per-layer rows grouped into steps by contiguous L index.

    Contiguity is the only key available: these rows carry no step id. A one-row run is the MTP
    head's own step, not a trunk step, which is why a row-count floor is applied later rather than
    averaging everything together -- the same shape trap the kind and op tables each taught.
    """
    rows = []
    for ln in text.splitlines():
        m = LAYRE.search(ln)
        if m:
            rows.append({"il": int(m.group(1)), "wait": float(m.group(2)), "cb": float(m.group(3)),
                         "submit": float(m.group(4)), "gpu": float(m.group(5)),
                         "union": float(m.group(6)), "gap": float(m.group(7))})
    tables, cur = [], []
    for r in rows:
        if cur and r["il"] != cur[-1]["il"] + 1:
            tables.append(cur)
            cur = []
        cur.append(r)
    if cur:
        tables.append(cur)
    return tables


def parse_step_rows(text):
    out = []
    for ln in text.splitlines():
        m = STEPRE.search(ln)
        if m:
            out.append({"step": int(m.group(1)), "layers": int(m.group(3)), "total": float(m.group(4)),
                        "wait": float(m.group(5)), "cb": float(m.group(6)), "submit": float(m.group(7)),
                        "ntok": int(m.group(8)), "gpu_sum": float(m.group(9)),
                        "union_sum": float(m.group(10)), "gap_sum": float(m.group(11))})
    return out


def _corr(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = (sum((x - ma) ** 2 for x in a)) ** 0.5
    db = (sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / (da * db) if da * db else None


def layer_report(text, min_rows=40):
    """None when absent -- a log without per-layer rows means the instrument was off."""
    tables = [t for t in parse_layer_tables(text) if len(t) >= min_rows]
    if not tables:
        return None
    steps = [t for t in tables if st.median([r["union"] for r in t]) < 20.0]
    # Steps whose pool fill dominates are cold-pool steps; excluding them is what makes the rest
    # comparable to a served request. The threshold is on the ratio, not on a raw ms, so it travels.
    cold = sum(1 for t in steps if sum(r["cb"] for r in t) > 0.5 * sum(r["wait"] for r in t))
    out = {"n_tables": len(tables), "n_steps": len(steps), "n_cold_like": cold, "families": {},
           "sums": {}, "ratios": {}, "n_step_rows": len(parse_step_rows(text))}
    within_corr = None
    for name, is_attn in (("gdn", False), ("full_attn", True)):
        v = [r for t in steps for r in t
             if r["il"] < 40 and ((r["il"] in FULL_ATTN) == is_attn)]
        if not v:
            continue
        med = {k: st.median([r[k] for r in v]) for k in ("wait", "cb", "submit", "gpu", "union", "gap")}
        med["n_layers"] = len({r["il"] for r in v})
        med["n_rows"] = len(v)
        med["gpu_over_union"] = med["gpu"] / med["union"] if med["union"] else None
        med["gap_share"] = med["gap"] / (med["gap"] + med["union"]) if (med["gap"] or med["union"]) else None
        out["families"][name] = med
    per_step = []
    for t in steps:
        b = [r for r in t if r["il"] < 40]
        per_step.append({"wait": sum(r["wait"] for r in b), "cb": sum(r["cb"] for r in b),
                         "submit": sum(r["submit"] for r in b), "gap": sum(r["gap"] for r in b),
                         "union": sum(r["union"] for r in b), "gpu": sum(r["gpu"] for r in b)})
    # Per-LAYER profile. Both questions that matter are answered here and nowhere else in this
    # instrument: WHICH layers carry the hook window, and whether it tracks that layer's own GPU
    # work (a per-layer sync would; bookkeeping driven by slot churn would not).
    by_il = {}
    for t in steps:
        for r in t:
            if r["il"] < 40:
                by_il.setdefault(r["il"], []).append(r)
    out["per_layer"] = {}
    for il, v in sorted(by_il.items()):
        out["per_layer"][il] = {"cb": st.median([r["cb"] for r in v]),
                                "union": st.median([r["union"] for r in v]),
                                "gap": st.median([r["gap"] for r in v]), "n": len(v)}
    if out["per_layer"]:
        tot = sum(d["cb"] for d in out["per_layer"].values())
        rank = sorted(out["per_layer"].items(), key=lambda kv: -kv[1]["cb"])
        out["cb_concentration"] = {"total_median_per_layer_ms": tot,
                                   "top4_share": sum(d["cb"] for _, d in rank[:4]) / tot if tot else None,
                                   "top8_share": sum(d["cb"] for _, d in rank[:8]) / tot if tot else None,
                                   "top_layers": [il for il, _ in rank[:8]]}
        # within-step correlation across layers, which removes the step-to-step scale
        within = []
        for t in steps:
            b = [r for r in t if r["il"] < 40]
            if len(b) >= 40:
                c = _corr([r["cb"] for r in b], [r["union"] for r in b])
                if c is not None:
                    within.append(c)
        within_corr = st.median(within) if within else None
    if per_step:
        med = {k: st.median([p[k] for p in per_step]) for k in per_step[0]}
        wall = med["wait"] + med["cb"] + med["submit"]
        out["sums"] = med
        out["sums"]["wall"] = wall
        hook = med["cb"] + med["submit"]
        out["ratios"] = {
            "gap_over_hook": (med["gap"] / hook) if hook else None,
            "gap_over_wall": (med["gap"] / wall) if wall else None,
            "hook_over_wall": (hook / wall) if wall else None,
            "gpu_over_union": (med["gpu"] / med["union"]) if med["union"] else None,
            # the engine's own cross-check, across steps rather than within one
            "corr_gap_hook": _corr([p["gap"] for p in per_step],
                                   [p["cb"] + p["submit"] for p in per_step]),
            "corr_gap_wait": _corr([p["gap"] for p in per_step], [p["wait"] for p in per_step]),
        }
        # set AFTER the literal above: assigning into out["ratios"] earlier is silently lost when
        # this dict is built (it rebinds the name). Cost of finding that out: one KeyError.
        out["ratios"]["within_step_corr_cb_union"] = within_corr
    return out


def layer_verdict(a):
    r = a.get("ratios") or {}
    if not r.get("gap_over_hook"):
        return "NO DATA"
    fam = a["families"]
    g, f = fam.get("gdn"), fam.get("full_attn")
    msg = []
    if g and f:
        if g["gpu_over_union"] >= 1.0 and f["gpu_over_union"] >= 1.0:
            msg.append("GPU is packed inside both families' spans (gpu/union >= 1) -> per-layer idle "
                       "is NOT the story")
        if abs(g["gap"] - f["gap"]) / max(g["gap"], f["gap"], 1e-9) < 0.25:
            msg.append("the cache_r-bearing (GDN) layers carry the SAME per-layer gap as full-attn "
                       "(%.2f vs %.2f ms) -> the copy does not present as idle" % (g["gap"], f["gap"]))
    msg.append("step-level GPU idle = %.0f%% of the wall, and it tracks the CPU hook window 1:1 "
               "(gap/hook=%.2f, corr=%.3f)" % (100 * r["gap_over_wall"], r["gap_over_hook"],
                                               r["corr_gap_hook"] or float("nan")))
    return "; ".join(msg)


def cmd_layers(args):
    text = open(args.log, errors="replace").read()
    a = layer_report(text)
    if a is None:
        print("no per-layer CGC-DECPROF rows in this log (instrument off, or CGC_DECODE_PROFILE unset)")
        return 1
    print("=" * 78)
    print("PER-LAYER: was the GPU working, or waiting?  (medians; ms)")
    print("=" * 78)
    print("log                 : %s" % args.log)
    print("trunk tables        : %d   (steps used: %d, of which cold-looking: %d)"
          % (a["n_tables"], a["n_steps"], a["n_cold_like"]))
    print("step summary rows   : %d  (used to cross-check the sums below)" % a["n_step_rows"])
    print()
    print("%-11s %5s %6s %6s %7s %7s %7s %7s %9s %9s" %
          ("family", "nLay", "wait", "cb", "submit", "gpu", "union", "gap", "gpu/uni", "gap share"))
    print("-" * 78)
    for name in ("gdn", "full_attn"):
        d = a["families"].get(name)
        if not d:
            continue
        print("%-11s %5d %6.2f %6.2f %7.2f %7.2f %7.2f %7.2f %9.2f %8.0f%%" %
              (name, d["n_layers"], d["wait"], d["cb"], d["submit"], d["gpu"], d["union"], d["gap"],
               d["gpu_over_union"], 100 * d["gap_share"]))
    s = a["sums"]
    r = a["ratios"]
    print()
    print("per-step sums (40 trunk layers, medians):")
    print("  wall = wait + cb + submit      = %8.1f ms  (%.1f + %.1f + %.1f)"
          % (s["wall"], s["wait"], s["cb"], s["submit"]))
    print("  hook window = cb + submit      = %8.1f ms  = %.1f%% of wall"
          % (s["cb"] + s["submit"], 100 * r["hook_over_wall"]))
    print("  gap (GPU idle, directly read)  = %8.1f ms  = %.1f%% of wall" % (s["gap"], 100 * r["gap_over_wall"]))
    print("  gpu / union                    = %8.2f     (overlap double counts; >1 hides span idle)"
          % r["gpu_over_union"])
    print("  gap / hook                     = %8.2f     <- the engine's own cross-check" % r["gap_over_hook"])
    print("  corr(gap, cb+submit)           = %8.3f     corr(gap, wait) = %.3f"
          % (r["corr_gap_hook"], r["corr_gap_wait"]))
    pl = a.get("per_layer") or {}
    if pl:
        cc = a["cb_concentration"]
        print()
        print("per-layer cb profile (median over steps, ms) -- WHERE the hook window lives:")
        for lo in (0, 20):
            hi = min(lo + 20, 40)
            print("  L%-2d-" % lo + "%-2d  " % (hi - 1) +
                  " ".join("%5.2f" % pl[i]["cb"] for i in range(lo, hi) if i in pl))
        print("  top 4 share of cb = %.0f%%   top 8 share = %.0f%%   (layers %s)"
              % (100 * cc["top4_share"], 100 * cc["top8_share"],
                 ",".join("L%d" % i for i in cc["top_layers"])))
        print("  within-step corr(cb, union) across layers = %s"
              % ("%.3f" % a["ratios"].get("within_step_corr_cb_union")
                 if a["ratios"].get("within_step_corr_cb_union") is not None else "n/a"))
        print("  reading: a flat ~0.2 ms floor with a few big layers = bookkeeping driven by those")
        print("  layers' churn. If instead cb tracked union, it would be a per-layer GPU sync.")
    print()
    print("VERDICT: %s" % layer_verdict(a))
    print()
    print("Read it this way: `union` is a SPAN, so idle inside it is invisible here; `gap` is the")
    print("idle BETWEEN segments and is the only idle this table can name. `gpu` double counts.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", default="prod25")
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--decode-predict", type=int, default=128)
    r.add_argument("--port", default="auto")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)
    o = sub.add_parser("ops")
    o.add_argument("log")
    o.set_defaults(fn=cmd_ops)
    a = sub.add_parser("analyze")
    a.add_argument("log")
    a.set_defaults(fn=cmd_analyze)
    l = sub.add_parser("layers")
    l.add_argument("log")
    l.set_defaults(fn=cmd_layers)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
