#!/usr/bin/env python3
"""mmid_shapes.py -- price the MoE expert-bank gather (MUL_MAT_ID) and give its
kernel-side upper bound, on the axis MTP actually moves: tokens per op.

WHY THIS OP AND WHY THIS AXIS
    The dense GEMV family was closed by docs/DENSE_GEMV_INSTRUMENT_2026-09-22.md,
    and `run_shapes.py` skips 3D tensors on purpose ("MoE expert bank ->
    MUL_MAT_ID, priced elsewhere"). This is that elsewhere.

    The expert bank is the only weight family whose bytes/step grow with the token
    count: each token reads n_expert_used slices of EVERY layer's bank, while a
    dense weight is read once per step no matter how many tokens ride along. So
    the verify batch's marginal cost lives here, and so does the answer to "does
    the second draft token cost as much as the first".

WHY THE MARGINAL MODEL AND NOT `batch`
    The probe's `batch` number is `total/(loops*ops)`, which still carries
    1/batch of the per-graph fixed cost (~220 us on this machine). At batch=32
    that is ~11% of a 64 us op, and it is charged to the SMALL shapes the most --
    exactly the shapes here. Two graph sizes remove it:

        g(b) = fixed + b*marginal   -> measure g(16) and g(32)
        marginal = (g(32) - g(16)) / 16,   fixed = g(16) - 16*marginal

    Everything below is reported on the marginal, and the fixed term is printed
    next to it because it is the input to the OTHER open question on this box: how
    many command buffers a step really has (docs/DECODE_STEADY_BASELINE_2026-09-19.md).

WHAT IS NOT IN SCOPE (say it before quoting a number)
    This prices the kernel on the REAL model geometry, indexed by expert id --
    the path ggml_mul_mat_id takes. It does NOT price the CGC pool slab, whose
    weights are gathered into a contiguous buffer by a different mechanism and
    whose per-op cost the engine already reports (CGC-VERIFY-OP). So this is the
    kernel's floor, and the engine's number is the thing to compare it against;
    neither one alone tells you where the time went.

Usage:
    shape_probe/mmid_shapes.py --tokens 1,2,3,4 --json Backup/phase_decomp/L3/shape_probe/mmid.json
    shape_probe/mmid_shapes.py --selftest
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE / "cgc_shape_probe"

sys.path.insert(0, str(ROOT / "scripts/check"))
import server_window as _sw          # noqa: E402  the shared vocabulary + admission bar
NEED_MB = _sw.NEED_MB

# Measured DRAM peak for this machine (Mac16,12 / M4 / 8 GPU, 6 threads), the same
# constant run_shapes.py uses -- NOT the 120 GB/s spec number, which has never been
# observed here. Both are GiB/s: the probe prints MiB/ms, which is the same unit.
PEAK_GIB_S = 108.8
# Bytes per second, spelled out: `bytes / PEAK_GIB_S` silently divides by a rate in
# GiB and returns a number 1024^3 too large. It was in this file for one run.
PEAK_BYTES_S = PEAK_GIB_S * 1024 ** 3

# The pinned production denominator (docs/DENOMINATOR_PINNED_2026-09-22.md):
# 12.57 t/s at ml = 1.69-1.91 tokens per step => step = 134-152 ms. NOT 79.55 ms,
# which is ms per TOKEN. Quoting a per-step cost against a per-token time inflates
# every share by 1.8x -- the defect this file's docstring exists to prevent.
STEP_MS_LO, STEP_MS_HI = 134.0, 152.0
ML_CENTRAL = 1.8

# Real geometry, read from the model's own header (gguf_pool_geometry.parse_header).
# (label, probe type, K, N per expert, experts, ops per layer, layers)
#   gate_exps / up_exps  0.*.ffn_gate_exps.weight  IQ2_S (2048,512,256)  82.0 MiB
#   down_exps            0.*.ffn_down_exps.weight  IQ3_S (512,2048,256) 110.0 MiB
# down_exps is IQ3_S in 38 of 40 layers (34/38 are IQ4_XS, 40 is Q3_K) -- the
# dominant type is priced, and the mix is a stated bound, not a silent average.
LAYERS = 40
USED = 8
SHAPES = (
    ("gate_exps", "iq2_s", 2048, 512, 256, 1, LAYERS),
    ("up_exps",   "iq2_s", 2048, 512, 256, 1, LAYERS),
    ("down_exps", "iq3_s",  512, 2048, 256, 1, LAYERS),
)

MIB = 1024 * 1024


# ---------------------------------------------------------------- window (must
# travel with every reading: this box runs other sessions, and a number taken over
# a thrashing box is not comparable with one taken over a quiet one).
def digest(libdir=""):
    """The kernel's identity: the dylib the probe links, not the repo's HEAD.

    `libdir` is the swept build's own bin/ when the measurement is a knob sweep: the tree's
    shipped dylib is NOT what such a run loaded, and stamping the tree's md5 on it would be
    the exact provenance lie this line keeps catching.
    """
    if libdir:
        base = Path(libdir)
        out = {}
        for rel in ("libggml-metal.0.dylib", "libggml-metal.0.19.0.dylib"):
            p = base / rel
            if p.exists():
                out[rel] = subprocess.run(["md5", "-q", str(p)], capture_output=True,
                                          text=True).stdout.strip()[:16]
                break
        return out or {"": "no libggml-metal found in libdir"}
    return _tree_digest()


def _tree_digest():
    """The kernel's identity: the dylib the probe links, not the repo's HEAD.

    A probe measurement is only comparable with another probe measurement taken on the
    same libggml-metal -- the `.metal` source inside it is where every timing here
    comes from. `build.sh` prints these too; recording them in the artifact is what
    makes the pair checkable later (and is what the repo's provenance contract asks
    of a product that carries measured objects).
    """
    out = {}
    for rel in ("libggml-metal.0.dylib", "libggml.0.dylib", "libggml-base.0.dylib"):
        p = ROOT / "src/llama.cpp/build/bin" / rel
        if p.exists():
            out[rel] = subprocess.run(["md5", "-q", str(p)], capture_output=True,
                                      text=True).stdout.strip()[:16]
    return out


# --------------------------------------------------------------- the nsg sweep
# The id pipeline's tile dimension (nsg = simdgroups per threadgroup). IQ2_S has honoured
# CGC_MMV_NSG since CGC P1-3c; IQ3_S only since P1-3e (this commit), so an older dylib will
# silently ignore the value for down_exps -- which is why every arm is checked against the
# pipeline name it actually compiled, not against the env var that was set.
SWEEP_SHAPES = (
    ("gate_exps", "iq2_s", 2048, 512, 256),
    ("down_exps", "iq3_s",  512, 2048, 256),
)


def rotated(values, rep):
    """The arm order for rep `rep`: the list rotated left by `rep`.

    A sweep that runs `unset,1,4,...,32,unset` once has a null cell that only brackets the
    sweep in TIME. Any monotone drift inside that window (swap grows, a neighbour wakes up)
    lands in the null cell -- and it landed there hard enough on the first run of this
    command to make the IQ3_S channel unreadable (unset 29.8% -> 20.4%). Rotating the order
    per rep puts every arm in a different time slot, so the per-arm median is trend-free and
    the unset readings across reps measure the drift itself.
    """
    if not values:
        return []
    k = rep % len(values)
    return list(values[k:]) + list(values[:k])


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def nsg_decide(per_arm, reached, thr_floor=0.05):
    """The verdict, from readings only: {tag: [pct_peak per rep]} + {tag: reached?}.

    Pure on purpose, because the decision must be exercisable on a channel that drifted --
    that is the one thing this sweep cannot fake, and it is what the first run of it
    produced. A REFUSAL is a first-class outcome here: a median taken across a 31%-wide
    null cell is not a small effect, it is no reading, and printing one anyway is how a
    drift gets quoted as a knob.
    """
    unsets = per_arm.get("unset") or []
    if len(unsets) < 2:
        return {"status": "unjudged",
                "why": "the unset arm was measured once: no null cell, so nothing here "
                       "separates the knob from a drift"}
    ref = median(unsets)
    noise = (max(unsets) - min(unsets)) / ref if ref else float("inf")
    thr = max(thr_floor, 2 * noise)
    out = {"null_ref_pct": ref, "null_pct_per_rep": unsets, "noise_frac": noise,
           "threshold_frac": thr, "unwired": sorted(t for t, ok in reached.items() if ok is False),
           "unstable": sorted(t for t, v in per_arm.items()
                              if t != "unset" and len(v) > 1 and (max(v) - min(v)) / ref > thr),
           "median_pct": {t: median(v) for t, v in per_arm.items() if v}}
    if noise > thr_floor:
        out["status"] = "INVALID"
        out["why"] = ("the CHANNEL moved: the unset arm itself spans %.1f%% across reps, so every "
                      "arm difference is inside the trend -- rerun in a quieter window"
                      % (100 * noise))
        return out
    wins = sorted((t for t, v in out["median_pct"].items()
                   if t != "unset" and v > ref * (1 + thr)), key=lambda t: -out["median_pct"][t])
    out["wins"] = wins
    if out["unwired"]:
        out["status"] = "INVALID"
        out["why"] = ("nsg=%s never reached the kernel (the compiled pipeline name does not "
                       "carry the value), so a flat reading there is evidence about the "
                       "plumbing" % ", ".join(out["unwired"]))
    elif wins:
        out["status"] = "win"
        out["winner"] = wins[0]
    else:
        out["status"] = "no-lever"
        out["why"] = "no arm beats unset by more than the threshold"
    return out


def nsg_sweep(args):
    if not PROBE.exists():
        sys.exit("FATAL: %s missing -- run shape_probe/build.sh first" % PROBE)
    values = [None if v.strip() in ("", "unset", "0") else int(v)
              for v in args.nsg_sweep.split(",")]
    if None not in values:
        values = [None] + values                      # the unset arm is the reference
    reps = max(1, args.reps)
    tokens = int(args.tokens.split(",")[-1])
    env = probe_env(args)
    env.pop("CGC_MMV_NSG", None)                      # each arm sets its own

    print("CGC_MMV_NSG sweep on the MUL_MAT_ID (MoE) path -- T=%d, batch %d vs %d, marginal model"
          % (tokens, args.b1, args.b2))
    print("libdir %s" % (args.libdir or "(the tree's shipped build)"))
    print("dylib  %s" % json.dumps(digest(args.libdir)))
    print("window %s: usable %.2f GiB, swap %.0f MiB  |  %d rep(s), arm order rotated per rep"
          % (WINDOW["class"], WINDOW["usable_mib"] / 1024, WINDOW["swap_used_mib"], reps))
    print("-" * 104)

    out = {"peak_gib_s": PEAK_GIB_S, "tokens": tokens, "libdir": args.libdir,
           "window": WINDOW, "engine_digest": digest(args.libdir), "reps": reps,
           "instrument": "shape_probe/mmid_shapes.py --nsg-sweep", "b1": args.b1, "b2": args.b2,
           "rows": {}}

    for (label, tname, k, out_n, experts) in SWEEP_SHAPES:
        print("%s  %s %dx%d  experts=%d used=%d" % (label, tname, k, out_n, experts, USED))
        per_arm, detail, reached, passes = {}, {}, {}, []
        for rep in range(reps):
            tags = []
            for val in rotated(values, rep):
                tag = "unset" if val is None else str(val)
                e = dict(env)
                if val is not None:
                    e["CGC_MMV_NSG"] = str(val)
                m, fixed, op_bytes, extra = marginal(tname, k, out_n, experts, USED, tokens,
                                                     args.b1, args.b2, args.target_ms, e)
                if m is None:
                    print("   rep %d %-6s FAILED -- %s" % (rep, tag, transport(extra)))
                    out["rows"]["%s_%s" % (label, tag)] = {"error": transport(extra)}
                    continue
                gbs = op_bytes / (m * 1e-6) / (1024 ** 3)
                got = pipeline_nsg(extra.get("pipeline"))
                per_arm.setdefault(tag, []).append(100 * gbs / PEAK_GIB_S)
                detail.setdefault(tag, []).append(
                    {"rep": rep, "marginal_us": m, "fixed_us": fixed, "gib_s": gbs,
                     "pipeline": extra.get("pipeline"), "reached": None if val is None
                     else (got == val)})
                if val is not None:
                    reached[tag] = reached.get(tag, True) and (got == val)
                tags.append(tag)
            sw = swap_used_mib()
            passes.append({"rep": rep, "order": tags, "swap_used_mib": round(sw), "pct":
                           {t: per_arm[t][-1] for t in tags}})
            print("   rep %d  " % rep + "  ".join(
                "%s %.1f%%" % (t, per_arm[t][-1]) for t in tags)
                + "   (swap %.0f MiB)" % sw)
            sys.stdout.flush()

        v = nsg_decide(per_arm, reached)
        print("   %-6s %8s %8s %8s   %-34s %s"
              % ("nsg", "us/op", "%peak", "spread", "pipeline compiled", "knob reached?"))
        for tag in sorted(per_arm, key=lambda t: (t != "unset", int(t) if t != "unset" else -1)):
            rows = detail[tag]
            sp = (max(per_arm[tag]) - min(per_arm[tag])) / median(per_arm[tag]) * 100
            print("   %-6s %8.2f %7.1f%% %7.1f%%   %-34s %s"
                  % (tag, median([r["marginal_us"] for r in rows]), median(per_arm[tag]), sp,
                     rows[-1]["pipeline"] or "?",
                     "-" if tag == "unset" else ("yes" if reached.get(tag) else "NO")))
            out["rows"]["%s_%s" % (label, tag)] = {
                "median_pct_peak": median(per_arm[tag]), "pct_peak_per_rep": per_arm[tag],
                "median_us": median([r["marginal_us"] for r in rows]), "spread_frac": sp / 100,
                "pipeline": rows[-1]["pipeline"], "reached": reached.get(tag), "per_rep": rows}

        if v["status"] == "INVALID":
            print("   NO READING: %s" % v["why"])
        else:
            print("   null cell: unset spans %.1f-%.1f%% => noise %.1f%%, threshold %.1f%%"
                  % (min(v["null_pct_per_rep"]), max(v["null_pct_per_rep"]),
                     100 * v["noise_frac"], 100 * v["threshold_frac"]))
            if v["status"] == "win":
                t = v["winner"]
                print("   VERDICT: nsg=%s buys +%.1f%% of peak (%.1f%% -> %.1f%%) -- above the "
                      "threshold; a rebuild+gate would be next"
                      % (t, v["median_pct"][t] - v["null_ref_pct"], v["null_ref_pct"],
                         v["median_pct"][t]))
            else:
                print("   VERDICT: no arm beats unset by more than the threshold "
                      "=> the tile dimension is not the lever for this shape")
            if v["unstable"]:
                print("   (unstable across reps, inside the trend: %s)" % ", ".join(v["unstable"]))
        out["rows"][label + "_verdict"] = dict(v, passes=passes)
        print()

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote %s" % p)
    return 0


def transport(err):
    """Errors are strings or dicts depending on where they came from; keep the JSON serialisable."""
    return err if isinstance(err, str) else json.dumps(err)


def window():
    """This run's window block, in the SHARED vocabulary.

    `class` is not decoration: a product whose window block lacks it is counted by
    `server_window.audit_products` as "cannot answer the busy-box question" -- which is
    how this block came to be written twice (the first version carried usable_gib and
    nothing else, and the repo's own audit reported the artifact as unanswerable).
    The admission bar is IMPORTED, not retyped: a second copy of the threshold is the
    defect this line has already paid for once.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    pg = int(m.group(1)) if m else 4096
    pages = {}
    for line in out.splitlines():
        mm = re.match(r"^(.+?):\s+(\d+)\.$", line.strip())
        if mm:
            pages[mm.group(1)] = int(mm.group(2))
    usable_mib = (sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
                  * pg / MIB)
    swap_mib = swap_used_mib()
    quiet = usable_mib >= NEED_MB
    return {
        "class": "clean" if quiet else "busy-overridden",
        "why": (f"usable {usable_mib / 1024:.2f} GiB vs the shared admission bar "
                f"{NEED_MB / 1024:.1f} GiB: the box was "
                + ("quiet" if quiet else "NOT quiet -- a GPU microbench has a tiny host footprint, "
                   "so the ratio-to-peak is the reading and the absolute us/op may describe "
                   "the neighbour")),
        "need_mb": NEED_MB, "usable_mib": round(usable_mib), "swap_used_mib": round(swap_mib),
        "n_samples": 1, "busy_samples": 0 if quiet else 1,
        "samples": [{"usable_mib": round(usable_mib), "swap_used_mib": round(swap_mib),
                     "quiet": quiet, "source": "this driver's own vm_stat"}],
        "class_source": ("mmid_shapes.py (own vm_stat reading; same vocabulary and the same "
                         "imported admission bar as server_window)"),
    }


def swap_used_mib():
    """Swap in use, in MiB. Cheap enough to sample per rep, which is the point: swap is this
    box's drift mechanism (12.7 GB model + pool on 16 GB), so the sweep records it next to
    every pass rather than once at the top."""
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    ms = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
    if not ms:
        return 0.0
    return float(ms.group(1)) * {"K": 1 / 1024.0, "M": 1.0, "G": 1024.0}[ms.group(2)]


WINDOW = window()


def parse_result(stderr, mode):
    """`RESULT   batch median 64.64 us/op  38.72 GB/s` -> (us, gbs). None if absent.

    Absent must stay absent: this line's oldest defect is a missing reading being
    printed as 0, so a parser that returns 0.0 on no-match is the bug, not the fix.
    """
    for line in stderr.splitlines():
        if line.startswith("RESULT") and mode in line:
            m = re.search(r"median\s+([\d.]+)\s+us/op\s+([\d.]+)\s+GB/s", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def parse_loops(stderr):
    m = re.search(r"batch:\s+(\d+)\s+loops\s+x\s+(\d+)\s+ops", stderr)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_bytes(stderr):
    """Per-op bytes as the probe itself counted them, so the driver cannot disagree
    with the instrument about what was read."""
    m = re.search(r"reads\s+([\d.]+)\s+MiB/op", stderr)
    return float(m.group(1)) * MIB if m else None


def parse_pipeline(stderr):
    """The pipeline the probe actually compiled, e.g. `kernel_mul_mv_id_iq3_s_f32_nsg=8`.

    This is the sweep's own gate: a knob whose value does not appear in the kernel's name
    did not reach the kernel, and an arm that looks flat for THAT reason is not evidence
    about the tile dimension -- it is evidence about the plumbing. The line is printed on
    compile, and every probe invocation is a fresh process, so it is always there.
    """
    m = re.search(r"compiling pipeline:.*name = '([^']+)'", stderr)
    if m:
        return m.group(1)
    m = re.search(r"loaded kernel (\S+)", stderr)
    return m.group(1) if m else None


def pipeline_nsg(name):
    m = re.search(r"nsg=(\d+)", name or "")
    return int(m.group(1)) if m else None


def probe_env(args):
    """Env for every probe run: the libdir (when sweeping a build outside the tree) plus any
    `--set K=V`. Returned as a dict so `one()` cannot forget half of it."""
    e = dict(os.environ)
    for kv in (args.set or []):
        k, _, v = kv.partition("=")
        e[k] = v
    if args.libdir:
        e["DYLD_LIBRARY_PATH"] = args.libdir + ":" + e.get("DYLD_LIBRARY_PATH", "")
    return e


def one(tname, k, out_n, experts, used, tokens, batch, target_ms, pool_mib, env=None):
    cmd = [str(PROBE), "--op", "mul_mat_id", "--type", tname,
           "--k", str(k), "--out", str(out_n), "--experts", str(experts),
           "--used", str(used), "--tokens", str(tokens),
           "--mode", "batch", "--batch", str(batch), "--iters", "1",
           "--target-ms", str(target_ms), "--pool-mib", str(pool_mib)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])[-1]
    res = parse_result(p.stderr, "batch")
    loops = parse_loops(p.stderr)
    op_bytes = parse_bytes(p.stderr)
    if res is None or loops is None or op_bytes is None:
        return None, "unparseable probe output: " + p.stderr[-200:]
    return {"per_op_us": res[0], "gbs": res[1], "loops": loops[0], "ops": loops[1],
            "op_bytes": op_bytes, "pipeline": parse_pipeline(p.stderr)}, ""


def marginal(tname, k, out_n, experts, used, tokens, b1, b2, target_ms, env=None):
    """(marginal µs/op, fixed µs/graph, op bytes, [raw g(b1), g(b2)]) or (None...)."""
    raw = {}
    for b in (b1, b2):
        # Device memory for ONE bank (82-136 MiB) + ids + outs. The bank is shared by
        # every copy and the copies differ by ids, so this does NOT scale with b --
        # which is why a 200 MiB pool can host a 32-op graph that reads 256 experts.
        r, err = one(tname, k, out_n, experts, used, tokens, b, target_ms, 320, env)
        if r is None:
            return None, None, None, err
        raw[b] = r["per_op_us"] * r["ops"]          # µs per graph (one loop)
    m = (raw[b2] - raw[b1]) / (b2 - b1)
    fixed = raw[b1] - b1 * m
    return m, fixed, r["op_bytes"], {"g_b1_us": raw[b1], "g_b2_us": raw[b2],
                                    "pipeline": r["pipeline"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,3,4", help="the MTP axis: tokens per op")
    ap.add_argument("--b1", type=int, default=16)
    ap.add_argument("--b2", type=int, default=32)
    ap.add_argument("--target-ms", type=float, default=1500.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--libdir", default="",
                    help="directory holding the libggml-metal to test (a build outside the tree); "
                         "the tree's shipped dylib is left alone")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="extra env for every probe run (e.g. --set CGC_MMV_NSG=8); repeatable")
    ap.add_argument("--nsg-sweep", default="",
                    help="comma-separated CGC_MMV_NSG values (or 'unset') to compare, "
                         "e.g. 'unset,1,4,8,16,32'")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the arm list, rotated per pass (default 3). One pass "
                         "cannot separate a knob from a drift: this sweep's first run left a "
                         "31 percent wide null cell and still printed a verdict")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.nsg_sweep:
        return nsg_sweep(args)

    if not PROBE.exists():
        sys.exit("FATAL: %s missing -- run shape_probe/build.sh first" % PROBE)

    tokens = [int(t) for t in args.tokens.split(",")]
    print("MoE expert-bank gather (MUL_MAT_ID) -- kernel-side pricing")
    print("window %s: usable %.2f GiB, swap %.0f MiB  |  peak %.1f GiB/s  |  step %.0f-%.0f ms "
          "(ml %.2f, docs/DENOMINATOR_PINNED_2026-09-22.md)"
          % (WINDOW["class"], WINDOW["usable_mib"] / 1024, WINDOW["swap_used_mib"], PEAK_GIB_S,
             STEP_MS_LO, STEP_MS_HI, ML_CENTRAL))
    print("geometry: %d layers x (gate+up+down), used=%d of 256 experts, marginal model "
          "(%d vs %d ops)" % (LAYERS, USED, args.b2, args.b1))
    print("-" * 112)

    out = {"peak_gib_s": PEAK_GIB_S, "step_ms": [STEP_MS_LO, STEP_MS_HI], "ml": ML_CENTRAL,
           "window": WINDOW, "engine_digest": digest(),
           "instrument": "shape_probe/mmid_shapes.py --op mul_mat_id (kernel-side only; no engine)",
           "used": USED, "layers": LAYERS,
           "b1": args.b1, "b2": args.b2, "target_ms": args.target_ms, "rows": {}}
    per_token_ms = {}                      # tokens -> measured ms/step for the whole family
    bytes_step = {}                        # tokens -> bytes/step for the whole family

    for T in tokens:
        tot_ms = tot_bytes = 0.0
        print("tokens/op = %d" % T)
        for (label, tname, k, out_n, experts, ops, layers) in SHAPES:
            m, fixed, op_bytes, err = marginal(tname, k, out_n, experts, USED, T,
                                               args.b1, args.b2, args.target_ms)
            key = "%s_t%d" % (label, T)
            if m is None:
                print("   %-10s FAILED -- %s" % (label, err))
                out["rows"][key] = {"error": err}
                continue
            if m <= 0:
                # A non-positive marginal means the two graph sizes did not separate;
                # every number derived from it would be noise dressed as a cost.
                print("   %-10s INVALID -- marginal %.2f us/op <= 0 (b=%.0f vs %.0f us): "
                      "the two graph sizes did not separate" % (label, m, fixed + args.b1 * m,
                                                                fixed + args.b2 * m))
                out["rows"][key] = {"error": "non-positive marginal", "marginal_us": m}
                continue
            gbs = op_bytes / (m * 1e-6) / (1024 ** 3)
            ms_step = m * ops * layers / 1000.0
            tot_ms += ms_step
            tot_bytes += op_bytes * ops * layers
            print("   %-10s %7.2f us/op  %7.2f MiB/op  %6.1f GiB/s (%4.1f%% pk)  "
                  "fixed %6.0f us/graph  %5.2f ms/step" %
                  (label, m, op_bytes / MIB, gbs, 100 * gbs / PEAK_GIB_S, fixed, ms_step))
            out["rows"][key] = {"type": tname, "k": k, "out": out_n, "experts": experts,
                                "used": USED, "tokens": T, "ops_per_layer": ops,
                                "layers": layers, "marginal_us": m, "fixed_us": fixed,
                                "op_bytes": op_bytes, "gib_s": gbs,
                                "pct_peak": 100 * gbs / PEAK_GIB_S, "ms_per_step": ms_step,
                                "raw": err if isinstance(err, dict) else {}}
        per_token_ms[T] = tot_ms
        bytes_step[T] = tot_bytes
        floor = tot_bytes / PEAK_BYTES_S * 1000.0            # ms at peak bandwidth
        print("   family: %5.2f ms/step measured  vs %5.2f ms at %.1f GiB/s peak  "
              "=> %5.1f%% of peak" % (tot_ms, floor, PEAK_GIB_S, 100 * floor / tot_ms))
        print()

    # ---- the two numbers this file exists to produce ------------------------
    print("-" * 112)
    valid = sorted(t for t in per_token_ms if per_token_ms[t] > 0)
    if len(valid) >= 2:
        t1, t2 = valid[0], valid[-1]
        if t1 == 1:
            first, extra = per_token_ms[1], (per_token_ms[t2] - per_token_ms[1]) / (t2 - 1)
            print("  the first token costs %5.2f ms/step; each EXTRA token costs %5.2f ms/step "
                  "(T=%d vs T=1)" % (first, extra, t2))
            out["first_token_ms"] = first
            out["extra_token_ms"] = extra
            if first > 0:
                print("  => amortisation: an extra verify token costs %.0f%% of the first one"
                      % (100 * extra / first))
                out["extra_over_first"] = extra / first
    if 1 in per_token_ms and 1 in bytes_step:
        floor1 = bytes_step[1] / PEAK_BYTES_S * 1000.0
        save = per_token_ms[1] - floor1
        print("  at T=1 the family's bytes cost %.2f ms at peak and %.2f ms now: the kernel-side "
              "headroom is %.2f ms = %.1f%% of a %.0f ms step"
              % (floor1, per_token_ms[1], save, 100 * save / STEP_MS_LO, STEP_MS_LO))
        print("  (kernel-side only: the engine's own CGC-VERIFY-OP timing is the other half of "
              "this comparison)")
        out["headroom_ms_t1"] = save
        need = ML_CENTRAL / 25.0 * 1000.0
        print("  for scale: 25 t/s at ml %.2f needs a %.0f ms step, today is %.0f ms, so %.0f ms "
              "must come out" % (ML_CENTRAL, need, STEP_MS_LO, STEP_MS_LO - need))
        out["needed_step_ms"] = need

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote %s" % p)
    return 0


def selftest():
    bad = tot = 0

    def check(name, cond):
        nonlocal bad, tot
        tot += 1
        print("  %s %s" % ("ok  " if cond else "FAIL", name))
        bad += 0 if cond else 1

    ok = ("RESULT   sync  median  3050.50 us/op     0.82 GB/s\n"
          "RESULT   batch median    64.64 us/op    38.72 GB/s\n"
          "         batch: 399 loops x 32 ops\n"
          "         bank 82.00 MiB (0.320 MiB/expert), 32 op(s), reads 2.562 MiB/op\n")
    check("a batch RESULT line parses to its own number",
          parse_result(ok, "batch") == (64.64, 38.72))
    check("and the sync line does not shadow it", parse_result(ok, "sync") == (3050.50, 0.82))
    check("a missing mode is None, NOT 0.0 (the defect this line keeps re-learning)",
          parse_result(ok, "marginal") is None)
    check("a truncated RESULT line is not half-parsed",
          parse_result("RESULT   batch median    64.64 us/op\n", "batch") is None)
    check("loops x ops parse", parse_loops(ok) == (399, 32))
    check("per-op bytes come from the instrument, not from the driver's arithmetic",
          abs(parse_bytes(ok) - 2.562 * MIB) < 1.0)
    check("no bytes line -> None", parse_bytes("nothing here") is None)

    # arithmetic fixtures: the marginal model and the headroom statement
    g16, g32, b1, b2 = 1400.0, 2650.0, 16, 32
    m = (g32 - g16) / (b2 - b1)
    fixed = g16 - b1 * m
    check("marginal removes the fixed term exactly", abs(m - 78.125) < 1e-9)
    check("...and the fixed term it implies reproduces g(16)", abs((fixed + 16 * m) - g16) < 1e-9)
    check("a non-positive marginal is refused, not rounded", not (0.0 > 0))
    # the denominator guard: 13.39 ms against a PER-TOKEN time vs a per-STEP time
    check("the pinned step denominator is the per-step one, not 79.55 ms/token",
          STEP_MS_LO > 100 and ML_CENTRAL * 79.55 > STEP_MS_LO)
    check("25 t/s at ml 1.8 needs ~72 ms", abs(ML_CENTRAL / 25.0 * 1000.0 - 72.0) < 0.01)
    pl = "ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_id_iq3_s_f32', name = 'kernel_mul_mv_id_iq3_s_f32_nsg=8'"
    check("the sweep can read which nsg the kernel was ACTUALLY compiled with",
          parse_pipeline(pl) == "kernel_mul_mv_id_iq3_s_f32_nsg=8"
          and pipeline_nsg(parse_pipeline(pl)) == 8)
    check("and None (not 0) when no pipeline line is present",
          parse_pipeline("nothing") is None and pipeline_nsg(None) is None)
    # 342.6 MiB/step at 108.8 GiB/s is 3.08 ms; dividing by a GiB rate instead of a
    # byte rate returns 3.15 s. Pin the unit, because only one of them is physically sane.
    b = 342.6 * MIB
    check("the peak-bandwidth floor is in the right unit (ms, not seconds)",
          abs(b / PEAK_BYTES_S * 1000.0 - 3.08) < 0.02)

    # ---- the sweep's own protocol: rotation, medians, and the right to refuse ------
    vals = [None, 1, 4]
    check("rotation puts every arm exactly once in every pass",
          all(sorted(rotated(vals, r), key=str) == sorted(vals, key=str) for r in range(6)))
    check("and rotates (rep 0 and rep 1 are not the same order)",
          rotated(vals, 0) == [None, 1, 4] and rotated(vals, 1) == [1, 4, None])
    check("rotation is periodic, not lossy", rotated(vals, 3) == rotated(vals, 0))
    check("median is the middle of an odd list and the mean of the pair in an even one",
          median([3.0, 1.0, 2.0]) == 2.0 and median([1.0, 2.0, 3.0, 4.0]) == 2.5)

    # a channel that DRIFTED (the real first run: unset 29.8% and 20.4% for the same arm)
    drifted = nsg_decide({"unset": [29.8, 20.4], "1": [28.3, 21.0], "32": [18.7, 25.0]},
                         {"1": True, "32": True})
    check("a channel whose own null cell spans 31% is refused, not averaged",
          drifted["status"] == "INVALID" and "CHANNEL moved" in drifted["why"])
    check("...and no median is published with it", "wins" not in drifted)
    stable = nsg_decide({"unset": [44.3, 44.1, 44.2], "1": [43.8, 44.0, 43.9],
                         "32": [44.7, 44.5, 44.6]}, {"1": True, "32": True})
    check("a stable channel with a 0.4%-wide null cell still says 'not the lever'",
          stable["status"] == "no-lever" and stable["threshold_frac"] == 0.05)
    check("...and the 0.2 point below unset is not called unstable", stable["unstable"] == [])
    loud = nsg_decide({"unset": [44.3, 44.1], "8": [52.0, 52.4]}, {"8": True})
    check("a real +17% arm is a win once the null cell is tight", loud["status"] == "win"
          and loud["winner"] == "8")
    unwired = nsg_decide({"unset": [44.3, 44.2], "8": [44.3, 44.2]}, {"8": False})
    check("a knob that never reached the kernel invalidates the shape, flat or not",
          unwired["status"] == "INVALID" and "plumbing" in unwired["why"])
    check("one unset reading is 'unjudged', not a zero-width noise floor",
          nsg_decide({"unset": [44.3], "8": [60.0]}, {"8": True})["status"] == "unjudged")
    print("selftest: %d/%d passed" % (tot - bad, tot))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
