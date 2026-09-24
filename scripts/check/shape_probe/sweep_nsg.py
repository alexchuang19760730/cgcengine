#!/usr/bin/env python3
"""sweep_nsg.py -- scan the Metal tile knob (CGC_MMV_NSG) on REAL model shapes.

Why marginal cost, not per-op time:
    every graph costs roughly the same fixed amount to build/submit/commit
    (~230-360 us here) no matter how many ops it holds. A single "us/op" number
    silently includes a slice of that constant, which is why a tiny 0.5 MiB GEMV
    looked like it ran at 13 GB/s. Two different graph sizes remove it:

        g(b) = fixed + b * marginal      -> measure g(32) and g(64)
        marginal = (g(64) - g(32)) / (64 - 32)

    `marginal` is what the step actually pays per extra dispatch. Comparing knob
    settings on anything else compares mostly the harness.

2026-09-23 -- what this upgrade adds, and why it was needed:

  * REACH GATE. nsg is a Metal function constant, so a value the type's getter does
    not read compiles the SAME pipeline and looks perfectly flat. The pipeline name
    is parsed per arm and compared with the value set; an arm whose name does not
    carry it is reported unwired and the channel is refused. This is the whole reason
    a rescan was owed: P1-3d (dense IQ4_XS/Q6_K/Q8_0) landed after the first dense
    sweep, so before it, three of this model's four dense types could not have moved
    for any value -- a flat reading there was evidence about the plumbing.
  * ROTATION + NULL CELL. A sweep that runs unset,2,4,... once has a null cell that
    only brackets the sweep in TIME, so any monotone drift (swap, a neighbour waking
    up) lands in it. Arms are rotated per rep and the unset arm is measured in every
    rep; if the channel's own unset spread exceeds the threshold the whole channel is
    REFUSED rather than reported as a median.
  * CENSUS SHAPES. The shape table is the per-step (type, k, n, count) census of the
    delivery carrier, taken from the GGUF header, so each row prices into ms/step.

Usage:
    shape_probe/sweep_nsg.py --libdir /tmp/cgc-nsgid-build/bin --reps 3 \
        --shapes iq4_xs:2048x8192:40 iq4_xs:4096x2048:40 ...
    shape_probe/sweep_nsg.py --selftest
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE / "cgc_shape_probe"

PEAK_GB_S = 108.8          # the M4 peak this repo measured and quotes (GiB/s)
STEP_MS = 143.0            # the delivery step the census is priced against

# Per-step census of `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X` (GGUF header, decode,
# MTP off). `count` is the number of these matmuls ONE step pays for:
#   iq4_xs 2048x8192  attn_qkv x30 (linear layers) + attn_q x10 (full-attn layers)
#   iq4_xs 4096x2048  ssm_out x30 + attn_output x10
#   iq4_xs 2048x4096  attn_gate x30
#   iq4_xs 2048x512   ffn_gate_shexp x40 + ffn_up_shexp x40 + attn_k x10 + attn_v x10
#   iq4_xs 512x2048   ffn_down_shexp x40
#   q6_k   2048x248320 output.weight (lm_head) x1  [token_embd is a lookup, not a matmul]
#   q8_0   blk.39 only (attn_q/k/v/output, ffn_{gate,up,down}_shexp) + blk.40 nextn.eh_proj
SHAPES = (
    ("iq4_xs", 2048, 8192, 40, "attn_qkv+attn_q"),
    ("iq4_xs", 4096, 2048, 40, "ssm_out+attn_output"),
    ("iq4_xs", 2048, 4096, 30, "attn_gate"),
    ("iq4_xs", 2048, 512, 100, "shexp gate/up + attn_k/v"),
    ("iq4_xs", 512, 2048, 40, "shexp down"),
    ("q6_k", 2048, 248320, 1, "lm_head"),
    ("q8_0", 2048, 8192, 1, "blk.39 attn_q"),
    ("q8_0", 4096, 2048, 2, "blk.39 attn_output + nextn.eh_proj"),
    ("q8_0", 2048, 512, 3, "blk.39 attn_k/v + shexp up"),
    ("q8_0", 512, 2048, 1, "blk.39 shexp down"),
)

BPW = {"iq4_xs": 4.25, "q6_k": 6.5625, "q8_0": 8.5, "f32": 32.0, "bf16": 16.0, "f16": 16.0}


def pool_for(tname, k, n, b):
    """Enough device memory for `b` DISTINCT weights (each op reads its own)."""
    per = k * n * BPW.get(tname, 4.0) / 8.0
    return int(b * per / (1024 * 1024)) + 256


def rotated(values, rep):
    """Arm order for rep `rep`: the list rotated left. Every arm lands in a different
    time slot across reps, and the unset arm (which is the null cell) is present in all."""
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


def worst_case_stat(vals):
    """Per-arm statistic for a ONE-SIDED contamination model: min, not median.

    Contention only ever SLOWS a GPU microbenchmark, so a slow reading is contaminated and a fast
    one is the better estimate of the kernel -- while `median` mixes the two and (worse) the
    marginal model amplifies any per-graph fixed-cost error by 1/(b2-b1). Measured on this box
    with a neighbour on the compositor: the same arm's %%-peak spanned 10.6%% across three reps,
    which is exactly the drift that refuses every dense channel when read as a median.
    The bias of a min is the opposite one (it can only overstate throughput), which is the
    direction this sweep wants: the question is how much the knob CAN recover.
    """
    return min(vals) if vals else None


def decide(per_arm, reached, thr_floor=0.05, stat=median):
    """The verdict from readings only: {tag: [pct_peak per rep]} + {tag: reached}.

    Pure, because the decision must be exercisable on a channel that drifted -- and both
    of this sweep's real runs (2026-09-22 20:00 and the 2026-09-23 rescan) produced at
    least one channel that had to be refused. A median taken across a wide null cell is
    not a small effect, it is no reading.
    """
    unsets = per_arm.get("unset") or []
    if len(unsets) < 2:
        return {"status": "unjudged",
                "why": "the unset arm ran once: no null cell, so nothing separates the knob "
                       "from a drift"}
    ref = stat(unsets)
    noise = (max(unsets) - min(unsets)) / ref if ref else float("inf")
    thr = max(thr_floor, 2 * noise)
    out = {"null_ref_pct": ref, "null_pct_per_rep": unsets, "noise_frac": noise,
           "threshold_frac": thr, "arms_pct": {t: stat(v) for t, v in per_arm.items() if v},
           "unwired": sorted(t for t, ok in reached.items() if ok is False),
           "median_pct": {t: stat(v) for t, v in per_arm.items() if v}}
    if out["unwired"]:
        out["status"] = "INVALID"
        out["why"] = ("nsg=%s never reached the kernel (the compiled pipeline name does not "
                      "carry the value): a flat reading there is evidence about the plumbing"
                      % ", ".join(out["unwired"]))
        return out
    if noise > thr_floor:
        out["status"] = "INVALID"
        out["why"] = ("the CHANNEL moved: the unset arm itself spans %.1f%% across reps, so "
                      "every arm difference is inside the trend" % (100 * noise))
        return out
    wins = sorted((t for t, v in out["median_pct"].items()
                   if t != "unset" and v > ref * (1 + thr)), key=lambda t: -out["median_pct"][t])
    out["wins"] = wins
    if wins:
        out["status"] = "win"
        out["winner"] = wins[0]
    else:
        out["status"] = "no-lever"
        out["why"] = "no arm beats unset by more than the threshold"
    return out


def parse(stderr):
    """(marginal_us_per_op, per_graph_fixed_us, pipeline_name) from one invocation."""
    loops = per_op = None
    for line in stderr.splitlines():
        if "batch:" in line and "loops" in line:
            loops = int(line.split("batch:")[1].split("loops")[0].strip())
        if line.startswith("RESULT") and "batch" in line:
            per_op = float(line.split("median")[1].split("us/op")[0].strip())
    m = re.search(r"name = '([^']+)'", stderr)
    pipe = m.group(1) if m else None
    return loops, per_op, pipe


def pipeline_nsg(name):
    m = re.search(r"nsg=(\d+)", name or "")
    return int(m.group(1)) if m else None


def run(tname, k, out_n, env, b1, b2, target_ms, libdir=None):
    """(marginal_us_per_op, per_graph_fixed_us, pipeline) or (None, None, error)."""
    res, pipe = {}, None
    for b in (b1, b2):
        cmd = [str(PROBE), "--type", tname, "--k", str(k), "--out", str(out_n),
               "--r", "1", "--mode", "batch", "--batch", str(b),
               "--iters", "1", "--target-ms", str(target_ms),
               "--pool-mib", str(pool_for(tname, k, out_n, b))]
        e = dict(os.environ)
        e.update(env or {})
        if libdir:
            e["DYLD_LIBRARY_PATH"] = libdir + ":" + e.get("DYLD_LIBRARY_PATH", "")
        p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=900)
        if p.returncode != 0:
            return None, None, (p.stderr or "")[-300:]
        loops, per_op, pipe_name = parse(p.stderr)
        if loops is None or per_op is None:
            return None, None, "no parse: " + p.stderr[-200:]
        res[b] = per_op * b            # us per GRAPH (one loop), not per op
        pipe = pipe_name or pipe
    return (res[b2] - res[b1]) / (b2 - b1), res[b1] - b1 * ((res[b2] - res[b1]) / (b2 - b1)), pipe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knob", default="CGC_MMV_NSG")
    ap.add_argument("--values", default="2,4,8,16,32")
    ap.add_argument("--shapes", default="", help='override the census table, e.g. '
                                                '"iq4_xs:2048x8192:40" ...')
    ap.add_argument("--b1", type=int, default=32)
    ap.add_argument("--b2", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--statistic", choices=("median", "min"), default="median",
                    help="per-arm statistic across reps. `min` is the right one on a busy box "
                         "(contention is one-sided and the marginal model amplifies fixed-cost "
                         "error); `median` is the default so an old run stays reproducible")
    ap.add_argument("--target-ms", type=float, default=1500.0)
    ap.add_argument("--libdir", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not PROBE.exists():
        sys.exit("FATAL: probe missing -- run build.sh")

    if args.shapes:
        shapes = []
        for spec in args.shapes.split():
            parts = spec.split(":")
            if len(parts) == 3:                      # type:KxN:count
                tname, dims, count = parts
                k, n = (int(x) for x in dims.split("x"))
            elif len(parts) == 4:                    # type:K:N:count
                tname, k, n, count = parts
                k, n = int(k), int(n)
            else:
                sys.exit("FATAL: --shapes entries are type:KxN:count or type:K:N:count (got %r)" % spec)
            shapes.append((tname, k, n, int(count), ""))
    else:
        shapes = SHAPES

    # ints, not the argv strings: the reach gate compares these against the nsg parsed out of
    # the pipeline name, and `16 == "16"` is False -- which reported every arm as "never reached
    # the kernel" on the first run of this upgrade (the gate's own class of defect: a comparison
    # that can only ever answer no, and reads as evidence about the plumbing).
    values = [None] + [int(v) for v in args.values.split(",")]
    reps = max(1, args.reps)
    print("knob %s   values %s   marginal model: (%d ops - %d ops)/%d   reps %d (rotated), "
          "statistic %s"
          % (args.knob, [v or "unset" for v in values], args.b2, args.b1, args.b2 - args.b1, reps,
             args.statistic))
    print("libdir %s" % (args.libdir or "(tree default)"))
    print("peak %.1f GiB/s   step %.0f ms   (the census is priced against these)"
          % (PEAK_GB_S, STEP_MS))
    print("-" * 108)

    stat = worst_case_stat if args.statistic == "min" else median
    out = {"knob": args.knob, "b1": args.b1, "b2": args.b2, "libdir": args.libdir,
           "reps": reps, "statistic": args.statistic, "peak_gib_s": PEAK_GB_S, "step_ms": STEP_MS,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "shapes": {},
           "provenance": provenance(args.libdir)}
    print("engine %s   thermal %s   window %s"
          % (out["provenance"]["libggml_metal"], out["provenance"]["thermal_at_start"].get("label"),
             out["provenance"]["window_at_start"].get("class")))
    family_ms = {"unset": 0.0}
    for (tname, k, n, count, label) in shapes:
        key = "%s_%dx%d" % (tname, k, n)
        op_mib = k * n * BPW[tname] / 8.0 / (1024 ** 2)
        print("  %-8s %-12s x%-4d %7.1f MiB/step  %s"
              % (tname, "%dx%d" % (k, n), count, op_mib * count, label))
        per_arm, reached, passes = {}, {}, []
        for rep in range(reps):
            tags = []
            for val in rotated(values, rep):
                tag = "unset" if val is None else str(val)
                m, fixed, pipe = run(tname, k, n, ({args.knob: str(val)} if val else {}),
                                     args.b1, args.b2, args.target_ms, args.libdir)
                if m is None:
                    print("     rep %d %-6s FAILED -- %s" % (rep, tag, str(pipe)[-120:]))
                    out["shapes"].setdefault(key, {})["error_%s_rep%d" % (tag, rep)] = str(pipe)[-200:]
                    continue
                # the op's own bytes at this shape, so the driver cannot disagree with the probe
                gib_s = (op_mib * (1024 ** 2) / (m * 1e-6)) / (1024 ** 3)
                pct = 100 * gib_s / PEAK_GB_S
                per_arm.setdefault(tag, []).append(pct)
                got = pipeline_nsg(pipe)
                if val is not None:
                    reached[tag] = reached.get(tag, True) and (got == val)
                tags.append(tag)
                out["shapes"].setdefault(key, {}).setdefault("arms", {}).setdefault(tag, []).append(
                    {"rep": rep, "marginal_us": m, "fixed_us": fixed, "pct_peak": pct,
                     "pipeline": pipe, "reached": None if val is None else (got == val)})
            passes.append({"rep": rep, "order": tags,
                           "pct": {t: per_arm[t][-1] for t in tags if per_arm.get(t)}})
            print("     rep %d  " % rep
                  + "  ".join("%s %.1f%%" % (t, per_arm[t][-1]) for t in tags if per_arm.get(t)))
            sys.stdout.flush()
        verdict = decide(per_arm, reached, stat=stat)
        # A drifted channel is refused BEFORE any median exists, so this dict is absent on that
        # path -- reading it unconditionally is a KeyError mid-sweep (the same defect class that
        # once crashed the id sweep after its first shape had already been measured).
        meds = verdict.get("median_pct") or {}
        out["shapes"][key].update({"verdict": verdict, "passes": passes, "op_mib": op_mib,
                                   "count": count, "label": label,
                                   "ms_per_step": {t: ms_from_pct(tname, k, n, count, p)
                                                   for t, p in meds.items()}})
        print("     => %-9s %s" % (verdict["status"], verdict.get("why", "")
                                   or "winner nsg=%s" % verdict.get("winner")))
        if not meds:
            print("        (refused before a median existed -- no percent-peak or ms/step printed)")
        for tag in sorted(meds, key=lambda t: (t == "unset", t)):
            ms = ms_from_pct(tname, k, n, count, meds[tag])
            family_ms[tag] = family_ms.get(tag, 0.0) + ms
            print("        nsg=%-6s median %5.1f%% peak   %6.2f ms/step"
                  % (tag, meds[tag], ms))
        # Best-case diagnostic, printed unconditionally and labelled as a bound rather than a
        # verdict: contention is one-sided, so a channel's fastest reading is the least
        # contaminated one. When the null cell refuses the channel this is the only statement
        # left -- "no arm ever beat unset by more than X" -- and it must never be quoted as a
        # win (a single lucky arm min can look like one).
        if per_arm.get("unset"):
            best_unset = min(per_arm["unset"])
            cand = {t: min(v) for t, v in per_arm.items() if t != "unset" and v}
            if cand:
                t_best = max(cand, key=lambda t: cand[t])
                gain = 100.0 * (cand[t_best] - best_unset) / best_unset
                best_ms = ms_from_pct(tname, k, n, count, cand[t_best])
                unset_ms = ms_from_pct(tname, k, n, count, best_unset)
                out["shapes"][key]["best_case"] = {"arm": t_best, "gain_pct": gain,
                                                   "ms_saved": unset_ms - best_ms}
                print("        best case (min over reps, an UPPER BOUND only): nsg=%s %+.1f%% "
                      "=> %.2f ms/step saved (of %.2f)"
                      % (t_best, gain, unset_ms - best_ms, unset_ms))
        sys.stdout.flush()

    if family_ms.get("unset"):
        print("-" * 108)
        print("dense family, priced (this table only):")
        for tag in sorted(family_ms, key=lambda t: (t == "unset", t)):
            d = family_ms[tag] - family_ms["unset"]
            print("   nsg=%-6s %6.2f ms/step   vs unset %+6.2f ms  (%+.2f%% of a %.0f ms step)"
                  % (tag, family_ms[tag], d, 100 * d / STEP_MS, STEP_MS))
        out["family_ms"] = family_ms
        out["family_delta_ms"] = {t: family_ms[t] - family_ms["unset"] for t in family_ms}
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n")
        print("wrote %s" % p)
    return 0


def provenance(libdir):
    """Engine digest + box state, so a row can be attributed later.

    This project's rule for measurement products: without the engine identity and the window the
    box was in, a row cannot be compared with anything -- and this sweep's whole history is rows
    that could not be compared (five runs, five unreadable null cells, one of them on a build whose
    knob never reached the kernel).
    """
    import hashlib
    dirs = [libdir] if libdir else [str(ROOT / "src/llama.cpp/build/bin")]
    digests = {}
    for d in dirs:
        p = Path(d) / "libggml-metal.0.19.0.dylib"
        if not p.exists():
            p = Path(d) / "libggml-metal.dylib"
        if p.exists():
            digests[Path(d).name + "/" + p.name] = hashlib.md5(p.read_bytes()).hexdigest()[:16]
    out = {"libggml_metal": digests, "probe": {}}
    try:
        pp = PROBE.resolve()
        out["probe"] = {"path": str(pp), "md5": hashlib.md5(pp.read_bytes()).hexdigest()[:16]}
    except OSError:
        pass
    try:
        sys.path.insert(0, str(ROOT / "scripts" / "check"))
        import thermal_pressure as tp
        out["thermal_at_start"] = tp.stamp()
    except Exception as e:
        out["thermal_at_start"] = {"level": None, "label": "UNREADABLE: %s" % e}
    try:
        import server_window as sw
        w = sw.decision()
        out["window_at_start"] = {"class": w.get("class"),
                                  "admits": w.get("admits"),
                                  "reclaimable_mb": w.get("reclaimable_mb"),
                                  "refused_by": w.get("refused_by")}
    except Exception as e:
        out["window_at_start"] = {"class": "unknown", "why": str(e)}
    return out


def ms_from_pct(tname, k, n, count, pct):
    """ms/step for `count` ops of (k,n) at `pct` % of peak."""
    op_mib = k * n * BPW[tname] / 8.0 / (1024 ** 2)
    gib_per_s = PEAK_GB_S * pct / 100.0
    return (op_mib * (1024 ** 2) * count) / (gib_per_s * (1024 ** 3)) * 1000.0


def selftest():
    """The decision rule, the rotation, and the pricing -- on fixtures, without a GPU."""
    bad = 0

    def chk(name, got, want, tol=1e-9):
        nonlocal bad
        ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else got == want
        if not ok:
            print("FAIL %s: got %r want %r" % (name, got, want))
            bad += 1

    # rotation: every arm visits a different slot, and the null cell stays present
    chk("rotated r0", rotated(["unset", "2", "4"], 0), ["unset", "2", "4"])
    chk("rotated r1", rotated(["unset", "2", "4"], 1), ["2", "4", "unset"])
    chk("rotated r2", rotated(["unset", "2", "4"], 2), ["4", "unset", "2"])

    # a flat channel, tight null cell -> no lever
    v = decide({"unset": [40.0, 40.4, 39.8], "2": [40.1, 40.0, 40.2]}, {"2": True})
    chk("flat status", v["status"], "no-lever")
    chk("flat threshold", v["threshold_frac"], 0.05)
    # a channel whose own null cell spans 30% -> refused, not reported
    v = decide({"unset": [40.0, 52.0, 40.0], "8": [60.0, 60.0, 60.0]}, {"8": True})
    chk("drifted status", v["status"], "INVALID")
    # the knob never reached the kernel -> refused even if an arm looks like a win
    v = decide({"unset": [40.0, 40.0, 40.0], "16": [50.0, 50.0, 50.0]}, {"16": False})
    chk("unwired status", v["status"], "INVALID")
    # one rep only -> unjudged (no null cell)
    chk("single rep", decide({"unset": [40.0]}, {})["status"], "unjudged")
    # the reach gate's own two failure modes: the parse, and the comparison's types
    chk("nsg parse", pipeline_nsg("kernel_mul_mv_iq4_xs_f32_nsg=16_ne12=1_r2=1_r3=1"), 16)
    chk("nsg parse absent", pipeline_nsg("kernel_mul_mv_iq4_xs_f32"), None)
    chk("reach compare is int-typed",
        pipeline_nsg("kernel_mul_mv_iq4_xs_f32_nsg=4_ne12=1") == int("4"), True)
    # the best-case diagnostic must be the MIN of an arm's reps, and cannot turn a refused
    # channel into a win
    unset_reads = [40.0, 52.0, 41.0]
    chk("best case uses min", min(unset_reads), 40.0)
    v = decide({"unset": unset_reads, "8": [46.0, 46.4, 45.8]}, {"8": True})
    chk("best case does not overturn a refusal", v["status"], "INVALID")
    # a real win, larger than the threshold
    v = decide({"unset": [40.0, 40.0, 40.0], "4": [46.0, 46.0, 46.0]}, {"4": True})
    chk("win status", v["status"], "win")
    chk("win winner", v["winner"], "4")
    # threshold is 2x the null spread when the null spread exceeds the floor
    v = decide({"unset": [40.0, 42.0, 40.0], "8": [44.0, 44.0, 44.0]}, {"8": True})
    chk("2x noise threshold", round(v["threshold_frac"], 4), 0.1)

    # marginal model: g(b) = fixed + b*marginal must invert exactly
    b1, b2, marg, fixed = 32, 64, 3.5, 200.0
    g1, g2 = fixed + b1 * marg, fixed + b2 * marg
    chk("marginal invert", (g2 - g1) / (b2 - b1), marg, 1e-9)
    chk("fixed invert", g1 - b1 * ((g2 - g1) / (b2 - b1)), fixed, 1e-9)

    # pricing: iq4_xs 2048x8192 x40 at 100% of peak must equal the census bytes / peak
    ms = ms_from_pct("iq4_xs", 2048, 8192, 40, 100.0)
    want = (2048 * 8192 * 4.25 / 8 * 40) / (PEAK_GB_S * (1024 ** 3)) * 1000.0
    chk("pricing at peak", ms, want, 1e-6)
    # and 50% of peak must cost exactly twice as long
    chk("pricing halves", ms_from_pct("iq4_xs", 2048, 8192, 40, 50.0), 2 * ms, 1e-6)

    # the census's own internal consistency: the four iq4_xs big shapes + lm_head should
    # total ~1.21 GiB/step, i.e. ~10.4 ms at peak -- the number the doc quotes
    total = sum(k * n * BPW[t] / 8 * c for (t, k, n, c, _) in SHAPES)
    ms_floor = total / (PEAK_GB_S * (1024 ** 3)) * 1000.0
    if not (9.5 <= ms_floor <= 11.5):
        print("FAIL census floor: %.2f ms, expected ~10.4" % ms_floor)
        bad += 1

    fails = bad + (0 if 9.5 <= ms_floor <= 11.5 else 1)
    print("selftest: %d/%d passed (census floor %.2f ms/step at %.1f GiB/s)"
          % (22 - fails, 22, ms_floor, PEAK_GB_S))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
