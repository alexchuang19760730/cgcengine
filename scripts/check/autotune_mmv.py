#!/usr/bin/env python3
"""autotune_mmv.py -- the wiring layer between three things that already exist.

Why this file exists
--------------------
The autotuner recipe has three pieces, and all three were already there:

  1. INVENTORY  -- `shape_roofline.carrier_projections(gguf)` reads the GGUF header
     and answers "which (quant type, K, N, n_mats, n_used) does this model actually
     have". That is the only honest source for the shape list.

  2. MEASUREMENT -- `test-backend-ops perf -o <op> -p <regex>` gives per-shape
     microseconds from real compiled Metal pipelines. THIS REPLACES the timing
     half of `m1_harness`: that harness shells out to llama-bench and regexes a
     whole-model t/s column, so it has (a) no shape isolation -- a win cannot be
     attributed to a shape, and (b) +/-27% single-arm noise, which is ~9x the
     3% threshold. A knob that moves one kernel by 5% is invisible there and
     obvious here.

  3. KNOBS -- CGC_MMV_NSG / CGC_MMV_NR0 / CGC_MMV_FUSE
     (ggml-metal-device.cpp:801-833 and ggml-metal-ops.cpp:2800-2830). NSG is a
     Metal *function constant* (FC_MUL_MV+0): the pipeline compiles and caches
     per value at runtime, so a sweep needs NO rebuild and NO .metal edit, and
     each row's dot product stays inside one simdgroup so bit-identity holds
     (M1 does not need re-baselining).

What was missing is not a feature: it is the layer that turns (1) into a `-p`
regex, drives (2) once per (3) setting, and decides with a pre-registered rule.
That layer is this file.

Rules encoded here (all "refuse rather than guess")
---------------------------------------------------
* ELIGIBILITY  NSG/NR0 only exist for IQ2_S and IQ3_XXS. Our carrier's ffn_down
  is IQ3_S, so it is REFUSED for tuning, not silently swept (a sweep over a knob
  the kernel ignores measures noise and looks like a result).
* EXISTENCE    Only cases present in the BASELINE capture are tuned. A shape the
  harness has no test case for is reported as missing and refuses (exit 1).
* DECISION     Paired ABBA order, median over reps, win must clear `--threshold`
  (default 3%) AND every rep must agree in sign (the M-W rule: 5/5 same
  direction). Otherwise the verdict is "no evidence", never "best guess".
* BIT-IDENTITY NSG/NR0 preserve per-row reduction order => no M1 re-baseline.
  FUSE is *claimed* bit-identical but measured divergent (777/800 rows) and was
  measured SLOWER end-to-end (15.94 vs 22.07 t/s, draft accept collapsed), so it
  is carried as `adoption=needs_m1_gate` + `prior=measured_worse`.
* VALUE        A per-shape win is not an end-to-end win. `--share` (fraction of
  the step this kernel family owns) is REQUIRED before any end-to-end number is
  printed; value = share x win%.

Measured noise floor of this instrument (first sweep, 2026-09-22)
-----------------------------------------------------------------
`--knob CGC_MMV_NSG --values 1,2,4,8,16,32 --reps 3` on the one decode shape:
unset (= N_SG_IQ2_S = 2) and the explicit `2` arm are BY DEFINITION the same
value, and they differed by 2.42%. So this box's per-cell noise here is ~2.4%
(against llama-bench's +/-27% single-arm noise), and the observed NSG trend
(1 -> 32: 0% -> 4.24%, monotone) is only ~2x that and failed the sign rule at
2/3 reps => verdict `no evidence`, not "NSG=32 is 4% faster".

Usage
-----
    # no GPU: show the grid it would run, and refuse to guess anything
    python3 scripts/check/autotune_mmv.py --gguf models/gguf/<carrier>.gguf --plan

    # the sweep (per-shape, real pipelines, no rebuild)
    python3 scripts/check/autotune_mmv.py --gguf models/gguf/<carrier>.gguf \
        --knob CGC_MMV_NSG --values 1,2,4,8,16,32 --reps 3 --share 0.085 \
        --json Backup/phase_decomp/L3/autotune_nsg.json

    # no GPU, no model: parser + knob-table + eligibility checks
    python3 scripts/check/autotune_mmv.py --selftest
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "src/llama.cpp/build/bin/test-backend-ops"
DYLIB = ROOT / "src/llama.cpp/build/bin/libggml-metal.dylib"   # symlink -> .0 -> .0.19.0

# --- knob table: the only place that knows what a knob is -------------------
# applies  -> quant types whose kernel the knob actually reaches. Anything else
#             is refused (a knob the kernel ignores measures noise).
# values   -> legal values; None means "env unset" (the compiled-in default).
# bitid    -> what happens to bit-identity (M1 = the row_fnv1a64 gate).
KNOBS = {
    "CGC_MMV_NSG": {
        "applies": {"iq2_s", "iq3_xxs"},
        "values": ["1", "2", "4", "8", "16", "32"],
        "default": None,                       # N_SG_IQ2_S = 2 (ggml-metal-impl.h:76)
        "bitid": "preserved (per-row dot product stays in one simdgroup) -> no M1 re-baseline",
        "src": "ggml-metal-device.cpp:818-833 (Metal function constant FC_MUL_MV+0)",
        "note": "runtime compile+cached per value: no rebuild, no .metal edit",
    },
    "CGC_MMV_NR0": {
        "applies": {"iq2_s", "iq3_xxs"},
        "values": ["8"],                       # only the literal "8" is accepted; else default 4
        "default": None,                       # N_R0_IQ2_S = 4 (ggml-metal-impl.h:75)
        "bitid": "preserved",
        "src": "ggml-metal-device.cpp:801-816",
        "note": 'only strcmp(e,"8")==0 is honoured; any other value silently falls back to 4',
    },
    "CGC_MMV_FUSE": {
        "applies": {"iq2_s", "iq3_xxs"},
        "values": ["1"],
        "default": None,
        "bitid": "CLAIMED bit-identical, MEASURED divergent 777/800 rows (CGC bit-bisect v8)",
        "src": "ggml-metal-ops.cpp:2800-2830",
        "prior": "measured_worse",
        "note": "15.94 vs 22.07 t/s and draft accept collapsed. Needs the M1 gate before "
                "adoption, and needs a GLU-shaped case (a bare MUL_MAT_ID will not reach it)",
    },
}

# MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048):
#                11922 runs -    86.69 us/run ...
CASE = re.compile(
    r"(?P<op>MUL_MAT_ID\w*)\(type_a=(?P<type>\w+),type_b=\w+,n_mats=(?P<n_mats>\d+),"
    r"n_used=(?P<n_used>\d+),b=\d+,m=(?P<m>\d+),n=(?P<n>\d+),k=(?P<k>\d+)\):")
US = re.compile(r"([\d.]+)\s+us/run")


# ---- the three pieces -----------------------------------------------------

def inventory(gguf):
    """(1) shape list, from the model itself. -> [(proj, spec)]"""
    sys.path.insert(0, str(ROOT / "scripts/check"))
    import shape_roofline as sr                      # refuses on unknown types
    carrier, meta = sr.carrier_projections(gguf)
    out = []
    for proj, c in carrier.items():
        out.append((proj, dict(c, n_mats=meta.get("expert_count"),
                               n_used=meta.get("expert_used_count"))))
    return out, meta


def case_regex(spec, n_decode_rows=1):
    """(1)->(2) the join: a shape becomes a `-p` regex for test-backend-ops.

    n=1 is the decode GEMV row count. n>1 cases exist in the harness and are
    prefill-ish; tuning on them would be tuning a shape the step does not have.
    """
    # NOTE: -p is std::regex_search over vars(), i.e. over
    #   "type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048"
    # so every field between the ones we pin must be named -- dropping type_b
    # silently matches nothing (that mistake cost one sweep).
    return (r"type_a=%s,type_b=\w+,n_mats=%s,n_used=%s,b=\d+,m=%s,n=%s,k=%s"
            % (spec["type"], spec["n_mats"], spec["n_used"], spec["N"],
               n_decode_rows, spec["K"]))


def measure(regex, env_extra, op="MUL_MAT_ID", timeout=900):
    """(2) one measurement. -> {(case_key): us}. Refuses (exit 1) on no rows."""
    env = dict(os.environ)
    env.update({k: v for k, v in env_extra.items() if v is not None})
    for k, _ in env_extra.items():
        if env_extra[k] is None:
            env.pop(k, None)
    cmd = [str(BIN), "perf", "-o", op, "-p", regex]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return parse_capture(p.stdout + p.stderr), time.time() - t0, cmd


def parse_capture(text):
    """-> {(type, m, n, k, n_mats, n_used): us}. A case with no us/run is dropped."""
    lines = text.splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = CASE.search(line)
        if not m:
            continue
        us = None
        for j in range(i, min(i + 8, len(lines))):
            u = US.search(lines[j])
            if u:
                us = float(u.group(1))
                break
        if us is None:
            continue
        g = m.groupdict()
        out[(g["type"], int(g["m"]), int(g["n"]), int(g["k"]),
             int(g["n_mats"]), int(g["n_used"]))] = us
    return out


# ---- decision -------------------------------------------------------------

def decide(base_samples, cand_samples, threshold=3.0):
    """(median-base%, median-cand%, verdict). Sign agreement across reps is
    required: a win that flips direction between reps is noise, not a win."""
    mb = statistics.median(base_samples)
    mc = statistics.median(cand_samples)
    if mb <= 0:
        return None, None, "refused: baseline median is 0"
    delta = (mb - mc) / mb * 100.0                 # positive = candidate faster
    pairs = list(zip(base_samples, cand_samples))
    # sign agreement is against the MEDIAN direction, not "always faster": a
    # candidate that is consistently slower is a LOSS, not "no evidence".
    same = sum(1 for b, c in pairs if ((b - c) > 0) == (delta > 0))
    if abs(delta) < threshold:
        return delta, mb, "no evidence (|%.2f%%| < %.1f%% threshold)" % (delta, threshold)
    if same != len(pairs):
        return delta, mb, "no evidence (sign agrees %d/%d reps)" % (same, len(pairs))
    return delta, mb, "WIN" if delta > 0 else "LOSS"


def provenance():
    """Both halves of the instrument matter: the kernels live in the dylib, but the
    SHAPES live in test-backend-ops (another session edits that file and rebuilds
    it). A number is only quotable with both md5s."""
    def md5(p):
        try:
            return hashlib.md5(Path(p).read_bytes()).hexdigest()
        except Exception:
            return None
    return {"dylib_md5": md5(DYLIB), "probe_md5": md5(BIN), "probe": str(BIN)}


# ---- commands -------------------------------------------------------------

def cmd_plan(gguf, knob, values):
    inv, meta = inventory(gguf)
    k = KNOBS[knob]
    print("model    %s" % gguf)
    print("experts  %s mats, top-%s" % (meta.get("expert_count"), meta.get("expert_used_count")))
    print("knob     %s  (%s)" % (knob, k["src"]))
    print("bit-id   %s" % k["bitid"])
    rows = []
    for proj, spec in inv:
        elig = spec["type"] in k["applies"]
        rows.append((proj, spec, elig))
        print("\n  %-14s type=%-8s K=%-5s N=%-5s  %s"
              % (proj, spec["type"], spec["K"], spec["N"],
                 "ELIGIBLE" if elig else "REFUSED (knob does not reach this type)"))
        print("      -p regex: %s" % case_regex(spec))
        if spec.get("variants"):
            print("      minority layers not represented: %s" % spec["variants"])
    n_elig = len({(s["type"], s["K"], s["N"]) for _, s, e in rows if e})
    if n_elig == 0:
        print("\nnothing to tune: no projection is of a type %s reaches (%s)"
              % (knob, sorted(k["applies"])))
        return 2
    print("\ngrid: %d eligible shape(s) x %d value(s) (+baseline) x reps"
          % (n_elig, len(values)))
    print("note: NSG default (env unset) is already the compiled-in N_SG_* value; "
          "include it in --values if you want it as an explicit arm")
    return 0


def cmd_run(gguf, knob, values, reps, share, op, out_json, timeout):
    k = KNOBS[knob]
    inv, meta = inventory(gguf)
    # dedupe by SHAPE: ffn_gate and ffn_up are the same (type,K,N) here, and a
    # duplicated regex would print the same case twice and read as two shapes.
    targets = {}
    for proj, spec in inv:
        if spec["type"] not in k["applies"]:
            print("  %-14s REFUSED (type %s: knob does not reach it)" % (proj, spec["type"]))
            continue
        ckey = (spec["type"], spec["N"], 1, spec["K"], spec["n_mats"], spec["n_used"])
        targets.setdefault(ckey, ([], spec))[0].append(proj)
    if not targets:
        sys.exit("FATAL: nothing eligible -- refusing to sweep a knob the kernel ignores")
    for ckey, (projs, _) in targets.items():
        print("  %-14s ELIGIBLE (case %s)" % ("+".join(projs), "/".join(map(str, ckey))))

    regexes = {ckey: case_regex(spec) for ckey, (_, spec) in targets.items()}
    # one combined regex: all eligible shapes in a single run (cheaper, and they
    # share the same compiled pipeline state)
    # plain alternation: std::regex here is ECMAScript, so do NOT use "(?:...)"
    combined = "|".join(regexes.values())
    settings = [None] + list(values)               # None = baseline (env unset)

    prov = provenance()
    print("dylib md5 %s | probe md5 %s  (numbers belong to these two binaries, "
          "not to the working tree)" % (prov["dylib_md5"], prov["probe_md5"]))
    print("baseline run first -- only cases that appear there will be tuned")
    base, dt, cmd = measure(combined, {knob: None}, op=op, timeout=timeout)
    print("  %d case(s) in %.0fs" % (len(base), dt))
    if not base:
        sys.exit("FATAL: baseline produced no measured case for regex %s\n  cmd: %s"
                 % (combined, " ".join(cmd)))

    samples = {s: {key: [] for key in base} for s in settings}
    for rep in range(reps):
        order = settings if rep % 2 == 0 else list(reversed(settings))   # ABBA
        for s in order:
            env = {knob: s}
            got, dt, _ = measure(combined, env, op=op, timeout=timeout)
            for key in base:
                if key in got:
                    samples[s][key].append(got[key])
                else:
                    print("  WARN: case %s missing at %s=%s -- dropped" % (key, knob, s))
        print("  rep %d/%d done" % (rep + 1, reps))

    print("\n%-9s %-22s %10s %10s %8s  %s"
          % (knob, "case", "base us", "best us", "delta", "verdict"))
    results = []
    for key in sorted(base):
        base_s = samples[None][key]
        if not base_s:
            continue
        best = None
        for s in settings:
            if s is None:
                continue
            cand = samples[s][key]
            if len(cand) != len(base_s):
                continue
            d, mb, verdict = decide(base_s, cand)
            if d is None:
                continue
            if best is None or d > best[0]:
                best = (d, s, mb, statistics.median(cand), verdict)
        if best is None:
            print("%-9s %-22s %10s %10s %8s  %s"
                  % ("-", "/".join(map(str, key)), "-", "-", "-", "no data"))
            continue
        d, s, mb, mc, verdict = best
        name = "%s m=%s k=%s [%s]" % (key[0], key[1], key[3],
                                      "+".join(targets[key][0]))
        print("%-9s %-22s %10.2f %10.2f %+7.2f%%  %s"
              % ("%s=%s" % (knob.replace("CGC_MMV_", ""), s), name, mb, mc, d, verdict))
        row = {"case": list(key), "knob": knob, "value": s, "base_us": mb,
               "best_us": mc, "delta_pct": d, "verdict": verdict,
               "base_samples": base_s, "cand_samples": samples[s][key]}
        if verdict == "WIN" and share:
            row["end_to_end_pct"] = d * share
            print("           end-to-end = %.2f%% x share %.4f = %+.2f%% (threshold 3%%)"
                  % (d, share, row["end_to_end_pct"]))
        elif verdict == "WIN":
            print("           end-to-end NOT printed: pass --share to translate")
        results.append(row)

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": str(gguf), "provenance": prov, "op": op,
                   "knob": knob, "values": values, "reps": reps, "share": share,
                   "settings": {str(s): {"/".join(map(str, k)): v
                                         for k, v in samples[s].items()} for s in settings},
                   "results": results},
                  open(out_json, "w"), indent=2)
        print("\njson -> %s" % out_json)
    return 0


def selftest():
    """No GPU, no model: parser, knob table, eligibility, decision rule."""
    fails = []

    txt = """
  MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048):                11922 runs -    86.69 us/run -  16.78 MFLOP/run - 193.53 GFLOPS
  MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=2,k=2048):                 5962 runs -   203.71 us/run -  33.55 MFLOP/run - 164.71 GFLOPS
  MUL_MAT_ID(type_a=iq3_s,type_b=f32,n_mats=256,n_used=8,b=0,m=2048,n=1,k=512):                 1000 runs -   999.99 us/run
  MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=9,k=2048):                    1 runs - (no number)
"""
    got = parse_capture(txt)
    if got.get(("iq2_s", 512, 1, 2048, 256, 8)) != 86.69:
        fails.append("parser: decode case -> %r" % got.get(("iq2_s", 512, 1, 2048, 256, 8)))
    if len(got) != 3:
        fails.append("parser: expected 3 rows (the 4th has no us/run), got %d" % len(got))

    # knob table integrity
    for name, k in KNOBS.items():
        if not k["applies"]:
            fails.append("%s: no applies set" % name)
        if not k["values"]:
            fails.append("%s: no values" % name)
    if KNOBS["CGC_MMV_NSG"]["values"] != ["1", "2", "4", "8", "16", "32"]:
        fails.append("NSG grid does not match the clamp 1..32")
    if KNOBS["CGC_MMV_FUSE"].get("prior") != "measured_worse":
        fails.append("FUSE lost its measured_worse prior -- it must not be adopted silently")

    # eligibility: our carrier's down is IQ3_S -> must be refused
    for name, k in KNOBS.items():
        if "iq3_s" in k["applies"]:
            fails.append("%s claims iq3_s, but NSG/NR0/FUSE only reach iq2_s/iq3_xxs" % name)
        if "iq2_s" not in k["applies"]:
            fails.append("%s does not reach iq2_s (our gate/up type)" % name)

    # decision rule
    d, _, v = decide([100, 100, 100], [94, 95, 96], 3.0)
    if not (v == "WIN" and d > 3):
        fails.append("decide: clear win -> %r %r" % (d, v))
    d, _, v = decide([100, 100, 100], [99, 99, 99], 3.0)
    if v == "WIN":
        fails.append("decide: 1%% must not be a WIN (below 3%% threshold)")
    d, _, v = decide([100, 90, 110], [95, 95, 95], 3.0)
    if v == "WIN":
        fails.append("decide: sign must agree on every rep (got %r)" % v)
    d, _, v = decide([100, 100], [110, 110], 3.0)
    if v != "LOSS":
        fails.append("decide: slower candidate must be LOSS, got %r" % v)

    # provenance must name BOTH binaries (kernels in the dylib, shapes in the probe)
    prov = provenance()
    if not (prov.get("dylib_md5") and prov.get("probe_md5")):
        fails.append("provenance: could not md5 both binaries (%s)" % prov)

    # the join: shape -> regex must name the decode row count n=1
    r = case_regex({"type": "iq2_s", "n_mats": 256, "n_used": 8, "N": 512, "K": 2048})
    if ",n=1," not in r:
        fails.append("case_regex must pin n=1 (decode GEMV), got %s" % r)

    print("selftest: %d checks failed" % len(fails))
    for f in fails:
        print("  FAIL: %s" % f)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--knob", default="CGC_MMV_NSG", choices=sorted(KNOBS))
    ap.add_argument("--values", default=",".join(KNOBS["CGC_MMV_NSG"]["values"]))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--share", type=float, default=None,
                    help="fraction of the decode step this kernel family owns; REQUIRED "
                         "before any end-to-end number is printed")
    ap.add_argument("--op", default="MUL_MAT_ID")
    ap.add_argument("--json", dest="out_json")
    ap.add_argument("--plan", action="store_true", help="print the grid, touch no GPU")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.gguf:
        sys.exit("FATAL: --gguf is required (the shape list must come from the model)")
    if not BIN.exists():
        sys.exit("FATAL: %s missing -- the measurement half is not built" % BIN)
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    if args.plan:
        return cmd_plan(args.gguf, args.knob, values)
    if args.run:
        return cmd_run(args.gguf, args.knob, values, args.reps, args.share,
                       args.op, args.out_json, args.timeout)
    sys.exit("FATAL: choose --plan, --run or --selftest")


if __name__ == "__main__":
    sys.exit(main())
