#!/usr/bin/env python3
"""sweep_ns g.py -- scan a Metal tile knob on REAL model shapes and price the win.

Why marginal cost, not per-op time:
    every graph costs roughly the same fixed amount to build/submit/commit
    (~230-360 us here) no matter how many ops it holds. A single "us/op" number
    silently includes a slice of that constant, which is why a tiny 0.5 MiB GEMV
    looked like it ran at 13 GB/s. Two different graph sizes remove it:

        g(b) = fixed + b * marginal      -> measure g(32) and g(64)
        marginal = (g(64) - g(32)) / (64 - 32)

    `marginal` is what the step actually pays per extra dispatch. Comparing knob
    settings on anything else compares mostly the harness.

Usage:
    shape_probe/sweep_nsg.py --knob CGC_MMV_NSG --values 1,2,4,8,16,32 \
        --shapes 2048x8192:iq4_xs,2048x512:iq4_xs --libdir /tmp/cgc-nsg-build/...

Two questions it answers, in order:
    1. does the knob even reach the kernel?  (pipeline name must change)
    2. is any shape better by more than the instrument's own noise?
Only then does it translate into % of a real step.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = HERE / "cgc_shape_probe"

PEAK_GB_S = 108.8
STEP_MS_LO = 79.5

# per-step bytes and dispatch count for the shapes worth caring about
# (from Backup/phase_decomp/L3/dense_bytes_census.py + the run_shapes pricing)
KNOWN = {
    ("iq4_xs", 2048, 8192):   dict(count=40,  bytes_per_step=340.0 * 1048576, label="attn_qkv"),
    ("iq4_xs", 4096, 2048):   dict(count=40,  bytes_per_step=170.0 * 1048576, label="ssm_out"),
    ("iq4_xs", 2048, 4096):   dict(count=30,  bytes_per_step=127.5 * 1048576, label="attn_gate"),
    ("iq4_xs", 2048, 512):    dict(count=100, bytes_per_step=53.1 * 1048576,  label="ffn_gate_shexp"),
    ("iq4_xs", 512, 2048):    dict(count=40,  bytes_per_step=21.2 * 1048576,  label="ffn_down_shexp"),
    ("f32", 2048, 256):       dict(count=40,  bytes_per_step=80.0 * 1048576,  label="router"),
    ("q6_k", 2048, 248320):   dict(count=1,   bytes_per_step=397.9 * 1048576, label="lm_head"),
}


def run(type_name, k, out_n, env, b1, b2, target_ms, libdir=None):
    """Return (marginal_us_per_op, per_graph_fixed_us) or (None, None)."""
    res = {}
    for b in (b1, b2):
        cmd = [str(PROBE), "--type", type_name, "--k", str(k), "--out", str(out_n),
               "--r", "1", "--mode", "batch", "--batch", str(b),
               "--iters", "1", "--pool-mib", str(pool_for(type_name, k, out_n, b))]
        # target time instead of the default 0.8 s: more loops = less noise in
        # the difference we are about to take.
        e = dict(os.environ)
        e.update(env or {})
        if libdir:
            e["DYLD_LIBRARY_PATH"] = libdir + ":" + e.get("DYLD_LIBRARY_PATH", "")
        out_json = Path("/tmp/cgc_probe_%s_%d_%d.json" % (type_name, k, b))
        cmd += ["--json", str(out_json)]
        p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=600)
        if p.returncode != 0:
            return None, None, (p.stderr or "")[-300:]
        loops = None
        for line in p.stderr.splitlines():
            if "batch:" in line:
                loops = int(line.split("batch:")[1].split("loops")[0].strip())
        per_op = None
        for line in p.stderr.splitlines():
            if line.startswith("RESULT") and "batch" in line:
                per_op = float(line.split("median")[1].split("us/op")[0].strip())
        if loops is None or per_op is None:
            return None, None, "no parse: " + p.stderr[-200:]
        res[b] = per_op * b          # us per GRAPH (one loop), not per op
    marginal = (res[b2] - res[b1]) / (b2 - b1)
    fixed = res[b1] - b1 * marginal
    return marginal, fixed, ""


BPW = {"iq4_xs": 4.25, "q6_k": 6.5625, "f32": 32.0, "q8_0": 8.5, "bf16": 16.0, "f16": 16.0}


def pool_for(tname, k, n, b):
    """Enough device memory for `b` DISTINCT weights (each op reads its own)."""
    per = k * n * BPW.get(tname, 4.0) / 8.0
    return int(b * per / (1024 * 1024)) + 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", default="CGC_MMV_NSG")
    ap.add_argument("--values", default="2,4,8,16,32")
    ap.add_argument("--shapes", default="2048x8192:iq4_xs,2048x512:iq4_xs")
    ap.add_argument("--b1", type=int, default=32)
    ap.add_argument("--b2", type=int, default=64)
    ap.add_argument("--libdir", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not PROBE.exists():
        sys.exit("FATAL: probe missing -- run build.sh")

    shapes = []
    for spec in args.shapes.split(","):
        dims, tname = spec.split(":")
        k, n = (int(x) for x in dims.split("x"))
        shapes.append((tname, k, n))

    values = [None] + [v for v in args.values.split(",")]
    print("knob %s   values %s   marginal model: (%s ops - %s ops)/%d"
          % (args.knob, [v or "unset" for v in values], args.b2, args.b1, args.b2 - args.b1))
    print("libdir %s" % (args.libdir or "(tree default)"))
    print("-" * 96)

    out = {"knob": args.knob, "b1": args.b1, "b2": args.b2, "libdir": args.libdir, "runs": {}}
    for tname, k, n in shapes:
        meta = KNOWN.get((tname, k, n), {})
        print("  %-12s %-12s count=%-4s %8.1f MiB/step"
              % (tname, "%dx%d" % (k, n), meta.get("count", "?"),
                 meta.get("bytes_per_step", 0) / 1048576))
        base = None
        for val in values:
            env = {args.knob: val} if val is not None else {}
            m, fixed, err = run(tname, k, n, env, args.b1, args.b2, 1500, args.libdir)
            label = val or "unset"
            if m is None:
                print("     NSG=%-6s FAILED %s" % (label, err))
                out["runs"].setdefault("%s_%dx%d" % (tname, k, n), {})[str(label)] = {"error": err}
                continue
            gbs = (meta.get("bytes_per_step", 0) / meta.get("count", 1)) / (m * 1e-6) / (1024 ** 3)
            ms = m * meta.get("count", 1) / 1000.0 if meta else 0.0
            if base is None:
                base = ms
                rel = 0.0
            else:
                rel = 100.0 * (base - ms) / base if base else 0.0
            print("     NSG=%-6s marginal %8.2f us  %6.1f GB/s (%4.1f%% of peak)  "
                  "%6.2f ms/step  vs unset %+6.2f%%   [fixed %.0f us]"
                  % (label, m, gbs, 100 * gbs / PEAK_GB_S if gbs else 0, ms, rel, fixed))
            sys.stdout.flush()
            out["runs"].setdefault("%s_%dx%d" % (tname, k, n), {})[str(label)] = {
                "marginal_us": m, "fixed_us": fixed, "gb_s": gbs, "ms_per_step": ms, "rel_pct": rel}
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
