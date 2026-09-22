#!/usr/bin/env python3
"""run_shapes.py -- price the dominant decode shapes through the probe, then TEST
WHETHER THE RESULT CAN CLOSE against the known production step time.

The probe can produce two very different numbers for the same shape:
    sync : one dispatch, wait for completion, repeat   -> single-op latency
    batch: N concurrent ops in one command buffer      -> steady-state throughput
They differ by up to 8x on this machine, so quoting either one alone is a coin
flip. The discriminator is arithmetic that anyone can check:

    priced_step = sum over shapes of (us_per_op x dispatches_per_step)
    compare with the production step time (79.5-97.9 ms)

A branch that prices ABOVE the real step time is charging the GPU for stalls it
never actually pays, and every number derived from it is garbage. This script
prints that verdict for both branches so nobody has to re-litigate it.

SUPERSEDED for quoting purposes (2026-09-22, see docs/DENSE_GEMV_INSTRUMENT_2026-09-22.md):
    the `batch` numbers from THIS script still contain a per-graph fixed cost
    amortised over `--batch` ops (220-660 us per graph here, measured), which
    makes small shapes look ~4x worse than they are. Use sweep_nsg.py's
    MARGINAL model instead. This script is kept for the closure argument: it is
    what proves the single-dispatch (`sync`) numbers cannot be quoted.

Usage:
    shape_probe/run_shapes.py --limit 20
    shape_probe/run_shapes.py --json Backup/phase_decomp/L3/shape_probe/pricing.json
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/check"))
sys.path.insert(0, str(ROOT / "Backup/phase_decomp/L3"))

PROBE = HERE / "cgc_shape_probe"

# MEASURED peak for this machine (Mac16,12 / M4 / 8 GPU), 6 threads. Not the 120
# GB/s spec number -- that one has never been observed here.
PEAK_GB_S = 108.8

# Production step time from unpolluted llama-bench runs (see MEMORY_PERF.md).
STEP_MS_LO, STEP_MS_HI = 79.5, 97.9

TYPE_MAP = {"IQ4_XS": "iq4_xs", "Q6_K": "q6_k", "F32": "f32", "F16": "f16",
            "BF16": "bf16", "IQ2_S": "iq2_s", "IQ3_S": "iq3_s", "IQ3_XXS": "iq3_xxs",
            "Q8_0": "q8_0", "Q4_0": "q4_0", "Q4_K": "q4_k", "Q5_K": "q5_k",
            "Q2_K": "q2_k", "Q3_K": "q3_k", "IQ4_NL": "iq4_nl"}

MIB = 1024 * 1024


def dominant_shapes(limit):
    """Group the model's tensors by (type, dims, role) -> [count, bytes].

    dims arrive in ggml `ne` order: dims[0] = K (reduction), dims[1] = N (rows
    out), dims[2] = expert count when present. 3D tensors are MoE expert banks
    and are priced elsewhere (they go through MUL_MAT_ID, not plain MUL_MAT).
    """
    import dense_bytes_census as cen
    import gguf_pool_geometry as gpg

    _ver, tensors, _ft, meta = gpg.parse_header(cen.MODEL)
    agg = {}
    for name, (tid, dims, _off) in tensors.items():
        nb = gpg.n_bytes(tid, dims)
        cls, _layer = cen.classify(name)
        key = (str(gpg.tname(tid)), tuple(int(d) for d in dims), cls)
        a = agg.setdefault(key, [0, 0])
        a[0] += 1
        a[1] += nb
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    return gpg, meta, rows[:limit]


def measure(tname, k, out_n, rows, mode, batch, iters, pool_mib, distinct=True):
    """One probe process; return its median us/op for `mode`."""
    cmd = [str(PROBE), "--type", tname, "--k", str(k), "--out", str(out_n),
           "--r", str(rows), "--mode", mode, "--batch", str(batch),
           "--iters", str(iters), "--pool-mib", str(pool_mib)]
    if not distinct:
        cmd.append("--shared-weight")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        return None, dt, (p.stderr.strip()[-200:] or "exit %d" % p.returncode)
    for line in p.stderr.splitlines():
        if line.startswith("RESULT") and (mode in line):
            try:
                return float(line.split("median")[1].split("us/op")[0].strip()), dt, ""
            except Exception:
                return None, dt, "parse error: " + line
    return None, dt, "no RESULT for mode=%s -- tail: %s" % (mode, p.stderr[-160:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rows", type=int, default=1, help="1=decode, 4=MTP K=4")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--min-mib", type=float, default=20.0,
                    help="skip shapes contributing less than this MANY MiB PER STEP")
    ap.add_argument("--max-pool-mib", type=int, default=1400)
    ap.add_argument("--classes", default="dense,lm_head",
                    help="which role classes to price (default: dense,lm_head)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not PROBE.exists():
        sys.exit("FATAL: %s missing -- run shape_probe/build.sh first" % PROBE)
    _gpg, meta, rows = dominant_shapes(args.limit)

    print("peak  %.1f GB/s (measured)  |  production step %.1f-%.1f ms  |  rows=%d"
          % (PEAK_GB_S, STEP_MS_LO, STEP_MS_HI, args.rows))
    print("-" * 108)

    out = {"settings": {"peak_gb_s": PEAK_GB_S, "step_ms": [STEP_MS_LO, STEP_MS_HI],
                        "batch": args.batch, "rows": args.rows},
           "shapes": {}}
    priced = {"sync": 0.0, "batch": 0.0}
    covered = 0

    for (tname, dims, cls), (n, nb) in rows:
        if len(dims) != 2:
            continue                      # MoE expert bank -> MUL_MAT_ID, priced elsewhere
        if cls not in args.classes.split(","):
            continue
        gt = TYPE_MAP.get(tname.upper())
        if gt is None:
            print("  SKIP %-8s %-14s (probe has no filler for this type)" % (tname, "x".join(map(str, dims))))
            continue
        per_t = nb / n
        if nb / MIB < args.min_mib:
            continue
        # how many ops worth of distinct weights can we still fit?
        budget = args.max_pool_mib - 256
        b = max(1, min(args.batch, int(budget / (per_t / MIB))))
        if b < 1:
            print("  SKIP %-8s %-14s (one tensor is %.0f MiB -- too big to batch)"
                  % (tname, "x".join(map(str, dims)), per_t / MIB))
            continue
        pool = int(b * (per_t / MIB) + 256)

        res = {}
        for mode in ("sync", "batch"):
            bb = b if mode == "batch" else 1
            pp = pool if mode == "batch" else int(per_t / MIB + 256)
            us, dt, err = measure(gt, dims[0], dims[1], args.rows, mode, bb,
                                  args.iters, pp)
            res[mode] = (us, err, dt)

        if res["sync"][0] is None and res["batch"][0] is None:
            print("  FAIL %-8s %-14s  %s" % (tname, "x".join(map(str, dims)), res["batch"][1]))
            continue

        line = "  %-8s %-14s n=%-4d %7.1f MiB/step" % (
            tname, "x".join(map(str, dims)), n, per_t * n / MIB)
        rec = {"type": tname, "dims": list(dims), "count": n,
               "bytes_per_tensor": per_t, "bytes_per_step": nb, "batch_ops": b}
        for mode in ("sync", "batch"):
            us, err, dt = res[mode]
            if us is None:
                continue
            gbs = (per_t / (us * 1e-6)) / (1024 ** 3)
            ms = us * n / 1000.0          # one dispatch per tensor per step
            priced[mode] += ms
            rec[mode + "_us"] = us
            rec[mode + "_gb_s"] = gbs
            rec[mode + "_ms_per_step"] = ms
            line += "  | %s %7.2f us %6.1f GB/s %5.1f%%pk %6.2f ms" % (
                mode[0].upper(), us, gbs, 100 * gbs / PEAK_GB_S, ms)
        covered += nb
        print(line, flush=True)
        out["shapes"]["%s_%s" % (tname, "x".join(map(str, dims)))] = rec

    print("-" * 108)
    print("priced %d shapes, %.1f MiB/step of dense bytes" % (len(out["shapes"]), covered / MIB))
    best_gbs = max([s["batch_gb_s"] for s in out["shapes"].values() if "batch_gb_s" in s] or [1.0])
    print("best-observed shape throughput: %.1f GB/s (%.0f%% of measured peak)"
          % (best_gbs, 100 * best_gbs / PEAK_GB_S))
    print("-" * 108)

    # ---- the bound nobody was allowed to quote until now -------------------
    # Everything above is a MEASUREMENT. This block is EXTRAPOLATION and is
    # labelled as such: "if this shape ran at X GB/s, the step would cost Y".
    def ms_at(bytes_per_step, gb_s):
        return (bytes_per_step / (gb_s * 1e9)) * 1e3

    print("  %-26s %9s %9s %9s %9s" % ("shape", "now ms", "@best", "@peak", "save ms"))
    save_best = save_peak = 0.0
    for key, s in sorted(out["shapes"].items(),
                         key=lambda kv: -kv[1].get("batch_ms_per_step", 0)):
        now = s.get("batch_ms_per_step", 0.0)
        at_best = ms_at(s["bytes_per_step"], best_gbs)
        at_peak = ms_at(s["bytes_per_step"], PEAK_GB_S)
        save_best += max(0.0, now - at_best)
        save_peak += max(0.0, now - at_peak)
        print("  %-26s %9.2f %9.2f %9.2f %9.2f"
              % (key, now, at_best, at_peak, max(0.0, now - at_best)))
    print("-" * 108)
    for label, sv in (("best-observed %.1f GB/s" % best_gbs, save_best),
                      ("absolute peak %.1f GB/s" % PEAK_GB_S, save_peak)):
        pct = 100.0 * sv / STEP_MS_LO
        print("  headroom if every dense GEMV hit %-24s %.2f ms = %.1f%% of a %.1f ms step -> %s"
              % (label, sv, pct, STEP_MS_LO,
                 "ABOVE 3%% threshold" if pct >= 3.0 else "below 3%% threshold"))
    print("  (EXTRAPOLATION: assumes nothing else in the step gets slower)")
    out["save_ms_at_best"] = save_best
    out["save_ms_at_peak"] = save_peak
    for mode in ("sync", "batch"):
        ms = priced[mode]
        verdict = "CLOSES" if ms <= STEP_MS_HI else "DOES NOT CLOSE"
        print("  %-6s branch prices dense at %7.2f ms/step vs step %.1f-%.1f ms  ->  %s"
              % (mode, ms, STEP_MS_LO, STEP_MS_HI, verdict))
        out[mode + "_ms_per_step"] = ms
        out[mode + "_closure"] = verdict
    out["batched_is_valid"] = (priced["sync"] > STEP_MS_HI) and (priced["batch"] <= STEP_MS_HI)
    if out["batched_is_valid"]:
        print("  => the single-dispatch number is latency, not cost: use the batch branch.")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
