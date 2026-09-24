#!/usr/bin/env python3
"""Per-shape roofline for the MoE expert projections: ceiling vs measured kernel.

Why this exists
---------------
"Can we sweep the ceiling for the shapes we want, then build the kernel?" needs two
kinds of number, and they come from different places:

  * the CEILING is arithmetic -- bytes are fixed by (K, N, top-k, bits/weight). Nothing
    to measure.
  * the ACHIEVED number is a measurement of the real compiled kernel.

`test-backend-ops perf` is the measurement half (per-op, per-type, per-shape, real
pipelines), and it already carries qwen36 cases. `scripts/check/m1_harness.py` is not:
its timing path shells out to llama-bench and regexes a t/s column, so it has no shape
isolation and cannot produce a per-shape number at all.

What this prints: bytes/token, the kernel's µs at our exact shape, the achieved GB/s,
that as a fraction of device peak, and the end-to-end value of a kernel win
(X = share of the step, Y = kernel headroom -> X x Y).

It refuses rather than guesses: an unknown quant type, or a (type, shape) with no
measurement, is a hard error (exit 1). A roofline table with a silently substituted
number is worse than no table.

Usage
-----
    # capture (real kernels, ~90 s; see note on provenance below)
    src/llama.cpp/build/bin/test-backend-ops perf -o MUL_MAT_ID -p n_mats=256 > cap.txt 2>&1

    python3 scripts/check/shape_roofline.py --capture cap.txt \
        --gguf models/gguf/<carrier>.gguf --gpu-busy-ms 97.90 --selftest

Provenance caveat: `test-backend-ops` links libggml-metal.dylib, so the numbers belong
to whatever dylib is on disk -- not to the sources in the working tree. Record the dylib
md5 with the capture (the .txt starts with the engine's device banner, and the caller
should stamp md5 separately).
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

# bytes/weight. Only types we can name honestly; anything else is refused.
BPW = {
    "iq2_s": 82 / 256, "iq2_xs": 74 / 256, "iq2_xxs": 66 / 256,
    "iq3_s": 110 / 256, "iq3_xxs": 98 / 256,
    "iq4_xs": 136 / 256, "iq4_nl": 18 / 32,
    "q2_K": 84 / 256, "q3_K": 110 / 256, "q4_K": 144 / 256,
    "q5_K": 176 / 256, "q6_K": 210 / 256,
    "q8_0": 34 / 32, "f16": 2.0, "f32": 4.0, "bf16": 2.0, "q4_0": 18 / 32,
}

# GGML type enum -> name, for the handful we care about.
GGML_TYPE = {0: "f32", 1: "f16", 2: "q4_0", 8: "q8_0", 10: "q2_K", 11: "q3_K",
             12: "q4_K", 13: "q5_K", 14: "q6_K", 21: "iq3_s", 22: "iq2_s",
             23: "iq4_xs", 30: "bf16"}

CASE = re.compile(
    r"MUL_MAT_ID\(type_a=(\w+),type_b=\w+,n_mats=(\d+),n_used=(\d+),b=\d+,"
    r"m=(\d+),n=(\d+),k=(\d+)\)")
US = re.compile(r"([\d.]+)\s+us/run")


# ---- carrier: which type+shape each expert projection actually has ----------

def gguf_tensor_table(path):
    """(name -> (dims, type_id)) + the MoE KVs. Header only; no tensor data read."""
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        sys.exit("FATAL: not a GGUF")
    struct.unpack("<I", f.read(4))          # version
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def skip(vt):
        if vt == 8:
            f.read(struct.unpack("<Q", f.read(8))[0])
        elif vt == 9:
            et = struct.unpack("<I", f.read(4))[0]
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                skip(et)
        elif vt in (0, 1, 7):
            f.read(1)
        elif vt in (2, 3):
            f.read(2)
        elif vt in (4, 5, 6):
            f.read(4)
        elif vt in (10, 11, 12):
            f.read(8)
        else:
            sys.exit(f"FATAL: unknown GGUF value type {vt} - refusing to guess")

    kv = {}
    for _ in range(n_kv):
        kl = struct.unpack("<Q", f.read(8))[0]
        key = f.read(kl).decode()
        vt = struct.unpack("<I", f.read(4))[0]
        if vt == 9:                          # arrays: we only need scalars
            skip(vt)
            continue
        if vt == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            kv[key] = f.read(n).decode("utf-8", "replace")
            continue
        if vt in (0, 1):
            kv[key] = struct.unpack("<B", f.read(1))[0]
        elif vt in (2, 3):
            kv[key] = struct.unpack("<h", f.read(2))[0]
        elif vt in (4, 5):
            kv[key] = struct.unpack("<i", f.read(4))[0]
        elif vt == 6:
            kv[key] = struct.unpack("<f", f.read(4))[0]
        elif vt == 7:
            kv[key] = bool(struct.unpack("<B", f.read(1))[0])
        elif vt in (10, 11):
            kv[key] = struct.unpack("<q", f.read(8))[0]
        elif vt == 12:
            kv[key] = struct.unpack("<d", f.read(8))[0]
        else:
            skip(vt)

    tensors = {}
    for _ in range(n_tensors):
        nl = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nl).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype = struct.unpack("<I", f.read(4))[0]
        f.read(8)                            # offset
        tensors[name] = (dims, ttype)

    def find(k):
        for kk, vv in kv.items():
            if kk.split(".")[-1] == k:
                return vv
        return None

    return tensors, {
        "block_count": find("block_count"),
        "expert_count": find("expert_count"),
        "expert_used_count": find("expert_used_count"),
        "embedding_length": find("embedding_length"),
        "expert_feed_forward_length": find("expert_feed_forward_length"),
    }


def carrier_projections(gguf):
    """-> {proj: {'type', 'K', 'N', 'layers'}} for the stacked expert tensors.

    Takes the dominant (most common) type per projection: our carriers are mixed
    (37 IQ3_S + 3 IQ4_XS + 1 Q3_K for down), and the dominant type is what a
    per-shape measurement can represent. The minority layers are reported so the
    approximation is visible rather than hidden.
    """
    tensors, meta = gguf_tensor_table(gguf)
    out = {}
    for proj in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        seen = {}
        for name, (dims, ttype) in tensors.items():
            if not name.startswith("blk.") or not name.endswith(proj + ".weight"):
                continue
            if len(dims) != 3:
                sys.exit(f"FATAL: {proj} is not a 3D stacked tensor in {gguf}")
            seen.setdefault((tuple(dims), GGML_TYPE.get(ttype, f"T{ttype}")), 0)
            seen[(tuple(dims), GGML_TYPE.get(ttype, f"T{ttype}"))] += 1
        if not seen:
            sys.exit(f"FATAL: no {proj} tensors in {gguf}")
        (dims, tname), n = max(seen.items(), key=lambda kv: kv[1])
        # stacked expert tensor is [K, N, n_experts] in ggml order
        K, N, n_ex = dims
        out[proj] = {"type": tname, "K": K, "N": N, "layers": n,
                     "variants": {f"{t} {d}": c for (d, t), c in seen.items() if c != n}}
    return out, meta


# ---- the measurement half --------------------------------------------------

def parse_capture(path):
    """-> {(type, m, k, n_mats, n_used, n): us}. Requires the 'us/run' line."""
    return parse_capture_text(Path(path).read_text(errors="replace"))


def parse_capture_text(text):
    """Same, from text. The 'us/run' number is on a later line than the case name,
    and a case whose number never appears is dropped rather than guessed."""
    lines = text.splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = CASE.search(line)
        if not m:
            continue
        t, n_mats, n_used, mm, nn, kk = m.groups()
        us = None
        for j in range(i, min(i + 6, len(lines))):     # us/run is on a later line
            u = US.search(lines[j])
            if u:
                us = float(u.group(1))
                break
        if us is None:
            continue
        out[(t, int(mm), int(kk), int(n_mats), int(n_used), int(nn))] = us
    return out


def roofline(meas, carrier, n_used, peak_gbs, dense_mb, gpu_busy_ms, step_ms,
             proxies=None):
    """per-projection rows. `proxies` maps a missing type to a measured stand-in.

    A proxy is a MODEL, not a measurement: it assumes the same achieved GB/s, so the
    µs is scaled by the bytes/weight ratio. The row is tagged PROXY and must not be
    quoted as measured.
    """
    proxies = proxies or {}
    rows = []
    for proj, c in carrier.items():
        key = (c["type"], c["N"], c["K"], 256, n_used, 1)
        proxy = None
        if key not in meas:
            src = proxies.get(c["type"])
            if src:
                skey = (src, c["N"], c["K"], 256, n_used, 1)
                if skey in meas:
                    proxy = (src, meas[skey] * BPW[c["type"]] / BPW[src])
            if proxy is None:
                sys.exit(
                    f"FATAL: no measurement for {proj} = {c['type']} "
                    f"(m={c['N']},k={c['K']},n_used={n_used},n=1) in the capture.\n"
                    f"       Add it to tests/test-backend-ops.cpp (one emplace_back) and\n"
                    f"       rebuild -- do NOT let this script invent a number.\n"
                    f"       (--proxy {c['type']}=<measured type> is available but tags\n"
                    f"        the row as modelled, so the total is not a measurement.)")
        bpw = BPW.get(c["type"])
        if bpw is None:
            sys.exit(f"FATAL: no bytes/weight for type '{c['type']}' - refusing")
        bytes_expert = c["K"] * c["N"] * bpw
        bytes_layer = bytes_expert * n_used
        us = proxy[1] if proxy else meas[key]
        rows.append({
            "proxy_from": proxy[0] if proxy else None,
            "proj": proj, "type": c["type"], "K": c["K"], "N": c["N"],
            "layers": c["layers"], "bpw": bpw,
            "mb_per_token": bytes_layer * c["layers"] / 1e6,
            "us_per_layer": us,
            "gb_s": bytes_layer / us / 1e3,
            "pct_peak": bytes_layer / us / 1e3 / peak_gbs * 100,
            "ideal_us": bytes_layer / (peak_gbs * 1e3),   # bytes / (GB/s) -> us
            "variants": c["variants"],
        })
    return rows


# ---- selftest ---------------------------------------------------------------

CAP = (
    "  MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048): \n"
    "               17883 runs -    55.62 us/run -  16.78 MFLOP/run - 301.62 GFLOPS\n"
    "  MUL_MAT_ID(type_a=q2_K,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048): \n"
    "               29805 runs -    41.09 us/run -  16.78 MFLOP/run - 408.33 GFLOPS\n"
    "  MUL_MAT_ID(type_a=iq3_xxs,type_b=f32,n_mats=256,n_used=8,b=0,m=2048,n=1,k=512): \n"
    "               11922 runs -    88.60 us/run -  16.78 MFLOP/run - 189.36 GFLOPS\n"
)

GATE = {"ffn_gate_exps": {"type": "iq2_s", "K": 2048, "N": 512, "layers": 39,
                         "variants": {}}}
DOWN = {"ffn_down_exps": {"type": "iq3_s", "K": 512, "N": 2048, "layers": 37,
                         "variants": {}}}
BADT = {"ffn_gate_exps": {"type": "iq9_z", "K": 2048, "N": 512, "layers": 39,
                         "variants": {}}}


def _selftest():
    bad = total = 0

    def chk(name, ok, detail=""):
        nonlocal bad, total
        total += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' - ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    def refuses(fn):
        try:
            fn()
            return False
        except SystemExit:
            return True

    meas = parse_capture_text(CAP)
    chk("parse: 3 cases", len(meas) == 3, f"got {len(meas)}")
    chk("parse: us comes from the us/run line",
        meas.get(("iq2_s", 512, 2048, 256, 8, 1)) == 55.62,
        str(meas.get(("iq2_s", 512, 2048, 256, 8, 1))))

    r = roofline(meas, GATE, 8, 120.0, 827.0, 97.9, None)[0]
    chk("math: bytes/token", abs(r["mb_per_token"] - 104.8) < 0.1, f"{r['mb_per_token']:.1f}")
    chk("math: achieved GB/s", abs(r["gb_s"] - 48.3) < 0.2, f"{r['gb_s']:.1f}")
    chk("math: % of peak", abs(r["pct_peak"] - 40.2) < 0.3, f"{r['pct_peak']:.1f}")
    chk("math: ideal us at peak", abs(r["ideal_us"] - 22.38) < 0.05, f"{r['ideal_us']:.2f}")

    chk("refuses: a needed shape is missing",
        refuses(lambda: roofline(meas, DOWN, 8, 120.0, 827.0, 97.9, None)))
    chk("refuses: a type with no bytes/weight",
        refuses(lambda: roofline(meas, BADT, 8, 120.0, 827.0, 97.9, None)))

    rd = roofline(meas, DOWN, 8, 120.0, 827.0, 97.9, None, {"iq3_s": "iq3_xxs"})[0]
    chk("proxy: tagged as modelled", rd["proxy_from"] == "iq3_xxs", str(rd["proxy_from"]))
    chk("proxy: scaled by bytes/weight", abs(rd["us_per_layer"] - 88.60 * 110 / 98) < 0.05,
        f"{rd['us_per_layer']:.2f}")
    chk("proxy: still refuses when the stand-in is missing too",
        refuses(lambda: roofline(meas, DOWN, 8, 120.0, 97.9, 97.9, None, {"iq3_s": "q5_K"})))

    print(f"selftest: {total - bad}/{total} passed")
    return 1 if bad else 0


def n_scaling(meas, n_used, n_mats=256):
    """Per extra token in the verify batch, for every (type, shape) measured at n>1.

    Returns [(key, measurements, per_token, increments, bytes_per_weight)] for groups that
    actually have more than one n. Nothing is inferred where n=1 only.
    """
    groups = {}
    for (t, m, k, nm, nu, n), us in meas.items():
        if nm == n_mats and nu == n_used:
            groups.setdefault((t, m, k), []).append((n, us))
    out = []
    for key, lst in sorted(groups.items()):
        if len(lst) < 2 or BPW.get(key[0]) is None:
            continue
        lst.sort()
        out.append((key, lst, [(n, us, us / n) for n, us in lst],
                    [lst[i][1] - lst[i - 1][1] for i in range(1, len(lst))],
                    BPW[key[0]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", help="test-backend-ops perf output")
    ap.add_argument("--gguf")
    ap.add_argument("--n-used", type=int, default=None,
                    help="top-k routed experts (default: read from the GGUF)")
    ap.add_argument("--peak-gb-s", type=float, default=120.0,
                    help="device RAM peak. 120 is the cited M4 spec; 100.5 is this box's "
                         "MEASURED large-stream value (Q6_K lm_head, 417 MB contiguous)")
    ap.add_argument("--peak-note", default="cited M4 spec, NOT measured on this box",
                    help="where the peak came from; printed verbatim next to the table")
    ap.add_argument("--dense-mb", type=float, default=827.0,
                    help="non-expert RAM traffic per token (docs/CEILING_STACK)")
    ap.add_argument("--gpu-busy-ms", type=float, default=97.90,
                    help="measured GPU busy per decode step (joint capture 09-20)")
    ap.add_argument("--step-ms", type=float, default=None)
    ap.add_argument("--proxy", action="append", default=[], metavar="TYPE=SRC",
                    help="measure SRC at the same shape and model TYPE from it, "
                         "scaling by bytes/weight. Tagged PROXY in the output.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    proxies = dict(p.split("=", 1) for p in args.proxy)

    if args.selftest:
        sys.exit(_selftest())
    if not args.capture or not args.gguf:
        sys.exit("FATAL: --capture and --gguf are required (or use --selftest)")

    carrier, meta = carrier_projections(args.gguf)
    n_used = args.n_used or meta.get("expert_used_count")
    if not n_used:
        sys.exit("FATAL: expert_used_count missing from the GGUF and --n-used not given")
    meas = parse_capture(args.capture)
    if not meas:
        sys.exit(f"FATAL: no MUL_MAT_ID cases parsed from {args.capture}")

    print(f"carrier: {args.gguf}")
    print(f"  layers={meta['block_count']} experts={meta['expert_count']} "
          f"top-k={n_used} d_model={meta['embedding_length']} "
          f"ff={meta['expert_feed_forward_length']}")
    for proj, c in carrier.items():
        v = c["variants"]
        print(f"  {proj:14s} {c['type']:6s} K={c['K']:<5} N={c['N']:<5} "
              f"{c['layers']} layers" + (f"  (+minorities: {v})" if v else ""))

    rows = roofline(meas, carrier, n_used, args.peak_gb_s, args.dense_mb,
                    args.gpu_busy_ms, args.step_ms, proxies)

    print(f"\n## per-token expert traffic and the kernel floor "
          f"(peak={args.peak_gb_s:g} GB/s -- {args.peak_note})")
    print(f"{'projection':14s} {'type':6s} {'MB/tok':>8s} {'us/layer':>9s} "
          f"{'GB/s':>7s} {'%peak':>6s} {'ideal us':>9s}")
    print("  (us/layer = one dispatch at n_used=%d, n=1 token; ideal_us = the same bytes "
          "at peak bandwidth)" % n_used)
    tot_mb = tot_us = 0.0
    modelled = []
    for r in rows:
        tag = r["type"] if not r["proxy_from"] else f"{r['type']}*"
        print(f"{r['proj']:14s} {tag:6s} {r['mb_per_token']:8.1f} "
              f"{r['us_per_layer']:9.2f} {r['gb_s']:7.1f} {r['pct_peak']:5.0f}% "
              f"{r['ideal_us']:9.2f}")
        if r["proxy_from"]:
            modelled.append(f"{r['proj']} {r['type']} <- modelled from "
                            f"{r['proxy_from']} (same GB/s) - NOT measured")
        tot_mb += r["mb_per_token"]
        tot_us += r["us_per_layer"] * r["layers"]
    ms = tot_us / 1e3
    print(f"{'TOTAL':14s} {'':6s} {tot_mb:8.1f} {tot_us:9.1f}   <- us/token, not us/layer"
          f"\n{'':14s} {'':6s} {'':8s}        -> {ms:.2f} ms GPU floor/token "
          f"=> {1000 / ms:.0f} t/s ceiling (MoE GEMV only)")
    for line in modelled:
        print(f"  * {line}")

    expert_share = ms / args.gpu_busy_ms * 100
    print(f"\n## X x Y (what a kernel win is worth end-to-end)")
    print(f"  X  share of the step : {ms:.2f} ms of {args.gpu_busy_ms:.2f} ms GPU busy "
          f"= {expert_share:.1f}%")
    best = {}
    for (t, m, k, nm, nu, n), us in meas.items():
        if (m, k, nm, nu, n) in [((r["N"], r["K"], 256, n_used, 1)) for r in rows]:
            best.setdefault((m, k), []).append((us, t))
    for (m, k), lst in sorted(best.items()):
        lst.sort()
        fastest, ftype = lst[0]
        print(f"  shape m={m},k={k}: fastest measured type = {ftype} ({fastest:.2f} us)"
              + "".join(f" | {t} {us:.2f}" for us, t in lst[1:]))
        cur = [r for r in rows if r["N"] == m and r["K"] == k][0]
        gain = cur["us_per_layer"] / fastest
        if gain > 1.02:
            print(f"    -> {cur['type']} {cur['us_per_layer']:.2f} -> {ftype} {fastest:.2f} "
                  f"= {gain:.2f}x on the kernel "
                  f"=> end-to-end {expert_share * (1 - 1 / gain):.1f}%")

    ns = n_scaling(meas, n_used)
    if ns:
        print("\n## verify batch: does one more drafted token cost less? (the MTP question)")
        print("   bytes are per token (each token carries its own top-k ids), so a FLAT")
        print("   per-token cost means the batch does not amortise the expert reads.")
        for (t, m, k), lst, pt, incr, bpw in ns:
            print(f"  {t} m={m},k={k}:")
            b1 = None
            for (n, us, per) in pt:
                by = n * n_used * m * k * bpw
                b1 = by / n if b1 is None else b1
                print(f"    n={n}  {us:9.2f} us  {per:7.2f} us/token  {by / us / 1e3:6.1f} GB/s"
                      + (f"   +{incr[n - 2]:.2f}" if n > 1 else ""))
            # the honest comparator: mean per-token cost of the batched cases vs n=1's
            multi = sum(us for n, us, _ in pt if n > 1) / sum(n for n, _, _ in pt if n > 1)
            am = 1.0 - multi / pt[0][2]
            print(f"    -> batched mean {multi:.2f} us/token vs n=1's {pt[0][2]:.2f} "
                  f"= {am * 100:+.1f}% "
                  f"=> {'NO amortisation' if am < 0.05 else f'amortised {am * 100:.0f}%'}"
                  f" (marginal {incr[-1]:.2f} us/token)")

    print(f"\n## effective bandwidth in situ")
    total_mb = args.dense_mb + tot_mb
    eff = total_mb / args.gpu_busy_ms
    print(f"  dense {args.dense_mb:.0f} MB + experts {tot_mb:.0f} MB = {total_mb:.0f} MB/token")
    print(f"  {total_mb:.0f} MB / {args.gpu_busy_ms:.2f} ms GPU busy = {eff:.1f} GB/s "
          f"= {eff / args.peak_gb_s * 100:.0f}% of peak")
    print(f"  => at peak the same traffic is {total_mb / args.peak_gb_s:.2f} ms "
          f"=> {1000 / (total_mb / args.peak_gb_s):.0f} t/s (the L4 wall)")
    if args.step_ms:
        print(f"  measured step {args.step_ms:.1f} ms => {1000 / args.step_ms:.2f} t/s "
              f"(per-step bandwidth {total_mb / args.step_ms:.1f} GB/s)")


if __name__ == "__main__":
    main()
