#!/usr/bin/env python3
"""Price the dense `mul_mat` weight traffic that `mul_mv` re-reads M times, per shape and per
kernel-family eligibility rule. READ-ONLY: parses an existing CGC_MM_DBG log. No GPU, no server.

WHY THIS EXISTS
---------------
`ggml_metal_op_mul_mat` picks its family from M (= ne11) in three gates
(ggml-metal-ops.cpp:2438 `ne11_mm_min = 8`, :2476-2500 the two small-batch type lists, and the
`CGC_MM_BITIDENT` opt-out). Only one of them, `mul_mv`, sets `nr1 = 1`
(ggml-metal-device.cpp:848) -- one src1 row per threadgroup -- so **every weight byte is fetched
M times**. The other two families derive a rows-per-threadgroup from M and read the weight ONCE.

That makes "which calls are eligible for which family" a *priced* question, and the price is
`(M - 1) * weight_bytes` per call. `CGC_MM_DBG=1` already prints M, ne00, ne01 and both operand
types for every call, so the answer needs no new instrumentation and no run -- but hand-arithmetic
over 26k lines is where the mistakes below came from. All of them are now structural:

  * accumulating a byte sum per key and then multiplying by the count again -- produced a
    427 TB "traffic" that looked like a units bug, and was one;
  * using lowercase `q6_k` for the group B literal while `ggml_type_name()` prints `q6_K`
    (the log's own spelling is authoritative) -- silently emptied group B and dropped the LM head;
  * using 0.265625 B/elem for IQ4_XS instead of 17/32 = 0.53125 (136 B per 256-element block),
    which halved the widest quantised tensors.

USAGE
-----
    python3 scripts/check/mul_mv_surface.py Backup/cgc_logs/llama_server_<stamp>.log
    python3 scripts/check/mul_mv_surface.py <log> --gguf models/gguf/<model>.gguf   # name shapes

`--gguf` turns `f32 2048->256` into `ffn_gate_inp.weight`. NOTHING more precise is possible or
wanted: 41 layers share these shapes, so the useful output is the tensor FAMILY (the name with
its `blk.N.` prefix stripped) plus how many layers carry it, never a layer number.
"""
import argparse
import collections
import re
import struct

# ---- the three gates, transcribed from ggml-metal-ops.cpp:2474-2500 -------------------------
# Group A: `ne11 >= 2 && ne11 <= 8` over this type list.
GROUP_A = {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
           "mxfp4", "iq4_nl"}
# Group B: `ne11 >= 4 && ne11 <= 8` over this type list (goes through the q4x4 ext dispatch).
# SPELLING MATTERS: these are `ggml_type_name()` outputs, and the log prints `q6_K`, not `q6_k`.
GROUP_B = {"q2_K", "q3_K", "q4_K", "q5_K", "q6_K"}
# mul_mm needs `ne11 > ne11_mm_min`; everything with ne11 <= 8 that is not in A or B falls to mul_mv.
NE11_MM_MIN = 8

# Bytes per stored element, block overhead included. Unknown -> 0 bytes, and the shape is still
# listed in the call histogram so an unmodelled type is VISIBLE rather than silently free.
BYTES_PER_ELEM = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32,        # 32 elems / 34 B
    "q6_K": 210 / 256,      # 256 elems / 210 B
    "iq4_xs": 17 / 32,      # 256 elems / 136 B  (= 4.25 bits/weight)
    "iq3_s": 110 / 256,
    "iq2_s": 82 / 256,
}
# VOCAB_THRESHOLD: the LM head is the only tensor whose output dimension is the vocabulary
# (248320 here). Its weight is the widest in the file, so its co-occurrence is also the cleanest
# available discriminator between "this pass evaluated logits" (decode/verify) and "it did not"
# (a prompt chunk). See the per-M slice below.
VOCAB_THRESHOLD = 100_000

LINE_RE = re.compile(r"MMDBG (\S+) ne11=(\d+) ne00=(\d+) ne01=(\d+) t0=(\S+) t1=(\S+)")

GGML_TYPE_ID = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
                9: "q8_1", 10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
                15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s",
                20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 24: "iq3_xxs_r4"}


def gguf_families(path):
    """(type_name, ne0, ne1) -> sorted set of tensor-name suffixes, for every >=2D tensor.

    Deliberately a SET of suffixes: 41 layers share every dense shape, so a dict keyed by shape
    would silently keep whichever layer was parsed last (`blk.40.*`, i.e. the MTP block) and
    label the whole family with it."""
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        f.read(4)
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if et == 8:
                    for _ in range(n):
                        rstr()
                else:
                    f.seek(n * size[et], 1)
                return ("array", n)
            return struct.unpack(scalar[t], f.read(size[t]))[0]

        meta = {}
        for _ in range(n_kv):
            k = rstr()
            meta[k] = rval(struct.unpack("<I", f.read(4))[0])

        fams = collections.defaultdict(set)
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<Q", f.read(8))[0]
            if len(dims) >= 2:
                suffix = re.sub(r"^blk\.\d+\.", "", name)
                fams[(GGML_TYPE_ID.get(tt, f"type{tt}"), dims[0], dims[1])].add(suffix)
    return fams, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--min-mib", type=float, default=8.0,
                    help="hide identities whose whole-run weight traffic is below this (MiB)")
    args = ap.parse_args()

    fams, meta = ({}, {})
    if args.gguf:
        fams, meta = gguf_families(args.gguf)
        print(f"GGUF {args.gguf}")
        for k in ("general.name", "qwen35moe.expert_count", "qwen35moe.embedding_length"):
            if k in meta:
                print(f"  {k} = {meta[k]}")
        print()

    # key = (type0, M, ne00, ne01) -> [calls, bytes_per_call, flip_a, flip_b, flip_iq4xs]
    shapes = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0])
    tags = collections.Counter()
    unparsed = 0
    with open(args.log, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("MMDBG"):
                continue
            m = LINE_RE.search(line)
            if m is None:
                unparsed += 1
                continue
            tag, n11, n00, n01 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            t0, t1 = m.group(5), m.group(6)
            tags[tag] += 1
            b = n00 * n01 * BYTES_PER_ELEM.get(t0, 0.0)
            # Both small-batch gates also require src1 == F32 and ne00 % 128 == 0.
            ok = (t1 == "f32" and n00 % 128 == 0)
            v = shapes[(t0, n11, n00, n01)]
            v[0] += 1
            v[1] = b
            v[2] += 1 if (tag == "mul_mv" and ok and t0 in GROUP_A and 2 <= n11 <= NE11_MM_MIN) else 0
            v[3] += 1 if (tag == "mul_mv" and ok and t0 in GROUP_B and 4 <= n11 <= NE11_MM_MIN) else 0
            # IQ4_XS is in NEITHER list today. This column is the (b) proposal, not a fact.
            v[4] += 1 if (tag == "mul_mv" and ok and t0 == "iq4_xs" and 2 <= n11 <= NE11_MM_MIN) else 0

    if unparsed:
        print(f"WARNING: {unparsed} MMDBG lines did not match the pattern; the table is incomplete\n")

    total_mv = sum(v[1] / 2 ** 20 * k[1] * v[0] for k, v in shapes.items()) / 1024

    def saved(idx):
        return sum(([v[2], v[3], v[4]][idx]) * v[1] / 2 ** 20 * (k[1] - 1)
                   for k, v in shapes.items()) / 1024

    print(f"family tags in log : {dict(tags)}")
    print(f"calls              : {sum(v[0] for v in shapes.values())}")
    print(f"weight traffic     : {total_mv:.1f} GiB  (mul_mv: each weight byte read M times)\n")
    # FOOTGUN, made visible: on an arm that ALREADY set CGC_MM_BITIDENT=0 the flipped calls are
    # tagged `small-batch`, not `mul_mv`, so the (a) column necessarily reads 0 -- which is
    # indistinguishable from "the knob did nothing" unless it is said out loud. The counterfactual
    # size of (a) therefore only exists on a CONTROL arm (one whose log is 100% mul_mv).
    sb = tags.get("small-batch", 0)
    if sb:
        print(f"NOTE: {sb} calls are already tagged `small-batch` in this log, so the (a) row below")
        print( "      reads 0 BY CONSTRUCTION (those calls are no longer `mul_mv`). The count that")
        print( "      prices (a) is the CONTROL arm's; the count that CONFIRMS it is this one.\n")
    print(f"  (a) CGC_MM_BITIDENT=0     flips {sum(v[2] for v in shapes.values()):>6} calls"
          f" -> saves {saved(0):7.1f} GiB")
    print(f"      of which group B      flips {sum(v[3] for v in shapes.values()):>6} calls"
          f" -> saves {saved(1):7.1f} GiB")
    print(f"  (b) IQ4_XS -> group A     flips {sum(v[4] for v in shapes.values()):>6} calls"
          f" -> saves {saved(2):7.1f} GiB   [needs a .metal instantiation, not just the gate]")
    print(f"  (a)+(b) union             saves {saved(0) + saved(2):7.1f} GiB"
          f"  ({100 * (saved(0) + saved(2)) / max(total_mv, 1e-9):.1f}% of traffic)")
    if sb:
        print(f"  OBSERVED `small-batch`    {sb:>6} calls   <- the flip itself, as executed")
    print()

    # ---- per-M slice: prompt chunk or decode/verify pass --------------------------------------
    # Discriminator: a pass that evaluates the LM head is a pass that wanted logits -- the last
    # chunk of a prompt, or a speculative verify step. A prompt chunk in the middle of a long
    # prompt does not. This is read off the data rather than assumed from the pool path's batch
    # size, because the pool path also emits partial chunks (2 and 4 both appear in the record).
    lm_shapes = {k for k in shapes if k[3] >= VOCAB_THRESHOLD}
    print("per-M slice (a pass is decode iff the LM head ran in it):")
    print(f"  {'M':>3} {'calls':>7} {'traffic GiB':>12} {'LM head?':>9} {'reading':>9}"
          f" {'(a) GiB':>9} {'(b) GiB':>9}")
    for n11 in sorted({k[1] for k in shapes}):
        calls = sum(v[0] for k, v in shapes.items() if k[1] == n11)
        traf = sum(v[1] / 2 ** 20 * n11 * v[0] for k, v in shapes.items() if k[1] == n11) / 1024
        has_lm = any(k in lm_shapes for k in shapes if k[1] == n11)
        sa = sum(v[2] * v[1] / 2 ** 20 * (n11 - 1) for k, v in shapes.items() if k[1] == n11) / 1024
        sb = sum(v[4] * v[1] / 2 ** 20 * (n11 - 1) for k, v in shapes.items() if k[1] == n11) / 1024
        print(f"  {n11:>3} {calls:>7} {traf:>12.1f} {('yes' if has_lm else 'no'):>9}"
              f" {('decode' if has_lm else 'prefill'):>9} {sa:>9.2f} {sb:>9.2f}")
    print()

    # ---- by identity ---------------------------------------------------------------------------
    def label(t0, n00, n01):
        names = fams.get((t0, n00, n01))
        if not names:
            return f"{t0} {n00}->{n01}"
        nm = "/".join(sorted(names)[:2])
        return f"{nm} x{len(names)}" if len(names) > 1 else nm

    by_id = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0, 0.0]))
    for (t0, n11, n00, n01), v in shapes.items():
        row = by_id[label(t0, n00, n01)][n11]
        row[0] += v[0]
        row[1] += v[2]
        row[2] += v[3]
        row[3] += v[4]
        row[4] = v[1]

    print(f"{'identity':46}{'M=1':>7}{'M=2':>7}{'M=3':>7}{'M=4':>7}{'M=8':>7}{'MiB/call':>10}"
          f"{'flips a/b/iq4':>16}")
    for key in sorted(by_id, key=lambda k: -sum(r[0] * r[4] * n for n, r in by_id[k].items())):
        row = by_id[key]
        if sum(r[0] * r[4] * n for n, r in row.items()) / (1 << 20) < args.min_mib:
            continue
        cells = "".join(f"{row[n][0] if n in row else '-':>7}" for n in (1, 2, 3, 4, 8))
        fa = sum(row[n][1] for n in row)
        fb = sum(row[n][2] for n in row)
        fi = sum(row[n][3] for n in row)
        mib = next(iter(row.values()))[4] / (1 << 20)
        print(f"{key[:45]:46}{cells}{mib:>10.3f}{f'{fa}/{fb}/{fi}':>16}")


if __name__ == "__main__":
    main()
