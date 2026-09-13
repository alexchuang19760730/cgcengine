#!/usr/bin/env python3
"""MTP head provenance gate: prove a GGUF carries the head that belongs to its base.

WHY THIS EXISTS
---------------
MTP accept rate is a property of a (base, head) PAIR, not of a head alone. The measured
proof lives in the Edge0 repo's `merge_mtp_head_edge0.py`: transplanting the donor Nail
head onto the Edge0 base dropped accept to **0.81%**, while the same head on its own base
reaches **52.33%**. The two Edge0 MTP artifacts on disk make that failure invisible to
every cheap check, because their blk.40 type sets are IDENTICAL:

    iq4_xs=7  f32=7  bf16=2  q2_K=2  q3_K=1  q8_0=1     (303.0 MiB each)

Same file size, same tensor count, same types. So "we always use the official head" cannot
be verified by reading a type table. It needs a *recorded identity* plus a gate that
refuses to run when the identity moved. This is that gate.

WHAT IT READS
-------------
The head is every tensor of the highest-numbered block (`blk.<block_count>.*` -- the MTP
layer) plus `output.weight` (the lm_head the draft and the target share; on Qwen3.6 GGUF
this is where the head's `Q6_K` pin lives, while `blk.40.*` itself is dense-IQ4_XS with
byte-copied experts).

Identity is a sha256 over `name / ggml type / nbytes / sha256(raw tensor bytes)` for each
head tensor, so it moves when ANY tensor of the head changes -- including a change that
preserves size AND type. That is the whole point: the 0.81% graft was exactly that kind of
change.

COMMANDS
--------
    fingerprint --gguf G [--out J]     record the head's identity (a sidecar)
    compare A B [--sigma]              what differs between two artifacts
    check --gguf G --expect J          PASS/FAIL against a recorded sidecar

`compare --sigma` also decodes the head's LOSSLESS tensors (BF16/F16/F32) and reports
    rel_l2       = ||a - b|| / ||a||
    d_over_sigma = RMS(a - b) / std(a)
Two storage types with the SAME trained weights land at ~0.005 and below on BF16; the
Nail-donor vs Edge0-own router measures 0.093 -- the signature of different trained
weights, which is why the donor head produced drafts the Edge0 base never accepts.

USAGE
-----
    python3 scripts/check/mtp_head_identity.py fingerprint \\
        --gguf models/gguf/Edge0-35B-Q4_0-MTP.gguf \\
        --out  models/gguf/Edge0-35B-Q4_0-MTP.mtphead.json
    python3 scripts/check/mtp_head_identity.py compare \\
        models/gguf/Edge0-35B-Q4_0-MTP.gguf models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf --sigma
    python3 scripts/check/mtp_head_identity.py check \\
        --gguf models/gguf/Edge0-35B-Q4_0-MTP.gguf \\
        --expect models/gguf/Edge0-35B-Q4_0-MTP.mtphead.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402
import numpy as np  # noqa: E402

# The lm_head is part of the head's identity: on Qwen3.6 GGUF the MTP head's `Q6_K` pin is
# on `output.weight`, and the draft shares it with the target, so a change there changes
# the accept rate just as much as a change in the block's own weights.
SHARED_LM_HEAD = "output.weight"

# Types we can decode exactly. Everything else (iq4_xs, q2_K, q3_K, q8_0 ...) is hashed as
# raw bytes only -- a quantised comparison would need a dequantiser and would confuse
# "different precision" with "different weights", which is the distinction this gate exists
# to make.
LOSSLESS = ("BF16", "F16", "F32")


def open_reader(path: Path) -> gguf.GGUFReader:
    return gguf.GGUFReader(str(path))


def field_value(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


def head_layer(reader) -> int:
    """Which block is the MTP head.

    Found SEMANTICALLY: the MTP layer is the block that carries the `nextn.*` tensors
    (`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`). That is a
    property of the head itself rather than of the architecture's arithmetic, and it is
    correct even when the metadata counts the MTP layer inside `block_count`.

    This matters here: on these Qwen3.6/Edge0 artifacts `qwen35moe.block_count` is **41**
    (blk.0..40 with the MTP layer included), so the naive `head = block_count` walks off the
    end and silently matches nothing but the shared lm_head -- i.e. a gate that always
    passes. Caught in the first run; kept as the documented reason for this rule.
    """
    layers = sorted({int(t.name.split(".")[1])
                     for t in reader.tensors
                     if t.name.startswith("blk.") and ".nextn." in t.name})
    if layers:
        return layers[-1]
    for k, f in reader.fields.items():
        if k.endswith(".block_count"):
            return int(f.contents()[0]) - 1
    raise SystemExit("mtp-head: no blk.<n>.nextn.* tensors and no *.block_count metadata")


def raw_bytes(t) -> bytes:
    """The tensor's stored bytes, exactly as they sit in the file.

    Hashed rather than interpreted: the identity must move on any change to the stored
    representation, and must NOT move because a reader normalised something on load.
    """
    return np.asarray(t.data).tobytes()


def head_tensors(reader, layer: int):
    prefix = f"blk.{layer}."
    out = [t for t in reader.tensors if t.name.startswith(prefix)]
    shared = [t for t in reader.tensors if t.name == SHARED_LM_HEAD]
    return sorted(out + shared, key=lambda t: t.name)


def to_f32(t):
    ty = t.tensor_type.name
    b = raw_bytes(t)
    if ty == "BF16":
        u = np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float32)
    if ty == "F16":
        return np.frombuffer(b, dtype="<f2").astype(np.float32)
    if ty == "F32":
        return np.frombuffer(b, dtype="<f4").astype(np.float32)
    return None


def degenerate(t) -> str | None:
    """Is this lossless head tensor numerically empty?

    A head can be byte-perfect in type/size and still be dead. The measured case: the Edge0
    own-head artifact stores `blk.40.ffn_gate_inp.weight` (the MTP head's expert ROUTER) at
    min=-3.3e-09 / max=+9.3e-10 / std=7.7e-12 while the other two artifacts carry the real
    router (std=0.0096, range +-0.14). A zero router makes every expert tie, so the draft
    stops following the target and accept collapses -- 0.81% measured. Nothing in the file
    size, the tensor count or the type table shows it.
    """
    f = to_f32(t)
    if f is None:
        return None
    peak = float(np.max(np.abs(f))) if f.size else 0.0
    if peak == 0.0:
        return "all-zero"
    # 1e-6 is deliberately far below any real weight scale (the smallest real head tensor
    # here peaks at 0.0089) and far above BF16 denormal dust (1e-9), so it separates
    # "collapsed by a conversion bug" from "legitimately small" without a tuned threshold.
    if peak < 1e-6:
        return f"collapsed (peak |v| = {peak:.3g})"
    return None


def fingerprint(path: Path) -> dict:
    r = open_reader(path)
    layer = head_layer(r)
    tensors = head_tensors(r, layer)
    if not tensors:
        raise SystemExit(f"mtp-head: {path} has no blk.{layer}.* -- it carries no MTP head")
    # A head that resolves to the shared lm_head alone means the layer rule found the wrong
    # block; fail loudly rather than emit an identity that can never move.
    if all(t.name == SHARED_LM_HEAD for t in tensors):
        raise SystemExit(f"mtp-head: blk.{layer}.* matched nothing -- refusing to fingerprint "
                         "the lm_head alone (that gate could never fail)")

    recs, dead = {}, {}
    for t in tensors:
        b = raw_bytes(t)
        recs[t.name] = {
            "type": t.tensor_type.name,
            "shape": [int(x) for x in t.shape],
            "nbytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
        why = degenerate(t)
        if why:
            dead[t.name] = why

    lines = [f"{n}|{v['type']}|{v['nbytes']}|{v['sha256']}" for n, v in sorted(recs.items())]
    identity = hashlib.sha256("\n".join(lines).encode()).hexdigest()

    types: dict[str, int] = {}
    for v in recs.values():
        types[v["type"]] = types.get(v["type"], 0) + 1

    return {
        "schema": "cgc.mtp_head_identity/1",
        "gguf": str(path),
        "gguf_bytes": path.stat().st_size,
        "block_count": field_value(r, f"{field_value(r, 'general.architecture', '')}.block_count"),
        "head_layer": layer,
        "n_tensors": len(recs),
        "head_bytes": sum(v["nbytes"] for v in recs.values()),
        "types": dict(sorted(types.items())),
        "degenerate": dead,
        "identity": identity,
        "tensors": recs,
    }


def load_fp(spec: str) -> dict:
    """Accept either a .gguf or a previously written sidecar."""
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"mtp-head: no such file: {p}")
    if p.suffix == ".json":
        fp = json.loads(p.read_text())
        if fp.get("schema") != "cgc.mtp_head_identity/1":
            raise SystemExit(f"mtp-head: {p} is not a head sidecar (schema={fp.get('schema')!r})")
        # The sidecar's tensor dict is authoritative; recompute nothing from disk so a
        # check against a recorded identity stays cheap on an 18 GB artifact.
        return fp
    return fingerprint(p)


def cmd_fingerprint(args) -> int:
    fp = fingerprint(Path(args.gguf))
    text = json.dumps(fp, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    if args.print or not args.out:
        print(text)
    print(f"\nidentity {fp['identity']}", file=sys.stderr)
    print(f"head     blk.{fp['head_layer']}.* + {SHARED_LM_HEAD}  "
          f"{fp['n_tensors']} tensors, {fp['head_bytes'] / 2**20:.1f} MiB", file=sys.stderr)
    print(f"types    {fp['types']}", file=sys.stderr)
    if fp["degenerate"]:
        print(f"DEAD     {fp['degenerate']}", file=sys.stderr)
        print("         a collapsed head tensor drafts tokens the base cannot accept", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args) -> int:
    a, b = load_fp(args.a), load_fp(args.b)
    same = a["identity"] == b["identity"]
    print(f"A {a['gguf']}")
    print(f"  identity {a['identity']}  head=blk.{a['head_layer']}  "
          f"{a['n_tensors']} tensors  {a['head_bytes'] / 2**20:.1f} MiB  {a['types']}")
    print(f"B {b['gguf']}")
    print(f"  identity {b['identity']}  head=blk.{b['head_layer']}  "
          f"{b['n_tensors']} tensors  {b['head_bytes'] / 2**20:.1f} MiB  {b['types']}")
    print()
    print(f"identity: {'SAME' if same else 'DIFFERENT'}")

    if same:
        return 0

    names = sorted(set(a["tensors"]) | set(b["tensors"]))
    diff_type, diff_bytes, diff_only_a, diff_only_b = [], [], [], []
    for n in names:
        ra, rb = a["tensors"].get(n), b["tensors"].get(n)
        if ra is None:
            diff_only_b.append(n)
            continue
        if rb is None:
            diff_only_a.append(n)
            continue
        if ra["sha256"] == rb["sha256"]:
            continue
        if ra["type"] != rb["type"] or ra["nbytes"] != rb["nbytes"]:
            diff_type.append(n)
        else:
            diff_bytes.append(n)
    print(f"\nsame name+type+size, different bytes : {len(diff_bytes)}")
    print(f"different type or size               : {len(diff_type)}")
    print(f"present only in A                    : {len(diff_only_a)}")
    print(f"present only in B                    : {len(diff_only_b)}")
    for n in diff_type[:20]:
        ra, rb = a["tensors"][n], b["tensors"][n]
        print(f"  [type] {n}: A {ra['type']}/{ra['nbytes']}  B {rb['type']}/{rb['nbytes']}")
    for n in diff_only_a[:10]:
        print(f"  [only A] {n}")
    for n in diff_only_b[:10]:
        print(f"  [only B] {n}")

    if args.sigma:
        print("\nlossless-tensor weight comparison (decoded exactly):")
        print(f"  {'tensor':<42} {'type':<6} {'rel_l2':>10} {'d/sigma':>10}")
        worst = 0.0
        n_cmp = 0
        for n in diff_bytes + diff_type:
            va, vb = a["tensors"][n], b["tensors"][n]
            if va["type"] not in LOSSLESS or vb["type"] != va["type"]:
                continue
            ra, rb = _tensor_by_name(a, n), _tensor_by_name(b, n)
            if ra is None or rb is None:
                continue
            fa, fb = to_f32(ra), to_f32(rb)
            if fa is None or fb is None or fa.shape != fb.shape:
                continue
            d = fa - fb
            rel = float(np.linalg.norm(d) / (np.linalg.norm(fa) + 1e-30))
            sd = float(np.sqrt(np.mean(d ** 2)) / (fa.std() + 1e-30))
            worst = max(worst, sd)
            n_cmp += 1
            print(f"  {n:<42} {va['type']:<6} {rel:10.5f} {sd:10.5f}")
        if n_cmp:
            print(f"\n  worst d/sigma = {worst:.4f}")
            print("  reading: <=0.005 -> same trained weights at a different precision; "
                  ">=0.05 -> different trained weights (do not graft)")

    return 1


def _tensor_by_name(fp, name):
    """Re-open the artifact for --sigma. Only the few lossless head tensors are read."""
    if fp.get("_reader") is None:
        fp["_reader"] = open_reader(Path(fp["gguf"]))
    for t in fp["_reader"].tensors:
        if t.name == name:
            return t
    return None


def cmd_check(args) -> int:
    exp = json.loads(Path(args.expect).read_text())
    got = load_fp(args.gguf)
    if got["degenerate"]:
        print(f"FAIL  the head is present but NUMERICALLY DEAD: {got['degenerate']}")
        print("      Identity matching cannot see this -- the bytes are there, the values are not. "
              "A collapsed router makes every expert tie, so accept collapses with it.")
        return 1
    ok = exp["identity"] == got["identity"]
    print(f"expect {exp['identity']}  ({Path(args.expect).name})")
    print(f"got    {got['identity']}  ({got['gguf']})")
    if ok:
        print(f"PASS  head blk.{got['head_layer']}.* + {SHARED_LM_HEAD}, "
              f"{got['n_tensors']} tensors, {got['head_bytes'] / 2**20:.1f} MiB, {got['types']}")
        return 0
    print("FAIL  the artifact carries a DIFFERENT head than the recorded one.")
    if exp.get("tensors") and got.get("tensors"):
        moved = [n for n in exp["tensors"]
                 if n in got["tensors"] and exp["tensors"][n]["sha256"] != got["tensors"][n]["sha256"]]
        print(f"      {len(moved)} head tensor(s) changed, e.g. {moved[:5]}")
        print("      MTP accept rate is a property of (base, head). A different head on this "
              "base drafts tokens the base does not accept.")
        print("      If the change was intentional, regenerate the sidecar and re-measure accept.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fingerprint", help="record the head's identity")
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", help="write the sidecar here")
    p.add_argument("--print", action="store_true", help="also print the JSON to stdout")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("compare", help="what differs between two artifacts")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--sigma", action="store_true",
                   help="decode lossless head tensors and report rel_l2 / d-sigma")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("check", help="PASS/FAIL against a recorded sidecar")
    p.add_argument("--gguf", required=True)
    p.add_argument("--expect", required=True)
    p.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
