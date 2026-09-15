#!/usr/bin/env python3
"""Census of ALL-ZERO expert rows in a GGUF, straight from the file.

Why this exists
---------------
The teardown pool-integrity scan (`llama_expert_cache: pool integrity: ... zero-regions=N`)
reports that N resident pool regions hold nothing but zero bytes. It cannot tell you
WHY, and the two possible reasons are opposite in consequence:

    the FILE row is zero too  ->  MODEL-ZERO. The correct matmul really is a multiply
                                  by zeros. The engine is doing the right thing and the
                                  alarm is a property of the checkpoint, not a defect.
    the file row is NON-zero  ->  ENGINE-ZERO. The fill never landed (or landed in the
                                  wrong slot) and the expert's contribution was silently
                                  dropped. This IS a real defect.

This script answers the file-side question once and for all: which (layer, kind, expert)
rows are all-zero in the GGUF itself. Any zero-region alarm whose owner_expert appears
here is MODEL-ZERO by construction and can be closed without another server run.

Two probe widths are reported per zero row, because they mean different things:
  probe    non-zero bytes in the first 4096 B (the width the engine's scan uses)
  full     non-zero bytes in the whole row stride
probe=0 and full=0  -> genuinely dead expert row
probe=0 and full>0  -> the 4 KiB probe is simply too narrow for this quantisation
                       (a probe limitation, not a dead expert)

Usage:
    /opt/homebrew/bin/python3 scripts/check/gguf_dead_expert_census.py
    /opt/homebrew/bin/python3 scripts/check/gguf_dead_expert_census.py --layers=0,1
    /opt/homebrew/bin/python3 scripts/check/gguf_dead_expert_census.py --model=PATH

Exit codes: 0 = census completed, 1 = model not found / malformed, 2 = usage.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(
    ROOT, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")

TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(?P<kind>gate|up|down)_exps\.weight$")
KIND_ORDER = {"gate": 0, "up": 1, "down": 2}

PROBE_BYTES = 4096


def nz_count(buf: bytes) -> int:
    return sum(1 for b in buf if b)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices to restrict to (default: all)")
    ap.add_argument("--probe", type=int, default=PROBE_BYTES)
    ap.add_argument("--json", default=None, help="also write the census as JSON here")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError:
        print("error: the `gguf` python package is required (pip install gguf)",
              file=sys.stderr)
        return 2

    only = None
    if args.layers:
        only = {int(x) for x in args.layers.split(",") if x.strip() != ""}

    reader = GGUFReader(args.model)
    print(f"model  : {args.model}")
    print(f"probe  : first {args.probe} B/row, then full-row recheck on zero rows")

    rows = []          # (layer, kind, kind_idx, expert, n_experts, stride, nz_probe, nz_full)
    fh = open(args.model, "rb")
    try:
        for t in reader.tensors:
            m = TENSOR_RE.match(t.name)
            if m is None:
                continue
            layer = int(m.group(1))
            kind = m.group("kind")
            if only is not None and layer not in only:
                continue
            ne = [int(x) for x in t.shape]
            if len(ne) < 3:
                continue
            n_experts = ne[2]
            stride = int(t.n_bytes) // n_experts if n_experts else 0
            if stride <= 0:
                continue
            base = int(t.data_offset)
            probe = min(stride, args.probe)
            for e in range(n_experts):
                off = base + e * stride
                fh.seek(off)
                p = fh.read(probe)
                nz_p = nz_count(p)
                if nz_p != 0:
                    continue
                fh.seek(off)
                f = fh.read(stride)
                nz_f = nz_count(f)
                rows.append((layer, kind, KIND_ORDER[kind], e, n_experts,
                             stride, nz_p, nz_f))
    finally:
        fh.close()

    total_rows = 0
    for t in reader.tensors:
        if TENSOR_RE.match(t.name):
            ne = [int(x) for x in t.shape]
            if len(ne) >= 3 and (only is None or
                                 int(TENSOR_RE.match(t.name).group(1)) in only):
                total_rows += ne[2]

    print(f"\nrows scanned : {total_rows}  (gate/up/down exps rows, all experts)")
    print(f"all-zero rows: {len(rows)}")
    if rows:
        print(f"\n{'layer':>5s} {'kind':<5s} {'kindidx':>7s} {'expert':>6s} "
              f"{'n_experts':>9s} {'stride':>8s} {'nz_probe':>8s} {'nz_full':>8s}  verdict")
        print("-" * 78)
        for layer, kind, ki, e, nx, stride, nz_p, nz_f in rows:
            verdict = "DEAD-ROW" if nz_f == 0 else "PROBE-TOO-NARROW"
            print(f"{layer:>5d} {kind:<5s} {ki:>7d} {e:>6d} {nx:>9d} {stride:>8d} "
                  f"{nz_p:>8d} {nz_f:>8d}  {verdict}")

        print("\nby layer:")
        per_layer = {}
        for layer, kind, ki, e, *_ in rows:
            per_layer.setdefault(layer, []).append((kind, ki, e))
        for layer in sorted(per_layer):
            items = per_layer[layer]
            kinds = sorted({k for k, _, _ in items})
            experts = sorted({e for _, _, e in items})
            print(f"  layer {layer:>3d}: {len(items)} zero rows  kinds={kinds}  "
                  f"experts={experts}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as w:
            json.dump({
                "model": args.model,
                "probe_bytes": args.probe,
                "rows_scanned": total_rows,
                "zero_rows": [
                    {"layer": l, "kind": k, "kind_idx": ki, "expert": e,
                     "n_experts": nx, "stride": st,
                     "nz_probe": np_, "nz_full": nf,
                     "verdict": "DEAD-ROW" if nf == 0 else "PROBE-TOO-NARROW"}
                    for l, k, ki, e, nx, st, np_, nf in rows
                ],
            }, w, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.json}")

    print("\nuse: any pool-integrity zero-region whose owner_expert is in the "
          "DEAD-ROW list above is MODEL-ZERO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
