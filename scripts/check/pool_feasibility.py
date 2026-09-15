#!/usr/bin/env python3
"""Decide, BEFORE launching anything, whether a (model, pool size) cell is usable.

Why this exists
---------------
`CGC_POOL_MAX_TOKENS` (the "cap") is not a pool parameter: llama-context.cpp clamps `n_batch`
to it whenever the L4 pool is active, so the cap IS the prefill chunk size. Two consequences
that this predictor turns into a verdict:

  P1  capacity   `cap * top_k <= usable_slots`  (slot 0 is a reserved ZERO slot)
                 usable_slots comes from the loader: every layer's expert tensor is shrunk to
                 `capacity = clamp(pool_bytes / (max_layer * per_slot), 8, 256)`, where
                 `per_slot` is the MAX over *decoder* layers of that layer's per-expert bytes.
  P2  liveness   `cap >= n_keep_tail + 1`, and `n_keep_tail = (n_rs_seq > 0) ? n_rs_seq + 1 : 0`
                 (llama-memory-hybrid.cpp:89, asserted at llama-batch.cpp:609).
                 `n_rs_seq = draft.n_max` when the spec type is draft-mtp/eagle3/dflash/dspark
                 (common.h `need_n_rs_seq`), and `run_server.sh` passes `--spec-draft-n-max 3`
                 with MTP on. So MTP on => cap >= 5; MTP off => n_rs_seq 0 => no floor.

Verdicts
--------
  USABLE          cap_max >= 8, the build's default cap. The cell can run at the SAME cap as
                  any other cell, so a cross-cell comparison is a POOL-SIZE experiment and the
                  pool-layout invariance claim (bit-identical at fixed cap) applies.
  PARTITION-REF   floor <= cap_max < 8. The cell runs, but only at a cap below the default, so
                  comparing it against a cap-8 reference measures the PREFILL STEP PARTITION,
                  not the pool. Its logits can inform partition sensitivity and nothing else.
  UNUSABLE        cap_max < floor. No cap satisfies both P1 and P2: cap big enough to fill the
                  slots violates the liveness floor, and vice versa. The pool path cannot run
                  (measured: the loader aborts, or the union overflows the layer's slots).

This is derived from the model's own GGUF header and the engine's own arithmetic -- no launch,
no GPU, no weights read (only the tensor TABLE is parsed).

Usage
-----
  python3 scripts/check/pool_feasibility.py --pools 2,4,6,8,10
  python3 scripts/check/pool_feasibility.py --pools 4 --models iq3,ornith --md
  python3 scripts/check/pool_feasibility.py --gguf path/to/model.gguf --pools 2,4,6,8 --mtp 0
  python3 scripts/check/pool_feasibility.py --pools 2,4,6,8,10 --json /tmp/feas.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GGUF_PY = os.path.join(ROOT, "src", "llama.cpp", "gguf-py")

# The build default for CGC_POOL_MAX_TOKENS (llama-context.cpp / cgc_pool_max_tokens()).
REF_CAP = 8
# run_server.sh default when MTP is on: --spec-draft-n-max 3  -> n_rs_seq = 3 -> floor 5.
DEFAULT_SPEC_N_MAX = 3
# run_server.sh default when SERVER_LAYER_CAPS is unset (MTP branch).
DEFAULT_LAYER_CAPS = "40-40:256"
# `capacity = clamp(pool_bytes / (max_layer * per_slot), 8, 256)`.
CAP_MIN, CAP_MAX = 8, 256
# The four expert tensor kinds the loader sums per layer (see compute_l4_pool_capacity).
KINDS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps")


def _load_matrix_module():
    """Import knifeedge_matrix so MODELS (paths, mtp, topk) has exactly one definition."""
    path = os.path.join(HERE, "knifeedge_matrix.py")
    spec = importlib.util.spec_from_file_location("km", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    return km


def layer_cap(layer: int, default: int, layer_caps: str | None) -> int:
    """Mirror `cgc_layer_cap` (llama-expert-cache.h): last matching `start-end:cap` wins."""
    if not layer_caps:
        return default
    cur = default
    for entry in layer_caps.replace(";", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            rng, cap = entry.split(":")
            start, end = rng.split("-")
            if int(start) <= layer <= int(end):
                cur = int(cap)
        except ValueError:
            continue
    return cur


def gguf_geometry(path: str):
    """Per-layer per-expert bytes + the layer count, from the GGUF tensor table only."""
    if GGUF_PY not in sys.path:
        sys.path.insert(0, GGUF_PY)
    import gguf  # noqa: PLC0415  (path is fixed up right above)

    reader = gguf.GGUFReader(path)

    def field(key, default=None):
        f = reader.fields.get(key)
        return f.contents() if f is not None else default

    arch = field("general.architecture")
    n_nextn = int(field(f"{arch}.nextn_predict_layers", 0) or 0)
    block_count = int(field(f"{arch}.block_count", 0) or 0)
    n_expert = field(f"{arch}.expert_count")
    topk = field(f"{arch}.expert_used_count")

    per_layer: dict[int, int] = {}
    per_layer_kinds: dict[int, dict[str, int]] = {}
    for t in reader.tensors:
        name = t.name
        if "_exps" not in name or "blk." not in name:
            continue
        try:
            il = int(name.split("blk.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if not any(k in name for k in KINDS):
            continue
        blk, tsize = gguf.GGML_QUANT_SIZES[t.tensor_type]
        ne = list(t.shape)
        if len(ne) < 2:
            continue
        row = (ne[0] // blk) * tsize
        eb = row * ne[1]
        per_layer[il] = per_layer.get(il, 0) + eb
        kind = next((k for k in KINDS if k in name), "?")
        per_layer_kinds.setdefault(il, {})[kind] = eb

    if not per_layer:
        raise RuntimeError(f"{os.path.basename(path)}: no blk.*_exps tensors found")

    max_layer = max(per_layer) + 1
    n_decoder = max_layer - n_nextn if n_nextn < max_layer else max_layer
    # `per_slot` is the MAX over decoder layers only: charging the trunk the MTP head's bytes
    # makes the trunk's pool depend on how the head happens to be quantised (measured harm:
    # a F16 head took the trunk from 118 to 33 slots/layer and produced 585 nil buffers).
    decoder = {il: b for il, b in per_layer.items() if il < n_decoder}
    per_slot = max(decoder.values()) if decoder else max(per_layer.values())
    binding_il = max(decoder, key=lambda i: decoder[i]) if decoder else max(per_layer, key=per_layer.get)
    return {"per_layer": per_layer, "per_layer_kinds": per_layer_kinds, "max_layer": max_layer,
            "n_nextn": n_nextn, "n_decoder": n_decoder, "per_slot": per_slot,
            "binding_il": binding_il, "block_count": block_count,
            "n_expert": n_expert, "topk": topk, "arch": arch}


def evaluate(geom, pool_bytes, topk, mtp, spec_n_max=DEFAULT_SPEC_N_MAX,
             layer_caps: str | None = None) -> dict:
    """Apply P1 + P2 and return the verdict for one (geometry, pool) cell."""
    denom = geom["max_layer"] * geom["per_slot"]
    raw = pool_bytes // denom if denom else CAP_MAX
    capacity = max(CAP_MIN, min(CAP_MAX, int(raw)))

    caps = DEFAULT_LAYER_CAPS if layer_caps is None else layer_caps
    per_layer_slots = [layer_cap(l, capacity, caps) for l in range(geom["max_layer"])]
    min_slots = min(per_layer_slots)
    usable = min_slots - 1                      # slot 0 is the reserved ZERO slot
    cap_max = usable // topk

    # P2: cap >= n_keep_tail + 1, n_keep_tail = n_rs_seq + 1 when n_rs_seq > 0.
    n_rs_seq = spec_n_max if mtp else 0
    floor = (n_rs_seq + 2) if n_rs_seq > 0 else 1

    if cap_max >= REF_CAP:
        verdict, why = "USABLE", (f"cap_max {cap_max} >= default cap {REF_CAP}: runs at the same "
                                  f"cap as every other cell, so a cross-cell diff is pool-only")
    elif cap_max >= floor:
        verdict, why = "PARTITION-REF", (f"cap_max {cap_max} < default {REF_CAP} (but >= floor "
                                         f"{floor}): the cell runs only at a smaller cap, so it "
                                         f"measures partition sensitivity, not pool layout")
    else:
        verdict, why = "UNUSABLE", (f"cap_max {cap_max} < floor {floor}: no cap satisfies P1 and "
                                    f"P2 at once (loader aborts or the union overflows)")

    return {"capacity_slots": capacity, "min_layer_slots": min_slots, "usable_slots": usable,
            "cap_max": cap_max, "floor": floor, "n_rs_seq": n_rs_seq,
            "union_max_at_cap_max": cap_max * topk,
            "union_max_at_ref": REF_CAP * topk,
            "headroom_at_ref": usable - REF_CAP * topk,
            "verdict": verdict, "why": why, "layer_caps": caps,
            "per_slot_bytes": geom["per_slot"], "binding_il": geom["binding_il"],
            "max_layer": geom["max_layer"], "n_nextn": geom["n_nextn"]}


def validate_against_probes(records, result_dir):
    """Cross-check the prediction against the harness' MEASURED cap-probe records.

    A predictor that is never compared to a measurement is just a nicer-looking guess, and the
    failure mode here is quiet: `capacity` depends on the GGUF's per-expert bytes AND on how the
    loader divides the pool, so an engine change moves it without touching this script. Measured
    stale case: `cap_probe_iq3_pool8gb.json` says 106 slots (recorded 2026-09-11) while the
    2026-09-14 logs say 143 -- the prediction (143) is right and the cached record is old, which
    is exactly the kind of thing that should be printed, not silently averaged away.
    """
    import glob
    print()
    print("=" * 118)
    print("VALIDATION vs measured cap-probe records (prediction = slots / usable / cap_max)")
    print("=" * 118)
    print(f"{'model':<22} {'pool':>5} {'predicted':>22} {'measured':>22}  result")
    print("-" * 118)
    n_match = n_bad = n_missing = 0
    for r in records:
        src = os.path.join(result_dir, f"cap_probe_{r['model']}_pool{r['pool_gb']}gb.json")
        if not os.path.exists(src):
            n_missing += 1
            continue
        m = json.load(open(src, encoding="utf-8"))
        ms = m.get("min_layer_slots") or m.get("n_slots")
        if ms is None:
            for t in m.get("tried", []):
                uf = t.get("union_fit") or {}
                if uf.get("n_slots"):
                    ms, mu = uf["n_slots"], uf.get("usable_slots")
                    break
            else:
                n_missing += 1
                continue
            mu = mu if mu is not None else ms - 1
        else:
            mu = ms - 1
        have_cap = m.get("cap")
        pred = f"{r['min_layer_slots']}/{r['usable_slots']}/{r['cap_max']}"
        meas = f"{ms}/{mu}/{have_cap if have_cap is not None else '-'}"
        ok = (ms == r["min_layer_slots"])
        n_match += ok
        n_bad += (not ok)
        note = "MATCH" if ok else "MISMATCH (stale record or engine changed)"
        print(f"{r['model'][:22]:<22} {str(r['pool_gb']) + 'G':>5} {pred:>22} {meas:>22}  {note}")
        if not ok:
            import datetime
            when = datetime.datetime.fromtimestamp(os.path.getmtime(src)).strftime("%Y-%m-%d %H:%M")
            print(f"{'':<22} {'':>5} cached: {src}")
            print(f"{'':<22} {'':>5}         recorded {when}; if that predates the current engine "
                  f"stamp the record is stale, not the prediction")
    print("-" * 118)
    print(f"{n_match} match, {n_bad} mismatch, {n_missing} without a measured record")
    return n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="iq3,ornith,iq4",
                    help="comma-separated keys from knifeedge_matrix.MODELS")
    ap.add_argument("--gguf", action="append", default=[],
                    help="extra GGUF path (repeatable); name is taken from the file stem")
    ap.add_argument("--pools", default="2,4,6,8,10", help="pool sizes in GiB")
    ap.add_argument("--topk", type=int, default=None, help="override expert_used_count")
    ap.add_argument("--mtp", default=None, choices=["0", "1"],
                    help="override MTP (sets the P2 floor); default = per-model from MODELS")
    ap.add_argument("--spec-n-max", type=int, default=DEFAULT_SPEC_N_MAX)
    ap.add_argument("--layer-caps", default=None,
                    help=f"LLAMA_EXPERT_CACHE_LAYER_CAPS value (default {DEFAULT_LAYER_CAPS!r})")
    ap.add_argument("--json", default=None, help="write the full record here")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check every prediction against the measured cap-probe records")
    ap.add_argument("--md", action="store_true", help="emit a markdown table instead of the text one")
    args = ap.parse_args()

    km = _load_matrix_module()
    pools = [int(x) for x in args.pools.split(",") if x.strip()]

    targets = []
    for key in [k for k in args.models.split(",") if k.strip()]:
        if key not in km.MODELS:
            print(f"[skip] {key}: not in MODELS", file=sys.stderr)
            continue
        targets.append((key, km.model_path(key), MODELS_MTP(km, key)))
    for p in args.gguf:
        targets.append((os.path.splitext(os.path.basename(p))[0], p, True))

    rows, records = [], []
    for name, path, mtp_default in targets:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} does not exist", file=sys.stderr)
            continue
        try:
            geom = gguf_geometry(path)
        except Exception as e:  # noqa: BLE001 - report, do not crash the whole table
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        topk = args.topk or geom["topk"] or 8
        mtp = (args.mtp == "1") if args.mtp is not None else mtp_default
        for gb in pools:
            r = evaluate(geom, gb * 1024 ** 3, topk, mtp, args.spec_n_max, args.layer_caps)
            r.update({"model": name, "path": path, "pool_gb": gb, "topk": topk, "mtp": bool(mtp),
                      "n_expert": geom["n_expert"], "block_count": geom["block_count"],
                      "arch": geom["arch"]})
            records.append(r)
            rows.append((name, gb, mtp, topk, r))

        print(f"[geom] {name}: arch={geom['arch']} layers={geom['max_layer']} "
              f"nextn={geom['n_nextn']} experts={geom['n_expert']} topk={topk} "
              f"per_slot={geom['per_slot']:,} B (binding blk.{geom['binding_il']}) "
              f"mtp={'on' if mtp else 'off'}", file=sys.stderr)

    if not rows:
        print("no cells to evaluate", file=sys.stderr)
        return 1

    if args.md:
        print("| model | pool | P1 slots (usable) | cap_max | floor | verdict | why |")
        print("|---|---|---|---|---|---|---|")
        for name, gb, mtp, topk, r in rows:
            badge = {"USABLE": "✅ USABLE", "PARTITION-REF": "⚠️ PARTITION-REF",
                     "UNUSABLE": "❌ UNUSABLE"}[r["verdict"]]
            print(f"| {name}{'' if mtp else ' (MTP off)'} | {gb} GiB | "
                  f"{r['min_layer_slots']} ({r['usable_slots']}) | {r['cap_max']} | "
                  f"{r['floor']} | {badge} | {r['why']} |")
    else:
        print()
        print("=" * 118)
        print("POOL FEASIBILITY (pre-launch, derived from the GGUF header + the engine's own arithmetic)")
        print("=" * 118)
        print(f"{'model':<22} {'pool':>6} {'mtp':>4} {'slots':>6} {'usable':>7} {'cap_max':>8} "
              f"{'floor':>6} {'verdict':>15}")
        print("-" * 118)
        for name, gb, mtp, topk, r in rows:
            print(f"{name[:22]:<22} {str(gb) + 'G':>6} {('on' if mtp else 'off'):>4} "
                  f"{r['min_layer_slots']:>6} {r['usable_slots']:>7} {r['cap_max']:>8} "
                  f"{r['floor']:>6} {r['verdict']:>15}")
        print("-" * 118)
        print("USABLE        cap_max >= 8: runs at the default cap -> cross-cell diff is pool-only")
        print("PARTITION-REF floor <= cap_max < 8: runs, but only at a smaller cap -> the diff is")
        print("              the prefill STEP PARTITION, so it can inform partition sensitivity only")
        print("UNUSABLE      cap_max < floor: no cap satisfies both P1 and P2 -> the pool path stops")
        print(f"P1: cap*topk <= min_layer_slots-1 | P2: cap >= floor (MTP on -> n_rs_seq="
              f"{args.spec_n_max} -> floor {args.spec_n_max + 2}; off -> no floor)")
        print(f"slots = clamp(pool_bytes / (max_layer * per_slot), {CAP_MIN}, {CAP_MAX}); "
              f"layer_caps={args.layer_caps or DEFAULT_LAYER_CAPS!r}")
        print("=" * 118)

    if args.validate:
        result_dir = os.path.join(ROOT, "Backup", "knifeedge_matrix")
        validate_against_probes(records, result_dir)

    if args.json:
        json.dump({"ref_cap": REF_CAP, "spec_n_max": args.spec_n_max,
                   "cells": records}, open(args.json, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}", file=sys.stderr)

    bad = [r for r in records if r["verdict"] != "USABLE"]
    if bad:
        print(f"\n{bad[0]['verdict']} cells: "
              + ", ".join(f"{r['model']}@{r['pool_gb']}GiB" for r in bad), file=sys.stderr)
    return 0


def MODELS_MTP(km, key):
    """MODELS stores mtp as "0"/"1"; normalise to a bool (default on)."""
    return str(km.MODELS.get(key, {}).get("mtp", "1")) == "1"


if __name__ == "__main__":
    sys.exit(main())
