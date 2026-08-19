#!/usr/bin/env python3
"""convert_paper_trace.py — convert ISCA'26 "Patterns behind Chaos" MoE trace to sim_gguf_cache.py format.

Paper: "Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale
MoE LLM Inference" (ISCA'26, arXiv:2510.05497). Traces:
  https://huggingface.co/datasets/core12345/MoE_expert_selection_trace   (gated)
  org layout: {model}/{dataset}/{subdir}/{n}.json

Actual per-request JSON schema in the HF dataset (verified on
cognitivecomputations/DeepSeek-R1-AWQ/mmlu/<subject>/<n>.json):

  [                       # one dict per step
    {                     # step 0 = PREFILL batch (one entry per prefill token)
      "0": null,         #   dense layers present as null (e.g. DSV3 layers 0-2)
      "3": [ [8 experts], [8 experts], ... ],   #  B entries, B = prefill token count
      ...
    },
    {                     # steps 1..N = DECODE, one token per step
      "3": [ [8 experts] ],                    #   inner batch dim is 1 (single sequence)
      ...
    },
    ...
  ]

Topology is per-model: DeepSeek-V3 = 61 layers (3 dense + 58 MoE), 256 experts,
top-8; Qwen3-235B = 63 layers; Llama4 = dense layers interleaved; Kimi K2 has a
different expert count. This converter derives the MoE layer set from the data
itself — dense layers (null values) are skipped and MoE layer ids are remapped
to consecutive 0..N-1.

sim_gguf_cache.py consumes flat text:
  layer_XX.bin,phase,step,hits,exp1 exp2 ... exp8
with N_LAYERS consecutive lines grouped as one token.

This converter:
  * derives the MoE topology from the data itself (distinct layer ids seen across
    all files) and remaps them to consecutive 0..N-1, so models with dense FFN
    layers interleaved (e.g. Llama4) or non-zero-based layer ids work unchanged;
  * emits the decode phase by default, optionally the prefill phase too
    (--emit-prefill) for pool-warmup experiments (paper insight Ob3: prefill
    routing predicts decode routing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"


def collect_input_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        )
        if not files:
            sys.exit(f"error: no *.json files under {input_path}")
        return files
    sys.exit(f"error: input path not found: {input_path}")


def load_layer_map(files: list[str]) -> dict[str, int]:
    """Collect distinct layer ids across all files, remap to consecutive 0..N-1.

    Layer ids are per-layer expert indices in the paper traces; dense layers are
    absent from step dicts (or null), so they simply never contribute. Sorting
    numerically keeps 0..N-1 aligned with the model's MoE layer order.
    """
    seen: set[int] = set()

    def scan_record(record: list) -> None:
        for step in record:
            if not isinstance(step, dict):
                continue
            for layer_str, val in step.items():
                if val is None:
                    continue  # dense layer (present as null), not a MoE layer
                try:
                    seen.add(int(layer_str))
                except (TypeError, ValueError):
                    pass

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except Exception as exc:
            print(f"warn: skipping {path}: {exc}", file=sys.stderr)
            continue
        scan_record(record)

    ordered = sorted(seen)
    if not ordered:
        sys.exit("error: no layer ids found in input traces")
    return {str(layer_id): idx for idx, layer_id in enumerate(ordered)}


def _layer_lines(layer_map: dict[str, int], step: dict, phase: str,
                 step_idx: int) -> list[str]:
    """Flatten one step dict (layer_id -> [ [experts], ... ]) into sim lines.

    Each inner entry is one token's top-k expert list for that layer. With
    per-layer list length B (prefill batch) or 1 (single-sequence decode),
    producing one line per entry keeps sim's grouping (N_LAYERS consecutive
    lines = one token) aligned with the trace.
    """
    lines: list[str] = []
    layers = sorted((layer_map[k], k) for k in step if k in layer_map)
    for mapped_idx, orig_key in layers:
        entries = step[orig_key]
        if entries is None:
            continue
        for experts in entries:
            if not isinstance(experts, list) or not experts:
                continue
            lines.append(
                f"layer_{mapped_idx:02d}.bin,{phase},{step_idx},"
                f"{len(experts)},{' '.join(map(str, experts))}"
            )
    return lines


def _split_prefill_decode(record: list) -> tuple[dict | None, list[dict]]:
    """Split the raw step list into (prefill_step, decode_steps).

    In the dataset layout step 0 holds the whole prefill batch (per-layer lists
    of B entries), and the remaining steps are single-token decode steps.
    """
    if not record or not isinstance(record, list):
        return None, []
    first = record[0]
    if not isinstance(first, dict):
        return None, []
    return first, [s for s in record[1:] if isinstance(s, dict)]


def _batch_size(step: dict) -> int | None:
    """Infer per-layer token count from the first non-dense layer."""
    for v in step.values():
        if isinstance(v, list) and v:
            return len(v)
    return None


def emit_decode(files: list[str], layer_map: dict[str, int], out) -> None:
    n_steps = 0
    n_lines = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except Exception as exc:
            print(f"warn: skipping {path}: {exc}", file=sys.stderr)
            continue
        _, decode_steps = _split_prefill_decode(record)
        for step_idx, step in enumerate(decode_steps):
            lines = _layer_lines(layer_map, step, PHASE_DECODE, step_idx)
            out.write("".join(f"{ln}\n" for ln in lines))
            n_lines += len(lines)
            n_steps += 1
    print(f"decode: {n_steps} steps, {n_lines} lines, {len(layer_map)} layers", file=sys.stderr)


def emit_prefill(files: list[str], layer_map: dict[str, int], out) -> None:
    n_tokens = 0
    n_lines = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except Exception as exc:
            print(f"warn: skipping {path}: {exc}", file=sys.stderr)
            continue
        prefill_step, _ = _split_prefill_decode(record)
        if not prefill_step:
            continue
        batch = _batch_size(prefill_step)
        if not batch:
            continue
        lines = _layer_lines(layer_map, prefill_step, PHASE_PREFILL, 0)
        if not lines:
            continue
        out.write("".join(f"{ln}\n" for ln in lines))
        n_lines += len(lines)
        n_tokens += batch
    print(f"prefill: {n_tokens} tokens, {n_lines} lines, {len(layer_map)} layers", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="paper trace JSON file, or directory of per-request JSON files")
    ap.add_argument("--output-decode", default="",
                    help="output flat trace (decode). default: <input-dir>/paper_decode.txt")
    ap.add_argument("--emit-prefill", action="store_true",
                    help="also emit prefill phase to <output-decode>.prefill.txt (for pool warmup)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="only convert the first N files (input dir listing order)")
    args = ap.parse_args()

    files = collect_input_files(args.input)
    if args.max_files:
        files = files[: args.max_files]
    print(f"input: {len(files)} file(s)", file=sys.stderr)

    layer_map = load_layer_map(files)
    print(f"topology: {len(layer_map)} MoE layers (remapped to 0..{len(layer_map)-1})",
          file=sys.stderr)

    out_decode = args.output_decode
    if not out_decode:
        base = args.input if os.path.isfile(args.input) else args.input
        out_decode = os.path.join(base, "paper_decode.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_decode)), exist_ok=True)

    with open(out_decode, "w", encoding="utf-8") as f:
        emit_decode(files, layer_map, f)
    print(f"wrote: {out_decode}")

    if args.emit_prefill:
        out_prefill = out_decode + ".prefill.txt"
        with open(out_prefill, "w", encoding="utf-8") as f:
            emit_prefill(files, layer_map, f)
        print(f"wrote: {out_prefill}")


if __name__ == "__main__":
    main()
