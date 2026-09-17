#!/usr/bin/env python3
"""knifeedge.constants — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import re


# Per-model launch facts. `mtp` selects whether to ask run_server.sh for the MTP draft head;
# `path` (optional, absolute) overrides models/gguf/<file>.
#
# The template is chosen by CGC_SERVER_PROFILE in run_server.sh, NOT by model kind: with no
# profile and no explicit overrides, EVERY model gets the same Qwen3-nothink-ChatML.jinja. Any
# note here claiming "iq3 -> embedded template" was wrong; two entries can only ever differ by
# their weights. gate_model_identity() enforces that they actually do.
MODELS = {
    "iq3": {
        "file": "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "Nail IQ3_XXS denseIQ4X MTP carrier (13.6 GB). In this checkout the name is a "
                "symlink -- see gate_model_identity: it must not point at another entry's file.",
    },
    "iq3ext": {
        # The real 13 GB IQ3_XXS-denseIQ4X lives on the external drive; the internal disk has no
        # room for it next to the 18.2 GB IQ4_XS (10 GiB free, 13 GB needed). Read speed there is
        # a measured 86 MB/s, so a small pool can be filled from it at run time.
        "file": "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "path": "/Volumes/AlexZhuang/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "real IQ3_XXS-denseIQ4X MTP, run straight from the external drive",
    },
    "iq4": {
        "file": "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "vanilla IQ4_XS (18.2 GB). NOT on disk in this checkout -- see the model-id line, "
                "which prints exists=False rather than silently comparing something else.",
    },
    "ornith": {
        # A different BASE, not a different quantisation of the same base: Ornith 1.5 35B is an
        # abliterated MLX-bf16 fine-tune, requantised here to a mixed q3_K/q4_K trunk plus a q8_0
        # MTP layer (123 expert tensors: q3_K=90, q4_K=30, q8_0=3).
        "file": "Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "17.6 GB, arch qwen35moe, block_count 41 + nextn_predict_layers 1, 256 experts, "
                "top_k 8, n_ff_exp 512 -> 589,824 B/expert for the q4_K kinds (the same per-slot "
                "geometry as the Edge0 Q4_0 entry, so a given pool maps to a comparable slot "
                "count) and 450,560 B/expert for q3_K. It carries its own head (fingerprint "
                "2e35283d4a95c0bc, 21 tensors, 1372.7 MiB, types F32=9/Q8_0=12, router alive) -- "
                "no graft, so this is a THIRD independently-paired (base, head) rather than a "
                "second carrier for an existing pair.",
    },
}


# `buffer is nil` -- the signature of the L3-B gather path repointing a Metal-resident FFN
# tensor at a HOST std::vector. One hit invalidates the combo AND explains numeric drift.
NIL_RE = re.compile(r"buffer is nil")



# Canonical probe. The point is not the arithmetic: it is that a working scaffold makes the
# model produce a short answer containing 42, while a broken one echoes the prompt, leaks a
# think marker, or returns empty. Whatever the pool size, EVERY config must pass this or the
# numbers below it are measuring the template, not the cache.
PROBE_PROMPT = "15+27 等於多少？請只輸出答案"



# `CGC-UNION: layer=1 union avg=38.2 min=16 max=64 of usable=34 (188%)  [WIDE: exceeds usable] gather=3/9`
# The `gather=N/M` tail is the only place the engine reports that the WIDE-union route was
# actually taken (llama-context.cpp, CGC_UNION_LOG). The WIDE marker is required in the pattern:
# a narrow layer can also print gather>0 for other reasons, and those would prove nothing about
# the wide-union path this gate is about.
GATHER_RE = re.compile(r"CGC-UNION: layer=(\d+).*?\[WIDE[^\]]*\].*?gather=(\d+)/(\d+)")



# Floor for `CGC_POOL_MAX_TOKENS`. Two independent lower bounds:
#   * arithmetic -- cap*topk must fit the pool (see union_fit_gate);
#   * liveness   -- n_batch is clamped to cap, and the MTP verify batch needs n_batch above
#                   n_keep_tail: measured, cap=4 aborted at llama-batch.cpp:609
#                   GGML_ASSERT(n_ubatch > n_keep_tail) in the startup warmup decode.
MIN_CAP = 6



# The feasibility prediction is a model of the LOADER (`compute_l4_pool_capacity` and
# `cgc_layer_cap`), so its provenance has to include the loader. Deliberately NOT merged into
# NUMERIC_SOURCES: that tuple stamps the oracle path, and a file added there would invalidate every
# existing reference (they carry no hash for it, so a diff would read as "numerics changed").
# Different question, different stamp.
POOL_GEOMETRY_SOURCES = (
    "src/llama.cpp/src/llama-model-loader.cpp",   # compute_l4_pool_capacity -> the slot count
    "src/llama.cpp/src/llama-expert-cache.cpp",   # cgc_layer_cap / LAYER_CAPS resolution
    "src/llama.cpp/src/llama-expert-cache.h",
)



# Human-readable names for the fields of `record_geometry_key`. A refusal that says "something
# differs" is useless; it has to name which of these moved and to what.
KEY_FIELDS = (("engine", "numerics code (source digest)"),
              ("pool_geometry", "pool-geometry code (loader/cache digest)"),
              ("binary", "built binary"),
              ("weights", "weights identity"),
              ("cap", "CGC_POOL_MAX_TOKENS"),
              ("mode", "prefill mode (standard / M2_stream)"),
              ("mtp", "MTP"),
              ("spec_n_max", "spec-draft-n-max"),
              ("layer_caps", "LAYER_CAPS"),
              ("extra_env", "extra env"),
              ("suite", "suite shape (profiles / questions / repeats / temperature / max_tokens)"))



# Source files that DECIDE the numerics. A reference oracle is a snapshot of the engine these
# files describe, so a change here invalidates it. Deliberately narrow: harness/probe scripts
# are excluded because editing them cannot move a single logit.
NUMERIC_SOURCES = (
    "src/llama.cpp/src/llama-expert-cache.cpp",
    "src/llama.cpp/src/llama-expert-cache.h",
    "src/llama.cpp/src/llama-context.cpp",
    "src/llama.cpp/src/llama-graph.cpp",
)



MODEL_SAMPLE_BYTES = 4 << 20        # 4 MiB from the head + 4 MiB from the tail



# [CGC 2026-09-15] How the expert cache participates in a dump. This is the field that was
# missing, and its absence is why the gate could report M1 19/19 for a year while the engine was
# wrong: every reference on this box was produced by `run_server.sh`, which ALWAYS passes
# `-expert-cache <budget>` (scripts/run_server.sh:587). So "reference vs candidate" was
# expert-cache-ON vs expert-cache-ON -- a pool-size invariance check wearing the costume of a
# correctness check. Only the states in GROUND_TRUTH_STATES may be compared against as truth.
EXPERT_CACHE_ON = "on"

EXPERT_CACHE_OFF = "off"                 # no cache object at all (budget 0)

EXPERT_CACHE_OFF_NOGATHER = "off_nogather"   # cache exists, hook + skip_load both off

EXPERT_CACHE_NOHOOK = "on_nohook"        # cache exists, skip_load on, hook off (4th arm)

GROUND_TRUTH_STATES = (EXPERT_CACHE_OFF, EXPERT_CACHE_OFF_NOGATHER)



# ============================================================================================
# CAP INVARIANCE -- a first-class gate (2026-09-14)
#
# What it proves: the SAME pool, dumped at TWO different `CGC_POOL_MAX_TOKENS`, must produce the
# same logits. This is the "reduction order / pool layout independence" claim -- the cap changes
# the union width (cap x topk), therefore which experts share the pool and in which order they are
# summed, while the pool SIZE is held fixed. A pool size may only ever differ in speed; a cap
# change may not differ at all, because both caps are legal configurations of the same engine.
#
# Why it has to be a harness gate and not a hand-run: the failure it looks for is invisible in
# every cheaper signal. Measured 2026-09-14 by hand -- cap=5 on an 8 GiB pool was 13-15% FASTER
# than cap=8 and scored M1 3/82 against the cap=8 reference. Nothing in the speed number, the nil
# scan, the union-fit arithmetic or the template gate distinguishes that from a clean config; only
# the logits do. The earlier hand procedure also could not be repeated cheaply, so the +13% lever
# sat unused for want of a gate rather than for want of a measurement.
#
# ALIGNMENT (this is the part that makes it valid): the dump's `step` is a per-process counter of
# dump CALLS, i.e. of ubatches, and the cap clamps n_batch -- so the two caps split one prompt into
# a different number of ubatches and number the same sequence position DIFFERENTLY. Keying on
# (step, token_idx) across two caps would compare unrelated rows and manufacture a verdict out of
# chunking. `pmax` (llama_memory_seq_pos_max) is the absolute sequence position, so the gate
# realigns both dumps on it and keeps only single-token (decode) rows -- in a multi-token dump
# every row carries the same ubatch-wide pmax, so no per-position key exists for them.
# ============================================================================================

CAP_INVARIANCE_DEFAULT_CAPS = "6,8"



FACT_PATTERNS = ((r"\[chat\]\s+model_kind=(\S+)", "model_kind"),
                 (r"\[chat\]\s+template_file=(\S+)", "template_file"),
                 (r"\[chat\]\s+template_kwargs=(\S+)", "template_kwargs"),
                 (r"\[chat\]\s+profile=(\S+)", "profile"),
                 (r"-expert-cache\s+(\d+)", "budget_bytes"),
                 (r"\[start\].*budget=(\d+)B", "budget_bytes"),
                 (r"Soft Pool init: L0=(\d+) L1=(\d+) \(n_slots=(\d+)", "pool"),
                 (r"n_slots=(\d+)", "n_slots"),
                 (r"LAYER_CAPS per-layer caps: total (\d+) slots", "total_slots"),
                 (r"min (\d+)/layer", "min_layer_slots"),
                 (r"\[guard\]\s+memory_mode=(\S+) class=(\S+)", "guard"))
