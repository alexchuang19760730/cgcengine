#!/usr/bin/env python3
"""m123_oracle_gate.py -- the per-commit M1/M2/M3 quality gate.

WHY THIS EXISTS
---------------
`precommit_e2e_gate.sh` is an HTTP quality harness: it grades ANSWERS. It cannot see whether the
forward pass still computes the same numbers. This gate covers that gap. It launches a server
through `run_server.sh` (so the profile, the env allowlist and the load path are the production
ones -- not a hand-built argv that silently drifts), sends the SAME deterministic probe the
reference oracle was produced with, and compares the fresh logits dump against the reference.

THE THREE METRICS ARE NEVER MERGED
----------------------------------
`cgc_logits_oracle_compare.py` reports them separately on purpose:

  M1 numeric identity    same row_fnv1a64  -> bit-identical logits
  M2 decision agreement  same argmax_token -> the chosen token did not change
  M3 topk set agreement  same top-N id set -> the candidate set did not change

M1 same + M2 different   should be 0; anything else is a tie-break bug.
M1 different + M2 same   the trap cell: M1 is the only side that can say "there IS a difference",
                         and greedy decoding is chaotic, so a drift that has not yet flipped a
                         choice can flip on the next token. PASS/FAIL here follows
                         `oracle_gate()`'s rule: ok == (M1 rate == 1.0 AND M2 rate == 1.0), and
                         the 2x2 cross-tab is printed so one number is never quoted for both.

CONFIG COMPARABILITY (added 2026-09-15 -- this is the whole point of the current revision)
-----------------------------------------------------------------------------------------
A cache-ON reference is a *relative* invariance check, and a relative check is only meaningful
between two runs of the SAME numerics-determining configuration. This gate used to ignore that,
and the result was a whole afternoon of chasing a "FAIL" that was really a category error:

  * `ref_iq3_pool8gb_M2_6144.jsonl` (2026-09-15 01:14) was dumped while `CGC_MM_BITIDENT` was
    still ABSENT FROM THE LAUNCH ALLOWLIST (`run_server.sh` only started forwarding it on
    2026-09-15, see the comment on the `-- MTP` block). `CGC_MM_BITIDENT=1` is pillar 1 of the
    bit-identical trio and it changes the M<=8 mul_mat family; without it the SAME logical row
    yields ULP-different values depending on batch size (`ggml-metal-ops.cpp:2395`). A dump taken
    without it can therefore never be bit-identical to a dump taken with it -- measured: only
    `CGC_MM_BITIDENT=0` reproduces the reference's step-0 row hash `e2578ac0ff15b37e`; every other
    single-knob arm lands on `a6302fd4d0ced8c0`.
  * Under the current production configuration the engine is *deterministic*: four independent
    launches produce byte-identical dumps (9/9 rows, 0 mismatches). So the old reference was
    stale, not the engine.

So the gate now resolves the launch environment on BOTH sides via `run_server.sh CGC_DUMP_ENV=1`
(which prints the fully-resolved `CGCENV`/`ENV`/`ARG` and exits before exec -- one source of
truth, no hand-written copy to drift), diffs the numerics-determining subset, and:

  * same config        -> normal M1/M2/M3 verdict.
  * different config   -> prints the diff and exits 2 (INVALID COMPARISON), NOT 1. A cross-config
                          diff is not a regression and must never be reported as one.
  * no sidecar on ref  -> warns that comparability cannot be verified, then compares anyway.

`--write-ref PATH` re-baselines in one step: it copies the fresh dump to PATH and writes a
`PATH.cap` carrying the same resolved config, so the NEXT gate run can prove comparability.
Every gate run also writes its own `Backup/m123_oracle_gate/cap_<tag>.json` for the same reason.

Usage:
  python3 scripts/check/m123_oracle_gate.py
  python3 scripts/check/m123_oracle_gate.py --profile prefill250 --env CGC_SPAC=1
  python3 scripts/check/m123_oracle_gate.py --write-ref Backup/knifeedge_matrix/ref_xxx.jsonl
  python3 scripts/check/m123_oracle_gate.py --allow-incomparable     # read a cross-config diff

Exit code: 0 = M1 and M2 both 1.0, 1 = any difference, 2 = harness error / not comparable.
"""
# This interpreter is Python 3.9.6: without the future import, PEP 604 annotations such as
# `port: int | None = None` are EVALUATED at def time and raise TypeError, which killed the whole
# gate (not just the annotated call) for every caller. Deferring annotation evaluation fixes that
# without changing a single line of the annotated code.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "scripts" / "check" / "cgc_logits_oracle_compare.py"
# v3, dumped 2026-09-15 22:33. Its 9 rows are BYTE-IDENTICAL to v2 (md5
# d2f0b9a01fc5404ebb04dcf3583e133f for both), so this is not a re-baseline of the numerics -- it
# corrects what the sidecar RECORDS.
#
# v2's cap says `CGC_OA_ASYNC=0`, but v2 was dumped at 17:21 against the PRESENCE-based gate
# (`getenv("CGC_OA_ASYNC") != nullptr`), while `run_server.sh` unconditionally puts the variable
# into SERVER_ENV. So that `0` selected the SEGMENTED dispatcher exactly as `1` does; the recorded
# value never had behavioural meaning. When the 21:36 fix made the gate value-aware and the
# launcher default was restored to 1 (preserving the effective behaviour of every profile that
# does not set the knob -- see dec-20260915-2215), `prefill250` resolved to `1` and the gate began
# reporting INVALID COMPARISON (`ENV.CGC_OA_ASYNC: ref='0' now='1'`) on EVERY run, i.e. the
# comparability check itself had gone dark.
#
# The verdict was never in doubt, and the dump proves it independently: M1 was 9/9 bit-identical
# against v2. The non-segmented path lands on `ff68c5a2`, not on the anchor `dc055e63`, so two runs
# that both reproduce `dc055e63` are both on the segmented path no matter what their env string
# says. A missing/incomparable config stamp must not be allowed to veto a bit-identity that the
# logits themselves establish -- and equally, `--allow-incomparable` is not the fix, because it
# discards the check for every future run too.
#
# `prefill250` now PINS CGC_SERVER_OA_ASYNC=1 in run_server.sh (eng-gate-0006: the knob a profile
# fails to state is the knob that drifts). v2 is kept on disk for the historical record.
#
# v3 RETIRED 2026-09-16 by the skip0 value-semantics fix; v4 is the current reference
# (Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v4_skip0off.jsonl).
#
# Read this before concluding anything from a v4-vs-v3 diff. The three readers of
# LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0 used to test PRESENCE (`getenv(...) != nullptr`), so the
# profile's `=0` meant ON and v3 was dumped with layer 0 OUT of the pool (39 pooled layers, and
# `CGC-DECPROF layers=39` on all 19 steps of that era's log). The fix made the readers parse the
# VALUE, so the profile's `=0` now means OFF: layer 0 joins the pool, 40 pooled layers, +142
# resident slots and +152 MiB of pool.
#
#   * `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1` (the pre-fix behaviour, reachable via
#     `CGC_SERVER_SKIP0=1`) reproduces v3 BIT-IDENTICALLY: M1 9/9, M2 9/9, M3 9/9, full-row
#     fnv1a64 9/9. That is the evidence that v3 was dumped with skip0 ON and that the predicate
#     change is inert in every other dimension.
#   * The new default does NOT reproduce v3: M1 4/9, M2 9/9, M3 6/9, cross-tab
#     `{same/same: 4, diff/same: 5, diff/diff: 0}` -- drift, not divergence.
#
# ★ The trap this comment exists for: the *resolved env string is identical on both sides* ("0"
# before the fix and "0" after it). The comparability check above is a string diff, so it CANNOT
# detect a value-semantics change and will report a plain M1 FAIL against v3 -- a category error,
# not a regression. Unlike the CGC_OA_ASYNC round (where the string itself moved, so the gate
# raised INVALID COMPARISON on its own), this class of change only ever gets caught by a human
# re-baselining on purpose. If a future knob change is described as "same env, different meaning",
# write a new reference; do not re-read the old one.
#
# v5 (ref_iq3_pool8gb_M2_6144_bitident_v5_spac.jsonl), dumped 2026-09-16 19:12. `prefill250`
# pinned CGC_SPAC=1 + CGC_SPAC_ALPHA=0.75 to unify the prefill and decode arms (run_server.sh:286),
# so "--profile prefill250" gained two ENV keys the v4 cap does not carry, and the gate went dark
# exactly like the CGC_OA_ASYNC round: `ENV.CGC_SPAC: ref='<absent>' now='1'` plus the same for
# _ALPHA, on every run.
#
# This is a NOMINAL re-baseline, and the evidence is not "M1 was 9/9 when I looked" -- it is that
# v5's file is BYTE-IDENTICAL to v4's (md5 a0a0ca742ca94e843c54b39981742738 for both), so the
# oracle rows were not re-derived, merely re-stamped. It also makes SPAC's claim precise: it
# changes WHICH experts are resident (victim choice + prefetch target) and demonstrably not the
# numbers, on this 9-row probe at this pool geometry. If SPAC ever does move the numerics, the gate
# now fails on M1 against v5 -- which is where it should fail.
#
# Note what is deliberately NOT done: CGC_SPAC is not added to DIAGNOSTIC_KEYS. That set suppresses
# a key from the comparability stamp, so putting it there would make the .cap stop recording it --
# i.e. the next reader could not tell a SPAC-on baseline from a SPAC-off one. The SLOT_TABLE_GPU
# precedent does not transfer: that knob is an experiment whose whole claim is bit-identity, and it
# is not a production default.
# v6 (ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl), dumped 2026-09-17 13:45.
#
# This re-baseline is NOT nominal, unlike every earlier one: it records a FIX.
# `expert_cache_on_topk` used to snapshot the top-k with a LINEAR read of
# `ggml_argsort_top_k`'s output, a view whose `nb[1]` is `n_expert*4` rather than `n_expert_used*4`.
# For every T >= 2 step it therefore read ranks k..2k-1 of token 0's sorted row for token >= 1 --
# legal expert ids, silently the wrong token -- and that vector is what writes the remap leaf, so
# all prefill and all batch verify routed tokens >= 1 through another token's experts. The read is
# nb-aware now; `CGC_IDS_LINEAR_READ=1` restores the old one in place for A/B on one binary.
#
# Measured, new binary vs v5: M1 5/9, M2 6/9, M3 5/9. Row by row the 4 changed rows are
# (0,0,'DEF') plus (5,1)(5,2)(5,3) -- and (5,0) is UNCHANGED while (5,1..3) change, which is the
# defect's own signature: on a 4-token step, token 0 keeps its routing and tokens >= 1 do not.
# A fresh launch against v6 is M1/M2/M3 9/9, and the two runs that failed against v5 produced
# byte-identical hashes (so the reference was stale, not the runs).
#
# Consequence for older evidence: M1/M2 passing across pools before this date shows INVARIANCE
# only. Every arm and every pool size made the same mistake, so cross-pool agreement could not see
# it -- the 2026-09-14 "M1 117/117" belongs to that category and must not be quoted as "the
# numbers were right". Only a v6 comparison can support that claim.
#
# Not a diagnostic key, and deliberately so: CGC_IDS_LINEAR_READ stays in the comparability stamp,
# because a reference dumped with the old read must never compare clean against a fixed binary.
DEFAULT_REF = ROOT / "Backup" / "knifeedge_matrix" / "ref_iq3_pool8gb_M2_6144_bitident_v7_20260926.jsonl"
RESULT_DIR = ROOT / "Backup" / "m123_oracle_gate"

# ★ The reference is an UNTRACKED asset (Backup/ is gitignored) and every verdict is measured against
# it, so its PATH is not enough: the summary must also say WHICH BYTES were compared. Until
# 2026-09-19 nothing pinned the reference's content, which means re-baselining or hand-editing that
# one file would silently redefine every future M1/M2 number with no artefact recording the change.
# Pinned here, next to DEFAULT_REF, in the same spirit as ORACLE_PINNED_ENV below.
#
#   --write-ref writes the .cap sidecar but does NOT update this table; after a deliberate
#   re-baseline, update the md5 here in the same commit, so the change is reviewable.
REF_PINS = {
    # v6 — the baseline the tree carried until 2026-09-26. It last produced 9/9 on 09-23 11:32
    # (pfx-ckpt2); every run after that reports 6/9, and the three differing rows are all
    # ctx_type='MTP' while the DEF rows still match. Kept so the old number stays reproducible.
    "ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl": "72d82a33ad79e0e69bc935acd24228f2",
    # v7 — re-baselined 2026-09-26 on tree 591d7cfac. WHY THIS IS A RE-BASELINE AND NOT A FIX:
    # the engine is reproducible (two independent dumps are byte-identical, md5 e1663cc10d, which
    # is exactly the ref the 12:29 fn_on run compared against at 9/9), so there is no numerical
    # bug to fix — the MTP bucket's semantics moved. See docs/M1M2M3_REBASELINE_2026-09-26.md.
    "ref_iq3_pool8gb_M2_6144_bitident_v7_20260926.jsonl": "e1663cc10d529571553e7105b4a7a51b",
}


def ref_md5(path: Path) -> str:
    """md5 of the reference file, streamed (these dumps are small, but no reason to hold them)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ★ The oracle's batch/ubatch are part of THE ORACLE'S IDENTITY, not a property of the profile.
#
# The `prefill250` profile used to carry batch=ubatch=6144 as its production default, which made
# "--profile prefill250" happen to reproduce the reference. On 2026-09-16 that default moved to
# 5632 for a production reason (measured: 6144 died on req2 0/5, 5632 survived 4/4) -- and the
# gate immediately reported INVALID COMPARISON on a 4-row diff (CGCENV.BATCH, CGCENV.UBATCH,
# ARG[27], ARG[29]) that was 100% my own default changing, while M1/M2/M3 sat at 9/9. Two things
# were wrong with that: (a) a production tuning silently redefined what the reference means, and
# (b) the fix "just pass --env CGC_SERVER_BATCH=6144" lives in a human's memory, which is exactly
# the kind of hand-written copy this file's docstring says must not exist.
#
# So the oracle's numerics-determining knobs are pinned HERE, once, next to DEFAULT_REF. A future
# production tuning cannot move the reference again. Re-baselining at a different config is still
# possible and still deliberate: `--write-ref` + `--no-pin-oracle-env` (see the flags in main()).
ORACLE_PINNED_ENV = ("CGC_SERVER_BATCH=6144", "CGC_SERVER_UBATCH=6144")

# knifeedge_matrix.PROBE_PROMPT, verbatim. The oracle dump is keyed on
# (step, token_idx, ctx_type), so a different prompt produces a different token sequence and the
# keys stop lining up -- the comparison would then report "0 common keys" instead of a verdict.
PROBE_PROMPT = "15+27 等於多少？請只輸出答案"

# Keys that only steer instrumentation. They must not count as a config difference, or every run
# would look incomparable to every reference (the dump path alone differs per tag).
DIAGNOSTIC_KEYS = {
    "CGC_DUMP_ENV", "CGC_LOGITS_ORACLE_DUMP", "CGC_LOGITS_ORACLE_TOPN",
    "CGC_LOGITS_ORACLE_FIRST_N", "CGC_SLOT_DBG", "CGC_MMID_MV_DBG", "CGC_MMID_ASSERT",
    "CGC_MMID_ASSERT_FATAL", "CGC_PREV_PF_DBG", "CGC_IDS_MAX_LINES", "CGC_SLAB_DBG",
    "CGC_MM_DBG", "CGC_SPAC_DBG", "CGC_CAP_DBG", "CGC_ROUTE_DUMP",
    # [CGC 2026-09-15 S1 slot-table] This key is deliberately in the "does not change the numbers"
    # set, because that IS its claim: CGC_SLOT_TABLE_GPU=1 moves the expert->slot mapping from a
    # host-written input leaf to a graph gather (get_rows over a per-layer table) and must produce
    # the same id vector. Leaving it out would make every S1 run report "incomparable" against the
    # reference, so the gate could never express the one proposition it exists to test. If the
    # claim is false the gate fails on the LOGITS, which is exactly where it should fail.
    "CGC_SLOT_TABLE_GPU",
    # [CGC 2026-09-26 fill-nocache] Same kind of claim as CGC_SLOT_TABLE_GPU above: F_NOCACHE on the
    # expert-file handle changes WHERE the bytes come from, never WHICH bytes are read (same pread,
    # same offsets, same destination buffers). Leaving the key out would make every knob-on run
    # report "incomparable" against a knob-off reference, i.e. the gate could not express the one
    # proposition the knob rests on. If the claim is false the gate fails on the LOGITS.
    "CGC_FILL_NOCACHE",
    # [CGC 2026-09-30 R6 witness] The third key of this kind, and the one whose claim is being
    # **tested** rather than presumed. `CGC_SEG_BATCH=1` submits the 41 decode segments as one,
    # which SKIPS the per-layer top-k hook (llama-context.cpp:4059 "CGC_SEG_BATCH=1 SKIPS THE
    # HOOK") and maintains the pool's members through the read-back path instead. The claim that
    # makes it comparable -- and therefore testable -- is: WHERE the members are maintained does
    # not change WHICH members are used (same id vector, same logits).
    # Why it has to live here rather than be measured around: `quote_gate.py`'s R6 blocks every
    # throughput reading from this arm *because its output is unverified*, and it names exactly one
    # release condition -- an M1/answer-hash witness on the server path (`m123_oracle_gate.py` IS
    # that path: run_server.sh + --env). Without this key, the very run meant to produce the witness
    # reports INCOMPARABLE against the reference, so the release condition could never be satisfied
    # by any measurement. Marking the claim comparable is not assuming it true: if it is false, the
    # gate fails on the LOGITS, which is exactly where it should fail.
    # ⚠ The registry's own text predicts which way this can go: with G3 (`CGC_ZERO_SLOT`) unarmed, a
    # non-resident selection falls back to `e % ns` and lands on a DIFFERENT real expert ⇒ a red M1
    # here is a live possibility, and it would be a finding about the arm, not a harness bug.
    "CGC_SEG_BATCH",
    # [CGC 2026-10-01 Y-accel · (b)+Y predict-ensure] CGC_SEG_BATCH_FAST collapses each decode step
    # into ONE async submit instead of the 41-segment wait->hook->submit loop. Three things make up
    # the arm, and the claim is about their COMBINATION, not about any one of them:
    #   (1) one extra capture node (ffn_moe_ids_cont, a contiguous copy of selected_experts) in the
    #       host-leaf decode graph -- a pure copy consumed by nothing else, so it cannot move a value;
    #   (2) before submit, each layer's remap leaf is written from the PREVIOUS token's predicted
    #       expert ids and those experts are sync-ensured resident -- the same mapping the per-segment
    #       hook writes (llama-context.cpp:7941-7968), just sourced from prediction;
    #   (3) after submit, the true ids are read back and each position's consumed slot (rd[i]) is
    #       compared against slot_table_safe(true_id): any difference means the prediction missed
    #       (or an ensure failed), and the WHOLE step is redone via the proven 41-segment loop.
    # The claim is therefore: a FAST step either reproduces the 41-seg step exactly, or it is thrown
    # away and recomputed by the 41-seg loop -- so the emitted logits are the 41-seg logits either
    # way. It is NOT a claim that prediction rarely misses: a high miss rate costs speed, never
    # correctness.
    # Registered (rather than left out) for the same reason as CGC_SEG_BATCH: without it, every FAST
    # run reports INCOMPARABLE against the reference and the release condition could never be met by
    # any measurement. Two holes would surface as a red LOGITS verdict, which is where they belong:
    # a divergence the post-submit comparison cannot see (e.g. an expert whose ZERO-slot mapping is
    # identical under both prediction and truth), or the id-collection this arm turns on
    # (llama-context.cpp:6569) not being inert after all.
    "CGC_SEG_BATCH_FAST",
    # [CGC 2026-10-02 線A · overlap fence] exp-overlap-partial-2026-10-02. The claim is NARROW and
    # stated per byte: CGC_OVERLAP_FENCE=k changes only the ORDER in which segments are committed
    # (segment i+1 is committed before the host runs segment i's top-k hook) and inserts a device
    # event wait at the head of that segment's first command buffer, released by the host only AFTER
    # the remap leaf write returns. Therefore: (a) the remap leaf's BYTES are identical -- they are
    # produced by the same host code from the same state, just earlier; (b) the WEIGHT BYTES the MoE
    # reads are identical -- residency/eviction is untouched (no segment is removed, no ensure is
    # skipped); (c) the logits are therefore expected to be unchanged. What DOES move is timing
    # only. If the claim is false the gate fails on the LOGITS, which is where it should fail.
    # NOT covered by this claim: k only fences the first k boundaries per graph, so a k that is too
    # small is not a correctness question at all -- the unfenced boundaries keep the upstream order.
    "CGC_OVERLAP_FENCE",
    # [CGC 2026-10-02 線A · leafonly D2] exp-overlap-leafonly, the refinement of the fence above.
    # The claim, stated per byte: CGC_LEAF_SPLIT=1 submits a fenced span as TWO graph_computes instead
    # of one -- part A = [seg_start, first `ffn_up-` node) which holds every remap reader and keeps
    # the event wait, and part B = [that node, seg_end] which begins with the shared expert
    # (`build_ffn(cur, ffn_up_shexp, ...)`, whose input is the pre-MoE `cur`) and reads NO remap.
    # Therefore: (a) part B commits before segment i's hook, and that is the entire point -- it is
    # leaf-independent work by construction, so there is no read-before-write to lose; (b) every byte
    # part B consumes from part A (moe_out -> ffn_out, then the residual into layer i+1) is ordered
    # by Metal's own hazard tracking, the same mechanism CGC_SUBMIT_AHEAD (below) already proved for
    # every non-remap tensor; (c) the remap leaf's own bytes and its read-after-write ordering are
    # untouched -- part A still waits on the event that is signalled only after the write returns.
    # What DOES move is when the dispatches are committed. If the claim is false the gate fails on the
    # LOGITS, which is where it should fail. ⚠ This registration also carries the cb-accounting fix
    # (seg_cb_end[]): with a two-part submit the old `done0 + (i+1)*bufs` hook wait would fire early
    # and that WOULD corrupt the remap, so the split and that fix are one proposition, not two.
    "CGC_LEAF_SPLIT",
    # CGC_SUBMIT_AHEAD is the RACY ceiling probe for the same axis (it mutates a remap buffer an
    # in-flight command buffer references). It is registered here ONLY so that a diagnostic run can
    # be compared on the same basis; an arm carrying it is expected to FAIL the identity check, and
    # that failure is the point (it is the "the probe really moved something" control).
    "CGC_SUBMIT_AHEAD",
    # [CGC 2026-09-30 線A · S2] The prefetch pair, registered for the same reason and with the same
    # caveat as CGC_SEG_BATCH: this is the mechanism the R6 witness is meant to TEST, and a knob that
    # makes a run "incomparable" cannot be tested by any measurement. The claim is narrow --
    # CGC_RHO_PROBE builds a per-layer shadow router and CGC_RHO_FILL turns its prediction into a
    # non-blocking batch prefetch of the SAME experts the per-layer fill would read, so which bytes
    # reach the MoE is unchanged; only WHEN they arrive differs. It is not a claim that the arm's
    # output is already correct: on the hook-less arm residency was measured at 57.1% versus 96.3%
    # on the honest path, and a non-resident selection falls back to `e % ns` and lands on a
    # DIFFERENT real expert. So the expectation this registration makes testable is directional:
    # if missing residency is the whole divergence, M1 goes 0/9 -> 9/9; if the divergence lives in
    # the single-submit graph itself, M1 stays red -- and that, not a green M1, is the finding.
    "CGC_RHO_PROBE", "CGC_RHO_FILL",
    # [CGC 2026-10-01 rho split · delivery flag] `CGC_RHO=1` is the flag the split
    # (`Backup/rho_split_2026-09-30/rho_delivery_split.patch`) introduces so the delivery arm can
    # run the mechanism with the meter OFF: `cgc_rho_meter()` gates the four delivery-path prints
    # on `CGC_RHO_PROBE`, so a delivery run's stderr contains zero `CGC-RHO-*` lines while the
    # prefetch itself still fires. Its claim is exactly numerics-neutral: the split changes WHERE
    # the meter prints, never WHICH experts are prefetched, so the id vector and logits must be
    # unchanged. Leaving the key out makes the delivery arm's oracle report INCOMPARABLE against
    # a probe-off reference (`ENV.CGC_RHO: ref=<absent> now=1`), i.e. the split's pre-registered
    # receipt (`rho_window.py` S4: 同 build、同一份參考、M1 9/9) could never be satisfied by any
    # measurement. If the claim is false the gate fails on the LOGITS, which is where it should.
    "CGC_RHO",
    # [CGC 2026-10-02 S1 single-submit arm] Four keys, one claim between them, registered for the
    # same reason as CGC_SLOT_TABLE_GPU above: without them every run of this arm reports
    # INCOMPARABLE against the reference and the one proposition it exists to test could never be
    # expressed by any measurement. The claim, split by key:
    #   CGC_S1_MIN_IL=0     layer 0 joins the GPU-table path. Its host leaf and the table implement
    #                       the SAME mapping (the S1 equivalence the 09-20 witness already measured
    #                       layer-by-layer), so which of the two the consumer reads cannot change
    #                       the answer -- unless the cross-backend copy this gate keeps layer 0 out
    #                       of is broken, which is exactly a LOGITS failure.
    #   CGC_S1_SINGLE_SUBMIT / CGC_S1_SS_VERIFY  one async submit per step plus the pre-submit table
    #                       publish; the witness only reads ids back after the drain, so it cannot
    #                       move a value. Whether the published table is CORRECT is not presumed
    #                       here: it is published from the live pool, the same source the per-segment
    #                       hook publishes from, and a wrong entry shows up as wrong logits.
    #   CGC_FAST_COST       timers only (ggml_time_us around existing calls); it has been on in every
    #                       A/B arm since 2026-10-01 and was never in this set, so a cost-split run
    #                       could not be compared against a reference at all. Registering it moves
    #                       no threshold: a timer that moved the numbers would fail on the LOGITS.
    "CGC_S1_MIN_IL", "CGC_S1_SINGLE_SUBMIT", "CGC_S1_SS_VERIFY", "CGC_FAST_COST",
}
# CGCENV scalars that are comparability-irrelevant (paths/timing only).
DIAGNOSTIC_CGCENV = {"LOG", "PORT"}


def _opener():
    """No-proxy opener: this box runs an HTTP proxy on 7897 that eats 127.0.0.1 requests."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url, timeout=5.0):
    return _opener().open(url, timeout=timeout).read()


def _post(url, payload, timeout=300.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with _opener().open(req, timeout=timeout) as r:
        return r.read()


def kill_servers(port: int | None = None):
    """Preflight cleanup. With a port, only that listener is targeted: `pkill -f llama-server`
    also matches every parallel session's server (http_duo.py:31,285). llama-bench has no port, so
    it CANNOT be scoped -- and the gate never launches it, while profile_duo.py (this repo's
    delivery instrument for t/s) does. See the listing/opt-in below.

    [CGC 2026-09-19] The `else` branch used to fall back to `pkill -9 -f llama-server` precisely
    when nothing was listening on our port -- i.e. the idlest moment on the box, and therefore the
    moment when the only servers that pattern can match are OTHER sessions'. "Free the port" means
    "kill our listener", and with no listener on that port there is nothing of ours to free. A
    pid-blind kill here cannot make our launch succeed; it can only break someone else's run, so
    the fallback is removed. The no-port call (used only where a port genuinely does not exist)
    keeps the pattern kill and says so.
    """
    if port is not None:
        pids = server_listeners(port)
        if pids:
            subprocess.run(["kill", "-9", *pids], check=False)
        # else: nothing of ours is on that port. Do NOT widen to a pattern kill.
    else:
        subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    # llama-bench is NOT a leftover of ours -- the gate never launches it, profile_duo.py does.
    # A pattern kill here can therefore only hit another session's in-flight measurement, and the
    # victim sees a dead bench rather than a dead gate. Same rule as run_server.sh's preflight:
    # list by default, signal only on explicit request (CGC_PREFLIGHT_KILL=all).
    _lb = subprocess.run(["pgrep", "-fl", "llama-bench"], capture_output=True, text=True)
    _hits = [l for l in _lb.stdout.splitlines() if l.strip()]
    if _hits:
        if os.environ.get("CGC_PREFLIGHT_KILL", "").strip() == "all":
            subprocess.run(["pkill", "-9", "-f", "llama-bench"], check=False)
            print("  [preflight] killed %d llama-bench (CGC_PREFLIGHT_KILL=all)" % len(_hits),
                  flush=True)
        else:
            print("  [preflight] %d llama-bench running, NOT killed (set CGC_PREFLIGHT_KILL=all "
                  "to kill); first: %s" % (len(_hits), _hits[0].strip()[:88]), flush=True)
    time.sleep(1.0)


def server_pids():
    out = subprocess.run(["pgrep", "-f", "build/bin/llama-server"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def server_listeners(port: int) -> list[str]:
    """PIDs listening on `port`. The pid-blind `server_pids()` above answers "is any server up",
    which is what the readiness/teardown loop wants; this one answers "is OURS up", which is what
    a teardown must use on a box shared with other sessions."""
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout
    return [x for x in out.split() if x.strip().isdigit()]


ENGINE_BIN = ROOT / "src" / "llama.cpp" / "build" / "bin"

# The artifacts, named as the loader SEES them: the version-less dylib is a symlink to the versioned
# file, and the loader maps the TARGET. Hashing the version-less name therefore means following the
# link -- not guessing a version string.
#
# Why this list is spelled this way (fixed 2026-09-26, measured on this box):
#   * it used to hash `libllama.0.0.279.dylib`, which is a STALE leftover in build/bin while
#     `libllama.0.dylib -> libllama.0.0.578.dylib` is what actually runs. So the digest recorded the
#     bytes of a file no process had loaded.
#   * it did not include `libllama-common.0.dylib` at all -- and `common/speculative.cpp` is compiled
#     INTO that dylib. The consequence was measured this same day: the pre-fix and the k_eff-fixed
#     engine produced summaries with IDENTICAL engine_digest (libllama.0.0.279=0cd5a628,
#     llama-server=054fb22f, ...), i.e. the one comparison the digest exists to make -- "is this the
#     same build?" -- was blind on the exact axis under test, and `m123_gate_window.digests_agree()`
#     would have certified two different engines as one.
#
# libllama-server-impl.dylib belongs here for the same reason as the rest: it is where the server's
# own logic lives, and `llama-server` is a ~50 KB launcher whose md5 does NOT move when that logic
# changes. Found by measurement on 2026-09-23: three runs with different server-context.cpp
# behaviour (6c/6e, four arms) all recorded llama-server=054fb22f04a01c5c, so the summaries could not
# have told them apart -- the exact failure mode this function's docstring describes.
# (Spelled without the audit module's filename on purpose -- and the same for the harness's: the
# shared window probe's audit decides "is this launcher gated?" by TEXT search for those names, so
# naming one here reports this file as newly gated and invites the ungated baseline to be lowered on
# a phantom. It did, twice, before this sentence was rewritten.)
ENGINE_ARTIFACTS = ("libllama.0.dylib", "libllama-common.0.dylib", "libggml-metal.0.dylib",
                    "libggml-base.0.dylib", "libllama-server-impl.dylib", "llama-server")


def engine_digest(bin_dir: Path | None = None) -> dict:
    """md5 of the linked engine artifacts, recorded with every verdict.

    Why this is not optional: on 2026-09-19 the same probe with the same env and argv produced
    PLAIN_MATCH TRUE at 02:57 and FALSE at 04:0x with ONE difference between them -- libllama was
    rebuilt in between -- and the artifact that had been linked at 02:54 was already gone from
    disk, so the build could not be bisected after the fact. An M1/M2/M3 verdict has the same
    exposure: "the gate passed" is only a statement about a build if the build is named. With the
    digest in the summary, a flip is build identity, not a mystery.

    `bin_dir` exists for the self-test: it is the only way to feed this function a stale file next
    to a symlink and check that the symlink wins.
    """
    bin_dir = ENGINE_BIN if bin_dir is None else Path(bin_dir)
    out = {}
    for name in ENGINE_ARTIFACTS:
        p = bin_dir / name
        if not p.exists():
            continue
        real = p.resolve()   # the file the loader maps; `name` stays the key so diffs compare links
        h = hashlib.md5()
        with open(real, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = {"md5": h.hexdigest()[:16], "mtime": int(real.stat().st_mtime),
                     "resolves_to": real.name}
    return out


# [CGC 2026-09-27] "same dylib" does NOT imply "same behaviour" -- recording the gap explicitly.
# `engine_digest()` hashes only the linked artifacts, so a binary linked under a dirty tree is
# recorded exactly like the same binary linked from clean HEAD: the tracked part is in the dylib
# and the uncommitted part is nowhere. A 2026-09-27 bisect trusted that identity and concluded
# "the engine never regressed", while the one clean 9/9 and every later 6/9 ran on the SAME
# dylibs -- the difference was the uncommitted sources, which the digest could not see. Paths alone
# were not enough either (they say *what* is dirty, not with what bytes), so we hash the content.
SRC_DIR_PREFIXES = ("src/llama.cpp/src/", "src/llama.cpp/ggml/src/", "src/llama.cpp/tools/server/",
                    "scripts/run_server.sh", "scripts/check/m123_oracle_gate.py")
MAX_DIRTY_HASHED = 40
DIRTY_HASH_MAX_BYTES = 8 << 20


def _content_md5(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "<unreadable>"
    return h.hexdigest()[:16]


def tree_dirty() -> dict:
    """Tracked-file diff state at verdict time. A gate PASS that does not say whether the tree
    matched HEAD cannot be reproduced later -- and this box runs parallel sessions that edit the
    same tree.

    `dirty_src_hashes` is the part that makes a dylib digest usable as an identity claim: it
    records the *bytes* of the uncommitted engine sources, which is exactly the part of a dirty-tree
    build that no artifact digest can ever cover.
    """
    info = {}
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout
        tracked = [l for l in st.splitlines() if l and not l.startswith("??")]
        info = {"head": head, "dirty_tracked": len(tracked),
                "dirty_paths": sorted(l[3:].strip() for l in tracked)[:20]}
        info["dirty_src_hashes"] = dirty_source_hashes("\n".join(tracked))
    except Exception:  # noqa: BLE001 - provenance must never fail the run it describes
        pass
    return info


def dirty_source_hashes(porcelain: str, root: Path | None = None) -> dict:
    """Content md5 of the uncommitted ENGINE sources listed in a `git status --porcelain` output.

    `root` is a parameter so the self-test can point it at a scratch tree; it defaults to the repo.
    Untracked lines (`??`) are skipped -- an untracked file is not part of the build at all. Files
    over DIRTY_HASH_MAX_BYTES are skipped rather than hashed whole: the cap exists so this stays
    cheap next to a 13 GB model load, and a source that big is already flagged by `dirty_tracked`.
    """
    base = ROOT if root is None else Path(root)
    out = {}
    for line in porcelain.splitlines():
        if not line or line.startswith("??"):
            continue
        rel = line[3:].strip()
        if not rel.startswith(SRC_DIR_PREFIXES):
            continue
        if len(out) >= MAX_DIRTY_HASHED:
            break
        fp = base / rel
        try:
            if fp.stat().st_size > DIRTY_HASH_MAX_BYTES:
                continue
            out[rel] = _content_md5(fp)
        except OSError:
            continue
    return out


def tee(name, text):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / name).write_text(text, errors="replace")


# ---------------------------------------------------------------------------
# [P0 2026-09-15] dump validity -- a gate that can accept a poisoned dump is worse than no gate
# ---------------------------------------------------------------------------
# Motivating incident: a Metal command buffer failed with Insufficient Memory, so the graph never
# ran and the dump was read straight out of the *previous* compute's output buffer. The file was
# structurally valid JSONL, so every consumer downstream treated it as ground truth. The engine now
# aborts on that failure and stamps `<dump>.invalid`; these checks are the second line of defence so
# a dump that is garbage for *any other* reason is also refused.

INVALID_SUFFIX = ".invalid"
# A real LM logit is O(10..100); over <=150K vocab that keeps |sum| under ~1e10 and |mean| under
# ~1e4. These bounds sit ~5-8 orders of magnitude away from anything a forward pass can produce,
# so they exclude garbage without ever risking a false positive on legitimate output.
MAX_ABS_LOGIT = 1.0e30
MAX_ABS_SUM = 1.0e15
MAX_ABS_MEAN = 1.0e9
# Shortest run of consecutive ids tied on the *exact* same bits that we treat as degenerate.
# The incident produced ids 59400..59407 all at 0.125.
MIN_TIED_RUN = 3


def validate_dump(path: Path):
    """Return (ok, reason). Never raises for a malformed record: that *is* a failure.

    Checks, in order of cheapness: the sidecar stamp written by the engine, then per-record
    numeric sanity, then the degeneracy shapes.
    """
    stamp = Path(str(path) + INVALID_SUFFIX)
    if stamp.exists():
        detail = stamp.read_text(errors="replace").strip()
        return False, f"engine stamped this dump invalid ({stamp.name}): {detail}"

    if not path.exists() or path.stat().st_size == 0:
        return False, "dump is missing or empty"

    n_rec = 0
    n_rows_checked = 0
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            return False, f"line {lineno} is not valid JSON: {e}"
        n_rec += 1

        for key, bound, label in (("sum", MAX_ABS_SUM, "|sum|"),
                                  ("mean", MAX_ABS_MEAN, "|mean|")):
            v = rec.get(key)
            if v is None:
                continue
            if not math.isfinite(v):
                return False, f"line {lineno}: {key}={v} is not finite"
            if abs(v) > bound:
                return False, f"line {lineno}: {label}={abs(v):.3e} exceeds {bound:.0e} (stale/failed compute)"

        top = rec.get("top") or []
        if not top:
            return False, f"line {lineno}: empty top-k (argmax_token={rec.get('argmax_token')})"
        if not math.isfinite(rec.get("argmax_logit", 0.0)):
            return False, f"line {lineno}: argmax_logit is not finite"
        if abs(rec.get("argmax_logit", 0.0)) > MAX_ABS_LOGIT:
            return False, f"line {lineno}: argmax_logit={rec['argmax_logit']:.3e} exceeds f32 sanity"

        run = 0
        for a, b in zip(top, top[1:]):
            consecutive = int(b["t"]) == int(a["t"]) + 1
            # exact-bit comparison: `==` on floats is what we want here, and it is what the
            # engine-side screen does too
            identical = float(b["v"]) == float(a["v"])
            run = run + 1 if (consecutive and identical) else 0
            if run >= MIN_TIED_RUN:
                return False, (f"line {lineno}: top-k contains {run + 1} consecutive ids tied on "
                               f"identical values (e.g. t={a['t']} -> t={b['t']}, v={a['v']})")
        n_rows_checked += 1

    if n_rec == 0 or n_rows_checked == 0:
        return False, "dump contains no rows"

    return True, f"{n_rec} records"


# ---------------------------------------------------------------------------
# launch-config resolution -- the thing that makes the comparison falsifiable
# ---------------------------------------------------------------------------
def resolve_launch(profile, extra_env, timeout=120.0):
    """Ask `run_server.sh` for the fully-resolved launch config.

    `CGC_DUMP_ENV=1` makes it print CGCENV/ENV/ARG at the exact point where every profile default
    and override has been applied, then `exit 0` BEFORE the exec. So this is cheap, side-effect
    free (apart from its pre-flight pkill, which is why it runs before anything is launched), and
    it can never drift from what the server would have received.
    """
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env["CGC_DUMP_ENV"] = "1"
    env.pop("CGC_DETACHED", None)
    for kv in extra_env:
        k, _, v = kv.partition("=")
        env[k] = v
    cp = subprocess.run(["bash", "scripts/run_server.sh"], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, timeout=timeout)
    cgcenv, envs, args = {}, {}, []
    for line in cp.stdout.splitlines():
        if line.startswith("CGCENV "):
            parts = line.split(None, 2)
            if len(parts) == 3:
                cgcenv[parts[1]] = parts[2]
        elif line.startswith("ENV "):
            k, _, v = line[4:].partition("=")
            envs[k] = v
        elif line.startswith("ARG "):
            args.append(line[4:])
    if not cgcenv:
        raise RuntimeError("run_server.sh produced no CGCENV block; CGC_DUMP_ENV is gone?\n"
                           f"--- stdout ---\n{cp.stdout[-2000:]}\n--- stderr ---\n{cp.stderr[-2000:]}")
    return {"cgcenv": cgcenv, "env": envs, "args": args}


def config_stamp(cfg):
    """The numerics-determining subset of a resolved launch, frozen for diffing."""
    return {
        "CGCENV": {k: v for k, v in cfg["cgcenv"].items() if k not in DIAGNOSTIC_CGCENV},
        "ENV": {k: v for k, v in cfg["env"].items() if k not in DIAGNOSTIC_KEYS},
        "ARG": cfg["args"],
    }


def _norm_path_value(s):
    """A path-valued entry compares by the BYTES it names, not by its spelling.

    Measured 2026-09-19: `CGC_SERVER_MTP=0` resolves MODEL to
    `models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, which is a SYMLINK whose realpath and size
    (13,663,116,512 B) are byte-for-byte those of the MTP carrier the reference was dumped with.
    The stamp diff nevertheless reported `CGCENV.MODEL` as one of 36 numerics-determining
    differences -- a difference in *spelling* on a check that decides whether a comparison is
    legitimate at all. That matters beyond cosmetics: this file's whole discipline is "a
    cross-config diff is not a regression", so a spelling-only diff makes a legitimate
    comparison look illegitimate and pushes the reader toward re-baselining (which would destroy
    the reference) or toward `--allow-incomparable` (which discards the check everywhere).

    Only absolute paths that exist are resolved; anything else is returned unchanged, so a knob
    value like `1` or `none` can never be mistaken for a filename. The claim is unchanged in
    strength -- different bytes still differ (realpath OR size moves), same bytes no longer do.
    """
    if not isinstance(s, str) or not s.startswith("/"):
        return s
    p = Path(s)
    try:
        if not p.exists():
            return s
        return f"{p.resolve()}|size={p.stat().st_size}"
    except OSError:  # noqa: BLE001 - an unreadable path is just a string
        return s


def diff_stamp(a, b):
    """Return a list of human-readable differences between two config stamps."""
    out = []
    for section in ("CGCENV", "ENV"):
        ka, kb = a.get(section, {}), b.get(section, {})
        for k in sorted(set(ka) | set(kb)):
            va, vb = ka.get(k, "<absent>"), kb.get(k, "<absent>")
            if va != vb:
                na, nb = _norm_path_value(va), _norm_path_value(vb)
                if na == nb:
                    continue  # same bytes, different spelling -- not a config difference
                out.append(f"{section}.{k}: ref={va!r}  now={vb!r}"
                           if na == va and nb == vb else
                           f"{section}.{k}: ref={na!r}  now={nb!r}")
    if a.get("ARG") != b.get("ARG"):
        sa, sb = a.get("ARG", []), b.get("ARG", [])
        for i in range(max(len(sa), len(sb))):
            va = sa[i] if i < len(sa) else "<absent>"
            vb = sb[i] if i < len(sb) else "<absent>"
            if va != vb and _norm_path_value(va) != _norm_path_value(vb):
                out.append(f"ARG[{i}]: ref={va!r}  now={vb!r}")
    return out


def stamp_notes(a, b):
    """Path-valued axes that differ by SPELLING only, i.e. the diffs diff_stamp dropped.

    Reported rather than swallowed: "0 config diffs" must mean "nothing numeric differs", and a
    reader has to be able to tell that apart from "the model path was spelled the same by luck".
    Silence about a dropped axis is how an instrument turns into an assumption.
    """
    out = []
    for section in ("CGCENV", "ENV"):
        ka, kb = a.get(section, {}), b.get(section, {})
        for k in sorted(set(ka) | set(kb)):
            va, vb = ka.get(k, "<absent>"), kb.get(k, "<absent>")
            if va != vb and _norm_path_value(va) == _norm_path_value(vb):
                out.append(f"{section}.{k}: ref={va!r}  now={vb!r}  -> same bytes, not a diff")
    return out


def read_cap(path):
    cap = Path(str(path) + ".cap")
    if not cap.exists():
        return None
    try:
        return json.loads(cap.read_text())
    except Exception:  # noqa: BLE001 - a corrupt sidecar is "no sidecar"
        return None


# ── (d) the gate's OWN pool pressure ────────────────────────────────────────────────────────────
#
# G2 gates changes to the pool's remap/hook path, but until 2026-09-19 the gate never recorded how
# much of that path its own probe exercised. The measured production runs log 12k-758k expert
# lookups at 61.6-96.4% hit rate; a single cold 400-token probe may be nowhere near the eviction
# code, and "the probe passed" would then mean much less than it looks like. So the summary now
# carries the probe run's own counters, read from the server log at teardown.
#
# NOT done on purpose: guessing which server log belongs to this run. The banner prints it
# (`[log]   <path>`), and if that line is absent the answer is recorded as "unknown" rather than
# filled in from a glob -- another run's counters would be a silent wrong answer.
POOL_TEARDOWN_MARKER = "llama_expert_cache: final stats:"

_FINAL_RE = re.compile(
    r"llama_expert_cache: final stats: runtime requests=(?P<requests>\d+) hits=(?P<hits>\d+) "
    r"misses=(?P<misses>\d+) \(hit rate (?P<hit_pct>[\d.]+)%\)\s+prewarm req=(?P<prewarm_req>\d+) "
    r"hit=(?P<prewarm_hit>\d+) miss=(?P<prewarm_miss>\d+)\s+resident=(?P<resident_mib>[\d.]+) MiB")
_MISS_RE = re.compile(
    r"llama_expert_cache: miss attribution: compulsory=(?P<compulsory>\d+) capacity=(?P<capacity>\d+)"
    r" \((?P<compulsory_pct>[\d.]+)% / (?P<capacity_pct>[\d.]+)% of (?P<miss_total>\d+)\)\s+"
    r"evictions=(?P<evictions>\d+)\s+layers_distinct_over_slots=(?P<over>\d+)\s+worst=(?P<worst>[^|\n]+)")
# The banner appends a CJK parenthetical right after the path (`...log（tail -f 同路徑）`), and CJK
# is non-space, so a plain \S+ swallows it and the path stops being a path. Stop at either paren.
_LOG_RE = re.compile(r"\[log\]\s+(?P<path>/[^\s（(]+)")


def parse_pool_counters(text: str) -> dict:
    """Pool counters from a finished server log.

    `found` is False WITH A REASON when the teardown stats are absent -- never a dict of zeros, so
    "the probe saw no pressure" and "nobody looked" stay distinguishable.
    """
    fin = _FINAL_RE.search(text)
    if not fin:
        return {"found": False,
                "reason": f"no '{POOL_TEARDOWN_MARKER}' line in the server log. That line is written "
                          f"at teardown, so the usual cause is reading the log while the server is "
                          f"still alive."}
    out = {"found": True,
           "requests": int(fin.group("requests")), "hits": int(fin.group("hits")),
           "misses": int(fin.group("misses")), "hit_pct": float(fin.group("hit_pct")),
           "prewarm_req": int(fin.group("prewarm_req")),
           "resident_mib": float(fin.group("resident_mib"))}
    m = _MISS_RE.search(text)
    if m:
        for k in ("compulsory", "capacity", "evictions", "over"):
            out[k] = int(m.group(k))
        out["worst"] = m.group("worst").strip()
    return out


def pool_counters_for_run(launch_log: Path, wait_s: float = 20.0) -> dict:
    """The pool counters of THIS gate run, or a reason they could not be read."""
    try:
        banner = launch_log.read_text(errors="replace")
    except OSError as e:
        return {"found": False, "reason": f"launch log unreadable: {e}"}
    m = _LOG_RE.search(banner)
    if not m:
        return {"found": False,
                "reason": f"the launch banner in {launch_log.name} has no '[log] <path>' line, so the "
                          f"server log cannot be attributed. Deliberately not guessed from a glob: "
                          f"another run's counters would be a silent wrong answer."}
    slog = Path(m.group("path"))
    if not slog.exists():
        return {"found": False, "reason": f"server log named by the banner does not exist: {slog}"}
    deadline = time.monotonic() + wait_s
    while True:
        pc = parse_pool_counters(slog.read_text(errors="replace"))
        if pc["found"] or time.monotonic() >= deadline:
            pc["source"] = str(slog)
            return pc
        time.sleep(0.5)


def selftest_engine_digest() -> int:
    """A digest that records a file nobody loaded is worse than no digest: it certifies identity.

    The incident this guards (2026-09-26) is in the comment on ENGINE_ARTIFACTS: two engines that
    differed in `common/speculative.cpp` (compiled into `libllama-common`) produced the SAME recorded
    digest, because the list hashed a stale versioned `libllama` and omitted `libllama-common`.
    """
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "libllama.0.0.999.dylib").write_bytes(b"nobody links this")     # the stale-file shape
        (d / "libllama.0.0.1.dylib").write_bytes(b"the linked bytes")
        (d / "libllama.0.dylib").symlink_to("libllama.0.0.1.dylib")
        (d / "libllama-common.0.0.2.dylib").write_bytes(b"common bytes")
        (d / "libllama-common.0.dylib").symlink_to("libllama-common.0.0.2.dylib")
        (d / "llama-server").write_bytes(b"launcher")
        got = engine_digest(d)
        cases += [
            ("the digest follows the symlink to the linked file",
             got["libllama.0.dylib"]["resolves_to"] == "libllama.0.0.1.dylib"),
            ("... and hashes ITS bytes, not the stale neighbour's",
             got["libllama.0.dylib"]["md5"] == hashlib.md5(b"the linked bytes").hexdigest()[:16]),
            ("a versioned name that nothing links is not invented into the record",
             "libllama.0.0.999.dylib" not in got),
            ("libllama-common is recorded (it holds common/speculative.cpp)",
             got.get("libllama-common.0.dylib", {}).get("md5")
             == hashlib.md5(b"common bytes").hexdigest()[:16]),
            ("missing artifacts are skipped, not faked",
             [k for k in got if k not in ENGINE_ARTIFACTS] == []),
        ]
    real = engine_digest()
    cases += [
        ("the real build/bin digest names the linked libllama",
         real.get("libllama.0.dylib", {}).get("resolves_to", "").startswith("libllama.0.0.")),
        ("... and includes libllama-common", "libllama-common.0.dylib" in real),
        ("no key is a versioned name nothing links",
         all(k in ENGINE_ARTIFACTS for k in real)),
    ]
    bad = 0
    for name, ok in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1
    print(f"engine-digest selftest: {len(cases) - bad}/{len(cases)} passed")
    return 0 if bad == 0 else 1


def selftest_tree_dirty_hashes() -> int:
    """The other half of "which build produced this verdict".

    `engine_digest` covers the linked artifacts; this covers the bytes that are in the working tree
    but in no artifact. A 2026-09-27 bisect read "identical dylib digests" as "the engine never
    regressed" and was wrong, because both verdicts ran on the same dylibs from different trees.
    """
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "src").mkdir()
        (d / "src" / "llama.cpp").mkdir()
        (d / "src" / "llama.cpp" / "src").mkdir()
        (d / "src" / "llama.cpp" / "src" / "llama-context.cpp").write_bytes(b"engine bytes")
        (d / "src" / "llama.cpp" / "ggml").mkdir()
        (d / "src" / "llama.cpp" / "ggml" / "src").mkdir()
        (d / "src" / "llama.cpp" / "ggml" / "src" / "ggml-backend.cpp").write_bytes(b"ggml bytes")
        (d / "scripts").mkdir()
        (d / "scripts" / "run_server.sh").write_bytes(b"launcher bytes")
        (d / "scripts" / "unrelated.py").write_bytes(b"NOT engine, must stay out")
        # porcelain format: XY<path>
        por = (" M src/llama.cpp/src/llama-context.cpp\n"
               " M src/llama.cpp/ggml/src/ggml-backend.cpp\n"
               " M scripts/run_server.sh\n"
               " M scripts/unrelated.py\n"
               "?? src/llama.cpp/src/new-file.c\n")
        got = dirty_source_hashes(por, root=d)
        cases += [
            ("an engine source under src/llama.cpp/src is hashed",
             got.get("src/llama.cpp/src/llama-context.cpp")
             == hashlib.md5(b"engine bytes").hexdigest()[:16]),
            ("ggml sources are hashed too",
             got.get("src/llama.cpp/ggml/src/ggml-backend.cpp")
             == hashlib.md5(b"ggml bytes").hexdigest()[:16]),
            ("run_server.sh is hashed (it decides which env the engine sees)",
             got.get("scripts/run_server.sh")
             == hashlib.md5(b"launcher bytes").hexdigest()[:16]),
            ("a non-engine script is left out", "scripts/unrelated.py" not in got),
            ("an untracked engine file is left out (it is not in any build)",
             "src/llama.cpp/src/new-file.c" not in got),
            ("a path outside the repo's own tree is not fabricated",
             "src/llama.cpp/llama.h" not in got),
        ]
        # size cap
        (d / "src" / "llama.cpp" / "src" / "huge.c").write_bytes(b"x" * (DIRTY_HASH_MAX_BYTES + 1))
        got2 = dirty_source_hashes(" M src/llama.cpp/src/huge.c\n", root=d)
        cases += [("a source over the size cap is skipped, not half-hashed", got2 == {})]
    real = tree_dirty().get("dirty_src_hashes") or {}
    cases += [
        ("the real tree records at least the gate script itself",
         "scripts/check/m123_oracle_gate.py" in real),
        ("... and every entry is 16 hex chars",
         all(len(v) == 16 and all(c in "0123456789abcdef" for c in v) for v in real.values())),
    ]
    bad = 0
    for name, ok in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1
    print(f"tree-dirty-hashes selftest: {len(cases) - bad}/{len(cases)} passed")
    return 0 if bad == 0 else 1


def selftest_pool_counters() -> int:
    """Positive cases and, more importantly, negatives: a parser that returned zeros for a missing
    line would make "no pressure" and "no reading" identical."""
    real = ("llama_expert_cache: final stats: runtime requests=20740 hits=12772 misses=7968 "
            "(hit rate 61.6%)  prewarm req=0 hit=0 miss=0  resident=6430.62 MiB file_reads=233544 "
            "pread_usec=6267007337\n"
            "llama_expert_cache: miss attribution: compulsory=3746 capacity=4222 (47.0% / 53.0% of "
            "7968)  evictions=7686  layers_distinct_over_slots=5  worst=layer 1 distinct=217 "
            "slots=143\n")
    p = parse_pool_counters(real)
    q = parse_pool_counters("llama_expert_cache: decode/pool (ensure_slot+batch) hits=1/2 (50.0%)\n")
    b = _LOG_RE.search("[log]   /tmp/llama_server_20260919_202004.log（tail -f 同路徑）\n")
    cases = [
        ("real teardown parses", p.get("found") is True),
        ("requests", p.get("requests") == 20740),
        ("hit_pct", p.get("hit_pct") == 61.6),
        ("prewarm_req", p.get("prewarm_req") == 0),
        ("capacity", p.get("capacity") == 4222),
        ("evictions", p.get("evictions") == 7686),
        ("layers_distinct_over_slots", p.get("over") == 5),
        ("worst kept as text", "layer 1" in str(p.get("worst"))),
        ("mid-run log (no teardown) => found False", q.get("found") is False),
        ("... and it names the cause", "teardown" in q.get("reason", "")),
        ("... and carries NO zero counters", "requests" not in q),
        ("banner path stops before the CJK tail",
         bool(b) and b.group("path") == "/tmp/llama_server_20260919_202004.log"),
        ("banner without the line => no match", _LOG_RE.search("[perf] batch=6144\n") is None),
        # Regression guard for the name collision: `pin` was bound twice in main() and the
        # second binding made the summary report ref_pinned=False forever.
        ("no bare `pin` binding survives in this module",
         re.search(r"^\s*pin\s*=", Path(__file__).read_text(encoding="utf-8"), re.M) is None),
    ]
    bad = 0
    for name, ok in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1
    print(f"pool-counter selftest: {len(cases) - bad}/{len(cases)} passed")
    return 0 if bad == 0 else 1


def cap_doc(cfg, note="", extra=None):
    """The provenance document. Same shape the dumper side uses, plus the resolved config."""
    stamp = config_stamp(cfg)
    doc = {
        "kind": "cgc-logits-oracle",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Kept so `knifeedge_matrix.py --oracle-abs` still refuses this as ground truth: every
        # reference on this box comes from run_server.sh, which passes -expert-cache <budget>
        # unless CGC_SERVER_EXPERT_CACHE_OFF=1. cache-ON supports relative invariance only.
        "expert_cache": "off" if cfg["cgcenv"].get("BUDGET") == "0" else "on",
        "expert_cache_source": ("run_server.sh passes -expert-cache <budget> unless "
                                "CGC_SERVER_EXPERT_CACHE_OFF=1, so this is a cache-ON dump and "
                                "must never be used as ground truth (--oracle-abs refuses them)"),
        "profile": cfg["cgcenv"].get("PROFILE"),
        "resolved": stamp,
        "note": note,
    }
    if extra:
        doc.update(extra)
    return doc


def write_cap(path, cfg, note="", extra=None):
    """Write the `<path>.cap` provenance sidecar next to a dump."""
    doc = cap_doc(cfg, note=note, extra=extra)
    Path(str(path) + ".cap").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return doc


def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="prefill250",
                    help="CGC_SERVER_PROFILE for the launch. The default reproduces the "
                         "configuration the M2 reference oracle was dumped under "
                         "(ctx 8192 / batch 6144 / oa_async=0 / PREFILL_STREAM=1 / SLAB_CAP=256), "
                         "which is a precondition for the comparison to mean anything. The "
                         "batch/ubatch half of that is not left to the profile -- it is pinned by "
                         "ORACLE_PINNED_ENV, so a production default change cannot move the "
                         "reference out from under the gate.")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--dump", default="/tmp/m123_oracle_gate.jsonl")
    # [CGC 2026-09-19] The default probe answers "42" and stops, so it always yields 9 oracle
    # records regardless of CGC_LOGITS_ORACLE_FIRST_N. That is enough to see that a knob moved the
    # logits, and NOT enough to say whether the choice survives -- greedy decoding is chaotic, and
    # five steps is five coin flips. Ask for a long generation when the question is "does it
    # matter", and keep the defaults so existing dumps stay reproducible.
    ap.add_argument("--probe-prompt", default=PROBE_PROMPT,
                    help="prompt for the dump run. Default is the short arithmetic probe; pass a "
                         "prompt that generates at length when you need many steps to compare.")
    ap.add_argument("--probe-max-tokens", type=int, default=48,
                    help="max_tokens for the dump run (default 48, matching the reference dumps).")
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=None,
                    help="port the server binds. Default: the resolved profile's CGCENV PORT "
                         "(prefill250 -> 8080). The gate PROBES this port and teardown CLEANS it, "
                         "so it must equal what the profile binds; a different value is refused "
                         "rather than measured against a socket nobody owns.")
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VAL handed to run_server.sh (its allowlist still applies; "
                         "a dropped variable prints nothing, so pass only documented knobs). "
                         "These WIN over ORACLE_PINNED_ENV, but the effective set is resolved and "
                         "printed before anything launches -- changing a pinned knob without "
                         "re-baselining will show up as INVALID COMPARISON, not as a silent pass.")
    ap.add_argument("--no-pin-oracle-env", action="store_true",
                    help=f"do not force {ORACLE_PINNED_ENV} -- take the profile's current defaults "
                         "instead. Only meaningful together with --write-ref, i.e. when you are "
                         "deliberately re-baselining the reference at a new configuration.")
    ap.add_argument("--write-ref", default=None,
                    help="after a successful probe, copy the fresh dump to this path and write "
                         "its .cap. This is how you re-baseline; the copied dump is the run that "
                         "just happened, so its provenance is exact by construction.")
    ap.add_argument("--ref-note", default="", help="free-text note stored in the .cap on --write-ref")
    ap.add_argument("--allow-incomparable", action="store_true",
                    help="downgrade a cross-config comparison from exit 2 to a loud warning. Use "
                         "it only to read a diff you already know is cross-config -- never to "
                         "turn it into a regression verdict.")
    ap.add_argument("--allow-empty-probe", action="store_true",
                    help="do not fail the run when the probe answer is empty. Default is to fail: "
                         "an empty answer plus a structurally-valid dump is the 2026-09-15 "
                         "stale-buffer signature, and comparing it would report a false 1.0.")
    ap.add_argument("--allow-invalid-ref", action="store_true",
                    help="skip dump-validity screening on the reference file. Default is to refuse "
                         "a reference that does not pass the same checks as a fresh dump.")
    ap.add_argument("--allow-ref-drift", action="store_true",
                    help="proceed even though the reference's md5 does not match REF_PINS. Use only "
                         "while re-baselining; the verdict stays meaningful only if you also update "
                         "REF_PINS in the same commit.")
    ap.add_argument("--pool-wait", type=float, default=20.0,
                    help="seconds to wait for the server log's teardown stats before recording the "
                         "probe's pool counters as unreadable (default 20).")
    ap.add_argument("--selftest", action="store_true",
                    help="run the pool-counter parser's and the engine-digest self-tests and exit "
                         "(no server is launched).")
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    ap.add_argument("--teardown-timeout", type=float, default=90.0)
    args = ap.parse_args()

    if args.selftest:
        return (selftest_pool_counters() | selftest_engine_digest()
                | selftest_tree_dirty_hashes())

    tag = args.tag or time.strftime("%Y%m%d_%H%M")
    ref = Path(args.ref)
    if not ref.is_absolute():
        ref = ROOT / ref
    if not ref.exists():
        print(f"ERROR: reference oracle missing: {ref}", file=sys.stderr)
        print(f"       create it with: {Path(__file__).name} --write-ref {args.ref}", file=sys.stderr)
        return 2

    # [P0 2026-09-15] A reference is the most dangerous file in the whole harness: everything is
    # measured against it, so a poisoned reference makes every future run look "identical". Screen
    # it with the same rules as a fresh dump, not with a weaker one.
    if not args.allow_invalid_ref:
        ref_ok, ref_reason = validate_dump(ref)
        if not ref_ok:
            print(f"ERROR: reference {ref} fails dump validity: {ref_reason}", file=sys.stderr)
            print("       Re-baseline from a healthy run, or pass --allow-invalid-ref to override "
                  "(the verdict will be meaningless).", file=sys.stderr)
            return 2

    # The reference's own bytes. A mismatch here is the one failure that would make two runs at
    # different times incomparable while every other artefact (engine digest, config stamp, tree)
    # still matches -- i.e. exactly the silent case, so refuse rather than warn.
    ref_hex = ref_md5(ref)
    ref_pin = REF_PINS.get(ref.name)
    if ref_pin is None:
        print(f"note: no REF_PINS entry for {ref.name}; recording md5 {ref_hex} without checking it")
    elif ref_pin != ref_hex:
        print(f"ERROR: reference {ref.name} has md5 {ref_hex} but REF_PINS pins {ref_pin}.", file=sys.stderr)
        print("       Every M1/M2 number below would be measured against different bytes than the "
              "one this repository has been quoting.", file=sys.stderr)
        if not args.allow_ref_drift:
            print("       Re-baseline deliberately (--write-ref) and update REF_PINS in the same "
                  "commit, or pass --allow-ref-drift to proceed anyway.", file=sys.stderr)
            return 4
        print("       --allow-ref-drift given: proceeding; treat the verdict as a new baseline.",
              file=sys.stderr)

    # Merge the oracle pin with the caller's --env BEFORE anything is printed or resolved, so the
    # line below reports the EFFECTIVE set. A pin that was silently overridden and a pin that was
    # silently dropped would otherwise look identical in the transcript -- and the second one is
    # the failure mode this whole knob exists to prevent.
    merged_env: dict[str, str] = {}
    if not args.no_pin_oracle_env:
        for kv in ORACLE_PINNED_ENV:
            k, _, v = kv.partition("=")
            merged_env[k] = v
    for kv in args.env:
        k, _, v = kv.partition("=")
        merged_env[k] = v
    extra_env = [f"{k}={v}" for k, v in merged_env.items()]

    print(f"=== M1/M2/M3 oracle gate (tag {tag}) ===", flush=True)
    print(f"  profile : {args.profile}", flush=True)
    print(f"  ref     : {ref.relative_to(ROOT)}  ({sum(1 for _ in ref.open())} records)", flush=True)
    if extra_env:
        # NOT named `pin`: that name is taken by the reference md5 check above, and reusing it
        # silently made the summary's `ref_pinned` always False (fixed 2026-09-19).
        pin_note = "" if args.no_pin_oracle_env else f"  (oracle pin: {list(ORACLE_PINNED_ENV)})"
        print(f"  extra   : {extra_env}{pin_note}", flush=True)

    # (1) resolve the launch config FIRST -- before anything is launched, and before kill_servers(getattr(args, "port", None))
    # so the pkill this incurs cannot race our own server.
    try:
        now_cfg = resolve_launch(args.profile, extra_env)
    except Exception as e:  # noqa: BLE001 - harness
        print(f"ERROR: could not resolve the launch configuration: {e}", file=sys.stderr)
        return 2
    now_stamp = config_stamp(now_cfg)

    # [CGC 2026-09-19] The probe target and the server's bind port were two pieces of state that
    # nothing tied together. `--port 8081` against a profile that binds 8080 gave a run that looked
    # healthy -- a dump appeared, because the startup anchor writes one record -- and then sat for
    # the full 300 s ready-timeout polling a socket nobody owned, followed by a teardown that
    # cleaned the wrong port. Take the port from the resolved config (by construction what
    # run_server.sh will bind) and refuse a mismatch; this sits before any kill_servers call.
    profile_port = now_cfg["cgcenv"].get("PORT")
    if profile_port is None:
        print("ERROR: the resolved profile carries no PORT; cannot tie --port to the launch.\n"
              f"       resolved CGCENV keys: {sorted(now_cfg['cgcenv'])}", file=sys.stderr)
        return 2
    try:
        profile_port = int(profile_port)
    except ValueError:
        print(f"ERROR: resolved PORT is not an integer: {profile_port!r}", file=sys.stderr)
        return 2
    if args.port is None:
        args.port = profile_port
        print(f"  port    : {args.port} (from profile {args.profile})", flush=True)
    elif args.port != profile_port:
        print(flush=True)
        print("!" * 74)
        print(f"  PORT MISMATCH: --port {args.port}, but profile {args.profile} binds {profile_port}.")
        print("  The gate PROBES --port and teardown CLEANS --port, so a mismatch would poll a")
        print("  socket nobody owns for the whole ready-timeout and then clean the wrong listener.")
        print("  Refusing to launch. Either drop --port (it defaults to the profile's port) or")
        print("  change the profile's PORT so the two agree.")
        print("!" * 74, flush=True)
        return 2

    # (2) comparability precondition. Both sides resolved by the same code path, so this is an
    # exact comparison, not a curated knob list that can go stale.
    ref_cap = read_cap(ref)
    comparable, cfg_diffs = True, []
    if ref_cap is None:
        print("  WARN    : reference has no .cap sidecar -- comparability cannot be verified. "
              "M1/M2/M3 below are unqualified.", flush=True)
    else:
        ref_stamp = ref_cap.get("resolved")
        if ref_stamp is None:
            print("  WARN    : reference .cap carries no resolved config (pre-dates the sidecar "
                  "format) -- comparability cannot be verified.", flush=True)
        else:
            cfg_diffs = diff_stamp(ref_stamp, now_stamp)
            comparable = not cfg_diffs
            for n in stamp_notes(ref_stamp, now_stamp):
                print(f"  note    : {n}", flush=True)
    if not comparable:
        print(flush=True)
        print("!" * 74)
        print(f"  REFERENCE IS NOT COMPARABLE ({len(cfg_diffs)} numerics-determining difference(s))")
        for d in cfg_diffs[:20]:
            print(f"    {d}")
        if len(cfg_diffs) > 20:
            print(f"    ... and {len(cfg_diffs) - 20} more")
        print("  A cross-config diff is NOT a regression. Re-baseline with:")
        print(f"    python3 scripts/check/m123_oracle_gate.py --profile {args.profile} \\")
        print(f"        --write-ref {ref.relative_to(ROOT)}")
        print("!" * 74, flush=True)

    kill_servers(getattr(args, "port", None))
    dump = Path(args.dump)
    for p in (dump, Path(str(dump) + ".cap"), Path(str(dump) + INVALID_SUFFIX)):
        if p.exists():
            p.unlink()

    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = args.profile
    env["CGC_LOGITS_ORACLE_DUMP"] = str(dump)
    env["CGC_LOGITS_ORACLE_TOPN"] = "8"
    # `run_server.sh` runs the server in the FOREGROUND and `wait`s on it unless CGC_DETACHED=1,
    # in which case it re-execs itself through a detach helper, prints `[detach] server PID=N` and
    # exits 0 (run_server.sh:43 / :1373). Without this the launcher would block until teardown --
    # and with stdout on a pipe it would block forever, because the server inherits the fd.
    env["CGC_DETACHED"] = "1"
    for kv in extra_env:
        k, _, v = kv.partition("=")
        env[k] = v

    # `run_server.sh` DETACHES its child with `set -m` and exits. It must therefore be launched
    # with stdout redirected to a FILE, never a pipe: the detached server inherits the fd, so
    # `subprocess.run(capture_output=True)` (or `communicate()`) blocks on a pipe EOF that only
    # arrives when the SERVER exits -- measured: a 420 s timeout while the server was healthy and
    # already serving. `start_new_session=True` keeps Ctrl-C on our side from reaching it, and we
    # signal the recorded leader PID explicitly during teardown.
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / f"cap_{tag}.json",
               cap_doc(now_cfg, note=f"resolved launch config for gate run {tag} "
                                     f"(provenance, not a reference)"))
    launch_log = RESULT_DIR / f"launch_{tag}.log"
    with launch_log.open("w") as lf:
        proc = subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=str(ROOT), env=env,
                                stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=180.0)
        except subprocess.TimeoutExpired:
            print("ERROR: run_server.sh did not return within 180s (it should detach and exit)",
                  file=sys.stderr)
            kill_servers(getattr(args, "port", None))
            return 2
    out = launch_log.read_text(errors="replace")
    print(out.rstrip(), flush=True)
    # The leader line carries the PID run_server.sh detached -- the only reliable handle, because
    # the banner's other PID-like numbers are the parent's.
    m = re.search(r"server PID=(\d+)", out)
    leader = int(m.group(1)) if m else None
    srv_log = None
    lm = re.search(r"\[log\]\s+([^\s（(]+)", out)
    if lm:
        srv_log = Path(lm.group(1))
    print(f"  leader  : {leader}   server log: {srv_log}", flush=True)

    base = f"http://127.0.0.1:{args.port}/v1"
    t0 = time.time()
    ready = False
    while time.time() - t0 < args.ready_timeout:
        try:
            _get(f"{base}/models", timeout=3.0)
            ready = True
            break
        except Exception:  # noqa: BLE001 - poll loop
            time.sleep(2.0)
    if not ready:
        print(f"ERROR: server did not become ready within {args.ready_timeout}s", file=sys.stderr)
        kill_servers(getattr(args, "port", None))
        return 2
    print(f"  ready   : {time.time() - t0:.0f}s", flush=True)
    # In the transcript, next to the dump path: a dump's provenance must include the probe that
    # produced it, or two dumps of different lengths look interchangeable in the write-up.
    print(f"  probe   : max_tokens={args.probe_max_tokens} "
          f"prompt={args.probe_prompt[:46]!r}", flush=True)

    payload = {"model": "local", "messages": [{"role": "user", "content": args.probe_prompt}],
               "temperature": 0.0, "max_tokens": args.probe_max_tokens}
    try:
        body = _post(f"{base}/chat/completions", payload)
        ans = json.loads(body).get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001 - harness
        print(f"WARN: probe request failed: {e}", flush=True)
        ans = ""
    print(f"  answer  : {ans.strip()[:60]!r}", flush=True)
    time.sleep(1.5)

    # SIGINT so the cache teardown stats are emitted into the server log.
    if leader:
        subprocess.run(["kill", "-INT", str(leader)], check=False)
    else:
        # `pkill -f build/bin/llama-server` is pid-blind and this box runs parallel sessions, so as
        # a teardown it can only ever hit somebody else's server -- the defect http_duo.py:31,285
        # already recorded. Match on the port string instead. The probe is the only thing we own.
        subprocess.run(["pkill", "-INT", "-f", f"--port {args.port}"], check=False)
    # [CGC 2026-09-19] The port-scoped sweep used to live in the `else` branch above, i.e. it was
    # unreachable in the one run that needed it: `leader` came from the `server PID=` line and
    # pointed at a pid that had already exited ("kill: 47306: No such process"), while the real
    # server kept listening. Always sweep the port AFTER the leader kill, never only instead of it.
    time.sleep(1.0)
    held = server_listeners(args.port)
    if held:
        subprocess.run(["kill", "-INT", *held], check=False)
    t1 = time.time()
    while time.time() - t1 < args.teardown_timeout:
        if not server_pids():
            break
        time.sleep(1.0)
    if server_pids():
        print("WARN: server did not exit on SIGINT; killing", flush=True)
        kill_servers(getattr(args, "port", None))
    print(f"  stopped : after {time.time() - t1:.0f}s", flush=True)

    if not dump.exists() or dump.stat().st_size == 0:
        print(f"ERROR: no oracle dump at {dump}; the server never wrote one. "
              f"CGC_LOGITS_ORACLE_DUMP must survive run_server.sh's env allowlist "
              f"(run_server.sh:989).", file=sys.stderr)
        return 2

    # [P0 2026-09-15] Refuse to build a verdict on a run that cannot be trusted. Order matters:
    # both checks sit BEFORE write_cap()/--write-ref, so an invalid run can never become a
    # reference and can never acquire a provenance sidecar that makes it look legitimate.
    if not ans.strip() and not args.allow_empty_probe:
        print(flush=True)
        print("!" * 74)
        print("  INVALID RUN: the probe returned an empty answer.")
        print("  An empty answer with a healthy-looking dump is the exact signature of the")
        print("  2026-09-15 Metal OOM (the graph never ran; the server replied HTTP 200 with")
        print("  stale bytes). M1/M2/M3 below would happily report 1.0 on such a dump, which")
        print("  is precisely the failure mode this gate must not have. Not comparing.")
        print(f"  Server log: {srv_log}")
        print("  Override with --allow-empty-probe only if the emptiness is known-unrelated.")
        print("!" * 74, flush=True)
        return 2

    ok_dump, reason = validate_dump(dump)
    if not ok_dump:
        print(flush=True)
        print("!" * 74)
        print(f"  INVALID DUMP: {reason}")
        print(f"  {dump} is not usable as an oracle reference and was NOT made one.")
        print("  Investigate the engine-side failure first; do not re-run until it is understood.")
        print("!" * 74, flush=True)
        return 2
    print(f"  dump    : {reason}, validated", flush=True)

    # The fresh dump's own provenance, carried with the fresh dump.
    # [CGC 2026-09-30 線A · S2-a] the probe prompt is a CLI argument, so it is NOT part of the
    # config stamp -- that is exactly why a low coverage_pct cannot be cleared today: there is
    # nothing on record to compare against. Record it on every dump and on every re-baseline.
    write_cap(dump, now_cfg, note=f"gate run {tag}",
              extra={"probe_answer": ans.strip(), "validated": True,
                     "probe_prompt_md5": hashlib.md5(args.probe_prompt.encode()).hexdigest()})

    # (3) re-baseline in one step when asked. Done BEFORE the comparison so a --write-ref run
    # also reports what the new baseline looks like against the old one.
    if args.write_ref:
        dst = Path(args.write_ref)
        if not dst.is_absolute():
            dst = ROOT / dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dump, dst)
        # a stale sidecar next to the destination would make the *new* reference look invalid
        stale = Path(str(dst) + INVALID_SUFFIX)
        if stale.exists():
            stale.unlink()
        write_cap(dst, now_cfg, note=args.ref_note or f"re-baselined by gate run {tag}",
                  extra={"probe_answer": ans.strip(), "probe_prompt_md5": hashlib.md5(args.probe_prompt.encode()).hexdigest()})
        print(f"  baseline: wrote {dst.relative_to(ROOT)} + .cap (this run is the new reference)",
              flush=True)

    report = RESULT_DIR / f"oraclecmp_{tag}.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run([sys.executable, str(COMPARE), "--a", str(ref), "--b", str(dump),
                         "--report", str(report)],
                        cwd=str(ROOT), capture_output=True, text=True)
    print(cp.stdout, flush=True)
    if cp.stderr.strip():
        print(cp.stderr, file=sys.stderr, flush=True)
    tee(f"cmp_{tag}.log", cp.stdout)

    if not report.exists():
        print("ERROR: compare produced no report", file=sys.stderr)
        return 2
    d = json.loads(report.read_text())
    met = d["metrics"]
    m1r, m2r, m3r = (met["numeric_identity"]["rate"], met["decision_agreement"]["rate"],
                     met["topk_set_agreement"]["rate"])
    pool_counters = pool_counters_for_run(launch_log,
                                          wait_s=float(getattr(args, "pool_wait", 20.0)))

    # How much of the dump actually had a counterpart in the reference. A shortfall means record
    # keys collided across DIFFERENT token sequences -- which is what happens when the probe prompt
    # differs from the one the reference was taken with -- and then M1/M2 compare unrelated content
    # while `comparable` still says True, because the prompt is a CLI argument and not part of the
    # config stamp. Recorded, not inferred, because the symptom (a low M1) reads like a regression.
    try:
        dump_records = sum(1 for _ in dump.open())
    except OSError:
        dump_records = 0
    n_common = met["numeric_identity"]["n"]
    coverage_pct = (100.0 * n_common / dump_records) if dump_records else None
    # [CGC 2026-10-07] Named once: `comparability_defects` below and the summary both read it, and
    # it used to be recomputed inline twice. `None` means "the sidecar carries no prompt md5", which
    # is UNVERIFIABLE rather than a mismatch -- the defect list treats it as one, fail-closed.
    _prompt_md5 = hashlib.md5(args.probe_prompt.encode()).hexdigest()
    probe_prompt_match = ((ref_cap or {}).get("probe_prompt_md5") is None
                          or (ref_cap or {})["probe_prompt_md5"] == _prompt_md5)
    # [CGC 2026-10-07] Fold the three extra preconditions into the verdict itself. `comparable` was
    # conjunctive over the config stamp alone, which is why eight 2026-10-02 summaries could all say
    # `comparable: true, config_diffs: []` while spanning two reference files. Fail-closed: an
    # absent sidecar field yields None => not comparable, because "unverified" must not read as
    # "verified". `config_diffs` is unchanged and still reported -- this only ever removes
    # qualifications, it can never invent a difference.
    _defects = [d_ for d_ in (
        (None if ref_pin == ref_hex else
         f"ref bytes differ from REF_PINS ({ref_hex} != {ref_pin})"),
        (None if (ref_cap or {}).get("probe_prompt_md5") is not None else
         "reference carries no probe_prompt_md5 (pre-dates the sidecar format)"),
        (None if probe_prompt_match else "probe prompt differs from the reference's"),
        (None if not dump_records or not n_common or n_common == dump_records else
         f"only {n_common}/{dump_records} dump records found a counterpart (key collision)"),
    ) if d_]
    comparable = comparable and not _defects
    summary = {
        "tag": tag, "profile": args.profile, "ref": str(ref), "ref_md5": ref_hex,
        "ref_pinned": (ref_pin == ref_hex), "dump": str(dump),
        # [CGC 2026-09-30 §R6 witness] The EFFECTIVE override set (`--env` merged over the oracle
        # pin). Without it a summary records what the numbers were but not WHICH ARM produced them,
        # so the paired witness checker (`scripts/check/r6_witness.py`) would have to trust a
        # hand-written label -- exactly the class of hand-written copy this file's docstring forbids.
        # Additive, so summaries written before today stay readable; the checker fail-closes when
        # the field is absent (it refuses to guess which side is the witness arm).
        "extra_env": dict(merged_env),
        "probe_prompt_md5": _prompt_md5,
        "ref_probe_prompt_md5": (ref_cap or {}).get("probe_prompt_md5"),
        "probe_prompt_match": (None if (ref_cap or {}).get("probe_prompt_md5") is None
                               else probe_prompt_match),
        "dump_records": dump_records,
        "coverage_pct": coverage_pct,
        "probe_answer": ans.strip(),
        "m1_numeric_identity": f"{met['numeric_identity']['equal']}/{met['numeric_identity']['n']}",
        "m2_decision_agreement": f"{met['decision_agreement']['equal']}/{met['decision_agreement']['n']}",
        "m3_topk_set_agreement": f"{met['topk_set_agreement']['equal']}/{met['topk_set_agreement']['n']}",
        "n_compared": met["numeric_identity"]["n"],
        "cross_tab": d["cross_tab"],
        # [CGC 2026-10-07] `comparable` used to mean "the two resolved CONFIG stamps agree", which
        # is necessary but NOT sufficient, and the 2026-10-02 overlap batch proved it: eight runs
        # all reported comparable=true / config_diffs=[] while being split across TWO different
        # reference files (md5 a3fbfb0c vs 0cf5c593). Their M1 readings ranged 6/222 .. 222/222 and
        # were NOT comparable -- they answered different questions. Three defects made the config
        # stamp insufficient, each of which silently turns "incomparable" into a confident number:
        #   (1) the reference's own BYTES (this run was measured against a different file),
        #   (2) the probe prompt (a CLI arg, deliberately not in the config stamp),
        #   (3) key collision in the dump, the symptom being coverage_pct far below 100.
        # Each is recorded first and folded in after, so `comparable` stays fail-closed on old
        # summaries that predate the fields (None => not comparable, never a free pass).
        "comparable": comparable,
        # Per-defect verdicts, so a reader can tell WHICH precondition broke instead of only that
        # the conjunction did. `config_diffs` alone used to be the sole evidence. Computed once
        # above and folded into `comparable`; repeated here purely so the summary is self-contained.
        "comparability_defects": (_defects or None),
        "config_diffs": cfg_diffs,
        "ok": bool(comparable and m1r == 1.0 and m2r == 1.0),
        "engine_digest": engine_digest(),
        "pool_counters": pool_counters,
        "tree": tree_dirty(),
        "report": str(report),
    }
    (RESULT_DIR / f"summary_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    pc = summary["pool_counters"]
    if pc.get("found"):
        print(f"  probe pool: requests={pc['requests']} hit={pc['hit_pct']}% "
              f"capacity={pc.get('capacity')} evictions={pc.get('evictions')} "
              f"over={pc.get('over')} prewarm={pc['prewarm_req']}")
    else:
        print(f"  probe pool: UNREADABLE -- {pc.get('reason')}")
    if coverage_pct is not None and coverage_pct < 100.0:
        # [CGC 2026-09-30 線A · S2-a] three-way, because "shortfall" has two very different causes
        # and only one of them invalidates the verdict. `unknown` stays loud on purpose: a legacy
        # reference that never recorded its prompt cannot clear the doubt, so it must not read as
        # if it had been checked.
        _rpm = (ref_cap or {}).get("probe_prompt_md5")
        if _rpm is None:
            print(f"  \u26a0 coverage: only {n_common} of {dump_records} dump records "
                  f"({coverage_pct:.1f}%) had a counterpart in the reference, and the reference's "
                  f"`.cap` records NO probe_prompt_md5 (pre-dates 2026-09-30) => PROMPT UNKNOWN: "
                  f"the prompt-mismatch explanation for the shortfall CANNOT be cleared, so the "
                  f"M1/M2 verdicts above are EVIDENCE, not a decision. Re-baseline (--write-ref) to "
                  f"make it decidable.")
        elif _rpm != hashlib.md5(args.probe_prompt.encode()).hexdigest():
            print(f"  \u26a0 coverage {coverage_pct:.1f}% and the reference was dumped with a "
                  f"DIFFERENT probe prompt (ref {_rpm} vs now "
                  f"{hashlib.md5(args.probe_prompt.encode()).hexdigest()}) => the M1/M2 verdicts "
                  f"above compare unrelated content. CANNOT_JUDGE.")
        else:
            print(f"  \u26a0 coverage {coverage_pct:.1f}% with an IDENTICAL probe prompt "
                  f"({_rpm}) => the shortfall is NOT a prompt artefact: the record keys diverged "
                  f"because this arm produced a different token sequence. That IS the finding, and "
                  f"the M1/M2 verdicts above are readable.")
    print("=" * 74)
    if not comparable:
        print(f"GATE {tag}: INVALID COMPARISON -- reference was dumped under a different "
              f"numerics-determining configuration ({len(cfg_diffs)} diffs).")
        print(f"  observed M1={summary['m1_numeric_identity']}  M2={summary['m2_decision_agreement']}  "
              f"M3={summary['m3_topk_set_agreement']}  n={summary['n_compared']} "
              f"(printed for information, NOT a verdict)")
    else:
        print(f"GATE {tag}: {'PASS' if summary['ok'] else 'FAIL'}   "
              f"M1(bit-identical)={summary['m1_numeric_identity']}  "
              f"M2(argmax)={summary['m2_decision_agreement']}  "
              f"M3(topk)={summary['m3_topk_set_agreement']}  n={summary['n_compared']}")
        print(f"cross-tab: {d['cross_tab']}")
    print(f"summary  : {RESULT_DIR / f'summary_{tag}.json'}")
    print("  build   : " + "  ".join(
        f"{k}={v['md5']}" for k, v in summary["engine_digest"].items()) or "  build   : (none)")
    tr = summary["tree"]
    if tr:
        print(f"  tree    : {tr.get('head')} dirty_tracked={tr.get('dirty_tracked')}")
    print("=" * 74)
    if not comparable:
        return 2 if not args.allow_incomparable else (0 if summary["ok"] else 1)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
