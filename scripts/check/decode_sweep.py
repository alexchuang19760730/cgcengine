#!/usr/bin/env python3
"""Decode arm sweep: restart the server per arm, run decode_bench, harvest cache final stats.

Why a driver and not a shell loop: every arm needs the SAME request shape, the SAME pool
accounting read back, and the server has to be SIGINT'd (not SIGKILL'd) so the expert-cache
teardown prints `final stats` / `miss attribution` / `read shape`. Those three lines are the
only way to attribute decode time to (miss count) vs (per-miss IO latency) — and they are
exactly what is missing from every decode number quoted in the docs so far.

Arms are declared below. Each is a dict of extra env vars layered on the base profile.
The driver is resumable: rows accumulate into --json keyed by tag, so a partial sweep is
still useful and re-running skips nothing already recorded (unless --force).

Usage:
    python3 scripts/check/decode_sweep.py --arms baseline,mtp-off,pool-4g --rounds 5
    python3 scripts/check/decode_sweep.py --report /tmp/decode_sweep.json
"""
# PEP 563: `int | None` in an annotation is evaluated at def time before 3.10, so importing this
# module on the default system python3 (3.9) raised `TypeError: unsupported operand type(s) for |:`. That
# is not a local inconvenience: llama_bench_matrix.harvest_bench_stats() imports this file to read a
# shape's stats AFTER llama-bench has already run the arm, so every prod_matrix / matrix arm came back
# rc=1 with a full measurement sitting in its stderr log and no rows in its JSON.
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_DIR = os.path.join(ROOT, "Backup", "cgc_logs")
SERVER_MATCH = "build/bin/llama-server"

# In-band thermal reading, imported so there is ONE parser and one "unreadable is not zero"
# rule (thermal_pressure.py). Decode had none of this until now, which is why the same arm on
# the same day read 6.6 / 9.75 / 10.24 / 16.17 / 6.95 t/s with nothing to attribute it to.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thermal_pressure as tp  # noqa: E402


def build_fingerprint():
    """The exact binaries a row's numbers belong to, discovered by GLOB over every shared library
    the run loads (`libggml*.dylib`, `libllama*.dylib`) plus the server binary itself.

    Why every row needs it: a sweep row without a fingerprint cannot be compared to anything --
    not to another arm, and not to itself a day later after a rebuild. The emitter in
    agent_harness/engine_loop REFUSES a row without one, so recording it here is what makes a run
    quotable at all. Recorded per row rather than per file so a rebuild mid-sweep is visible
    instead of silently averaging two different binaries.

    [CGC 2026-09-15 measurement-hygiene fix] Until now this hashed a hand-picked THREE files:
    llama-server, libggml-metal and libllama. That list omits libggml-base, which is where
    ggml-backend.cpp lives -- i.e. the scheduler and the `CGC_OA_ASYNC` gate -- and it omits
    libggml-cpu, which is where the mul_mat_id crash signature of the 20:18 SIGSEGV lives
    (build/ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o proves the mapping against the
    CMake object layout, not by inference). Consequence, measured: the pair recorded at 21:34
    (the OA_ASYNC gate still read presence, so both "noasync" arms bit-copied their segmented
    counterparts) and the pair recorded at 21:41 (gate value-aware, p25-slotgpu-noasync =
    {ff68c5a2}) carried the SAME server/metal/llama triple -- two runs whose scheduler code
    differed, stamped comparable by the very tool whose rule is "compare only within one
    fingerprint". See lessons.jsonl eng-mh-0007.

    Two things this shape buys, beyond coverage:
      * keys are version-free (`libggml-base`, not `libggml-base.0.19.0`), so a version bump
        cannot silently degrade an entry to "missing" (which is what the old literal filenames
        would have done) nor silently re-point to a different file;
      * the KEY SET is itself the fingerprint's shape. Old rows carry 3 keys, new rows carry 8;
        a comparison that mixes them is flagged by the key set rather than by nobody noticing.
        Rows recorded before this change are therefore not comparable to rows after it on the
        libggml-base dimension -- which is exactly the fact this change exists to expose."""
    fp = {}

    def add(key, path):
        try:
            with open(path, "rb") as fh:
                fp[key] = hashlib.md5(fh.read()).hexdigest()[:12]
        except OSError:
            fp[key] = "missing"

    bin_dir = os.path.join(ROOT, "src/llama.cpp/build/bin")
    add("server", os.path.join(bin_dir, "llama-server"))
    for pat in ("libggml*.dylib", "libllama*.dylib"):
        for path in sorted(glob.glob(os.path.join(bin_dir, pat))):
            base = os.path.basename(path)
            # only the real versioned files: `libggml-base.0.19.0.dylib`, not the `.dylib` /
            # `.0.dylib` symlinks to them (same bytes three times is not more information).
            m = re.match(r"^(lib[a-z0-9]+(?:-[a-z0-9]+)*)\.\d", base)
            if m is None:
                continue
            add(m.group(1), path)
    return fp


# tag -> extra env. Base = whatever `--profile` says (default `prefill250`), so the arms below are
# only "the variable under test" RELATIVE TO THAT PROFILE -- and the profile is not a passive
# default, it pins knobs. Two consequences worth stating, because both have already bitten:
#
#   * The `p25-*` arms and the block from `p25-mmap` down assume you passed `--profile prod25`.
#     Run them on the default and they test prod25-ish knobs on a prefill250 base (ctx 8192,
#     -b 5632, SLAB_CAP 256) -- a different experiment, not a wrong one, but not the recorded one.
#   * `spac-on` is degenerate on BOTH bases now: `prod25` has always pinned CGC_SPAC=1, and from
#     2026-09-16 `prefill250` pins it too (run_server.sh:335, the prefill/decode unification). For
#     a real SPAC A/B use `--profile prod25 --arms p25-nospac,baseline`, or the equivalently named
#     `prefill250:CGC_SPAC=0` arm that `llama_bench_matrix.py` accepts.
ARMS = {
    "baseline":      {},
    "mtp-off":       {"CGC_SERVER_MTP": "0"},
    # The pool-size curve, as named in NEXT_ACTIONS task 3 (8 GiB control vs 3/4/6 GiB). Each is a
    # different `-exper-cache`, i.e. a different SLOT COUNT -- measured 143 slots at 8 GiB,
    # 71 at 4 GiB -- so a smaller pool is not "the same run with less RAM", it is a different
    # cache. Pair them in one window with a reversed arm order; do not read them across windows.
    "pool-3g":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "3221225472"},
    "pool-4g":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "4294967296"},
    "pool-6g":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "6442450944"},
    "pool-10g":      {"CGC_SERVER_EXPERT_CACHE_BYTES": "10737418240"},
    # budget 0 = expert cache off entirely: every expert is read where the loader put it, so
    # decode has no pool, no remap and no fill. This is the "zero-IO-work" upper bound and the
    # single most informative arm -- if decode does NOT go up here, the bottleneck is compute,
    # not the cache, and no amount of pool/fill tuning can reach 25 t/s.
    #
    # [2026-09-15 fix] This arm used to set CGC_SERVER_EXPERT_CACHE_BYTES=0, which does NOT
    # disable the cache: run_server.sh still appends `-expert-cache 0`, the L4 load-time path
    # still runs (skip_load on, CPU placement, ne02 shrunk), and the resulting tensor geometry
    # disagrees with what the hook expects -- measured: every request HTTP 500, which is why the
    # arm never produced a number. The only launch that is actually cache-free OMITS the flag,
    # which is what CGC_SERVER_EXPERT_CACHE_OFF=1 does (run_server.sh:827). Same bug and same fix
    # in `nocache-phase` below.
    "nocache":       {"CGC_SERVER_EXPERT_CACHE_OFF": "1"},
    # P1 prefill-protect. The build default is OFF and run_server.sh never set it, so every
    # server-profile decode number recorded so far is the "after a prefill" case. The code
    # comment at llama-context.cpp:4987 records 22.2 t/s steady-state with it on.
    "protect-on":    {"CGC_PREFILL_PROTECT": "1"},
    "protect-noprefetch": {"CGC_PREFILL_PROTECT": "1", "CGC_SERVER_NO_PREFETCH": "0"},
    # ---- tested on the prod25 base profile (see --profile) ----
    # `--no-mmap` makes the dense tensors anonymous pages that the OS may swap; the 09-12 note
    # in run_server.sh measured swap used 5.8GB / free 61MB as a top speed cause. This machine
    # is at 3.1GB of a 4GB swap file, so this is a live hypothesis, not a theory.
    "p25-mmap":      {"CGC_SERVER_LOAD_MODE": "mmap"},
    "p25-noasync":   {"CGC_SERVER_OA_ASYNC": "0"},
    "p25-nospac":    {"CGC_SPAC": "0"},
    "p25-mtpoff":    {"CGC_SERVER_MTP": "0"},
    "p25-mmap-mtpoff": {"CGC_SERVER_LOAD_MODE": "mmap", "CGC_SERVER_MTP": "0"},
    "p25-noident":   {"CGC_MM_BITIDENT": "0"},
    "prefetch-on":   {"CGC_SERVER_NO_PREFETCH": "0"},
    "spac-on":       {"CGC_SPAC": "1"},
    "verify-off":    {"CGC_SERVER_VERIFY_DECODE": "0"},
    "draft-off":     {"CGC_SERVER_DRAFT_DECODE": "0"},
    # ---- 2026-09-15: MTP acceptance is the whole ballgame ----
    # 09-05 (27.71 t/s) amortised 622 tokens over 156 decode steps: draft acceptance 0.9974,
    # mean len 3.99, 144 ms/step. Today prod25-mtpoff spends only 96 ms/step but yields ONE
    # token per step (10.43 t/s), and prod25 with MTP on is SLOWER still (7.6-8.0 t/s) while
    # emitting a degenerate repeat. So per-step cost did NOT regress -- MTP did.
    # The three `bit-identical` pillars were flipped from default-0 to default-1 earlier today
    # and prod25 forces all three; MTP_NO_WARMUP in particular touches the draft context's
    # init, so it is the prime suspect. Test them one at a time, plus DRY (which penalises
    # repetition on the TARGET sampler only -- a systematic target/draft mismatch).
    "p25-mtp-on":    {},                       # same-session bare prod25 reference
    # [CGC 2026-09-18 §EN-149 path 2] S1 to production. Three arms, all on the production shape
    # (prod25 + Nail + MTP on), so `p25-s1-prod` differs from `p25-mtp-on-diag` by ONE variable
    # (CGC_SLOT_TABLE_GPU) and `p25-s1-keepleaf` adds one more (CGC_S1_KEEP_LEAF), which keeps the
    # host leaf consumed and therefore keeps the per-layer drain. That trio separates S1's two
    # halves: (diag -> keepleaf) = what the in-graph gather COSTS, (keepleaf -> s1prod) = what
    # removing the leaf/drain PAYS. Without the middle arm a null result is unattributable.
    "p25-mtp-on-diag":   {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1"},
    "p25-s1-keepleaf":   {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_KEEP_LEAF": "1"},
    "p25-s1-prod":       {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_SLOT_TABLE_GPU": "1"},
    # [CGC 2026-09-18 §EN-152] Two cheap readings in one run.
    #   p25-mmd-trace : which mul_mat family each MUL_MAT takes (CGC_MM_DBG=1). attn_qkv / attn_gate
    #                   / ssm_out / ffn_*_shexp are all IQ4_XS, and IQ4_XS appears in NEITHER
    #                   small-batch list (ggml-metal-ops.cpp:2479: group A needs one of
    #                   IQ4_NL/etc with ne11 in [2,8]; group B needs Q2_K..Q6_K with ne11 in [4,8]),
    #                   while mul_mm needs ne11 > ne11_mm_min = 8. So at decode/verify (ne11 in
    #                   {1,2,4}) every dense projection should fall through to the plain mul_mv.
    #                   This arm verifies that from the instrument, not from the source.
    #   p25-s1-churn  : CGC_S1_TABLE_CHURN=1 turns on the consumed-subset churn counter that the S1
    #                   slot-table line reports as "(not instrumented)". That number is the
    #                   precondition for "publish all layers' tables once before the step" (§EN-150).
    "p25-mmd-trace":     {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1", "CGC_MM_DBG": "1"},
    # [CGC 2026-09-18 §EN-154] (a) PRICING `CGC_MM_BITIDENT=0`. Everything else from today says
    # the dense M<=8 GEMVs are cheap in *bytes per call* but are read ne11 times: mul_mv sets
    # `nr1 = 1` (ggml-metal-device.cpp:848), so one threadgroup computes ONE token column and
    # every weight byte is fetched ne11 times. The small-batch mat-mv family is the fix that
    # already exists -- it derives r1ptg from ne11 (nxpsg/r1ptg table in ggml_metal_op_mul_mat)
    # so one threadgroup covers all ne11 columns and the weight is read ONCE -- and BITIDENT=1
    # is the only reason it is unreachable. This pair prices that, with NO source change.
    #   * BOTH arms carry identical diagnostics, including CGC_MM_DBG, so the family choice is
    #     self-attesting: the control arm must print ONLY `mul_mv` and the test arm must print
    #     `small-batch` at exactly the calls predicted from the existing MMDBG log.
    #   * `-on` PINS the value 1 instead of relying on the run_server.sh default, so a future
    #     change of that default cannot silently collapse this pair into a no-op (a null result
    #     would then be unattributable). `-off` pins 0; run_server.sh:2014-2017 forwards nothing
    #     for "0", so the child's getenv sees the literal "0" -> `e[0] != '1'` -> old path.
    # Prediction from Backup/cgc_logs/llama_server_20260918_133430.log (26401 mul_mv calls, all
    # group A, group B EMPTY): 9747/26401 calls flip (f32 9100 = the routers, q8_0 555, bf16 200),
    # 61.4 -> 8.6 GiB of eligible weight traffic (-52.8 GiB of the run's 256.3 GiB total). IQ4_XS
    # (63.0% of that traffic) is in NEITHER small-batch list and cannot flip -- that is (b).
    "p25-mmflip-on":     {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_MM_DBG": "1", "CGC_MM_BITIDENT": "1"},
    "p25-mmflip-off":    {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_MM_DBG": "1", "CGC_MM_BITIDENT": "0"},
    "p25-s1-churn":      {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_TABLE_CHURN": "1"},
    # [CGC 2026-09-18 §EN-144] The FIRST same-checkpoint MTP-off arm. Every earlier server-side
    # "MTP off" arm set only CGC_SERVER_MTP=0, and run_server.sh:154 turns that into
    # MODEL_DEFAULT=$Q36 = Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -- a DIFFERENT checkpoint with no nextn
    # head (verified verbatim with CGC_DUMP_ENV=1, not inferred). So every historical server-side
    # MTP off/on comparison (12.62 vs 9.82, 6.14 vs 5.81) crossed TWO variables at once. This arm
    # pins CGC_SERVER_MODEL to the production Nail checkpoint, so the ONLY difference from
    # p25-mtp-on is CGC_SERVER_MTP. Treat every pre-existing "MTP off" arm as a cross-checkpoint
    # probe, and diff the CGCENV MODEL line before quoting any MTP off/on pair.
    "p25-nail-mtpoff": {"CGC_SERVER_MODEL": "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
                        "CGC_SERVER_MTP": "0"},
    "p25-pillars-off": {"CGC_SERVER_MTP_NO_WARMUP": "0",
                        "CGC_SERVER_NO_SEQ_RM_PROBE": "0",
                        "CGC_MM_BITIDENT": "0"},
    "p25-nowarmup":  {"CGC_SERVER_MTP_NO_WARMUP": "0"},
    "p25-probe-on":  {"CGC_SERVER_NO_SEQ_RM_PROBE": "0"},
    "p25-nodry":     {"CGC_SERVER_DRY_MULTIPLIER": "0"},
    # ---- the money arm ----
    # Measured on 09-15: MTP-on runs are 88.4% CAPACITY misses (68298 of 77266) over a 143-slot
    # pool, 553.8 s of pread inside a 92.6 s wall, plus a `prewarm req=22902 hit=0 miss=22902`
    # pass that is 100% wasted work and evicts 22902 slots on the way. So CAPACITY misses --
    # not compulsory traffic, not per-read latency (1941us vs 1739us on 09-05, +12%) -- are what
    # kills decode.
    #
    # CGC_PREFILL_PROTECT exists for exactly this and has NEVER been reachable from a server
    # profile: cgc_prefill_protect_on() is a bare getenv()!=nullptr test, and run_server.sh only
    # added it to the allowlist today, so every server decode number on record is the
    # "after a prefill" case. The mechanism is direct -- a decode HIT stamps
    # slot_decode_reserved[l][slot]=1 (llama-expert-cache.cpp:748-752), and pick_slot will not
    # evict a protected slot -- so it converts the hot decode working set into pinned residency.
    # The in-code measurement is 22.2 t/s steady-state with it on vs 8.1-8.9 after a prefill.
    "p25-protect":   {"CGC_PREFILL_PROTECT": "1"},
    "p25-protect-mtpoff": {"CGC_PREFILL_PROTECT": "1", "CGC_SERVER_MTP": "0"},
    # 09-05's operating point, rebuilt: 4 GiB pool (71 slots) WITH the L0/L1 tier split active.
    # run_server.sh pins SOFT_POOL_L0/L1 to 0, so the partition cannot be enabled from a profile
    # even though 09-05 ran with L0=32 L1=32. This arm exists to test whether the partition --
    # not the capacity -- is what collapsed the miss rate.
    "p25-softpool":  {"CGC_SERVER_EXPERT_CACHE_BYTES": "4294967296",
                      "CGC_SOFT_POOL_L0": "32", "CGC_SOFT_POOL_L1": "32"},
    # Speed-max combination. The 2026-09-15 sweep isolated two independent wins:
    #   pillars-off  -> reads 285279 -> 136290 (-52%), 6.48 -> 7.83 t/s, and loopiness 0.0
    #                   (the ONLY MTP-on arm whose answer was not a repeated sentence)
    #   nodry        -> draft acceptance 0.735 -> 0.812, 6.48 -> 7.48 t/s
    # They touch different terms (IO volume vs acceptance), so they should compose. NOTE this arm
    # deliberately gives up the three bit-identical pillars, so it is a SPEED datapoint only --
    # it cannot be a candidate production config until the bit-identical gate passes.
    "p25-nodry-pillars": {"CGC_SERVER_DRY_MULTIPLIER": "0",
                          "CGC_SERVER_MTP_NO_WARMUP": "0",
                          "CGC_SERVER_NO_SEQ_RM_PROBE": "0",
                          "CGC_MM_BITIDENT": "0"},
    # Fill-pool concurrency. LLAMA_EXPERT_CACHE_WORKERS was the literal 8 in run_server.sh and
    # had never been swept, yet it is the only knob controlling how many expert preads are in
    # flight. The measurements say latency -- not bandwidth -- is the binding term, so this is
    # the cheapest untested lever on the whole path.
    "p25-workers16": {"CGC_SERVER_WORKERS": "16"},
    "p25-workers32": {"CGC_SERVER_WORKERS": "32"},
    # ---- P0/P1 phase decomposition 2026-09-15 ----
    # These arms exist to ATTRIBUTE the time inside one decode step, not to certify a throughput
    # number, so one round each is the intent. CGC_PHASE_TIMING adds the per-step
    # build/alloc/inputs/compute split (CGC-PHASE, every 32 steps) and CGC_M2_PROFILE adds one
    # line per (layer,kind) slab fill carrying pool-vs-disk bytes and ms. Together with the
    # teardown counters (pread_usec / fill_batch_usec / file_reads) divided by n_decoded, the
    # step cost decomposes into: disk syscalls, pool gather, GPU, and everything else.
    #
    # `nocache` is the decisive control and the reason this set is worth a run: with no pool,
    # no remap and no fill, the only remaining costs are graph build + Metal. If decode does
    # NOT improve there, the ~100 ms/token is NOT an IO/cache problem and no pool or fill knob
    # can reach the budget.
    "p25-mtpoff-phase":  {"CGC_SERVER_MTP": "0", "CGC_PHASE_TIMING": "1", "CGC_M2_PROFILE": "1"},
    "nocache-phase":     {"CGC_SERVER_EXPERT_CACHE_OFF": "1",
                          "CGC_PHASE_TIMING": "1", "CGC_M2_PROFILE": "1"},
    # MTP-on counterpart of p25-mtpoff-phase: MTP is the one arm where the capacity-miss flood
    # is 88% (measured 09-15), so its fill_wait should be dramatically larger. The pair is what
    # makes the fill_wait counter informative -- a single arm cannot tell a clean 0 from a broken
    # instrument.
    "p25-phase-mtp":     {"CGC_PHASE_TIMING": "1", "CGC_M2_PROFILE": "1"},
    "p25-phase-w32":     {"CGC_SERVER_MTP": "0", "CGC_SERVER_WORKERS": "32",
                          "CGC_PHASE_TIMING": "1", "CGC_M2_PROFILE": "1"},
    # n_cb sweep, phase-timed. `compute` is graph_compute()'s wall time and is the ONLY phase
    # left once build/alloc/inputs (~0.13 ms) and fill_wait (~0) are subtracted. The question this
    # sweep answers is what `compute` IS: MTLCommandBuffer encoding is CPU-side and parallelises
    # across n_cb command buffers, GPU execution does not. So if compute falls steeply from
    # n_cb=1 to 8 and is flat after, the remaining term is GPU execution (Metal kernel work is
    # the lever). If it keeps falling past 8, or if adding WORKER threads (not encode threads)
    # raises it -- measured: workers=32 pushed compute 104 -> 156 ms -- then `compute` is
    # CPU-side contention and no further kernel micro-optimisation can reach it.
    "p25-cb1-phase":     {"CGC_SERVER_MTP": "0", "CGC_SERVER_N_CB": "1",
                          "CGC_PHASE_TIMING": "1"},
    "p25-cb2-phase":     {"CGC_SERVER_MTP": "0", "CGC_SERVER_N_CB": "2",
                          "CGC_PHASE_TIMING": "1"},
    "p25-cb16-phase":    {"CGC_SERVER_MTP": "0", "CGC_SERVER_N_CB": "16",
                          "CGC_PHASE_TIMING": "1"},
    # [2026-09-15] The decisive arm. CGC_DECODE_PROFILE is the fork's built-in M0 decode profiler
    # (ggml-backend.cpp:1965, "CGC-DECPROF"): per step it splits the segmented dispatch into
    #   wait   = time blocked on the previous segment's GPU completion
    #   cb     = eval-callback + encode time for the segment
    #   submit = command-buffer submit/commit time
    # plus a per-layer attribution. `compute` (CGC-PHASE) is n_cb-invariant at ~101 ms, so the
    # question is what those 101 ms are: GPU execution (wait) or CPU-side dispatch (cb/submit).
    # No amount of Metal kernel work can reach a `cb`-dominated step, and no amount of node
    # fusion can reach a `wait`-dominated one. CGC_DECODE_PROFILE_ALL=1 adds every layer.
    "p25-decprof":       {"CGC_SERVER_MTP": "0", "CGC_DECODE_PROFILE": "1",
                          "CGC_DECODE_PROFILE_ALL": "1"},
    # [CGC 2026-09-18 engine line] The layer-0 question. The per-layer DECPROF attribution, taken as
    # a median over 47 steady steps of the MTP-ON production shape (8 GiB pool,
    # Backup/phase_decomp/m3_layergpu2_20260917.json), gives layer 0 gpu=17.88 / union=9.25 ms
    # against a median of 2.47 / 1.77 -- 7.2x and 5.2x, with all other 40 layers inside
    # 1.37-2.87 / 1.19-1.82. Layer 0's host-side numbers are NORMAL (wait 1.49, cb 1.55,
    # submit 0.14), so the excess is GPU-side, and the two candidate mechanisms need separating:
    #   (a) segment 0 really carries extra work (the input-embedding split, a non-pooled blk.0,
    #       or a different node count), or
    #   (b) segment 0 is a normal layer that pays the head-of-step launch ramp, because the
    #       segmented dispatcher drains the queue at every segment boundary and the step therefore
    #       starts on an idle GPU.
    # GGML_SCHED_DEBUG=2 prints every split's backend and full node list, so counting segment 0's
    # nodes against segment 1's separates (a) from (b) with no new instrument. MTP is left ON
    # deliberately: that is the production setting the 7.2x was measured in.
    "en-sched":          {"GGML_SCHED_DEBUG": "2", "CGC_DECODE_PROFILE": "1",
                          "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 engine line] The shape scan that settles L0. `docs/L0_ATTRIBUTION_2026-09-18.md`
    # found the L0 anomaly (gpu 7.2x / union 5.2x the per-layer median, 47 steady steps) in the
    # ntok=1 layout that `llama-bench` produces -- llama-bench cannot speculate, so its decode steps
    # are single-token -- but NOT in the ntok=2/4 layout that a real MTP verify produces. Two
    # candidate mechanisms remain (segment 0 carries extra work vs segment 0 pays the head-of-step
    # launch ramp) and they need the same statistic in both shapes before either can be named.
    #
    # DECISION RULE, written before the run:
    #   ratio = L0 gpu / median(other 39 layers' gpu), per arm, median over steady steps
    #     ratio >= 3 in ntok=1 only      -> layout-dependent (candidate: segment 0's extra work)
    #     ratio >= 3 in BOTH             -> layer-independent (candidate: head-of-step ramp)
    #     ratio <  2 in both             -> no reproducible anomaly; the 7.2x was that run's artifact
    # The ratio is taken WITHIN a run on purpose: arm position alone is worth +12.5% (eng-mh-0061),
    # so a cross-arm level comparison would be measuring position, not shape.
    "en-l0-mtpoff":      {"CGC_SERVER_MTP": "0", "CGC_DECODE_PROFILE": "1",
                          "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1"},
    "en-l0-mtpon":       {"CGC_DECODE_PROFILE": "1",
                          "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 node-level GPU time] CGC_GPU_NODES=1 adds the per-NODE-KIND GPU table
    # (CGC-GPUNODE:) built from each command buffer's own GPUStartTime/GPUEndTime plus the contiguous
    # node range that buffer encoded (ggml_metal_graph_compute's n_nodes_0 / n_nodes_per_cb split).
    # No sampling and no MTLCounterSampleBuffer. SELF-CHECK: the line's `seg_busy` must equal its own
    # `layer gpu_sum` -- both sum the same per-segment Metal busy time, so a non-zero `delta` means
    # the ranges or the buffer/node mapping are wrong. CGC_GPU_TIMING=1 is REQUIRED for that
    # denominator to exist: dp_lay_gpu[] is only filled from the same accessor, so without it
    # `layer gpu_sum` reads 0.00 and `delta` is vacuously 0 (measured, first run). CGC_DECODE_PROFILE
    # supplies the step cadence; ALL is deliberately NOT set, because the per-layer detail is already
    # known and 40 lines per step would bury the kind table.
    "en-nodes":          {"CGC_GPU_NODES": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 node-level GPU time TRACE] The vocabulary question. The first table had
    # `(other)` 16.1% / `node` 13.6% / `cache` 12.8% and NO ffn_moe_gate/up/down row, and two very
    # different causes fit that: (a) those nodes are simply named something else after the pool
    # repoint, or (b) they ARE there but share a command buffer with several small nodes, so the
    # node-count split hands them only 1/N of their own buffer's duration. This arm prints the RAW
    # node names and the range size of the three hottest command buffers per sampled step, plus a
    # range-size histogram, which separates the two in one line. It also prints every kind instead of
    # the top 14 -- the first revision hid 8 of 22 kinds, and those 8 carried 26.9% of the step, so
    # "present but ranked below the cut" and "absent" looked identical.
    "en-nodes-trace":    {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_TRACE": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 per-NODE command buffers] The end of the dilution story -- and its ceiling.
    #
    # TRACE showed the ranges reach 64 nodes; MATRIX showed the resulting least-squares recovery is
    # not identifiable (rank 15/22, negative coefficients) because a segment's buffers overlap and
    # their durations are therefore not additive. The remaining way to make the kind table a
    # MEASUREMENT is to stop guessing inside a range: shrink the ranges.
    #
    # TWO knobs, and they are NOT equally available:
    #   * CGC_CB_N_MAIN lowers the encoder's `n_main = MAX(64, 0.1*n_nodes)` floor, i.e. how many of
    #     a segment's FIRST nodes the main thread's single buffer eats. FREE: it changes no count.
    #   * CGC_SERVER_N_CB (= SERVER_N_CB, default 8) widens the worker slices, but every extra slice
    #     is one more MTLCommandBuffer per segment. NOT free, and NOT unbounded -- see below.
    #
    # MEASURED CEILING (2026-09-18 01:48, n_cb=127 + n_main=1): the server never becomes ready. It
    # is not slow, it is BLOCKED: `sample <pid>` puts the main thread 100% in
    #   ggml_metal_graph_compute -> commandBufferWithUnretainedReferences -> _MTLCommandBuffer
    #   initWithQueue: -> _dispatch_semaphore_wait_slow -> semaphore_wait_trap
    # i.e. Metal throttles command-buffer CREATION against the number in flight, and with 128 buffers
    # per segment the creation loop blocks BEFORE dispatch_apply, so no worker ever encodes. This is
    # the real reason per-node timestamps are not free: the granularity is capped by Metal, not by
    # the timestamp API. (Retracts the "no MTLCounterSampleBuffer needed" reading of the previous
    # round: it holds for the per-COMMAND-BUFFER level, not for the per-NODE level.)
    #   => the wide-n_cb arms below are DELIBERATELY GONE. Do not re-add n_cb >= 64: it deadlocks at
    #      model load, before any token is generated.
    "en-fine-ctl":       {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_TRACE": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # The one free knob. n_main=1 leaves the main-thread buffer holding the segment's FIRST node
    # (the MoE topk) instead of its first 64, so the widest slice drops from 64 nodes to
    # ceil((N-1)/8) ~ 13 at the production n_cb=8 -- a 5x improvement with the SAME 9 buffers per
    # segment, hence no Metal-throttle risk at all. The MoE gate/up/down land in the first worker
    # slice instead of inside the 64-node bucket.
    "en-fine-nmain":     {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_TRACE": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1"},
    # Both knobs, cb count doubled only (17 buffers per segment instead of 9 -> ~34 in flight, well
    # under whatever the throttle is). Widest slice ~7 nodes. This is the finest granularity that
    # can be reached without risking the deadlock above; it is a probe of whether n_cb can be raised
    # at all, so it runs LAST in any sweep.
    "en-fine-nmain16":   {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_TRACE": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "16"},
    # [CGC 2026-09-18 where is the mass] The kind table buckets nodes by a FIXED prefix vocabulary
    # (ggml-backend.cpp:2100-2109) and drops whatever matches no entry into "(other)" -- measured to
    # be the single largest bucket of a decode step (cntw 20.2% / ub 62.8% of segment busy). This arm
    # just turns on CGC_GRPH_DBG, which prints every node of the first 6 graph_computes with its name
    # and op. The point is NAME RESOLUTION: which real names are falling through to "(other)", and
    # therefore whether the un-attributed fifth of the GPU time is a subsystem or an artefact of the
    # vocabulary being too short. No instrumentation overhead beyond stderr at load.
    "en-grph":           {"CGC_GRPH_DBG": "1"},
    # [CGC 2026-09-18 does a VIEW cost GPU time] The decisive arm for "should we merge the shape
    # chain": the op-keyed table plus the fine command-buffer granularity, so that a buffer holding a
    # single op is common and `uni` (us per node for that op) is an exact division. Compare VIEW /
    # RESHAPE / PERMUTE against MUL_MAT_ID. If the shape ops are ~0 us/node then the 40% of a layer
    # that is views costs nothing and the census's "66% is data movement" is a NODE-CENSUS fact with
    # no GPU consequence; if they are not, the shape chain is the largest addressable block.
    # CGC_CB_N_MAIN=1 is free (no extra buffers) and CGC_SERVER_N_CB=16 costs 17 buffers/segment.
    "en-ops":            {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "16"},
    "en-ops-coarse":     {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18] The same table with the slices pushed to TWO nodes. Reason: at n_cb=16 the
    # slices are ~7 nodes and almost nothing is single-op -- measured, only 2 of 27 ops ever landed in
    # a buffer whose nodes were ALL the same op, and one of those two samples (VIEW) implied 3.5x the
    # whole step, i.e. unusable. A layer's node order interleaves ops, so the only way to get
    # single-op samples is to shrink the slice until it is smaller than the shortest same-op run --
    # and the runs that exist are real: the MoE gate/up pair is two consecutive MUL_MAT_ID, and the
    # MoE output combine is EIGHT consecutive VIEWs of ffn_moe_weighted. At n_cb=63 the slice is
    # ceil(97/63) = 2 nodes, which puts both of those inside a single buffer.
    # RISK: Metal throttles command-buffer creation; 129 buffers/segment deadlocked at load
    # (ggml-metal-context.m's GGML_METAL_MAX_COMMAND_BUFFERS comment). 64/segment is the next rung
    # down, so this arm is the probe of where that ceiling actually is.
    "en-ops-2node":      {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "63"},
    # [CGC 2026-09-18 work-weighted NAME table] The answer to "which named subsystem actually
    # carries the decode step", for the three families the M3 question is about: ffn_moe_* vs
    # cache (the CGC pool's own staging) vs attn_*. Same instrument as `en-nodes`, except the name
    # table now splits each command buffer over the named nodes that ENCODE something instead of
    # over all named nodes, and prints `*bywork` rows ordered by that column plus a
    # work-attributed/residual headline.
    #
    # Deliberately at the DEFAULT command-buffer granularity (no CGC_CB_N_MAIN / CGC_SERVER_N_CB):
    # that is the arm whose `cntw` column was unusable, and the claim to test is precisely that the
    # work-weighted column is usable THERE -- it divides the main-thread buffer's >=64-node range by
    # ~1/4 instead of by 64, without needing finer slices at all.
    "en-work":           {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 NAMING] Same table PLUS the graph dump, so one run both proves the new names
    # are in the graph (CGC-GRPH) and shows the re-split table (CGC-GPUNODE/CGC-GPUOPS). The three
    # names this arm exists to verify: ffn_moe_add (the n_expert_used-1 intermediate terms of the
    # MoE expert-output reduce, llama-graph.cpp:2790/2799), gdn_state / gdn_out_raw
    # (ggml_gated_delta_net, models/delta-net-base.cpp:402/572). Before them, 240 + 30 nodes were
    # bucketed into `node` -- the largest single unnamed cluster in the table.
    "en-named":          {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_GRPH_DBG": "1"},
    # [CGC 2026-09-18 down-combine reachability] The fused path promised in llama-graph.cpp:2675
    # ("One kernel replaces 8 down GEMVs + 7 adds") is an AND of SIX conditions, and the fused
    # Metal kernel is Q3_K-only (ggml-metal.metal:11886, ggml-metal-device.cpp:1309 -- the pipeline
    # name is a hard-coded snprintf, and the IQ types would additionally need a lookup table,
    # smem != 0). This arm answers "how many of the 40 layers actually take it?" by reading the
    # gate itself: CGC_DOWN_COMBINE_AUDIT=1 prints one CGC-DCAUDIT line per MoE layer per graph,
    # and the op table's ADD / MUL_MAT_ID node counts are the cross-check (a fused layer loses its
    # 6 `ffn_moe_add` nodes and its 8th MUL_MAT_ID).
    "en-dc-on":          {"CGC_DOWN_COMBINE": "1", "CGC_DOWN_COMBINE_AUDIT": "1",
                          "CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # The control: byte-identical except the one flag, so the audit lines and the node counts are
    # the only moving parts. Interleave with en-dc-on; do not run them in separate windows.
    "en-dc-ctl":         {"CGC_DOWN_COMBINE": "0", "CGC_DOWN_COMBINE_AUDIT": "1",
                          "CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18] The IQ3_S / IQ4_XS variants (40/40 coverage on the flagship) + the
    # multi-token shape. Two things changed under these arms:
    #   * the fused kernel now has three type variants -- Q3_K was the only one, while the model's
    #     ffn_down_exps is IQ3_S x37 + IQ4_XS x3 -- dispatched by ggml_type_name in the pipeline
    #     getter (the smem differs per type and is not cosmetic: 0 / 512*4 / 32*4);
    #   * CGC_DC_MULTITOK admits the VERIFY shape (n_tokens 2/4), which is what every trunk graph
    #     is under MTP. Without it the fused path is structurally unreachable in production even
    #     with the type gate open. It is read in BOTH llama-graph.cpp and ggml-metal-ops.cpp.
    # So the expected signature of `en-dc-mt` is CGC-DCFUSED on il 0..39 (the IQ layers), whereas
    # `en-dc-ctl` (single-token, Q3_K-only) could only ever fire on il=40. Compare the two.
    "en-dc-mt":          {"CGC_DOWN_COMBINE": "1", "CGC_DC_MULTITOK": "1",
                          "CGC_DOWN_COMBINE_AUDIT": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    "en-dc-mt-ctl":      {"CGC_DOWN_COMBINE": "0", "CGC_DC_MULTITOK": "1",
                          "CGC_DOWN_COMBINE_AUDIT": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 down-combine REAL benefit] The shipped model is the WRONG SUBJECT for this
    # A/B. Its audit says trunk = iq3_s/iq4_xs with ZERO of 1120 trunk rows being Q3_K, so the
    # fused path can only ever serve the MTP draft head (il=40) -- one layer out of 41. Any t/s
    # difference between en-dc-on and en-dc-ctl on that model is therefore noise BY CONSTRUCTION,
    # and reporting "no effect" from it would be a statement about the model, not about the fusion.
    #
    # The second model already on disk has ffn_down_exps = Q3_K x30 (29 of them in the 40 trunk
    # layers) and can therefore actually exercise the fused path:
    #     models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf
    #     down_exps types: {'Q3_K': 30, 'Q4_K': 10, 'Q8_0': 1}  (41 MoE layers)
    #
    # MTP is forced OFF in the Ornith arms so the trunk graph is a plain decode (n_tokens == 1,
    # llama-graph.cpp:2683). That is not a convenience: under MTP the trunk is a VERIFY graph
    # (n_tokens 2 or 4) and the fused path is structurally unreachable whatever the type is.
    # Read the two pairs as different questions:
    #     en-dc-on / en-dc-ctl          (Nail)   -> coverage 1/41; expected to be a null result
    #     en-dc-orn-on / en-dc-orn-ctl  (Ornith) -> coverage 29/40; this is where benefit shows
    # ⚠️ Ornith is 16.4 GB vs Nail's 12.7 GB. Check free memory first -- this machine has been
    #    within 100 MiB of the load failing.
    "en-dc-orn-on":      {"CGC_SERVER_MODEL": "models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf",
                          "CGC_SERVER_MTP": "0", "CGC_DOWN_COMBINE": "1",
                          "CGC_DOWN_COMBINE_AUDIT": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    "en-dc-orn-ctl":     {"CGC_SERVER_MODEL": "models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf",
                          "CGC_SERVER_MTP": "0", "CGC_DOWN_COMBINE": "0",
                          "CGC_DOWN_COMBINE_AUDIT": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18 aligned A/B -- the arm that was missing] "Is Ornith really 1.6x faster than
    # Nail, or was the comparison invalid?" The two en-dc-orn-* arms above cannot answer it: they
    # also carry CGC_DOWN_COMBINE(_AUDIT), and they were measured on a different build, in a
    # different thermal state, and -- the one that matters -- with `p25-gputime`'s own flag set
    # missing from the Nail side of the pair. This arm is the EXACT definition of `p25-gputime`
    # (Nail: MTP off + gpu timing + decode profile) plus one single thing: the model file.
    # Differ-by-one is the whole point; pair it as
    #     p25-gputime (Nail, MTP off)  vs  orn-p25 (Ornith, MTP off)
    # interleaved ABBA (run forward, then reversed), never forward-only.
    # ⚠️ Ornith is 16.36 GiB and this box is 16 GB -- Nail is 12.72 GiB. The load has come within
    #    100 MiB of failing before; check the gate first.
    "orn-p25":           {"CGC_SERVER_MTP": "0", "CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_SERVER_MODEL": "models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf"},
    # [CGC 2026-09-18 Ornith + MTP] "Can Ornith carry MTP at all, and what does it buy?"
    # The head is structurally complete and already identified -- 21 tensors incl.
    # blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm} plus the shared output.weight,
    # `cgc.mtp_head_identity/1` identity 2e35283d, `degenerate: {}` = nothing flagged. So this is
    # ONE FLAG away: the exact definition of `orn-p25` minus `CGC_SERVER_MTP=0`. The earlier
    # en-dc-orn-* arms switched MTP off for a different reason (a plain trunk graph, n_tokens == 1,
    # is the only shape the down-combine gate accepts) -- not because Ornith cannot do MTP.
    # ⚠️ THE GATE IS MEMORY, NOT CORRECTNESS. Ornith is 16.36 GiB on a 16 GB box, the MTP head is
    #    1.34 GiB, and layer 40's 256 pool slots are another 0.80 GiB (0.80 of a 2 GiB budget in
    #    the M6 census). Expect load-time or request-time OOM before anything else.
    #    Escape hatch (existing, no source change): CGC_SERVER_LAYER_CAPS shrinks `40-40:256`,
    #    e.g. `...;40-40:64`. That is a SECOND variable -- report it as its own arm, never folded
    #    into this one.
    # ⚠️ Also: MTP-on makes the trunk a VERIFY graph (n_tokens 2/4), so the down-combine fusion is
    #    structurally unreachable in this arm. Do not read its t/s as "fusion works on Ornith".
    # ⛔ NEVER RUN (decision 2026-09-18, user): the Ornith line is CLOSED. The aligned ABBA
    #    (`p25-gputime` vs `orn-p25`, same build, both MTP off, n_predict 96) put Nail at 12.24 and
    #    Ornith at 8.21 -- Nail is 1.49x FASTER -- and Ornith also reads 32% MORE bytes per token.
    #    The earlier "Ornith 15.72-19.65" did not reproduce (7.94-8.47 on this build); the BUILD,
    #    not the thermal state, is the prime suspect. See memory 2026-09-18 EN-125/EN-126.
    #    Kept rather than deleted, as the record of a question that was asked and answered.
    "orn-p25-mtp":       {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_SERVER_MODEL": "models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf"},
    # The granularity cross-check: same table at 2-7 node slices, where the count-weighted column
    # was already known to CHURN (MUL_MAT 27.3% -> 11.1%, GET_ROWS 4.3% -> 9.0%). Any kind whose
    # work-weighted share is stable between `en-work` and `en-work-fine` is quotable; one that moves
    # is still granularity-dependent and must not be ranked. Read the two side by side, never alone.
    "en-work-fine":      {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "16"},
    # [CGC 2026-09-18 §EN-149] differ-by-one from en-work-fine: the expert pool shrunk from 8 GiB
    # to 6 GiB. Deliberately single-variable, so it decides whether `dnqkv_proj`'s 5.5% is a
    # GPU-side latency cost or the machine's memory pressure (wqkv is a DENSE weight, so unlike the
    # experts it never goes through the pool; at 8 GiB the launch already reports
    # "[budget] OVERSUBSCRIBED by 4838 MiB", and 0.49 ms per 7.3 MB works out to ~15 GB/s, which is
    # swap/compressed-memory order, not DRAM order).
    "en-work-fine-pool6": {"CGC_GPU_NODES": "1", "CGC_GPU_OPS": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "16",
                          "CGC_SERVER_EXPERT_CACHE_BYTES": "6442450944"},
    # The same granularity, but dumping one CGC-NSM line per command buffer (range + duration +
    # per-kind node counts) instead of the aggregate table. Why it is needed even now: the printed
    # table's `ub` is PER KIND, so a family of co-located kinds (ffn_moe_gate/up/down all live in
    # the same ranges) has its family bound either as max(ub) or sum(ub) depending on whether the
    # ranges coincide -- and that gap is [31.6%, 44.2%] of seg_busy, i.e. the M3 threshold falls
    # INSIDE it. The per-buffer dump removes the ambiguity: the family bound becomes the SUM over
    # buffers containing at least one family member (a valid upper bound), and the lower bound the
    # sum over buffers whose nodes are ALL family members.
    # Measured granularity of this configuration (en-fine-nmain16): range histogram 1=58 2=80
    # 3-7=2372 8+=0 -- no buffer exceeds 7 nodes and the 64-node bucket is gone.
    "en-fine-matrix":    {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_MATRIX": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_GPU_TIMING": "1", "CGC_CB_N_MAIN": "1", "CGC_SERVER_N_CB": "16"},
    # [CGC 2026-09-18 node-level GPU time MATRIX] The fix for the dilution the TRACE arm exposed.
    # n_main = MAX(64, 0.1*n_nodes) (ggml-metal-context.m:1099) means every segment's main-thread
    # command buffer holds >=64 nodes, so a count-weighted split divides anything living in it by
    # >=64 (measured: ffn_moe_gate cntw=1.91 ms vs ub=122.37 ms of 271.94 ms). This arm dumps one
    # CGC-NSM line per buffer -- dur plus the per-kind node counts -- so `dur_i = sum_k cnt_ik * t_k`
    # can be solved offline (least squares) for the per-kind per-node cost t_k, which is the number
    # M3's "ffn_moe_* >= 40%?" rule actually needs.
    "en-nodes-matrix":   {"CGC_GPU_NODES": "1", "CGC_GPU_NODES_MATRIX": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-18] The other way to answer M3's question, and the cheap one: the per-OP-TYPE
    # breakdown that was already in the tree. `CGC_VERIFY_OP_TIMING` fires only for T>1 (its target
    # predicate is `node->src[2]->ne[1] > 1`), i.e. the MTP verify batch -- the production shape --
    # so it answers "what share do gate/up/down carry" for the path that actually runs, without the
    # node-range dilution that made CGC_GPU_NODES unusable for a threshold test. It is NOT the T=1
    # shape M3's leave condition is written against; report it as the verify-path number it is.
    # It also had to be added to run_server.sh's allowlist first -- it was silently dropped before.
    "en-verify-op":      {"CGC_VERIFY_OP_TIMING": "1", "CGC_DECODE_PROFILE": "1"},
    # [2026-09-15] Fusion re-test, now that run_server.sh can actually pass CGC_MMV_FUSE through
    # (it could not before -- the var was missing from the launcher allowlist, so every earlier
    # fusion A/B through this launcher measured a fusion that was never enabled). The fused
    # dispatch logs one GGML_LOG_WARN per run, so the arm is also self-verifying: no WARN = the
    # fusion did not engage and the arm's number is meaningless.
    # CGC_GLU_FUSED_DOWN rides along because §8.113's "+6.5%" was recorded for the pair.
    "p25-fuse":          {"CGC_SERVER_MTP": "0", "CGC_MMV_FUSE": "1",
                          "CGC_SERVER_GLU_FUSED_DOWN": "1"},
    "p25-fuse-phase":    {"CGC_SERVER_MTP": "0", "CGC_MMV_FUSE": "1",
                          "CGC_SERVER_GLU_FUSED_DOWN": "1",
                          "CGC_PHASE_TIMING": "1"},
    # ---- 2026-09-15: split the 72 ms `wait` into GPU execution vs launch/completion latency ----
    # CGC-DECPROF says 91% of a decode step is `wait`, but `wait` is a CPU-side spin on cgc_done,
    # so it cannot say whether the GPU was actually busy for 72 ms or finished early and left the
    # window to launch + completion-report latency. The two readings imply OPPOSITE work:
    #   GPU-busy high  -> n_tokens=1 GEMV is occupancy/latency bound; batch a layer's 8 expert
    #                     GEMVs into one dispatch (roadmap M3 item 3) and raise per-dispatch
    #                     parallelism. Touches ggml-metal kernels.
    #   GPU-busy low   -> the GPU is starving; restore inter-segment overlap instead of draining
    #                     the queue at every segment boundary (the remap-leaf dependency that
    #                     forces the drain, see the CGC_SUBMIT_AHEAD comment in ggml-backend.cpp).
    #                     Touches the dispatcher, not a single kernel.
    # CGC_GPU_TIMING makes the Metal completion handlers record each buffer's own
    # GPUStartTime/GPUEndTime; the dispatcher prints `CGC-GPUTIME:` per 8 steps with
    # wait / gpu_busy_sum / gpu_union / gap. Decision rule: busy/wait >= 70% -> execution;
    # <= 40% -> latency. Either way `unsupported=` in the line must be 0, or the platform is
    # not reporting the timestamps and the line means nothing.
    # CGC_DECODE_PROFILE rides along so `wait` comes from the instrument that produced the
    # 72.56 ms figure, in the same run.
    "p25-gputime":       {"CGC_SERVER_MTP": "0", "CGC_GPU_TIMING": "1",
                          "CGC_DECODE_PROFILE": "1"},
    # [CGC 2026-09-17 §EN-14] The anchor with the churn/digest diagnostics ON. Two arms can only be
    # diffed on a stderr stream that BOTH of them print, and CGC_S1_TABLE_CHURN was only carried by
    # p25-slotgpu-churn -- so the anchor arm was silent and every SLOT-OWNER comparison would have
    # had one empty side. Keeping it a separate arm rather than adding the flag to p25-gputime means
    # every earlier measurement taken with the plain anchor stays reproducible as it was.
    "p25-gputime-churn": {"CGC_SERVER_MTP": "0", "CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1",
                          "CGC_S1_TABLE_CHURN": "1"},
    # MTP-on counterpart (same reason as p25-phase-mtp): a single arm cannot tell a clean 0
    # busy-time from a broken instrument.
    "p25-gputime-mtp":   {"CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1"},
    # [2026-09-15] The CEILING probe for the whole "restore inter-segment overlap" family.
    # CGC_SUBMIT_AHEAD=1 submits segment i+1 BEFORE the top-k hook of segment i writes its remap
    # leaf, which is exactly the raciness the current order was introduced to fix. It is therefore
    # WRONG and its output is expected to be corrupt (md5 must change -- if it does not, the flag
    # never reached the process and the number is meaningless). It is worth a run anyway because
    # it deletes the entire CPU window between the poll and the commit in one switch: no
    # double-buffered remap design can beat this, so if decode does not speed up here, that whole
    # family is dead and the remaining lever is the GPU work itself.
    "p25-submit-ahead":  {"CGC_SERVER_MTP": "0", "CGC_SUBMIT_AHEAD": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 S1 slot-table] The first CORRECT member of the "restore inter-segment
    # overlap" family. CGC_SLOT_TABLE_GPU=1 makes the expert->slot mapping a graph node
    # (get_rows over a per-layer table) instead of a host-written input leaf. Stage S1 deliberately
    # does NOT touch the dispatcher, so n_segs stays at 40 and the speedup here should be ~0 --
    # what this arm has to prove is that the output is bit-identical to `p25-gputime` (same md5,
    # same logits). That gate is the precondition for S2 (drop the segment drain) and S3 (collapse
    # the segments), which is where the x1.78 that p25-submit-ahead measured becomes reachable
    # without corrupting the answer.
    "p25-slotgpu":       {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # Same arm with the S1 diagnostics on (CGC-S1: graph / CAPTURE / hook / EXPECT). Use it when
    # p25-slotgpu diverges and you need to know WHERE: the hook traces say whether the table was
    # captured at all and whether the hook found it, and EXPECT prints the slot vector the host
    # just published for this step's ids -- the number to put next to CGC-MMID-ASSERT's `first=`.
    # Never use this arm for a throughput number: it writes to stderr from inside the hot path.
    "p25-slotgpu-dbg":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_DBG": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 S1 cleanup] p25-slotgpu-ident (CGC_S1_IDENT: constant-0 table) and
    # p25-slotgpu-tag (CGC_S1_TAG: 1000+layer table) are gone, together with the knobs themselves.
    # Both were deliberately-WRONG table publishes whose only job was to infer, from whether an
    # out-of-range assertion fired, whether the host write reaches the buffer the GPU reads. The
    # kernel-side ids capture (CGC_IDS_CAPTURE, ggml-metal-ops.cpp) answers the same question
    # directly -- it reads the value mul_mat_id actually consumed -- so both the inference and the
    # two standing invitations to leave an invalid mapping switched on were removed.
    #
    # [CGC 2026-09-15 §9.18 next step] The output-capture pair. Both arms carry CGC_S1_OUT_CAP
    # identically, because it PINS ffn_moe_down via ggml_set_output and pinning perturbs the
    # allocator layout: a one-sided pin would make the two arms differ by the pin rather than by the
    # mechanism under test. Never a gate arm -- see cache_down_out_tensors in llama-context.h.
    "p25-outcap-base":    {"CGC_SERVER_MTP": "0", "CGC_S1_OUT_CAP": "1",
                           "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-outcap": {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_OUT_CAP": "1",
                           "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 §9.18 ladder] The prefilling counterpart of the pair above. The decode capture
    # answered "the arms differ at layer 0 of decode step 0" -- which localizes nothing, because
    # decode step 0's layer-0 input comes from the KV the PREFILL wrote and the two prefills already
    # disagree (M1 numeric gate 5/9). In prefill, layer 0's input is the embedding: identical across
    # arms by construction. So the first layer at which these two arms differ names the carrier
    # instead of inheriting it. CGC_S1_OUT_LAYERS=0-3 keeps the pin to four layers: a prefill
    # ffn_moe_down is [2048, 8, ntok], which is 16 MB per layer at a 250-token prompt but 402 MB per
    # layer at a 6144-token chunk, and an all-layer pin in every graph already blew the Metal
    # envelope once (kIOGPUCommandBufferCallbackErrorOutOfMemory at warmup, both arms).
    "p25-outcap-pre-base":   {"CGC_SERVER_MTP": "0", "CGC_S1_OUT_CAP": "pre",
                              "CGC_S1_OUT_LAYERS": "0-3",
                              "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-outcap-pre": {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1",
                               "CGC_S1_OUT_CAP": "pre", "CGC_S1_OUT_LAYERS": "0-3",
                               "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 §9.18 ladder, layer depth] The 0-3 window answered something clean but partial:
    # on the prefill-stream graphs (182 and 22 tokens, all 256 experts resident) layers 0-3 agree,
    # and on the pool-path graphs (ntok=2 warmup, ntok=4) layer 0 agrees while layers 1-3 do not. The
    # 0-3 window cannot tell those apart from "the first pooled layer is 4" or "nothing after 3
    # diverges", because it stops at 3. These arms are the NEXT rung, 4-7.
    #
    # Why the rungs are FOUR layers and not the whole depth in one arm (CONVENTIONS A8): pinning 11
    # layers on prefill graphs dies at the load-time warmup decode with the same Metal OOM as pinning
    # 40 on every graph, while 4 layers runs -- and at ntok=2 eleven pinned tensors hold only 1.4 MB,
    # so the limit is pins-per-graph and not bytes. A sparse all-depth window is therefore NOT a
    # cheaper alternative to a dense narrow one; it is the variant that aborts.
    "p25-outcap-depth-base": {"CGC_SERVER_MTP": "0", "CGC_S1_OUT_CAP": "pre",
                              "CGC_S1_OUT_LAYERS": "4-7",
                              "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-outcap-depth": {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1",
                                 "CGC_S1_OUT_CAP": "pre",
                                 "CGC_S1_OUT_LAYERS": "4-7",
                                 "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 premise B] Does the published table's CONTENT move between decode steps?
    # Prints a per-graph total on stderr plus a teardown line, and decides whether publication is
    # redundant work (premise B holds -- the segment boundary has lost its reason to exist) or
    # inherently per-step (premise B fails, and D3's n_segs collapse does not follow from it).
    # Only decode steps (n_tokens == 1) are counted; prefill rewrites the table for reasons nobody
    # is questioning and would drown the answer.
    "p25-slotgpu-churn":  {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_TABLE_CHURN": "1",
                           "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-15 S1 localization] CGC_S1_MIN_IL gates which layers get the GPU table, so the
    # arm bisects the divergence instead of arguing about it.
    #
    # RESOLVED by rung 1 (Backup/phase_decomp/s1_bisect_20260915.json, --rounds 2 --n-predict 24,
    # build server 131bc5316ebf / metal 9729ad35ddd1 / llama bcf32c3f0917):
    #
    #   p25-gputime     md5 {dc055e63}          stable   <- anchor
    #   p25-slotgpu-l39 md5 {dc055e63}          stable
    #   p25-slotgpu-l38 md5 {dc055e63}          stable
    #   p25-slotgpu-l20 md5 {dc055e63}          stable
    #   p25-slotgpu     md5 {29ca694a,b8c705cc} UNSTABLE (2 rounds -> 2 values)
    #
    # The two possibilities this arm was built to separate:
    #   one-layer arm already unstable -> the divergence is not in the mapping at all; the mere
    #                                     PRESENCE of the extra nodes (CONT + GET_ROWS + VIEW per
    #                                     layer) perturbs the graph/segment layout globally;
    #   one-layer arm bit-identical     -> the mapping is correct and the divergence appears at a
    #                                     specific layer, which the arm ladder then localizes.
    #
    # => the SECOND branch. l20 already adds 20 layers' worth of CONT+GET_ROWS+VIEW+table nodes
    #    (~80 nodes) and is still bit-identical AND stable, so "extra nodes perturb the layout
    #    globally" is refuted; the divergence is confined to the layers l20 does not touch, i.e.
    #    1..19. Note also the SHAPE of the failure: the full arm is not stably wrong but UNSTABLE,
    #    which rules out a plain deterministic mapping error and points at ordering/state.
    #    Rung 2 (below) samples 1..19.
    #
    # Run these with a baseline in the same sweep (`--arms p25-gputime,p25-slotgpu-...`) so all
    # rows share one build fingerprint -- and one n_predict, which the digest is scoped to.
    "p25-slotgpu-l39":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "39",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l38":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "38",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l20":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "20",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # ---- Rung 2: rung 1 excluded 20..39, so the culprit is in 1..19 --------------------------
    # The arms are NESTED (min_il=m covers layers m..39), so the LARGEST m that is still
    # bit-identical is exactly one less than the culprit layer:
    #   l2  identical -> the culprit is layer 1 (the only layer l2 still excludes)
    #   l6  identical -> culprit in 2..5       l10 identical -> culprit in 6..9
    #   l14 identical -> culprit in 10..13     l18 identical -> culprit in 14..17
    #   none identical -> culprit in 18..19
    # Layer 0 is never a candidate: CGC_S1_MIN_IL's default of 1 exists precisely to keep layer 0
    # on the host leaf (its MoE FFN runs on CPU/BLAS, so a GPU-computed id vector would need a
    # cross-backend copy -- the shape of the 20:18 SIGSEGV).
    "p25-slotgpu-l2":    {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "2",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l6":    {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "6",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l10":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "10",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l14":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "14",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    "p25-slotgpu-l18":   {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_MIN_IL": "18",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # ---- Rung 2 RESULT (Backup/phase_decomp/s1_bisect2_20260915.json, --rounds 3 --n-predict 24) --
    #
    #   p25-gputime     md5 {dc055e63}          stable   <- anchor
    #   p25-slotgpu-l20 md5 {dc055e63}          stable   (rung 1)   serves 20..39, 20 layers
    #   p25-slotgpu-l18 md5 {4991e6d2}          stable   serves 18..39, 22 layers
    #   p25-slotgpu-l14 md5 {65bd254b}          stable   serves 14..39, 26 layers
    #   p25-slotgpu-l10 md5 {1c5a05df}          stable   serves 10..39, 30 layers
    #   p25-slotgpu-l6  md5 {7042e5e4}          stable   serves  6..39, 34 layers
    #   p25-slotgpu-l2  md5 {0d472bf5,51548f50} UNSTABLE serves  2..39, 38 layers
    #   p25-slotgpu     md5 {29ca694a,b8c705cc} UNSTABLE serves  1..39, 39 layers
    #
    # Read naively this says "the bad layer is 18 or 19". That reading is NOT licensed by this
    # experiment, and the reason is structural: CGC_S1_MIN_IL is a PREFIX gate, so every arm serves a
    # contiguous SUFFIX of the stack. The digest is monotone in the LEFTMOST served layer, and the
    # count of served layers moves together with it -- 20 served is identical, 22 served is not. So
    # the same table also fits "any arm serving >= 21 layers diverges", which is a threshold/resource
    # effect and needs a COMPLETELY different fix from a wrong mapping at one layer. A nested-arm
    # ladder cannot separate those two, no matter how many rungs are added: it has only one degree of
    # freedom. Do not add a rung 3 of the same shape.
    #
    # What DOES separate them is comparing the two id vectors directly at every layer, which is what
    # `p25-keepleaf` exists for. In that arm mul_mat_id consumes the host leaf while every S1 node is
    # still built, so the post-sync readback can walk all captured layers and print gather-vs-leaf
    # (same=1/0). Mapping error -> some layer prints same=0 and the arm's digest is bit-identical to
    # the baseline (the extra nodes are then provably inert). Threshold effect -> every layer prints
    # same=1 and the arm's digest still differs, which would mean the ids were never the problem.
    #
    # Run it with the baseline in the SAME sweep: the whole comparison is a same-fingerprint claim.
    "p25-keepleaf":      {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_KEEP_LEAF": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # [CGC 2026-09-17 §EN-14] keepleaf with the churn/digest diagnostics on. This is the arm that
    # separates "the S1 nodes existing at all" from "the GPU table being CONSUMED": it builds every
    # S1 node but leaves mul_mat_id eating the host leaf, and it is bit-identical to the anchor. So
    # if the pool's reverse map matches the anchor here and differs in p25-slotgpu-churn, the pool
    # difference tracks the consumption, not the graph shape.
    "p25-keepleaf-churn": {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_KEEP_LEAF": "1",
                           "CGC_S1_TABLE_CHURN": "1",
                           "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # Same, with the diagnostics on. POST (gather vs leaf, every layer), EXPECT (table[ids] at hook
    # time) and EQUIV (published table vs slot_table_safe) all ride on CGC_S1_DBG. Throughput from
    # this arm is meaningless -- it writes to stderr from inside the hot path -- so it is tagged as a
    # probe by NEG_MARKERS in traces/emit_episodes.py.
    "p25-keepleaf-dbg":  {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_KEEP_LEAF": "1",
                          "CGC_S1_DBG": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # ---- RESULT of the keep-leaf control, and what it leaves open ------------------------------
    #
    # p25-keepleaf is BIT-IDENTICAL to the baseline (dc055e63, stable, 24 tokens, same prefix) with
    # all 39 layers served and every S1 node present, and its pool counters match the anchor exactly
    # (misses 13040, file_reads 36360). So the 156 extra nodes are numerically inert, and no
    # served-layer-count threshold can be involved -- which also removes the ambiguity rung 2 could
    # not resolve, licensing the per-layer reading: min_il=20 identical and min_il=18 different pin
    # the fault to layer 18 or 19.
    #
    # The per-layer readback then killed the ids: p25-keepleaf-dbg prints gather-vs-leaf for every
    # captured layer, and for TOKEN 0 the two agree at EVERY layer, 6 samples each. (Token 1 differs
    # at every layer -- that is the padding row whose layout/n_tokens quirk is recorded in
    # eng-src-0005, and it is provably benign because min_il=20 carries 20 layers of it and is still
    # bit-identical.) Note the aggregate `same=` field reads 0 because it compares the whole vector;
    # the meaningful comparison is per token, which is why it has to be split by hand.
    #
    # So both explanations S1 was built around are dead. What remains between the real arm and the
    # control is the ABSENCE of the leaf node -- and the header comment on cache_slot_table_tensors
    # says that absence was never intended ("the leaf itself is still built when this is on: it is
    # simply not consumed"). p25-buildleaf measures exactly that documented design.
    "p25-buildleaf":     {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "CGC_S1_BUILD_LEAF": "1",
                          "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # ---- RESULT of the build-leaf arm: IT DOES NOT RUN, and that is the finding ---------------
    #
    # p25-buildleaf aborted at server start: `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)` from
    # ggml_gallocr_allocate_node <- ggml_gallocr_reserve_n_impl <- ggml_backend_sched_reserve <-
    # llama_context::graph_reserve. It never produced a digest, so it is NOT evidence about the
    # logits -- it is evidence about the allocator. With BUILD_LEAF set, `slots` is not expanded and
    # not consumed, so the S1 chain (table/cont/get_rows/view) is orphaned; `slot_table` survives
    # only because line ~2190 expands it unconditionally, which makes it a graph ROOT WITH NO
    # CONSUMER -- and the new `remap` leaf is a second one. The scheduler assigns no backend to a
    # root that nothing consumes and ggml-alloc then aborts on buffer_id = -1.
    #
    # So the header comment on cache_slot_table_tensors ("the leaf itself is still built when this
    # is on: it is simply not consumed ... a pure scheduler/dispatch change") describes something
    # the allocator will not accept. It was wrong in two ways, not one: the branch is an if/else so
    # the leaf is NOT built (eng-gate-0003), and building it unconsumed is not viable either. The
    # experiment is retired; do not re-add it without an innocuous consumer for the leaf.
    #
    # ---------------------------------------------------------------------------------------
    # [CGC 2026-09-15 S1 timing] CGC_SERVER_OA_ASYNC=0 pair -- the ONE variable left after the
    # value-level explanations died.
    #
    # By 21:29 the ids were exonerated twice over and on the real arm, not on a control:
    #   * p25-slotgpu-dbg, CGC-S1: EQUIV-* = 2067 lines, mismatch=0 on every one. The published
    #     table equals slot_table_safe() for every selected expert, so the "publish clamps a
    #     non-resident expert to slot 0 while safe returns -1" drift the two functions were written
    #     to have is NOT happening here -- consistent with the CGC-SLOT-TABLE clamp counter staying
    #     at 0 lines in every S1 log of the day.
    #   * CGC-S1: POST (the gather, read after ggml_backend_sched_synchronize) vs CGC-S1: EXPECT-*
    #     (what the hook published for THIS step's ids): all 39 layers x all 6 sampled graph builds
    #     agree, in every row the probe prints. On the real arm. With no leaf anywhere.
    # Together with p25-keepleaf being bit-identical, that leaves no value difference at all between
    # the real arm and the control, which is only possible if the difference is WHEN the table write
    # lands relative to the segment that reads it.
    #
    # That is exactly the hazard llama-graph.cpp describes at the S2 comment: the segmented async
    # dispatcher (CGC_OA_ASYNC, prod25 = on) submits segment i+1 only after the hook of segment i
    # has written the leaf, and the drain exists BECAUSE the leaf is a host-written input. S1 moves
    # the mapping into a graph node; if the drain is keyed to the leaf rather than to the hook, S1
    # loses the ordering guarantee and the gather can read the table before the host wrote it.
    #
    # This pair is the test: OA_ASYNC=0 reverts to the non-segmented dispatcher, so the ordering
    # question disappears. Compare WITHIN the pair -- both arms must carry the same value of this
    # switch or the digests are not comparable (CONVENTIONS.md rule A5 is about length; this is the
    # same class of mistake with a build switch).
    #   slotgpu-noasync bit-identical to gputime-noasync -> the drain really was keyed to the leaf,
    #       and S1's fix is to make the dispatcher drain on the hook (or on the table) instead.
    #   slotgpu-noasync still different -> the ordering is not it either, and the remaining suspect
    #       is the ggml-alloc buffer assignment (which the now-dead BUILD_LEAF arm was meant to
    #       probe); the instrument for that is GGML_SCHED_DEBUG=2 node-level GET_CAUSE, not another
    #       digest.
    #
    # FIRST RUN OF THIS PAIR WAS A NO-OP, and the reason is a gate bug now fixed at
    # ggml/src/ggml-backend.cpp:1745. The segmented dispatch was selected with
    # `getenv("CGC_OA_ASYNC") != nullptr`, so CGC_OA_ASYNC=0 -- and an empty string, and every
    # profile run_server.sh emits -- selected the SEGMENTED branch. The two arms therefore re-ran
    # the segmented arms exactly: md5 sets {dc055e63} and {29ca694a, 672585db, b8c705cc}, pool
    # counters misses 13040 / 14296 and file_reads 36360 / 39948, all identical to the segmented
    # run. The gate is now value-aware (unset/empty/nonzero = segmented, "0" = off), which keeps
    # prod25's recorded digests reproducible because prod25 sets "1".
    #
    # So the ordering hypothesis is UNTESTED, not refuted. Any earlier row whose declared purpose
    # says "non-segmented" is a segmented row; see lesson eng-gate-0005.

    # Baseline for the noasync pair: non-segmented dispatch, host-written leaf.
    "p25-gputime-noasync": {"CGC_SERVER_MTP": "0", "CGC_SERVER_OA_ASYNC": "0",
                            "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # S1 with non-segmented dispatch. Bit-identical to p25-gputime-noasync would mean the
    # segment-boundary ordering was the cause and the fix is to keep the drain tied to the hook.
    "p25-slotgpu-noasync": {"CGC_SERVER_MTP": "0", "CGC_SERVER_OA_ASYNC": "0", "CGC_SLOT_TABLE_GPU": "1",
                            "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # The keep-leaf control with non-segmented dispatch. This is now the decisive pair, because the
    # gate fix changed what the noasync arms mean. Measured with the gate fixed (build fingerprint
    # 2026-09-15 21:37):
    #   p25-gputime-noasync md5 {dc055e63} stable, misses 13040, file_reads 36360
    #       -- identical to the SEGMENTED anchor, so the dispatcher is numerically neutral when the
    #          ids come from the host leaf.
    #   p25-slotgpu-noasync md5 {ff68c5a2} stable, misses 16717, file_reads 47574
    #       -- a value that appears in neither the segmented S1 set {29ca694a, 672585db, b8c705cc}
    #          nor the anchor. So the ordering hypothesis is REFUTED (the arm is still wrong with
    #          the hazard removed) and, separately, the S1 answer DEPENDS on the dispatcher:
    #          segmented = 3 values across 3 rounds, non-segmented = 1 stable value.
    # That splits the failure into a non-deterministic part introduced by the segmented dispatch and
    # a deterministic part underneath it. This arm asks whether the extra nodes are still inert once
    # the dispatch is not there:
    #   keepleaf-noasync bit-identical to gputime-noasync -> the nodes are inert again, so the
    #       residual deterministic difference is the ABSENCE of the leaf node (allocator layout),
    #       and the instrument is GGML_SCHED_DEBUG=2 per-node GET_CAUSE.
    #   keepleaf-noasync different -> the S1 nodes are only inert under the segmented dispatcher,
    #       which would make the segmented path's bit-identity a coincidence of ordering and would
    #       move the suspect back onto the nodes themselves.
    # Throughput from this pair is not a candidate number: genuinely non-segmented costs ~10x
    # (6.95 -> 0.72 t/s for the baseline). Note that fact is itself the evidence the gate fix works,
    # and it is the size of the prize S2/S3 are chasing.
    "p25-keepleaf-noasync": {"CGC_SERVER_MTP": "0", "CGC_SERVER_OA_ASYNC": "0", "CGC_SLOT_TABLE_GPU": "1",
                             "CGC_S1_KEEP_LEAF": "1",
                             "CGC_DECODE_PROFILE": "1", "CGC_GPU_TIMING": "1"},
    # ---------------------------------------------------------------------------------------
    # Backend-assignment pair (2026-09-15 20:18 finding)
    #
    # The S1 run at 20:18 died with SIGSEGV inside ggml_compute_forward_mul_mat_id **on
    # libggml-cpu** (parsed from ~/Library/Logs/DiagnosticReports/llama-server-2026-09-15-201852.ips;
    # stack: ggml_backend_cpu_graph_compute <- ggml_backend_sched_graph_compute_async <-
    # llama_context::graph_compute <- ... <- common_init_from_params). That signature appears in NO
    # other run of the day -- the only other crashes are SIGBUS in fill_pool_direct/prewarm and
    # SIGABRT from ggml_abort. It means at least one mul_mat_id was assigned to the CPU backend
    # while its weight operand is a pool-repointed tensor living in a Metal buffer.
    #
    # This pair runs the SAME profile twice with ggml's own scheduler dump on (GGML_SCHED_DEBUG=1
    # prints "## SPLIT #N: <backend> # M inputs" plus the input tensor names for every split; =2
    # additionally prints EVERY node with its assigned backend and the reason code in GET_CAUSE).
    # Diff the two logs: if S1 introduces a CPU split that the baseline does not have, the whole
    # `id_oob with F32 bit patterns` mystery is answered -- the consumer is not reading the gather
    # output at all, it is a different backend reading a different buffer. Throughput from these two
    # arms is meaningless (the dump is on); use them only for the split table.
    #
    # IMPORTANT (measured 2026-09-15 20:23, first attempt): with GGML_SCHED_DEBUG=1 the dump is
    # emitted via GGML_LOG_DEBUG, which the default llama verbosity threshold (INFO) FILTERS OUT --
    # the log contained zero "## SPLIT" lines, which is indistinguishable from "exactly one split".
    # Both print sites in ggml_backend_sched_print_assignments are now promoted to GGML_LOG_WARN
    # (still gated on sched->debug), so =1 and =2 are both readable at default verbosity.
    #
    # REFUTED by that first run: the split STRUCTURE is identical between base and S1 (18 graph
    # builds each, every build = [CPU, MTL0, CPU(attn_post_norm residual), MTL0], 72 split headers
    # in both). So "S1 changes which backend the moe nodes land on" is NOT the mechanism at graph
    # granularity. What is still open is the node-level cause code, which is why this arm is =2.
    "p25-sched-base":    {"CGC_SERVER_MTP": "0", "GGML_SCHED_DEBUG": "2",
                          "CGC_DECODE_PROFILE": "1"},
    "p25-sched-slotgpu": {"CGC_SERVER_MTP": "0", "CGC_SLOT_TABLE_GPU": "1", "GGML_SCHED_DEBUG": "2",
                          "CGC_DECODE_PROFILE": "1"},
}


def port_listener(port: int | None) -> list[str]:
    """PIDs listening on `port`, via lsof. The handle that is actually ours."""
    if port is None:
        return []
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout
    return [x for x in out.split() if x.strip().isdigit()]


def killed(port: int | None = None):
    """Stop THIS arm's server. Port-scoped when a port is known: `pkill -f <pattern>` is
    pid-blind and on this shared box it takes out parallel sessions' servers too (the defect
    recorded at http_duo.py:31,285). The pattern form is kept only as the no-port fallback."""
    pids = port_listener(port)
    if pids:
        subprocess.run(["kill", "-INT", *pids], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(6)
        left = port_listener(port)
        if left:
            subprocess.run(["kill", "-9", *left], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        return
    subprocess.run(["pkill", "-INT", "-f", SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    subprocess.run(["pkill", "-9", "-f", SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def wait_ready(ctrl_path, timeout=240):
    """run_server.sh prints `[detach] server ready` once the child is health-checked, and its
    own `[log] <path>` line names the file the SERVER writes (which is where `listening on
    http`, `print_timing` and the expert-cache teardown actually land). Polling the shim's
    stdout for `listening on http` never matches -- the shim only relays its own messages."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(ctrl_path):
            try:
                with open(ctrl_path, "r", errors="replace") as f:
                    txt = f.read()
                if "[detach] server ready" in txt:
                    return True
                if "error:" in txt or "FAILED" in txt:
                    return False
            except OSError:
                pass
        time.sleep(2)
    return False


LOG_LINE_RE = re.compile(r"\[log\]\s+([^\s（(]+)")


def server_log_of(ctrl_path):
    """The server's own log path, as announced by run_server.sh."""
    try:
        txt = open(ctrl_path, "r", errors="replace").read()
    except OSError:
        return ""
    m = LOG_LINE_RE.search(txt)
    return m.group(1) if m else ""


def start(env_extra, ctrl_path, profile="prefill250"):
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env.update(env_extra)
    open(ctrl_path, "w").close()
    with open(ctrl_path, "a") as fh:
        subprocess.Popen(["./scripts/run_server.sh", "--detach"],
                         cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return wait_ready(ctrl_path)


# NOTE ON WHITESPACE: the C++ teardown pads fields with TWO spaces
# (`... misses=20080 (hit rate 92.4%)  prewarm req=0 ...`, `... slots=39  worst=layer 2 ...`).
# An earlier revision hard-coded single spaces, so FINAL_RE/ATTR_RE silently matched nothing and
# every sweep row reported `hit None% miss None` while SHAPE_RE (which used \s+) worked. Use \s+
# between every field -- never a literal space -- so the harvest cannot silently degrade again.
FINAL_RE = re.compile(r"final stats: runtime requests=(\d+)\s+hits=(\d+)\s+misses=(\d+)\s+"
                      r"\(hit rate ([\d.]+)%\)\s+prewarm req=(\d+)\s+hit=(\d+)\s+miss=(\d+)\s+"
                      r"resident=([\d.]+) MiB\s+file_reads=(\d+)\s+pread_usec=(\d+)\s+"
                      r"fill_batch_usec=(\d+)")
ATTR_RE = re.compile(r"miss attribution: compulsory=(\d+)\s+capacity=(\d+)\s+"
                     r"\(([\d.]+)% / ([\d.]+)%\s+of\s+(\d+)\)\s+evictions=(\d+)\s+"
                     r"layers_distinct_over_slots=(\d+)\s+worst=layer (\d+)\s+distinct=(\d+)\s+slots=(\d+)")
SHAPE_RE = re.compile(r"read shape: jobs=(\d+)\s+bytes=(\d+).*us/job=(\d+)\s+effective_rate=([\d.]+) MiB/s")
# `slot print_timing: id 0 | task 88 | draft acceptance = 0.99740 (383 accepted / 384 generated),
# mean len = 3.99` -- the LAST one in the log is the steady-state value. This is the number that
# decides whether the MTP speculative path nets a speedup: at mean len ~4 each decode step
# amortises 4 tokens; at mean len ~1 it is pure overhead.
DRAFT_RE = re.compile(r"draft acceptance = ([\d.]+)\s+\(\s*(\d+) accepted\s*/\s*(\d+) generated\),\s*"
                      r"mean len =\s*([\d.]+)")
# Server-measured throughput, straight from slot print_timing (`tg`). Independent of the
# client-side timing decode_bench derives, so the two cross-check each other.
TG_RE = re.compile(r"n_decoded =\s*(\d+),\s*tg =\s*([\d.]+) t/s")


def loopiness(text):
    """Degeneration detector for the sampled answer.

    A collapsed draft/verify path does not crash -- it emits a coherent-looking sentence over
    and over (`p25-nowarmup`: `Here, the user request is genuinely ambiguous, ask a sharp
    question.` repeated to the token limit). md5 stability across rounds CANNOT see this (the
    loop is perfectly deterministic), so every arm must report it: `answer_stable=True` next to
    a loopiness near 1.0 is a broken arm, not a stable one.

    Measured as the fraction of REPEATED WORD 4-GRAMS, not lines and not fixed-size chunks.
    Two earlier revisions failed on real data: comparing stripped lines scored every MTP-on loop
    0.00 (the repeats were separated by `. `, or differed in one word -- `don'guess.` vs
    `don't guess.` -- so each line was unique), and half-overlapping 32-char chunks scored
    `p25-probe-on` 0.00 even though its sample is visibly one sentence twice, because the repeat
    period (~66 chars) did not line up with the 32/16 window. Word 4-grams are period-agnostic
    and ignore punctuation and whitespace, which is what makes them survive the sample being
    truncated to ~200 chars. 0 = no phrase reused, 1 = every 4-gram is a repeat."""
    words = " ".join(text.split()).split()
    k = 4
    if len(words) < 2 * k:
        return 0.0
    grams = [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


def harvest(log_path):
    try:
        txt = open(log_path, "r", errors="replace").read()
    except OSError:
        return {}
    out = {}
    if (m := FINAL_RE.search(txt)):
        req, hit, miss, rate, p_req, p_hit, p_miss, res, reads, pu, fbu = m.groups()
        out.update({
            "requests": int(req), "hits": int(hit), "misses": int(miss),
            "hit_rate_pct": float(rate),
            "prewarm_req": int(p_req), "prewarm_miss": int(p_miss),
            "resident_mib": float(res), "file_reads": int(reads),
            "pread_usec": int(pu), "fill_batch_usec": int(fbu),
        })
        rt = int(miss) + int(p_miss)
        if rt > 0:
            out["per_miss_us"] = round(int(pu) / rt, 1)
    if (m := ATTR_RE.search(txt)):
        comp, cap, cp, cap_p, tot, ev, over, wl, wd, wns = m.groups()
        out.update({
            "miss_compulsory": int(comp), "miss_capacity": int(cap),
            "capacity_pct": float(cap_p),
            "evictions": int(ev), "layers_over_slots": int(over),
            "worst_layer": int(wl), "worst_distinct": int(wd), "worst_slots": int(wns),
        })
    if (m := SHAPE_RE.search(txt)):
        jobs, b, usj, rate = m.groups()
        out.update({"io_jobs": int(jobs), "io_bytes": int(b),
                    "io_us_per_job": int(usj), "io_effective_mib_s": float(rate)})
    if (ms := DRAFT_RE.findall(txt)):
        acc, a, g, ml = ms[-1]
        out.update({"draft_accept": float(acc), "draft_accepted": int(a),
                    "draft_generated": int(g), "draft_mean_len": float(ml)})
    if (ms := TG_RE.findall(txt)):
        n, tg = ms[-1]
        out.update({"n_decoded_server": int(n), "tg_server": float(tg)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--json", default="/tmp/decode_sweep.json")
    ap.add_argument("--profile", default="prefill250")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if args.report:
        rows = json.load(open(args.report))
        print(f"{'tag':16s} {'decode':>7s} {'dec_min':>8s} {'prefill':>8s} {'hit%':>6s} "
              f"{'miss':>7s} {'us/miss':>8s} {'MiB/s':>7s} {'reads':>7s} {'acc':>6s} "
              f"{'mlen':>5s} {'loop':>5s} {'stable':>6s} {'thermal':>15s}")
        for r in rows:
            ml = r.get('draft_mean_len')
            acc = r.get('draft_accept')
            acc_s = f"{acc:.3f}" if acc is not None else "-"
            ml_s = f"{ml:.2f}" if ml is not None else "-"
            # `launch/worst` from the measured rounds themselves, falling back to the sweep
            # bookends for rows recorded before this instrumentation existed. A row with no
            # level at all prints `--`, so an unattributed number looks unattributed instead
            # of looking merely terse.
            th_l = (r.get('thermal_launch') or {}).get('label') \
                or ((r.get('thermal_sweep') or {}).get('launch') or {}).get('label')
            th_w = (r.get('thermal_worst') or {}).get('label') \
                or ((r.get('thermal_sweep') or {}).get('worst') or {}).get('label')
            th_s = f"{str(th_l)[:3]}/{str(th_w)[:3]}" if (th_l or th_w) else "--"
            print(f"{r['tag']:16s} {r.get('decode_tps_median', 0):7.2f} "
                  f"{r.get('decode_tps_min', 0):8.2f} {r.get('prefill_tps_median', 0):8.2f} "
                  f"{r.get('hit_rate_pct', 0):6.1f} {r.get('misses', 0):7d} "
                  f"{r.get('per_miss_us', 0):8.1f} {r.get('io_effective_mib_s', 0):7.1f} "
                  f"{r.get('file_reads', 0):7d} {acc_s:>6s} {ml_s:>5s} "
                  f"{r.get('loopiness', 0):5.2f} {str(r.get('answer_stable')):>6s} "
                  f"{th_s:>15s}")
        return

    rows = []
    if os.path.exists(args.json):
        try:
            rows = json.load(open(args.json))
        except Exception:  # noqa: BLE001
            rows = []
    have = {r["tag"] for r in rows}

    for tag in args.arms.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if tag not in ARMS:
            print(f"unknown arm {tag!r}; known: {','.join(ARMS)}", file=sys.stderr)
            continue
        if tag in have and not args.force:
            print(f"[skip] {tag} already recorded")
            continue

        print(f"\n===== arm {tag}  env={ARMS[tag]} =====", flush=True)
        # Thermal bookends for the whole arm. `pre_kill` is the state we carried in;
        # `launch` is the state the arm is launched into -- the moment the prefill work
        # showed to be the one that carries the claim (read 0 -> 6/6 runs >= 250 t/s;
        # read 1 or 2 -> 0/21). Read BEFORE the model loads, because loading 13 GB is
        # itself a thermal event. Per-round readings come from decode_bench, which stamps
        # both sides of every request.
        thermal_sweep = {"pre_kill": tp.stamp()}
        killed(getattr(args, "port", None))
        thermal_sweep["launch"] = tp.stamp()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ctrl_path = os.path.join(LOG_DIR, f"arm_{tag}_{stamp}.ctrl.log")
        if not start(ARMS[tag], ctrl_path, args.profile):
            print(f"[FAIL] {tag}: server never became ready; see {ctrl_path}", flush=True)
            continue
        srv_log = server_log_of(ctrl_path)
        print(f"  ready; ctrl={ctrl_path}\n         srv={srv_log}", flush=True)

        bench_json = f"/tmp/decode_sweep_bench_{tag}.json"
        if os.path.exists(bench_json):
            os.remove(bench_json)
        subprocess.run(["/opt/homebrew/bin/python3", "scripts/check/decode_bench.py",
                        "--rounds", str(args.rounds), "--warmup", str(args.warmup),
                        "--n-predict", str(args.n_predict), "--tag", tag,
                        "--json", bench_json],
                       cwd=ROOT)
        bench = json.load(open(bench_json))[-1] if os.path.exists(bench_json) else {}
        thermal_sweep["post_bench"] = tp.stamp()

        # SIGINT (not -9) so the teardown stats are printed, then wait for them to land.
        subprocess.run(["pkill", "-INT", "-f", SERVER_MATCH],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(2)
            if srv_log and os.path.exists(srv_log) and \
                    "final stats" in open(srv_log, errors="replace").read():
                break
        subprocess.run(["pkill", "-9", "-f", SERVER_MATCH],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        thermal_sweep["teardown"] = tp.stamp()
        thermal_sweep["worst"] = tp.worst(list(thermal_sweep.values()))

        row = {"tag": tag, "env": ARMS[tag], "log": srv_log,
               "thermal_sweep": thermal_sweep}
        row.update({k: v for k, v in bench.items() if k != "tag"})
        row["loopiness"] = loopiness(bench.get("sample", "") or "")
        row.update(harvest(srv_log))
        row["build"] = build_fingerprint()
        rows.append(row)
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=2)
        print(f"  -> decode {row.get('decode_tps_median')} t/s  "
              f"hit {row.get('hit_rate_pct')}%  miss {row.get('misses')}  "
              f"us/miss {row.get('per_miss_us')}  io {row.get('io_effective_mib_s')} MiB/s  "
              f"accept {row.get('draft_accept')} / mean_len {row.get('draft_mean_len')}  "
              f"loop {row.get('loopiness')}",
              flush=True)
        print(f"     thermal: launch={thermal_sweep['launch']['label']}"
              f"({thermal_sweep['launch']['level']})  "
              f"worst={thermal_sweep['worst']['label']}  "
              f"rounds={row.get('thermal_hist') or '{} (bench not reached)'}",
              flush=True)

    print(f"\nsaved -> {args.json}")
    subprocess.run(["/opt/homebrew/bin/python3", __file__, "--report", args.json], cwd=ROOT)


if __name__ == "__main__":
    main()
