# Decode step budget: verify is 2.4x cheaper per position, and 20% of the wall is outside

Date: 2026-09-19 14:00. Box: 16 GB M4 Air, production profile `prefill250`, 8 GiB pool
(143 slots/layer), greedy, `CGC_FORCE_TEMP0=1`, memory-guarded paired launch
(`CGC_SERVER_MTP=0` then `=1`, back-to-back, same driver, same box).

## 0. The numbers belong to a named build

| artifact | md5[:16] |
|---|---|
| `libllama.0.dylib` | `aab787412e572550` |
| `libggml-metal.0.dylib` | `1be366306c604669` |
| `llama-server` | `054fb22f04a01c5c` |

These are **identical to `summary_anchor0919b.json`**, the M1/M2/M3 = 9/9 PASS anchor. So the
step budget below and the bit-identity gate describe the *same* binary. Raw logs:
`Backup/cgc_logs/llama_server_20260919_135943.log` (off), `..._140009.log` (on).

Arm identity was taken from the running process's **argv** (`--spec-type draft-mtp` present,
`--spec-draft-n-max 3`), not from a log filename.

## 1. End to end: MTP is a small WIN, not a wash

| arm | tokens | ms/token | t/s |
|---|---|---|---|
| MTP off | 92 | 111.04 | 9.01 |
| MTP on | 92 | 107.88 | 9.27 |

`content_head` byte-identical. This is now the **third** paired reading and it disagrees with the
"wash/parity" phrasing of the 13:52 pair (off 117.32 / on 118.74): the sign is not stable at the
1–3% level. What *is* stable is that the absolute numbers move ±8% between launches for reasons
outside the engine (carried swap: 3.44 GB -> 3.57 GB -> 4.49 GB across the three pairs).

## 2. Per step: the verify step is 2.4x cheaper per POSITION

`CGC-DECPROF` step summaries, steady steps only (`step >= 3`), `layers=40` = base model,
`layers=1` = the nextn/MTP head:

| regime | n | total | wait | cb | submit | per position |
|---|---|---|---|---|---|---|
| plain, ntok=1 (off) | 11 | 92.08 | 70.57 | 22.52 | 2.89 | **92.08 ms** |
| verify, ntok=4 (on) | 39 | **152.60** | 114.36 | 35.23 | 4.01 | **38.15 ms** |
| draft, ntok<=4 (on) | 53 | 1.42 | 1.24 | 0.01 | 0.11 | — (1.4 ms/call) |

Two things follow, and both are direct measurements rather than differences of measurements:

* **cb/token falls 2.6x** (22.52 -> 8.81 ms) because 4 positions share one union. The fill is
  union-driven, not token-driven -- so widening a step is also the *cheapest* way to reduce fill.
* **The drafter is 1.42 ms/call**, i.e. ~74 ms of 9925 ms (0.7%). This kills the earlier
  "draft forward = 28-51 ms per draft token" figure for good. That number was `mtp1 - ngram1`;
  measured directly, a draft call is 1.4 ms.

## 3. A two-parameter fit, and why it is not yet evidence

Two regimes give two parameters exactly, which means the fit has **zero degrees of freedom** --
it describes the data, it does not test it:

```
step(ntok) = F + m * ntok      F = 71.9 ms fixed per step,  m = 20.2 ms per token
```

Check: `71.9 + 20.2 = 92.1` (plain ✓), `71.9 + 4*20.2 = 152.7` (verify ✓).

The *shape* this implies is the important part and it is testable: **~72 ms of a decode step is
paid once per step regardless of how many tokens the step carries.** If that holds, then
tokens/step is the cheapest lever available, because the 72 ms is already sunk. Falsification is
one cheap run: measure `ntok=2` and `ntok=8` and see whether both land on the line. Today the only
`ntok=2` point is step 2 of a launch (warm-up, 554 ms) and is unusable.

## 4. Where the fixed 72 ms lives

Per-layer GPU counters (`CGC-GPUTIME`), median over steady steps:

| arm | wait | gpu_union | union/wait | gap | gap/wait |
|---|---|---|---|---|---|
| off | 72.1 | 75.7 | **105%** | 27.2 | 38% |
| on | 115.0 | 98.8 | **86%** | 51.7 | 45% |

`gpu_union / wait` ~ 86-105% means **the GPU is busy for essentially the whole wait window**.
There is no large idle pool to reclaim: the earlier "35.3% idle" reading does not reproduce on
this build. `gpu_busy_sum` exceeds `wait` (113-138%) because GPU work from a step overlaps into
the next step's window, so it is not a utilisation figure -- only `union/wait` is interpretable.

Per layer that is ~1.9 ms. The weight traffic it must move is ~0.9 MB (8 experts x ~111 KB at
IQ3_XXS for a 3B-active model), i.e. **~7 us at 120 GB/s**. So the two walls are not where the
I/O narrative put them:

* bandwidth floor for a step: 3e9 params x 3.06 bit / 8 / 120 GB/s = **9.6 ms/token**.
  Measured hot-path GPU time ~29-38 ms/token => **~3-4x off the floor.**
* therefore decode is not bandwidth-bound and not I/O-bound at the floor. It is bound by
  **per-layer fixed cost** (~1.9 ms x 40 layers), i.e. op count, launch/dispatch and sync
  latency at M=1 (GEMV): the bytes per op are ~7 us and the op costs ~1.9 ms.

## 5. 20% of the wall is inside no `llama_decode` call

The step index runs to 198; 95 summaries are printed, so the profiled set is a subset and the
time budget must be closed from the *counts*, not by summing printed lines.

Expected split at 96 tokens, accept 0.51, k=3 (`E = 1.903` tok/round): **50.5 base steps + 151
draft steps = 202 total**, against 198 observed indices -- consistent, which independently
confirms both the accept rate and the round structure.

```
time inside llama_decode (on)  = 50.5*152.60 + 151*1.42 = 7914 ms  of  9925 ms wall
UNATTRIBUTED                   = 2011 ms = 20.3%  =  39.9 ms per round
plain arm                      = 92*92.08 = 8471 ms of 10215 ms wall -> 17.1% (19 ms/token)
```

So ~20% of decode time happens outside the profiled step, in both arms. It is not an MTP
artefact (the plain arm has its own 17%), but it is the **largest single unattributed block** and
nothing in the current instrumentation can see it. This is the same class of gap as the project's
other invisible-by-construction costs: real, large, and only findable by putting a counter where
there is currently none.

## 6. What this means for 25 t/s

Using the measured round cost (194.3 ms/round = 152.6 step + 41.7 unattributed) and `E(a)`:

| accept | E | t/s at today's cost | t/s with the 20% removed |
|---|---|---|---|
| 0.51 (today) | 1.90 | 9.79 | 12.2 |
| 0.70 | 2.53 | 13.04 | 16.6 |
| 0.80 | 2.95 | 15.19 | 19.3 |
| 0.90 | 3.44 | 17.70 | 22.5 |
| 1.00 | 4.00 | 20.59 | **26.2** |

* Accept alone cannot reach 25 (ceiling 20.6 t/s at today's step cost, even with a perfect drafter).
* Removing the unattributed 20% is worth **+26% today** and is bit-identity neutral.
* Reaching 25 requires the per-step fixed cost to fall: with accept 0.80, `F` must go
  **72 -> ~17 ms**, i.e. ~0.42 ms/layer instead of ~1.9 ms/layer. That is a kernel/graph change
  (fewer ops, fewer syncs, wider M), not a cache change.

Ranked next steps:

1. **Attribute the 20%** (39.9 ms/round). Cheapest, no bit-identity risk, largest immediate win.
   Counter pairs around the speculative round in `common/speculative.cpp`.
2. **Validate the F/m line** with `ntok=2` and `ntok=8` before building anything on it.
3. **Raise tokens/step** (accept or larger k): each marginal token costs 20.2 ms while 71.9 ms is
   already sunk -- the cheapest yield available while `F` stands.
4. **Attack `F`** (1.9 -> 0.42 ms/layer). The only route to 25 t/s; needs a falsifiable target.

Dead, with reasons: split-MMID fill overlap (any H/C split that is not a prefix changes the
left-to-right add association at `llama-graph.cpp` combine => not bit-identical; a prefix before
the first cold position is typically 0-1 of 8 positions, so there is nothing to overlap).
Prefetch (the dominant miss is first-touch/compulsory). Replacement policy (simulated: LRU-2 -6%,
two-touch filter +35% worse, frequency pinning +3% worse; only oracle Belady wins).

## 7. The k sweep: two negative results that change the plan

Added 14:12. Driver `/tmp/k_sweep.py`, arms k=1 (ntok=2), k=7 (ntok=8), k=3 (control), MTP on,
same profile/pool. Raw: `Backup/cgc_logs/llama_server_20260919_1412*.log`, `..._141304.log`.

**A. The widest verify is not reachable at the production geometry.** k=7 aborted on the very
first request:

```
ggml_metal_synchronize: error: command buffer 8 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
CGC-METAL-FAIL: command buffer 8 failed (status 5) ... refusing to return stale output
run_server.sh: line 2281: 23514 Abort trap: 6
```

This is the same failure the launcher already documented for `ub=6144` at the same 8 GiB pool
(0/5 launches survived, always `status 5` OOM). So **ntok=8 cannot be measured at this geometry**,
which kills the "widen tokens/step" plan as stated: at 8 GiB the usable decode width stops at 4.

**B. `F` and `m` are not identifiable, so the line in section 3 cannot be tested this way.**
k=1 completed (123.22 ms/token, E=1.51 -> ~186 ms/round) but its per-step `cb` is 6-12x the k=3
run's -- 72, 130, 252, 275 ms against 35 ms -- at a *narrower* union, which `cb` should make
cheaper, not 7x dearer. Same log, same build; what differed was the machine state (the k=7 OOM
left swap at 4287 MB, and the arm launched at free=71 MB).

So the three width points are each taken in a different memory state, and `cb` -- the term that
varies most -- is a function of that state, not of the width. Combining them yields
`F+m*2 = 186`, `F+m*4 = 205` => `m = 9.5`, `F = 167 ms`, which disagrees with section 3's
`m = 20.2`, `F = 71.9` and means **neither pair is measuring the same quantity**.

And there is no way out of this with the current instrument: within one launch the decode width
is a constant (`CGC-PHASE-SPLIT` sets one width per launch), so width can only be varied across
launches, where the pool state is not held fixed. **The fixed-vs-marginal split is therefore not
measurable on this box as instrumented.** Section 3's numbers must be read as a description of two
offsets, not as a model with a fixed and a marginal part.

Consequence for the plan: item 3 ("raise tokens/step") is blocked twice over -- width 8 OOMs the
Metal budget at 8 GiB, and the exchange rate between width and step cost is not measurable. The
surviving levers are the two that do not need a width sweep: attribute the 20% (item 1), and
attack per-layer cost (item 4).

## 8. Is the 20% real, or is it the instrumentation that found it?

Everything in section 5 came from runs with `CGC_DECODE_PROFILE`, `CGC_DECODE_PROFILE_ALL` and
`CGC_GPU_TIMING` all armed -- and `_ALL` printed ~1700 per-layer rows for 95 steps, i.e.
synchronous writes into the decode loop. The 20% must therefore be checked against a run with the
profilers off before anyone spends a rebuild on counters inside the loop.

Driver `/tmp/noprof_arms.py`, both arms with the instruments **off**, two requests each (the
second reuses the pool: `cache_prompt: false`, so the KV is not reused but the *expert pool* is).
Raw: `Backup/cgc_logs/llama_server_20260919_1415*.log`, `/tmp/noprof.json`.

| arm | rep1 (cold pool) | rep2 (warm pool) |
|---|---|---|
| MTP off | 117.72 ms/tok (8.49 t/s) | **75.28 ms/tok (13.28 t/s)** |
| MTP on | 144.60 ms/tok (6.92 t/s) | **90.62 ms/tok (11.04 t/s)** |

**The profilers are not the cause.** 117.72 (off, unprofiled) is not faster than 111.04 (off,
profiled at 14:00), so the ~20% is not an artefact of the instrumentation that found it. This
compares two launches, so it bounds the claim rather than proving it -- it rules out "the gap is
my own logging", which was the reason to hesitate before writing loop counters.

### Three things this run says that the earlier ones could not

**1. Pool warmth, not batch width, is the dominant variable -- and it is worth 42-54 ms/step.**
This is a *within-launch* comparison (same server, same width, same build, same machine state),
which is the control section 7 said did not exist:

| | cold | warm | delta |
|---|---|---|---|
| MTP off | 117.72 | 75.28 | **-42.4** |
| MTP on | 144.60 | 90.62 | **-54.0** |

That is 3-5x larger than anything the width sweep was trying to resolve, which is why the width
sweep could not resolve it. `cb` is not a small correction term in the cold state: it is the
largest single component of a decode step.

**2. 13.28 t/s is the warm baseline -- and it independently matches the project's own 14 t/s.**
Every decode figure in this document (and in the 25 t/s gap arithmetic built on it) was measured
on request 1 of a fresh server, i.e. **cold pool**. The warm figure 13.28 t/s agrees with the
`run_server.sh`-over-HTTP number the project has been quoting (`~14 tok/s`), so the two families
are measuring the same thing and the colder ones are the outliers, not the other way round.

**3. MTP is a measured LOSS in the warm state, not a wash and not a win.** 90.62 vs 75.28 =
**-20.4%**, and -22.8% in the cold state. Every earlier reading of "parity" came from
single-request (cold) pairs whose spread (±8% across launches) was the same size as the effect
being looked for. In the state that production actually serves from, the sign is stable: the
4-wide verify step costs 172 ms warm (90.62 x 1.903) against 75.28 ms for a plain step, so the
yield of 1.9 tokens is bought at 2.28x the step cost. Break-even needs accept 0.80 (E=2.95 ->
58.4 ms/tok, a 1.55x win over 13.28 t/s); today's accept is 0.51.

### What the 25 t/s target now looks like

Against the **warm** baseline (75.28 ms/token = 13.28 t/s), the gap is 1.88x, not the 2.4-2.8x
that the cold numbers implied. The arithmetic is unchanged in form and the conclusion survives:

| accept | E | ms/token at today's 172 ms verify step | t/s |
|---|---|---|---|
| 0.51 (today) | 1.90 | 90.5 | 11.04 |
| 0.70 | 2.53 | 68.0 | 14.71 |
| 0.80 | 2.95 | 58.3 | 17.15 |
| 0.90 | 3.44 | 50.0 | 20.00 |
| 1.00 | 4.00 | 43.0 | 23.26 |

Accept alone still cannot reach 25 (ceiling 23.3 t/s even with a perfect drafter), so the same two
terms are needed: fewer ms per verify step **and** a higher accept. But the target is now reachable
by a smaller margin than the cold baseline suggested, and the 20% unattributed time (section 5)
becomes worth more, not less: removing it is the cheapest way to buy the required step-cost cut.

## 9. The 20% does not exist -- it was my own median-vs-mean artefact

Added 14:29. Section 5 claimed ~20% of the decode wall fell outside any `llama_decode`. That
claim is **wrong**, and the instrument that disproves it was already in the source.

`server-context.cpp` already wraps the four loop phases with timers that were never switched on:
`t_pre_decode`, `t_decode`, `t_post_decode`, `t_sampl`, printed by `DEBUG_TIMINGS` (a commented-out
`// #define` at line 2953). Enabled per section 10, MTP off, three requests (128 cold then 2x384):

| rep | t_pre_decode | t_decode | t_post_decode | t_sampl | wall ms/token |
|---|---|---|---|---|---|
| rep1 cold | 0.14 | 206.50 | 1.22 | 1.21 | 124.19 |
| rep2 warm | 0.16 | 189.41 | 1.83 | 1.81 | 134.89 |
| rep3 warm | 0.17 | 176.10 | 1.98 | 1.93 | 115.26 |

(`t_sampl` is nested inside `t_post_decode` for the non-speculative path -- the timer at
server-context.cpp:4092 sits inside `post_decode` -- so it is a breakdown of `t_post_decode`, not a
fifth phase. `t_decode` is a *per-call* average and therefore dominated by the prompt decode
(~5.3 s), which is why it is ~176 rather than ~115; solving `(3P + 276G)/279 = 176.1` with
`G = 120` gives `P = 5337` ms, matching the prompt rate of ~310 ms/token at 17 tokens.)

**Every phase outside the decode is ~2 ms per call. pre + post + sampl = 2.2 ms out of ~110 = 2%,
not 20%.** And `t_post_decode` -- which contains the `assign(p, p + n_vocab)` full-vocab logits copy
for each speculative position -- is 1.2-2.0 ms, so that copy is not a lever either.

Where the 20% came from: section 5 multiplied the **median** step (92.08 ms) by the token count and
compared it against a **mean**-derived wall (111.04 ms/token). `111.04 / 92.08 = 1.206` -- the
two-figure "20%" is exactly that ratio, i.e. the right-skew of the step distribution, not missing
time. The budget closes: the server's own `predicted_per_token_ms` *is* the mean, so comparing it to
`median x count` was guaranteed to invent a gap.

Consequences:

* **Item 1 ("attribute the 20%") is closed with nothing to fix.** There is no host-side block to
  reclaim. The decode time is inside `llama_decode`, which makes item 4 (per-layer cost) not just
  the best lever but the *only* one left on the compute side.
* The sampler is **1.9 ms/call (1.6%)**. It is not worth optimising, and neither is the logits copy.
* Every earlier "median x N vs wall" accounting in this document is now known to be invalid; only
  sums of means should be used for budget claims.

## 10. Turning the instrument on, and what it costs

One line, upstream's own switch -- no new counters, because this project has repeatedly been burned
by hand-written counters that measured the wrong thing:

```diff
-// #define DEBUG_TIMINGS
+#define DEBUG_TIMINGS     // server-context.cpp:2953
+// [CGC 2026-09-19] note: also enables llama_synchronize(ctx_tgt) after each decode (~line 3057),
+// so absolute timings are NOT comparable to a build with it off. Read the bucket RATIOS.
```

`cmake --build build --target llama-server -j6` rebuilds only `server-context.cpp` ->
`libserver-context.a` -> `libllama-server-impl.dylib` -> `llama-server`. Digests confirm the blast
radius, which matters because the build directory is shared with parallel sessions:

| artifact | before | after |
|---|---|---|
| `libllama.0.dylib` | `aab787412e572550` | **unchanged** |
| `libggml-metal.0.dylib` | `1be366306c604669` | **unchanged** |
| `libllama-server-impl.dylib` | `a87bfd2e80a086b4` | `762b8af267c91f88` |
| `llama-server` | `054fb22f04a01c5c` | unchanged (thin exe) |

So the core engine digest stays the M1/M2/M3 anchor's, and `llama-bench` -- a separate binary that
does not link `libllama-server-impl` -- is unaffected. **The cost is real though: the extra
`llama_synchronize` per decode made this arm ~115-135 ms/token against 75-117 in the unprofiled
run, i.e. tens of ms/token.** It is a diagnostic build, not a production one.

**Current state: reverted, and the artifact digest proves it.** The build directory is shared with
parallel sessions and other sessions identify a number by the digest it was taken on, so a
diagnostic that slows the production path -- or even a comment-only source change, which shifts the
line numbers the binary embeds -- must not be left in place:

```
libllama-server-impl.dylib   a87bfd2e80a086b4   <- rebuilt back to the original, byte-identical
libllama.0.dylib             aab787412e572550   <- never changed (the M1/M2/M3 anchor build)
libggml-metal.0.dylib       1be366306c604669   <- never changed
llama-server                 054fb22f04a01c5c   <- never changed
```

The return to `a87bfd2e80a086b4` after reverting the source also re-confirms build determinism:
same source state in, identical artifact out. To re-enable the diagnostic, apply the diff above and
rebuild that one target (~1 min); the findings do not depend on it being on.

## 11. The kernel-family hypothesis (item 1 of the plan), and why it is the top item

> **REVISED 2026-09-19 — read §15 before acting on this section.** The mechanism assumed below (that
> the 4-wide verify "runs on `mul_mv` instead of the kernel built for it") does not hold for this
> model's weights: its expert tensors are IQ2_S / IQ3_S / IQ4_XS, which the small-batch family admits
> at *no* batch size. The ceiling is 2.4% of expert weight bytes, and the A/B's 15.5% effect is six
> times that — so the effect was never this mechanism. What follows is the reasoning as it was written
> *before* the check, kept because the way it failed is the useful part.

`CGC_MM_BITIDENT=1` is a production default (`run_server.sh:2098`) and it is a **bit-identity
pillar**. Its own comment says what it costs (`ggml-metal-ops.cpp:2451-2470`):

```
ne11=2 -> nxpsg=16/r1ptg=2, ne11=3 -> nxpsg=8/r1ptg=3, ne11=1 -> not even eligible (mul_mv)
CGC_MM_BITIDENT=1 ... all ne11 <= 8 land on mul_mv
Perf note: M in [2,8] loses the small-batch GEMV (falls back to mul_mv)
const int ne11_mm_min = 8;   // break-even where the mat-matrix kernel becomes more efficient
```

The small-batch mat-mv family exists precisely for BS [2, 8] — the speculative-verify regime — and
the bit-identity mode bypasses it for every type. So the 4-wide verify runs on `mul_mv`
(matrix-**vector**) instead of the kernel built for it, which is a candidate mechanism for the
measured **172 ms verify step vs 75 ms plain step (2.28x for 1.9 tokens)**.

The proposed experiment has a **sharp built-in control**, because the source says `ne11=1` is not
eligible for that family:

| | ne11 | BITIDENT=1 | BITIDENT=0 | predicted change |
|---|---|---|---|---|
| plain step | 1 | mul_mv | mul_mv | **none** |
| verify step | 4 | mul_mv | small-batch mat-mv | **large** |

If both move equally, the hypothesis is refuted. This is an UPPER-BOUND measurement only: the
BITIDENT=0 arm is expected to break M1/M2, so it may tell us whether an M-invariant small-batch
kernel is worth building, and nothing more.

Staged driver `/tmp/bitident_ab.py`, arms b1 / b0 / b1 (the two baselines bracket the test, so
drift can be distinguished from effect), read from the `ntok=4` steady step summaries:

```bash
CGC_SERVER_PORT=8199 CGC_SERVER_PROFILE=prefill250 CGC_SERVER_MTP=1 CGC_DECODE_PROFILE=1 \
CGC_DUMP_ENV=1 CGC_MM_BITIDENT=0 bash scripts/run_server.sh
# then compare ntok=1 and ntok=4 CGC-DECPROF step medians against the BITIDENT=1 arm
```

## 12. Pool warmth as a mechanism (items 6-8), and the damage to undo first

The `cb` collapse is certain (20.41 -> 4.80 -> 3.66 ms, thrash -> 19.37), but a slab prefill leaves
damage that `cb` does not explain: after thrash, rep4's `cb` recovered to 4.63 ms while `wait`
stayed at 78.72 ms against 63.88 ms warm. Two things to measure, both staged:

1. **Per-layer location of the residue** — `CGC_DECODE_PROFILE_ALL=1` prints per-layer rows, so the
   before/after thrash comparison shows whether `gpu_union` degrades globally or in a few layers.
2. **Does a re-prewarm restore the rate?** — if a warmup pass after prefill returns `wait` to 63.88,
   then the fix is "end every prefill by re-warming the decode set" (the `SLAB_POOL_HANDOFF`
   direction), not a change to the pool's contents.

Why this also matters to the parallel S1 work: the effect is measured *as a slab-prefill effect*
(2000-token slab chunk reads all 256 experts per layer), so the size of the slab and what it does to
the decode working set is directly relevant to `Backup/phase_decomp/s1_prefill2025_*`.

## 13. Measurement hygiene this run produced (do not re-learn)

* **Never `pkill -f build/bin/llama-server`.** It matches every other session's server. It did kill
  one here (a `-expert-cache 1073741824`-class 10 GiB server that was not ours; our profile is
  8 GiB). Cleanup in a probe must be `kill <pid on our own port>`, and a probe should run on a
  dedicated port (`CGC_SERVER_PORT=8199`) so it never needs to touch anyone else's.
* **Never sum medians and call it a wall.** Section 9 exists because of that.
* **Never trust a single request.** Cold vs warm is worth more than any effect measured in
  sections 2-4.

## 14. The two lanes are now one harness (run this, not §11/§12 by hand)

§11 and §12 describe the two experiments as prose with exact commands. Running them by hand is how
earlier numbers ended up mixing regimes (cold vs warm, launch drift, interference from a parallel
session). They are now one script that waits for a real window, runs both lanes, compares the arms
itself, and writes the judging report:

```bash
python3 scripts/check/decode_window_harness.py run            # waits, runs, then self-reports
python3 scripts/check/decode_window_harness.py render-sample  # check the layout, no measurement
python3 scripts/check/test_decode_window_harness.py           # verdict + render self-test
```

The harness encodes the corrections this run paid for:

* **Lane A is `b1 / b0 / b1`** so two BITIDENT=1 baselines bound the box's drift. An effect inside the
  spread is reported `INCONCLUSIVE`, and `BASELINE_AGREE_PCT` (5%) is both a noise floor and a drift
  budget -- if the two baselines disagree by more than it, the run is not admissible and says so.
* **A window, not a launch.** It waits for no llama process anywhere, both our port and 8080 idle,
  free >= 4 GB, and no writes to `Backup/cgc_logs` for 90 s. An arm is `done` only after every
  request returned; an interrupted arm is re-queued and never recorded as data.
* **It signals only its own port.** Cleanup targets the pid holding `CGC_SERVER_PORT` (8199) and
  refuses to start if that port is taken. There is no `pkill -f llama-server` anywhere in it -- that
  pattern killed another session's 10 GiB server once, and is banned here for that reason.
* **Anchor guard.** Every arm is checked against `ANCHOR` (the digests the M1/M2/M3 gate passed on).
  A build that moved is marked `anchor-mismatch` and reported as INCOMPLETE with the cause, instead
  of being stamped with a digest nothing covers (`PROBE_ANCHOR=none` opts out for a new campaign).
* **Mean, never median x count** (§9), and every decode figure carries its pool regime.
* **The report is written from the first second and refreshed while waiting.** An earlier version
  wrote it only when the whole run finished, so a harness still waiting for a window (or killed
  while waiting) left no report at all and "why is there no output?" was unanswerable on disk. It
  now reports `INCOMPLETE` plus the live blocker and a `SIGTERM`/`SIGINT` handler so killing it
  still writes what is known. A report with no measurement in it says so in a banner; do not read
  the two `INCOMPLETE` lines as a result either way.

## 15. Item 1's premise is void for this model: the knob cannot reach the expert weights

§11 rested on an **eligibility** claim, and eligibility is a property of the weights the model
ships — not of the knob. Checked against the real GGUF (path resolved by asking the launcher for
`CGCENV MODEL`, not hard-coded):

| type | tensors | of which expert | expert MiB | admitted to small-batch |
|---|---:|---:|---:|---:|
| IQ2_S | 78 | 78 | 6396 | 0 |
| IQ3_S | 39 | 39 | 4290 | 0 |
| IQ4_XS | 253 | 3 | 408 | 0 |
| Q2_K | 2 | 2 | 168 | 168 |
| Q3_K | 1 | 1 | 110 | 110 |
| Q8_0 / F32 / BF16 / Q6_K | 380 | 0 | — | admitted, but no experts |

The small-batch family admits exactly two disjoint type lists — `{F32,F16,BF16,Q1_0,Q2_0,Q4_0,
Q4_1,Q5_0,Q5_1,Q8_0,MXFP4,IQ4_NL}` at ne11 2–8, and `{Q4_K,Q5_K,Q6_K,Q2_K,Q3_K}` at ne11 4–8
(`ggml-metal-ops.cpp:2476-2504`) — plus `ne00 % 128 == 0`. IQ2_S, IQ3_S and IQ4_XS are on neither.

**Ceiling: 278 of 11,372 MiB = 2.4% of expert weight bytes.** Any kernel-family saving on the verify
step is capped near 2.4% of its expert-matmul cost — and that assumes the small-batch kernel were
*infinitely* faster on those bytes.

### The two instruments disagree by 6x, and that disagreement is the result

| instrument | claim about the verify step |
|---|---|
| the weights (source-level) | at most **2.4%** |
| the A/B (measured, §14 lane A) | **15.5%** (b1 166.4 → b0 140.6 ms) |
| the drift lane (measured) | the same configuration moved **−60%** between launches (block 2: 337.2 → 133.6) |

An effect six times the mechanism's ceiling is not the mechanism. "The effect is real" and "the
effect is zero" are both wrong readings of §11; the correct one is **item 1 does not apply to this
model**. The −8.1% reported earlier sat inside a 17.5% baseline spread; it is now excluded on
mechanism grounds as well, which is a stronger statement than "underpowered".

### Why interleaving could not squeeze the baselines under 5%, and the run was stopped at 5/12

| block | arms (verify step, ms) | within-block span |
|---|---|---|
| 0 | b1 325.7 / b0 345.3 / b1b 392.5 | 18.8% — and the **same-config** pair b1/b1b differs by **20.5%**, worse than §11's original 17.5% |
| 1 | b1 337.2 / b0 133.6 | **86.5%** |

`cb` is constant across every arm (8.7–10.8 ms), so this is not pool state; the entire swing is
GPU-side `wait`. The cause is that the box is shared: another session's
`llama-bench -p 0 -n 192 -d 512 --spec-type draft-mtp -r 3` ran on the same GPU, alongside
Spotlight bulk-import (`mdbulkimport`), Time Machine `backupd` and two Electron apps at 40%+ CPU.

Pairing cancels a drift that is slow relative to one arm. Here the regime flips **faster than one
arm takes** (~4 min), so a pair's two members are never in the same regime. Three consequences,
now enforced in code rather than asserted in prose:

* a block whose arms span more than `REGIME_FLIP_PCT` (20%) is reported as `regime faster than one
  arm` instead of being silently averaged;
* a single pair is never called "controlled" — one ratio has no spread by construction, which is
  exactly how a one-block run would have produced a false `resolved`;
* when a ceiling and an A/B both exist and the A/B exceeds the ceiling by more than 2x, the report
  says **`not attributable`** outright.

**One GPU serves one trustworthy measurement.** Our A/B running beside theirs contaminates both
sets of numbers, so stopping was the correct move — not adding blocks.

### The shape that could answer it (if a future model makes it live again)

Not a cross-launch A/B. The only shape without a cross-launch component times both kernel families
**inside one process, seconds apart** — the method used earlier for GDN. `CGC_MM_BITIDENT` is read
into a function-local `static const` (`ggml-metal-ops.cpp:2469-2472`), so it cannot be toggled in a
live server; a standalone ggml-metal micro-benchmark (M = 2/3/4/8 `mul_mat` at the real shapes, run
once with and once without the env) costs seconds per rep and pairs its arms ~2 seconds apart
instead of ~4 minutes. Given the 2.4% ceiling this is **not** on the 25 t/s path.

## 16. Decode width, and what F is actually made of (two passes, both refused by the box)

Asked for: the S(ntok) curve for ntok = 1..4 with MTP **on**, and F split per layer into attention /
MoE gather / eval-sync / dispatch. Tool: `scripts/check/sntok_curve.py`.

Design, and why it looks the way it does: `speculative.n_max` cannot be changed per request
(`server-schema.cpp:200` sits inside `#if 0` — "we disable speculative parameter adjustments for
now"), so each width needs its own launch. A launch is 1.7 min warm-cached or ~4 min cold, and this
box's regime flips in about that long, so **every test arm is bracketed by ntok=4 anchors co-measured
minutes away** rather than by an anchor from earlier in the run.

| pass | anchors (ntok=4) | arm cost | anchor spread | verdict |
|---|---|---|---|---|
| 1 | 185.3 / 189.9 / 134.1 ms | ~4 min | **34.5%** | `regime faster than one arm` |
| 2 | 145.7 / 327.3 / 212.3 / 128.9 ms | ~1.7 min | **97.5%** | `curve not admissible` (raw vs bracketed fits disagree 171% on F, 1234% on c) |

**The fact that settles it**: pass 2 logged **zero** `window lost` / `llama running` events. Every arm
ran inside a guard-approved quiet window — no other llama process, both ports free — and the anchors
still spanned 2.5x. So the regime is ambient (Spotlight bulk-import, TimeMachine, GUI load), and
"wait for a quiet window" does not remove it. **The curve is not obtainable on this box by any ladder
affordable here**, and the harness says so instead of averaging it.

### The exception that gives the answer anyway

`--spec-draft-n-max 0` was the intended route to a 1-token step inside the MTP-on env block. It
**aborts on the first decode**:

```
llama-context.cpp:2961: GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed
```

This falsifies the repo note that k=0 is a legal baseline (`spec_cost_curve.py` relied on it). So the
DESIGN GAP (no ntok=1 base step with MTP on) stands — now with a mechanism, not an assumption. The
ntok=1 step is measured with MTP **off** instead, and the curve extrapolated to ntok=1 agrees with it
to **-2.5%** (72.9 vs 71.1 ms): the env block does not materially move F, which is exactly the
caveat the gap was about.

**F = 71.1 ms/step = 1.81 ms/layer** (40 base layers; 77.3 ms/token). Composition, as sums of
per-layer medians (arm r0 = MTP off, ntok=1, the only direct reading of F):

| bucket | sum ms | per layer | share |
|---|---:|---:|---:|
| `union` — GPU span | 50.5 | 1.26 | **70%** |
| `gap` — GPU idle between segments | 14.5 | 0.36 | 20% |
| `cb` — expert fill (slot mgmt + blocking I/O) | 4.2 | 0.10 | 6% |
| `submit` — dispatch of the layer's segment | 3.0 | 0.08 | 4% |

Three things follow, and they answer the "1.9 -> 0.5 ms/layer" sentence:

1. **At most 30% of F is addressable as overhead.** Zero every non-GPU millisecond and F is still
   1.26 ms/layer — **2.5x** the target. The 25 t/s ask is therefore a statement about **GPU execution
   (attention + MoE math)**, not about dispatch, prefetch or cache fill.
2. **The GDN layers are the expensive ones**, not the full-attention ones: +0.48 ms/layer `union`
   over the 10 attention layers (30 layers => 14 ms/step = 20% of F). The target needs 52 ms/step of
   cut, i.e. **3.6 gaps of that size** — one perfect GDN fix is a fraction of the ask.
3. **The warm fill lives in the first 5 layers**: at ntok=4 warm, 92% of `cb` sits in L0-L4 (8.92 of
   9.75 ms) with L5+ at 0.83 ms; cold it is spread (L0-L4 10.40, L5+ 21.95). Reproduced across both
   passes and both anchors. No step total can show this, and it is the actionable form of the "cb
   collapsed" claim the earlier docs assert three times without a layer split.

### Limits of this instrument (quoted with the numbers above, not after them)

* `gpu` sums `end-start` over a segment's buffers and **double-counts overlapping buffers** (it can
  exceed `wait`); `union` is the honest span and is what the table uses.
* The four buckets **do not strictly partition the step**: closure runs 75-121% across arms (worst off
  25%), so composition percentages carry that width. At ntok=1 closure is 102%, which is the width
  quoted for F's split.
* Absolute attention vs MoE cannot be split without per-node timing. The GDN/full-attention contrast
  bounds the differential only; a common per-layer MoE cost cancels in it.
* Artifacts: `scripts/check/sntok_curve.py`, `/tmp/sntok_curve.json` (pass 2),
  `/tmp/sntok_curve_pass1.json` (pass 1). No engine source was modified by either pass.

## 17. Boundaries

* `free` was 60 MB during both arms; swap grew 128 MB (off) / 922 MB (on) across the pair. The
  ratio is defensible because both arms ran back-to-back on one machine state; the absolute
  ms/token is not, and it moved 111 -> 108 -> 118 -> 107 across four launches.
* The accept rate (0.51 = 76/149) is from the MTP-on arm of the 13:52 pair, not from this run;
  this run's `n_decoded`-based round count (198 indices vs 202 predicted) is *consistent* with it
  but is a consistency check, not a re-measurement.
* `F`/`m` is a 2-point, 2-parameter fit. See section 3.
* Single prompt, greedy, temp 0 -- nothing here covers rejection sampling at temp > 0.
* `gpu_busy_sum`/`gap` are not utilisation figures (see section 4).
* **rep1-vs-rep2 conflates two things**: a warm expert pool, and the passing of any one-time
  first-request cost (Metal shader first-use, graph warm-up). Both would produce the same 42-54
  ms/step drop. Separating them needs a third request (does rep3 == rep2?) plus a deliberately
  pool-thrashing request before it. Until that runs, "pool warmth" is the leading explanation, not
  a demonstrated one; what *is* demonstrated is that "request 1 of a fresh server" and "a later
  request" differ by more than any effect measured in sections 2-4.
* Every rate in sections 1-4 is therefore a **cold-pool** rate. Section 8 is the only warm one.
* Pool warmth is measured three times and the **wall** effect is not stable: -36% (14:15, no
  profilers), -27% (14:25, DECPROF on), **~0%** (14:29, DEBUG_TIMINGS on). The one constant across
  all three is `cb`: 20.41 -> 4.80 -> 3.66 ms with the thrash driving it back to 19.37. So the fill
  collapsing is certain; its translation into wall time depends on scheduling/overlap and is masked
  when a per-decode `llama_synchronize` is present (the DEBUG_TIMINGS build). A production-like
  third reading with 3+ reps and no instrumentation is still needed to pin the wall gain.
* Section 9's `t_decode` is a per-call average mixing prompt decodes with token decodes, so it is
  not a per-token figure. Only its relation to the wall (that it accounts for essentially all of it)
  is being claimed.

Machine left clean (0 servers, 0 drivers), and no engine source was modified by any of the runs
above -- the five build artefacts were returned bit-for-bit to the gated anchor, which is also how
build determinism got re-confirmed. The only files this work added are the harness and its
self-test in `scripts/check/` (§14) plus scratch drivers in `/tmp`.

Additional boundaries for §15:

* §15's 2.4% is a **byte** share used as a proxy for a **time** share. It assumes uniform cost per
expert weight byte; a mix of IQ2_S / IQ3_S / IQ4_XS satisfies that approximately, not exactly, so
treat 2.4% as an order-of-magnitude ceiling rather than a computed limit.
* §15 is about *this* model. On weights whose experts are Q4_K/Q5_K/Q6_K (admitted at ne11 4-8) the
premise is live and the ceiling is 100% -- the gate reports that case as `premise bounded` with a
100% share, which is how it can say both "live" and "void" without being asked which model.
* The drift lane's five completed arms are a **distribution of this machine's regimes**, not
measurements of BITIDENT. They must not be averaged into a single number; the two blocks disagree by
more than any effect the lane was built to detect, and that is the point of keeping the rows.
* The family gate reads the weights the launcher reports (`CGCENV MODEL`), so a profile change that
swaps the model cannot leave the gate describing a model nobody loaded.
* The three failures the new self-tests caught (`resolved` from a single pair; a missing
`expert_eligible` key; a stray paren) are recorded here because two of them were *conclusions*, not
typos: the harness exists to stop exactly that class of number from reaching a report.

Report for this run: `docs/DECODE_WINDOW_2026-09-19.html` (regenerated by
`scripts/check/decode_window_harness.py run --analyze-only`).
