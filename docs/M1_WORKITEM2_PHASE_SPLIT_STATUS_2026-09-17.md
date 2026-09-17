# M1 work items 2 + 3 — phase split and cap demotion (2026-09-17)

Status: **implemented and verified**; two defects found and fixed on the way, one open decision.
Binary: `build/bin/libllama.0.0.279.dylib`, built 16:41 (source edits 16:32 phase rule, 16:41 banner).

## What was implemented

**Work item 2 — the phase is decided from the request's token count, before graph construction.**
The phase had been an emergent property of one expression (`n_tokens <= cgc_pool_max_tokens()`)
evaluated independently at six sites: the routing-mask leaf, the remap/S1 leaf, hot prewarm,
prefetch/SPAC refresh, the hook's large-batch branch, and the `n_batch` clamp. Six copies is six
chances to disagree, and the failure is silent: one side builds a phase while the other serves a
different one, which is either `tensor buffer is nil` or a remap leaf filled with the previous
step's ids. Both were measured during work item 1.

The predicate now lives in `src/llama-cgc-canon.h`'s sibling `src/llama-cgc-phase.h`, both sides
call it, and the env var is read in exactly one place:

    cgc_decode_width(slots, top_k, cap) = min(cgc_decode_bound(slots, top_k, cap), T_prefill - 1)
    cgc_select_graph_phase(n_tokens, decode_width) = n_tokens <= decode_width ? DECODE : PREFILL

  * DECODE (`CGC_GRAPH_DECODE`): pool path, remap leaf / S1 table, FFN reads pool regions by slot.
  * PREFILL (`CGC_GRAPH_PREFILL`): whole-layer slab, `ne[2] = n_expert = 256`, **raw** ids, pool
    not involved.

T_prefill (`CGC_PREFILL_THRESHOLD`, default 512) is folded in **once**, in `cgc_decode_width`, so
the clamp and the predicate cannot disagree. An earlier revision re-checked the threshold inside
the predicate and the two diverged for `CGC_PREFILL_THRESHOLD <= cap` — the clamp allowed a step
that the predicate then sent to the prefill graph with no slab armed and raw ids against shrunk
tensors. `scripts/check/phase_split_selftest.cpp` pins this (and pins that the threshold is
currently non-binding, so raising the cap ceiling makes the gate speak up instead of silently
changing which graph serves which step).

**Work item 3 — cap demoted to the decode graph's width ceiling.** Not derived from slots, and no
longer the phase switch. Measured with real geometry, same pool (`routable=142 slots`, top_k=8 →
bound `floor(142/8) = 17`):

    cap=8   -> decode graph width=8   (bound=8)    cap is the binding ceiling
    cap=64  -> decode graph width=17  (bound=17)   cap lowered 64 -> 17 and raised nothing

## Evidence

### A. The failure case the old predicate mis-routed

Old predicate: `n_tokens <= cap`. With cap=64 a 35-token step took the **pool** path even though
its worst-case union (35 × 8 = 280) exceeds the 142 routable slots — the shape that produced
`buffer is nil` historically. New predicate: 35 > 17 → PREFILL graph. Measured (log
`llama_server_20260917_165758.log`, cap=64, profile prefill250):

    CGC-PHASE-SPLIT: cap=64 routable=142 slots / top_k=8 -> decode graph width=17 tokens (bound=17, ...)
    CGC-PREFILL-STREAM: il=0 kind=0 ntok=35 experts=256 slab=110.00 MiB filled=115343360 bytes db=1
    llama_expert_cache: verify-strict: refused=0  zero_mapped_selected=0

The first two lines are quoted verbatim from that log. The third is the log's own correctness
counter (note the double space — a single-space grep does not match it), and it is what actually
establishes the claim: `refused=0` means no cold expert was refused a real fill, and
`zero_mapped_selected=0` means the ZERO-slot pollution path (the mechanism that made results
pool-size dependent before 2026-09-11) never fired. An earlier revision of this document wrote
`nil=0  assert=0` here: substantively true (the log contains no `buffer is nil` and no
`GGML_ASSERT`), but it was a paraphrase presented as a quote, and grep for it returns nothing in
any log — corrected because this document's whole subject is unverifiable numbers.

### B. The phase flips with the decode width alone (same pool, same prompt)

12-token prompt, both arms, only cap differs:

| cap | decode width | phase for the 12-token step | slab served it |
|-----|--------------|------------------------------|----------------|
| 8   | 8            | 12 > 8 → PREFILL             | **yes** (`slab ntok=[12]`) |
| 64  | 17           | 12 ≤ 17 → DECODE             | no |

### C. Correctness gate with the new binary (vs the v6 reference)

    GATE phase_p2: PASS   M1(bit-identical)=9/9  M2(argmax)=9/9  M3(topk)=9/9  n=9

Read from the artifact rather than transcribed:
`Backup/m123_oracle_gate/summary_phase_p2.json` →
`cross_tab = {"num_eq_dec_eq": 9, "num_eq_dec_ne": 0, "num_ne_dec_eq": 0, "num_ne_dec_ne": 0}`,
`comparable = true`, `ok = true`. An earlier revision of this document hand-typed that dict and
got it wrong (it omitted `num_eq_dec_ne` and repeated `num_ne_dec_ne`); the verdict numbers were
right, the transcription was not. Quote the file, not the memory of the file.

Independent corroboration that the decode path is untouched: the accept harness produces
**58.25% / 180-of-309**, byte-identical to the pre-change build.

### D. Prefill and decode on this commit

Prefill, `scripts/check/prefill_certifiability.py --runs 5`, pp2048 @ `-b/-ub 5632`, profile
prefill250 (its own resolved command; 3 of 5 runs completed before the session was killed):

| run | free at launch | inactive | usable% | swap in use | pp2048 t/s |
|-----|----------------|----------|---------|-------------|------------|
| 1 | **3.40 GiB** | 3.83 | 52.1% | 2489 MiB | **246.86 ± 7.09** |
| 2 | 7.15 GiB | 2.37 | 61.8% | 4125 MiB | 134.83 ± 24.84 |
| 3 | 7.01 GiB | 2.63 | 61.8% | 4636 MiB | 110.70 ± 20.59 |

Decode, `scripts/check/mtp_accept_ab.py --arms nail,nail_nomtp`, same build, 8 GiB pool:

| carrier | accept | acc/gen | prefill t/s | decode t/s |
|---------|--------|---------|-------------|------------|
| nail (MTP on) | 58.25% | 180/309 | 15.61 | **8.83** |
| nail_nomtp | n/a | 0/0 | 14.83 | **6.38** |

**Shape label — read the ratio, not the absolutes.** These two numbers are tied to their harness:
`mtp_accept_ab.py`, prompts of 12–15 tokens, thermal HEAVY, swap 2.5–4.6 GiB. They are NOT
comparable with the 10.78 / 12.62 t/s family measured earlier on a quieter box, and the `prefill
t/s` column is per-request fixed overhead (a 15-token prompt at 2 t/s is ~7 s of setup), not a
prefill rate. What IS quotable is the **within-shape ratio: MTP-on / MTP-off = +38%** (same
harness, same session, same pool) — and the accept counters (58.25%, 180/309), which reproduce the
pre-change build exactly and therefore also corroborate the gate PASS.

## Two defects found (both fixed)

1. **The phase banner was unobservable.** It used `LLAMA_LOG_INFO`, and the production launcher
   runs at `verbosity = 3` (`common_params_print_info`), so every INFO line from this file is
   suppressed. Measured: the banner string was present in the lib (`strings` hit) and absent from
   all 900+ lines of the run's log — an invisible phase decision, in the milestone where the phase
   decision *is* the mechanism. Every other CGC diagnostic in this engine already uses
   `fprintf(stderr, ...)`. Converted; the banner, the clamp line and the "NOT capped" line now
   appear in every run's log.

2. **`cap=2` aborts the server, and the abort is nowhere near the phase code.** With the slab
   unarmed the clamp applies (`n_batch 2048 → 2`), which forces `n_ubatch = 2`, and the engine's
   own warmup decode then trips `GGML_ASSERT(n_ubatch > n_keep_tail)` in
   `llama-batch.cpp:609` via `llama_memory_hybrid::init_batch → split_equal`. Pre-existing clamp
   semantics (the old clamp used cap as well), now reachable and documented; cap is documented as
   clamped to `[2, 64]`, which reads as "2 is legal" and is not.

## Decision: the default profile does NOT arm the slab (option C), and says so out loud

`CGC_PREFILL_STREAM=1` and `CGC_GATHER_SLAB_CAP=256` are defaulted **inside the `prefill250`
profile block only** (`scripts/run_server.sh`). A default-profile launch therefore logs
`prefill slab NOT armed`, keeps the `n_batch` clamp, and serves every prefill through the pool
path — i.e. the phase split is currently a *gate/profile* behaviour, not the product default.

**Why not auto-arm, in the reviewer's words:** arming from runtime state (free memory, or a fit
computation) would put part of the configuration in memory state that the `.cap` fingerprint
does not record. D5's comparability is read off that fingerprint, so two runs with identical
`.cap` files could then have different memory layouts and the gate could not see it — the same
failure eng-gate-0006 names ("the profile cell nobody writes is the cell that drifts"), with the
drift source moved from a missing env var to unrecorded memory state. Three facts support staying
gated: (1) the slab is prefill-only and M3's criteria are decode, so arming it contributes
nothing to the M3 verdict; (2) its cost is coupled to chunk width while the memory margin is
already spent (prefill250's own constraint is an 8 GiB pool — 10 GiB dies on the first request);
(3) the slab's *benefit* is still unadjudicated (M2 exit conditions 2 and 3 lack an instrument;
`db=1` is structural evidence only). Auto-arming now would trade an unproven gain for a known,
already-exceeded cost.

**Cheap strengthening implemented (both review asks):**

1. One line of visibility, printed by the launcher (not the binary) so that no fingerprint moves
   — `CGC_SERVER_PROFILE` is a shell variable and is deliberately absent from `SERVER_ENV`:

       [arm]   slab OFF (profile=off): prefill chunks wider than the decode width take the POOL path \
               with the n_batch clamp on; arm with CGC_PREFILL_STREAM=1 and CGC_GATHER_SLAB_CAP>=n_expert
       [arm]   slab ON  (profile=prefill250, CGC_PREFILL_STREAM=1, CGC_GATHER_SLAB_CAP=256): ...

   Verified for three cases via `CGC_DUMP_ENV=1` (default → OFF, `prefill250` → ON, explicit
   `CGC_PREFILL_STREAM=0` → OFF), with `SERVER_ENV` unchanged in all three. "Why did this chunk
   take the pool path" is now answerable from the launch log itself.
2. If it ever becomes the default: that requires a **one-time re-baseline** (no existing profile's
   `.cap` contains `CGC_PREFILL_STREAM`), and the profile should then write `=0`/`=1` explicitly
   rather than expressing the state as absence.

### The measurement that would reopen it, specified (option D)

Three corrections to the obvious experiment, all structural, all testable before running it:

**(i) It must be a PREFILL measurement. On decode the two arms are identical by construction, so
a decode-side A/B can only return 0.000 — a shape that manufactures its own false negative.**
From this document's own predicate: `decode_width = min(floor(142/8), T_prefill-1) = min(17, 511)
= 17`, and decode is `T = 1` (MTP verify 2–4) ⇒ always `≤ 17` ⇒ always `PREFILL? no` — the DECODE
graph, which is the pool path. The whole-layer slab never participates in decode. §C's identical
accept counters (58.25%, 180/309) are the empirical half of the same statement. Measure at
**chunk > 17**.

**(ii) It is not a single-variable experiment.** For a chunk wider than the decode width the
unarmed arm *also* eats the `n_batch` clamp (the launcher's own `[arm] OFF` line says "with the
n_batch clamp on"), so the arms differ in two things at once:

| arm | what actually runs for a 2048-token prompt |
|-----|--------------------------------------------|
| armed | 1 × width-2048 PREFILL graph (whole-layer slab) |
| unarmed | ≈120 × width-17 pool-path graphs |

So the honest claim is "**slab + no clamp**" vs "**pool + clamp**". Third arm to separate them:
unarmed with `-ub 17`. If the per-token cost of unarmed(`-ub 17`) ≈ unarmed(`-ub 2048`), then the
entire gap is **graph-evaluation count** — i.e. a clamp penalty, not a slab acceleration.

**(iii) The landing point is the PROFILE cell (option D), not runtime memory.** If armed wins, the
conclusion is "not arming costs the clamp penalty", which is an argument for the default — but its
correct home is the profile table (every pool-regime profile writes `CGC_PREFILL_STREAM=1`), which
is fingerprintable. That is exactly the shape the objection to auto-arming does *not* cover, so
this measurement has real decision power even though it argues for a default.

**Protocol, since launch order dominates here:** interleave the arms, ≥3 reps, discard rep 1, take
paired medians, and record `free at launch` + `swap in use` for every launch (the sequence effect
in §D falsified the original free-memory hypothesis and made carried swap the live suspect).
**Null cell:** `-p 16` on both arms must come out bit-identical — both are on the decode side, so
any difference would mean the arms differ through something other than phase, and the instrument
is what is broken, not the slab.

**What would change this decision:** a same-build measurement that the armed slab is faster than
the pool path on *any* profile. That would remove fact (3) above — and the right response would
then be to write that evidence into M2's exit conditions 2/3 and pick a baseline, not to make the
default depend on runtime memory.

## What this does NOT establish

* Only the pool-path decode width was exercised across two caps. `T_prefill` remains non-binding
  and untested as a binding threshold.
* The prefill/decode numbers above are from a box that is concurrently used (`swap` 2.5–4.9 GiB,
  thermal HEAVY throughout); the pp2048 figures are a sequence effect, not a spec — see below.
* No 48-question run, no MTP acceptance change, no 4/6/10 GiB pool sweep on this build.

## Falsified on the way: the whitepaper's prefill hypothesis

`prefill_certifiability.py`'s stated hypothesis is that throughput tracks **free memory at launch
monotonically**, and that making free memory a precondition turns 250 into a spec. These three
launches say otherwise — the fastest run had the *least* free memory:

| run | free at launch | swap in use | pp2048 t/s |
|-----|----------------|-------------|------------|
| 1 | 3.40 GiB | 2489 MiB | 246.86 |
| 2 | 7.15 GiB | 4125 MiB | 134.83 |
| 3 | 7.01 GiB | 4636 MiB | 110.70 |

Free memory and throughput are **anti-monotone** here, while two other variables are monotone in
the right direction: swap in use (2489 → 4636 MiB) and run index. So the state that moves the
number is **carried swap from the previous launch**, i.e. a sequence effect, not a per-launch
draw. Two consequences:

* The instrument's design — N consecutive independent launches, take the spread — is itself
  biased: launch 1 is systematically the fastest because it starts from the least-swapped box.
  A "5-run spread" therefore conflates the mechanism with the ordering.
* Any A/B on this box must **interleave** arms (and ideally purge/reboot between them), which is
  what the canon/phase A/Bs in this document did.

## The rotated A/B: decode after a slab-served prefill (run 20260917_2001)

The protocol was rebuilt for one question — *is decode systematically colder after a slab-served
prefill?* — because the fixed-order grid could not answer it: arms ran slab → pool8 → pool17 every
rep, so launch position and arm were perfectly collinear, and this box's own §158 finding is that
the first launch in a sequence is systematically the fastest (carried swap). Now the order rotates
by rep and the report proves the balance from the same `arm_order()` the driver uses:

| rep | order |
|---|---|
| 1 (discarded) | `phase-slab` -> `phase-pool8` -> `phase-pool17` |
| 2 | `phase-pool8` -> `phase-pool17` -> `phase-slab` |
| 3 | `phase-pool17` -> `phase-slab` -> `phase-pool8` |
| 4 | `phase-slab` -> `phase-pool8` -> `phase-pool17` |

Mean launch position per arm over the counted reps: **2.0 / 2.0 / 2.0 — BALANCED.**

### Verdict: SYSTEMATIC COLD, confirmed

Decode is paired per rep (same rep = same launch sequence position for that arm), reported for two
instruments that measure decode *inside the same launch* as the prefill it follows: `long` (the
decode tokens of the 2233-token request itself) and `short_postlong` (steps 2..9 of the next
request). `first_step` is deliberately excluded from this claim: its wall contains a 12-token
prefill, which on the slab arm is the already-known short-chunk penalty, so it would measure the
prefill defect and not the pool's temperature.

| cell | vs | per-rep paired ratio (slab/base) | median | reps colder/warmer | verdict |
|---|---|---|---|---|---|
| `long` | `phase-pool8` | 0.288, 0.207, 0.254 | 0.254 | 3 / 0 | SYSTEMATIC COLD |
| `long` | `phase-pool17` | 0.275, 0.243, 0.176 | 0.243 | 3 / 0 | SYSTEMATIC COLD |
| `short_postlong` | `phase-pool8` | 0.287, 0.131, 0.542 | 0.287 | 3 / 0 | SYSTEMATIC COLD |
| `short_postlong` | `phase-pool17` | 0.178, 0.237, 0.226 | 0.226 | 3 / 0 | SYSTEMATIC COLD |

"Systematic" is held to the strict reading: **every** counted rep must agree in direction, not a
majority (3-of-3 here). At n=3 with one rep pointing 4x the other way, "2 of 3" is a split, not a
verdict — the stricter rule was forced by a selftest case, not chosen for comfort.

### The counters name the mechanism

| | slab arm | pool8 arm |
|---|---|---|
| pool high-water mark (`resident`) | 6,429.53 MiB | 6,430.62 MiB |
| pool requests | 8,835 (decode only) | 475,415 (prefill + decode) |
| hit rate | 86.3 % | 98.3 % |
| misses: compulsory / capacity | **1,210 / 0 (100 % / 0 %)** | 6,445 / 1,868 (77.5 % / 22.5 %) |
| file reads | 75,264 | 23,994 |
| cumulative pread | 3,612 – 4,977 s | 57 – 116 s |

The pool is **full in both arms at the same high-water mark**, so this is not a small pool — it is a
pool that the slab prefill never fills. Every one of the slab arm's 1,210 decode-side misses is a
*first touch* (0 capacity), i.e. exactly the signature of "the prefill did not publish". The
arithmetic closes: 1,210 cold reads at a few tens of ms each, spread over the pool's 8 worker
threads, is a few hundred ms per decode step — which is what the arm measures (1.64–3.72 t/s, i.e.
270–610 ms/step) against the pool arms' 5.7–8.7 t/s.

### The honest limit of this result

The grid cannot separate two components that point the same way:

1. **The pool is not published** by a slab-served prefill (the slab re-points `wt->data` and
   returns) — the 1,210/0 compulsory-only misses are direct evidence.
2. **The box is left under pressure by the slab regime's own I/O**: 75k reads and 3.6–5.0 ks of
   cumulative pread versus 24k reads and 57–116 s, and the slab arm ends at free = 8 % against the
   pool arms' 13–14 % (before-states are comparable, 61–76 %).

Both are cured by the same piece of work and neither is cured by tuning: an arm that arms the slab
**and** publishes the union into the pool would discriminate them, and that arm does not exist yet.
The prefill-side prize is also unaffected (4.43x on the long chunk, measured this run).

### Decode non-regression is now a formal gate, and it FAILS

`decode_gate()` compares the slab arm against `phase-pool8` (today's default: unarmed slab, clamp =
cap 8, so "does arming the slab cost decode?" is a question about that arm), per cell, at an
explicit ±10 % tolerance, and needs ≥2 usable launches per side before it will decide anything:

| cell | slab | pool8 | ratio | verdict |
|---|---|---|---|---|
| `short_warm` | 3.72 | 8.70 | 0.428 | **FAIL** |
| `long` | 1.94 | 7.68 | 0.253 | **FAIL** |
| `short_postlong` | 1.64 | 5.72 | 0.287 | **FAIL** |
| `first_step` | – | – | – | NOT MEASURED (degenerate, see below) |

So M1's "decode must not regress" exit criterion is now **measured and failing** for a default
arming of the slab. Together with the near-threshold prefill penalty (a 12-token chunk is ~28x
slower through the slab at matched graph count), this strengthens option C: not a default, and the
precondition for revisiting it is a prefill→pool handoff that removes the cold start, not a knob.

### Five instrument defects found by this run (all fixed, all silent)

1. **`first_step` decode was an artifact.** With `n_predict=1` the server attributes ~0 ms to the
   single predicted token (it is emitted with the prefill batch) and reports ~1e6 "t/s". The raw run
   wrote that into the cell, the gate compared 1e6 against 1e6, and it **PASSED** — a green cell that
   measured nothing. Values at or above `DECODE_ARTIFACT_TPS` are now refused with the reason, and
   the gate refuses them independently of the loader (defence in depth).
2. **Pool counters were collected per shape, not per launch** — 16 rows for a 4-launch arm, every
   counter 4x the truth. Ratios were unaffected, which is why it survived a glance at the verdicts
   and was caught only by the report's own table being 4x too long.
3. **`resident` was never parsed**, so the column that distinguishes "cold pool" from "small pool"
   was silently empty in every row.
4. **The launch-log → server-log regex never matched** (`\d+\d+` where the filename has
   `date_time`, i.e. `\d+_\d+`); the fallback to the stored JSON then made it look like a working
   parse that was merely missing a field.
5. **Aggregation trusted the `--reps` flag over the files on disk**, so a truncated grid could be
   reported as if the missing reps had been measured.

The harness's own `--selftest` now pins all five (99 checks), plus the rotation being a Latin
square, the cold-decision rule, and the refusal thresholds.

## Files

    src/llama.cpp/src/llama-cgc-phase.h           new: the single predicate + its two knobs
    src/llama.cpp/src/llama-graph.{h,cpp}          phase plumbed to the graph; both leaves gated by it
    src/llama.cpp/src/llama-context.{h,cpp}        decode width computed once, clamp + hook use it
    scripts/check/phase_split_selftest.cpp         new offline gate (11 assertions)
    scripts/run_server.sh                          CGC_PREFILL_THRESHOLD in the allowlist
    scripts/check/phase_split_ab.py                the rotated A/B: waits for a window, rotates arms,
                                                   attributes each timing to the graph that served it,
                                                   gates decode, and writes its own report (99 selftests)
    docs/PHASE_SPLIT_AB_REPORT_2026-09-17.md       generated by that script from the result JSONs
    Backup/phase_split_ab/20260917_2001/           the run this section quotes (12 launches, 4 reps)
