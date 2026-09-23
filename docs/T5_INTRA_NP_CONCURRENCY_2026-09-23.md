# T5 (intra-process concurrency): `-np 2` buys +66% — but not in the delivery regime

**Date** 2026-09-23 · carrier Nail IQ3_XXS-denseIQ4X · profile `prod25` · pool **4 GiB** ·
engine `llama-server 054fb22f04a0`, `libggml-metal 128b6048870f`, `libllama 971ade0f947d`
· products `Backup/t5_intra_np/intra_np.json` + `probe_<arm>.json`
· tool `scripts/check/concurrency_probe.py` (11/11 selftest) + driver `Backup/phase_decomp/t5_intra_np.py`

## 0. One line

**Two concurrent requests inside one 4 GiB server do not buy throughput when MTP is on — they cost
10% of it (pair/serial 0.90) — and they buy +66% when MTP is off (1.664).** The delivery regime is
MTP-on, so task 5's "total efficiency" premise is not rescued by this route; the same route is
demonstrably real one configuration away.

**Second result (§6), found by the same instrument and worth more than the first:** in the MTP-on
delivery regime **no request ever reuses the conversation prefix** — all nine requests re-prefilled
the full 204–208-token prompt at ~100 ms/token, 59% of the request wall. That tax is **one env var**
(`CGC_SERVER_NO_SEQ_RM_PROBE`): flipping it restores reuse (208 → 4 tokens) and takes the serial
request wall from 35.5 s to 15.0 s (**+133% aggregate**, decode untouched). It is not yet a
recommendation — that flag exists to protect bit-identity, and §6b names where the two can be
separated instead of traded off.

## 1. What was already dead, and what was left

Task 5 wanted `0.79 × 2 = 1.58`: two 4 GiB servers, each 21% slower, for 58% more total. That
premise is already refuted (2026-09-23, `docs/T5_PARALLEL_AND_4G_BASELINE_2026-09-23.md`): the
launcher refuses any second llama process (`cgc_memory_guard_req()` returns `req_other=0` for every
class) and one arm is oversubscribed by itself (model 13030 + pool 4096 MiB vs 16384 MiB physical),
so the available number is `0.79 × 1`.

The remaining way to buy the same thing is N concurrent requests in **one** process, which needs
`-np N` (the launcher's `CGC_SERVER_CONCURRENCY`, `run_server.sh:1199`). That cell had never been
measured — `decode_bench.py` sends one request at a time and has no concurrency support.

## 2. Design

Arms, A/B/A so that launch-ordinal drift cannot be mistaken for the effect, plus the one arm that
decides *where* the effect lives:

| arm | `-np` | extra env | why |
|---|---:|---|---|
| `np1-a` | 1 | — | current config. **The falsifier:** with one slot the second request must queue, so its pair aggregate must land back at the serial number. |
| `np2` | 2 | — | the question, in the delivery regime (MTP on). |
| `np1-b` | 1 | — | reproducibility of the control. |
| `np2-mtpoff` | 2 | `CGC_SERVER_MTP=0` | is the serialisation in the spec path, or below it? |
| `np1-noseqrm` | 1 | `CGC_SERVER_NO_SEQ_RM_PROBE=0` | §6b: is the 21 s prompt tax a knob or structural? |

Per arm, the probe runs 3 reps of **serial** (1 request) and **pair** (2 requests released through a
`threading.Barrier`, so thread-start jitter cannot masquerade as a shorter overlap window), in
alternating order, rep 0 dropped.

Two throughput definitions, both reported, and they are 2.5× apart on this shape:

* **per-request t/s** — the server's own `predicted_n / predicted_ms`. The only quantity comparable
  with a `decode_bench.py` baseline.
* **aggregate t/s** — total tokens ÷ wall of the whole batch. This is "total efficiency". On
  prod25/MTP-on a single request spends ~21 s of its ~35 s wall in prompt processing, so the same
  run reads 7.4 per-request and 3.0 aggregate. Every aggregate below is therefore printed with
  `non_gen_pct` (the share of the batch wall that is *not* generation).

## 3. Result

| arm | `-np` | serial agg | pair agg | **ratio** | per-req serial | per-req pair | non-gen (pair) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `np1-a` | 1 | 3.096 | 3.201 | 1.034 | 7.724 | 7.829 | n/a † | AT THE BOUNDARY |
| `np1-b` | 1 | 2.909 | 3.041 | 1.045 | 6.441 | 7.035 | n/a † | AT THE BOUNDARY |
| **`np2`** | 2 | 2.904 | 2.614 | **0.900** | 7.019 | **4.803** | n/a † | **NO GAIN (slots serialise)** |
| **`np2-mtpoff`** | 2 | 8.595 | 14.306 | **1.664** | 8.977 | 7.587 | **5.7%** | **PARALLEL BUYS THROUGHPUT** |

† These three arms were measured by the probe's first revision, which did not record the prompt
fields (§7.2), so the split is not in their product. From the server log it is **58.9%** for
`np1-a` (task 0: prompt 20.896 s of 35.502 s total). Per-request and aggregate columns come from
the same products and are unaffected.

Read three ways:

1. **The control is honest.** `-np 1` gives 1.034 and 1.045, twice — `-np` alone does not move the
   single-stream path, and the probe does not manufacture a win where there is none. (Their pool
   counters are also **byte-identical**: `misses 153285`, `evictions 153101`, `compulsory 7894`,
   `capacity 145391`. Two independent launches, same numbers.) The two controls' *absolute* levels
   do drift — serial aggregate 3.096 vs 2.909 (6.6%), per-request 7.724 vs 6.441 (20%, `np1-b` ran
   into HEAVY) — which is exactly why the contrast is taken inside one arm.
2. **MTP on: concurrency is negative.** Pair aggregate 0.90× of serial, and the individual
   requests' own decode rates collapse from 6.84–7.39 t/s (solo, same server) to 1.65–5.23 t/s.
3. **MTP off: concurrency works exactly as advertised.** 8.595 → 14.306 aggregate: each of the two
   streams keeps 85% of its solo rate and the total is +66%. The pair requests are symmetric to the
   millisecond (rep 1: `14.658 s / 14.659 s` eval, both 7.37 t/s; rep 2: `13.838 s / 13.838 s`,
   both 7.80 t/s).

Every arm started NOMINAL (`np2-mtpoff` stayed NOMINAL throughout). The pair/serial contrast is
inside a single arm, so launch drift and thermal cancel on the ratio; absolute levels do not.

## 4. The mechanism, from the engine's own per-request timings

The probe only sees walls. The server log prints each request's own prompt/eval split, which is what
turns "0.90×" into a mechanism. **MTP on, the three pairs** (each pair = two requests released
together):

| pair | slot 0 prompt | slot 0 eval | slot 1 prompt | slot 1 eval |
|---|---|---|---:|---|
| A (cold pool) | 19.6 s @10.6 | **65.3 s @1.65** | **65.9 s @3.16** | 17.8 s @6.07 |
| B | 21.9 s @9.5 | **45.8 s @2.36** | **46.2 s @4.51** | 13.3 s @8.11 |
| C | 20.6 s @10.1 | **45.9 s @2.35** | **46.2 s @4.50** | 16.9 s @6.39 |
| *solo (serial arm, same server)* | *20.9–22.2 s @9.4–10.1* | *14.6–15.8 s @6.84–7.39* | | |

The slow slot's eval time equals the other slot's prompt duration to within 0.6–0.9%, in all three
pairs. Reconstructed timeline for pair C:

```
[0.0 .. 20.6]  slot0 prompt alone, at solo rate
[20.6 .. 66.8] slot0 decode ∥ slot1 prompt     <- one pass, two sequences
                  slot0 decode 2.35 t/s (solo 7.4)   slot1 prompt 4.50 t/s (solo ~10)
                  their sum 6.85 = the solo decode rate 6.84-7.20
[66.8 .. 83.7] slot1 decode alone, 6.39 t/s
```

So: **the two prompts never batch with each other, the two decodes never overlap at all, and the
only overlap that exists (one slot's prefill against the other's decode) redistributes the same
throughput rather than adding to it** — the sum of the two co-running streams lands back on the
single-stream number. That is also why the pair is *slower* than two serial requests: 2 × serial
wall is 71–75 s, the measured pair walls are 81.5 / 83.9 / 103.4 s.

The pool sides with this: `np2` records `misses 167634 / capacity 159732` against the controls'
`153285 / 145391` — +9% misses for the identical token workload, i.e. two sequences' expert sets
share (and thrash) one 4 GiB residency. The miss counts are per arm, not per token; the direction is
what is quotable.

**MTP off, the same two requests** (rep 1 of `np2-mtpoff`): both slots finish together at
`15.44 s / 15.42 s` with `eval 14.66 s @ 7.37` each. No asymmetry, no interleaving: a normal
two-sequence batch, which is what `-np 2` is for.

## 5. So can task 5 be rescued?

**Not by this route, in the regime the line ships.** The observation is specific: `-np`, the
launcher knob, the probe and the hardware are all fine — the configuration that cannot batch is
MTP-on. Since `prod25` (MTP on) is the decode delivery regime, "run two things at once for free" is
not available there, and the +66% belongs to MTP-off work.

One caveat that must travel with that sentence: `run_server.sh`'s MTP block is all-or-nothing — with
`CGC_SERVER_MTP=0` a whole env set disappears (`CGC_NO_PREFETCH`, `CGC_VERIFY_DECODE`,
`CGC_DRAFT_DECODE`, `CGC_WARM_NPAST`, `CGC_MTP_NO_WARMUP`, `CGC_MM_BITIDENT=1`,
`CGC_NO_SEQ_RM_PROBE`, and the `LAYER_CAPS 40-40:256` override). So this experiment separates
**"MTP-on configuration" from "MTP-off configuration"**, not "the MTP head" from "everything else".
Both arms' own serial-vs-pair contrasts are internally valid; attributing the difference to the
draft/verify chain specifically needs the MTP-on arm restarted with that block's members applied
individually.

## 6. Byproduct, and it is not small: prompt processing under MTP

The probe's per-request rows make a second thing visible for the first time, and it is the reason
the MTP-on aggregate looks so low (58.9% non-generation on `np1-a` vs **5.7%** on `np2-mtpoff`):

| regime | requests 1–2 prompt (208 tok) | request 3+ prompt | request wall |
|---|---:|---:|---:|
| MTP **on** (prod25) | 20.9 s @10.0 (req 1), 19.6 s @10.6 (req 2) | **all 9 of 9 requests re-prefilled 204–208 tokens: 20.6–22.2 s @9.4–10.1 for the request that had the pass to itself, 46.2–65.9 s @3.2–4.5 for its partner when the two overlapped (§4)** | 35–37 s |
| MTP **off** | 17.5 s @11.9 (req 1), 18.6 s @11.2 (req 2) | **0.33–0.79 s for 2–4 tokens × 5** — served from the KV cache | 11.6–13.6 s |

So: with MTP off the third and later requests reuse the conversation prefix (the server's
`prompt_save` / `prompt_load` path, `server-task.cpp:1865+`); with MTP on **no request reuses it**,
all nine prefilling the full prompt at ~100 ms/token, i.e. **the prompt costs about as much as a
decode step**. For a target of "prefill 250+ t/s" that is the largest single number in this table,
and it is paid per request rather than once per session.

Two corrections to the first version of this section:

* **The two arms load the same weights.** The MTP-off arm logs a different model *path*
  (`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`), which is a **symlink to the same file** —
  `os.path.realpath` and size (13.663 GB) are identical to the MTP-on arm's. The log's
  `unused tensor blk.40.*` warnings are the MTP head sitting in the file, unused because no spec
  context was created. Verified rather than assumed, because a different GGUF would have made this
  section unquotable. What differs between the arms is `--spec-type draft-mtp` plus the MTP env
  block, and nothing else.
* **`CGC_WARM_NPAST` is not a prefix-reuse knob** (it was listed as one here). The engine reads it
  at `llama-context.cpp:6309-6317` as the fast-path warm-up gate: `cgc_warm_gate = n_past <
  CGC_WARM_NPAST`, and while that gate is true the ZERO-slot fast path is disabled in favour of the
  exact `ensure_batch` path. Its two values are visible in the logs as `CGC-WARM verify n_past=…
  warm=…`: **`warm=0` with MTP on** (prod25 sets `DENSE_IQ4X=1`, and `run_server.sh:2165-2171`
  turns that into `CGC_WARM_NPAST=0`) and **`warm=2048` with MTP off** (the variable is never
  exported outside the MTP block, so the engine default 2048 applies). It gates *decode*, not
  reuse.

The one arm-specific difference that *does* fingerprint in the logs is `CGC_NO_SEQ_RM_PROBE=1`,
which `run_server.sh:2232` sets inside the MTP block: the MTP-on log contains
`skipping seq_rm probe (CLI parity / explicit override)` and the MTP-off log does not.
`server-context.cpp:1390-1405` shows it forcing `ctx_tgt_seq_rm_type =
COMMON_CONTEXT_SEQ_RM_TYPE_PART` instead of asking `common_context_can_seq_rm()` — i.e. it overrides
how a sequence's KV may be removed, which is the same machinery the server consults before reusing
a prefix. No `failed to restore` / `prompt cache update` line appears in either log, so the reuse is
not failing loudly; it is either not matching or not attempted.

### 6b. The separating experiment was run: **it is a knob** — and the knob was set for bit-identity

`np1-noseqrm` = the control arm (`-np 1`) with exactly one env flipped: `CGC_SERVER_NO_SEQ_RM_PROBE=0`.

| | control `np1-a` | `np1-noseqrm` |
|---|---:|---:|
| prompt tokens evaluated, request 1 / 2 / 3 | 208 / 204 / 206 | **208 / 4 / 4** |
| serial aggregate t/s | 3.096 | **7.198 (+133%)** |
| serial request wall | 35.5 s | **15.0 s (2.4×)** |
| non-generation share (requests 2–3) | 59% | **3.8% / 4.5%** |
| per-request decode t/s | 7.72 | 7.51 (unchanged) |
| `skipping seq_rm probe` in the log | 1 | **0** |
| draft acceptance | 0.37908 (58/153, len 2.14) | **0.37908 (58/153, len 2.14)** |

Reach is checked, not assumed: the engine's own `skipping seq_rm probe (CLI parity / explicit
override)` line disappears, `--spec-type draft-mtp` is still on the command line, `warm=0` is
unchanged, and speculation **ran** (the acceptance line above is bit-identical to the control's). So
the entire 2.4× is prompt processing, and decode is untouched.

**But the flag was added deliberately, and un-setting it re-opens what it closed.** The two sides:

* `server-context.cpp:1390-1405`. With the probe skipped (the prod25 default) the server *hard-codes*
  `ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_PART`. With the probe running, the type comes
  from `common_context_can_seq_rm(ctx_tgt)` — and PART is evidently the reuse-hostile setting, which
  is how one startup probe ends up deciding whether every later request re-prefills.
* The reason it was hard-coded (`speculative-simple.cpp:184-195`, "CGC bit-bisect v5"): the probe
  **decodes 2 tokens `[0,0]` through the full trunk** to test compatibility — that fills expert-cache
  slots and bumps LRU ticks, and the `llama_memory_clear` inside it does **not** reset the expert
  cache, leaving the pool state divergent before prefill chunk 0 (recorded there as *chunk0 L4 rows
  0-2 divergence*).

So the honest reading is not "turn it off". It is: **the prefill tax and the bit-identity fix are the
same knob**, therefore the fix belongs where the two can be separated — run the probe before the pool
is armed, or reset the pool/LRU after it — rather than choosing between speed and identity. Until
that is done, `np1-noseqrm`'s +133% is a measurement, not a recommendation: it has not been through
M1/M2/M3.

**What reuse does to the output, and to the hazard's own indicators** (the strongest checks available
from these products, none of which is M1/M2/M3):

* **Every request in every arm produced the same answer**: md5 `3958f37c`, 108 tokens — **all 27
  requests** across `np1-a` (0 of 9 reusing a prefix), `np1-noseqrm` (8 of 9) and `np2-mtpoff` (7 of
  9). A restored prefix therefore did not change this prompt's result. prod25 is greedy, so this is a
  meaningful comparison and not a sampling coincidence.
* **The hazard's own counters are clean in the reusing arm**: `MTP fast path … cold(ZERO)=0 (0.0%)`
  (identical to the control's 0.0%), and the teardown check prints `pool integrity: owner-set
  slots=3055 zero-regions=0 zero-prefix-only=0 [OK: every resident slot holds non-zero bytes]` —
  also identical to the control. The ZERO-slot contamination the v5 note is about leaves no trace
  here.
* **What that does *not* prove**: this is text md5, not logits (M1) and not argmax over the gate's
  nine probes (M2/M3); it is one short single-turn prompt, whereas prefix reuse matters most across
  turns; and "the output did not change here" is not "the pool-state divergence is gone".

**One open question decides how cheap the fix is.** *Why* does a compatibility probe enable prefix
reuse? Two readings: (a) `ctx_tgt_seq_rm_type` (hard-coded PART vs probed) is consulted by the
prefix-reuse path, in which case the fix is to make PART reuse-friendly; (b) the probe's side effect
— it decodes 2 real tokens through the full trunk — is what leaves the context in a state that
`prompt_save`/`prompt_load` can serialize, in which case the fix is a non-polluting one-time warm
state instead of the compatibility check. They imply different work, and one experiment separates
them: raise the server's log level so the `prompt cache` / `updating prompt cache` / `failed to
restore` trace lines surface, and see which arm even enters the save/load path.

One more consequence worth stating: the pool counters are not comparable across these two arms at
all (`hit 64.0% → 54.0%`, `misses 153285 → 62975`, `evictions 153101 → 62791`) — consistent with 8 of
9 full prefills disappearing, and enough on its own to make any cache-side conclusion quoted across
this flag meaningless.

### 6c. The open question is answered, the fix is in, and it works by default (one binary, A/B)

The trace run named the failure rather than leaving it as a hypothesis. `--log-verbosity 4` on the
control arm shows, once per request:

```
slot   operator(): id 0 | task 79 | forcing full prompt re-processing due to lack of cache data
    (likely due to SWA or hybrid/recurrent memory, see .../pull/13194#issuecomment-2868343055)
```

and the cache entry it is about carries **`checkpoints: 0`** (`317 tokens, checkpoints: 0, 66.734 MiB`).
The chain is four lines of existing code, no probing needed:

| step | code | what it does here |
|---|---|---|
| 1 | `server-context.cpp:1390` | probe skipped (prod25 default) ⇒ `ctx_tgt_seq_rm_type = PART` — **hard-coded** |
| 2 | `server-context.cpp:3651` | `do_checkpoint = … && (type == FULL \|\| type == RS \|\| n_swa > 0)` ⇒ false for PART, and `n_swa = 0` / `is_swa_any = 0` (model print_info) ⇒ **0 checkpoints** |
| 3 | `server-context.cpp:3402` | reuse computes `n_past = lcp(slot.prompt, input) = 208` correctly |
| 4 | `server-context.cpp:3566` | `pos_min >= pos_min_thold` ⇒ checkpoint search ⇒ none ⇒ `do_reset` ⇒ `pos_next = 0; n_past = 0` ⇒ **full re-prefill** |

So `PART` is not a label, it is a **claim about the memory** — "seq_rm can put this trunk back at an
arbitrary prefix" — and a recurrent (hybrid GDN) trunk cannot honour it. Reading (b) in 6b is dead:
reuse has nothing to do with the probe's side effect on serialization. The claim is simply false, and
it costs one full prefill per request.

**The fix** (`server-context.cpp`, the probe-skip branch): answer the question from metadata instead.
`common_context_can_seq_rm` decides `RS` with the single query `llama_n_rs_seq(ctx) > 0`
(`common.cpp:1526`), which needs no decode — so the branch can stop lying *without* running the
2-token compatibility decode the flag exists to avoid (the decode that pollutes the pool before
prefill chunk 0, `speculative-simple.cpp:184-195`). Metadata where metadata answers; the decode probe
is still used when it does not (`n_rs_seq == 0`), so the dense/attention path is untouched. An explicit
`CGC_SEQ_RM_TYPE=PART|RS|FULL|NO` override was added at the same time — the only way to compare the
three answers on one binary, and the guard against a wrong *value*, which is what this bug was.

**A/B: one binary (`engine 054fb22f04a0`), profile prod25, same probe, one variable.**

| | `np1-pfx-part` (`CGC_SEQ_RM_TYPE=PART`, the old behaviour) | `np1-pfx-meta` (default) |
|---|---:|---:|
| prompt tokens evaluated, request 1 → 9 | 208 / **208 / 208 / 208 / 208 / 208 / 208 / 208 / 208** | 208 / **4 / 4 / 4 / 4 / 4 / 4 / 4 / 4** |
| request wall | 19157–20083 ms (prompt) + 15006 ms (decode) | prompt **489–634 ms**, decode 14471 ms |
| serial aggregate t/s | (control arm, 6b) 3.096 | **7.214** |
| `forcing full prompt re-processing` | 9 lines | **0 lines** |
| context checkpoints | `checkpoints: 0` | **`created context checkpoint 1 of 32 … n_tokens = 182, 63.172 MiB`**, 3 kept per entry (256.329 MiB) |
| reach line | `CGC_SEQ_RM_TYPE=PART forces seq_rm_type on trunk and draft` | `answered from metadata: trunk RS (recurrent trunk)` |

Reach is checked on both sides, not assumed: the forced arm reproduces the defect **on the same
binary**, and the metadata arm's log contains no re-processing line at all. `+133%` on the aggregate
and `2.4×` on the request wall are therefore attributable to this one variable.

**Correction to 6b's table.** Its acceptance row reads `0.37908 (58/153, len 2.14)` for `np1-noseqrm`
"bit-identical to the control's", and that is true of **request 1 only**. The full log of that arm
(`llama_server_20260923_100339.log`) contains **both** values, and the second is the reused-request
value:

| arm | acceptance on the full-prefill requests | on the reused requests |
|---|---|---|
| `np1-pfx-part` (reuse off) | 0.37908 (58/153), len 2.14 | — (all 9 are full-prefill) |
| `np1-pfx-meta` (reuse on) | 0.37908 (58/153), len 2.14 | **0.36538 (57/156), len 2.10** |
| `np1-noseqrm` (reuse on, 6b) | 0.37908 (58/153), len 2.14 | **0.36538 (57/156), len 2.10** (same pair) |

So the acceptance shift is a property of **prefix reuse**, not of the `RS` type choice — both routes to
reuse show the identical pair — and `RS` is exactly what the engine's own probe answers for this model
(`checkpoints-line=0` in every probe-run log, i.e. it never returned `FULL`). What changes on a reused
request is the boundary: the restored prefix means the last prompt token is evaluated in a 4-token
batch rather than as the tail of a 208-token chunk, which moves the logits at the boundary enough to
change one draft token out of 156, and not enough to change the answer.

**What this does and does not establish.**

* **Established**: reuse is reachable in the delivery regime (prod25, MTP on, k=3, 4 GiB pool) by
default, with no decode at load time; the prompt tax falls from ~19 s to ~0.5 s per request; every one
of the 27 requests in three arms still answers md5 `3958f37c` / 108 tokens (greedy).
* **Not established — and it is the gate before this becomes a default**: M1/M2/M3 at the logits level
in an MTP-on regime. The existing gate **cannot** cover it: `m123_oracle_gate.py:487-489` dumps its
reference with `CGC_SERVER_MTP=0`, and with MTP off the whole MTP env block does not run, so
`CGC_NO_SEQ_RM_PROBE` is not exported, `cgc_no_seq_rm_probe` is false, and the changed branch is **not
executed at all**. A green gate there says nothing about this change — it is dead code in that
configuration. The test that would cover it is a paired logits dump in the MTP-on regime
(`--env CGC_SERVER_MTP=1`, `--write-ref` with `CGC_SEQ_RM_TYPE=PART`, then compare the default run
against it), and the acceptance column above predicts the answer is *not* "identical": reuse moves
the boundary. Whether that movement is bit-identical **by M1's definition** (row_fnv1a64 on logits) is
the open question.
* **Cost**: 3 checkpoints × 63.2 MiB kept per cached prompt entry (256.3 MiB vs 66.7 MiB for the state
alone), inside the server's 8192 MiB prompt-cache limit. On a 16 GiB box that is real and must be
counted against the pool before this is enabled everywhere.

### 6d. The logits gate was run in the MTP-on regime: **M1 is not invariant** — so the fix stays opt-in

6c left one question open and named the only test that could answer it. It has now been run, and the
answer is negative. Protocol: the gate itself, `--profile prefill250` (so the pinned
`CGC_SERVER_BATCH/UBATCH=6144` apply to both arms), `CGC_SERVER_MTP=1` on both, one env difference.

| run | config | compared against | verdict |
|---|---|---|---|
| `pfxref-part` | MTP=1, `CGC_SEQ_RM_TYPE=PART` (the pre-fix claim) | v6 reference | **M1 9/9 M2 9/9 M3 9/9** — and its dump md5 is `72d82a33ad79e0e69bc935acd24228f2`, **byte-identical to the pre-registered v6 reference** (dumped days earlier, MTP=0) |
| `pfx-meta` | MTP=1, default (metadata ⇒ RS) | `pfxref-part`'s dump | **M1 FAIL (1/9)**, M2 PASS (9/9), M3 6/9 |
| `pfx-meta2` | MTP=1, default, independent run | `pfx-meta`'s dump | **M1 9/9 M2 9/9 M3 9/9** ⇒ the RS configuration is deterministic |

`pfx-meta2` is the control that makes the middle row attributable: the same configuration reproduces
its own dump bit-for-bit, so the 8 differing rows are the variable and not run-to-run noise. The gate
labels the PART-vs-metadata comparison `INVALID COMPARISON` because the two configs differ — which is
the experiment, not a defect in it: the question being asked is exactly "is this change M1-neutral?",
and within the gate's own definition of M1 (`row_fnv1a64` over the dumped logits) the answer is no.

Where it moves (`RS` vs `PART`, keyed on `(step, token_idx, ctx_type)`, all 9 rows present in both):

| step | ctx_type | n_tokens | RS sum | PART sum | delta | rel | row identical | argmax | pmax |
|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | DEF | 1 | -489276.67 | -489276.67 | 0.00 | 0.000% | **yes** | yes | yes |
| 1 | DEF | 1 | -289804.02 | -283952.87 | -5851.14 | -2.06% | no | yes | yes |
| 2 | MTP | 1 | -462869.73 | -460189.56 | -2680.17 | -0.58% | no | yes | yes |
| 3 | MTP | 1 | -459147.80 | -459731.44 | +583.64 | +0.13% | no | yes | yes |
| 4 | MTP | 1 | -678664.85 | -678009.35 | -655.51 | -0.10% | no | yes | yes |
| 5 | DEF | 4 | -686997.34 | -679660.79 | -7336.55 | -1.08% | no | yes | yes |
| 5 | DEF | 4 | -566610.42 | -544559.14 | -22051.27 | -4.05% | no | yes | yes |
| 5 | DEF | 4 | -591384.56 | -590541.32 | -843.24 | -0.14% | no | yes | yes |
| 5 | DEF | 4 | -506154.47 | -491092.85 | -15061.63 | -3.07% | no | yes | yes |

The shape is the mechanism's signature: **step 0 is bit-identical** (nothing has been done differently
yet) and **every step after it drifts**, in all three context types. `argmax` and `pmax` are identical in
all nine rows while the row sum moves 0.1–4% — the same near-tie/rounding signature the acceptance
shift in 6c predicted, at the logits level.

**What this means for the fix.** The +133% prompt-processing result in 6c stands, and so does the
mechanism. What does *not* stand is the implicit claim that the change is free: making the trunk's
claim truthful also changes which route the speculative rollback takes (`server-context.cpp:3253/4183`:
`FULL || (RS && draft.size() > n_rs_seq)` — true for RS with k=3, false for PART), and that route
change is visible in the logits. So:

* **`CGC_SEQ_RM_TYPE` must stay opt-in**; the metadata answer is correct as a *description* but is not
M1-neutral as a *default*. A default that quietly costs M1 is the same class of defect 6b was about.
* The reuse need and the rollback need are separable, and that is where a safe fix lives. Prefix reuse
only requires `do_checkpoint` (`:3651`) to be true so that some checkpoint exists to restore; the
rollback route is a *different* predicate (`:3180/3253/4183`). A fix that creates checkpoints **without
changing `ctx_tgt_seq_rm_type`** — i.e. add the "this memory cannot reposition itself" clause at
`:3651` and leave the PART claim alone — would give reuse while keeping the rollback route the
reference was dumped under. That is the next experiment, and it is now falsifiable in one gate run
per arm, exactly as this one was.
* **Boundary**: this dump is a *single* request, so it isolates the type claim's effect on the spec
path; it does **not** cover reuse (which needs a second request). The reuse path's own logits effect
is still unmeasured, and the acceptance evidence in 6c says it is real.

### 6e. The separation is real and it works for reuse — but it does **not** buy M1, and the bisect says why

6d left a concrete plan: serve the two needs separately, because reuse only needs a checkpoint to
exist (`:3699`) while the speculative rollback route is chosen elsewhere (`:3253/:4183`). That is now
implemented and measured. Code shape:

* `ctx_tgt_seq_rm_type` goes back to the **CLI-parity value** (`PART`), i.e. exactly what the reference
dump was taken under. The metadata answer from 6c is gone (it is still reachable deliberately through
`CGC_SEQ_RM_TYPE`).
* Checkpoint creation gets its own switch: member `cgc_prefix_reuse_ckpt` (set in `load_model`), one new
clause at the `do_checkpoint` gate, and `CGC_PREFIX_REUSE_CKPT` / `CGC_SERVER_PREFIX_REUSE_CKPT` wired
through the launcher's allowlist.

**Reuse is recovered** (`np1-ckpt`, prod25, MTP on k=3, 4 GiB pool, verbosity 4):

| arm | serial prompt tokens per rep | pair | `forcing full prompt re-processing` | per-request t/s |
|---|---|---|:--:|---:|
| `np1-trace` (pre-fix) | 208 / 208 | 208/208 … | 9 | 7.623 |
| `np1-pfx-part` (`CGC_SEQ_RM_TYPE=PART`) | 208 / 208 / 208 | 208/208 … | 9 | 5.61 |
| `np1-pfx-meta` (6c: metadata ⇒ RS) | 208 / **4** / **4** | 4/4 4/4 4/4 | **0** | 7.534 |
| **`np1-ckpt`** (`CGC_PREFIX_REUSE_CKPT=1`) | 208 / **4** / **4** | 4/4 4/4 4/4 | **0** | **8.198** |

Reach is witnessed, not assumed: `prefix reuse enabled by context checkpoints` ×1, `created context
checkpoint` ×11, request 1 pays 20234 ms for 208 tokens and every later request ~0.5 s for 4. The
and-rolled-baseline comparison is a 4 GiB prod25 baseline of 8.30 t/s, and this arm's per-request
median is 8.198 — but that is a timing read under a recorded busy-box override
(`reclaimable=6165MB<6500`, swap 5.1 GiB), so the claim rests on the counter (208 → 4, a property of
the code path) and the t/s only corroborates it.

**M1 is still not met, and the bisect localises it.** Same gate protocol as 6d, one env difference:

| run | config vs `pfxref-part` (PART, MTP=1) | dump md5 | M1 |
|---|---|---|---|
| `pfx-ckpt` | PART **+** `CGC_PREFIX_REUSE_CKPT=1` | `467554b0f086712f79f1fbcbf12a8f94` | **FAIL (1/9)**, M2 9/9, M3 6/9 |
| (`pfx-meta`, 6d, for comparison) | RS instead of PART | `467554b0f086712f79f1fbcbf12a8f94` | FAIL (1/9) |
| reference | PART, no clause | `72d82a33ad79e0e69bc935acd24228f2` = **the v6 pre-registered reference, byte-for-byte** | 9/9 |

The two failing dumps are **byte-identical**, which is the useful part: the rollback route is
irrelevant (it differs between those two runs and the dump does not). The single carrier is **checkpoint
creation itself**, and the reason is in the same function that gates it:

```cpp
// server-context.cpp:3786-3800 -- "process the last few tokens of the prompt separately in order to
// allow for a checkpoint to be created" (upstream PR #20288)
if (do_checkpoint) {
    static const int checkpoint_offsets[] = {4 + n_ubatch, 4};
    ...  if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) { should_break = true; }
```

With `do_checkpoint` true, the prefill's **tail is a different batch**: without it the 208-token prompt
is 26 chunks of 8; with it the last chunk is 4. That is a reduction-order change, and it is exactly the
signature 6d measured (step 0 identical, every later step drifts, `argmax`/`pmax` unchanged, the row sum
moving 0.1–4%). It is also consistent with the acceptance shift in 6c (0.37908 → 0.36538): the tail
tokens are evaluated in a different batch composition.

**So "reuse and M1 together" is not reachable while the split depends on the feature.** The split has to
become a property of the *engine configuration* instead, and there are two ways:

1. **Make the tail split unconditional** (always break at the same offsets, checkpoints or not). Then
reuse on/off differ in nothing but reuse, and the M1 reference has to be re-baselined **once**, as a
declared act (`--write-ref` + update `REF_PINS`), which is the re-baseline path the gate already
supports. Cost: one extra small chunk per prompt on every run, including non-reuse ones.
2. **Checkpoint only at boundaries the chunker already produces** (multiples of `n_batch`) instead of
breaking to manufacture one. Then the split is untouched and the reference stands; the checkpoint simply
sits at the last natural chunk start (e.g. 200 instead of 204 for this prompt), so reuse restores a few
tokens less per request.

Both are one-clause changes at the same site, and both are falsifiable the same way this one was: one
gate run per arm against a dump whose provenance is pinned. What is **not** yet measured: the restore
path's own logits effect (the 6d/6e dumps are single-request, so they exercise creation, not restore).

### 6f. Route 2 works: reuse **and** M1, by taking the checkpoint at a boundary the chunker already makes

6e proposed two routes and 6e's own bisect made route 2 the cheap one (no re-baseline, every existing
experiment stays valid). It is implemented and verified. The mechanism was not one break but **two**,
and both had to be neutralised: upstream's block at `:3786-3800` (`checkpoint_offsets = {4 + n_ubatch, 4}`)
*and* the user-message-start break above it. Both manufacture a chunk boundary, so with checkpoints on
the prefill's split changes -- hence the reduction order -- and that was the whole of the M1 failure.

```cpp
const bool do_checkpoint_split = do_checkpoint && !cgc_prefix_reuse_ckpt;   // the two breaks
// creation (`create_checkpoint`) still keys off do_checkpoint, untouched
```

With `CGC_PREFIX_REUSE_CKPT=1`, the checkpoint is taken at the last chunk start the chunker produces
anyway, and the split is bit-for-bit what it is with the feature off.

| | result |
|---|---|
| **M1** (MTP=1, PART, CKPT vs the pinned reference) | **PASS 9/9**; M2 9/9; M3 9/9 |
| dump md5 | `72d82a33ad79e0e69bc935acd24228f2` — **byte-identical to the reference, and to the pre-registered v6 reference** |
| reuse | `serial rep0:208 rep1:8 rep2:8`, `pair 8/8 8/8 8/8` |
| `forcing full prompt re-processing` | **0 lines** (pre-fix: 9) |
| checkpoint actually used | `created context checkpoint 1 of 32 (pos_min = 199, pos_max = 199, n_tokens = 200, 63.208 MiB)` |
| per-request t/s | 7.968 (busy-box override recorded: `reclaimable=6165MB<6500`); pre-fix `np1-pfx-part` 5.61 |

**Both halves of the user's prediction hold, and one came out better.**

* The predicted cost is exact: reuse now restores to `n_past = 200` instead of 204, so 8 tokens are
re-evaluated per request instead of 4 -- 8 against the ~204 that re-prefilling costs, i.e. ~3.9% of the
waste survives and ~96% of it is gone.
* What was not predicted: the **memory cost halves as well**. With natural boundaries there is one
checkpoint per prompt entry instead of three -- `checkpoints: 1, 129.942 MiB` against
`checkpoints: 3, 256.329 MiB` -- because the two manufactured boundaries were also the two extra
snapshots. On a 16 GiB box that matters more than the 4 tokens.

**Instrument defect found while doing this** (same class as the round's other entries, and it was hiding
my own A/B): `m123_oracle_gate.engine_digest()` did not hash `libllama-server-impl.dylib`, which is where
the server's logic actually lives -- `llama-server` is a ~50 KB launcher whose md5 does **not** move when
`server-context.cpp` changes. Every one of 6c/6e's four arms therefore recorded the identical
`llama-server=054fb22f04a01c5c` while running materially different server code, so the summaries could not
have told them apart after the fact. Fixed (added to the tuple); the new build line reads
`… libllama-server-impl.dylib=60fb7910a8bb7d38 llama-server=054fb22f04a01c5c`. `decode_window_harness.digest()`
already had it.

**Still open**: the *restore* path's own logits effect. Every dump here is a single request, so it
exercises creation, not restore; reuse on/off will only be fully M1-neutral by construction once a
multi-request dump compares a restored prefix against a recomputed one. And whether `prod25`'s profile
should turn the knob on by default is now a supported decision rather than a hopeful one -- the evidence
(M1 9/9, byte-identical dump, ~96% of the prompt tax recovered, half the cached-state memory) points one
way, but a profile default is a deliberate act, not a byproduct of a fix.

## 6g. The restore path's own logits — the gap §6e/§6f left open

§6e/§6f ended on a stated boundary: **every dump in this project is two of them away from the thing
prefix reuse actually does.** `m123_oracle_gate.py` sends ONE request, so it can only see checkpoint
*creation*; the restore path — the 208→8 prompt collapse — had never been dumped. This round dumps it.

**Two new instruments** (neither existed before):

| what | where | why it has to be its own thing |
|---|---|---|
| `pfx_restore_probe.py` | `Backup/phase_decomp/` | one server, N identical **serial** requests (the concurrency probe's pair phase would put two sequences in one dump), dump on; then splits the dump into per-request files and **asserts** the segment count equals the requests sent |
| `--align pos` | `scripts/check/cgc_logits_oracle_compare.py` | the global `step` counter cannot align two dumps whose evaluation sequences differ. `pos` keys each row on its own identity `(ctx_type, n_tokens, pmax, token_idx)` and compares a key's records as a **multiset** (measured: ~50 MTP keys per request hold *two* records with different hashes, so "one key, one value" would have been a silent lie) |

**Setup, and the single-variable claim is read off the products rather than asserted:** prod25 shape,
4 GiB pool, MTP on k=3, `CGC_SERVER_SEQ_RM_TYPE=PART` pinned in **both** arms (so the rollback route
is not a variable — §6d's confound), same build (`libllama=cbea2d5bef81`,
`libllama-server-impl=60fb7910a8bb`), `build stable within each arm: True`. The env diff between the
two arms is exactly `CGC_SERVER_PREFIX_REUSE_CKPT: 0 → 1` plus the dump path.

| comparison (MTP **on**) | keys | bit-identical | differing | record-count mismatch |
|---|---:|---:|---:|---:|
| req 0, two launches (instrument check) | 306 | **306** | 0 | 0 |
| **req 1: full prefill vs restore** (the measurement) | 305 | 198 | **104** | 3 |
| req 2: full prefill vs restore | 305 | 198 | 104 | 3 |
| restore vs restore, same arm | 309 | **309** | 0 | 0 |
| **ordinal control**: two *full-prefill* requests in the reuse-free arm | 305 | 205 | **100** | 0 |
| two *full-prefill* requests, ordinals 1 vs 2, reuse-free arm | 305 | **305** | 0 | 0 |

MTP **off** (same widget, both arms on `cbea2d5bef81`; here the flag is a **no-op**: the two arms'
whole dumps are the same file, md5 `855e0d8f66354b724fc4f5a7deee23f0`):

| comparison (MTP **off**) | keys | bit-identical |
|---|---:|---:|
| full prefill vs restore (arm a) | 108 | **108** |
| full prefill vs restore (arm b) | 108 | **108** |
| restore vs restore, two launches | 108 | **108** |
| req 0, two launches | 109 | **109** |

### Three readings, in the order the evidence supports them

1. **Where reuse was already the mechanism, restore is bit-exact.** MTP off: full prefill vs restore
   is 108/108, cross-launch identical, and the two arms produce byte-identical dumps — because in
   that configuration the seq-rm probe runs, `can_seq_rm` answers `RS`, and checkpoints were already
   being created (§6c). So `CGC_PREFIX_REUSE_CKPT` adds nothing there, which is the clean way to say
   "the property is not the flag's doing".
2. **Under MTP on, the 100 draft-head keys that differ are NOT the flag's doing.** The reuse-free arm
   is the falsifier: within it, request 0 vs request 1 differ in **exactly those 100 keys**
   (`('MTP',1,*)` rows), while requests 1 vs 2 differ in **none**. Set check: the ordinal-difference
   set is a subset of the cross-arm difference set (`E∩C=100, E−C=0, C−E=7`). The extra 7 keys are
   `('DEF',4,*,1..3)` — verify-batch rows at the *proposed-token* positions, i.e. the run-to-run draft
   proposal differing, which is the same thing §6c measured as the accept shift. The same numbers
   (198/305, 100 MTP + 7 verify) reproduced on a **different build** earlier today (`971ade0f947d`),
   so they are a property of the configuration, not of one launch.
3. **The trunk's own logits — the ones that decide the first generated token — are identical either
   way.** `('DEF',1,207,0)` (the prompt tail) and the first verify batch `('DEF',4,211,0..3)` are
   byte-identical between full prefill and restore.

### What this does to the "default it?" question

The measurement says the question is **not yet answerable at M1 under MTP on** — and the reason is not
reuse. Under MTP on, the draft head's logits are a function of the *process's history*, not of the
request: request 0 differs from every later request in 100 rows **in every arm, including the arm with
no reuse at all**, and later requests are then perfectly reproducible (305/305). The M1 definition the
gate enforces ("bit-identical logits") is therefore defined, for MTP-on, **only for the first request** —
and the gate dumps one request, so it cannot see this. Two consequences, both concrete:

* Enabling prefix reuse in a rolling multi-request MTP-on workload cannot be certified by the current
gate, and *not* enabling it cannot either — they are both "request ≥ 1". That certification needs a
dump of the second request, which is what this section is, and it is a different instrument, not a
longer run of the gate.
* Under MTP off the answer is already yes, exactly (108/108), for a workload that is symmetric in the
ordinal.

All 12 requests in the four arms answered with the same md5 (`3958f37c`, 108 tokens), i.e. none of the
above moves a greedy decision **on this probe** — the same boundary §6d drew.

Artifacts: products `Backup/pfx_restore/{pair1-a,pair1-b,mtp0-ckpt0,mtp0-ckpt1}.json`, dumps and their
per-request splits beside them (`dumps.json` carries the md5s), comparison report
`/tmp/pfx_full_vs_restore_mtp1.json` (regenerable from the dumps).

## 7. Instrument defects found and fixed this round

1. **The null cell compared walls to a per-request baseline.** `--baseline-tps 8.30` is
   decode_bench's *per-request* number, but the first revision fed it the wall-based serial
   aggregate and printed **−62.7% / −65.0%** for arms whose actual per-request rate was 6.44–7.72
   (−22.4% … −6.9% in a swap-heavier regime). Fixed: the null cell now reads `serial_per_req_tps`,
   and the selftest pins it (`null cell delta is -15.2% not -65%`).
2. **The rows did not carry `prompt_n` / `prompt_ms` / `prefill_tps`**, so the product could not
   distinguish "concurrency is slow" from "this request shape is mostly prefill" — the number that
   decided §4's reading was only recoverable from the server log. Kept now, with `gen_s` /
   `non_gen_s` / `non_gen_pct` per batch, and an absent timing is `None` rather than `0.0`
   (the repo's standing absence-as-zero rule; pinned by a selftest case).
3. The driver re-ran its whole arm list and clobbered `intra_np.json`; adding one arm would have
   deleted the controls. It now merges by tag.
4. The three pre-fix products still carried the superseded `null_cell`. Rather than re-measure
   (a definition is not a measurement), the probe gained `--reanalyze <product>`: it re-derives
   `verdict` from the batches already in the file and marks it `verdict_reanalyzed`. All four
   products have been re-derived, so no product in `Backup/t5_intra_np/` now holds the −65%.
5. **`server_window.audit()` decides "is this launcher gated?" by TEXT search**, and it searches for the
   *filenames* of the gating modules. Documenting the digest defect in §6f inside
   `m123_oracle_gate.py` therefore reclassified that gate as newly gated -- `window_gate` printed
   `PROGRESS: 1 launcher(s) now gated … Run 'update'`, which is an invitation to lower the ungated
   baseline **on a comment**. It happened twice in a row (first the harness's name, then the probe's).
   Worked around in the comment; the audit itself is still comment-flippable and should ignore comments,
   because a gate that can be advanced by prose is a gate that cannot be trusted in the other direction.
6. **`m123_oracle_gate.engine_digest()` omitted the file the server's logic lives in** (`§6f`), so four
   arms with materially different server code recorded the same `llama-server` md5. Fixed.
7. **`pfx_restore_probe.witnesses()` read the server log *before* teardown**, and the pool-integrity
   lines are written during teardown — so the product recorded `pool_integrity_lines: 0`, which reads
   as "the pool is clean" when it actually meant "the line had not been printed yet". The same class
   as the absent-as-zero defects above. Fixed, and the four existing products re-derived from their
   logs; the field is now named `*_lines` with a `note` that 0 means *the line is absent*.
8. **`t5_parallel_baseline.engine_md5()` has the §6f defect too** (no `libllama-server-impl.dylib`),
   which matters here because *this* change lives in that file. Not fixed in the shared helper (it is
   another line's file): the probe records the impl digest as an extra key of its own, and it also
   records the digest **at the end of the arm** so a rebuild during a run is flagged instead of
   silently compared — which happened today (12:10 and 12:20 both moved `libllama`), and one MTP-on
   pair from the first pass had to be thrown away because its two arms were on two builds.

## 8. Boundaries

* **Not concluded:** whether the serialisation is the draft/verify chain or the env block that
  `CGC_SERVER_MTP=1` brings (§5).
* **Not measured:** `-np 3+` (only 2 slots were run), and `-np 2` under `prefill250` (chunked-pre-
  fill, MTP on) — where §6's prefix-reuse difference would interact with the 6144-token chunk path.
* The absolute levels here are swap-heavier than the 4 GiB baseline (swap 4.5–5.1 GB of 6.1 GB used,
  `window` BUSY by the harness floor of 8000 MB in every arm, launcher floor passing at ~40%).
  They are quoted against the same-regime controls inside each arm; they are not quotable against
  the 8.30 t/s baseline except in the per-request column, and even there only with the regime noted.
* `np2-mtpoff`'s `null_cell` prints +8.2% against the 8.30 baseline that was taken **with MTP on** —
  a cross-regime comparison. It is printed because the tool prints it; it means nothing. The
  within-arm 1.664 does.
* **The engine was rebuilt twice while §6g's arms were being taken** (12:10 and 12:20, `libllama`
  `971ade0f947d → 92adbd07d3c9 → cbea2d5bef81`, by a parallel session). Only comparisons whose *two*
  arms share one digest are quoted; the first MTP-on pair was discarded for exactly that reason and
  re-run. Every product carries the digest at the arm's start **and** end plus a
  `BUILD_MOVED_DURING_RUN` flag, because "same build" cannot be checked after the fact from a record
  that holds one number.
* §6g's dumps were written to `/tmp` (the engine opens the path it is given); copies and their
  per-request splits are in `Backup/pfx_restore/`. A `/tmp` reaper would make the numbers
  unquotable, which is why they were copied before the write-up.
* The arms above were taken with the **launcher's** guard only (free% ≥ 40, zero other llama
  processes); the shared window probe was not asked, and by its own floor
  (`NEED_MB = 8000`) this box refuses in every arm. The driver now asks it
  (`server_window.require_first`) before the first launch, with `--need-mb` recorded in each
  record and its samples in `window_samples`; the default is the harness floor, and a lower value
  is an explicit operator decision, not a default. A refusal is written to
  `Backup/t5_intra_np/refusals.json` rather than over the arm's measured record.
