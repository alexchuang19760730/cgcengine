# Routing trace: when do the two arms first feed different experts?

2026-09-17 · repo `flashkv-devserver` · branch `demo/sweet-spot-windows-fix`

## Why this exists

The POOL-row classifier localised the A/B divergence to

```
FIRST DIFF: graph 31, ffn_moe_down-1.pool, row 0 -- ids differ, own 214/19 [ROUTING]
             Different experts were fetched: look UPSTREAM of the gather, at routing.
```

but could only say that for the three nodes that carry POOL rows (layer 1's gate/up/down) and only
for the passes whose slots survived. The question one layer up — WHEN does routing first disagree —
is answerable from the same two logs, because `path=MV` rows carry the ids `mul_mat_id` actually
consumed for **every** MoE node (40 trunk layers x gate/up/down = 120 names), in submission order.

Tool: `Backup/mem_probe/trace_routing.py <logA> <logB> [--list N]` (splits both streams into passes
at their own per-pass marker, compares ids as sequences AND as sets, prints the first differing
pass and the first differing node in submission order).

Logs (same run, same binary `libggml-metal` sha256 `c6590288a9228efe…`, pool 4 GiB, MTP 0, cap 8):
`Backup/cgc_logs/ids_dst_capture/p25-gputime-churn_llama_server_20260917_113148.log` (A) vs
`…p25-slotgpu-churn_llama_server_20260917_113245.log` (B).

**Post-fix pair used by §9 and §10** (same profile, pool and MTP; built from the tree that carries the
record-width fix): `p25-gputime-churn_llama_server_20260917_120059.log` vs
`p25-slotgpu-churn_llama_server_20260917_120247.log`. Any pre-fix number in this file is a number
produced by the 8-word record and should be re-taken before it is quoted again — that is the whole
point of §10, and it applies to the "SAME" verdicts as much as to the "DIFF" ones.

> ## ⚠ STATUS — Answers 1-3 below were decided with a broken ids record. Read §10 before quoting them.
>
> Everything down to §9 was written against a build whose ids capture printed **8 of the 16/64 ids a
> pass actually submits**: the submission width is `n_ids = k * n_tokens` (8 x T), while
> `CGC_IDS_STRIDE` was 8 and the printer used the stride as its word count (and the kernel overran 7
> neighbour slots besides). Every "identical ids" verdict below is therefore a verdict about **token 0
> alone**.
>
> Re-measured after the fix (§10), same A/B pair: the arms do **not** diverge at pass 3. They diverge at
> **pass 0**, on **117 of 120** MoE nodes, because token 1's experts are read from **slot 0** —
> `CGC-SLOT-TABLE-CLAMP: site=pool verify il=N clamped=185/256  <-- TABLE/LEAF EQUIVALENCE BROKEN`,
> 16 times, once per layer, in the arm's own log.
>
> The pre-fix text is kept verbatim as the record, with its correction attached to each claim. §9 and
> §10 are post-fix and supersede it.

## Answer 1: the ids first differ at pass 3 — in prefill, in 38 of 40 layers  **[RETRACTED — see §10]**

**CORRECTION.** The table below is not wrong about what it measured — it is wrong about what that
represented. With the fixed record the same pair gives `pass 0: 120 compared, 117 differ`, first
differing pass **0**, and the per-layer spread is 39 of 40 layers from the first pass, not 38 layers
from the third. The "pass 3 cliff" was token 0 holding while token 1 diverged: the two tokens of a
pass sit in the same record, 8 words apart.

`n_ids` of the layer-0 gate row says which passes are prefill and which are decode
(it is 8 x T): passes 0-1 T=2, passes 2-26 T=8, 27 T=2, 28-29 T=4, **pass 30+ T=1 (decode)**.

| pass | kind | nodes compared | differ | layers |
|---|---|---|---|---|
| 0 | prefill T=2 | 120 | **0** | — |
| 1 | prefill T=2 | 120 | **0** | — |
| 2 | prefill T=8 | 120 | **0** | — |
| **3** | prefill T=8 | 120 | **114** | **2 … 39** |
| 4 … 29 | prefill | 120 | 114 | 2 … 39 |
| **30** | **decode T=1** | 120 | **120** | **0 … 39** |
| 31 … 34 | decode | 120 | 120 | 0 … 39 |

Per-layer first-difference: **pass 3 → layers 2..39 (38 layers); pass 30 → layers 0 and 1.**
So layers 0 and 1 hold their routing for the whole prefill and only flip on the first decode step.

**And it is not a reordering.** At pass 3, of the 114 differing nodes:
`identical sequence = 6 (layers 0,1)`, `same set / different order = 0`,
**`different experts (the SET differs) = 114`**. Example, `ffn_moe_gate-2`:
A `[14,3,44,16,42,31,28,65]` vs B `[14,3,46,63,16,50,26,2]` — only 3 of 8 shared.

## Answer 2: routing is DOWNSTREAM — the first carrier is a value difference in layer 1's MoE

The DST stream (node outputs, layers 0-2, 41 passes) differs from **pass 0**, and the differing set
is stable at 14 nodes: `ffn_moe_out-1`, `l_out-1`, and **all twelve layer-2 nodes**.

Layer 0 is never in the differing set (tail-32 window). Within layer 1, at pass 0, `attn_norm-1`, `z-1`(see note),
`gate-1`, `conv_input-1`, `conv_output_raw-1`, `linear_attn_out-1`, `attn_residual-1`,
`attn_post_norm-1`, **`ffn_moe_logits_raw-1`** and **`ffn_moe_weights_norm-1` are all SAME**, while
**`ffn_moe_out-1` DIFFERS**. Together with the pool-row verdict for the same window
(ids SAME, `own=` SAME, bytes SAME, no CONTENT/SLOT-REUSED/RELAYOUT row in the compared graphs)
the input side of layer 1's MoE is identical on both arms, and the output is not:

```
same ids (order too) + same owner + same pool bytes + same gate weights  ->  different MoE output
```

So the causal order is

```
pass 0   ffn_moe_out-1 DIFFERS          <- first carrier, inside layer 1's MoE combine
pass 0   l_out-1, layer-2 nodes DIFFER  <- it propagates; layer 2's router LOGITS already differ
pass 0-2 ids still IDENTICAL            <- RETRACTED, see below: true for token 0 only
pass 3   ids flip in 38 of 40 layers    <- RETRACTED: the flip is at pass 0, 117/120 nodes
pass 30  layers 0-1 flip too            <- first decode step
```

**CORRECTION (post-fix measurement, §10).** Both `IDENTICAL` lines above are artefacts. Token 1's ids
were never identical: pass 0, `ffn_moe_gate-1`, token 0 `[5,1,10,12,8,2,4,0]` on **both** arms, token 1
`[11,3,50,7,20,55,9,6]` (leaf) vs `[0,0,0,0,0,0,6,0]` (table) — slot 0, the ZERO slot. The knife edge
is therefore not three passes away and not an amplifier of a sub-ulp value: it is the first pass, in
the ids themselves. The supporting "same ids + same owner + same bytes + same weights" sentence above
held only because all four were read at token 0.

What *does* survive is the amplification measurement itself, from §9: one fp32 reassociation of the
8-term expert sum — correctness-preserving, no data touched — flips routing on 24 → 108 of 120 nodes.
So routing IS an amplifier of sub-ulp differences in general; it simply is not what is being measured
in this A/B.

## What this rules out, and what it points at

Ruled out as the first carrier: the pool's expert selection (ids identical until pass 3), the pool's
layout (`own=` identical), the pool's contents (bytes identical, no CONTENT row), the gate weights
(`ffn_moe_weights_norm-1` SAME), and the router input/logits for layer 1 (SAME).

**CORRECTION (post-fix).** The first item is wrong and it is the decisive one. "ids identical until
pass 3" was read at token 0; at token 1 the two arms' ids differ **at pass 0**, and the difference is
that the table arm reads **slot 0** (`clamped=185/256`, §10.2). Everything in this list was therefore
computed on the one eighth of each record that happened to agree — the exclusion was of token 0's
expert selection, not of the arms' expert selection. What survives from this section: the pool's
*contents* are not the carrier (`CONTENT=0`), which §10 does not touch — the bytes are the same, the
**ids that index them are not**.

Remaining candidate: **the routed expert computation and its reduction for layer 1**, where all
named inputs are identical but the output is not. The nameable mechanisms are the accumulation order
/ precision of the weighted sum over the 8 routed experts (slot order vs expert-id order, fp16 vs
fp32 accumulate), and a kernel/path difference between the two arms (the S1 arm's table is a
GPU-computed node; the host-leaf arm's is a llama-side remap) that changes the reduction tree without
changing the operands.

Falsifiable next test: canonicalise the reduction order and precision for the MoE FFN (fixed order
by expert id, fp32 accumulate) and re-run this trace. If `ffn_moe_out-1` becomes SAME and the first
difference moves to pass 3's ids only, the carrier is the reduction; if the pass-0 value difference
persists, it is kernel/path selection, not arithmetic order.

**OUTCOME (§9, §10).** The test was run, in a stronger form than proposed. Canonicalising the ids
cannot be done without permuting the routing weights with them (the weights are gathered *by* the ids
and consumed positionally), so the probe reassociated the 8-term fp32 `add` chain instead: same terms,
same pairing, different summation order.

* Reassociation alone reproduces the whole A/B signature — value difference from graph 0, then a
  routing flip — at **pass 3** on `24/120` nodes, reaching `108/120`: so the *sensitivity* the
  hypothesis was about is real and large.
* But it is not this A/B's carrier: the arms share the builder's association and their fusion counts
  are identical (`fuse=` vectors agree for `36/36` DST names), and the actual divergence turns out to
  be in the **ids at pass 0** (token 1 → slot 0), which this section could not see.
* The candidate named here as "the reduction order (slot order vs expert-id order)" is therefore
  **excluded**, and the candidate named as "a kernel/path difference" is confirmed in a narrower,
  mechanical form: `CGC-SLOT-TABLE-CLAMP` in the publish path (§10.2).

## How strong each verdict is (read before quoting any "SAME" above)

* The DST stream is **not a whole-tensor digest in this run**. The capture ran with the default
  `WORDS=32 TAIL=1` (`CGC_TENSOR_CAPTURE_HASH` unset): every DST verdict is about the **LAST 32
  ELEMENTS** of that node's output. The MV ids are exact and complete (they are the ids themselves),
  but every "`attn_post_norm-1` SAME" below means *same at the tail-32 window*. If a node's `ne0`
  differs from its neighbour's, the two windows do not even cover the same token span -- that is the
  r5 lesson (`head window IS token 0`), and it is why the window is named in every result.
  To turn this into a whole-tensor statement, re-run with `HASH=1`.
* Because of that window, the `z-1` oddity below is *not* evidence of a value difference.
* **[added post-fix] The ids record was truncated, and that mattered more than the DST window.** The MV
  stream printed `CGC_IDS_STRIDE` = **8** words per row while a row's real width is `k * n_tokens`
  (16 at T=2, 64 at T=8), so every ids verdict in this document is about **token 0**. This is the one
  boundary that changed a conclusion (pass 3 → pass 0, see §10), as opposed to merely narrowing a
  claim. Ids are exact and complete *within the token that was printed*, which is what made the error
  invisible: an exact reading of the wrong eighth looks exactly like an exact reading.

## Honest boundaries

* The MV stream is a **fixed 4096 slots** and a pass costs ~117 of them: it covers passes 0-34
  (pass 34 partial, 16 of 120 nodes) while the run has 41 passes. Everything about "when routing
  first differs" is inside the covered window; nothing is claimed about the tail. (Post-fix the record
  is 8x wider, so a run covers *fewer* passes for the same slot count — 35 here, not 41 — which is the
  price of the correction.)
* The DST capture covers layers 0-2 only (36 node names), so "layer 0 is clean" means those twelve
  layer-0 nodes, not layer 0 in isolation from the rest of the trunk.
* **The `z-1` oddity**: `z-1` differs at pass 1 only, while its upstream (`conv_*`, `linear_attn_out`)
  and downstream (`attn_post_norm-1`, `ffn_moe_*`) are SAME at that pass and at pass 0. With window
  captures this has a boring explanation: `z-1`'s `ne0` differs from its neighbours', so the tail-32
  window covers a different span for it than for them, and a SAME/diff mix across differently-shaped
  nodes is expected (r5). It is therefore recorded as a window artefact, NOT as a value difference,
  and it changes nothing about the pass-0/pass-3 structure (which rests on the exact MV ids plus a
  differing set that is stable across all 41 passes).
* The DST digests are per-element 32-bit hashes: this trace can say WHICH node and WHICH pass, not
  by how much. Magnitude needs the magnitude instrument, not this one.
* Same-arm control for the POOL stream passed (run2 vs run3 `p25-gputime-churn`: SAME=105 DIFF=0).
  The DST stream's control is the one recorded in the r6/r7 rounds, not re-run here.

---

# §9 reduction order: the instrument was blind, and the answer is that order is the amplifier

2026-09-17 (second half) · same repo/branch · two build changes, four capture runs, three comparisons.

## §9.1 First, the instrument was reading one token in eight

Every ids comparison above — "pass 3: 114 of 120 nodes differ", "identical sequence = 6", "same set /
different order = 0" — was made from a **truncated** record. The submission is

```
cgc_submit(ctx, bid_ids, dst, slot, CGC_IDS_STRIDE, n_ids = ne20 * ne21, 0, 0);
```

i.e. the row's real width is **k * n_tokens** (8 x T here), while `CGC_IDS_STRIDE` was **8** and the
printer used the stride as the word count. Two consequences, both silent:

| | effect |
|---|---|
| **truncated reading** | the MV printer emitted exactly 8 ids per pass. At T=2 that is token 0; at T=8 it is one token of eight. "The ids are identical" was a statement about **1/8 of the operands**, and token 1 (the token the recurrence carries forward) was never looked at. |
| **slot overrun** | the kernel writes `n_ids` words at `slot * stride`, so a T=8 row (64 words) ran 8 slots wide into its 7 successors — and those successors are written by the nodes that come *after* it. The surviving content of a slot was therefore the last writer's, not its own node's. |

Fix (both, one constant): `CGC_IDS_STRIDE = 64` (= 8 top-k x 8 tokens, the pool path's `n_tokens`
bound; buffer cost 4096 x 64 x 4 B = 1 MiB) and the printer clamps to the row's own recorded `n_ids`.

Verified live, same log format: `path=MV n_ids=16 ... ids=[64,6,5,4,12,1,2,39,0,7,8,11,10,13,3,9]` —
both tokens of a T=2 pass, where the old build printed the first eight and stopped. Distribution over
a run: `n_ids=16` x240, `n_ids=64` x2160 (pre-fix runs recorded the same `n_ids` but printed 8).

## §9.2 The probe: `CGC_ADD_ORDER=rev`

The FFN aggregation is a chain of `ggml_add`s over the k already-weighted expert contributions, in the
**position** order of the ids array. `CGC_ADD_ORDER=rev` reverses the *association* of that chain:
`(((c0+c1)+c2)+...)` becomes `(((c7+c6)+c5)+...)`. fp32 addition is not associative, so this is a real
arithmetic change and a mathematically identical one.

Why this shape and not "sort the ids": the routing weights are gathered **positionally**
(`weights = ggml_get_rows(probs, selected_experts)` and consumed positionally by `ggml_mul`), so
permuting the ids alone pairs expert A's output with expert B's weight — a silent wrong answer, not a
reordering. Permuting the *chain* leaves the pairing untouched. It is also the **strongest** form of the
test: if a full reversal is bit-identical, then no canonicalisation (ascending expert id, slot order,
anything) can change the result either, because each of them is a permutation of the same terms.

The banner is a raw `fprintf(stderr, ...)` after two failed attempts through the logging macros
(`LLAMA_LOG_INFO` and `LLAMA_LOG_WARN` were both absent from these arm logs while the same run carried
3847 `CGC-` lines; the string is verifiably in the loaded dylib and `llama_log_internal` has no level
filter). The engine's own `[CGC] Shutdown reminder` line proves raw stderr arrives, so the arm identity
rides that channel — an arm that cannot name itself in its own log is not an arm.

## §9.3 Result: one reassociation reproduces the entire A/B signature

Arm A = host leaf (`p25-gputime-churn`), 4 GiB pool, `load_mode=none`, MTP off, cap 8.
`trace_routing.py` compares the ids `mul_mat_id` actually consumed, per pass, all 120 MoE nodes.

| comparison | what differs | passes 0-2 | pass 3 | passes 5-11 | first diff |
|---|---|---|---|---|---|
| **P1** base vs rev — same binary, only `CGC_ADD_ORDER` | aggregation association | **0 diffs** | **24 / 120** | 72 → 75 → 93 → 96 → 108 | **pass 3** |
| **P2** base vs rev — second pair (env proven by the banner) | aggregation association | **0 diffs** | **24 / 120** | the same 72/75/93/96/108 | **pass 3** |
| **P3** **control** base vs base, no env in either | nothing | 0 | 0 | 0 | **never** |

And the DST stream (`analyze_capture_nodes.py`, layers 0-2): in P1 the **values** differ from graph 0
(`DST SAME=10 / DST DIFF=11`) while the ids are still SAME for graphs 0-3, and the ids then flip at
graph 4. In P3 the analyzer's own verdict is `every captured node agreed in every compared graph`.

So the sequence measured on the *deliberate* reassociation is:

```
graph 0     ffn_moe_out-1 etc. DIFF      <- the reassociation itself (by construction)
graphs 0-3  ids still identical           <- the value difference has not crossed a top-k boundary
graph 4     ids flip on 24/120 nodes     <- the cliff, at PASS 3
later       up to 108/120               <- nearly the whole trunk
```

which is the *same shape* as the A/B trace in Answer 1 (value difference first, cliff at pass 3,
trunk-wide after). The magnitude differs (24 vs 114 at pass 3, by the end 108 vs 114) — the probe's
perturbation is one reassociation, the A/B's is whatever the arms do differently.

## §9.4 What this settles, and what it does not

**Settled: the "coin flip" is not mysterious, and it is not a pool-size property.** A single
correctness-preserving change to the *order* of an 8-term fp32 sum is sufficient to flip routing on
90% of the trunk's MoE nodes, starting at pass 3. Any sub-ulp perturbation anywhere upstream is
therefore *expected* to become a trunk-wide routing difference three passes later. That is why pool
size, cap, and the ZERO-slot / nil-buffer bugs all presented as the same symptom: they were different
sources feeding the same amplifier.

**Also settled (by exclusion, in this round):** the fusion grouping is NOT the amplifier channel. The
DST records carry the dispatcher's `fuse=` count per node, and the A/B arm pair's fuse vectors are
**identical for all 36 names** (`0/36` differences), so the two arms do not differ in how many
operations the bin dispatcher merged.

**Not settled: where the A/B's own first perturbation enters.** The trace in Answer 2 says layer 1's
`ffn_moe_out` differs at pass 0 with identical ids, identical router logits and identical normalized
weights. If the association is the same in both arms (same builder, same fusion counts) and the
operands are the same, that output cannot differ — so one of those "identical" claims is still wrong,
and the candidates are now narrow and ordered:

1. **the expert-weight bytes behind the consumed ids** — the POOL-row digest compares rows at the
   same *slot index* in both arms; in a layout where the same expert sits at different slots, "row k
   SAME" and "ids SAME" can both be true while the *pairing* is not;
2. **the association** — not the fusion count (excluded above) but a different node ORDER inside the
   fusion window, which the `fuse` counter does not record;
3. **an uncalibrated operand**: `ffn_moe_weights_*` are outside the DST list for layer 1's MoE
   internals (`ffn_moe_gate/up/down-1.dst` are among the nine names the analyzer reports as NEVER
   CAPTURED), so "the FFN's own inputs are identical" is asserted from the router side only.

The next measurement should be (1): make `compare_pool_row.py` compare rows keyed by the **expert id**
the engine believes it placed there (`own=` is already in the record) rather than by slot index, so a
pure relayout stops reading as a content difference and a genuine content difference cannot hide
behind one.

## §9.5 Reproduction

```bash
# build (the only two files touched)
cmake --build src/llama.cpp/build -j8 --target llama-server

# base, then the same thing with the aggregation reversed
R13_TAG=r16_base R13_LOAD_MODE=none R13_POOL_BYTES=4294967296 python3 Backup/mem_probe/launch_r13.py
R13_TAG=r17_rev  R13_LOAD_MODE=none R13_POOL_BYTES=4294967296 CGC_ADD_ORDER=rev \
    python3 Backup/mem_probe/launch_r13.py          # log must contain: CGC-ADD-ORDER: ... REVERSED

python3 Backup/mem_probe/trace_routing.py A.log B.log            # ids, per pass, all 120 nodes
python3 Backup/analyze_capture_nodes.py A.log B.log --graphs 0   # DST values, layers 0-2
```

## §9.6 Honest boundaries of §9

* `CGC_ADD_ORDER` is a **probe, not the fix**. Both A/B arms already share the builder's left-to-right
  association, so a canonical order cannot by itself make them agree; what the probe establishes is
  that the *sensitivity* is large enough to explain the whole A/B signature. Presenting it as "the
  carrier" would overstate it: it is the amplifier, measured.
* P1's two arms are the same binary; P2's two arms differ by one `fprintf` (a log line, no arithmetic).
  Neither pair is a binary-identical comparison, and neither is quoted for throughput.
* The probe changes the *text* the model produces (routing differs from pass 3 on), so no P1/P2 run may
  be quoted for quality or for the M1/M2/M3 gate; the gate reference is unaffected because the flag is
  off by default.
* The ids stream is still 4096 slots and a T=8 pass costs 64 of them per node — wider records mean the
  ids stream now covers **fewer passes** per run (35 vs 41 here). The pass-3 conclusion is inside the
  covered window in every run above; nothing is claimed about the tail.
* `n_ids=16`/`64` splits the record into tokens by the operand's own `nb21`; the token that a given
  eighth belongs to is inferred from that stride, not printed per element.

---

# §10 The fixed instrument overturns Answer 1, and names the carrier

The same A/B pair (`p25-gputime-churn` vs `p25-slotgpu-churn`, run `r16`, 4 GiB pool, cap 8), re-read
with the full-width record:

| | ids first differ | nodes at that pass |
|---|---|---|
| old record (8 words = token 0 only) | **pass 3** | 114 / 120 |
| **fixed record (all 16/64 words)** | **pass 0** | **117 / 120** |

**Answer 1's "the ids first differ at pass 3, in prefill" was an artefact of reading one token.** The two
arms already disagree about which expert to fetch in the *first* prefill pass, on 117 of 120 MoE nodes.
There is no three-pass incubation period in the A/B; the "cliff at pass 3" was token 0 staying identical
while the rest of the batch did not.

## §10.1 What the difference actually is: token 0 vs token 1

Raw ids for `ffn_moe_gate-1` / `ffn_moe_down-1`, pass 0 (T=2, so the first 8 words are token 0's):

```
A leaf   row0: tok0=[5,1,10,12,8,2,4,0]  tok1=[11,3,50,7,20,55,9,6]
B table  row0: tok0=[5,1,10,12,8,2,4,0]  tok1=[ 0,0, 0,0, 0,0,6,0]   <-- 7 of 8 are slot 0
A leaf   row1: tok0=[23,16,17,26,19,25,65,14]  tok1=[24,15,21,13,18,53,51,22]
B table  row1: tok0=[23,16,17,26,19,25,65,14]  tok1=[65,13, 0, 0, 0, 0,26, 0]
```

Token 0 is **identical** in both arms — which is precisely why every previous round, reading words 0-7,
concluded "the table equals the leaf on every consumed id". Token 1 mostly reads **slot 0**.

## §10.2 The engine already counts this, and had been saying so

The arm's own audit fires 16 times, once per layer:

```
CGC-SLOT-TABLE-CLAMP: site=pool verify il=1 clamped=185/256
  <-- TABLE/LEAF EQUIVALENCE BROKEN: gather reads slot 0 (another expert's weights)
      while the host leaf writes -1 (loud)
```

`clamped = 185/256` on every layer: the **published** slot table clamps negative entries to 0, where the
host leaf writes `-1` (deliberately loud). So a GPU gather over that table sends 185 of 256 experts to
slot 0 — the ZERO slot — for every token whose experts fall in the clamped set. This is the
"ZERO-slot contamination" class the project fixed on the *leaf* path (`CGC_SYNCFILL_COLD`); the
**published table path never got the fix**, and the equality audit between the two was reading eight
words.

Two consequences worth stating separately:

* `CGC-ZERO-MAPPED = 0` in **both** arms, because that counter lives on the leaf path (`expert_cache_on_topk`).
  The S1 arm's losses were counted by `CGC-SLOT-TABLE-CLAMP` instead — a different counter, in a different
  file, printed once per layer and therefore easy to miss in a 4 000-line log.
* This also closes the "is it a capture artefact?" question for the zeros above: the clamp is reported by
  the **host**, on the publish path, not by the device capture. The device and the host agree that the
  table says 0.

## §10.3 So which branch of the hypothesis is it?

**Neither, as posed — and both of its diagnoses were wrong in an instructive way.**

* The reduction order is **not** the A/B's carrier: the arms' association is the same (same builder, and
  the `fuse=` vectors are identical for all 36 DST names — `0/36`), and the divergence is present at
  pass 0 with the ids themselves differing.
* The "different kernel/path" branch is closer, but the concrete answer is narrower and totally
  mechanical: **`CGC-SLOT-TABLE-CLAMP` — the published table maps cold experts to slot 0 while the leaf
  maps them to -1, so the S1 arm silently reads the zero page for them.**
* §9's probe still stands on its own footing, as the **amplifier calibration** rather than the carrier:
  one fp32 reassociation of the 8-term sum (correctness-preserving, no data touched) is enough to flip
  routing on 24 → 108 of 120 nodes. That is the measured gain of the path from "a sub-ulp difference" to
  "a different trace", and it is why a single mis-mapped token 1 becomes a trunk-wide A/B difference.

## §10.5 The output side was never instrumented — it was never *requested*

`ffn_moe_gate/up/down-*.dst` appearing in the "CHAIN but NEVER captured" list is not a missing hook:
`ggml_metal_op_mul_mat_id` already calls `cgc_dst_capture(ctx, bid_dst, op)` on **both** dispatch paths
(MV at `ggml-metal-ops.cpp:4786`, MM/map0 at `:4663`). The nine names were absent because the
`NODES` list in `Backup/run_ids_dst_capture.sh` does not contain them. One env var, no rebuild:

```bash
NODES="attn_norm-1,z-1,gate-1,conv_input-1,conv_output_raw-1,linear_attn_out-1,attn_residual-1,\nattn_post_norm-1,ffn_moe_logits_raw-1,ffn_moe_weights_norm-1,ffn_moe_gate-1,ffn_moe_up-1,\nffn_moe_down-1,ffn_moe_out-1,l_out-1" HASH=1 POOL=1 R13_TAG=r19_mmid \
  R13_LOAD_MODE=none R13_POOL_BYTES=4294967296 python3 Backup/mem_probe/launch_r13.py
```

(`CGC_DST_FILTER_MAX` is 2048 bytes, so a 15-name list is nowhere near the truncation that bit the
23-name list in the r5 round; the run's own `CGC-TENSOR-CAP enabled:` banner is the check.)

With all three `mul_mat_id` outputs in the filter, whole-tensor digests, **same round as the ids and
the clamp counter**, layer 1:

```
pos 22 attn_post_norm-1.dst          first_diff = 30
pos 23 ffn_moe_logits_raw-1.dst      first_diff = 30
pos 24 ffn_moe_weights_norm-1.dst    first_diff = 31
pos 25 ffn_moe_gate-1.dst            first_diff =  1     <-- the routed gate projection
pos 26 ffn_moe_up-1.dst              first_diff =  1
pos 27 ffn_moe_down-1.dst            first_diff =  1
pos 28 ffn_moe_out-1.dst             first_diff =  1

=> earliest divergence: graph 1, node ffn_moe_gate-1.dst
   earlier in the chain: attn_norm-1=SAME z-1=SAME gate-1=SAME conv_input-1=SAME
   conv_output_raw-1=SAME linear_attn_out-1=SAME attn_residual-1=SAME attn_post_norm-1=SAME
   ffn_moe_logits_raw-1=SAME ffn_moe_weights_norm-1=SAME
```

**This is the reading §10.3 predicted, and it is what makes the two sides of the disagreement the
same statement.** The expert *choice* is identical (router input, logits and the normalized routing
weights are all SAME until graph 30/31 — those are functions of the router input and the chosen ids
only), while the `mul_mat_id` **output** differs from graph 1. The only stage between the two is
**which slot the selected expert is read from** — and that is exactly what `CGC-SLOT-TABLE-CLAMP
clamped=185/256` and the raw token-1 ids (`[0,0,0,0,0,0,6,0]` vs `[11,3,50,7,20,55,9,6]`) say.

'Routing differs' and 'routing is identical' were both being reported about the same arm pair, and
both are true: the **decision** is identical, the **placement lookup** is not.

## §10.6 The clamp attribution is NOT established — the arm's own counter says so

§10.2 and §10.3 named the publish-path clamp as the carrier. That attribution has to be withdrawn, on
the S1 arm's own teardown line from the same run:

```
llama_expert_cache: S1 slot-table: publishes=1599 clamped_selected=0 clamped_table=295815
                   changed_entries=1472 consumed_changed=329 consumed_unchanged_publishes=61
```

`clamped_selected` counts exactly the case that would matter — a **selected** expert whose live slot is
negative while the published table says 0 (`llama-context.cpp:4455-4463`, `st[e] < 0 && tb[e] == 0`) —
and it is **0**. The 295815 is the whole-table count, i.e. the non-resident majority of a 71-slot pool
over 256 experts, which no consumer reads; the counting function's own comment says that number is
"large in EVERY run" and therefore carries no information. So: **the 0s in the captured ids are not
clamps of selected experts.**

The mechanism therefore has to be one of the following, and the code cannot separate them:

| # | candidate | what would confirm it |
|---|---|---|
| 1 | `zs >= 0` → `dst[e] = zs` (the reserved ZERO-slot branch, which does NOT increment `clamped`) | cannot hold on this run: that branch makes `clamped` 0 for the layer, and the log shows `clamped=185/256` on all 16 layers, i.e. `zs < 0` everywhere |
| 2 | `slot_table[e] == 0` genuinely (slot 0 is a live slot when `zs < 0`) | `CGC_S1_DBG=1` → `CGC-S1: EQUIV-pool il=1 ... table= safe= zero_slot=` for the selected experts: `table == safe == 0` means the cache really assigned slot 0 |
| 3 | the published table is **stale**: published before the step's union ensure, so the selected experts read back as they were one step earlier | the same EQUIV line with `table != safe` (and `mismatch > 0`) |

One env var selects between 2 and 3 (`CGC_S1_DBG=1` on the table arm; the print already exists at
`llama-context.cpp:4390`). Until that runs, the honest statement of the carrier is:

> the expert **decision** is identical (router input, logits, weights all SAME until graph 30/31) and the
> **placement lookup** differs from graph 1 onward; *why* the lookup returns 0 for six of token 1's eight
> experts is **not yet established**. Two candidates, one 4-minute run apart.

What §10.1-§10.2 *do* establish, and what does not depend on the attribution at all:

* the pre-fix "ids identical until pass 3" reading was token-0-only, and the full-width record shows the
divergence on the first chunk graph (117/120 nodes);
* token 0's ids are byte-equal on both arms, token ≥1's are not;
* `clamped=185/256` per layer and 16 `CGC-SLOT-TABLE-CLAMP` lines are real — they are simply about the
unselected majority, which is why they were the wrong thing to read as the cause.

## §10.4 What to do next, in order

> **REVISED 2026-09-17 12:50 — item 1 below is retracted and item 2 is replaced. Read §11 first.**
> The clamp does **not** touch any consumed id: counted race-free *inside* the publish loop it is
> `sel_wrong=0` on every layer. And the `id_oob` evidence item 2 was going to re-measure is a
> host-pointer artifact, not a statement about the kernel.

1. ~~**Fix the publish path's clamp**~~ — retracted, see §11.2. The 16 `CGC-SLOT-TABLE-CLAMP` lines
   are about the unselected majority; they are not an acceptance test for anything the consumer reads.
2. **Re-run the A/B with the fixed record** and expect pass 0 to become 0/120. Until then, *no* S1
   measurement (speed or quality) is comparable with the leaf arm, including the earlier "the extra
   nodes are inert" conclusion — that one was also decided on an 8-word reading.
3. Only then is §9's question (does a canonical reduction order matter?) worth asking again — and note
   that §9 already answers the part that matters: any *reordering* is inert-or-amplifying, never
   stabilising, so a canonical order is a robustness measure for the gate, not a repair for this bug.

## §11 The clamp is exonerated, and the probe that blamed it does not read the kernel's operand

Measured 2026-09-17 12:38–12:47 on the S1 table arm (`p25-slotgpu-churn`, `CGC_SLOT_TABLE_GPU=1`,
4 GiB pool / 71 slots, cap 8), with a rebuild that moved the selected-subset count into the publish
loop. Logs: `Backup/cgc_logs/ids_dst_capture/p25-slotgpu-churn_llama_server_20260917_123843.log`
(pre-change) and `..._124555.log` (post-change).

### §11.1 The table is correct for every CONSUMED id

`llama_expert_cache_publish_slot_table` now takes the step's raw ids and returns, from the same loop
that writes the entries, how many consumed ids were published as a placeholder and how many
consumed ids landed on a slot not owned by their expert. Result on all 39 layers of every step:

```
CGC-SLOT-TABLE-CLAMP: site=pool verify il=1 clamped=185/256 (no consumed id affected: sel_wrong=0)
```

`CGC-S1-CLAMP-SELECTED` never fired (count 0). So `clamped=185/256` is entirely about experts nobody
reads this step, and the answer to "is the table really 0 for what the consumer reads" is **no**.

Why the old counter said something different: it re-read the *live* table after publishing
(`st[e] < 0 && tb[e] == 0`). A background fill landing between the two loops flips `st[e]`
non-negative and the entries that *were* published as 0 are then counted as fine. It reported 0 on
every layer of every run — including runs where the consumer's operand provably was not the table's
values, i.e. it said "nothing wrong" about the one thing it exists to detect. That is the
instrument-not-in-the-room failure this project has now hit in three different files.

### §11.2 `id_oob` is a host-pointer artifact, not a kernel reading

Two probes read the same mmid ids tensor in the same run and disagree:

| probe | how it reads | result |
|---|---|---|
| `CGC-MMID` / `CGC-MMID-ASSERT` | `ids->data` (host pointer) | `id_oob=16/16`, `id_oob=64/64`, first values `1065319088, 1038952758, …` = F32 bit patterns (~0.05–0.93, router-probability range), `ids_offs=459392` |
| `CGC-IDS-CAP` (ring) | device blit from the backend buffer at the mmid node | `[7,36,6,12,10,1,8,9,0,15,33,0,0,0,19,0]` — small ints; words 0–7 equal the published table exactly |

For `il=1, graph 0` the ring's first eight words are `[5,1,10,12,8,2,4,0]` and the published table's
`pub` for the same eight experts is `5,1,10,12,8,2,4,0`. Identical. So the device **is** reading the
host-published map, and every "the GPU consumed an out-of-range id" statement in this file's earlier
sections rests on a pointer that does not address the device operand. Consequences to carry forward:

* `id_oob=0` on the leaf arm vs `id_oob>0` on the table arm is **not** a comparison of what the two
  kernels read. The generated ids differ in *type* (leaf: annotated host ids; table: gathered
  operand), so the two arms' OOB counts were never the same quantity.
* The earlier reading "the ids operand is float garbage on the table arm" is withdrawn here.

### §11.3 What is still genuinely unexplained, and it is one line wide

The ring's words 0–7 match the table and words 8–15 do **not** (token 0 is right, token ≥1 is not):
`[5,1,10,12,8,2,4,0, 0,0,0,0,0,0,6,0]` against `pub = 5,1,10,12,8,2,4,0, 11,3,50,7,20,55,9,6`. The
last word *does* match the table's 16th entry, which is what a partially stale buffer looks like when
one later write happens to land. Zeros grow with chunk size (T=1: 0 zeros; T=2: ~8/16; T=8: ~30–48/64)
and the runtime graphs that produce them are declared `ids ne=[8,2]`, `n_ids=16`, `nbi1=32`.

Next experiment, and the only one worth running: print, for **every** build (not the first six, which
are the 1-token reserve build and print `ids_flat_ne0=8 slots_ne=[8,1]`), the runtime `slots->ne[0]`,
`slots->ne[1]` and the element count ggml-alloc actually reserved for that node, next to the
consumer's `n_ids`. If the gather's output is sized for one token while the mmid reads `k*T`, the
stale tail is explained by construction and the fix is to build the S1 nodes from
`selected_experts->ne[1]` rather than from the local `n_tokens` (the code already documents that the
two can disagree: "build_moe_ffn's `n_tokens` is NOT ubatch.n_tokens").

### §11.5 FOUND: the index vector is correct for token 0 and wrong for every token ≥ 1

Three questions were open. All three now have answers, and the last one is the bug.

**Shapes are not it** (`CGC-S1: SHAPE`, 2026-09-17 12:51, deduplicated by signature, every distinct build):
`n_tokens == selected_experts->ne[1]` in all five runtime shapes, `ids_flat == k*T`, `slots_nbytes == 4*k*T`
(32/64/128/192/256 for T=1/2/4/6/8). No MISMATCH line ever printed.

**The gather is faithful.** With the index vector pinned (`CGC_S1_PIN_IDS`) the post-synchronize readback
finally passes its own validity test (`ids_src_valid=1`) and reports `gather_vs_table=[0]` for all 16
positions — i.e. `gather[k] == table[idx[k]]` everywhere, tail included. The published table is
therefore read exactly as written, and the earlier `sel_wrong=0` result (race-free, inside the publish
loop) already said the table is right for the consumed set.

**What is wrong is the index vector that feeds the gather** (`ggml_cont(selected_experts)`), measured
against the hook's own ids in the same run:

```
il=1  hook = [193 105 229 249 220 106 181  84 | 237 163  50 218  20  55 222 212]
      idx  = [193 105 229 249 220 106 181  84 | 161 103 250  74 110 121 212  99]
      agree 8/16, DISAGREE at positions 8..15
```

Same on every layer (0..7 at least, and the pattern held for all 39 in earlier runs): token 0's eight
indices are exact, token ≥ 1's are a *foreign but legal* vector — every value in [0, 256), so nothing
asserts. The published table is ~72% non-resident zeros, which is exactly why the corruption surfaced as
"mostly zeros with occasional small slots" rather than as a crash, and why the gather's own output
looked plausible enough to be read as a mapping problem for several rounds.

Mechanism, to be confirmed by the fix: `ggml_cont(selected_experts)` copies a *view* whose
`nb[1] = n_expert*4` with `ne = [k, T]`. If the copy reads contiguous k-blocks instead of honouring
`nb[1]`, token 1's row becomes ranks 8..15 of token 0's sorted list — legal ids, unrelated to the
router's choice for token 1, which is what the numbers show. The CONT was added to satisfy
`ggml_reshape_1d`'s `ggml_is_contiguous` assert, so the assert passed while the data did not.

Consequence for the S1 line: this is not a pool, clamp, allocator, or kernel-lookup problem. The
host-leaf arm is unaffected because the host writes that vector, which is why it is bit-identical to
the baseline and the table arm is not.

### §11.6 The fix (option A) is implemented but NOT yet correct — blocker is identified

Implemented on 2026-09-17 13:00-13:08, `CGC_SLOT_TABLE_GPU=1` arm only:

* graph: `ffn_moe_ids_leaf` (2-D `[k, T]` I32, `ggml_set_output`, graph root) replaces
  `ggml_cont(selected_experts)` as the source of the index vector; the node structure around it
  (CONT -> RESHAPE_1D -> GET_ROWS -> RESHAPE_2D) is kept because two alternative structures aborted
  the reserve build with `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)` (the root reached only through
  a view; and a 2-D index producing a 3-D gather).
* hook: writes the step's raw ids into that tensor at both remap-leaf sites, next to the table publish.
* `llama-context.h`: `cache_ids_tensors[il]`, filled from the eval callback as `ffn_moe_ids_leaf` is seen.

Result: the server runs, the hook fires (16 publish lines, `sel_wrong=0`), the write takes the right
values -- and the gather still reads **zeros** for all 16 positions on every layer:

```
CGC-S1-IDS-WRITE: il=1 t=0x1306f17c0 data=0x12a1e4500 ne=[8,2] n=16 v0=193 v8=237
CGC-S1: POST il=1 gather=[0 x16] idx=[0 x16]
        gather_data=0x12a25a640  ids_src_data=0x12a256540  ids_leaf_data=0x12a1e4500
```

`ids_leaf_data` sits in the same buffer as `table_data` (`0x12a1e4100`, 1 KiB apart) — the buffer whose
writes demonstrably reach the device, since the table's contents are what the gather returned in §11.5.
`ids_src_data` is an arena address (~40 KiB away). So the tensor the hook writes through
`cache_ids_tensors[il]` is **not** the tensor the executing graph copies from.

Most likely cause, and the next thing to check: the map is keyed by layer only, so it holds whichever
build registered last, while several graphs of the same shape exist over a run. The fix is to write the
ids from the **eval-callback branch for `ffn_moe_ids_leaf` itself** (where `cur` is by construction the
node of the graph being encoded), using the ids the logits callback stored a moment earlier -- rather
than reaching into a map from the logits hook.

State: the default path (no `CGC_SLOT_TABLE_GPU`) is behaviourally unchanged -- both the new publish
counters and every S1 change are gated on the arm's own flags; only debug prints are always visible.
The arm itself is now *worse* than before this change (it used to get token 0 right and only the tail
wrong; with the index vector unwritten it gets nothing), so do not quote anything from this arm until
the map issue is fixed.

### §11.4 Code changed for this (uncommitted)

* `llama-expert-cache.h` / `.cpp`: `llama_expert_cache_publish_slot_table` takes optional
  `sel_ids`/`n_sel_ids` and returns `out_sel_clamped`/`out_sel_wrong` from inside the write loop; the
  contract comment records why post-hoc counting is wrong. New debug-only `CGC_S1_TAG=<v>` shifts
  resident entries and tags placeholders with `1000+v` so a `0` in a capture can be told apart from
  "never written"; documented as never-for-quality.
* `llama-context.cpp`: the racy post-hoc loop is gone; the two counts are now distinct, and the
  clamp message distinguishes "no consumed id affected" from the new `CGC-S1-CLAMP-SELECTED`.


---

## §12 r31–r33 (2026-09-17 13:1x): the carrier was the HOST's top-k read, not the device gather

§11.5 chased a device-side index vector that "was correct for token 0 and wrong for token >= 1"
and was about to rebuild that vector from the host. Both halves of that plan were wrong, and the
measurement that settled it is one line of tensor geometry.

### 12.1 The decisive reading

`expert_cache_on_topk` reads the top-k like this:

```c
const int32_t * ids = (const int32_t *) t->data;
std::vector<int32_t> ids_snap(ids, ids + n_tokens * n_expert_used);   // LINEAR indexing
```

With a print of the tensor's own layout next to it (r32, S1 arm):

```
CGC-TOPK-SHAPE il=1 t_ne=[8,2,1,1] t_nb=[4,1024,2048,2048] t_op=38   ids_stride_expected=32
CGC-S1: SELSHAPE il=1 sel_ne=[8,1] sel_nb=[4,1024] ops=38
```

`t` is `ggml_argsort_top_k`'s output: a **view into a [n_expert, T] argsort**, so `nb[1] = 1024 =`
`n_expert*4`, not `k*4 = 32`. Linear indexing therefore read, for token `t >= 1`, element
`t*k + j` of a row-major walk over that view -- i.e. **ranks k..2k-1 of token 0's sorted list**.
Legal expert ids, silently the wrong token. The device-side `CONT(selected_experts)` is nb-aware
and was right all along; that is why the two disagreed, and the disagreement was read here for
two rounds as a gather bug.

Consequence, and it is not cosmetic: that snapshot is what writes the remap leaf, so **every
T >= 2 step (all prefill, all batch verify) routed tokens >= 1 through another token's experts**.
It stayed invisible because it is deterministic -- every arm and every pool size made the same
mistake -- so cross-pool invariance (M1/M2) passed while the ids were wrong for all of them.

### 12.2 Retired: the host-written index vector (r31)

`ffn_moe_ids_leaf`, a 1-D I32 root written by the hook and consumed directly by the S1
`GET_ROWS`, scheduled fine and the host readback confirmed both the address and the values:

```
ids_src_data == ids_leaf_data (0x11146c500)   ids_src_valid=1
idx = [193 105 229 249 220 106 181 84 | 237 163 50 218 20 55 222 212]
```

yet the gather returned `[0 x16]` (`gather_vs_table=[15]`): the device read zeros for an index
vector whose host mirror held those numbers. A host-written root consumed by a device `GET_ROWS`
is not delivered to the device (the table and the remap leaf are consumed by `mul_mat_id`, not by
a device gather), so no host filling strategy can rescue that design. The index vector therefore
went back to being device-produced (`CONT -> RESHAPE_1D -> GET_ROWS`), which is also the only
variant with a speed rationale: it needs no host readback of the router at all.

### 12.3 The fix and what it verifies

`llama-context.cpp`: the snapshot is now nb-aware (`row = data + t*nb[1]`, `j`-th id at
`row + j*nb[0]`), so tokens >= 1 get their own top-k.

Same-run evidence (r33, 4 GiB, `ARMS=p25-gputime-churn,p25-slotgpu-churn`, N_PREDICT=12):

| check | before | after |
|---|---|---|
| `CGC-S1: POST ... MISMATCH` lines | 936 | **0** |
| `gather_vs_table` | `[15]` (r31) / `[n/a]` | **`[0]`** -- `gather[k] == table[idx[k]]`, all positions |
| ids axis, leaf arm vs S1 arm | differed from graph 1 | **IDENTICAL for graphs 0..9**, 1 row is a node that exists only in the S1 capture |
| device pool rows, leaf vs S1 | A `OK=96 LOST=0`, B `n/a=96` (no equivalence) | **SAME=12 DIFF=0** |

### 12.4 Honest limits

* **No speed claim.** The S1 arm measured `decode 6.76 t/s / prefill 10.49 t/s, hit 72.0%,
  miss 12205` on a box whose thermal state was `NOMINAL` at that moment; the leaf arm's numbers
  in the same run are `0.00` with `HEA/UNR` thermal, so there is no A/B. Nothing here shows S1 is
  faster -- and it cannot be, structurally: `expert_cache_on_topk` runs in **both** arms, so S1
  never removed a host step. Its measured value is as an instrument (device-side pool rows).
* **Existing oracles are stale by construction.** The fix changes ids for T >= 2 in *every* arm,
  including the host-leaf baseline, so any 8 GB reference produced before r33 encodes the old
  routing. The provenance sidecar gate should refuse those comparisons -- that refusal is correct,
  not an obstacle: a new reference has to be generated.
* Not re-verified in this round: M1/M2/M3 gates (they need the regenerated reference), the 48-q
  suite, and `--no-mmap`/6-8 GiB cells. All changes are uncommitted.
* `s1`'s arm is left OFF by default (`run_server.sh` passes only `CGC_SLOT_TABLE_GPU`, unset).


---

## §13 r34–r37: the nb-aware fix measured -- reference regenerated, gates green

Same binary as §12 (built 13:24). Instrument: `scripts/check/m123_oracle_gate.py` (launches through
`run_server.sh`, dumps the 9-row logits oracle, compares M1/M2/M3, and resolves the launch config on
BOTH sides so a cross-config diff is reported as INVALID COMPARISON rather than as a regression).

### 13.1 Which oracle rows the fix changed (the direct answer)

New binary vs the pre-fix reference `..._v5_spac.jsonl`: **M1 5/9, M2 6/9, M3 5/9** -- a real
numeric change, not a tie-break (`num_eq_dec_ne = 0`).

Row by row, `v5_spac` vs the regenerated `..._v6_nbaware.jsonl`:

| key (step, token_idx, ctx) | row_fnv1a64 identical |
|---|---|
| (0, 0, DEF) | **no** |
| (1, 0, DEF) | yes |
| (2,0) (3,0) (4,0) MTP | yes |
| (5, 0, DEF) | **yes** |
| (5, 1, DEF) | **no** |
| (5, 2, DEF) | **no** |
| (5, 3, DEF) | **no** |

The `(5,k)` pattern is the mechanism's own signature: step 5 is a 4-token step, **token 0 is
unchanged and every token >= 1 changed** -- exactly "tokens >= 1 were routed through another
token's experts". The MTP rows are unchanged.

Open, and honestly not explained here: `(0,0,'DEF')` also changed, i.e. the prefill chunk's
`token_idx 0` row. A causal first chunk would not depend on tokens >= 1, so either this dump's
`token_idx` for step 0 does not mean what it does for step 5, or the chunk's MoE is not ordered the
way the key suggests. It does not affect the verdicts (M2 flips there too), but it is the one row
whose change the mechanism above does not account for.

### 13.2 New baseline is self-consistent

`--write-ref Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl` (+ `.cap`,
carrying the resolved config and the note). A fresh launch against it: **M1 9/9, M2 9/9, M3 9/9**.
The two runs that FAILED against v5 (r34_oldref, r34_newref) produced byte-identical hashes, so the
engine is deterministic across launches and the failure was the reference, not the run.

### 13.3 Pool invariance against v6 (4/6/8 GiB)

| pool | M1 | M2 | hit% | misses (compulsory/capacity) | resident | `buffer is nil` |
|---|---|---|---|---|---|---|
| 4 GiB | 9/9 PASS | 9/9 | 72.2% | 1303 (1302 / 1) | 3048 MiB | none |
| 6 GiB | 9/9 PASS | 9/9 | 82.8% | 1047 (1047 / 0) | 4728 MiB | none |
| 8 GiB (ref) | 9/9 | 9/9 | -- | -- | -- | -- |

Miss attribution is ~100% **compulsory** at both smaller pools (capacity 1 of 1303 at 4 GiB), i.e.
the cache is not being thrashed -- misses are first touch. **Caveat:** pool bytes are part of the
gate's compared config, so the 4/6 GiB cells carry `INVALID COMPARISON`; they were read with
`--allow-incomparable` on purpose, because the pool IS the axis under test. They are sanctioned
cross-config reads, not clean gate PASSes. The probe is N=9 on one prompt, not the 88-step
knifeedge vector.

### 13.4 Prefill hit rate before/after (counters, not timings)

Same config (profile `prod25`, pool 4 GiB, `N_PREDICT=12`, S1 arm), consecutive runs:

| run | binary | hit% | misses | io MiB/s | decode t/s |
|---|---|---|---|---|---|
| r30addr 13:07 | pre-fix (linear ids) | 57.1% | 31847 | 177 | 6.37 |
| r31fix 13:12 | `ids_leaf` (retired, §12.2) | 58.7% | 30648 | 176 | 7.21 |
| r33 13:27 | **nb-aware ids** | **72.0%** | **12205** | **211** | 6.76 |

The hit/miss/io columns are cache counters and survive thermal drift; the t/s column does not
(r33 was `NOMINAL`, r30/r31 unrecorded), so only the counters are quoted as an improvement:
**hit 57.1% -> 72.0%, misses -62%**. Consistent with the preview of §12: token >= 1 had never
loaded the experts it actually routed to.

### 13.5 Tooling added

`Backup/mem_probe/detach_run.py <log> <cmd...>` -- generic double-fork detach, because every long
harness here dies when the calling tool reaps its process group (the failure looks like a crash
with no output). Note also that `m123_oracle_gate.py`'s `--port` is the GATE's probe port while
`run_server.sh` decides the server's; passing anything other than 8080 makes the gate poll a port
nothing listens on and time out at 300 s.


---

## §14 r38–r41: MTP accept re-measured -- the fix meets the M4 exit, and accept moved the wrong way

`scripts/check/mtp_accept_ab.py` (measures the server's own `timings.draft_n_accepted/draft_n`, gates
on head liveness through `mtp_head_identity.py`). Carrier `nail` (Nail's head on Nail's base, head
identity `219b27c4f9ee`), pool 8 GiB, `n_predict=96`, 3 requests per arm.

### 14.1 Why there is a knob (and why a 3-day-old baseline would not have been a before/after)

The only earlier accept number is `Backup/cgc_logs/mtp_accept_ab.json` (2026-09-14): nail
**70.4%** accept at decode 6.74 / prefill 6.00 t/s. That was a different binary *and* a different
dispatch regime, so comparing across it measures three days of engine work, not this fix.

So the pre-fix read was made switchable instead: `CGC_IDS_LINEAR_READ=1` restores the linear
snapshot of `expert_cache_on_topk` (the §12 defect) in the DEFAULT path's own code, forwarded by
`run_server.sh`'s allowlist. Same build, same pool, same dispatcher, same sampler, same prompts.

### 14.2 Same-binary A/B (build 14:12)

| arm | accept | acc/gen | prefill t/s | decode t/s |
|---|---|---|---|---|
| MTP off (denominator) | n/a | 0/0 | 19.24 | **9.82** |
| MTP on, `CGC_IDS_LINEAR_READ=1` (pre-fix) | **73.81%** | 155/210 | 7.30 | **8.62** |
| MTP on, nb-aware fix (default) | **58.25%** | 180/309 | **21.74** | **12.62** |

* **M4's exit condition is met**: MTP-on 12.62 >= MTP-off 9.82 (+28.5%). In the pre-fix arm MTP was
  net **negative** (8.62 < 9.82) -- i.e. the feature was being correctly rejected on a measurement
  that was itself broken.
* The fix is worth **+46% decode** and **~3x prefill** against the pre-fix read on one binary.

### 14.3 The counterintuitive part, stated plainly

**Accept went DOWN (73.81% -> 58.25%) while throughput went UP (+46%).** Both are real, and they are
not in tension: in the pre-fix arm the verify batch's tokens >= 1 were computed with another token's
experts, so "accepted" was a comparison against the wrong distribution -- and it *inflated* the
metric. Any M4 decision keyed on accept alone would have kept the defect and rejected the fix.

The mechanism of the inflation is NOT established here. What is established: the measured pair on one
binary, and that pre-fix accept was inversely related to throughput. The 2026-09-14 number (70.4%) sits
in the same inflated family as today's pre-fix arm (73.8%), so every accept number recorded before the
fix should be treated as belonging to that regime.

### 14.4 Limits

* 3 requests per arm, one carrier, one pool. The prefill delta (7.30 -> 21.74) is partly a cache
  effect -- the pre-fix arm loads experts it never routed to, which is the same mechanism as §13.4's
  hit rate (57.1% -> 72.0%), not an independent speed claim.
* Thermal is not controlled (`prefill` varied 38 -> 12.8 within one arm).
* Not run: `edge0head` / `graft` carriers, other pools, and the 48-q suite.
* `CGC_IDS_LINEAR_READ` reproduces a defect on purpose; never quote quality or throughput from that
  arm as an engine property.


---

## §15 Handover items raised against §13/§14 (2026-09-17 14:3x)

### 15.1 `DEFAULT_REF` was left pointing at v5 -- carried, now fixed

§13.2 regenerated the baseline but did not move `m123_oracle_gate.py`'s `DEFAULT_REF`, so every gate
invocation WITHOUT `--ref` kept comparing against a reference the fix had invalidated: it reported a
false `M1 5/9 FAIL` on a correct binary. Fixed: `DEFAULT_REF` now points at
`ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl`, with a dated comment recording why this
re-baseline is *not* nominal (unlike v2->v5, which were re-stamps) and what a v5 comparison now
means. `CGC_IDS_LINEAR_READ` is deliberately NOT added to `DIAGNOSTIC_KEYS`, so a reference dumped
with the old read can never compare clean against a fixed binary. Verified by a gate run with no
`--ref`: it resolves v6 and passes (r42).

### 15.2 Evidence status of every pre-fix `M1` pass -- downgraded to "invariance only"

A cross-pool `M1`/`M2` pass before today proves **invariance**, not correctness: the defect was in
the *shared* code path, so every arm and every pool size mis-routed tokens >= 1 in the same way and
agreed with each other. The 2026-09-14 "M1 117/117" belongs to that category and must be quoted as
"the pools agreed", never as "the numbers were right". Only a v6 comparison can support the latter.

### 15.3 §13.3 vs the "+1-3% capacity churn" upper bound: two instruments, different loads

| reading | instrument / load | capacity share of misses |
|---|---|---|
| §13.3 | oracle gate probe: **N=9, one chat turn** | ~0% (4 GiB 1/1303; 6 GiB 0/1047) |
| the other report's §4.6 | a throughput load used for a churn bound | ~49% |

These cannot describe the same load, and §13.3 must not be read as refuting the churn bound: a
9-row single-turn probe is a *short* load, and a cache that is still filling has no capacity
evictions to report. The correct statement is that the two instruments measure different loads; the
adjudication is a same-load, same-build counter comparison at two slot counts (hit/misses/io
counters only -- not t/s), which is also that report's gate 4. Result recorded in §16.


---

## §16 "Did the fix cost prefill speed?" -- measured, and the answer is no

Same binary (14:12), carrier `nail`, pool 8 GiB, `n_predict=96`, 3 requests. The knob from §14.1 is
the control, so the two rows inside each profile block differ ONLY in the ids read.

| chat profile | ids read | accept | prefill t/s | decode t/s |
|---|---|---|---|---|
| default (auto) | pre-fix (`CGC_IDS_LINEAR_READ=1`) | 73.81% | 7.30 | 8.62 |
| default (auto) | **fixed (nb-aware)** | 58.25% | **21.74** | **12.62** |
| `prefill250` | pre-fix | 61.95% | 2.18 | 3.39 |
| `prefill250` | **fixed** | 60.60% | **2.93** | **11.79** |

MTP-off denominators (same binary, same pool): default profile decode 9.82 t/s; `prefill250` decode
9.53 t/s. So **MTP-on >= MTP-off in both profiles** (12.62 vs 9.82; 11.79 vs 9.53), which is M4's
exit line, and the fix is ahead of the pre-fix read on **both** columns in **both** profiles.

### 16.1 Why the absolute prefill numbers are so low, and why that is not a regression

`prefill250`'s prefill column (2.2-2.9 t/s) is **not** a profile verdict: the harness's prompts are
**12-15 tokens**, so that column measures per-request overhead (15 tokens at ~2 t/s is ~7 s of fixed
cost), not prefill throughput. The profile comparison on this same binary and load is 21.74 (default
profile) vs 2.93 (`prefill250`) -- a 7x spread on a metric whose denominator is 15 tokens. Neither
number should be quoted as "prefill speed" without a long prompt; the 2600-token prefill probe is
the instrument for that.

What the table does support: the ids fix does not cost prefill, in either regime, and it is the only
change between the rows it separates.

### 16.2 Why the capacity-churn adjudication (§15.3) is still open

It cannot be read off these runs: the accept harness's server logs contain **zero** `expert_cache`
lines (checked: `grep -c expert_cache` = 0 for both arms), i.e. its server is torn down without the
graceful-shutdown stats dump, so no hit/miss/io counters are captured. The instruments that DO print
them are the oracle gate's server log (short load by construction) and the decode-sweep bench
(prints hit%/miss/io per arm). The adjudication therefore needs a decode-sweep at two slot counts,
same build and same load, reading counters only -- which is exactly that report's gate 4.


---

## §17 MTP attribution audit (raised against §14/§16 while r45 ran)

Two flags were raised: (a) a guard reporting `MTP=1 但 libllama-common.0.dylib 沒有 -DMTP_SUPPORT`,
which would mean artifacts disagree with source and every MTP number needs re-attribution; (b) the
runs were over-subscribed (21222 MiB static vs 16384 MiB physical). Both were checked statically --
no server was started, the box is someone else's.

### 17.1 What the guard actually tests

`scripts/run_server.sh:365`, verbatim criterion -- ONE string in ONE file:

```sh
_cgc_common="$(dirname "$BIN")/libllama-common.0.dylib"
if [ -e "$_cgc_common" ] && ! strings -a "$_cgc_common" 2>/dev/null | grep -q 'MTPDBG mtp_ctor'
```

`BIN` is hard-wired to `$ROOT/src/llama.cpp/build/bin/llama-server` (line 76), so the file is the
build tree's own symlink. `MTPDBG mtp_ctor` lives in `common/speculative.cpp:1312`, inside
`#ifdef MTP_SUPPORT` (first guard at :1310) -- i.e. the string is a proxy for "the guarded MTP
sites were compiled in". The guard's own comment says the criterion was chosen so the gate can also
NOT warn when it should not, because a gate that always fires becomes background noise.

### 17.2 The artifact in use HAD the code -- attribution stands

| artifact | time | criterion hits |
|---|---|---|
| `build/common/.../flags.make` | CXX_FLAGS `-DMTP_SUPPORT` | present |
| `build/common/.../speculative.cpp.o` | 09-16 12:11 | **3 / 3** |
| `build/bin/libllama-common.0.0.277.dylib` (in use before 14:12) | 09-16 12:11 | **3 / 3** |
| `build/bin/libllama-common.0.0.279.dylib` (in use from 14:12 today) | 09-17 14:12 | **3 / 3** |

Every run in this document (oracle gates r34-r37/r42, accept arms r38-r45, the 14:12 binary) ran
against a library with the guarded sites present, so the accept/decode deltas keep their
attribution: the drafting really ran (`MTP=1` -> `draft 180/309`; `MTP=0` -> `draft 0/0`, same
carrier/head), and it ran on the build where the `[CGC MTP fix]` sites exist.

### 17.3 The single warning is unexplained and NOT reproducible -- file it as an instrument defect

It appears exactly once, in `gate_r42_defaultref` (launch 14:27:59), and I cannot reproduce it:

* the guard's own command now prints `FOUND` for the symlink in use;
* it also prints `FOUND` under a minimal env (`env -i PATH=/usr/bin:/bin`) and through the same
  relative `dirname` resolution;
* there was **no build activity** in 14:26-14:30 (0 files under `src/llama.cpp/build` with an mtime
  in that window), so it was not a relink race;
* the accept arms launched minutes later (14:29, 14:34, 14:39) contain **zero** such warnings.

So the warning fired while the criterion was in fact satisfied -- meaning it cannot distinguish "the
code is absent" from "the probe did not run". That is precisely the failure mode its own B13 note
says it must avoid, and it is the reason a flag like this costs more than it returns: it invites a
false re-attribution of correct numbers. The audit above is what settles attribution here; the guard
still needs a self-test.

### 17.4 Which artifacts DO lack it (for dating older measurements)

Criterion hits = 0: `build/bin/libllama-common.0.0.{239,190,182,100}.dylib`,
`Backup/pre_mtp_rebuild_20260916/*`, and the `deploy-harmonyos/macos` copies `0.0.148`/`0.0.140`.
Any measurement attributed to those needs the "no MTP code" reading, not this one.

### 17.5 Over-subscription confirmed in my own runs

`[budget] 靜態需求 21222 MiB（model resident 13030 + pool 8192） vs 實體 16384 MiB` -- printed by
r40 and r43. So every t/s in §14/§16 is from an over-subscribed machine and is quoted **only** as a
same-binary, same-load A/B; the counters (accept counts, hit/miss) are the findings. Also visible in
the r42 server log: `L4 metal pool: 143 slots/layer` at 8 GiB -- the geometry the other report's
§4.6 names.
