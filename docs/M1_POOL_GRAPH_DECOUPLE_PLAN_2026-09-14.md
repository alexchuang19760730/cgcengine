# M1 — pool/graph decoupling: plan, and two corrections to the early-verification claims

Date: 2026-09-14 · Branch: `demo/sweet-spot-windows-fix` @ `b699db191` · Box: MacBook Air M4 16 GB

Sources being reconciled: `scripts/check/m1_early_verification/` (three early tests, untracked)
vs the actual code, plus the M0 measurements in `DECODE_PROFILE_AND_UNION_OVER_SLOTS_2026-09-13.md`.

---

## 0. STATUS 2026-09-14 (measured, this binary)

**§3.1 (the decode-side gather slab) is implemented and verified. M1's first two exit conditions
are met; M1 as a milestone is NOT complete** — its work items 1 and 2 (loader keeps the expert
tensor at full width, graph splits by phase) have not been started, and M2 depends on them.

| exit condition | state |
|---|---|
| M1/M2/M3 = 100% @ 4/6/8/10 GiB, incl. `union > slots` | **✅ 117/117 on all of 2/4/6/8/10 GiB** (`CGC_POOL_MAX_TOKENS=8` fixed on both sides; reference regenerated with the same binary) |
| `union-routable` PASS | **✅** `PASS mode=gather ... wide route: layers=8 steps=16/65` at 2 GiB, with `buffer is nil` = 0 |
| prefill chunk 2048 accepted, no OOB/NaN/nil | ⏳ needs work item 2 |
| RSS @ 4/6/8/10 GiB, slab reported separately | ⏳ total RSS measured (4.16/5.28/6.49/7.88/9.32 GB), slab not split out |
| decode must not regress | ⏳ not measured this round (`--gates-only` leaves tok/s empty) |
| work item 1: loader, full-width expert tensor, independent pool buffer | ❌ not started |
| work item 2: phase split (`T ≥ 512` → whole-layer slab) | ❌ not started |
| §7 pool-path gate off-by-one (`n_slots` → `usable_slots`) | ⏳ one line, and only safe **now** that the slab exists |
| **cap invariance** (same pool, different `CGC_POOL_MAX_TOKENS` → M1 = 100 %) | **❌ FAILS — measured 3/82 this round, see §6.** Not an exit condition before; it is one now |

Reproduce:

```bash
python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 8 --dump-oracle \
    --pool-cap 8 --gates-only --kill-existing --force          # reference
python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 2,4,6,10 --pool-cap 8 \
    --oracle-ref Backup/knifeedge_matrix/oracle_iq3_pool8gb.jsonl --gates-only --force
```

`--pool-cap 8` is load-bearing: `--pool-cap auto` probes for a cap that keeps even the smallest pool
on the pool path, i.e. **it steers away from the very case this gate exists for**.

Two more things worth remembering, both of which cost a run today:

- **A gate must be able to produce the evidence it judges.** The 2 GiB cell scored 117/117 while
  `gate-union-routable` reported `FAIL mode=unreachable layers=0 steps=0/0` — not because the wide
  route was not taken, but because `CGC_UNION_LOG` had never been set, so the `gather=N/M` lines the
  gate reads did not exist. The harness now sets it unconditionally.
- **The launcher's `env` allowlist silently drops unknown `CGC_*` variables**, so "the knob did
  nothing" and "the knob was never set" look identical in the log. Every new diagnostic flag must
  land in the allowlist in the same commit.

---

## 1. Two of the three early-verification claims do not survive contact with the code

### 1.1 "Accumulation order follows slot index, not expert id" — **already false**

`llama-context.cpp` builds the gather path's union like this:

```cpp
std::sort(uni.begin(), uni.end());
// the L3-B gather buffer is laid out in sorted-union order, so remap must use the expert's
// position in the sorted union (insertion-order umap is NOT valid after the sort).
std::unordered_map<uint32_t, uint32_t> uidx;
for (size_t k = 0; k < uni.size(); ++k) uidx[uni[k]] = (uint32_t) k;
```

and then writes **position in the sorted union** into the remap leaf:

```cpp
const uint32_t slot = uidx.count(e) ? uidx[e] : 0;   // position in the SORTED union
rd[i + j * n_expert_used] = (int32_t) slot;
```

So the gather buffer is laid out in ascending expert-id order, and the index written is a
function of the **routing set**, not of the pool. Separately, `mul_mat_id` sums each token over
its `top_k` slots **in the argsort order given by the ids array** — which is canonical in *both*
the pool and the gather path. There is therefore no known slot-dependent summation order, and
the measured 4/6/8/10 GiB invariance (117/117 M1) is consistent with that.

`test_canonical_gather.py` is a **scaffold, not a test**: it looks for a
`CGC_CANONICAL_GATHER_ORDER` env var that does not exist and prints "當前代碼可能尚未支援" when it
does not find it. Its premise was taken from the roadmap text rather than verified, which is why
it disagrees with the code.

**Consequence for M1: "canonical gather order" is not a work item.** The work item is a *gate*
that asserts it (see §4), not an implementation.

### 1.2 "union > slots aborts" — **true only in one branch; the branch that actually runs is silent**

The abort does exist (`llama-expert-cache.cpp:687` and `:800`), but both are conditioned on
`no fill is in flight`:

```cpp
if (!in_flight) {
    fprintf(stderr, "FATAL ensure_batch layer=%u: %zu distinct experts exceed the %u usable "
                    "pool slots and no fill is in flight — cannot assign; aborting\n", ...);
    abort();
}
```

Measured at 2 GiB (`n_slots=35`): **0 aborts, 234 × `buffer is nil`, M1 2/42, first divergence at
step 2 with `sum` 55 % apart.** So in practice the region does not abort — it produces **silently
wrong numbers**, which is strictly worse than the abort the early test predicted.

## 2. The real mechanism of the failure — it is structural, not numerical

```cpp
std::vector<std::vector<uint8_t>> cache_gather_buf;      // llama-context.h:367 — plain HOST memory
...
wt->data = cache_gather_buf[kind].data();                // llama-context.cpp:4419
```

The gather path repoints an expert tensor that lives in a **Metal buffer** at a **host
pointer Metal has never been told about**. Metal then fails its buffer lookup —
`ggml-metal-device.m:2026: error: tensor '%s' buffer is nil` — and reads nothing. The codebase
already knows this: `llama-model-loader.cpp:1115` records "the 33-slot run logged 585
`buffer is nil` errors on blk.1..39 expert tensors".

So the union>slots region is **broken by construction**: there is no legal place to put the
weights. That is exactly why the region needs a real slab, and it is why the early test 3 result
(Metal can allocate a 280 MB+ contiguous buffer and batch-matmul a 768 MB whole layer) is the
useful one of the three.

## 3. What M1 actually has to build

Two things, in this order.

### 3.1 A Metal-backed scratch slab, capacity fixed by a constant (not by the pool)

Replace the host `cache_gather_buf` with a slab allocated from Metal's buffer type, sized to a
**fixed capacity `C` experts**, chosen independently of the pool budget. `C = 64` for M1 — the
measured size is in §3.1.1 — **89 MiB live / 157.5 MiB allocated on Nail at `C = 64`**, which is
larger than the single-stride arithmetic below suggests because the GGUF is MIXED QUANT (the slab
is sized per kind by that kind's max stride; §3.1.1 carries the correction) — and `C`
must be a compile-time constant for the reason given just above.

> Correction (2026-09-14): this paragraph previously said "64 × 1.122 MB ≈ 72 MB per kind, ≈ 288 MB
> across four kinds". Both halves were wrong — 1.122 MB is the sum over the three projections, not a
> per-kind slice, and kind 3 does not exist in either model. See §3.1.1 and §3.1.3.

**Why `C` must be a constant and not `usable_slots`:** if the union is processed in several
passes, each token's sum becomes an outer sum over passes. The tree shape is then a function of
the group boundaries, and if those boundaries move with the pool the rounding moves with them —
which is exactly the M1 failure the gate exists to prevent. Fixing `C` makes the grouping a
function of the routing set alone.

### 3.2 A pass loop, only on the wide-union path

```
n = |sorted union|
if n <= usable_slots:            # today's pool path, unchanged
    one pass, remap by pool slot
else:                            # the region that is broken today
    for p in range(ceil(n / C)):         # pass order = ascending expert id, fixed by C
        slab = sorted_union[p*C : (p+1)*C]
        fill slab from pool (resident members) then from disk (the rest)
        remap each token's ids to their position within THIS pass's chunk
        dispatch the layer's MoE and ACCUMULATE the partial output
```

The inner sum stays in argsort order; the outer sum is in pass order; both are pool-independent.
This is the only part of M1 that is a graph change, because the MoE needs an accumulator that
spans passes.

Compatibility note: prefilling with `C = 256` (union saturated) is the M2 whole-layer path, and
its boundaries are fixed at 256, so it is canonical too — the two milestones share this primitive.

### 3.1.1 Measured geometry, and the real slab size (2026-09-14)

Read straight out of the GGUFs with `scripts/gguf_retensor.py list` at `blk.1` (256 experts), for
the two models this branch actually runs — the launcher's `auto` profile picked Nail, Edge0 is the
target model once its weights are in:

| kind | tensor | type | bytes / expert (Nail) | bytes / expert (Edge0) | present? |
|---|---|---|---|---|---|
| 0 | `ffn_gate_exps.weight` | iq2_s / q4_0 | 335,872 | 589,824 | ✓ both |
| 1 | `ffn_up_exps.weight` | iq2_s / q4_0 | 335,872 | 589,824 | ✓ both |
| 2 | `ffn_down_exps.weight` | iq3_s / q4_0 | 450,560 | 589,824 | ✓ both |
| 3 | `ffn_gate_up_exps.weight` | — | — | — | **absent in both** |
| | **sum / expert** | | **1,122,304 B = 1.0702 MiB** | **1,769,472 B = 1.6875 MiB** | |

Two facts that change the design:

- **kind 3 is dead code in both models.** Neither GGUF has `ffn_gate_up_exps`, so the graph never
  emits `ffn_moe_gate_up`, `cache_ffn_tensors[il][3]` stays `nullptr`, and the `for (kind < 4)` loop
  only ever repoints **three** tensors. The slab is 3 buffers, not 4, and any code that assumes 4
  live kinds is over-allocating.
- **Cross-check:** 287,309,824 B ÷ 256 = 1,122,304 B exactly, which reproduces the "1.122 MB/expert"
  figure the earlier miss-path analysis used. The geometry is consistent across two independent reads.

**Slab size = `C × sum`:**

| C | Nail IQ3_XXS | Edge0 Q4_0 | what C corresponds to |
|---|---|---|---|
| **64** | **68.5 MiB** | **108.0 MiB** | **M1: `cap × top_k` = 8 × 8 (the §2.1 measured ceiling)** |
| 128 | 137.0 MiB | 216.0 MiB | `cap` 16 |
| 176 | 188.4 MiB | 297.0 MiB | `cap` 22 = the 10 GiB geometric ceiling |
| **256** | **274.0 MiB** | **432.0 MiB** | **M2 whole-layer**; also the absolute ceiling (union ≤ `n_expert`) |

> **CORRECTION 2026-09-14 (supersedes the table above).** The table assumes **one stride per kind**,
> and that is false: these GGUFs are **mixed quant**. Read straight out of `blk.*`, the *distinct*
> `(kind, stride)` combinations are **8** on Nail — gate/up are `iq2_s | iq3_s | q2_K`
> (335,872 / 450,560 / 344,064 B per expert) and down is `iq3_s | iq4_xs` (450,560 / 557,056).
> The rule `C × sum` therefore has no single "sum" to multiply.
>
> What the implementation actually does (and what the measurements below are from): **one slab per
> kind, sized by that kind's MAX stride over the layers that request it** — a larger slab always
> serves a smaller request, because `ggml_nbytes = union × this-layer stride ≤ cap × max_stride`.
> That also avoids a real bug the `(kind, stride)` keying produces: a supersede allocates a fresh
> buffer and frees the old one while *other layers' live tensors still point into it*.
>
> | kind | slab at `C = 64` (per-kind max stride) | measured alloc lines (`CGC-GATHER-SLAB`) |
> |---|---|---|
> | 0 `ffn_gate_exps` | 27.5 MiB (stride 450,560) | 20.50 MiB then 27.50 MiB |
> | 1 `ffn_up_exps` | 27.5 MiB (stride 450,560) | 20.50 MiB then 27.50 MiB |
> | 2 `ffn_down_exps` | 34.0 MiB (stride 557,056) | 27.50 MiB then 34.00 MiB |
> | | **live boundary 89.0 MiB** | **cumulative allocated 157.5 MiB** |
>
> **So the honest M1 cost is 89 MiB live / 157.5 MiB allocated on Nail at `C = 64`**, not 68.5 and not
> 288 MB. The 68.5→157.5 gap is a known wart: supersedes are not reclaimed until context
> destruction. Fix: size each kind's slab from the **whole model's** max stride at first touch, or
> free superseded slabs in the graph-build restore path (the only place it is safe). M2's `C = 256`
> scales from these numbers: ≈ **300 MiB** live (**600 MiB** double-buffered) on Nail, and that is
> the figure to budget — 15% of a 4 GiB pool.

### 3.1.2 What M1's RSS actually costs — the correction

**§3.1's "≈ 288 MB" was ~4× too high.** The error was a units mistake: 1.122 MB is the sum over the
three projections, not the per-kind slice, and there is no 4th kind. **The correct M1 number is
68.5 MiB (Nail) / 108.0 MiB (Edge0).**

The honest range is **0 to +68.5 MiB (Nail) / +108.0 MiB (Edge0)**, not a flat +68.5:

- `cache_gather_buf[kind]` is a `std::vector<uint8_t>` that grows to `uni.size() * exp_bytes` and
  **never shrinks**. So in a config where the wide-union path fires (2 GiB), the host already holds
  up to 68.5 MiB — a Metal slab of the same size is then a *relocation* on unified memory, not new
  residency.
- But in the **default 10 GiB config the gather path essentially never fires** (the pool path always
  wins, §2.1's second block: `gather=0/8`), so those host vectors stay tiny and the slab would be
  genuinely new resident Metal memory.

Therefore: **allocate the slab lazily, on the first wide union, per kind.** A default-config server
then pays nothing, and the cost appears exactly in the configs that need it. Either way 68.5/108 MiB
is ≤ 1/5 of the ±0.5 GiB budget in §4, so M1's RSS exit condition is not at risk — but the earlier
number would have made M2 look cheaper than it is.

**The genuinely new RSS belongs to M2, not M1**: `C = 256` is 274 MiB (Nail) / 432 MiB (Edge0), and
that is the figure to budget when the whole-layer path lands.

### 3.1.3 C recommendation

**`C = 64` for M1; `C = 256` for M2; the rule is `C = cap × top_k`, clamped at `n_expert` (256).**

- `64` is the correct *minimum* for M1 because §2.1 **measured** the union ceiling to be exactly
  `cap × top_k = 8 × 8 = 64` — not an estimate, and not the 28.7 mean that earlier notes quoted.
- `C` must not be `usable_slots`-derived: a runtime `C` would make the pass boundaries move with the
  pool, which is precisely the M1 failure the invariant gate exists to prevent (§3.1 above).
- Practical note for whoever implements it: because `C = 64` and `cap = 8`, the pass loop of §3.2 is
  provably unnecessary in M1 (`ceil(64/64) = 1`), so §3.2 can land with M2 without leaving an
  untested path in M1.

### 2.1 The union ceiling is `cap × top_k` and it IS reached (measured 2026-09-14)

`CGC_UNION_LOG=1` was added to `llama-context.cpp` (per-layer union + chosen path + that layer's
usable slot count) and to `scripts/run_server.sh`'s `env` allowlist, turning the §5 inference into
an observation. One 36-token request at 2 GiB (`n_slots=35`, `usable=34`, MTP mode):

```
CGC-UNION: layer=1 union avg=41.8 min=16 max=64 of usable=34 (188%)  [WIDE: exceeds usable] gather=4/9
CGC-UNION: layer=2 union avg=43.0 min=16 max=64 of usable=34 (188%)  [WIDE: exceeds usable] gather=4/8
CGC-UNION: layer=3 ... max=64 ... gather=4/8     (layers 4..8 identical in shape)
```

and in the same run, `buffer is nil` = **468**. The chain is therefore observed end to end:
`union = 64 > usable 34 → gather path (4 of 9 sweeps) → host cache_gather_buf → Metal nil →
wrong values`. Evidence file: `Backup/cgc_logs/union_evidence_2gib_20260914.txt`.

Two numbers that change the design:

- **64, not ~29.** The earlier "union is only 28.7/256 = 11 %" figure is the *mean* over layers
  (avg 41.8–43 here). The **max is exactly `cap × top_k = 8 × 8 = 64`** at layers 1–8 — the
  ceiling is reached, so the slab's fixed capacity must be **`C >= 64`**, not 48. §3.1's "C = 64"
  is the minimum correct value, and its "≈ 72 MB per kind" figure follows from it.
- **The wide layers are again 1–8, not the tail.** That matches the churn data (`worst=layer 1
  distinct=253`, `layer 2 distinct=250`) and is a second, independent reason prerouter cannot
  help here: its head starts at layer 7.

One caveat on the run: the second `CGC-UNION` block in the same log shows `max=32 of usable=34
(94 %) gather=0/8` — i.e. after warm-up the union settled to 32 (4-token ubatches) and took the
pool path. So the wide-union region is entered during the 8-token prefill ubatches, not during
steady-state 4-token verify, which is consistent with `union == n_tokens × top_k` under
`cap = 8`.

## 4. Gates for M1 (the milestones' invariant gate, plus the new case)

1. `union-fit` → `union-routable`: `union ≤ usable_slots` **or** the pass loop ran.
2. **A new M1/M2 case with `union > slots`.** Today it fails (2/42), so this gate cannot be
   tightened before §3 lands — it is the acceptance test for §3, not a regression it must avoid.
3. **Canonical-order assertion** (replaces the attachment's test 1): run the *same* prompt at two
   pool sizes that both take the wide-union path, and require **M1 = 100 %** between them. That is
   the direct empirical form of "the order does not depend on the pool". Do this once `§3` fixes
   the nil, because the nil currently masks any order effect.
4. Existing: M1/M2 100 % at 4/6/8/10 GiB with a regenerated reference; RSS ±0.5 GiB; decode must
   not regress.
5. **Cap invariance** (added 2026-09-14, see §6): the *same* pool, two different
   `CGC_POOL_MAX_TOKENS`, must give M1 = 100 %. It currently fails at 8 GiB: 3/82 aligned rows.
   This is the sharpest form of gate 3, because it moves the pool-layout variable without moving
   the pool at all.

## 5. Honest status

- **Nothing in §3 is implemented.** This document is the plan plus the two corrections; the
  corrections are the part that changes what the next session should do first.
- ~~The causal chain was only inferred.~~ **Closed 2026-09-14 — see §2.1.** `union` and the chosen
  path now log per layer (`CGC_UNION_LOG=1`), and the chain is *observed*: union max = 64 >
  usable 34 → gather path → 468 `buffer is nil` in the same run. The reference arm's degenerate
  text is still true, and is the reason §4's acceptance test belongs on logits (M1) rather than on
  the text.
- **New finding, §7:** a one-expert-wide off-by-one in the pool-path gate, live in MTP mode,
  deliberately left unfixed until the slab lands.
- `scripts/check/m1_early_verification/` is untracked and was not authored here; it is left alone.

---

## 6. The pool cap *is* the prefill chunk size — and the invariant is cap-conditional (measured 2026-09-14)

### 6.1 The coupling nobody had written down

`llama-context.cpp` caps `n_batch` to `cgc_pool_max_tokens()` whenever the L4 pool is active:

```cpp
if (model.expert_cache_pool_capacity > 0 && cparams.n_batch > 1) {
    const uint32_t pmax = cgc_pool_max_tokens();
    if (cparams.n_batch > pmax) cparams.n_batch = pmax;   // -> n_ubatch = pmax too
}
```

The reason given in the comment is correct and load-bearing: the loader **shrinks each expert
tensor's `ne[2]` to the pool capacity**, so a batch wider than the cap would read raw router ids
(`0..n_expert-1`) against a capacity-slot tensor → OOB → NaN. Two consequences:

- **The prefill chunk size *is* `CGC_POOL_MAX_TOKENS`.** Not `-ub`, not the profile's `-b`: those
  are overwritten by this cap. Grepping a server log for `ntok=` shows prefill in 8-token chunks at
  the default (two consecutive 8-token batches at `pmax=7` and `pmax=15`), and the harness prints
  `[cap] fixed cap=N` beside it.
- **The "prefill computes over the full expert weights" path is unreachable** while the shrink is
  on (the comment in `expert_cache_on_topk` describes it as live). That path cannot exist until
  work item 1 (full-width tensor + independent pool buffer) lands — the same dependency M2 has.

### 6.2 The measured curve — bigger chunks are *worse*

8 GiB pool on Nail IQ3_XXS-denseIQ4X, MTP off, 3 requests of ~100 prompt tokens each, same binary,
`Backup/cgc_logs/m1_cap_sweep_evidence_20260914.txt`:

| `CGC_POOL_MAX_TOKENS` | prefill t/s | decode t/s | note |
|---|---|---|---|
| 2 | 20.03 / 17.23 / 15.09 | 9.6 | **aborts at load with MTP on** |
| 4 | 9.66 / 9.19 / 9.24 and 10.19 / 9.20 / 9.17 | 8.1–9.5 | two independent runs |
| 8 (default) | 8.55 / 8.31 / 8.30 and 6.48 / 8.24 / 8.35 | 7.3–7.6 | reproducible to ~1 % |
| 16 | 6.00 / 6.19 / 5.90 | 6.9 | −28 % |
| 32 | **0.37** (190.69 s for 71 tokens) | — | killed; slab C=256 = **356 MiB** |

So the union is a **working-set lever, not an amortisation lever**: `union(n)` grows
super-linearly while the resident pool is fixed (142 usable at 8 GiB), so by 32 tokens the per-step
footprint no longer fits and every expert is a cold fill. The earlier hypothesis — "a bigger chunk
amortises one union over more tokens, so prefill rises with the cap" — is **falsified**.

### 6.3 The floor is architectural, not a knob

`cap = 4` (and `2`) abort at load:

```
llama-batch.cpp:609: GGML_ASSERT(n_ubatch > n_keep_tail) failed
```

`n_keep_tail = n_rs_seq + 1` (llama-memory-hybrid.cpp:89) — the recurrent/linear-attention rollback
window of this hybrid model — so `n_ubatch ≥ 5` is required on this architecture regardless of MTP.
With MTP the same bound is implied again by the verify batch (`n_max + 1 = 4`). `cap = 5` loads and
runs; `cap = 4` and below cannot.

### 6.4 And the +13 % is **not free**: cap invariance fails

A `cap = 5` oracle dump at the **same 8 GiB pool** compared against the `cap = 8` reference
(`Backup/knifeedge_matrix/oracle_iq3_pool8gb_cap5.jsonl` vs `oracle_iq3_pool8gb.jsonl`, compared by
`(step, token_idx, node, ctx_type)`):

| gate | result |
|---|---|
| M1 (`row_fnv1a64` byte-identical) | **3/82** |
| M2 (`argmax_token`) | **15/82** |
| M3 (top-k id set) | **3/82** |
| max abs(sum) delta | 353 991 |

First divergence at step 3 (steps 0–2 identical), and the `cap = 5` preflight **degenerated**:
`echo=True`, `finish=length`, content looping `15+27=42 / ### 解釋：` — the `cap = 8` baseline
answers `15+27 等於 42。` and stops.

**This is a correction to §1.1.** "No known slot-dependent summation order" is true *across pool
sizes at a fixed cap* (which is the 117/117 measurement), but **false across caps at a fixed pool**:
moving the chunk changes which union each step gathers, and the logits move with it. The M1
invariant is therefore **cap-conditional today, and that was never gated**. Note the confound that
makes this evidence rather than proof: the candidate session also generated different text (the
loop), so the 82 aligned rows are aligned *up to* the divergence, not throughout. The mechanism to
test next is the one §1.1's own code shows: the pool path indexes by **slot** and the wide path
indexes by **position in the sorted union**, so a step that changes path (or union composition)
changes the per-token summation order; float addition is not associative.

### 6.5 What this changes for the plan

- **Work item 2 (union gather) does not feed prefill as a speed lever.** The binding constraint on
  prefill throughput is the pool's working set, so widening the chunk makes things worse; the only
  structure that can beat it is M2's whole-layer sequential read (read the layer's experts once, in
  order, for a large chunk) — which needs work item 1's full-width tensors. This round gives M2 its
  first live memory number to budget against: `C=256` = **356 MiB** measured (110 + 110 + 136 MiB
  for kinds 0/1/2, Nail).
- **The cheapest real prefill number on the table is now `cap = 5` (+13–15 %) — and it is
  inadmissible** until the reduction order is made canonical, because it fails §4's new gate 5.
- **Gate 5 is cheap**: two oracle dumps at one pool, no code change, ~2 min. It should become a
  first-class harness case rather than something run by hand.

---

## 7. A live off-by-one in the pool-path gate (found 2026-09-14, deliberately not fixed yet)

`llama-context.cpp` gates the pool path on the **raw** slot count:

```cpp
const uint32_t n_slots = llama_expert_cache_slots_per_layer_l(cache, (uint32_t) il);
if (llama_expert_cache_pool_active(cache) && uni.size() <= n_slots) {
```

But `llama_expert_cache_slots_per_layer_l()` returns `slots_l()` (raw), while the reserved ZERO
slot is the **last** one (`llama_expert_cache_zero_slot()` == `slots_l() - 1`,
`llama-expert-cache.cpp:178-183`), and `zero_slot_enabled()` is true exactly when
`CGC_VERIFY_DECODE` or `CGC_DRAFT_DECODE` is set — i.e. in MTP mode, which is the launcher's
default (its banner prints `MTP / draft-mtp`).

So in MTP mode the pool path can only hold `usable = n_slots - 1` real experts while this gate
admits `union <= n_slots`: a **one-expert-wide window (`union == n_slots`)** that reaches
`ensure_batch` and **aborts** (`distinct experts exceed the usable pool slots`). This is the
mechanism behind the earlier "caps32 abort" observation.

Measured consequence: at 2 GiB `usable = 34`, so the window is `union == 35`. This prompt did not
hit it (its unions were 16..64, i.e. either <= 34 or >> 35), which is exactly why it stayed
invisible for so long.

**The fix is one line — `llama_expert_cache_usable_slots()` instead of `slots_l()` — and it must NOT
land before §3.1.** A union in `(usable, n_slots]` would then be routed to the gather path, which
§2 shows is broken by construction (host buffer → Metal nil). A loud abort is the lesser evil until
the slab exists. So the gate change ships *with* the slab, and until then `CGC_UNION_LOG` reports
`usable` beside `max` so the window is at least visible (`[WIDE: exceeds usable]`).

### 7.1 Order of work this implies for M1

1. Slab + pass loop (§3), with `C >= 64` fixed by a constant.
2. **Then** the gate off-by-one (§7) — one line, now safe.
3. **Then** the M1/M2 `union > slots` case can be tightened to an acceptance test (§4.1/§4.2),
   because it will finally be a case that passes.
