# L3 / M1 — the delivery step ledger, and why the gap's non-fill part is 8.5%

**Date** 2026-09-20 · carrier Nail IQ3_XXS-denseIQ4X (blk.40 MTP), MTP on k=3, pool 8 GiB, prefill250
profile, temp 0.4. Evidence: `Backup/cgc_logs/llama_server_20260920_023021.log` (the same run the
work-row in `DECODE_STEP_ROW_POPULATIONS_2026-09-20.md` came from; 107 ntok=4 work rows, segs=41).

## 0. Provenance of every t/s quoted below (read this before citing the ledger)

| number | how it was obtained | source |
|---|---|---|
| **12.57 t/s** | **measured** (llama-bench, delivery anchor) | `prod_profile.py`, `prod25-stream`, 2026-09-20 12:30, NOMINAL throughout, ±2.26 |
| 268 ms | **reverse-derived** from 12.57 and mean_len 3.375 | `CROSS_LINE_STATUS_AND_TARGET_MATH_2026-09-20.md:139` |
| 3.375 | engine median `mean_len`, **G6 arm** (accept median 0.7976, per-task 2.41–4.0) | `Backup/cgc_logs/joint_step_accept_g6.json` |
| 247.98 ms | **measured** work-row step median (ntok=4, n=107, segs=41) | `Backup/cgc_logs/llama_server_20260920_023021.log` |
| **13.61 t/s** | **derived**: 3.375 / 0.24798 — same round, two instruments | `G7_DELIVERY_ARITHMETIC_2026-09-20.md` (§4 lists it as NOT the official reading) |

Two boundaries that must travel with the ledger:

1. **The 58.25% accept family and the 3.375 mean_len are different arms.** 58.25% (180/309,
   post-canon-order, `M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md`) belongs to the 12.57 t/s arm;
   its per-position acceptance implies mean_len ~2.1, not 3.375. Do not multiply or add the two.
2. **247.98 ms excludes the draft forward.** The same round logs 107 shadow rows (segs=2, 2.05 ms,
   1:1 with the work rows) ⇒ a whole round is ~250.03 ms ⇒ 13.50 t/s (inference, not a direct read).

The ledger's internal cross-check does hold: `union 149.24 + gap 95.17 = 244.41` vs `247.98` is 1.4%.

## 1. The ledger (ntok=4 work rows, medians)

```
total 247.98 = wait 159.77 + cb 74.18 + submit 10.09        (residual +3.94)
GPU: gpu_busy_sum 181.56   union_sum 149.24   gap 95.17
```

**`gpu_busy_sum` is NOT a floor — do not subtract from it.** Its own definition
(`ggml-backend.cpp:2049`) is `Σ(end-start) over the segment's n_cb+1 buffers, **overlap counted
twice**`; the delivery row's ratio is `181.56/149.24 = 1.22`, i.e. it is inflated by exactly that
overlap. The honest lower bound on a step is **`union_sum` = 149.24 ms** (the spans are chained, so
they cannot be hidden from each other), which puts the ceiling of *any* overlap work at
`3.375 / 0.14924 = 22.61 t/s` — not the 18.59 that this ledger's earlier draft derived from
`gpu_busy_sum`. That number is withdrawn in `docs/L3_ASYNC_FILL_PLAN_2026-09-20.md` §6.

`gap` is not a free-floating idle: the engine's own definition (`ggml-backend.cpp` — sampled in
`hook_seg`, `gt_prev_end → g[2]`) is *previous segment's GPU end → this segment's GPU start*, i.e.
the host-serial window. So the built-in cross-check applies:

```
gap 95.17  vs  cb + submit 84.27   ->  unexplained +10.90 ms/step   (1.13x)
```

⇒ **The gap contains almost nothing else.** It is the fill (`cb`), the per-segment scheduler call
(`submit`), and ~0.27 ms/segment of commit→GPU-start latency (×41 segments).

| what | ms/step | share | removable? |
|---|---:|---:|---|
| `cb` — the union's bytes: pool hits + disk/page-cache misses | 74.18 | 29.9% | no: bytes. 8 GiB pool over a 12.2 GB expert set (see §3) |
| `submit` — `ggml_backend_graph_compute_async` per segment ×41 | 10.09 | 4.1% | only by not paying it 41× (scheduler re-split per view) |
| launch/pickup — commit → GPU first buffer start ×41 | 10.90 | 4.4% | only behind a GPU-side fence (Metal shared event) |
| `wait` — CPU polling for segment completion | 159.77 | 64.4% | GPU executing (`union 149.24`); this is the work |

**M1's prize, stated honestly: `submit` + pickup = 21 ms/step = 8.5%** — 13.61 → 14.8 t/s. Not the
+40~50% priced earlier (that number had the *instrumented* arm's `cb` share, 19.4%, as denominator;
in the delivery work-row `cb` is 29.9%).

## 2. Why the overlap cannot be had without a fence

The M1 design put the await on the first consumer's eval callback. It does not exist:

1. `hook_seg(i)` fires on the **argsort(il)** that ends segment i; the union's consumer,
   `mul_mat_id(il)`, is the **first node of segment i+1**. Everything else in segment i+1
   (attention(il+1), router(il+1)) depends on MoE(il)'s output. There is no union-independent work
   ahead of the consumer to submit early.
2. The dispatcher's three callback sites in `ggml-backend.cpp` are all `ask=false` (after the node
   computed); `ask=true` exists only on the generic path.
3. "Submit earlier" = `CGC_SUBMIT_AHEAD=1`, which this repo labels **racy**. Measured: step
   167 → 50 ms — with `non-finite values: 993280 of 993280` and output `文摘…`. Not a slow path;
   a wrong one.

So the only safe form of the 21 ms is: commit segment i+1 early with a Metal shared-event wait that
the fill's preads signal. `ggml-metal` does not expose that today ⇒ new device-level API in the
Metal backend, ~2–3 h, bounded upside 8.5%, and the fail-closed property comes from the event, not
from timing.

## 3. The bytes side, for the first time exact (from the GGUF, not from notes)

Read from the carrier's own header (`qwen35moe`):

```
block_count 41 · embedding_length 2048 · expert_count 256 · expert_used_count 8
expert_feed_forward_length 512 · nextn_predict_layers 1
expert tensor shape = [2048, 512, 256] (gate/up), [512, 2048, 256] (down)  -> one expert = the
  3-projection triple the loader's per_slot measures
capacity = budget / (41 x per_slot)             [llama-model-loader.cpp:1188]
budget 8192 MiB -> 143 slots/layer              <- engine's own census line
budget 4096 MiB ->  71 ;  budget 10240 MiB -> 179   <- same line in the sweep arms
=> per_slot ~ 1.47 MB, and capacity is LINEAR in the budget knob
coverage = 143/256 = 56% of each layer's experts; pool fills 5976 slots (min 143, blk.40 = 256)
```

⇒ `LAYER_CAPS ... total 5976 slots (min 143/layer)` is **not a sizing bug**: the number is the
budget arithmetic itself. Misses stay at ~38% of requests (6961/18306) and 78% of them are first
touches, which no predictor can reach (that is L1/L2's closure, now explained by geometry rather
than by policy).

One question is **open and cheap to close** (do not quote either way yet): capacity is sized from
the **max** per-slot across layers while each layer's region is sized by its **own** per-slot
(`LLAMA_EXPERT_CACHE_LAYER_CAPS` already allows per-layer values), and the measured pool footprint
is `resident=6430.62 MiB` (L2 control arm) against an 8192 MiB budget. If those two facts mean the
budget is not fully spent, capacity could rise without a bigger budget; if the per-slot *max* is
what the census charges every layer, the pool is exactly the budget and the only lever left is
changing the budget (and therefore RSS on a box already over-committed) — i.e. a quantization
decision. The discriminator is one printed line: total pool bytes at init. My earlier arithmetic in
this section claimed the whole expert set is ~12.2 GB and that the pool "cannot hold it" — that
number came from a hand-multiplied per-slot and is **withdrawn** until the offsets-based total is
measured.

## 4. Widening the verify batch is negative (this retires the k lever)

Same run, work rows, segs=41:

| ntok | step ms | per verified token | gpu_sum | cb |
|---:|---:|---:|---:|---:|
| 3 | 197.47 | 65.8 | 156.79 | 55.04 |
| 4 | 247.98 | 62.0 | 181.56 | 74.18 |
| 8 | 605.36 | 75.7 | 472.86 | 168.68 |

4 → 8 tokens costs **2.44×** the step: nothing amortizes. k=3 is at a local optimum, and the
"more tokens per step amortizes the round trips" premise is false in this regime (it holds only in
the sense that `wait` grows sub-*linearly*: 160 → 410 for 2× tokens — but `cb` grows 2.27× and
dominates).

## 5. What this leaves (the only two sides with numbers)

1. **GPU-occupied wall: 149.24 ms/step for 4 tokens = 37 ms/token.** The expert set touched per
   token is ~1.2 GB (56% resident + misses) but only ~0.6 GFLOP of math: the step is
   launch/latency-bound at ~2.5% of the memory-bandwidth bound (~14 ms/token). 25 t/s at
   mean_len 3.375 needs a 135 ms step — *below* today's GPU-occupied 149 ms, so **this side must
   move for 25 t/s to exist at all** (line A's G4 / kernel+dispatch, not the gap).
2. **Accept / mean_len (M4).** The step cost is already paid; accept moves tokens, not time.
   accept 0.58 → 0.80 is +40% tokens per step = 13.6 → ~19 t/s with no engine change. That is
   the largest implementable lever that is still in this line's reach.
