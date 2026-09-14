# Cap invariance is now a harness gate (knifeedge_matrix.py)

## The claim it tests

`CGC_POOL_MAX_TOKENS` (the cap) sets the union width `cap x topk` per layer. Changing it changes
**which** experts are resident and **in what order** the MoE accumulates -- the pool *layout* --
while leaving the pool *size* fixed. A cap change is therefore a legitimate configuration of the
same engine, and it must be **numerically inert**. Pool *size* is allowed to change speed; pool
*layout* is not allowed to change a single bit.

> ### CORRECTION 2026-09-14 — the premise above is only half true
>
> The cap does **not** only change the layout. It is clamped onto `n_batch`, and `n_ubatch` follows
> from it, so a cap change also changes **how many tokens each prefill forward pass processes**.
> Measured (`docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md`): three arms differing *only* in that
> partition — `cap5` (5,5,5…), `cap8` (8,8,8…,4,4,4) and `cap8` with `-ub 5` (5,3,5,3…) — produce
> three **different** completions, while a **fixed cap across pools 2/4/6/8/10 GiB is bit-identical
> (M1 41/41)**. The divergence is already in the last prefill output (`pos=59`), where layer 0's
> routing is identical in all three arms and layer 1's is a three-way split; `gather=0` in every
> layer of every arm, so the pool↔gather switch and the expert-reduction order (already *k*-ordered,
> not slot-ordered) are both innocent.
>
> So the two invariances must be read separately, and only one of them is a requirement:
>
> * **pool-layout invariance** — *same cap*, pool 2/4/6/8/10 GiB → **PASSES**. This is what
>   "4 GB vs 8 GB must be byte-identical" asks for.
> * **cap invariance** — *same pool*, different cap → **FAILS**, because it is a *partition* change,
>   and the model's forward pass is not partition-invariant (batch-shape-dependent reduction order →
>   ulp → discrete routing flip → ~1-logit output difference). No expert-cache change can fix that.
>
> The gate now records and prints each arm's prefill step partition and states which knob moved
> (`pair["attribution"]`), so a FAIL says *what changed* instead of implying layout damage. The
> verdict thresholds are unchanged.

Until now that was a hand procedure with a one-shot finding and no way to re-run it:

> 2026-09-14, by hand: an 8 GiB pool at `cap=5` was 13-15% **faster** than `cap=8`, and scored
> M1 3/82 against the `cap=8` reference. Same binary, same pool, same model. So the speed lever
> was real, and so was the numeric damage.

Nothing cheaper distinguishes that pair from a clean config: the speed number looks good, the
`buffer is nil` scan is empty, the union-fit arithmetic passes (48 <= 142 usable slots) and the
no-think scaffold renders fine. Only the logits say anything -- which is why it has to be a gate.

## Running it

```bash
# one pool, two caps, two launches
python3 scripts/check/knifeedge_matrix.py --cap-invariance \
    --models iq3 --pools 8 --cap-invariance-caps 6,8

# the gate's own determinism control: two INDEPENDENT runs at one cap.
# Without this, an M1 failure between two caps could always be "the engine is not repeatable".
python3 scripts/check/knifeedge_matrix.py --cap-invariance \
    --models iq3 --pools 8 --cap-invariance-caps 8,8
```

Dumps are cached and reused only while their own sidecar still proves they describe *this*
computation (same cap, same engine stamp, same weights); otherwise they are re-dumped, and
`--force` re-dumps unconditionally. Every pair writes
`Backup/knifeedge_matrix/capinv_<model>_pool<gb>gb.json` (plus the two realigned dumps and the
raw comparison reports). `--summary` prints the cap section alongside the pool matrix, because
it is the verdict that qualifies every `cap` column there.

## The part that makes it valid: alignment

The dump's `step` field is a per-process counter of dump **calls**, i.e. of ubatches -- and the
cap clamps `n_batch`, so the two caps split one prompt into different numbers of ubatches and
number the same sequence position differently. Keying on `(step, token_idx)` across two caps
compares unrelated rows. Three attempts, each falsified on real dumps:

| attempt | what it did | measured failure |
|---|---|---|
| raw keys | compare the dumps as-is | `step` counts ubatches -> unrelated positions |
| single-token only | drop `n_tokens > 1` | 117 rows -> 21, of which **2** were the base model (MTP verify batches dominate) |
| request from position decrease | increment a request counter when position drops | 6 vs 8 "requests": MTP draft rejection ROLLS BACK the memory, and how often depends on the accept pattern, i.e. on the numerics under test |

What works: in a causal batch the rows cover contiguous positions ending at `pmax`
(`llama_memory_seq_pos_max` after the ubatch), so row `token_idx` of an n-token dump sits at
`pmax - (n-1) + token_idx`. That is a real per-token position for prefill chunks, single decode
steps and verify batches alike. The key is `(position, occurrence)` -- the occurrence index
handles both a second request restarting at position 0 and a rollback recomputing a position.
For two dumps running the same requests in the same order, the k-th computation of position p
aligns with the k-th computation of position p, whatever the chunking.

Two comparisons are made and the verdict comes from the first:

* **DEF** -- the target model's own context. This is the claim.
* **all ctx** -- everything including the MTP draft head, reported alongside. The draft head has
  its own layout sensitivity, but it must not be allowed to speak for the model.

## Verdict semantics (M1 and M2 are never merged)

| verdict | meaning |
|---|---|
| `PASS` | M1 full (bit-identical logits) with coverage >= threshold and n >= floor: the cap changed the layout and nothing moved |
| `DRIFT` | M1 not full, M2 full: same decisions, different bits. **Not a pass** -- greedy decoding is chaotic and this can flip at any later token |
| `FAIL` | M1 and M2 both differ |
| `INCONCLUSIVE` | (only reachable from a full M1) not enough aligned steps to certify inertness |
| `REFUSED` | the pair's provenance does not hold: different engine code or different weights |

Divergence is weighed before coverage, deliberately. Coverage and the step floor gate the *PASS*
direction only: a non-full M1 is positive evidence, and it is not weakened by imperfect overlap
-- imperfect overlap is in fact its **consequence**, because once a token differs the two
generations diverge and stop sharing positions. An earlier version required >=95% coverage for
any verdict, which made the gate silent on exactly the pair it exists for (measured: 93.5%
coverage with M1 2/29).

## Measured results (Nail IQ3_XXS-denseIQ4X, MTP on, 8 GiB pool, same binary)

**Determinism control -- two independent runs, both cap=8:**

```
cap8 vs cap8: PASS   DEF M1=35/35  M2=35/35  M3=35/35  n=35 coverage=100.0%
                     ALL(ctx) M1=60/60  M2=60/60  n=60
```

Two separate server launches, 10 s each to healthy, 60 rows each, aligned 60/60. This is what
licenses reading the next row as a statement about the cap rather than about run-to-run noise.

**The real pair -- cap=6 vs cap=8:**

```
cap6 vs cap8: FAIL   DEF M1=2/29  M2=7/29  M3=2/29  n=29 coverage=93.5%
                     ALL(ctx) M1=3/49  M2=11/49  n=49
                     first divergence at pos=25 occ=0 (DEF): same argmax (16),
                     sum -591919.81 vs -610423.32
```

Both dumps: `nil=ok`, RSS 8.16/8.17 GB, healthy in 25 s/10 s. The first divergence is at
**position 25**, in the base context, at the same argmax token -- i.e. drift before decisions,
the state the metrics split exists to name. Note also that M2 (7/29) is far above M1 (2/29):
reading M2 alone would have called this mostly-agreeing.

## Consequences

1. **The cap is a partition knob, not a free speed knob — and not a layout knob either.** Any pool
   sweep that changes the cap between cells is comparing different *forward passes* (the cap is
   clamped onto `n_batch`; `n_ubatch` follows), not merely different pool layouts. `summary_table()`
   already warns when a model's rows carry more than one cap, and this gate is the numeric half of
   that rule. Prefer pinning the cap and sweeping the pool: that pair is measured bit-identical
   (M1 41/41 at cap 8 across 2/4/6/8/10 GiB), so a pool sweep at a fixed cap is a clean experiment.
2. **`cap=5`'s +13-15% stays unavailable**, and the reason is now precise: it re-partitions the
   prefill, and the model's forward pass is not partition-invariant. It is *not* pool damage and
   *not* the pool↔gather switch (measured `gather=0` in every layer of every arm) — so the fix, if
   one is ever wanted, is to decouple the partition from the cap, not to reorder the MoE reduction.
   The gate is the entry condition, and it is now cheap to re-run (dumps are cached; the pair
   re-compares in seconds).
3. **The earlier hand figure (M1 3/82) was measured with the ubatch-numbering alignment**, i.e.
   with keys that mix positions across chunkings. The direction survives the correct alignment
   (2/29 DEF), but the numbers from that run should be quoted as superseded.
