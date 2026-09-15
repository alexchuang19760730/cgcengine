# cap=5 vs cap=8 — localisation, and why the proposed fix cannot work

**Date** 2026-09-14 · **Engine** `e8b85393d` (+uncommitted gate edits) · **Model** `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`
**Pool** 8 GiB, MTP off, temp 0, one long request (~400-token target, 59-token prompt)
**Binary** unchanged for all measurements below (no source edit, no rebuild)

The question was: *which step makes the logits change when `CGC_POOL_MAX_TOKENS` moves, and can a
canonical expert-reduction order (sort by expert id instead of slot / sorted-union position) bring
cap invariance back to M1 = 100 %?*

**Answer: the step that changes is the first one whose prefill partition differs, and no — a
canonical reduction order cannot fix this, because both caps already run the pool path and the
reduction order is already cap-independent.** The knob that moves the numerics is the **prefill step
partition** (`n_batch`/`n_ubatch`), not the pool layout. Evidence below, then what this means for
the gate.

---

## 1. Where the divergence starts

Two arms, same binary, same pool, same prompt, same request count, **MTP off** so every decode row
is single-token:

| arm | prefill step partition (`CGC-IDS`, run-length) | prefill steps |
|---|---|---|
| cap5 | `2x1, 5x11` | 12 |
| cap8 | `2x1, 8x6, 4x3` | 10 |

Oracle dump comparison (key = true sequence position, not the dump's ubatch counter):

```
=== cap5 vs cap8   common rows: 391
  M1 (row bits)     : 1/391
  M2 (argmax_token) : 17/391
  first M1 divergence: pos=59  ntok=1  DEF
      sum -586802.375 vs -603048.726   argmax 198 vs 271
```

`pos=59` is the **last prompt position** — i.e. the divergence is already in the *prefill output*.
Everything before it that can be aligned (the two single-token warmup steps at `pmax=1`) is
bit-identical; **no decode step is ever bit-identical**, because the generation has already forked.

### It is not rounding

The same tokens sit in both top-8, with ~1-logit gaps:

| token | cap5 | cap8 | Δ |
|---|---|---|---|
| 96608 | 9.826461 | 8.761765 | **+1.065** |
| 271 | 10.282467 | 11.217863 | −0.935 |
| 198 | 10.291896 | 10.917831 | −0.626 |
| 95772 | 10.095130 | 9.530836 | +0.564 |

A ~1-logit gap needs a ~0.1 relative hidden-state difference, not an ulp. So a **discrete** decision
changed somewhere and cascaded.

### Which layer

Aligned on the last chunk (`pmax=59`), comparing the 8 expert ids selected *for the last token of
that chunk*:

```
 il  cap5                     cap8                     iso(cap8,ub5)
  0  [38,21,114,182,163,77,24,60]  [38,21,114,182,163,77,24,60]  [38,21,114,182,163,77,24,60]   <-- ALL THREE AGREE
  1  [181,110,42,170,138,51,136,148] [193,42,238,148,138,181,136,18] [249,205,148,193,42,142,170,18]
  2  [27,117,135,255,187,222,57,233]  [255,172,123,222,233,135,27,200]  [172,57,253,121,222,87,233,10]
```

**Layer 0 routing is identical in all three arms; layer 1 routing is a three-way split.**
Layer 0 is a GatedDeltaNet (linear-attention) layer (`full_attention_interval = 4`), whose sequential
scan plus the batch-shaped projections around it are exactly the partition-sensitive machinery. The
perturbation is introduced in layer 0's computation — small enough not to flip layer 0's own router,
large enough to flip layer 1's.

---

## 2. The decisive isolation: cap8 with `-ub 5`

`CGC_POOL_MAX_TOKENS` is clamped onto `n_batch`, and `n_ubatch = min(n_batch, params.n_ubatch)`.
So `cap=8` with `-ub 5` keeps **cap-8 pool geometry** (slots, slab capacity = cap·top_k) while
chunking the prefill in 5-token units:

| arm | pool geometry | prefill partition | result |
|---|---|---|---|
| cap5 | cap-5 | `2x1,5x11` | argmax@59 = **198** |
| cap8 | cap-8 | `2x1,8x6,4x3` | argmax@59 = **271** |
| **iso = cap8 + `-ub 5`** | **cap-8** | `2x1,5x1,3x1,5x1,3x1,...` (5,3,5,3…) | argmax@59 = **95772** |

**The ISO arm matches neither reference — it produces its own third answer.** It has cap8's pool
and *neither* reference's partition. Three partitions → three different completions.

And the gather path is not involved at all:

```
=== gather usage, all layers, all three arms ===
  cap5:  407× "gather=0/8",  1× "gather=0/9"
  cap8:  391× "gather=0/8",  1× "gather=0/9"
  iso :  231× "gather=0/8",  1× "gather=0/9"
```

`gather=0` everywhere. Max union at cap5 is 40 (= 5·top_k8), at cap8 64, against **143 usable slots** —
both comfortably in the pool path. So:

- There is no pool-path ↔ gather-path switch to unify.
- The expert reduction (`ggml_add` chain over `cur_experts[i]`, `i = 0..n_expert_used-1`) already
  accumulates in **selected-expert *k* order**, which comes from the router's top-k and is identical
  in all three arms at layer 0. It is not slot-ordered and not sorted-union-ordered.

**The proposed fix therefore has no effect on this failure.** The gather path is dead code in these
runs, and the reduction order is already pool-layout-independent.

---

## 3. What *does* hold: pool-layout invariance

The property the product actually needs — *same configuration, different pool size* — is already
bit-identical. Using the recorded cap-8 dumps:

| comparison (cap = 8) | M1 (row bits) | M2 (argmax) | verdict |
|---|---|---|---|
| pool 2 GiB vs 8 GiB | 41/41 | 41/41 | **PASS** |
| pool 4 GiB vs 8 GiB | 41/41 | 41/41 | **PASS** |
| pool 6 GiB vs 8 GiB | 41/41 | 41/41 | **PASS** |
| pool 10 GiB vs 8 GiB | 41/41 | 41/41 | **PASS** |

So the two invariances must be separated:

- **Pool-layout invariance** — *same cap*, pool 2/4/6/8/10 GiB → **passes**. This is what
  "4 GB vs 8 GB must be byte-identical" asks for, and it is satisfied.
- **Cap invariance** — *same pool*, different `CGC_POOL_MAX_TOKENS` → **fails**. But a cap change is
  not a layout change: by the `n_batch` clamp it is a **partition** change, and the model's own
  forward pass is not partition-invariant (batch-shape-dependent kernel reduction order → ulp →
  discrete routing flip → ~1-logit output difference).

The old framing — "the cap changed the pool LAYOUT and the logits did not move" — was wrong. The cap
changes the layout *and* the partition; only the partition is causal, and nothing in the expert-cache
code can make the model's forward pass partition-invariant.

---

## 4. What the gate now does

`cap_invariance_gate` records and prints the **prefill step partition of each arm** and states which
knob actually moved:

```
[capinv] cap6 vs cap8: FAIL  DEF M1(bit-identical)=2/29 M2(argmax)=7/29 ...
[capinv]   attribution: prefill step partition: cap6=2x1,6x3,2x2,4x8 vs cap8=2x1,8x2,2x1,4x10
           -- the two caps chunk the prompt differently, so this compares two forward passes
```

- new helper `step_partition(log_path)` → `compressed` run-length of prefill `ntok`, `prefill_steps`
- `dump["partition"]` per arm, `pair["partition"]` + `pair["attribution"]` per pair
- attribution is one of: *partition differs* (comparing two forward passes) / *partition identical*
  (then a divergence really would be layout/slab) / *unknown* (no `CGC-IDS` recorded)

The verdict logic is unchanged (`FAIL` / `DRIFT` / `PASS` / `INCONCLUSIVE` / `REFUSED`); only the
interpretation is now explicit in the artifact.

---

## 5. Options, honestly

1. **Keep the gate, correct the label.** Cap invariance is a *partition-sensitivity* report, not a
   requirement. Pool-layout invariance is the requirement and it passes. (Recommended.)
2. **Make the partition independent of the cap.** This is the intent of M1 work item 1
   (`CGC_POOL_SPLIT`), and it is the *only* route to cap invariance — but it failed for an unrelated
   reason (the wide full-width-expert path faults Metal; see `M1_POOL_SPLIT_COST_2026-09-14.md`), and
   it is **not possible at small pools anyway**: one pool-path step must hold the whole union, so
   `cap` cannot exceed what the pool's slots accommodate (union = `cap · top_k`). At 4 GiB with 59
   slots, `cap` is bounded near 7; the cap is a *consequence* of the pool, not an independent knob.
3. **Make the forward pass partition-invariant.** Hard: it means every per-token reduction keeps the
   same association regardless of batch width, across 40 layers including the GDN scan. Not a
   localised change, and it would cost speed.
4. **Pin the cap and vary only the pool.** Already the sane operating mode, and already passing.

---

## 6. Reproduce

```bash
cd /Users/alexchuang/Documents/flashkv-devserver

# the gate, with the partition attribution now in the record
python3 scripts/check/knifeedge_matrix.py --cap-invariance \
    --models iq3 --pools 8 --cap-invariance-caps 6,8

# the isolation arm (cap8 geometry, cap5-sized chunks): partition 5,3,5,3...
python3 .tmp_cap_iso.py 8 5 iso

# compare any two dumps on true sequence position, M1 and M2 reported separately
python3 .tmp_cap_iso_cmp.py /tmp/capinf_5.jsonl /tmp/capiso_iso.jsonl cap5 iso
```

Evidence: `Backup/cgc_logs/llama_server_20260914_101328.log` (cap5), `..._101703.log` (cap8),
`..._102255.log` (iso); dumps `/tmp/capinf_{5,8}.jsonl`, `/tmp/capiso_iso.jsonl`;
`Backup/knifeedge_matrix/capinv_iq3_pool8gb.json` (gate record with partitions).

**Honest caveats.** The long-form localisation used one prompt per arm, and the harness probe prompt
is shorter — the *partition shapes* differ between the two (the harness shows long `4xN` tails from
`n_keep_tail`), so the recorded gate numbers are not the same partition sequences as §1's. The
conclusion rests on the *direction* (partition differs ⇒ output differs; partition identical ⇒
bit-identical), which both datasets show. M1 was checked at cap 6 vs 8 and cap 5 vs 8 and via the
cap8-vs-cap8 determinism control; not at every cap. Layer attribution is from routing ids only —
no intermediate activation dumps exist, so "introduced in layer 0" is inference from
`layer 0 identical / layer 1 differs in all three arms`, not a direct measurement.
