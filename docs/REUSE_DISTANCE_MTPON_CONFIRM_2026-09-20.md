# The reuse-distance shape survives into the delivery regime — and the delivery point sits on the curve

**Date:** 2026-09-20 · **Engine change:** none · **Leftover processes:** 0

Follow-up to `docs/REUSE_DISTANCE_RESULT_2026-09-20.md`, which measured MTP-off at 4 GiB. This is
the same instrument on the **delivery regime**: MTP **on**, **8 GiB** pool, same carrier (Nail),
same prompts, `CGC_NO_PREFETCH=1`.

## The two captures

| | MTP-off | **MTP-on (delivery)** |
|---|---|---|
| pool | 4 GiB | **8 GiB** |
| `n_slots`/layer | 71 | **143**, usable **142** |
| `ntok` mix | 1 (dominant) | **4 (dominant)**, +1/2/7/8 |
| calls | 668 | 354 |
| engine misses | 35,764 | **19,006** (dump) / 19,159 (`final stats`) |
| replay (pure LRU) | 36,312 (+1.5%) | **20,470 (+7.7%)** |
| entry agreement | 98.8% | **93.7%** |

Geometry resolved from the engine's own line: `LAYER_CAPS per-layer caps: total 5976 slots
(avg 145.8/layer, min 143/layer)` — and **5976 = 40 x 143 + 256 exactly**, so layers 0-39 carry
143 slots and only the MTP head (layer 40) carries 256. MTP-on sets `CGC_VERIFY_DECODE`, so
`zero_slot_enabled()` is true and one slot per layer is reserved: **usable = 142**. Layer 40 is
excluded (`--drop-layers 40`) since its cap is different.

## D1 — the shape is confirmed, in the unit that matters

Stack distance = **distinct experts accessed** since that expert's previous access. A re-demand is
a hit iff distance < S, so a capacity miss carries distance >= S by construction.

| | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| MTP-off, S=71 | 85 | 105 | 134 | 164 |
| **/ S** | **1.20** | **1.48** | **1.89** | **2.31** |
| MTP-on, S=142 | 154 | 167 | 189 | 213 |
| **/ S** | **1.08** | **1.18** | **1.33** | **1.50** |

**This is the answer to the question the follow-up was asked.** In both regimes the capacity misses
cluster at **1.1-2.3 x S**. The absolute distances differ (105 vs 167) exactly in proportion to the
capacity (71 vs 142), and the ratio is stable and slightly *tighter* in the delivery regime.

Interpretation: a capacity miss happens when the expert was last used **one to two pool-widths of
distinct experts ago**. That is the eviction boundary itself. It means the misses are neither
"re-used immediately" (which would be distance ~0 and would point at the fill path) nor
"unreachably far" (which would close the pool/policy family). They are **at the boundary**, which
is why both a bigger pool and a better victim rule convert them.

## D2 — the delivery point on the pool curve

MTP-on, same trace:

| slots/layer | memory | misses | per call |
|---|---|---|---|
| **142 (delivery)** | **7.9 GiB** | **20,470** | **57.8** |
| 284 (>= 256 = all experts resident) | 15.9 GiB | 8,403 | 23.7 |
| 568 | 31.7 GiB | 8,403 | 23.7 |

The MTP-off curve, for the same two ratios: 71 -> 142 -> 284 gave 36,312 -> 13,374 -> 7,921
(54.4 -> 20.0 -> 11.9 per call). So on both curves **doubling the pool removes ~60% of the
misses**, and the plateau is the **compulsory floor** (a layer has 256 experts, so 284 slots/layer
means every expert resident).

**The delivery geometry is the 142-slot point, and it is above the floor by 2.4x.** Reaching the
floor needs 14.3 GiB of pool, which a 16 GB box cannot hold beside a 12.7 GB model. So the pool is
past the steep part but has not reached the floor — consistent with, and now quantified against,
the project's earlier "placement is closed" conclusion.

## D3 — the victim rule, at both geometries

| geometry | LRU | Belady OPT | gap |
|---|---|---|---|
| MTP-off, S=71 | 36,312 | 19,267 | **46.9%** |
| **MTP-on, S=142 (delivery)** | **20,470** | **12,153** | **40.6%** (15.9 ms/call of cb) |
| MTP-on, S=284 (all resident) | 8,403 | 8,403 | 0% |

**At the delivery geometry, LRU is 40.6% above the offline optimum, at zero memory cost.** The gap
closes exactly when eviction stops (S=284), which is the signature of a policy-limited regime
rather than an artifact. OPT is clairvoyant; a real policy captures a fraction.

## What changed versus the first report, and what did not

**Not changed (the point of the exercise):** the D1 shape (1.1-2.3 x S) and the D2 knee
(+~60% misses on halving, plateau at all-resident). Two independent regimes, one shape.

**Changed / corrected while doing this — three more real defects, all caught by the tool:**

1. **D1's unit was wrong.** It measured distance in **calls**, which no capacity can be compared
   against: one MTP-on call carries ~32 experts, so "39 calls" is ~1250 expert-demands. The
   call-unit reading was `p50/S = 0.27` for MTP-on — an artifact. In the correct unit it is **1.18**.
   The first report's MTP-off reading `[28, 49, 99, 175]` calls was likewise not the quantity its
   own READ text described.
2. **The plain stack-distance form is not an identity for multi-expert atomic batches.** It
   mismatched the LRU replay in *both* directions on 160 random cases, because the classic theorem
   assumes one access at a time. It is now used only inside the self-test, restricted to singleton
   demands, where it genuinely cross-checks. D1 uses the replay's own distances, which honour
   atomic batches.
3. **The replay could evict an expert its own batch was reading.** It now excludes the current
   demand set from victim candidates, matching the engine's pass1/pass2 with `batch_owned`. A new
   self-test pins that invariant (a batch of 2 served at cap 2 costs exactly 2 misses).

**The exact-match gate still fails**, and worse in MTP-on (93.7% entries, +7.7% totals) than MTP-off
(98.8%, +1.5%). The direction is the same in both: **pure LRU over-predicts, so the engine is
better than LRU** — consistent with `slot_decode_reserved`. Modelling that flag explicitly still
over-protects (20,747), so its exact scope is not reproduced. D2/D3 carry the agreement stated.

## One invalid cross-check, stated rather than dropped

The `demand members vs engine requests` check (valid for MTP-off, where it agreed within 1.3%) is
**not valid for MTP-on**: it reports 297,518 vs 59,783. The ZERO-slot fast path serves verified
tokens without going through the pool path that increments `requests`, so that counter no longer
counts demand members. Reported as invalid rather than quietly suppressed. The engine's own two
numbers *do* agree in this regime (dump 19,006 vs final 19,159, 0.8%).

## Deliverable summary

The delivery regime was the open question, and it answers cleanly:

* **D1 shape: confirmed.** Capacity misses at 1.08-1.50 x S in the delivery regime (1.20-2.31 x S
  MTP-off). The misses sit at the eviction boundary.
* **Delivery point on D2: S=142 -> 20,470 misses / 354 calls = 57.8/call**, against a compulsory
  floor of 23.7/call, at 7.9 GiB and 15.9 GiB respectively.
* **D3 at the delivery geometry: 40.6% of misses are policy-addressable for free memory**; the
  lever order from the first report holds (smaller experts > victim rule > bigger pool).

## Caveats

- **No timing is quoted.** The capture reports accept 55.35% / mean_len 2.610 / decode 11.97 t/s and
  those are recorded in provenance only; this instrument measures routing, and routing is computed
  upstream of the cache, which is also why a busy box does not corrupt it (`CGC_WINDOW_OVERRIDE=1`,
  `--need-mb 5400`, both in provenance).
- The **miss counts** differ between the two engine-side sources by 0.8% in MTP-on and 0% in
  MTP-off; the dump is used as the validation set in both.
- `ntok` mix means "per call" is not "per token". Cross-regime per-call numbers are not comparable;
  the **ratios to S** are.
