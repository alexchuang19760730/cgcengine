# The capacity misses are addressable — and the victim rule, not the pool, is the lever

**Date:** 2026-09-20 · **Engine change:** none · **Leftover processes:** 0

> **Read with its follow-up.** This report's capture is MTP-**off** at 4 GiB. The delivery regime
> (MTP-**on**, 8 GiB, S=142) was captured later the same day —
> `docs/REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md` — which **confirms the D1/D2 shape** and is the
> source to cite for the **delivery geometry's numbers** (D1 1.08-1.50 × S; D3 40.6%). This file's
> **D1 section and its D3/lever figures were corrected** accordingly; the superseded values are kept
> inline so the correction is auditable.

## The question

`cb` is the device (`docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md`): ~82 MiB/step of miss bytes at
~1.6 GiB/s, engine/device 0.98, throughput flat in read shape. Nothing engine-side is left. The
bytes can only fall if there are fewer misses or smaller experts, and the miss census is
25.6% compulsory / 74.4% capacity. Compulsory is unreachable. **Are the 74.4% capacity misses
reachable by a pool or a policy, or are they long-distance re-touches nobody can serve?**

## Method, and the gate that had to pass first

`CGC-IDS` logs the complete demand set per (call, layer); `LLAMA_EXPERT_CACHE_MISS_DUMP` logs the
engine's own miss record. Replay LRU over the demand sequence at the run's own slot count and
require the replay's miss multiset to equal the engine's.

Capture: `--arms nail_nomtp` (MTP **off**, so one clean demand set per layer per call),
`--pool-gb 4`, `--n-predict 400`, `CGC_NO_PREFETCH=1`. 668 calls, 40 layers, 26,720 id lines,
engine `requests=224897 hits=189133 misses=35764`, `n_slots=71/layer` (partition disabled;
MTP-off means `zero_slot_enabled()` is false, so usable == n_slots).

### The pre-registered rule FAILED, and the failure was productive

| replay variant | misses | vs engine | entry agreement |
|---|---|---|---|
| pure LRU | 36,312 | +548 (+1.5%) | **98.8%** |
| LRU + `slot_decode_reserved` (my model of it) | 43,120 | +7,356 | — |

**Exact equality did not hold.** D1 below is a direct measurement over the demand sequence and
does not depend on the replay; D2/D3 are replay-level and carry the 98.8%/±1.5% agreement. Nothing
here is a timing.

Getting to 98.8% required finding **two real defects**, both of which had already produced a wrong
answer:

1. **`pmax` is not a call identity.** It is `llama_memory_seq_pos_max` — the position *within one
   request* — and a capture holds several requests, each restarting its positions. Grouping by
   `pmax` merged 668 calls into 408 groups and silently deleted demands: 21,717 replayed misses
   against the engine's 35,764. Calls are now recovered from **file order** (the hook fires once
   per layer per call, so a repeating layer index starts a new call), cross-checked against the
   engine's own `requests` counter.
2. **D1 counted every re-demand, not capacity misses.** The first version reported quartiles
   `[1, 3, 11, 35]` — which reads as "everything is a hit, just evicted". That was an artifact:
   ordinary hits dominate re-demands, and they are by definition recent. Fixed: only a re-demand
   that found the slot gone carries a distance. The reading moved to `[28, 49, 99, 175]`, i.e. the
   opposite conclusion.
3. **…and that second reading was STILL in the wrong unit** (found later the same day). It measured
   distance in **calls**, and a call carries several experts, so "distance 49" is not "49 experts
   ago". In the correct unit the MTP-off reading is `[85, 105, 134, 164]`. **The D1 section below is
   updated and the superseded numbers are kept so the correction stays auditable**; the conclusion
   survived and was strengthened, but `p50/S` went from 0.69 to 1.48 — i.e. the original text's
   "the typical miss is re-demanded well inside one pool-width" was wrong.

The residual +1.5% has a named mechanism: the engine's victim rule is **not** plain LRU — a decode
hit sets `slot_decode_reserved` (llama-expert-cache.cpp:1043) and `pick_slot` refuses to evict a
protected slot while an unprotected one exists (line 550). That makes the engine slightly *better*
than LRU, in the direction observed. My own model of the flag over-protects (43,120), so the flag's
exact scope is not reproduced; the net effect is bounded at +1.5% in the engine's favour.

## D1 — the reuse distances sit AT the eviction boundary. The misses ARE reachable.

> **This section was corrected the same day, twice.** The numbers first published here were in
> **calls**, which no capacity can be compared against (a call carries several experts). The table
> below is the correct unit — **distinct experts accessed since that expert's previous access**, the
> unit in which "distance < S" is exactly the LRU hit criterion. `docs/REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md`
> recomputes it in **both** regimes; cite that for the delivery geometry.

Capacity-miss distance, in distinct experts (MTP-off row n = 28,391; MTP-on row n = 12,067):

| regime | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| MTP-off, **S = 71** | 85 | 105 | 134 | 164 |
| **/ S** | **1.20** | **1.48** | **1.89** | **2.31** |
| MTP-on, **S = 142** (delivery) | 154 | 167 | 189 | 213 |
| **/ S** | **1.08** | **1.18** | **1.33** | **1.50** |

**Superseded (call unit), retained for audit:** `p25 p50 p75 p90 = 28, 49, 99, 175`, `p50/S = 0.69`,
`p90/S = 2.46`. The `p50/S = 0.69` there supported "the typical miss is re-demanded well inside one
pool-width", which the correct unit contradicts.

Both branches of the hypothesis are still wrong, but for the opposite reason to what the call unit
suggested:

* **p50 ≈ 1.2-1.5 × S**: the typical capacity miss is re-demanded about **one pool-width** of
  distinct experts later — just past the eviction boundary.
* **p90 ≈ 1.5-2.3 × S**: even the tail stays within a few pool-widths.

So the misses are neither "re-used immediately" (distance ~0, which would point at the fill path)
nor "unreachably far" (which would close the pool/policy family). They sit **at the boundary**, and
the stable ratio across two different capacities (71 and 142) is what makes this a property of the
routing rather than of one pool size. That is why both a bigger pool AND a better victim rule
convert them — see D2 and D3.

## D2 — the pool lever is real, and the delivery geometry sits ON its knee

Replay at multiples of 71 slots (LRU, same trace):

| slots/layer | memory | misses | per call | change |
|---|---|---|---|---|
| 71 | 4.0 GiB | 36,312 | 54.4 | — |
| **142** | **7.9 GiB** | **13,374** | **20.0** | **−63%** |
| 284 | 15.9 GiB | 7,921 | 11.9 | −78% |
| 568 | 31.7 GiB | 7,921 | 11.9 | −78% |

A layer has 256 experts, so **284 slots/layer = every expert of every layer resident**: the plateau
is the compulsory floor, not a policy limit. That single fact reframes the curve — on this model
the pool can never beat 11.9 misses/call, and getting there costs 15.9 GiB, which a 16 GB box
cannot hold alongside a 12.7 GB model.

**The delivery pool is the 142-slot row** (8 GiB, measured earlier at `n_slots=143`). So the
delivery geometry is already past the steep part of the knee: halving the pool (4 GiB) costs +171%
misses; doubling it (15.9 GiB) buys only another −41%, for 8 GiB more. **This is the first
quantified basis for the project's earlier "placement is closed" conclusion** — it was right, but
about the *pool*, and the curve says why rather than asserting it.

## D3 — the victim rule is the under-exploited lever, and it is free

Belady OPT at the same capacity (offline upper bound — it sees the future):

| slots/layer | LRU | OPT | gap | % of misses |
|---|---|---|---|---|
| 71 | 36,312 | 19,267 | 17,045 | **46.9%** (17.29 ms/call of cb) |
| **142** | **13,374** | **9,222** | **4,152** | **31.0%** |
| 284 | 7,921 | 7,921 | 0 | 0.0% |

At **142 slots on THIS trace** (MTP-off; delivery-sized, not the delivery regime) LRU is 31% above
the offline optimum, with zero extra memory. At 71 slots it is 47%. The gap closes exactly when
eviction stops happening (S=284), which is the expected signature of a genuinely policy-limited
regime rather than a measurement artifact.

**Delivery figure:** measured on the actual delivery geometry (MTP-on, S=142) in the follow-up,
the gap is **40.6%** — cite that number, not this 31%. It is larger because an MTP-on call demands
~32 experts atomically, which makes pure LRU's victim choice worse.

This is new for this project: all prior "placement" work was **static** (pin a top-K set from a
frequency profile), which the record closed as diffuse. A **dynamic victim rule** is a different
question and this is the first number that says it has room. Caveat stated: OPT is clairvoyant, and
a real policy (ARC/LFU/LIRS-class) captures some fraction of the gap, not all of it. Read D3 as
"how much headroom exists", not "what you get".

## What this means for the 25 t/s target

* The compulsory floor is 11.9 misses/call ≈ 11.9 × 1.11 MiB / 1.6 GiB/s = **8.1 ms/call of cb that
  no pool and no policy can remove**. That is the irreducible device read.
* Pool: already at the knee. Not the lever.
* **Policy: 31-41% of misses at the delivery capacity, for free.** 31% is the MTP-off 142-slot
  point on this trace; at the **actual delivery geometry** (MTP-on, S=142) the confirmed reading is
  **40.6%** — see `docs/REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md`, which is the one to cite for
  the delivery number. At 142 slots that is ~6-8 misses/call ≈ 4-6 ms/call if fully captured, and
  the realistic fraction is lower.
* **Quantisation remains the only lever with a linear, unbounded payoff**, because cb scales
  directly with bytes per miss.

So the ordering of the bytes-side levers, now measured rather than argued:

1. **Smaller experts** (linear, unbounded, but a quality trade).
2. **Victim rule** (**40.6%** of misses at the delivery geometry, free memory, needs a real policy
   and a bit-identity argument — eviction order does not change arithmetic order, so M1 should
   survive, but that must be proven, not assumed).
3. **Bigger pool** (past the knee; poor return).

## Artifacts

- `scripts/check/reuse_distance.py` — the tool, now **32 self-test checks, 0 failing**: a
  **must-fail control for the defect that actually happened** (two calls sharing a `pmax` must be
  two demand events, never one merged set), a randomized cross-check of the Belady heap against a
  brute-force reference (180 cases), the atomic-batch invariant, and a refusal path when no engine
  miss record exists. (It was 25 checks when this file was written; the three defects below added
  the rest.)
- `Backup/phase_decomp/REUSE_DISTANCE_20260920.conditions.md` — pre-registration.
- `Backup/phase_decomp/reuse_distance_result.json`, `miss_dump_off.txt`
- `Backup/cgc_logs/llama_server_20260920_131425.log` — the capture.

## Caveats

- **Regime**: this capture is MTP-OFF (ntok=1, ~8.3 experts/call/layer). The delivery regime is
  MTP-ON (ntok=4, ~4x the demands per call), so **per-call counts do not transfer**. What transfers
  is the *shape*: the distance/S ratios and the position of the delivery geometry on the D2 curve.
  **The follow-up was done**: `docs/REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md` captures MTP-on at
  8 GiB (S=142) and confirms the shape (1.08-1.50 × S there vs 1.20-2.31 × S here).
- **Box was busy**, and the launch went through `CGC_WINDOW_OVERRIDE=1` with `--need-mb 5400` (both
  recorded in provenance). That is acceptable *because routing is computed by the MoE gate upstream
  of the cache*, so memory pressure cannot change which experts are selected — and for the same
  reason **no timing from this run is quoted anywhere**.
- The `first version of this tool` failures (pmax merging, D1 counting hits, and the call-vs-`
  distinct-experts unit) are recorded above rather than quietly fixed, because all three produced
  plausible-looking wrong numbers.
- The pre-registered exact gate **failed** at 98.8%. D2/D3 rest on that 98.8%. In the MTP-on
  follow-up it is worse (93.7%), because that path has more mechanisms than pure LRU models.
