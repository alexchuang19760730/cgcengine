# M-F5 — the per-layer barrier, attributed: it is not removable overhead

**Date** 2026-09-20 · carrier Nail IQ3_XXS-denseIQ4X, MTP on k=3, pool 8 GiB, `prefill250`, temp 0.4,
completion door · instruments: `CGC-EBSPLIT` / `CGC-FSSPLIT`, both under the existing `CGC_HOOK_SPLIT`.

## 0. One line

**M-F5 is CLOSED, negative, with numbers.** The 0.46 ms/layer term F1 reported inside `ensure_batch`
is **not** host bookkeeping and **not** a scheduling artefact that can be moved off the critical
path: it is the layer's own union fill, and the only lever that moves it is **fewer or smaller
bytes**. The predicted **+6% is not available**, and the falsifier that would have delivered it was
executed and failed.

## 1. Why attribution had to come first

F1 fitted `cb_layer = 0.46 + 0.695·m` and named the 0.46 as a fixed per-layer term paid whenever a
layer has ≥1 miss. M-F5's whole premise is that this term is separable overhead. F1's own rule
applies to M-F5: **you cannot remove a term you have not attributed.** So the first deliverable is
not a fix, it is a named term.

Three candidates were **excluded by reading the code before spending a window**:

| candidate | why it is not it |
|---|---|
| `cgc_check_batch_invariant()` | early-returns unless `LLAMA_EXPERT_CACHE_BATCH_INVARIANT` is set (default off) |
| `cgc_exact_cache_verify_post_fill()` | early-returns unless `CGC_EXACT_CACHE_VERIFY` is set (default off) |
| `fill_wait_us` / the prefetch wait | the header states it only counts waits on prefetch-queued slots, not this path |

## 2. The instrument

Four timers inside `llama_expert_cache_ensure_batch`, read from a clock only (no behaviour change),
gated on the **existing** `CGC_HOOK_SPLIT` so the launcher allowlist needed no edit:

`assign` (the locked two-pass placement) · `collect` (file-offset math) · `fill` (submit + advise +
wait) · `publish` (the locked table write + WIN_PIN). Plus `submit` vs `wait` inside
`fill_segments_pool`, which has exactly one caller (this one).

## 3. Result 1 — it is not host bookkeeping (this refutes the F5 hypothesis)

Steady state, delivery shape, `n=3200` calls:

| phase | µs/call | share |
|---|---|---|
| assign (locked 2-pass + `pick_slot` + eviction) | **3.1** | 0.13% |
| collect | **1.9** | 0.08% |
| publish (locked table write + WIN_PIN) | **0.4** | 0.02% |
| **fill** | **2316** | **99.8%** |

Every host-side thing F5 suspected — lock acquisition, the batch mask, the publish, the eviction
path — is **5.4 µs of 2322**. So the barrier lives entirely in the fill.

Cross-check that the split reproduces F1 rather than replacing it: at `miss/call = 2.53`,
F1's fit predicts `0.46 + 0.695×2.53 = 2.22 ms`; the split measures `2321 µs`. **4% apart.**

## 4. Result 2 — the fill is submit + wait, and submit is the read-ahead hint

| | µs/call |
|---|---|
| submit (the caller's own CPU) | **780** |
| wait (pool round-trip: wake → pread → notify → caller wakes) | **1521** |

`submit`'s only non-trivial work is the `F_RDADVISE` loop. Tested with the pre-existing
`LLAMA_EXPERT_CACHE_NO_RDADVISE` (which had to be allowlisted — it was silently dropped before):

| | rdadvise ON | rdadvise OFF |
|---|---|---|
| submit | **779.6** | **6.8** (−99%) |
| wait | **1520.8** | **2312.4** (+52%) |
| **fill** | **2107.6** | **2146.9 (flat)** |
| decode | 11.63 t/s | 11.81 t/s |

**The hint is 99% of submit and buys ~790 µs of wait back.** So the two halves are *conserved* while
the hint is serialised on the decode critical path. The comment above `rdadvise_range` calls the
syscall cheap and non-blocking; the split measures **~260 µs per fcntl call**. That comment is
wrong for this volume and is now contradicted by a number rather than by an argument.

## 5. Result 3 — the trade is CAUSALLY ordered, not merely conserved (the falsifier)

If the halves are merely conserved, moving the hint to its own thread should keep the benefit and
drop the cost. Implemented (`LLAMA_EXPERT_CACHE_ADVISE_ASYNC=1`, default OFF, advisory only — no
result depends on it), rotated-order A/B (sync,async | async,sync):

| | sync (default) | async advisor |
|---|---|---|
| submit | ~880 µs | **9 µs** |
| wait | ~1536 µs | **4291 µs** |
| **fill** | **~2250 µs** | **~3962 µs (+76%)** |
| decode | 10.27 – 11.84 t/s | **8.72 – 8.91 t/s (−25%)** |

**Worse, and for a structural reason**: the hint only helps when it is issued **before** the preads
start. Deferring it makes it arrive late *and* contend with reads already in flight. So the submit
cost is **structurally on the critical path**, and this lever is dead. The flag is kept as the
recorded falsifier, with the verdict written into the header and the launcher comment so it cannot
be re-enabled on the assumption that it helps.

## 6. Result 4 — there is no resolved fixed term

58 steady-state samples across all arms, `fill` vs `miss/call`:

```
fill = -78.6 + 882.4 * miss_per_call        R2 = 0.438
```

- **fixed term ≈ 0** (−0.08 ms/layer) within the observed range.
- slope **0.882 ms/miss**; against the measured 1.11 MiB/miss that is **~1.26 GB/s** — page-cache
  copy speed, reproducing F1's own 1.6 GB/s reading.

⚠ **Honest limit on this fit**: `miss/call` only spans 2.47–2.90 in this regime, so the intercept is
poorly determined — a 0.46 ms intercept is *not* excluded by this data. What the data does establish
is that `fill` tracks misses with no resolvable constant *given a fixed shape*; the shape that would
resolve it (miss/call ≈ 1) is not reachable at the delivery pool.

## 7. What this closes, and what it does not touch

**Closed:** the "per-layer barrier is removable overhead" family. Its three sub-hypotheses are all
falsified — host bookkeeping (0.2% of the term), a free hint (conserved and causally ordered), and
an off-critical-path hint (−25% decode). The term is the union's bytes at page-cache speed.

**Consequence for the roadmap:** `M-F5`'s "+6% (19.3–19.4 in their shape)" **does not exist**, so
`M-25` cannot count it. The remaining term inside `cb` is still the **per-miss 51 ms (77% of cb)**,
and that is the bytes term — moved by **pool size or quantisation**, i.e. the axis this line already
measured as: 142 slots (delivery) 13,374 misses → 284 slots 7,921 (the compulsory floor).

**Not touched by this result — and the distinction matters for L2/L3:** everything above is about
making the **fill cheaper**. L3's target is different: stop the **consumer** blocking on it, i.e.
hide an unchanged fill behind the *same or a neighbouring* layer's GPU work. This round does not
refute or support L3; it removes one of L3's two justifications by showing the fill itself has no
free 13 ms inside it. It does, however, sharpen L3's pre-registration: a saving that reappears as
GPU idle is not a saving, and `fill` cannot be reduced by scheduling.

## 8. Artifacts

- `scripts/run_server.sh` — `LLAMA_EXPERT_CACHE_NO_RDADVISE` and `LLAMA_EXPERT_CACHE_ADVISE_ASYNC`
  allowlisted, each with the measured verdict in its comment.
- `src/llama.cpp/src/llama-expert-cache.{h,cpp}` — the four-timer split, the submit/wait split, the
  advisor thread (default off, documented as WORSE).
- `Backup/phase_decomp/ebsplit_capture.sh`, `rdadvise_ab.sh`, `advise_async_ab.sh`;
  `Backup/phase_decomp/rdadvise/`, `Backup/phase_decomp/advise/`.
- Engine logs: `llama_server_20260920_1524{24}`, `_1531{33}`, `_1532{33}`, `_1533{22}`, `_1534{22}`,
  `_1535{24}` — each arm's own log, recorded per launch rather than by "newest".

## 9. Verification

Both A/Bs are 2×2 rotated-order on one binary; the instrument is witnessed in the shipped dylib
(`strings` shows `CGC-EBSPLIT` / `CGC-FSSPLIT`), not in the plan. M1/M2/M3 bit-identity is re-gated
separately (see the memory entry). No absolute t/s is claimed beyond the within-pair ratios: every
launch ran at `--need-mb 5400`, **below the documented 8000 MB floor**, recorded in each arm's
provenance as `below_documented_floor: True`.
