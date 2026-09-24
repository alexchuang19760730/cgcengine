# F2-R — a design that can actually test overlap

Date 2026-09-20. Supersedes the F2 arm design in
`Backup/phase_decomp/F2_F5_OVERLAP_AND_AUDIT_20260920.conditions.md`, which could not answer its own
question for two independent reasons (§1). Everything below is grounded in measurements made today,
not in the record's recollection of them.

---

## 1. Why F2 measured nothing, in two independent ways

**(a) The profile had the prefetch machinery switched off.** The F2 grid ran `--profiles qa-zh`.
From the launcher's own in-order env dump (`CGC_DUMP_ENV=1`, which prints `SERVER_ENV` in the order
`env "${SERVER_ENV[@]}"` will consume it):

| profile | `CGC_SPAC` | `CGC_NO_PREFETCH` | `CGC_EVICTED_RING` | pool |
|---|---|---|---|---|
| `qa-zh` (what F2 used) | **absent** | 1 | 0 | 10 GiB |
| `prod25` (the 25 t/s pedigree) | **1** (alpha 0.75) | 1 | 0 | 8 GiB |
| `prefill250` | **1** (alpha 0.75) | 1 | 0 | 8 GiB |

qa-zh sets `CGC_NO_PREFETCH=1` and does **not** enable SpAc. The once-per-step prefetch — the only
mechanism in this engine that can overlap anything — was therefore *off*, and `prefetch=0/0` is
already explained before residency is discussed at all. **F2 tested a prefetch in the one profile
family where prefetch does not run.**

**(b) The per-layer triggers found nothing to fetch.** The layer-ahead and prev-token hooks sit in the
per-layer `on_topk` callback and are **not** gated by `CGC_NO_PREFETCH`, so they did run — and the
shared prediction structure read out as 40/41 layers valid with **320 of 320** predicted experts
already resident (8 GiB **and** 4 GiB). `resident = all of them` is a property of the geometry:
143 slots/layer at 8 GiB, 71 at 4 GiB, against a per-token demand of 8.

So (a) explains the counter and (b) explains the mechanism. Neither is a statement about overlap.

## 2. The reframe: two questions, and only one of them needs a predictor

F1's structure (`docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md`) is
`cb ≈ 0.46·L + 0.695·M` with `L ≈ 29/40` layers and `M ≈ 74` misses per step, i.e. **13 ms barrier
+ 51 ms per-miss**. Those two terms answer to completely different questions:

* **Q-scheduling** — when a fill is needed, must the consumer block? *Prediction-free.*
* **Q-prediction** — is there an expert nameable early, needed soon, and not resident?

F2 only ever addressed Q-prediction, and it did so in the wrong profile. The 13 ms barrier is
entirely Q-scheduling and has never been tested here.

## 3. The ceiling arithmetic — what is actually at stake

F1's miss census (long delivery run): **25.6% compulsory / 74.4% capacity-evicted.**

* Compulsory = the expert has never been touched. **No predictor can cover it** — there is nothing to
  predict from. That sets a hard floor.
* Capacity-evicted = it recurred. Coverable **in principle**, but only if named before the demand.

⇒ Ceiling of the entire prefetch family ≈ **74.4% × 51 ms ≈ 38 ms of cb**, i.e. `cb` 66 → ~28 ms,
and that is an *optimistic* bound (it assumes every capacity miss is named in time). Note also that a
covered layer also escapes its barrier (`cb(m=0) = 0.01 ms`), so a fully-covered step approaches
F4's warm bound of 0.4 ms — but only the coverable share can get there.

**⇒ The entire prefetch family is worth at most ~38 ms of a ~220 ms step. It cannot reach 25 t/s on
its own, by construction.** That should be stated before spending a window on it.

## 4. L1 — dose-response + the residency census (cheap, decisive, shipping regime)

No code. One profile, one env group, K swept. This measures whether the live predictor has any cold
candidates at all when asked for more.

```
CGC_SERVER_PROFILE=prod25 \
CGC_SPAC_DBG=1 CGC_SPAC_K={8,32,64} \
CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1 CGC_GPU_TIMING=1 CGC_HOOK_SPLIT=1 \
LLAMA_EXPERT_CACHE_BATCH_DBG=1 \
  ./scripts/run_server.sh
```

Read: `CGC-SPAC-MEM layer=N topK=K resident=R (X%) queued=Q` (per layer — this is the census),
`prefetch=N/M` + `prefetch drop breakdown` from final stats, and `cb` / its HOOK_SPLIT parts.

**Pre-registered dispositions**

| reading | verdict | consequence |
|---|---|---|
| `resident%` at K=64 still ≥ 90% | **coverage is not a lever** | skip L2; the pool is 17.9× over-provisioned and no predictor can be cold |
| `resident%` falls at larger K **and** cb falls monotonically | coverage **is** a lever | the slope gives ms per covered miss; L2 becomes worth running |
| `queued` rises but cb is flat | coverage is fine, **timing** is not | the fills don't reach the demand path in time — that is L3's problem, not L1's |

Corroborating record (label it as such — different build and pool, 2026-09-08): the instrument's own
line read `CGC-SPAC: feeds=547 queued=4 n_prefetch=1702 dropped=71`, i.e. **4 queued against a
K=8 × 41-layer = 328 candidate set**. That is the same 1-in-80 cold rate the per-layer triggers showed
from the other direction, and it is why L1's first row is the expected one.

## 5. L2 — the only prefetch source whose candidates are non-resident by construction

`CGC_EVICTED_RING` (llama-expert-cache.cpp:3402) is a per-layer ring of experts recorded **as
`pick_slot` evicts them**. So its candidates are, by construction, the ones that are *not* resident —
the one property no other source has. It is consumed only inside the `pf_hist` branch
(llama-context.cpp:2113), which is unreachable while SpAc is on (`if (spac_on) … else if (pf_hist) …`).

```
CGC_SERVER_PROFILE=prod25 \
CGC_SPAC=0 CGC_SERVER_NO_PREFETCH=0 CGC_PREFETCH_SRC=hist CGC_EVICTED_RING={0,16,64} \
LLAMA_EXPERT_CACHE_PREFETCH_DBG=1 CGC_TAIL_DBG=1 \
  ./scripts/run_server.sh     # with MTP off
```

**Regime caveat, stated before the run:** SpAc is the pool's *membership driver* — the `prod25`
comment records that turning it off leaves the decode working set non-resident and the output
degenerate. So L2 measures the ring in a **different regime**, and its `cb` is **not comparable** to
the delivery `cb`. L2 answers a mechanism question only: *when a genuinely cold candidate exists, does
a step-early fill cover the demand?*

**Pre-registered:** the ring's candidates are non-resident by construction, so
`PFDBG` lines from `prefetch_slot` **must** appear and `prefetch=` must be > 0. If they do not, that
is a **code defect** (the ring is not being populated), not a null result — read `pick_slot` next.
Then: does `ensure` (HOOK_SPLIT) drop against the `RING=0` control of the same regime?

## 6. L3 — the only lever that does not need a predictor, and the only one with real upside

The 51 ms per-miss term is 74 misses costing 0.695 ms each for ~1.11 MiB of bytes. Those bytes move at
**1.6 GB/s** (F1) — copy speed. So the cost is overwhelmingly **latency, not bandwidth**, and the
engine pays it **synchronously**: `ensure_batch` issues the union's fills and waits
(`outstanding == 0`) before `drain_layer` returns. Nothing about that wait requires a prediction; it
requires not blocking.

Target: issue the layer's fills and return; let the first consumer of the layer's weights take the
wait. Overlap it with the *same* layer's GPU work and the previous layer's FFN tail.

Measure with `fill_wait_us` — the **only** fill-path counter accumulated by the calling thread, so it
is directly comparable to a step's wall clock (the header says so explicitly, and adds that
`pread_usec` cannot be used for this at all).

**Pre-registered:** PASS on a drop in `fill_wait_us` **and** total step, with `CGC_GPU_TIMING` watched
— a saving that reappears as GPU idle is not a saving. Falsifier: `fill_wait_us` falls but `total` is
flat because `wait` rises.

⚠️ This needs its own slot space to be MTP-verify-safe: `CGC_NO_PREFETCH` exists precisely because a
background fill can overwrite a slot the GPU is mid-verify reading. So the change must give
prefetches slots the verify batch cannot be reading — the free-slot-only discipline layer-ahead uses
is the existing precedent.

## 7. Instrumentation: what was blocking this, and what is still missing

**Closed today** (same defect class as `LLAMA_EXPERT_CACHE_BATCH_DBG`, which is why it was findable):
`CGC_SPAC_DBG`, `CGC_PREFETCH_SRC`, `CGC_EVICTED_RING`, `CGC_TAIL_DBG` were **not in the launcher's
env allowlist**. These are not merely instruments — they are the *selectors* of the prefetch source,
so "prefetch had nothing to do" and "prefetch was not selectable" were indistinguishable from the
launcher. Verified end-to-end after the fix, using the in-order dump:

```
ENV CGC_EVICTED_RING=0        <- the hardcoded element (run_server.sh:1352)
ENV CGC_EVICTED_RING=16       <- the allowlist passthrough, appended later
=> `env` keeps the LAST duplicate, so the override wins (verified directly, not assumed)
```

**Still missing, and it matters for L2's interpretation:** the layer-ahead trigger has **no counter of
its own**. Today it can only be diagnosed second-hand, through the *prev-token* consumer's
`valid_layers` / `resident` line, because both read the same `prev_token_expert_ids`. A counter at
the trigger distinguishing "entered but predicate false" from "never entered" would make each
prediction source independently falsifiable.

## 8. Cost, order, and the expected outcome

| lane | cost | what it decides |
|---|---|---|
| **L1** | ~15 min (3 loads ≈ 2.5 min + ~1.5 min requests each) | whether a cold candidate exists at all in the shipping regime |
| **L2** | ~30 min (6 launches, MTP off) | whether a step-early fill of a genuinely cold candidate covers the demand |
| **L3** | 2–3 h code + rebuild + 2 launches | the 13 ms barrier and most of the 51 ms latency term |

**Order L1 → L3 → L2.** L1 is nearly free and its first row is the likely one; L3 is the only lane
with real upside and it is prediction-free; L2 is the most expensive per unit of information and
carries a regime caveat that limits what its number can mean.

**Honest expectation.** The pool holds 143 slots/layer against 8 needed per token (17.9×), and the
live predictor already finds ~99% of its candidates resident. So L1 most likely returns
**"no cold candidates"**, which closes the prefetch family with a number rather than a shrug — and
makes L3 the whole of the remaining plan. That is why L1 is stated first as a *decision*: its most
likely outcome is a negative one, and the design should be read as buying a decision, not a speedup.

## 9. What this design still does not do

* No absolute t/s claim is set up for L2 (regime caveat) — only a mechanism comparison.
* L1's dose is `CGC_SPAC_K`, but SpAc also drives *membership* (alpha 0.75 EMA). A larger K moves the
  membership and the prefetch dose together, so a cb change under L1 is **not** attributable to
  prefetch alone unless alpha is held and the `resident%` reading separates the two.
* Nothing here addresses prefill; this is a decode-step design only.
