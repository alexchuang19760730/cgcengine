# Is a layer's idle window bought by its MISSES or by a fixed per-layer cost?

**Date** 2026-09-20 · carrier `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`, MTP on, 143 slots/layer
(LAYER_CAPS 5976 slots), pool 8 GiB, hit rate 65.4%, 76 decode steps per layer · four same-config
repeats: logs `222434`, `222954`, `223046`, `223153` (the gap-split capture arms).

## 0. One line

**The window is per-miss, with no fixed per-layer component: `idle_host` ≈ 0.6-0.9 ms x misses/step
(median and mean agree; F1's established per-miss cost is 0.695 ms) and a miss-free step costs at most
~0.2-0.3 ms per layer, against 1.8-2.8 ms/step on L0-L5.** The six churny layers are not special
because their hook is heavier -- they are special because they carry **3.82 misses/step against 0.39**
for the other 34. So the lever is where the misses are, not a fixed per-layer code cost.

## 1. Population discipline (this is where the answer could have gone wrong)

Three series, one run, one phase:

| series | source | what it is |
|---|---|---|
| `idle_host` | `CGC-GAPSPLIT-L` | ms per layer per 40-segment graph (CPU window) |
| `cb` | `CGC-DECPROF all` | ms per layer per 41-segment graph (the top-k hook) |
| `misses` | `BATCHDBG` | misses per `ensure_batch` call with >= 1 miss |

* **Prefill is excluded, and it is not a small correction**: the decode window is everything after the
  last `CGC-POST:` line, and the excluded prologue holds **59% of all misses** (3273 of 5561 on
  `223046`). Fitting across both phases would have been mostly a prefill regression.
* **Draft graphs are excluded**: the log carries two graph families (`segs=41` and `segs=2`, the
  single-layer MTP draft graphs), so per-layer rows are used only from their own family's full graph.
* **The two families name the same graph with DIFFERENT segment counts** -- the gap splitter says 40
  where DECPROF says 41 (one of them does not count the nextn block). My first version hardcoded 41 for
  both and silently returned `live=0` on a perfectly good log; each family's value is now derived and
  printed.
* **Medians AND means, because one of them is degenerate here**: a layer whose misses land in one step
  out of three has a *median* step with no miss, so its median window is ~0 by construction. The median
  is the robust pooled reading; the mean is what carries a rare miss's cost. Both are reported, and
  they agree on the slope.
* **In this population `idle_host` == `hook`** (hook is 100% of the window, median over layers): the
  per-layer tail and submit are ~0 in these rows, so "the window" and "the hook" are the same quantity
  here. The dichotomy in the question collapses; the window is the hook.

## 2. The fit (all four runs)

```
                n    slope(med)  intercept   R2     rmse   fixed_share
222434  idle~miss 40     0.690      -0.273   0.98   0.109    -1.11
        (means)  40     0.635      +0.205   0.99   0.064     0.83
222954  idle~miss 40     0.671      -0.265   0.97   0.112    -1.11
        (means)  40     0.590      +0.214   0.99   0.072     0.90
223046  idle~miss 40     0.877      -0.310   0.98   0.127    -0.89
        (means)  40     0.853      +0.299   0.98   0.114     0.86
223153  idle~miss 40     0.756      -0.267   0.99   0.092    -0.89
        (means)  40     0.683      +0.257   0.99   0.078     0.86
```

`cb~miss` is identical to `idle~miss` (the hook IS the window), so it is not a second result.

Within-group (the check that a pooled slope is not just a cluster difference):

```
223046  within L0-L5  n= 6  slope(med) 0.958  intercept -0.589  R2 0.98 | slope(mean) 0.845 R2 0.98
        within rest   n=34  slope(med) 0.029  intercept  0.039  R2 0.33 | slope(mean) 1.263 R2 0.64
        mean window: L0-L5 2.639 ms/step | rest 0.641 ms/step
```

The rest group's *median* slope of 0.03 is the degeneracy above, not evidence that their misses are
cheap: with means, the same 34 layers show **0.81-1.26 ms/miss, R² 0.56-0.65**. So the per-miss law
holds inside both groups; what differs is the miss *rate*.

Leverage was reported and handled: the largest residual is L4 in every run (3.2-3.5x rmse), the refit
without it gives slope 0.674-0.880 and **the same verdict**.

## 3. The decision, and its boundary

**MISS_BOUND in all four runs** (corr 0.97-0.99, fixed_share <= 0 in the median fit, <= 0.10 in the
mean fit). Consequences:

* A code change that neither reduces misses nor overlaps them **cannot move this window** -- there is
  essentially nothing else in it. `assign`/`collect`/`publish` were already measured at 0.3% of the
  ensure block (`docs/HOOKSUB_L0L5_ATTRIBUTION_2026-09-20.md`), and this fit closes the other half: no
  fixed per-layer hook cost is visible either.
* L0-L5's 6 layers owning 1.8-2.8 ms/step of window is a **miss-rate** fact (3.82/step vs 0.39), so
  whatever is done has to change which experts are resident when those layers run.

**What this does NOT decide: locality vs overlap.** Both are "miss-targeting": one makes misses rarer
(coverage, prefetch, eviction policy), the other makes them cheaper on the critical path (async fill).
This fit cannot separate them, and it should not be quoted as if it did -- the separating measurement is
the F1 per-miss decomposition (barrier 0.46 ms x ~29 layers + per-miss latency) and L3's structural
blocker, not this regression.

## 4. Instrument notes worth keeping

* The rule is committed in the tool's docstring, and its first draft was wrong: `slope*mean/total` is
  the same thing as the intercept share when the intercept is positive and **unbounded (2.1, >1) when
  it is not** -- so it printed "slope share 2.11" on a fit where there was nothing to share.
  `fixed_share = intercept / mean(window)` cannot do that, and it is the same quantity where the
  original was defined.
* A missing BATCHDBG series is INVALID, not "no correlation"; a layer with no fill call is named, not
  dropped; `idle_host >= hook` is checked (the window contains its hook).
* `scripts/check/gap_vs_miss.py`: 11/11 selftest, including both synthetic regimes (miss-driven ->
  MISS_BOUND, fixed -> FIXED_BOUND), the no-boundary refusal, the other-graph exclusion count, and
  "one 1000x outlier row cannot move the verdict" (medians).

## 5. Reproduce

```
python3 scripts/check/gap_vs_miss.py --log Backup/cgc_logs/llama_server_20260920_223046.log
```

Needs a log carrying `CGC-GAPSPLIT-L` (`CGC_HOOK_SPLIT=1`, `CGC_GPU_TIMING=1`,
`CGC_DECODE_PROFILE=1`, `CGC_DECODE_PROFILE_ALL=1`) plus `LLAMA_EXPERT_CACHE_BATCH_DBG=1` -- all four
are in the launcher's allowlist.
