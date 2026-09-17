# The pool budget is a first-order confound — production ladder, 2026-09-17 23:35–23:40

**Status: raw measurement, not a whitepaper.** Same build as the rest of the day (`libllama.0.0.279.dylib`
and `libllama-common.0.0.279.dylib`, mtime 19:35:52 / 19:35:59, md5 == HEAD). No rebuild during or
after. Raw artefacts: `Backup/cgc_logs/en_prod8g_20260917/{a,b}.{json,md,log}`,
`/tmp/gpu_split4/b` (4 GiB, MTP off), `/tmp/en_gpusplit_mtpon/a` (4 GiB, MTP on).

Cell: production profile `prefill250`, production decode shape `-p 0 -n 128 -d 512 -b 512`, reps 3,
`CGC_GPU_TIMING=1 CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1`. The only difference between the
two pool rows is `--expert-cache` (`CGC_SERVER_EXPERT_CACHE_BYTES`), which the `prefill250` profile
sets to 8 GiB when it is not overridden.

## 1. The 2×2 (platform_ts; every arm launched NOMINAL)

| pool | MTP off | MTP on |
|---|---|---|
| 4 GiB | 6.147 | 6.480 |
| **8 GiB (production)** | **10.624** | **10.444** |

**8 GiB reproduces the recorded production value (10.78) to within 3 %.** So the day-long
"10.78 ↔ 5–6.5" discrepancy that both lines attributed to thermal state, swap and neighbours is
**the expert-cache budget**, not the machine.

## 2. What the pool buys is entirely CPU-side (4 GiB → 8 GiB, both MTP on, step medians)

| reading | 4 GiB | 8 GiB | ratio |
|---|---|---|---|
| step total | 136.94 ms | **85.65 ms** | **×0.63** |
| GPU span `union_sum` | 76.53 | **75.66** | **×0.99 (unchanged)** |
| GPU idle `gap_sum` | 54.25 | **19.95** | ×0.37 |
| CPU hook `cb` | 36.75 | **9.98** | ×0.27 |
| `wait` | 79.33 | 72.41 | ×0.91 |
| GPU idle share of the GPU clock | 41.5 % | **20.9 %** | |
| overlap ceiling `total/union` | **1.79** | **1.13** | |

**The GPU work is identical (75.66 vs 76.53 ms).** The whole 1.6× is the removal of
`cb`-side blocking — slot management and blocking expert fills, i.e. churn.

## 3. Three consequences

1. **§3b of the other line's status doc is pool-biased.** Its "GPU idle 35.3 %" and "the idle term is
   worth ~1.5×" were measured at 4 GiB. At the production pool the idle is **20.9 %** and the overlap
   lever is worth **1.13×**, not 1.5×. Its absolute numbers (step 135.21 ms, 6.15 t/s) carry the same
   bias.
2. **M3's exit condition still fails at production, but the gap is 1.14×, not 2.32×.** The GPU span
   alone is 75.66 ms ⇒ **13.2 t/s ceiling** with a zero-cost CPU. 15 t/s needs the span to fall to
   66.7 ms.
3. **The 8-env bundle that `CGC_SERVER_MTP=1` pulls in is neutral at the production pool**
   (10.444 vs 10.624 = −1.7 %, inside noise). At 4 GiB the same pair reads +5.4 % with a −12 % GPU
   span — i.e. that apparent "treatment effect" was pool starvation, not a property of MTP.
   ⚠️ `llama-bench` cannot speculate at all (`tools/llama-bench/` has zero hits for
   `speculat|draft|mtp`), so neither pool row measures MTP itself; they measure the env bundle.

## 4. What this does to "25 t/s"

At production, a steady step is 85.65 ms ⇒ 11.7 t/s, with a **13.2 t/s** zero-gap ceiling. 25 t/s
needs 40 ms/token. If the server's MTP really yields ~2.2 tokens per step at the same step cost, then
2.2 / 0.0857 s = **25.7 t/s** — so the whole question now hangs on **one number: tokens per step under
speculation at the production pool**, which `llama-bench` cannot produce. Only the server path with a
per-step profile can, and its step structure is not isomorphic to this one.

## 5. Method rule this run establishes

**An A/B on a non-production pool manufactures treatment effects.** Four readings today were damaged
this way: the 4 GiB MTP on/off delta (+5.4 %), the 4 GiB "GPU idle 46 % / 35 %", the derived
"overlap is worth 1.5–1.8×", and the 8 GiB "layer 0 carries 47 %" (that one from a last-wins
per-layer aggregation, see `.workbuddy/memory/2026-09-17.md` §EN-90). Any decode-cell A/B must run at
the profile's production pool, or be labelled a low-pool probe.
