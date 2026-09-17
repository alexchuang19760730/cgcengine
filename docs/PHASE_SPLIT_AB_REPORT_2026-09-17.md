# Phase-split A/B: is the armed slab faster, or is the clamp merely slower?

- date: 2026-09-17 20:38
- commit: `6c24c5f2d`  (dirty: 22 path(s))
- profile: `prefill250`;  arms: `phase-slab` {}, `phase-pool8` {'CGC_PREFILL_STREAM': '0'}, `phase-pool17` {'CGC_PREFILL_STREAM': '0', 'CGC_POOL_MAX_TOKENS': '64'}
- build: llama-server `054fb22f04a01c5cb93b362883dd3f82`, libllama `0dd3a4b7c91413e7c6b4f8b205edd94d`, libggml-metal `037c931ec6169f60189e7fde19b1232e`
- model: `/Users/alexchuang/Documents/flashkv-devserver/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf` (pool 8589934592 B, load_mode none, port 8080)
- arm order: rotated per rep (`arm_order`: slab/pool8/pool17 -> pool8/pool17/slab -> ...), so over reps 2..N each arm occupies each launch position exactly once and position cancels out of the paired comparison
- raw: `Backup/phase_split_ab/20260917_2001/` (per-launch JSON, launch/server logs, watch.log)

## Result (n=4 launches/arm, interleaved; launch 1 discarded as the cold one)

| arm | short_warm per launch | long per launch | first_step per launch | short_postlong per launch | short_warm median | long median | first_step median | short_postlong median |
|---|---|---|---|---|---|---|---|---|
| `phase-slab` | [1.45, 1.32, 1.4, 1.23] | [155.26, 119.78, 114.85, 105.0] | [1.02, 1.43, 1.88, 1.17] | [1.43, 1.52, 1.63, 1.16] | 1.32 | 114.85 | 1.43 | 1.52 |
| `phase-pool8` | [17.05, 34.26, 23.98, 34.25] | [17.4, 19.45, 17.04, 16.72] | [7.36, 7.29, 7.53, 5.94] | [31.54, 30.84, 32.67, 29.4] | 34.25 | 17.04 | 7.29 | 30.84 |
| `phase-pool17` | [49.18, 51.22, 29.6, 37.05] | [35.13, 23.03, 26.89, 25.95] | [12.11, 9.47, 6.5, 10.44] | [42.57, 39.06, 46.93, 46.47] | 37.05 | 25.95 | 9.47 | 46.47 |

## Verdict

- **pool8/pool17 (long, same 2233-token chunk)** — pool8/pool17 = 0.657 -> **pool17 is 1.52x faster**; a clamp-width gap alone, so part of any 'slab beats pool8' number is really the clamp's penalty rather than the slab's gain
- **slab/pool17 (long)** — slab/pool17 = 4.426 -> **slab is 4.43x faster**; one PREFILL graph with no clamp vs ~132 pool graphs; two variables, the narrowest of the long-chunk comparisons
- **slab/pool8 (long, end-to-end)** — slab/pool8 = 6.74 -> **slab is 6.74x faster**; the regime-level difference (slab + no clamp vs pool + clamp); never quote it as 'the slab'
- **slab/pool17 (short_warm, one graph per request)** — slab/pool17 = 0.036 -> **pool17 is 27.78x faster**; matched graph count, so this is the slab mechanism itself: a 12-token chunk is served by one graph either way
- **slab arm, short_warm/short_postlong** — 0.868 -> the same 12-token chunk measures 1.15x apart in two cache states. Cache state, not the phase rule, decides this cell.
- **option D (arm the slab by default?)** — CONDITIONAL. Long chunks: the slab regime is 6.74x the pool8 regime. Near-threshold chunks: a 12-token request is 28.07x SLOWER through the slab than through the pool at matched graph count, and the phase rule sends every chunk wider than the decode width there. So a default arming needs a floor on chunk size (the `T_prefill` threshold), which this grid did not sweep: that sweep is the experiment this result calls for.
- **decode gate, short_warm** — **FAIL** slab 3.72 vs pool8 8.7 t/s (x0.428, tolerance -10%)
- **decode gate, long** — **FAIL** slab 1.94 vs pool8 7.68 t/s (x0.253, tolerance -10%)
- **decode gate, first_step** — NOT MEASURED -- no decode medians (usable launches: slab 0, pool8 0); refused: rep1:1000000.0 t/s; rep2:1000000.0 t/s; rep3:1000000.0 t/s
- **decode gate, short_postlong** — **FAIL** slab 1.64 vs pool8 5.72 t/s (x0.287, tolerance -10%)
- **arm rotation (position 1/2/3 per arm over counted reps)** — BALANCED -- {'phase-slab': 2.0, 'phase-pool8': 2.0, 'phase-pool17': 2.0} over reps [2, 3, 4]. Position was perfectly collinear with arm in the fixed-order grid; this is what makes the paired decode comparison below admissible.
- **decode after slab prefill (long) vs phase-pool8** — **SYSTEMATIC COLD (every counted rep colder)** median paired ratio x0.254 (3/3 reps colder, 0/3 warmer, tolerance ±10%; per-rep [0.288, 0.207, 0.254])
- **decode after slab prefill (long) vs phase-pool17** — **SYSTEMATIC COLD (every counted rep colder)** median paired ratio x0.243 (3/3 reps colder, 0/3 warmer, tolerance ±10%; per-rep [0.275, 0.243, 0.176])
- **decode after slab prefill (short_postlong) vs phase-pool8** — **SYSTEMATIC COLD (every counted rep colder)** median paired ratio x0.287 (3/3 reps colder, 0/3 warmer, tolerance ±10%; per-rep [0.287, 0.131, 0.542])
- **decode after slab prefill (short_postlong) vs phase-pool17** — **SYSTEMATIC COLD (every counted rep colder)** median paired ratio x0.226 (3/3 reps colder, 0/3 warmer, tolerance ±10%; per-rep [0.178, 0.237, 0.226])
- **decode pool: cold or thrashing?** — NOT thrashing: 4838 compulsory vs 0 capacity misses across the slab arm's launches (hit rate [86.4, 86.3, 86.3, 86.3]). The pool is holding what it holds; the misses are first touches, which publishing a prefill union cannot remove on its own.

## Decode non-regression gate

Baseline is `phase-pool8` (today's default: unarmed slab, clamp = cap 8); tolerance -10%. Decode medians are over launches 2..N, paired by rep index under rotated arm order, and reported per cell so a pass cannot hide a failing cell.

| cell | slab decode t/s | pool8 decode t/s | slab/pool8 | verdict |
|---|---|---|---|---|
| short_warm | 3.72 | 8.7 | 0.428 | **FAIL** |
| long | 1.94 | 7.68 | 0.253 | **FAIL** |
| first_step | None | None | None | **NOT MEASURED** |
| short_postlong | 1.64 | 5.72 | 0.287 | **FAIL** |

A cell the server called degenerate is NOT MEASURED with the refusal quoted, not PASS/FAIL: a cell where both sides carry the same artifact (e.g. `n_predict=1`, where the single predicted token is emitted with the prefill batch) ratios to 1.0 and would PASS while measuring nothing. Refusals recorded this run: `first_step` rep1:1000000.0 t/s; `first_step` rep2:1000000.0 t/s; `first_step` rep3:1000000.0 t/s; `first_step` rep4:1000000.0 t/s

### Arm rotation (the confound this grid was rebuilt to remove)

Counted reps: [2, 3, 4] (rep 1 discarded as cold, per launch 1 = cold caches). Mean launch position per arm: {'phase-slab': 2.0, 'phase-pool8': 2.0, 'phase-pool17': 2.0} -- BALANCED.

| rep | order |
|---|---|
| 2 | `phase-pool8` -> `phase-pool17` -> `phase-slab` |
| 3 | `phase-pool17` -> `phase-slab` -> `phase-pool8` |
| 4 | `phase-slab` -> `phase-pool8` -> `phase-pool17` |

### Is decode systematically colder after a slab-served prefill?

The hypothesis: the slab path re-points `wt->data` and returns without publishing into the pool, so the decode that follows starts on a colder pool than the same decode after a pool-served prefill. Tested with the two instruments that measure decode *inside the same launch* as the prefill they follow, paired per rep. `first_step`'s wall is deliberately not used here: it contains a 12-token prefill, which on the slab arm is the already-known short-chunk penalty, so it would measure the prefill defect and not the pool's temperature.

| cell | vs | per-rep paired ratios (slab/base) | median | reps colder/warmer | verdict |
|---|---|---|---|---|---|
| `long` | `phase-pool8` | [0.288, 0.207, 0.254] | 0.254 | 3/0 | **SYSTEMATIC COLD (every counted rep colder)** |
| `long` | `phase-pool17` | [0.275, 0.243, 0.176] | 0.243 | 3/0 | **SYSTEMATIC COLD (every counted rep colder)** |
| `short_postlong` | `phase-pool8` | [0.287, 0.131, 0.542] | 0.287 | 3/0 | **SYSTEMATIC COLD (every counted rep colder)** |
| `short_postlong` | `phase-pool17` | [0.178, 0.237, 0.226] | 0.226 | 3/0 | **SYSTEMATIC COLD (every counted rep colder)** |

### Expert-cache counters per launch (who was cold, and why)

`compulsory` = first touch of that expert in this process; `capacity` = evicted then needed again. The slab arm's pool is only ever read by its DECODE steps (its prefill goes to the slab), so its compulsory count is a direct read of how cold decode was.

`pread s` is CUMULATIVE across the pool's worker threads, so it is a cost an arm paid, not wall time. `resident MiB` is the pool's high-water mark: a pool that is full while its first-touch misses stay high is cold, not small.

| arm | rep | hit% | hits/misses | compulsory | capacity | evictions | file reads | pread s | resident MiB | prefetch issued/dropped | drops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `phase-slab` | 1 | 86.4 | 7643/1208 | 1208 | 0 | 1018 | 75465 | 4243.9 | 6429.53 | 685/112 | {'#3 drain_cleared': 112} |
| `phase-slab` | 2 | 86.3 | 7625/1210 | 1210 | 0 | 1022 | 75360 | 3740.2 | 6429.53 | 670/110 | {'#3 drain_cleared': 110} |
| `phase-slab` | 3 | 86.3 | 7638/1211 | 1211 | 0 | 1037 | 75258 | 3612.0 | 6429.53 | 638/90 | {'#3 drain_cleared': 90} |
| `phase-slab` | 4 | 86.3 | 7645/1209 | 1209 | 0 | 1016 | 75294 | 4977.0 | 6429.53 | 673/117 | {'#3 drain_cleared': 117} |
| `phase-pool8` | 1 | 98.3 | 467102/8313 | 6445 | 1868 | 8201 | 23994 | 98.4 | 6430.62 | 4/0 | - |
| `phase-pool8` | 2 | 98.3 | 467102/8313 | 6445 | 1868 | 8201 | 23994 | 57.5 | 6430.62 | 4/0 | - |
| `phase-pool8` | 3 | 98.3 | 467102/8313 | 6445 | 1868 | 8201 | 23994 | 80.9 | 6430.62 | 4/0 | - |
| `phase-pool8` | 4 | 98.3 | 467102/8313 | 6445 | 1868 | 8201 | 23994 | 109.6 | 6430.62 | 4/0 | - |
| `phase-pool17` | 1 | 97.6 | 330631/8159 | 6336 | 1823 | 8047 | 23130 | 60.0 | 6430.62 | 4/0 | - |
| `phase-pool17` | 2 | 97.6 | 330631/8159 | 6336 | 1823 | 8047 | 23130 | 115.7 | 6430.62 | 4/0 | - |
| `phase-pool17` | 3 | 97.6 | 330631/8159 | 6336 | 1823 | 8047 | 23130 | 84.8 | 6430.62 | 4/0 | - |
| `phase-pool17` | 4 | 97.6 | 330631/8159 | 6336 | 1823 | 8047 | 23130 | 85.8 | 6430.62 | 4/0 | - |

## Which graph served each request

`evidence=line` means a `CGC-PREFILL-STREAM` line carried that request's exact token count. `evidence=predicate` means the line was not available (the slab trace is capped at **4 lines per process**, so later requests usually have none) and the classification comes from the phase rule the engine itself applies.

| arm | rep | cap | decode width | slab | cell | prompt tok (log) | served_by | evidence | hooks | nil | nan/assert |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `phase-slab` | 1 | 8 | 8 | armed | short_warm | 12 | slab | line | 0 | 0 | 0 |
| `phase-slab` | 1 | 8 | 8 | armed | long | 2233 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 1 | 8 | 8 | armed | first_step | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 1 | 8 | 8 | armed | short_postlong | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 2 | 8 | 8 | armed | short_warm | 12 | slab | line | 0 | 0 | 0 |
| `phase-slab` | 2 | 8 | 8 | armed | long | 2233 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 2 | 8 | 8 | armed | first_step | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 2 | 8 | 8 | armed | short_postlong | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 3 | 8 | 8 | armed | short_warm | 12 | slab | line | 0 | 0 | 0 |
| `phase-slab` | 3 | 8 | 8 | armed | long | 2233 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 3 | 8 | 8 | armed | first_step | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 3 | 8 | 8 | armed | short_postlong | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 4 | 8 | 8 | armed | short_warm | 12 | slab | line | 0 | 0 | 0 |
| `phase-slab` | 4 | 8 | 8 | armed | long | 2233 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 4 | 8 | 8 | armed | first_step | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-slab` | 4 | 8 | 8 | armed | short_postlong | 12 | slab | predicate | 0 | 0 | 0 |
| `phase-pool8` | 1 | 8 | 8 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 1 | 8 | 8 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 1 | 8 | 8 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 1 | 8 | 8 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 2 | 8 | 8 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 2 | 8 | 8 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 2 | 8 | 8 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 2 | 8 | 8 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 3 | 8 | 8 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 3 | 8 | 8 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 3 | 8 | 8 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 3 | 8 | 8 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 4 | 8 | 8 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 4 | 8 | 8 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 4 | 8 | 8 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool8` | 4 | 8 | 8 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 1 | 64 | 17 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 1 | 64 | 17 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 1 | 64 | 17 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 1 | 64 | 17 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 2 | 64 | 17 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 2 | 64 | 17 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 2 | 64 | 17 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 2 | 64 | 17 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 3 | 64 | 17 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 3 | 64 | 17 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 3 | 64 | 17 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 3 | 64 | 17 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 4 | 64 | 17 | not armed | short_warm | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 4 | 64 | 17 | not armed | long | 2233 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 4 | 64 | 17 | not armed | first_step | 12 | pool | predicate | 0 | 0 | 0 |
| `phase-pool17` | 4 | 64 | 17 | not armed | short_postlong | 12 | pool | predicate | 0 | 0 | 0 |

## What this does and does not settle

- It settles whether arming the slab as a **profile default** has a speed justification (decision D: put `CGC_PREFILL_STREAM=1` in the pool-regime profile cells, rather than choosing it at runtime from memory state, which would not be fingerprintable).
- It does **not** settle correctness: bit-identity across pool sizes belongs to the oracle gate (`Backup/m123_oracle_gate`, `--profile prefill250`), not to this script.
- The decode numbers in the raw JSON are incidental: decode is T=1 and can never enter the prefill graph, so a decode-side slab A/B is a false negative **by construction**.

## Caveats that must travel with these numbers

- One launch of this model on this box has been measured from 110 to 247 t/s for a fixed command, so only the PAIRED ratios within this interleaved session are trustworthy.
- Launch order dominates and the live variable is carried swap, not free memory (`docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md` section D). Per-launch free/swap is in each JSON's `before`.
- `short_warm` and `short_postlong` are the same request in two cache states; quote them separately or not at all.
- Both engine traces are capped (slab 4 lines, `CGC-HOOK` 80 lines per process), so trace absence is never evidence of the other path; see the `evidence` column.
- The window gate is the launcher's own (`free_pct >= 40`, no other llama process). Every poll is in `watch.log`.
