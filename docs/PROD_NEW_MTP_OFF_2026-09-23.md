# prod-new (MTP off) — the production cell, measured 2026-09-23 20:59:45

**One line**: on a clean window (no rival llama process, thermal NOMINAL for all 150 samples),
the profile's own production cell gives **prefill 283.0 t/s** (the 250+ target, met) and
**decode 11.72 t/s** (the profile's own 13–14 target, **missed by 1.3–2.3**).

Identity — this is what makes the two numbers quotable:

| field | value |
|---|---|
| HEAD | `155ee2474` |
| instrument | `commit_bench.py --record-only` → `llama_bench_matrix --arms prod-new` (llama-bench) |
| shape | `-p 2048 -n 128 -d 512 -r 3` |
| server binary | `054fb22f04a01c5cb93b362883dd3f82` |
| `libllama.0.0.578.dylib` | `64ecc0c8d70d5a2b32298cf4d0c18f82` |
| `libggml-metal.0.19.0.dylib` | `128b6048870fcc55949b6ab28fa4bd3b` |
| `libllama-server-impl.dylib` | `60fb7910a8bb7d38a6b8bb6d31808ffc` |
| profile env | MTP off · `b=ub=5632` · ctx 8192 · pool 8 GiB · `PREFILL_STREAM=1` · `SLAB_CAP=256` · `SPAC=1` · `OA_ASYNC=1` · `MM_BITIDENT=1` · `PREFIX_REUSE_CKPT=1` (no-op with MTP off) |
| wall | 76.6 s |
| machine window | zero llama processes at launch; the neighbour session's `llama-bench` has lstart 21:04:27, i.e. **after** this run ended 21:01:02 |
| swap in use | 5526 MB at launch → 5678 MB at end (held by processes outside this run) |

| cell | t/s | samples | sd | target |
|---|---:|---|---:|---|
| prefill `p2048` | **283.01** | 293.79 / 280.17 / 275.08 | 9.68 | 250+ → **met** |
| decode `n128 d512` | **11.72** | 12.39 / 11.03 / 11.73 | 0.68 | 13–14 → **missed** |

Decode at 11.72 t/s is 85.3 ms/token. The pool counters from the same run:

- hit **96.3 %** (124 138 / 128 920), misses 4 782 = **compulsory 4 030 (84.3 %)** + capacity 752 (15.7 %)
- resident **6 197 MiB of 8 192** — the pool is 24 % empty, so capacity is not what is binding
- worst layer 2: 192 distinct experts against 143 slots; 5 layers over slots
- I/O: 82 119 jobs / 2.32 GiB ⇒ 23.8 ms per job, **1.0 MiB/s effective**

⚠ Those I/O counters are **not** used above to claim "decode is I/O bound". `pread_usec` sums to
1 950 s inside a 76.6 s run, so its aggregation unit is not established; and this cell ran with
`CGC_DECODE_PROFILE` **off**, so there is no per-layer `wait / cb / submit` breakdown to separate
I/O wait from compute. The cell answers *what prod-new delivers*, nothing finer.

## Two defects this run exposed (neither is the profile's numbers)

1. **`commit_bench.py` cannot read its own profile.** `extract()` selects on
   `row["params"]["test"]`, but the matrix rows carry `n_prompt / n_gen / n_depth` at the top level
   (and this build's llama-bench JSON has `test: None`). It therefore returns `(None, None)` and the
   verdict print dies with `ValueError: Unknown format code 'f' for object of type 'str'` — no
   verdict, and no `[commit-gate] prefill=… decode=…` line to paste into a commit message, which is
   the one thing `NEXT_ACTIONS_2026-09-23.md` mandates. The numbers above were read out of the
   matrix JSON by hand. (File is untracked, so it is not fixed here.)
2. **The gate's paths are shared defaults.** `/tmp/commit_bench/` and `/tmp/commit_bench_result.json`
   are what every invocation writes unless told otherwise; a concurrent session overwrote this run's
   `.stderr.log` at 21:04:27. This product deliberately carries **no log-derived number** because of
   that. Two agents running the mandated gate at once will silently swap each other's evidence.

## What the reading does and does not settle

- It settles the deliverable question for this profile right now: **prefill is above target, decode
  is below it.**
- It does **not** settle whether the cache/prefetch carriers help decode, because there is no
  same-build control arm in this cell. That needs `prod-new` with the carriers off
  (`SPAC=0 OA_ASYNC=0`, and the slab/stream arm as a third cell) under the repo's ABBA protocol
  (≥300 s cool-down, interleaved arms, paired medians) — three arms × 3 reps, one quiet window.

The binary measured here also carries **ungated debug prints** (`CGC-SEG` every 160 segments,
`CGC-HOOK` for the first 80 routed layers, `CGC-POST/PRE/WARM` 24 times each); they are in `HEAD`,
not from the working diff. By this repo's own rule, an instrumented reading cannot be placed next to
an uninstrumented anchor — this one is instrumented.
