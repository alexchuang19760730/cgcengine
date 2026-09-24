# The five-number run: one clean window, delivery shape, all five numbers in one log

Date: 2026-09-20, 22:40–22:59. Tool: `Backup/phase_decomp/five_number_run.py` (selftest 12/12).
Run evidence: `Backup/phase_decomp/five_number_20260920_224505.json`,
log `Backup/phase_decomp/five_20260920_224615/llama_bench_prod25_..._p0_n128_d512_r3.stderr.log`.

## 1. What was asked, and what was written down first

§EN-351 left "is decode 25 reachable" undecided because the step floor had been read at two
incompatible values from two different runs:

| reading | `union_sum` | `cb` | regime |
|---|---:|---:|---|
| 023021 log | **149.24 ms** | 74.18 (29.9%) | high swap |
| 214450 log | **97.90 ms** | 11.26 (9.2%) | startup free 84% |

`cb` was shown to be 6.6× water-level sensitive (§EN-350), so the two runs are not comparable.
The criteria were fixed **before** the run and are evaluated by the tool, not by the reader:

```
union_sum ~= 149                       -> branch A: 25 is dead, rewrite the target
union_sum ~= 98-110 and total 120-135  -> branch B: 25 is an ml >= 3.06-3.4 problem
anything else                          -> INDETERMINATE, decide nothing
```

## 2. What actually happened

| time | attempt | outcome |
|---|---|---|
| 22:40 | sentinel | **DEGRADED** 174.88 t/s (0.62× of the 281.51 median) — no run |
| 22:45 | sentinel + run | sentinel **HEALTHY 250.19** (floor 239.28) → run completed, rc=0, 106 s |
| 22:51 | sentinel | **DEGRADED** 197.87 (0.70×) — refused, correctly |
| 22:57 | after 4 min idle (thermal 0, free 83%) | refused: another session's `harness.py run prod_profile` |
| 22:58 | — | that session launched `llama-server` (pid 514); window lost |

**One completed run in a sentinel-certified window.** The warm-skip-64 (true delivery) repeat is
still owed; the box was taken before it could be taken.

## 3. The five numbers (one log, one arm, one window)

```
mean_len     2.612    (384 generated tokens / 147 work rows)
union_sum    175.59 ms
gap_sum      101.84 ms
cb            76.18 ms
step          285.85 ms  (DECPROF work-row median)   -- and 561.83 ms as the round (see §5)
```

`wait` 185.06, `submit` 17.79, 518 step rows total, 147 of them work rows
(`ntok == 4 && gap_sum > 0`).

**Pre-registered verdict: branch A.** `union_sum` 175.59 ≥ 135. The branch-B band (88–125) was
never approached — not in the median, not in the best third (see §6).

## 4. Two parser defects, both caught by construction, not by inspection

**(a) `mean_len` 0.871 — a number that cannot exist.** llama-bench reports `n_gen` **per rep**
(128) while CGC-DECPROF counts rounds across **all** reps (147). Dividing 128 by 147 gives 0.871,
which is below the hard floor of 1.0 token per round. Fixed to `GEN * REPS / cycles` = 2.612, and
the floor is now enforced: `mean_len` outside [1, 4] returns `INVALID` instead of a branch. This is
the third time on this line that a quotient was assembled from two regimes — the failure
`mean_len_direct.py` was written to prevent.

**(b) The cycle/token pairing does not actually depend on `--warm-skip`.** Run 1 used
`--warm-skip 0` reasoning that the counts had to be aligned. They do not: warm-skip shrinks the
counted tokens and the counted rounds by the same fraction, so `mean_len = GEN * REPS / cycles`
holds either way. Run 1 paid a real cost for that mistake (the cold pool was averaged into the
t/s) for no benefit.

## 5. ★ `step` is not one number, and the gap is bigger than we thought

| | this run (bench) | 214450 (server) |
|---|---:|---:|
| `step_decprof` (DECPROF work-row `total`) | 285.85 ms | 129.57 ms |
| `step_round` (`mean_len / tps`) | **561.83 ms** | **188.47 ms** |
| share of the round outside the profiled window | **49%** | **31%** |

`CGC-DECPROF total` is the graph_compute window only. The draft model, sampling and the rest of the
round sit outside it. The §EN-351 isoline `step = mean_len × 40 ms` is a t/s statement, so it
needs `step_round`, not `step_decprof` — using the profiled number overstates the headroom by
roughly 2×. Here the honest headroom is `104.49 − 561.83 = −457 ms`, i.e. 25 t/s is not a
near-miss; it is ~5× away.

## 6. ★ The run was not in steady state — and it is not the cold pool

Splitting the 147 work rows into thirds (one per rep):

| segment | n | `total` | `union_sum` | `cb` |
|---|---:|---:|---:|---:|
| rep 1 | 49 | 261.91 | **159.95** | 87.04 |
| rep 2 | 49 | 279.76 | 175.59 | 81.21 |
| rep 3 | 49 | 294.46 | **201.32** | 65.53 |

The cold-pool hypothesis predicts the opposite: slow first, faster later. What happened is a
**monotone +26% rise in `union_sum` across 106 s of full load** with `cb` falling — a GPU clock
walking down, i.e. the box heating during the measurement. This is the same phenomenon the sentinel
caught at 22:40 and 22:51 from the outside; here it is visible *inside* a single run.

Consequence: a single ~106 s full-load decode run on this box does not have a steady state to
report. Every historical decode number on this line was taken under this drift.

## 7. What this does and does not settle

**Settled:** in the delivery shape, tonight, on a sentinel-HEALTHY window, `union_sum` reads
**160–200 ms** — the branch-A regime. The 97.90 reading was not reproduced.

**Also settled, and new:** `cb` was 76.18 ms even though the box idled at **83% free** before the
run. §EN-350 attributed the 74.18-vs-11.26 `cb` split to the water level; startup-free is evidently
not the discriminator, so that explanation needs revising (candidates: the 8 GiB pool being
resident-and-dirty by the time decode starts, or the shape — `-d 512` / `-c 4096` here versus the
server's short context).

**Not settled:**

1. **Provenance.** `llama-bench` is the 22:16 build. The working tree carries another session's
   uncommitted engine edits (`llama-context.cpp` +191 at 22:30, `llama-expert-cache.cpp` +380 at
   22:28, both in `process_ubatch` / `expert_cache_on_topk` — the hot path). This run measures the
   built binary, not the tree. Fingerprints are in the JSON.
2. **The warm-skip-64 repeat.** Run 1 includes the cold pool (`--warm-skip 0`), so its 4.65 t/s is
   not a delivery number and the decomposition is inflated by an unquantified amount.
3. **Instrumentation cost.** `CGC_DECODE_PROFILE_ALL` + `CGC_GPU_TIMING` are on; `union_sum` is a
   GPU-clock span and should be robust to printing, but the 49%-outside gap in §5 is large enough
   that the instrumentation cannot be assumed free.

## 8. Recommendation

Treat branch A as the working hypothesis and act on it: **25 t/s is not reachable at k=3 in the
delivery shape**; the target should be rewritten (16) or the recipe changed. Do not re-open the
question with more arithmetic — the next useful measurement is a warm-skip-64 repeat in a window
that is certified both before *and* after the run, since §6 shows the box can degrade during it.
