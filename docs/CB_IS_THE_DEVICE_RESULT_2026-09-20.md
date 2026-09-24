# cb is the device, not the engine — F1's two terms are one physical quantity

**Date:** 2026-09-20 · **Status:** measurement closed · **Engine change: none · Processes: none**

## The question this answers

F1 decomposed the decode `cb` (the MoE top-k hook) into two "structural terms":

```
cb_layer = 0.46 ms  +  0.695 ms x m        (m = that layer's miss count that step)
             ^barrier      ^per-miss
   x ~29 miss-bearing layers/step, ~74 misses/step
   =>  barrier 13 ms (19%)  +  per-miss 51 ms (77%)  = 64.7 ms vs 66.6 measured (96%)
```

The next step was assumed to be two engine changes. **Both terms are the device.** There is no
engine-side inefficiency of 64 ms to recover, and the one remaining engine-side item is ~6–13 ms.

## 1. The engine is the device (ratio 0.98)

Two independent readings, same 0.37 MiB segment shape, same file:

| source | per-job cost | implied rate |
|---|---|---|
| engine's own fill audit (09-19) | **1.698 ms/job** | 226 MiB/s per worker |
| bare `pread` on this device, 8 concurrent cold reads | **1.731 ms/job** | 214 MiB/s per worker |
| **engine / device** | **0.98x** | — |

The engine pays essentially exactly the device's price for a cold segment. There is nothing
between 1.70 and 1.73 ms for a code change to remove.

## 2. F_NOCACHE: the earlier "cold" was never verified

The previous probe inferred coldness ("pass 1 is probably cold, pass 2 is hot"). Its first
version of this sweep read up to **2.6 GB/s** on a device documented at 823 MB/s — i.e. it was
measuring page cache, and a "device ceiling" claim built on it would have been unfounded.

This run sets `fcntl(fd, F_NOCACHE, 1)` and **checks** it:

```
cache check: same 8 offsets read twice -> 1.71 ms then 1.70 ms (repeat/first = 0.99)
ok: no speedup on repeat -> the device is being read, not the cache
```

The script refuses to report (exit 3) if a repeat is ≥1.4x faster. That guard is the reason the
table below is quotable.

## 3. Throughput is flat in read shape — the merge-width lever is dead

F_NOCACHE, 8 workers, 24 MiB per size, 3 rounds, disjoint offsets, median:

| size | jobs | wall | aggregate | ms/job |
|---|---|---|---|---|
| 0.37 MiB | 64 | 14.6 ms | **1647 MiB/s** | 1.731 |
| 0.75 MiB | 32 | 15.9 ms | 1509 MiB/s | 3.506 |
| 1.50 MiB | 16 | 16.6 ms | 1443 MiB/s | 7.124 |
| 3.00 MiB | 8 | 16.0 ms | 1500 MiB/s | 11.173 |
| 6.00 MiB | 4 | 17.5 ms | 1373 MiB/s | 12.041 |
| 12.00 MiB | 2 | 15.6 ms | 1540 MiB/s | 12.690 |

Spread is 1373–1647 MiB/s (a 1.20x band) with **no monotone trend**. Note the shape of the
experiment: the 12 MiB row uses only **2** threads and still reaches 1540 MiB/s, while the
0.37 MiB row uses 8 threads for 1647 MiB/s — the ceiling is a property of the device, not of how
many streams are aimed at it.

**⇒ Widening the merge is worth nothing.** This is F3's verdict reached by its *correct* reason:
F3 was closed earlier because `effective_rate x workers` (1760 MiB/s) was read as "above copy
speed". The real reason is stronger — the device's aggregate throughput does not respond to read
size, so there is no larger shape to buy.

## 4. What this does to F1's two terms

| term | F1's reading | what it actually is | reducible by code? |
|---|---|---|---|
| **per-miss 0.695 ms** | "disk latency per miss" | **device bytes**: 1.11 MiB at the device's 1.6 GiB/s = 0.68 ms | **No.** Only fewer/smaller bytes |
| **barrier 0.46 ms** | "per-layer barrier" | a **two-regime fit artifact**: the regression put queueing-curve mass into the intercept. Direct residual (device wall for the same 3·m jobs vs measured cb_layer) is **+0.16…+0.28 ms** at m≤4 and **negative** at m≥6 | **Partly — 6–13 ms/step total** |

The negative residuals at m≥6 are informative: the engine is **faster** than my scattered reads at
high m, because merge-read collapses file-contiguous experts into fewer jobs. So the engine's
I/O is already better than the naive shape in the regime where volume is largest.

Honest statement of the engine-side headroom: **6 ms/step** (residual median 0.134 ms x ~29
miss-bearing layers) to **13 ms/step** (F1's regression intercept 0.46 ms x 29). Call it
**3–6% of a 220 ms step**: 16.1 t/s → 16.6–17.1 t/s.

## 5. Why cb cannot be engineered away

The miss set is ~74 experts/step x 1.11 MiB = **82 MiB/step**, and 82 MiB at the device's
1.6 GiB/s is **50 ms** — which is the measured per-miss term almost exactly. Those bytes must
move. `cb` is the engine *waiting* for them, and it is a **bandwidth floor**, not a latency
artifact: no overlap, no prefetch, and no submit-order change can move a byte faster than the
device moves it.

This is also why the prefetch family failed (F2R): the predictor's candidates were 100% already
resident, and even a perfect predictor cannot help a bandwidth floor — it can only move *when*
the wait happens, not how long the bytes take.

## 6. Where the remaining levers actually are

Since per-miss is bytes, only two things move it, and both are on the bytes side:

1. **Fewer misses.** F1's census: **25.6% compulsory / 74.4% capacity**. Compulsory (first-ever
   touch) is unreachable by any predictor or policy. Only the 74.4% capacity share is policy-
   addressable: 74.4% x 51 ms = **38 ms**, and that requires a perfect policy.
   Upper bound if cb fell to its floor (13 ms compulsory + ~10 ms overhead): step 220 → 177 ms
   → **18.9 t/s**. Still short of 25 on cb alone.
2. **Fewer bytes per miss.** cb scales linearly with expert byte size. This is the lever already
   on the project's record ("the remaining gap lives in the IQ2_S experts") — and it is a
   quantisation choice, not an engine change.

**⇒ The user-facing conclusion: neither F1 term is an engine defect.** Doing "both engine
changes" recovers 3–6%, because the two terms are one physical quantity — the device — and the
engine already runs at 0.98x of it.

## 7. Artifacts

- `Backup/phase_decomp/device_read_size_sweep.py` — this measurement, with the F_NOCACHE
  bypass check that refuses to report uncached-looking numbers
- `Backup/phase_decomp/pread_cost_probe.py` — the engine/device per-job comparison
- `docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md` — the F1 decomposition this revises
- `docs/F2_F5_OVERLAP_AND_AUDIT_RESULT_2026-09-20.md` — F2–F5

## 8. Honest caveats

- The engine's 1.698 ms/job is from a **09-19** log; the device's 1.731 ms/job is measured now.
  The 0.98x ratio is therefore cross-day. Both are the same file and the same shape, and the
  probe's cache check passes, but this is not a same-window paired measurement.
- 6 writers on this box at the time; the device ceiling may be partly contention. That makes the
  ceiling *lower* than a quiet box, which **strengthens** the conclusion (a quiet box would not
  make the engine relatively better).
- `barrier` is given as a range (6–13 ms) on purpose: the regression intercept and the direct
  residual disagree, and picking one would be false precision.
