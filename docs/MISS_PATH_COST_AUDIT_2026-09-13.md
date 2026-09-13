# Expert miss-path cost audit — is 823 MB/s the disk or a page-cache hit?

Date: 2026-09-13 · Branch: `demo/sweet-spot-windows-fix` · Model: IQ3_XXS-denseIQ4X (13.66 GB),
8 GiB pool, `mtp` profile, `--no-mmap`, box = MacBook Air M4 16 GB.

## Geometry first (so the benchmark reads the engine's real ranges)

From the GGUF's own consecutive tensor offsets (my hand-typed type-size table was wrong — it
claimed 32 KiB/expert where the file says 1.07 MiB):

| item | value |
|---|---|
| expert layers | 41 (`blk.0`…`blk.40`) |
| per layer | `ffn_down_exps` IQ3_S 115.3 MB + `ffn_gate_exps` IQ2_S 86.0 MB + `ffn_up_exps` IQ2_S 86.0 MB |
| per expert | 1 122 304 B = **1.07 MiB** (down 450 560, gate 335 872, up 335 872) |
| one layer, 256 experts | 287.3 MB (**287 MiB** — matches the 277 MiB/layer figure from the churn work) |
| model total expert data | 41 × 287.3 MB ≈ **11.8 GB of the 13.66 GB file** |

## Method, and why attempt 1 was wrong

`F_NOCACHE` marks an fd so reads are not cached. **It does not force a miss on a page that is
already resident** — so a "cold" measurement is only cold on a region the process has never
touched. Attempt 1 ignored this: an earlier HOT pass had primed the regions it then called cold,
producing 15–20 GB/s "device" reads (impossible). Attempt 2 takes every cold measurement from a
disjoint virgin region and measures the `F_NOCACHE` semantics directly:

```
A) 5 consecutive reads of a VIRGIN 32 MiB region, F_NOCACHE
   rep0  29.14 ms  (1098 MiB/s)   <- true device
   rep1  22.83 ms  (1402 MiB/s)
   rep4  22.68 ms  (1411 MiB/s)
```
rep0 ≈ rep1 (1.3×), so `F_NOCACHE` really is bypassing the cache here: ~1.4 GB/s is the device,
not a hit. (Contrast with a primed read of the same bytes: 7.7 GB/s.)

## The numbers

| regime | measured |
|---|---|
| **Device cold sequential**, 256 MiB virgin, single read | **1404 MiB/s** (182 ms) |
| Device cold, 64 MiB virgin | 1225 MiB/s |
| Device cold, 64 KiB single read | 0.443 ms → **fixed floor ≈ 0.44 ms per cold call** |
| Device cold, 1 MiB | 0.950 ms |
| Device cold, 4 MiB | 2.499 ms (1601 MiB/s) |
| Device cold, 16 MiB | 11.031 ms (1451 MiB/s) |
| **Page-cache HOT**, same 64 MiB, primed | **7721 MiB/s** (8.29 ms) |

So on this box: **a miss costs ≈ 5.5× a page-cache hit per byte**, and a cold call carries a
~0.44 ms floor that dominates anything under ~1 MiB.

## In situ: what the engine actually does

Instrumented `preadv` jobs and bytes (new counters, printed mid-run by `CGC-RIG-SNAPSHOT` — the
teardown line is unreliable, see below). One request = 67-token prefill + 32-token decode:

| metric | value (identical across 4/5 repetitions) |
|---|---|
| bytes fetched | 5 094 604 800 B = **4.86 GiB per request** |
| preadv jobs | 15 951 per request |
| mean job | **0.30 MiB** |
| µs per job | 1 838 – 2 772 |
| per-worker rate | 115 – 174 MiB/s |

## Verdict 1 — 823 MB/s is the DEVICE, not a cache hit

* A page-cache hit is **7721 MiB/s** here. 823 MB/s is 9× slower, so it is not cache.
* Aggregate across the default 8 pool workers, same 0.305 MiB per job:

  | run | µs/job | implied aggregate | % of the 1404 MiB/s ceiling |
  |---|---|---|---|
  | in-situ run 1 | 2772 | 880 MiB/s | 63 % |
  | in-situ run 2 (MERGE arm) | 1838 | 1327 MiB/s | **94 %** |

  So the device runs at **63–94 %** of its cold ceiling depending on the run; per-job latency is
  queueing-inflated (0.305 MiB should cost ~0.55 ms cold single-threaded, we see 1.8–2.8 ms), and
  at the better end it is effectively saturated.
* Either way the "823 MB/s" figure was never the device's serving rate: it is a per-worker rate
  (115–182 MB/s × 8 workers) or the *payload* rate over the whole request wall (18.98 GiB over
  44 s of 4 requests = 442 MiB/s), both of which fold in compute.

## Verdict 2 — Prefetch cannot REMOVE this cost

The bytes are structural, and one earlier hope is now dead:

* Per request the engine must fetch 4.86 GiB because the per-layer working set (~149 experts)
  exceeds the pool (143 slots/layer) — so re-reads are not an artifact of eviction policy.
* **Bigger merged reads are NOT a lever.** A/B with `LLAMA_EXPERT_CACHE_NO_MERGE=1`, same bytes:

  | arm | jobs | bytes | MiB/job | µs/job | per-worker MiB/s |
  |---|---|---|---|---|---|
  | MERGE (default) | 63 804 | 20 378 419 200 | 0.305 | 1838 | 174 |
  | NO_MERGE | 67 104 | 20 378 419 200 | 0.290 | 2040 | 149 |

  Merging collapses only **4.9 %** of calls (the fill batches are per-step unions, e.g. ≤ 4
  tokens × topk 8 = 32 experts/layer, in which adjacent-expert pairs are rare). MERGE is 17 %
  better per worker (174 vs 149 MiB/s) but that is well inside the run-to-run spread, and with
  8-way concurrency already pushing the device to 63–94 % there is little headroom for fewer/
  larger calls to buy throughput — only per-call latency, which the queue hides.

## Verdict 3 — what prefetch CAN buy, and its ceiling

I/O is roughly *half* the request wall (summed `pread_usec` 176.9 s over 4 requests ÷ 8 workers
≈ 5.5 s, against an ~11 s request). So the prize is not bandwidth, it is **removing the
serialization between I/O and compute**: a prefetcher that fills layer *L+1* during layer *L*'s
FFN caps the request at max(io, compute) instead of io + compute, i.e. **up to ~1.8×** — and it
cannot go beyond that, because the device is already saturated and the byte count does not move.

Ordering therefore changes: the remaining lever is **overlap**, and it is bounded at ~1.8×. The
previously-suspected lever (merge into bigger sequential reads) is measured dead: only 4.9 % of
calls collapse, and the device is already 63–94 % busy.

Caveat on the 1.8×: it assumes I/O and compute can be fully decoupled, which for decode is not
free — the routing that decides layer L+1's experts only exists after layer L+1's attention, so
the prefetch window is the current layer's FFN, not the whole step.

## Incidental finding: teardown telemetry is unreliable — and there is a Metal shutdown assert

Two independent reasons every arm's counters used to vanish:

1. **Stop path.** `scripts/run_server.sh` stops the server with `kill -INT`. Sending `SIGTERM`
   (or `SIGINT` to the whole process group, which delivers it twice) makes llama-server print
   `Received second interrupt, terminating immediately.` and skip `~llama_expert_cache()` — so
   `final stats` / `read shape` never print. Signal the **script only** and let its trap forward
   exactly one INT.
2. **A real bug:** even on a clean single INT the shutdown can abort —
   `ggml-metal-device.m:657: GGML_ASSERT([rsets->data count] == 0) failed`, reached via
   `ggml_print_backtrace → ggml_abort → ggml_metal_device_free`. The Metal residency sets are
   not fully released before device free. This is pre-existing, is on the exit path only, but it
   is why teardown telemetry is a coin flip and why the mid-run `CGC-RIG-SNAPSHOT` is now the
   primary channel.

## Artifacts

- geometry: `/tmp/gguf_geom.py` → `/tmp/gguf_expert_geom.json`
- benchmark: `/tmp/miss_path_bench2.py` (virgin regions + F_NOCACHE semantics)
- in situ: `/tmp/miss_insitu2.py`, A/B log `/tmp/miss_ab.log`
- new counters: `n_read_bytes` + `read shape` teardown line + `pread_usec`/`read_bytes` in
  `CGC-RIG-SNAPSHOT` (all inert unless the rig env var is set)
