# The expert-cache → Metal channel, both ends: it violates all three principles, and the shipped instrument cannot see it

**Date** 2026-09-23 (22:00–22:35) · carrier Nail IQ3_XXS-denseIQ4X · worktree `flashkv-devserver`
@ `2cf45a3d7` (`demo/sweet-spot-windows-fix`)

## 0. One line

The channel is not limited by the SSD (warm-measured **0.93 GB/s at one expert-blob, 2.85 GB/s at
depth 8**; the layout offers **86–110 MiB contiguous per (layer, kind)**) nor by Metal. It is limited
by its own shape: **one 0.34–0.56 MB read per (layer,expert,kind), merged only on file-adjacency,
joined to zero once per layer** — and the line this repo quotes for "how big are our reads"
(`read shape: … MiB/job`) **divides two different populations**, so it is not a physical read size.

## 1. Producer end (expert cache → pool), read from the code and the file

| property | value | where |
|---|---|---|
| read unit | 1 × (layer, expert, kind) = **335,872 / 344,064 / 450,560 / 557,056 B** | GGUF's own table: `blk.20.ffn_down_exps` ne=[512,2048,256] IQ3_S, nbytes/256 = 450,560 |
| what the layout offers | expert dim is **ne[2]** ⇒ 8 adjacent ids = **3.44 MiB contiguous**; whole (layer,kind) = **86–110 MiB contiguous** | `llama-model-loader.cpp:1817` (`entry.file_offset = offs + e*expert_bytes`) |
| merge policy | sorted by (file_idx, file_offset); **file-contiguous runs only**; gate/up/down are three different tensors ⇒ never merge with each other | `llama-expert-cache.cpp:3335-3377` |
| batch = one layer's misses | `ensure_batch` flattens the layer's misses (≤8 experts × 3 kinds) and submits them, then **`pool_done_cv.wait(outstanding == 0)`** | `:1290-1310` (default branch), `:3395` |
| concurrency | **8** persistent workers (`LLAMA_EXPERT_CACHE_WORKERS`, not overridden by prod-new) | `:3684-3698` |
| speculative path | `prefetch_slot` skips resident experts, then the fill runs **on the single bg thread** (or inline on the caller via `ensure_slot`), one segment per read, no merge with anything | `:1422-1443`, `:3148`, `:989`, `fill_segments_merged_serial` (added 2026-09-23, in HEAD) |

## 2. Consumer end (pool → Metal)

| property | value | where |
|---|---|---|
| landing | pread writes **directly into the expert tensor's own host-visible Metal buffer** (`pool_ext`) — zero copy, correct | `adopt_pool_region`, `cgc_gather_slab_get` comment |
| wide-union / prefill slab | cap × whole-model max stride = 256 × 557,056 ≈ 142.5 MiB/kind ⇒ **≈356 MiB** for 3 kinds | `cgc_gather_slab_get` |
| GPU dependency | `mul_mat_id` indexes rows by **slot index**; the CPU writes that remap leaf **once per layer**, and to do so it pulls the ids back to the host (`tensor_get_async` + `synchronize`) ⇒ **41 device→host→device round trips per step** | `ggml-backend.cpp:1650-1660` |
| dispatch order | submit seg[i] → **wait for all of its command buffers** → run the hook (which may block on I/O) → submit seg[i+1] | `CGC_OA_ASYNC` branch, `ggml-backend.cpp:1782-1810` |

Both ends therefore stop the world per layer, and they stop it *alternately*.

## 3. Physical probe (the missing measurement)

`scripts/check/expert_read_shape.py` — same file, same offsets from the GGUF table, `F_NOCACHE` so
the probe neither reads nor pollutes the page cache.

| shape | run 1 (22:0x, quiet) | run 2 (22:33, neighbour loading) |
|---|---:|---:|
| 1 expert (450,560 B) depth 1 | 456 µs · 0.92 GB/s | 452 µs · 0.93 GB/s |
| 1 expert depth 8 | 332 µs · **1.26 GB/s** | 147 µs · 2.85 GB/s |
| 8 experts (3.44 MiB) depth 1 | 1,210 µs · 2.78 GB/s | 1,523 µs · 2.20 GB/s |
| 8 experts depth 8 | 1,003 µs · **3.35 GB/s** | 299 µs · 11.5 GB/s |
| 64 experts (27 MiB) depth 8 | 5,813 µs · 4.62 GB/s | 1,917 µs · 14.0 GB/s |

**The régime is not controlled, and that is a finding of its own**: the two runs are 25 minutes
apart on the same shapes and differ by **3.4×**, because another session's `llama-bench` started at
22:25:16 and re-warmed the file (and shared the device). `F_NOCACHE` stops the probe from populating
the cache; it cannot make the read cold. So:

* **quote neither number as the device's ceiling or floor.** A cold reading requires the probe to be
  the first reader in a quiet window, and neither of these was.
* what survives both régimes is the **ordering**: the current unit at depth 8 is 2.3–2.8× faster than
  at depth 1, and multi-MB units are another 1.2–4× faster. The shape question is real; the magnitude
  is régime-dependent.

## 4. The verdict on the three principles

1. **CPU/GPU overlap: violated, and it is the first-order one.** The producer joins to zero once per
   layer (`pool_done_cv.wait`); the consumer waits out the whole segment before the hook and only then
   submits the next. Neither resource runs while the other does. In the delivery régime the hook's own
   cost is `cb` ≈ **42 ms median / 50.4 ms mean per step** (`CB_DELIVERY_SETTLED_2026-09-23`), i.e.
   50–58% of an 86 ms step.
2. **Bandwidth: violated as a consequence.** The run's own counters say **82,119 reads over 76.6 s
   wall = 1,072 reads/s**; the run's decode phase is 3 × 128 tokens at ~11.7 t/s ≈ **33 s**, so
   ≈**2,500 reads/s during decode** — against a depth-1 capability of ~2,200 reads/s *per worker*
   (452 µs, run 1). The 8 workers are therefore busy for roughly a seventh of the decode window; the
   per-layer join is the candidate, and it costs nothing in correctness to remove (the reads for layer
   *L+1* do not depend on layer *L*'s ids).
3. **Parallelism: violated.** Depth is a fixed 8, raised and drained once per layer, and the
   speculative path is a single thread (or inline on the caller). The saturating shape — 3.44 MiB per
   read at depth 8 — is **free from the file layout**; it needs no model change and no extra RAM.

## 5. The instrument defect (found while doing this; in HEAD)

`fill_job` (`llama-expert-cache.cpp:33-45`) **increments `n_reads` but not `n_read_bytes`**.
`n_read_bytes` is added only by the merged-iov branch (`:153`) and by the pool worker
(`:3240`, `:3254`). The printed line

```
llama_expert_cache: read shape: jobs=N bytes=B (0.04 MiB/job)  us/job=…  effective_rate=… MiB/s
```

therefore has a **numerator from the pool path and a denominator that also counts the bg/prefetch
single-segment reads** — and the more the merge fails, the smaller the printed MiB/job becomes. The
0.04 MiB/job figure quoted in the 2026-09-23 prefetch comment is that artefact, not a physical read
size. The same line's `us/job` is bounded by the issuer count: **23,755 µs/job × 82,119 jobs = 1,951 s
of summed thread time inside a 76.6 s wall**, while 8 workers cap that at 613 s — so at most one of
`us/job`, `effective_rate` and the path attribution can be right, and the line does not say which
path it is describing. (The `fill_segments_concurrent` spawn path *can* exceed the 8-worker bound, and
that is exactly the point: without a path tag the line is uninterpretable.)

**Until this is fixed, `MiB/job` and `effective_rate` must not be cited** — including for the
"unmerged small reads flood the device" claim, whose direction may well be right but whose number
this line cannot support.

## 6. The two A/B runs that finished in the same window

`ab_interleave.py` (interleaved, paired per-rep ratios, ≥300 s cooling, build fingerprint, window
gate); products `Backup/phase_decomp/ab_nospac_prodnew.json`, `ab_protect_prodnew.json`.

| arm | paired ratios | verdict |
|---|---|---|
| `CGC_SPAC=0` vs baseline | r1 1.097, r2 1.098, r3 0.980 | paired median **+9.7%**, but baseline drifts 11.65→13.66 (+17.3%) across the round; using r2/r3 only ⇒ **+3.9%**. Honest range **+4–10%**, not a clean number |
| `CGC_PREFILL_PROTECT=1` vs baseline | r1 1.027, r2 1.044 | **+3.6%**, and this round's baseline is 12.81/12.82/12.86 (**±0.2%** — the tightest baseline measured in this line) |

Both point the same way: **I/O the engine produces on its own initiative is net cost in the delivery
régime**, which is what a shape that cannot keep the device busy looks like.

## 7. The shape that satisfies the three principles (ordered by payoff, not by elegance)

1. **Stop joining per layer.** Issue layer *L*'s misses and continue; await at layer *L*'s own consumer
   (the mechanism exists as `fill_begin`/`fill_await` but is structurally blocked — the segmented path
   only calls back *after* the segment completes, and moving it one segment later produced all-NaN
   logits). Bound from run 1: at the *current* read unit, depth 8 is worth **1.26 GB/s** whenever the
   workers are not idle.
2. **Raise the read unit to MBs.** Because the expert dim is ne[2], a multi-expert slice is one
   pread: 3.44 MiB at depth 8 measured **3.35 GB/s** (run 1). This is second-order to (1) but it is
   free of new metadata — it only needs the fill set to be *bigger than one layer's misses*, e.g. the
   whole layer (which the prefill slab path already does: 283 t/s prefill).
3. **Demote speculative I/O.** §6 says removing it is worth +4–10%; §1 says its reads are single
   segments on one thread, i.e. the worst possible issue pattern for the channel it competes with.
4. **Fix the instrument first** (§5) — otherwise (1)–(3) cannot be scored, because the line that would
   score them reports a number that is not a read size.

## 8. Boundaries

* **No engine change and no rebuild in this round.** `llama-expert-cache.cpp` is clean at
  `2cf45a3d7`; the counter fix (§5) is a one-line change, but committing it triggers pre-commit check 8
  (source ⇒ build artifacts staged + rebuilt), and a rebuild now would compile *other sessions'*
  uncommitted engine edits (`ggml-backend.cpp` 18:33, `llama-context.cpp` 18:28, `qwen35moe.cpp`
  16:42) into a binary this commit would then own. Left to whoever rebuilds next, on purpose.
* **The read-path numbers in §3 are warm.** See §3: no quiet-window, first-reader probe has been run.
* **`pread_usec` and `file_reads` in `final stats:` are the same counter family** and inherit the same
  ambiguity; `CB_DELIVERY_SETTLED` already withdrew the "74.18 ms" figure that was read off it.
* Zero residue: the probe starts no server and leaves no process; the only `llama-bench` on the box at
  22:25:16 is another session's (PPID 81918), started after my measurements.

## 9. Commands

```
python3 scripts/check/expert_read_shape.py --model models/gguf/<model>.gguf
CGC_SHAPE_MODEL=models/gguf/<model>.gguf python3 scripts/check/expert_read_shape.py --selftest
python3 scripts/check/decode_carrier_ab.py --json Backup/phase_decomp/ab_nospac_prodnew.json
python3 scripts/check/ab_interleave.py --arms baseline,protect-on --reps 3 --rounds 3 \
    --warmup 1 --n-predict 120 --profile prod-new --min-idle-s 300 --json <out>.json
```
