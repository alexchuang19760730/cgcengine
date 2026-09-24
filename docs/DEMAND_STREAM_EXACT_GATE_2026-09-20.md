# The replay gate is now exact: 40/40 and 41/41 layers, by construction

**Date:** 2026-09-20 · **Engine change:** demand-stream instrument (`llama-expert-cache.cpp/.h`) ·
**Leftover processes:** 0 · **No timing is quoted anywhere in this document**

The previous report (`REPLAY_GATE_RESOLUTION_2026-09-20.md`) ended on a negative: the replay gate
could not be made exact by modelling the victim rule, because the residual was **demand the trace
cannot see**. It named the fix — instrument the demand array rather than the hook's union — and that
is what this delivers.

| arm | pool | slots/layer | per-layer miss counts | multiset | replay hits = engine hits | replay misses = engine misses |
|---|---|---|---|---|---|---|
| MTP-off (`nail_nomtp`) | 4 GiB | 71 | **40/40 exact** | **identical** | 95,050 = 95,050 | 19,127 = 19,127 |
| MTP-on (`nail`) | 8 GiB | 143 (142 usable) | **41/41 exact** | **identical** | 23,843 = 23,843 | 8,610 = 8,610 |

Four independent agreements per arm, against two different engine counters (`n_hits`, `n_misses`) plus
the `MISS_DUMP` multiset and its per-layer breakdown. The residual is **zero** and is no longer a
tolerance: the replay consumes the engine's own input stream, so a future mismatch is a
policy-modelling error rather than a missing input.

## 1. What was added

`LLAMA_EXPERT_CACHE_DEMAND_DUMP=<path>` logs every operation that changes a layer's residency or LRU
order, one line per call, written **while `cache->m` is held** so the file order is the mutation order:

| tag | site | payload |
|---|---|---|
| `S` / `s` | `ensure_slot`, counted / prewarm | expert |
| `B` | `ensure_batch` — one **atomic batch** | the union |
| `N` | `ensure` (the map path) | expert |
| `T` | `touch` — the MTP verify fast path | the union |
| `P` | `prefetch_slot` **reserved** a slot | expert, slot |
| `D` | `prefetch_slot` dropped (nothing evictable) | expert |
| `L` | a queued fill **landed** (bg published `slot_table`) | expert, slot |
| `Z` | `zero_reserved_slot` | — |
| `X` | `prepopulate` (installs the identity mapping) | slot count |
| `Y` / `U` | `pin_experts` / `unpin_all` | union / — |

Two design points are load-bearing rather than cosmetic:

* **`P` is logged at the decision, not at entry, and `L` exists at all.** `prefetch_slot` writes
  `slot_owner` at *queue* time while `slot_table` is only written when the bytes land. A stream that
  carried only `P` would be unmodelable — it could not say when a reservation became a hit. `L` is
  that event. (And a `prefetch_slot` that finds the expert already resident logs **nothing**, so
  "no event" cannot be misread as an effect.)
* **The header is `v2`, and the reader rejects `v1`.** The first cut logged `P` at entry with one
  payload field. Same tag, different payload — a v1 stream parsed as v2 would read `ids[1]` off a
  one-field event and model a reservation that never happened. Both arms' v1 dumps are on disk and
  are now **refused** (`exit 2`), which is the behaviour that was verified.

Reader: `scripts/check/reuse_distance.py --demand-dump … --miss-dump …`, which replays with the
engine's own rule reduced to the terms that are live here — pass-1 claim / pass-2 evict, argmin
`last_use`, strict `<` for the tie-break, reservations not evictable, `touch` moving LRU without
touching any counter. **57 self-tests, 0 failing**, including must-fail controls for each: a batch's
own member surviving, a repeated cold expert counting per occurrence, `touch` reordering LRU,
`ensure_slot` misses being excluded from the compared list, and refusals for `Y`/`U`/unknown tags,
a v1 stream, a wrong-arity `P` and an unpaired `L`.

## 2. Three findings that fell out of building it

**(a) The MTP-off "+2840 requests over demand" was the prewarm, and it is now measured directly.**
The census counts `S=2840`, and 2840 = **71 slots × 40 layers** exactly: `prewarm_hot` calls
`ensure_slot` with `count=true` for each of its top-71 picks, and those land in `n_requests`. The
previous report attributed +1.26% to "demand the trace cannot see" — correct in spirit, now exact.

**(b) `prefetch=0/0` could not distinguish "never called" from "called 100k times, all no-ops" —
and the answer is the second one.** In the v1 capture the hook made **100,137** `prefetch_slot` calls
in the MTP-off arm; in the v2 capture that arm made **zero** reservations; and `final stats` reports
`prefetch=0/0` in *both* runs, because `prefetch=%zu/%zu` is `n_prefetch/n_prefetch_dropped` and the
already-resident early return incremented **neither**. So the counter was structurally blind to the
only thing that was happening.

That is the mechanism behind the prefetch family being repeatedly judged "no candidates" in earlier
rounds: the hook prefetches exactly the union members it is about to demand, so by the time it asks,
they are resident. Fixed with a third counter — the engine now also prints

```
llama_expert_cache: prefetch detail: reserved=<n> dropped=<n> already_resident=<n>
```

⚠ **This counter post-dates the captures below.** The captures were taken on libllama
`7aba58c6…`; the counter is in `95030f01…`. The `0/0` readings above are from the captured logs and
are quoted as they were emitted.

**(c) The two arms are not one variable apart — so they must not be read as an MTP on/off A/B.**
The census is the evidence: MTP-off shows prefetch activity, MTP-on shows `T` and `Z` (the verify
fast path) and **no** prefetch at all. `run_server.sh` sets `CGC_NO_PREFETCH`, `CGC_VERIFY_DECODE`,
`CGC_DRAFT_DECODE` and `CGC_MM_BITIDENT` only inside its `SERVER_MTP=1` block, so `MTP=0` is a block
of env changes, not one flag. This is the same asymmetry flagged for `plain_match_ab.py`; it applies
to every MTP-off arm in this line, including the earlier reuse-distance captures. Each arm's gate is
valid against its own run's log — but a *difference between the arms* is not attributable to MTP.

## 3. Provenance and how to reproduce

| | MTP-off | MTP-on |
|---|---|---|
| demand stream | `Backup/phase_decomp/demand_off.txt` (15,760 events) | `Backup/phase_decomp/demand_on.txt` (14,535 events) |
| engine miss record | `Backup/phase_decomp/miss_off.txt` (19,127) | `Backup/phase_decomp/miss_on.txt` (8,457) |
| server log (counters, `MISS_DUMP`) | `Backup/cgc_logs/llama_server_20260920_140519.log` | `Backup/cgc_logs/llama_server_20260920_141049.log` |
| capture record | `Backup/phase_decomp/exact_off.json` | `Backup/phase_decomp/exact_on.json` |
| gate output | `Backup/phase_decomp/exact_gate_off.json` | `Backup/phase_decomp/exact_gate_on.json` |
| driver | `Backup/phase_decomp/exact_gate_capture.sh` | (same script, arm 2) |
| `final stats` | `requests=114177 hits=95050 misses=19127` | `requests=32453 hits=23843 misses=8610` |

Same model (`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`), temp 0.4, 3 prompts, both on
libllama `7aba58c613289566a08deb2d04647488`; `--need-mb 5400` (below the launcher's documented
8000 MB floor, recorded in each provenance block).

```bash
# the gate — offline, no server, no window needed
python3 scripts/check/reuse_distance.py \
  --demand-dump Backup/phase_decomp/demand_off.txt \
  --miss-dump   Backup/phase_decomp/miss_off.txt \
  --json        Backup/phase_decomp/exact_gate_off.json
# exit 0 exact / 1 mismatch / 2 refused
```

To re-capture: `bash Backup/phase_decomp/exact_gate_capture.sh` (both arms, same two dumps).
**Never quote a speed from an instrumented run** — the dump writes a line per event under the cache
mutex.

## 4. What this does and does not cover

**Does:** the decode-side pool path for both arms, at both geometries, including the MTP head's own
256-slot layer, `touch`, prepopulate, the zero-slot reservation and the prewarm.

**Does not:**
* **Measured-but-untested.** `P`/`L` are implemented and unit-tested, but **no reservation occurred
  in either capture**, so the reservation path is exercised by the self-tests only. That is stated
  rather than glossed: an untested-but-modelled site is where a silent divergence would live.
* **Refused, not modelled:** `Y` (pin_experts) and `U` (unpin_all). Both are placement features that
  are off in these profiles; the gate exits 2 rather than dropping the events.
* **Decode only.** Neither arm took the prefill slab path, so this says nothing about prefill's
  demand stream.
* **Policy, not quality.** The gate proves the replay models the cache. It says nothing about the
  model's output, and nothing about speed.
