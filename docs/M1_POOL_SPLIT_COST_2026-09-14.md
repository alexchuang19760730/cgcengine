# M1 work item 1 — full-width expert tensors + an independently allocated pool: measured cost

Status: **implemented, measured, and rejected.** The configuration is env-gated (`CGC_POOL_SPLIT=1`,
default OFF) and now logs a loud warning at enable time. The shipping default (adopted-region pool)
is unchanged and verified healthy in the same session.

**Errata and follow-up (2026-09-16).** Two corrections and one resolution, all in §4:

1. Blocker B's evidence was **mis-transcribed**. The capture says `st[1]=2`, not `st[1]=1`; there is
   no slot shared by two experts anywhere in it. See §4's errata note.
2. The knob named in §2 and §6, `CGC_IDENT_VERIFY`, **does not exist in the source**. The identity
   slot verifier is `LLAMA_EXPERT_CACHE_VERIFY_IDENTITY=1`. The reproduce command in §6 has been
   corrected.
3. Blocker B is **root-caused and fixed** (`ensure_batch` two-pass assignment, §4.1). Blocker A is
   untouched, so the rejection in §5 stands as written.

## 1. What "work item 1" asked for, and why it looked attractive

The plan (`M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md`) notes the coupling at `llama-context.cpp:263`:
with the L4 pool active, `n_batch` is clamped to `cgc_pool_max_tokens()` (default 8). The reason is
that the loader **shrinks** every expert tensor's `ne[2]` to the pool capacity, so a batch wider than
the capacity would index a capacity-sized tensor with full-range expert ids (`0..255`) -> OOB -> NaN.

Work item 1 proposed removing the shrink: keep the expert tensors full width, allocate the pool as
its own buffer, and lift the clamp so prefill chunks 512/2048 become legal. The hoped-for prize was
prefill throughput, with the pool decoupled from the tensor geometry.

## 2. What was implemented

| Piece | Where |
|---|---|
| `CGC_POOL_SPLIT=1` parse (numeric, `=0` is OFF) | `llama.cpp:~376` |
| Loader keeps `ne[2]` full width for expert tensors | `llama-model-loader.cpp` (`expert_cache_pool_split`) |
| Pool owns one buffer per kind (gate/up/down) | `llama_expert_cache_pool_split_alloc` |
| Per-(layer,kind) region adoption against that buffer | `llama_expert_cache_pool_split_adopt` |
| Geometry installed deterministically at graph build (`data` + `buffer` + `ne[2]` together) | `llama-context.cpp` `graph_get_cb` |
| Identity prepopulate actually **preads** into an owned pool | `llama-expert-cache.cpp` `llama_expert_cache_prepopulate` |

The last one is a real, reusable fix. The legacy path can say "slot `e` holds expert `e`" for free,
because the loader pre-read experts `0..capacity-1` into the tensor storage that *is* the adopted
pool region. An owned pool is a fresh allocation, so the same call would have marked uninitialised
Metal memory as resident. `prepopulate` now performs the identity pread when
`llama_expert_cache_pool_owned()` — the same bytes, the same order, and it is byte-checked at load by
the existing identity-slot verifier (`LLAMA_EXPERT_CACHE_VERIFY_IDENTITY=1`).

## 3. Measured

All numbers from one binary, Nail IQ3_XXS-denseIQ4X, MTP ON, 8 GiB pool budget, 16 GB box, no other
server resident.

**Pool allocation (the "cost" the item was asking about):**

| kind | size | Metal buffers |
|---|---|---|
| 0 (gate) | 1886.02 MiB | 1 |
| 1 (up) | 1886.02 MiB | 1 |
| 2 (down) | 2549.94 MiB | 1 |
| **total** | **6.32 GiB** | **3** |

Two facts worth keeping. First, the pool is **3 Metal buffers, not 120** — the residency-set
explosion that killed the earlier P1 mmap-stream attempt (120 buffers -> `requestResidency` IOKit
per buffer -> load hang) does **not** recur under this structure, and `buffer is nil` never fired
(0 occurrences). Second, 6.32 GiB is what a 143-slot pool costs at full width, against an 8 GiB
budget: the shrink was not hiding a multiplier.

**Baseline (shipping default, same binary, same session) — healthy:**

| prompt | prefill | decode | RSS |
|---|---|---|---|
| short (11 tok) | 5.41 tok/s | 5.05 tok/s | 9.40 GB |
| long (164 tok) | 11.38 tok/s | 7.87 tok/s | 9.40 GB |

(That baseline ran without the nothink template kwargs, so the model emits ` thinking` and the numbers
are slightly pessimistic; it is used here only as a liveness/regression check.)

**Split mode: unusable.** It loads, adopts all 120 regions, and then fails numerically.

## 4. Why it fails — two independent blockers

**Blocker A: the wide path is the one configuration Metal is known not to read.** Lifting the clamp
let a 16-token prefill batch build with the wide geometry, and the log says so explicitly:

```
CGC-POOL-SPLIT-GEOM: il=1 kind=1 ntok=16 mode=wide ne2=256 data=0x351d45de0 buf=0x1400832c0
```

A wide step means Metal reading a 256-expert tensor through the model's weight buffer — exactly the
configuration the loader shrink exists to avoid (plan §1.1: `buffer is nil`; and the P1 mmap-stream
residency explosion). The clamp was therefore restored: with the clamp, **every** step (decode, MTP
verify, prefill chunk) runs the pool path, which is the only expert-weight path Metal is known to
read correctly here. Prefill chunk size is bounded by pool capacity in *both* modes; that bound is an
invariant of the design, not an artifact of the shrink.

**Blocker B: the pool path itself mis-assigns slots under an owned pool.** With the clamp restored
and the identity prefill in place, routing was still degenerate and the run still died:

```
CGC-WARM verify n_past=1 warm=0 fast=0
CGC-HOOK: il=1 ntok=2 ids=[0 1 2 3 4 5 6 7]
CGC-PRE:  il=1 st[0]=-1 st[1]=1 st[2]=2 st[3]=3 st[4]=4 st[5]=5 st[6]=6 st[7]=7
CGC-POST: il=1 st[0]=1  st[1]=2 st[2]=3 st[3]=4 st[4]=5 st[5]=6 st[6]=7 st[7]=8
CGC-SLOT: il=1 st[0]=1  st[1]=2 st[2]=3 st[3]=4 remap=[1 2 3 4 5 6 7 8]
CGC-HOOK: il=2 ntok=2 ids=[-1096546439 ...]    <- NaN bit pattern
```

> **Errata (2026-09-16).** The first version of this section quoted `CGC-POST: il=1 st[0]=1 st[1]=1`
> and read it as "expert 0 and expert 1 share slot 1 — two experts own one slot". The capture,
> `Backup/cgc_logs/llama_server_20260914_094821.log` lines 322-326, says `st[1]=2`. With that
> correction the three lines are mutually consistent term by term — `remap[i] == st[ids[i]]`, and
> `CGC-SLOT`'s four printed values are simply `st[0..3]`. Nothing in the capture shows two experts
> on one slot, and no such line exists in any 09-14 log.

The real anomaly is a **uniform +1 shift of the whole layer's expert->slot map**, and 8/8 experts
being re-assigned instead of 1/8: **one cold expert turned a hits-only batch into a full batch of
misses.** Three facts in `ensure_batch` / `pick_slot` compose into it, and none of them is wrong on
its own:

* `prepopulate`'s identity fill leaves the pool **full** and every `slot_last_use` at **0**, so the
  LRU victim is the lowest index that is not skipped.
* `pick_slot`'s eviction **clears the victim's slot-table entry**
  (`slot_table[layer*n_expert + evicted] = -1`, `llama-expert-cache.cpp:~555`). An eviction is
  therefore not private to the miss that caused it: it converts a resident member into a miss.
* `ensure_batch` filled `batch_owned` — the mask whose stated job is "the slots this batch reads,
  never evict mid-batch" — **incrementally, inside the same loop that assigned the misses.** Only
  members visited *before* a given miss were protected. A member visited *after* it had not been
  added to the mask yet and was still an eviction candidate.

Batch `ids=[0..7]` against a full pool: expert 0 is cold -> miss -> evicts the slot holding the next
member -> that member is now cold -> miss -> evicts the slot holding the next one -> cascade. Each
member is handed the slot of the member after it, which is exactly the `st[e] = e+1` in the capture.
The reason the first victim is slot 1 rather than slot 0 is that slot 0 was not an eviction
candidate at that instant (busy, or pinned); the log alone cannot pin which `pick_slot` skip fired,
and the cascade does not depend on it — any first victim at index `k` produces the same +1 shift for
every member after `k`.

The NaN at `il=2` is the *downstream* signature, not the cause: with the map shifted by one, every
expert reads another expert's weights, and on a pool whose resident range ends at the last usable
slot the top of the range is shifted past it — an OOB read of the pool region, i.e. the class that
produces NaN rather than merely-wrong logits. (This last step is inference from the shifted remap
plus the observed signature; the shift itself is proven, see §4.1.)

The reason the legacy path never hit this is that there the pool **is** the expert tensor's own
storage, so "slot `e` holds expert `e`" is literally true before any fill runs, `pick_slot`'s
free-slot scan (`owner[i] < 0`) can see genuinely free slots, and — decisively — the pool is never
full at the moment a batch is assigned, so the eviction that starts the cascade never fires.

### 4.1 Blocker B: root-caused and fixed (2026-09-16)

`ensure_batch` now assigns in **two passes**: pass 1 claims the slot of every *resident* member,
pass 2 then assigns the misses — so a victim chosen in pass 2 can never be a slot this batch reads.
Pass 1 also **adopts an in-flight fill** (`slot_owner[s] = e` with `slot_queued`/`slot_loading` set
but `slot_table[e]` not yet published): the old hit test looked at `slot_table[e]` alone, so such an
expert read as cold and pass 2 would have given it a second slot. A new counter
(`n_hit_adopted_queued`) makes that path visible instead of silent.

The fix is a reordering, so the evidence has to be a **discriminating** one. It is
`scripts/check/expert_cache_ensure_batch_order.cpp`: no model, no server, no I/O — a synthetic cache
in exactly the post-`prepopulate` state (pool full, every `slot_last_use` 0, one cold member whose
slot is held by an in-flight fill) handed a 16-member batch. It links the real
`llama_expert_cache_ensure_batch` out of `libllama`, so it tests the shipped function, not a copy.

| arm | `st[e]` after the batch | hits/misses | exit |
|---|---|---|---|
| HEAD (one-loop) | `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 0/16 | FAIL |
| fixed (two-pass) | `16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15` | 15/1 | PASS |

The HEAD row is the 09-14 capture's mapping reproduced **bit for bit** (`st[0]=1 st[1]=2 ... st[7]=8`)
and its 0/16 hit count is the "one cold expert becomes a full batch of misses" claim. The fixed row
leaves every resident member exactly where it was and gives the single cold expert a slot no member
is reading.

Two honest limits on this evidence:

* It is a **unit** test of the assignment order, not a served request. The live-run confirmation
  could not be made on this box: the split configuration dies in warmup with
  `CGC-METAL-FAIL: ... status 5 (Insufficient Memory)` at both 10 GiB and 4 GiB pool budgets on a
  16 GB machine, before any request is served. A 4 GiB fixed-arm run did reach **layer 39** (the
  09-14 run died at il=2) and reported `CGC-BATCH-INVARIANT: il=1/il=2 OK` on the way, but that run
  also OOM'd in warmup, so it is a liveness improvement, not a quality result.
* **Corrected 2026-09-16 — see §4.2.** An earlier version of this section claimed that in a
  **non-split** run (the shipping default) the two-pass split is a **no-op**, on the grounds that the
  pool is never full during *warmup*: misses land in free slots and LRU eviction never fires. The
  observation itself is real — with the fix stashed and the same 4 GiB configuration re-run, the old
  and new binaries produced **bit-identical** `CGC-PRE`/`CGC-POST`/`CGC-SLOT` output — but it was
  measured on **warmup**, which is not the workload this code serves. A served prefill chunk against
  the 8 GiB pool touches 256 experts, the pool *does* fill, and pass 2 *does* evict; §4.2 counts
  exactly that. What the bit-identical result establishes is only that nothing was broken where
  nothing was full — it is *not* a test of the fix, and it is not a statement about the configuration.

A gate (`LLAMA_EXPERT_CACHE_BATCH_INVARIANT=1`, plumbed through `run_server.sh`'s `SERVER_ENV`
allow-list so a production launch can actually set it) now re-checks the post-conditions it used to
assume — distinct slots, `slot_table[e]` reads back, `slot_owner` agrees — on every layered batch.
It is verified, not assumed, because the original failure was **silent**: every selected expert still
had a slot, just the wrong one.

### 4.2 The reordering, measured on the configuration we actually ship (2026-09-16)

Everything §4.1 could say about the *live* path was on the **split** configuration — the one that
cannot finish warmup on this 16 GB box. That made the live evidence circular: the only workload that
could exercise the fix was a workload that does not run. This round moved the observation onto the
**shipping default** by adding three counters to the cache and printing them from its destructor, then
running the production entry point on the branch tip (`HEAD=0a4bfd73c`,
`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`, 256 experts, 40 layers):

```bash
CGC_SERVER_PROFILE=prefill250 LLAMA_EXPERT_CACHE_BATCH_INVARIANT=1 ./scripts/run_server.sh
# then one chat request, then SIGTERM -- the destructor is the only place these stats are printed
```

| quantity | measured | what it settles |
|---|---|---|
| prefill, 2873 tokens | **227.62 t/s** (12622.1 ms) | the profile runs at the tip |
| `batch_evict_batches` | **325** | pass 2 **does** evict on a served request ⇒ the two-pass *order* is load-bearing here, not a no-op |
| `batch_invariant_checks` | **331** | the gate ran on every layered batch, not only during warmup |
| `batch_invariant_violations` | **0** | none of its post-conditions (distinct slots, `slot_table[e]` reads back, `slot_owner` agrees) was violated |
| `CGC-BATCH-INVARIANT` OK / VIOLATIONS | 24 / **0** | the gate's own output agrees with the counters |
| final cache stats | `hits=9545/12149 (78.6%)`, `resident=6430.62 MiB` | the run did real work, not a stub |

Why the two counts differ is part of the point: `batch_evict_batches` (325) counts `ensure_batch`
calls that evicted, the OK log line is throttled to `il<=2` (hence only 24 lines, with a per-line `n`
ranging 3–32), and `batch_invariant_checks` (331) counts **all** `ensure_batch` calls — including the
zero-miss decode batches that the gate's previous placement skipped entirely. Read together they say:
evictions happen, and every batch that could have been checked was. The 24 lines are a *sample* for
the reader; the 331 is the coverage claim.

That retires the "no-op on the shipping configuration" reading. The reordering is exercised **and**
verified green on the configuration we actually ship — the two claims §4.1 could only make for the
configuration that OOMs. The unit test above still matters for the opposite reason: it is the only arm
where the *unfixed* behaviour is observable side by side.

**Boundary this measurement exposed, and does not cover.** The **second** request on this build kills
the server with a SIGSEGV that is not in this code path — it is in the MTP draft path
(`llama_get_embeddings_nextn` → `ggml_metal_synchronize`), and it reproduces with the
batch-invariant gate both on and off. The first request is unaffected (292.07 t/s cold, 227.62 t/s
warm). Full evidence in `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §11.

## 5. Conclusion and recommendation

**Reject work item 1 as specified.** On this engine the pool must remain the expert tensor's own
storage. Decoupling them is not a refactor of the pool; it is a rewrite of the ownership model
(`slot_owner` / `slot_table` / `batch_owned` / prepopulate identity) *and* of how Metal is allowed to
see expert weights, and the measured prize is not prefill throughput — the clamp survives either way.

Blocker B being fixed (2026-09-16, §4.1) **does not change this verdict**, and it is worth being
explicit about why, because the two blockers were the stated reason for the rejection. Blocker B was
a genuine engine bug in `ensure_batch`, and it is now repaired in the shipping code — but it was a bug
in the *path the split experiment exercised*, not a property of the ownership model. Blocker A is
untouched and is structural: with the clamp restored, prefill chunk size is bounded by pool capacity
in both modes, so work item 1's actual prize (512/2048-token prefill chunks) does not exist even in
the configuration where the split works. The experiment's negative result therefore stands, and it
now stands on one blocker instead of two.

What the experiment did buy:

* A **real engine fix**: `ensure_batch`'s batch-ownership mask never implemented the exact-set
  invariant its own comment claimed, which made any assignment against a FULL pool order-dependent
  (§4.1). That is a latent bug in the shipping path too — it is simply unreachable there while the
  pool is never full at assignment time. With this repair, a future prefetch/pinning variant that
  does fill the pool no longer silently corrupts the map.
* A **discriminating regression test** for it, `scripts/check/expert_cache_ensure_batch_order.cpp`
  (runs in milliseconds, no model needed), plus the `LLAMA_EXPERT_CACHE_BATCH_INVARIANT` gate.
* The pool-as-own-buffer structure costs **6.32 GiB in 3 buffers** and does not trip the residency
  problem that killed the earlier mmap-stream attempt. If a future design needs a decoupled pool, the
  allocation side is already proven cheap.
* `prepopulate` now fills an owned pool correctly instead of marking empty memory resident, and that
  fill is byte-verified at load.
* A clean negative result, with the exact failure signature, so the next attempt starts from the
  ownership model rather than from the loader.

The genuine lever for prefill remains M2 (whole-layer sequential streaming), which needs work item 1
in the sense of *wide tensors as a read source* — but as a **host-side source that Metal never maps**,
fed to the pool by the existing pread/gather machinery, not as a tensor the graph can dispatch on.

## 6. Reproduce

**Blocker B's mechanism and fix — fully reproducible, no model needed:**

```bash
cd src/llama.cpp
clang++ -std=c++17 -O1 -I include -I ggml/include -I src \
  ../../scripts/check/expert_cache_ensure_batch_order.cpp \
  -L build/bin -lllama -Wl,-rpath,$PWD/build/bin \
  -o ../../scripts/check/expert_cache_ensure_batch_order
DYLD_LIBRARY_PATH=build/bin ../../scripts/check/expert_cache_ensure_batch_order
# case 2 was added 2026-09-16; the test now runs both and prints ALL PASS (2/2)
#
# === CASE 1: order dependence (pool full, one cold member) ===
# post   : st[0]=16 st[1]=1 st[2]=2 ... st[15]=15
# misses : 1  hits: 15
# counters: batch_evict_batches=1 adopted_queued=0
#   (an eviction during assignment is what makes the two-pass ORDER load-bearing)
# === CASE 2: in-flight fill adoption ===
# post   : st[0]=0 st[1]=1 ... st[7]=7       misses: 0  hits: 8  adopted_queued: 1
#   (the in-flight fill was adopted; one expert, one slot; no second assignment)
#
# The one-loop arm is the historical comparison for CASE 1:
# HEAD   -> post: st[0]=1 st[1]=2 ... st[15]=16   misses: 16  hits: 0   FAIL
```

It links `llama_expert_cache_ensure_batch` out of the built dylib, so it exercises the shipped
function. Rebuild the library from the same tree state before rebuilding the test — the struct is
passed by pointer across the dylib boundary and a stale-dylib/fresh-header pair crashes inside
`ensure_batch` instead of failing to link (the test's header comment has the details).

**The split configuration itself — NOT reproducible as of 2026-09-16.** The probe that produced the
§3/§4 numbers lived in `/tmp/m1split_probe.py` and was never committed; `/tmp` has since been
cleared, so it is gone. Reconstructing it needs the flag set `CGC_POOL_SPLIT=1 CGC_POOL_SPLIT_DBG=1`
against a model + profile that fits this box, and on a 16 GB machine it now dies in warmup
(`CGC-METAL-FAIL: ... status 5 (Insufficient Memory)`) at both 10 GiB and 4 GiB pool budgets before
serving a request. The identity verifier, if you re-run it, is
`LLAMA_EXPERT_CACHE_VERIFY_IDENTITY=1` — **not** `CGC_IDENT_VERIFY=1`, which is the name §2 and §6
used before the 2026-09-16 errata and which exists nowhere in the source.

Evidence logs: `Backup/cgc_logs/llama_server_20260914_094239.log` (pre-fix) and
`Backup/cgc_logs/llama_server_20260914_094821.log` (post-fix, lines 322-326 are the §4 capture).
`Backup/` is gitignored; the numbers above are the extract, and because the logs cannot be committed
they are the only durable copy of the raw lines — not the probe that generated them.
