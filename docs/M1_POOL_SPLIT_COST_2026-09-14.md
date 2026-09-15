# M1 work item 1 — full-width expert tensors + an independently allocated pool: measured cost

Status: **implemented, measured, and rejected.** The configuration is env-gated (`CGC_POOL_SPLIT=1`,
default OFF) and now logs a loud warning at enable time. The shipping default (adopted-region pool)
is unchanged and verified healthy in the same session.

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
the existing identity-slot verifier (`CGC_IDENT_VERIFY`).

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
and the identity prefill in place, routing is still degenerate and the run still dies:

```
CGC-HOOK: il=1 ntok=2 ids=[0 1 2 3 4 5 6 7]
CGC-PRE:  il=1 st[0]=-1 st[1]=1 st[2]=2 ... st[7]=7
CGC-POST: il=1 st[0]=1  st[1]=1 ...            <- expert 0 and expert 1 share slot 1
CGC-SLOT: il=1 remap=[1 2 3 4 5 6 7 8]
CGC-HOOK: il=2 ntok=2 ids=[-1096546439 ...]    <- NaN bit pattern
```

Two experts ended up owning one slot. The remap then reads that region twice, and the layer's output
is a NaN, which propagates to il=2 and faults the Metal encoder (SIGSEGV, no `GGML_ASSERT`, no
`buffer is nil`). This is the same terminal failure signature as the pre-fix run, i.e. the fix did
not reach the cause.

The reason the legacy path never hits this is structural, not incidental: there the pool **is** the
expert tensor's own storage, so "slot `e` holds expert `e`" is literally true before any fill runs,
and `pick_slot`'s free-slot scan (`owner[i] < 0`) can see genuinely free slots. Once the pool is a
separate allocation, the identity/ownership bookkeeping has to be re-derived from scratch, and the
batch-ownership mask that the 2026-08-30 fix introduced (`batch_owned`) is not sufficient on its own.

## 5. Conclusion and recommendation

**Reject work item 1 as specified.** On this engine the pool must remain the expert tensor's own
storage. Decoupling them is not a refactor of the pool; it is a rewrite of the ownership model
(`slot_owner` / `slot_table` / `batch_owned` / prepopulate identity) *and* of how Metal is allowed to
see expert weights, and the measured prize is not prefill throughput — the clamp survives either way.

What the experiment did buy:

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

```bash
# baseline (healthy)
python3 /tmp/m1split_probe.py base 8

# split (aborts by design after the alloc/adopt log lines)
CGC_POOL_SPLIT=1 CGC_POOL_SPLIT_DBG=1 CGC_IDENT_VERIFY=1 python3 /tmp/m1split_probe.py split 8
```

Evidence logs: `Backup/cgc_logs/llama_server_20260914_094239.log` (pre-fix) and
`Backup/cgc_logs/llama_server_20260914_094821.log` (post-fix). `Backup/` is gitignored; the numbers
above are the extract.
