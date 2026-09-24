# Why the replay gate cannot be made exact by modelling the victim rule

**Date:** 2026-09-20 · **Engine change:** none · **Leftover processes:** 0

The request was: model `slot_decode_reserved` correctly so `reuse_distance.py`'s exact-match gate
passes instead of relying on a 98.8% tolerance.

**The premise is false, and it is false from the source rather than from an argument.** The rule was
never engaged in either capture, so modelling it can only make the replay *worse* — which is what
the earlier run already measured before I went looking.

## 1. `slot_decode_reserved` is provably inert in these captures

```cpp
// llama-expert-cache.cpp:912
const bool defer_decode = defer_decode_protect && cgc_prefill_protect_on();

// :470
static bool cgc_prefill_protect_on() {
    if (g_prefill_protect_override >= 0) return g_prefill_protect_override != 0;
    static const bool env_on = getenv("CGC_PREFILL_PROTECT") != nullptr;
    return env_on;
}
```

`CGC_PREFILL_PROTECT` is unset by default, and `run_server.sh:1984` only forwards it when set. It
was set in **neither** capture. So `defer_decode == false`, and both consultation sites are dead:

* `:550` `if (defer_decode && pass < 2 && slot_decode_reserved[i]) { n_defer_skip++; continue; }`
* `:595` `if (defer_decode && slot_decode_reserved[best_slot]) { ... = 0; }`

The flag is still **set** on every decode hit (`:1042`, guard `if (!defer_decode_protect)`, and
`defer_decode_protect` is `phase == CGC_PHASE_PREFILL` at the call site, false in a decode-only run)
— so it is written and never read. Measured consequence: replaying it took MTP-on from 20,470 to
20,747 misses (worse) and cut entry agreement, exactly as a rule-that-was-off should.

**⇒ Change made:** that variant is now behind `--prefill-protect`, off by default, with the source
lines in its `--help`. Previously the tool offered it unconditionally, which invited exactly this
false conclusion.

## 2. The real residual: `n_requests` counts fills, not demand

`n_requests` counts **fills**, not demand. The only increment in `llama-expert-cache.cpp` is
`cache->n_requests += n;` at `:968`, inside `fill_pool_direct` — the body of `ensure_batch`
(`:906`) — reached again from `llama_expert_cache_ensure` (`:3580`). Nothing else bumps it, and the
reason is specific to this engine:

```cpp
// llama-context.cpp:6499 — the MTP verify fast path (cgc_fast_eligible)
llama_expert_cache_zero_reserved_slot(cache, (uint32_t) il);
// LRU-touch resident experts only (no fill, no wait): keeps hot slots' freshness so
// the next step's pick_slot does not hand them out to a cold fill.
llama_expert_cache_touch(cache, (uint32_t) il, uni.data(), uni.size(),
                         cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP);
```

`llama_expert_cache_touch` (`:221`) refreshes LRU order and bumps `n_fast_union` / `n_fast_cold`
telemetry — it **never bumps `n_requests`**. So when the fast path serves a layer's whole union, only
the experts that are *actually cold* get counted, via the verify-strict prologue's
`ensure_batch(cold, ...)`. `requests` therefore counts **fills**, and in a regime that takes that
path it is *expected* to sit far below the trace's demand.

`CGC-IDS` is emitted by the hook and shows the union, so the trace sees **demand**. That makes a
testable claim for the regime where demand *does* go through `ensure_batch`: the miss residual should
be the **size** of the invisible demand (`requests - trace demand`). The tool now checks it and
reports the verdict instead of a hand-waved tolerance:

| | MTP-off | MTP-on |
|---|---|---|
| demand from the trace | 222,057 | 297,518 |
| engine `requests` (= fills) | 224,897 | 59,783 |
| requests - demand | **+2,840 = 1.26%** | **-237,735 (negative = fills < demand)** |
| replay miss residual | **+1.53%** | +7.71% |
| ratio | **1.21x -> CONSISTENT** | **INVALID** |

**MTP-off passes the falsifier at 1.21x** — there the demand and the fills are the same events, so
"demand the trace cannot see" is a real, bounded quantity and the residual is explained.

**MTP-on is INVALID, not passed**: the counter is *below* the trace's demand because the fast path
serves fills-free. The authoritative witness is the engine's own `refused` counter, printed on the
same line:

```
MTP-off  llama_server_20260920_131425.log: verify-strict: refused=0    zero_mapped_selected=0
MTP-on   llama_server_20260920_132729.log: verify-strict: refused=525  zero_mapped_selected=0
```

`refused` counts layer-calls where a still-cold selected expert forced the fast path to be abandoned
for the exact path (`llama-context.cpp:6483`, gated on `cgc_cold_after > 0`). **0 vs 525 is the
regime difference**, and it is a counter rather than a line count.

> **Self-correction (found by re-checking my own citations).** An earlier draft of this section
> cited `515 CGC-SYNCFILL lines in MTP-on, 0 in MTP-off` and `129 zero-slot lines`. Both are
> **print-capped line counts, not counters**: `CGC-SYNCFILL` prints only when `n_cold_filled > 0 && il <= 1`
> (`:6468`) and the `ZERO-slot refused` line only when `il <= 1 || n_verify_strict_refused <= 8`
> (`:6484`), so line counts measure *how many times the logger fired*, which is why my `129` was
> really 128 and why none of those numbers can be quoted as refusal totals. That is the exact
> defect this project keeps logging — reading a print cap as a count. The real numbers are the two
> `refused=` above, and the direction is unchanged; the *magnitudes I quoted before were wrong.*

## 3. Why no victim-rule change can ever fix this: the residual is uniform

A new per-layer check answers a question the global multiset cannot: is the replay wrong in a few
layers (a rule modelled badly) or a little in every layer (a trace limit)?

```
MTP-off: per-layer miss counts exactly equal: 0/40 layers
         worst: 22 L32 890/868, 22 L5 1077/1055, 21 L34 913/892, 20 L33 910/890
MTP-on : per-layer miss counts exactly equal: 0/41 layers
         worst: 62 L35 236/174, 55 L33 316/261, 52 L8 289/237, 51 L18 255/204
```

**0 of 40 layers match**, and the deltas are 20-22 for *every* layer — a small, near-constant
positive offset, not a concentrated failure. A mis-modelled eviction rule produces the opposite
signature: a few layers badly wrong. So:

* The residual is **not** a policy mismatch. It is a roughly constant amount of extra pool traffic
  per layer that the trace does not record — consistent with §2's 1.3-1.5%.
* **No amount of victim-rule modelling makes this gate exact.** The information is not in the trace.

## 4. What actually made it exact — DONE (see `DEMAND_STREAM_EXACT_GATE_2026-09-20.md`)

> **Superseded forward pointer.** The instrument this section proposed has since been built and the
gate is now **exact: 40/40 and 41/41 layers, multiset identical**, with the replay's hit and miss
> totals independently equal to the engine's own `n_hits`/`n_misses`. The paragraph below is kept as
> the analysis that motivated it, not as open work.

Instrument the demand **array**, not the hook's union. A dump of `site, layer, ...` at every site that
changes residency or LRU order lets the replay consume the engine's own input stream **by
construction**, making the gate an identity rather than a fit. What that work added, beyond the plan
here: `touch` had to be a logged site (it moves LRU while bumping no counter), and prefetch needed a
**pair** of events (`P` reservation + `L` landing) because it writes `slot_owner` at queue time and
`slot_table` only later. Until that existed, the honest position was:

* **D1/D2/D3 stay replay-level**, with the residual attributed (1.26%/1.53%, ratio 1.21x) and the
  MTP-on residual explicitly unattributed.
* **No per-layer claim** can be made from these captures at the strict level (0/40).
* The tool now records all of this in the JSON (`residual` block) so a consumer cannot quote a
  number without the qualification travelling with it.

## 5. Honest status of the gate

| gate level | status |
|---|---|
| global multiset equal | **FAILED** (98.8% / 93.7%) |
| global totals within 2% | passes for MTP-off (+1.5%), fails for MTP-on (+7.7%) |
| residual attributed to a named, measurable mechanism | **MTP-off CONSISTENT 1.21x; MTP-on INVALID** |
| per-layer counts equal | **FAILED 0/40 and 0/41** |
| exact, by construction | **DONE** — `DEMAND_STREAM_EXACT_GATE_2026-09-20.md` (40/40, 41/41) |

The requested outcome was "pass, not tolerate". The finding is that the gate cannot be *passed* by
better modelling, and the reason is now measured rather than assumed — plus a falsifier that would
have caught it if I had guessed wrong.

## Provenance — the exact files every number above comes from

Both captures are the same model (`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`, temp 0.4,
3 prompts) and differ only in MTP and pool:

| | MTP-off | MTP-on |
|---|---|---|
| server log (`CGC-IDS`, `final stats`, `refused=`) | `Backup/cgc_logs/llama_server_20260920_131425.log` | `Backup/cgc_logs/llama_server_20260920_132729.log` |
| pool / slots per layer | 4 GiB / 71 | 8 GiB / 143 (usable 142) |
| engine miss dump | `Backup/phase_decomp/miss_dump_off.txt` | `Backup/phase_decomp/miss_dump_mtpon.txt` |
| replay output JSON | `Backup/phase_decomp/reuse_distance_off.json` | `Backup/phase_decomp/reuse_distance_mtpon.json` |
| capture record (accept, geometry, provenance) | `Backup/cgc_logs/reuse_offset_capture.json` | `Backup/cgc_logs/reuse_mtpon_capture.json` |
| `final stats` | `requests=224897 hits=189133 misses=35764` | `requests=59783 hits=40624 misses=19159` |

Reproduce (no server needed — both are offline replays of a captured log):

```bash
# MTP-off, S = 71
python3 scripts/check/reuse_distance.py \
  --log      Backup/cgc_logs/llama_server_20260920_131425.log \
  --miss-dump Backup/phase_decomp/miss_dump_off.txt --slots 71

# MTP-on, S = 142 after the ZERO-slot reservation; layer 40 is the MTP head (256 slots)
python3 scripts/check/reuse_distance.py \
  --log      Backup/cgc_logs/llama_server_20260920_132729.log \
  --miss-dump Backup/phase_decomp/miss_dump_mtpon.txt \
  --slots 143 --zero-slot --drop-layers 40
```

Source lines cited above are in this worktree: `src/llama.cpp/src/llama-expert-cache.cpp`
(`:221` `touch`, `:470` `cgc_prefill_protect_on`, `:550`/`:595` the protected-slot consultations,
`:906` `ensure_batch`, `:968` the only `n_requests +=`, `:2662` the stats line) and
`src/llama.cpp/src/llama-context.cpp` (`:6468` `CGC-SYNCFILL` print guard, `:6483`
`n_verify_strict_refused`, `:6499` the fast-path guard, `:6507` the `touch()` call).

## Artifacts

- `scripts/check/reuse_distance.py` — `--prefill-protect` (off), the per-layer exactness panel, the
  residual-attribution falsifier, `residual` in the JSON, and (added later) the exact
  `--demand-dump` gate. 57 self-test checks, 0 failing
  (`python3 scripts/check/reuse_distance.py --log /dev/null --selftest`).
- `Backup/phase_decomp/reuse_distance_{off,mtpon}.json` — both include the `residual` block.

**Leftover processes: 0. No timing is quoted anywhere in this document** — both readings are counter
and multiset comparisons over a captured log, so they do not depend on the box's memory state.
