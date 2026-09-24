# M4 — the draft's distribution was never the request's (implemented, not yet measured)

**Date** 2026-09-20 · carrier Nail IQ3_XXS-denseIQ4X, delivery profile (prefill250 / prod25-stream).
Files touched: `common/speculative.{h,cpp}`, `tools/server/server-context.cpp`, `scripts/run_server.sh`
(the first three were **clean** at HEAD `656cfdf1b` before this change; `run_server.sh` already
carried other sessions' edits — my hunks are only the two blocks quoted below). Built: `BUILD_RC=0`,
witness `libllama-common.0.0.279.dylib` (13 `MTPDBG` strings ⇒ `-DMTP_SUPPORT` is in this artifact).

## 1. What was wrong (a knob that could not be turned on)

`CGC_MTP_SAMPLER_PARITY` exists in the engine to make the draft chain *be* the target's chain
(`speculative.cpp`, MTP ctor: parity ⇒ `sparams = params.sampling` instead of the legacy
`{TOP_K=10}`, and the draw is the chain's sample instead of its argmax). Two independent reasons it
could not do anything in the delivery configuration:

1. **The launcher never set it.** It appeared only as a pass-through
   (`if [ -n "${CGC_MTP_SAMPLER_PARITY:-}" ]`), and no profile sets it — so every MTP arm ever run
   here had it absent. (Pass-through was also presence-based: a literal `=0` would have meant ON.)
2. **Even with it set, it mirrored the wrong chain.** The draft chain is built **once, at init**,
   from `params_base.sampling` (`server-context.cpp:1285`), and the launch line pins that to
   `--temp 0`. A request then samples at **its own** temperature (the delivery runs temp 0.4), and
   nothing ever handed that request's chain to the draft. So parity would have bought a *greedy*
   draft mirrored against a *greedy* default — not against what the target actually samples.

## 2. What is implemented

| change | where |
|---|---|
| `common_speculative_set_sampling(spec, sampling)` — public API, iterates the impls | `speculative.h`, `speculative.cpp` |
| base-class virtual `set_sampling(...)`, default no-op (only a draft that is *defined* to mirror the target has anything to do) | `speculative.cpp` |
| MTP override: rebuilds the per-seq draft chains from the request's config; returns early when parity is off (legacy chain is not derived from the target) or when nothing usable was handed down | `speculative.cpp` |
| ctor's inline chain build factored into `build_samplers(sampling)` — one builder, two callers | `speculative.cpp` |
| call site: right after the request's own sampler is created (`slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling))`) | `server-context.cpp:1906` |
| **Default unchanged** (`CGC_MTP_SAMPLER_PARITY` still off ⇒ byte-identical behaviour), value-aware pass-through, `echo` banner | `run_server.sh` |
| `--spec-draft-p-min` plumbed (engine early-stops the draft chain on low confidence; the flag existed and the launcher never passed it ⇒ every arm so far ran `p_min=0`) | `run_server.sh` (opt-in via `CGC_SERVER_MTP_P_MIN`) |

## 3. The A/B (next window; no numbers claimed here)

```bash
# parity + a confidence floor, delivery profile, same binary, interleaved, >= 3 reps
CGC_MTP_SAMPLER_PARITY=1 CGC_SERVER_MTP_P_MIN=0.1 ./scripts/run_server.sh
# control = the same command without the two variables
```

Acceptance must be **joint**, because accept moves tokens while the step grows with them (G6
measured **+11% mean_len but +14% step ⇒ net −0.45 t/s**):

- `mean_len` **and** the step from the **same round** (`scripts/check/mtp_ruler.py` exists precisely
  because they have been reported separately twice), plus accept at its post-canon-order semantics.
- The hard ceiling is structural: delivery is `k=3` ⇒ `mean_len ≤ 4.0`. Against the citable
  baseline (12.57 t/s, `prod25-stream`, step 268 ms reverse-derived) that is **14.9 t/s at an
  unchanged step** — not the +40% that mixing the 58.25% accept arm with the 3.375 mean_len arm
  suggests (see `L3_M1_GAP_LEDGER_2026-09-20.md` §0).
- M1/M2/M3 oracle gate on the control arm only: the draft distribution changes *generated text*,
  not the engine's kernels, and the gate's probe is greedy, where both chains coincide.

## 4. What this does not fix

`step` itself (247.98 ms of work-row, of which 149.24 ms is GPU-occupied and 74.18 is the union's
bytes). M4 buys tokens per step; 25 t/s still needs the step and/or the GPU-occupied wall to move —
see the ledger's §5.
