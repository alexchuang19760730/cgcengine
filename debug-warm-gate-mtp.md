# Debug Session: warm-gate-mtp
- **Status**: [OPEN]
- **Issue**: `CGC_WARM_NPAST=256` fixes short-prompt `0000` degeneration, but breaks long-prompt MTP decode: `CGC_WARM_NPAST=0` gives normal accept and output, while `CGC_WARM_NPAST=256` collapses to `accept 0%` and only 1 decoded token.
- **Debug Server**: http://127.0.0.1:7777/event
- **Log File**: `.dbg/trae-debug-log-warm-gate-mtp.ndjson`

## Reproduction Steps
1. Run short prompt with MTP and confirm warm gate avoids `0000` output.
2. Run long prompt with MTP and compare `CGC_WARM_NPAST=256` vs `CGC_WARM_NPAST=0`.
3. Inspect per-context warm-gate signals during verify and draft decode.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Warm gate reads `n_past` from the wrong context during MTP draft decode, so draft fast path is disabled even after long-prompt prefill. | High | Low | Rejected so far: current evidence is dominated by target/verify batches, not draft batches. |
| B | Warm gate should apply only to verify decode, but it also gates draft decode and starves speculative acceptance. | High | Low | Inconclusive: no direct draft-path evidence yet. |
| C | `llama_memory_seq_pos_max(get_memory(), 0)` is valid for target ctx but not for draft ctx at the point `expert_cache_on_topk()` runs. | High | Medium | Rejected for the current regression: verify-side evidence already explains the bad path. |
| D | The long-prompt regression is caused by an unrelated side effect in exact-load fallback after the gate, not by the gate condition itself. | Medium | Medium | Rejected: `CGC_WARM_NPAST=0` removes the long dense `0000` regression. |
| E | The regression depends on layer-specific cache state, so the gate condition is correct but the fallback path leaves the remap/pool state inconsistent for MTP draft. | Low | Medium | Inconclusive. |

## Log Evidence
- `.dbg/trae-debug-log-warm-gate-mtp.ndjson` shows early long-prompt dense runs repeatedly entering verify-side gate checks with `ctx="verify"` and `n_tokens=8`, e.g. `n_past=15/23/.../175`, which means the warm gate is acting on target-side multi-token batches during the long path.
- With the original gate, `CGC_WARM_NPAST=0` removes long dense tail degeneration (`/tmp/mtp_long_dense_warm0.out` has no `0000`), while `CGC_WARM_NPAST=256` reproduces it (`/tmp/mtp_long_prod.out` tail collapses to `0000...`).
- Short prompts need a non-zero warm gate: `CGC_WARM_NPAST=0` reproduces `The capital of France is a 100000...` in `/tmp/mtp_short_warm0.out`, while `CGC_WARM_NPAST=8` and `=16` both keep normal `Paris` output.
- Production-shell workaround is validated: after updating [scripts/run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh), default short runs use `CGC_WARM_NPAST=8`, long dense runs use `CGC_WARM_NPAST=0`, and all four production test paths avoid `0000`.
- Pre-commit script now contains runtime production acceptance in [scripts/check_build_tracked.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/check_build_tracked.sh): `RUN_CGC_PROD_ACCEPT=1` runs `short_base`, `long_base`, and `long_dense`, checks for `0000`, and reports `t/s / TPOT / accept / hit rate`.
- Latest acceptance run (`BIN_DIR=src/llama.cpp/build/bin RUN_CGC_PROD_ACCEPT=1 scripts/check_build_tracked.sh`) passed all three production cases: short base `13.354 t/s / 74.88 ms`, long base `24.729 t/s / 40.44 ms`, long dense `21.373 t/s / 46.79 ms`, all with zero `0000`.

## Verification Conclusion
- Confirmed: the regression is warm-gate related, not a separate denseIQ4X bug.
- Confirmed: one fixed threshold does not fit every production path.
- Current workaround: use shell-level defaults in [scripts/run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh) so short prompts keep a small warm gate and long dense prompts avoid the bad `256` gate.
- Runtime acceptance is now moved into precommit coverage (same script, gated by `RUN_CGC_PROD_ACCEPT=1` for heavy runs).
- Remaining work: cold-machine performance confirmation, then decide whether to turn `CGC_ACCEPT_*_MIN_TPS` into hard speed gates.
