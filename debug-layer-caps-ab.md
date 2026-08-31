# [OPEN] Debug Session: layer-caps-ab

## Metadata
- Session ID: `layer-caps-ab`
- Started At: 2026-08-31
- Scope: Evaluate `LLAMA_EXPERT_CACHE_LAYER_CAPS` redistribution on long dense steady production MTP.

## User Symptom
- User wants to continue exploring production-safe performance levers after `warm`, `MMV_FUSE`, and the current pin profile showed no useful gain on long dense steady.

## Falsifiable Hypotheses
| ID | Hypothesis | Confidence | Status |
|---|---|---:|---|
| H1 | The current production default `40-40:256` remains the best point for long dense steady; disabling it will clearly reduce `hit/accept/t/s`. | 0.74 | Pending |
| H2 | Rebalancing some capacity back from the draft layer to trunk layers can recover throughput without reintroducing `0000`. | 0.46 | Pending |
| H3 | The current `40-40:256` profile has already captured nearly all useful benefit; deviations mostly add churn or overhead. | 0.69 | Pending |
| H4 | If `hit/accept` move but `t/s` does not, the current bottleneck is no longer the slot allocation profile. | 0.63 | Pending |

## Evidence Log
- Prior repo evidence already favors draft-layer residency: [scripts/run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh) documents that `40-40:256` reduced misses `11574 -> 2622` and improved steady throughput by about `+0.8 t/s` in the original ABBA study.
- Candidate matrix run on the current production-safe path (`./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady`):
  - `off` (`N30CACHE_LAYER_CAPS=0`): `26.069 t/s`, `38.36 ms`, `accept 98.383%`, `hit 56.4%`, no `0000`, output SHA256 `595b...49ee`
  - `256` (`40-40:256`): `18.336 t/s`, `54.54 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`, output SHA256 `e661...d614`
  - `224` (`40-40:224`): `18.507 t/s`, `54.03 ms`, `accept 98.118%`, `hit 88.6%`, no `0000`, output SHA256 `46e9...9559`
  - `192` (`40-40:192`): `18.651 t/s`, `53.62 ms`, `accept 98.784%`, `hit 87.5%`, no `0000`, output SHA256 `595b...49ee`
- The first matrix arm (`off`) clearly benefited from early machine state. Control reruns under a warmed machine state:
  - `256_b`: `20.652 t/s`, `48.42 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`
  - `off_b`: `15.604 t/s`, `64.09 ms`, `accept 98.383%`, `hit 56.4%`, no `0000`
- Output identity is stable per arm: `off == off_b` byte-identical; `256 == 256_b` byte-identical. This means the reruns changed performance regime, not the generated text.
- Partial rollback arms do not help:
  - `224` is slower than warmed `256`, lowers accept, and changes output.
  - `192` is also slower than warmed `256`, lowers hit/accept, and its output collapses back to the same byte stream as `off`.

## Conclusion
- H1: Confirmed. The current production default `40-40:256` remains clearly better than `off` once machine-state drift is controlled: much higher `hit`/`accept` and materially better throughput in the warmed comparison (`20.65 vs 15.60 t/s`).
- H2: Rejected. Rolling capacity back from the draft layer to trunk layers (`224`, `192`) does not recover throughput in this environment and degrades at least one of `hit`, `accept`, or output identity.
- H3: Confirmed. `40-40:256` has already captured the useful benefit for this workload; partial rollbacks mostly erode the draft-layer advantage.
- H4: Partially confirmed. `hit/accept` remain strongly coupled to `LAYER_CAPS` choice, but the absolute `t/s` headline is also highly sensitive to machine-state drift, so throughput comparisons need same-regime controls.
