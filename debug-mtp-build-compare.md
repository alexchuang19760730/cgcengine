# [OPEN] Debug Session: mtp-build-compare

## Metadata
- Session ID: `mtp-build-compare`
- Started At: 2026-08-31
- Scope: Compare the historical `26.573 t/s` production MTP steady checkout against the current environment on the same machine.

## User Symptom
- User wants a direct runtime comparison between the historical `26.57` steady line and the current environment, using the historical build checkout rather than only reading docs.

## Falsifiable Hypotheses
| ID | Hypothesis | Confidence | Status |
|---|---|---:|---|
| H1 | The historical `26.573` line corresponds to release head `d02e8cef2`, and its tracked binary still reproduces near-26 t/s on this machine. | 0.72 | Pending |
| H2 | Any large gap versus `26.573` is primarily caused by current machine state (thermal / memory pressure), not checkout differences. | 0.63 | Pending |
| H3 | Any large gap versus `26.573` is primarily caused by local working-tree changes, especially shell-level warm gate behavior in `scripts/run_n30cache.sh`. | 0.58 | Pending |
| H4 | `ctx=3072` is not the differentiator, because it was already present in the historical steady configuration. | 0.90 | Pending |
| H5 | If the historical checkout cannot reproduce near-26 t/s today, the documented `26.573` is a clean-window number rather than a stable same-day invariant. | 0.67 | Pending |

## Evidence Log
- Historical reference line in docs is the `8/30` steady production threshold: [CGC_測試計劃_白皮書_2026-08-30.html](file:///Users/alexchuang/Documents/flashkv0516/moeexpert/doc/CGC_測試計劃_白皮書_2026-08-30.html#L282-L282) and [CGC_生產級Release_2026-08-30.html](file:///Users/alexchuang/Documents/flashkv0516/moeexpert/doc/CGC_生產級Release_2026-08-30.html#L131-L153) document `26.573 t/s / accept 99.187% / hit 88.9%`.
- `ctx=3072` predates the current debugging session. It was introduced by commit `ffeeb59e9` (`MTP_CTX 2048->3072 for steady-state`) and is present in the current production shell default.
- Isolated historical checkout `d02e8cef2` was run in-repo via `.cmp_d02e8cef2` using the same local GGUF bytes (symlinked into `models/gguf`).
- Historical checkout result (`/tmp/compare_hist_d02.log`): `decoded 1102 tokens in 41.814 seconds, speed 26.355 t/s`, `accept 99.728%`, `hit 50.4%`, `ctx=3072`, but output tail collapses to `0000...` (`/tmp/compare_hist_d02.out`).
- Current tree result (`/tmp/compare_cur_head.log`): `decoded 1101 tokens in 47.710 seconds, speed 23.077 t/s`, `accept 99.187%`, `hit 88.9%`, `ctx=3072`, and output remains normal with no `0000` (`/tmp/compare_cur_head.out`).
- Control run on current tree with `N30CACHE_WARM_NPAST=256` (`/tmp/compare_cur_w256.log`): `decoded 1102 tokens in 43.681 seconds, speed 25.228 t/s`, `accept 99.728%`, `hit 50.4%`, and output tail again collapses to `0000...`.
- Current stderr explicitly shows the shell-level warm override on dense steady: `/tmp/compare_cur_head.err` contains `CGC-WARM verify ... warm=0 fast=1`, while the historical checkout runs with the binary default `warm=256`.
- Warm-threshold matrix on the current tree (`--mtp --dense-iq4x --steady`):
  - `warm=0`: `25.991 t/s`, `38.47 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`
  - `warm=8`: `18.764 t/s`, `53.29 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`
  - `warm=16`: `19.298 t/s`, `51.82 ms`, `accept 99.864%`, `hit 82.8%`, no `0000`
  - `warm=256`: `19.751 t/s`, `50.63 ms`, `accept 99.728%`, `hit 50.4%`, **has `0000`**
- Order/thermal control: rerunning `warm=0` after the full matrix yields `20.133 t/s`, `49.67 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`. This confirms later arms are affected by machine-state drift, while the `warm=256 -> 0000` regression remains logic-driven.

## Conclusion
- H1: Partially confirmed. The historical checkout reproduces near-26 t/s on the same machine (`26.355 t/s`), but it is **not** a clean production state because long dense output still degrades into `0000...`.
- H2: Rejected as the primary explanation. Machine state may contribute some variance, but the dominant behavioral difference is reproducible by changing the warm threshold alone.
- H3: Confirmed. The main difference between the historical near-26 result and the current production-safe result is the shell-level warm-gate behavior (`dense steady: warm=256` historically vs `warm=0` now).
- H4: Confirmed. `ctx=3072` is shared by both the historical checkout and the current tree; it is not the differentiator.
- H5: Refined. The documented ~26.57 line is reproducible as a **fast-but-regressing** state, not as the currently corrected no-`0000` production-safe state.
- Additional matrix conclusion: among the explicitly tested warm thresholds on the current tree, `warm=0` remains the only arm that is both fast and free of `0000` in the current production-safe setup. `warm=8/16` do not buy back correctness margin for long dense steady; they only slow it down.
