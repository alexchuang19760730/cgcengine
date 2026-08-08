# Debug Session: dflash-scheduler-path [OPEN]

## Problem

- Symptom: `M4 inference` / `oMLX + dflash` path still leaves evidence that `GenerationBatch` is imported indirectly during `fullgraph_capture`.
- Goal: trace `DFlashEngine.start()` internal path, identify where `scheduler` is still pulled in, then rerun `M4 inference` and `M5`.

## Hypotheses

1. `DFlashEngine.start()` directly imports or instantiates something from `omlx.scheduler`.
2. `DFlashEngine.start()` triggers a fallback-engine initialization path that imports `scheduler`.
3. A package-level `__init__` or indirect helper import still eagerly loads `scheduler`.
4. `M4 inference` and `M5` do not fail at the same runtime edge; `M5` may already be past this bug.
5. The actual trigger occurs inside runtime-context/model-loading helpers, not at `DFlashEngine` construction time.

## Plan

1. Add instrumentation only around `DFlashEngine.start()` and closely related helper calls.
2. Reproduce `M4 inference` and inspect runtime evidence.
3. Confirm or reject hypotheses.
4. Apply minimal fix only after evidence is collected.
5. Rerun `M4 inference` and `M5`.

## Evidence So Far

- Confirmed: `DFlashEngine.start()` no longer fails first on `GenerationBatch`; moving `scheduler` import out of `engine_core.py` top-level removed that blocker.
- Confirmed: `DFlashEngine.start()` also no longer fails on missing `mlx.core.new_thread_local_stream`; `engine_core.py` now tolerates older MLX APIs by falling back to `new_stream`.
- Confirmed: the next blocker is architectural, not another scheduler edge. `omlx/engine/dflash.py` still imports `dflash_mlx.*` at runtime, and repo docs/tests explicitly treat `dflash-mlx` as a required dependency for real DFlash execution.
- Constraint from user: do not rely on external `dflash-mlx` installation.

## Current Conclusion

- With the current repo state, `oMLX` contains the `DFlashEngine` wrapper, but the actual DFlash runtime is not vendored into this repository.
- Therefore, `M4/M5` cannot truthfully pass on the `dflash` path without either vendoring/replacing the `dflash_mlx` runtime inside the repo, or explicitly degrading to a non-DFlash engine and reporting that downgrade.

## Fixes Applied

- Moved `scheduler` imports in `omlx/engine_core.py` out of module top-level so `DFlashEngine.start()` no longer pulls `GenerationBatch` in via eager import.
- Added MLX API compatibility fallback in `omlx/engine_core.py`: use `mx.new_thread_local_stream()` when available, otherwise fall back to `mx.new_stream()`.
- Added a repo-local vendored `dflash_mlx` compatibility package under `ComputeGraphCompiler-main/dflash_mlx/`.
- The vendored runtime currently implements the minimum API surface needed by `DFlashEngine`:
  - runtime config/context
  - target bundle loading
  - draft placeholder bundle
  - token/summary event types
  - prefix-cache compatibility shim
  - target-only streaming generation bridge via `mlx_lm.generate.stream_generate()`

## Verification

- `M4 inference` rerun at `/tmp/cgc_dflash_debug_m4_vendor1/` now passes the inference gate:
  - `compile_mode = "omlx_dflash"`
  - `engine = "dflash"`
  - `omlx_flashmoe_ondemand_gate.status = "PASS"`
- `M4` overall still fails only on the formal training-side `distributed_gate` (`world_size=1`, `enable_nccl=false`), which is outside this DFlash bug.
- `M5` rerun with fingerprint lock is now in real build/bench execution under `/tmp/cgc_dflash_debug_m5_vendor2/`; it is no longer failing immediately on `GenerationBatch`, `new_thread_local_stream`, or external `dflash-mlx` import.
