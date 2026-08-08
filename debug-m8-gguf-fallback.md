# Debug Session: m8-gguf-fallback [OPEN]

## Scope
- Investigate why `m8` fails for local `gguf` model routing.
- Verify whether fallback should remain `edge_cloud_bridge` or be redirected to a stable pass path.

## Symptoms
- `m8` fails after local inference rejection.
- Runtime evidence shows `selected_route = m73_edge_cloud`.
- Runtime evidence shows `selected_backend = edge_cloud_bridge`.
- Local evidence shows `NotADirectoryError` while loading `demo-local.gguf` via MLX.

## Hypotheses
1. `gguf` single-file paths are incorrectly sent into MLX directory-style loading.
2. Local backend selection prioritizes `omlx_mlx_lm` before `llama.cpp` for `gguf`.
3. The first root cause is the local backend decision, and cloud timeout is secondary.
4. Current `edge_cloud_bridge` target is unstable for this path and cannot be treated as a reliable fallback.

## Plan
1. Inspect route-selection and local backend dispatch code.
2. Add instrumentation only around local backend selection and fallback decision.
3. Reproduce `m8` and collect pre-fix evidence.
4. Apply minimal fix after evidence confirms the root cause.
5. Re-run and compare pre-fix vs post-fix evidence.
