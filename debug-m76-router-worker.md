# Debug Session: m76-router-worker [OPEN]

## Scope
- Fix `inst2` worker/runtime failure and bring at least 2 instances to `READY`.
- Produce true parseable `MiniCPM5` router evidence so `selected_route_parse_status = PASS`.

## Constraints
- Keep runtime evidence truthful; no fabricated PASS.
- Preserve reproducible evidence before each fix iteration.

## Initial Hypotheses
1. `inst2` worker runtime is missing a Python package or repo path, so Ray worker imports fail during `SchedulerActor` creation.
2. `inst2` gateway start script and worker environment diverge, so the driver can import modules that the worker cannot.
3. `MiniCPM5` router prompt/output contract is too loose for the current Ollama model, causing masked or non-JSON output.
4. Router evidence parsing is overly strict relative to the actual model output shape, so a usable route is present but not extracted.
5. Remote multi-instance topology is partially stale, so even with one fix we may still need a second instance bootstrap path to reach 2 `READY`.

## Evidence Log
- Confirmed: old router prompt + extraction path failed; historical output was masked `****`.
- Confirmed: new Ollama JSON-schema path returns parseable JSON, latest route is `edge_router`.
- Rejected: current blocker is no longer `SchedulerActor` import failure.
- Rejected: current blocker is no longer `chunked_prefill_size must be divisible by page_size`.
- Confirmed: both `inst1` and `inst2` now fail at backend startup with Ray low-memory worker killing around shard loading 76-77%.

## Current Root Cause
- `inst1`/`inst2` reach gateway alive state, but backend workers are killed by Ray memory protection before readiness.
- Latest probe reports `OutOfMemoryError(... memory usage threshold of 0.950000 ...)` for both active instances.

## Latest Status
- Router evidence: PASS
- 2-ready target: BLOCKED by Ray low-memory killer
- Session: Open
