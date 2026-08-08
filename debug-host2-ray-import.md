# Debug Session: host2-ray-import
- **Status**: [OPEN]
- **Issue**: host2 inst1/inst3 backend fails to start after scheduler crash instrumentation sync, with `ImportError: cannot import name '_get_effective_model_parallel_size' from 'sglang.srt.entrypoints.engine'`.
- **Debug Server**: N/A
- **Log File**: N/A

## Reproduction Steps
1. Sync patched runtime files to host2.
2. Relaunch host2 `inst1` and `inst3`.
3. Observe backend deployment failure in `ray_serve_sglang_backend.log`.
4. Inspect import traceback for `sglang.srt.ray.engine -> sglang.srt.entrypoints.engine`.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | host2 live `sglang.srt.ray.engine` expects `_get_effective_model_parallel_size`, but live `sglang.srt.entrypoints.engine` is older and does not export it. | High | Low | Pending |
| B | host2 startup mixes repo-tree and `site-packages`, so imports resolve to different code versions. | High | Med | Pending |
| C | the latest sync updated `ray/engine.py` but did not sync its dependent `entrypoints/engine.py`, creating a partial upgrade. | High | Low | Pending |
| D | after startup is restored, the original timeout will move downstream into scheduler/decode rather than import/bootstrap. | Med | Med | Pending |

## Log Evidence
- Local repo evidence:
  - `sglang/srt/ray/engine.py` imports `_get_effective_model_parallel_size` from `sglang.srt.entrypoints.engine`.
  - `sglang/srt/entrypoints/engine.py` defines `_get_effective_model_parallel_size(...)` at line ~1434.
- Access evidence:
  - SSH to host2 succeeds with password `Gen@song123`.
- Runtime sync evidence:
  - `sync_runtime_files --host-label host2` successfully synced `entrypoints/engine.py`, `ray/engine.py`, `ray/scheduler_actor.py`, and later `managers/scheduler.py` into both repo-tree and live `site-packages`.
- Startup recovery evidence:
  - `inst1` and `inst3` both recovered to `GET /health = 200` and `GET /v1/models = 200`.
- Request-path evidence:
  - For both inst1 and inst3, `PrefillAdderTrace` showed `pending_swa_budget=384` and `rem_swa_tokens=128` after prefill budget update.
  - The first replay still timed out because scheduler crashed inside debug trace code:
    - `_scheduler_budget_trace_fields(...) -> tree_cache.evictable_size() -> NotImplementedError`
  - After guarding the trace-only metric collection, both gateways returned real `/v1/completions` responses in about 64s.
  - Response shape for both instances:
    - `completion_tokens=8`
    - `finish_reason="length"`

## Verification Conclusion
- Hypothesis A: Confirmed. Host2 had a partial runtime version skew; grouped sync of `entrypoints/engine.py` and Ray files removed the startup import blocker.
- Hypothesis B: Confirmed in practice. Host2 required repo-tree and live `site-packages` to be updated together.
- Hypothesis C: Confirmed. The missing grouped sync was the direct cause of the import mismatch.
- Hypothesis D: Partially confirmed. After backend recovery, the next blocker was not SWA headroom but our own scheduler budget trace calling an unimplemented cache metric. Once guarded, single-question real generation succeeded on both `inst1` and `inst3`.
- Current status:
  - Minimal objective achieved: host2 `inst1/inst3` single-question requests now complete.
  - No additional SWA headroom is required for this minimal proof path.
  - Residual risks remain in latency (~64s for 8 tokens), output quality, and host2 disk pressure.
