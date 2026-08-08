# [OPEN] Debug Session: swa-post-admission-drain

## Goal
- Explain why host2 resolves `swa=512` at startup, but `rem_swa_tokens` becomes `0` immediately after the first `can_run_list` admission.

## Scope
- Runtime evidence only.
- No business-logic modification before evidence is collected.

## Current Facts
- Host2 startup now shows:
  - `DSV4 pool sizes: full=1024, swa=512`
  - `Memory pool resolve post-constraint: constrained=1024 ... swa=512`
- Replay still times out.
- Current first observed failure moved to:
  - `[PrefillAdderTrace] stage='no_token' reason='budget_state'`
  - `rem_total_tokens=508`
  - `cur_rem_tokens=512`
  - `rem_swa_tokens=0`
  - `can_run_list_size=1`

## Falsifiable Hypotheses
1. `rem_swa_token_offset` jumps to `512` in the first `add_one_req()` path and consumes the entire SWA budget.
2. `token_to_kv_pool_allocator.swa_available_size()` drops from `512` to `0` immediately after first reservation, independent of offset.
3. `tree_cache.swa_evictable_size()` remains `0`, so there is no reclaimable headroom after the first admission.
4. Another path after `can_run_list.append(req)` mutates SWA accounting before `budget_state()` runs.
5. The current `PrefillAdderTrace` is too late in the flow, so we need before/after snapshots around the first `add_one_req()` to isolate the exact decrement point.

## Plan
1. Inspect `schedule_policy.py` and related SWA allocator methods.
2. Add narrow instrumentation around first-admission SWA accounting.
3. Sync instrumentation to host2 live runtime paths.
4. Replay one short request on `inst1` and `inst3`.
5. Compare pre/post admission SWA fields and pin the first drain point.

## Instrumentation Added
- Added narrow `PrefillAdderTrace` hooks in `schedule_policy.py` for:
  - `add_one_req_pre_lock`
  - `add_one_req_post_lock`
  - `update_prefill_budget_before`
  - `update_prefill_budget_after`
  - `budget_state_no_token`
- Added traced fields:
  - `swa_available_size`
  - `swa_evictable_size`
  - `rem_swa_token_offset`
  - `rem_swa_tokens`
  - `swa_needed`
  - `paged_extend_input_len`
  - `pending_swa_budget`

## Runtime Evidence
- Replayed one short request on host2 after syncing the patched `schedule_policy.py` into live `site-packages` and scoped-restarting `inst1` / `inst3`.
- `inst1` evidence (`rid='3c2f513553e8472aa2bfd29953b5280d'`):
  - `add_one_req_pre_lock`
    - `swa_needed=384`
    - `swa_available_size=512`
    - `swa_evictable_size=0`
    - `rem_swa_token_offset=0`
    - `rem_swa_tokens=512`
  - `update_prefill_budget_before`
    - `paged_extend_input_len=256`
    - `pending_swa_budget=512`
    - `rem_swa_tokens=512`
  - `update_prefill_budget_after`
    - `rem_swa_token_offset=512`
    - `rem_swa_tokens=0`
    - `rem_total_tokens=508`
    - `cur_rem_tokens=512`
  - `budget_state_no_token`
    - `rem_swa_tokens=0`
- `inst3` evidence (`rid='d297615dedfa44bda34bfc2ae62379db'`) is identical.

## Conclusion
- The first post-admission SWA drain point is now pinned exactly:
  - it is **not** `swa_available_size()` collapsing on its own
  - it is **not** `tree_cache.swa_evictable_size()` changing
  - it is the scheduler itself increasing `rem_swa_token_offset` inside `_update_prefill_budget()`
- The reason the whole budget is consumed is:
  1. pre-lock admission checks use `swa_needed=384`
  2. but `_update_prefill_budget()` calls `_swa_budget_for_req(extend_input_len)` **after** page-aligning `extend_input_len` to `256`
  3. so the actual deducted SWA budget becomes:
     - `max(256, sliding_window_size=128) + page_size=256`
     - `= 512`
  4. initial runtime SWA headroom is exactly `512`
  5. therefore:
     - `rem_swa_token_offset: 0 -> 512`
     - `rem_swa_tokens: 512 -> 0`
     - the very first `can_run_list` admission exhausts all SWA budget

## Hypothesis Status
1. `rem_swa_token_offset` jumps on the first admission and consumes the budget:
   - **Confirmed**
2. `swa_available_size()` collapses independently:
   - **Rejected** (`512` stays `512` across the traced before/after point)
3. `swa_evictable_size()` changes and causes the drop:
   - **Rejected** (`0` stays `0`)
4. A later path after `can_run_list.append(req)` consumes SWA unexpectedly:
   - **Refined**: the consuming path is specifically `_update_prefill_budget()`
5. The previous trace point was too late:
   - **Confirmed**

## Updated Next Step
- The next smallest debugging question is no longer "why does `rem_swa_tokens` become 0?"
- That is now answered.
- The next smallest decision is:
  - whether the code should reserve SWA using the **pre-lock admission budget (`384`)**
  - or whether host2 must be provisioned with **more than 512 SWA tokens** so that the current conservative reservation still admits one request plus decode slack
