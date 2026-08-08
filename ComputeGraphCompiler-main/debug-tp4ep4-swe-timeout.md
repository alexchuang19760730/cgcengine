# Debug Session: tp4ep4-swe-timeout
- **Status**: [OPEN]
- **Issue**: TP4/EP4 four-instance health checks pass, but real single-question inference times out on both gateway `/v1/completions` and backend `/generate`.
- **Debug Server**: http://127.0.0.1:7777/event
- **Log File**: .dbg/trae-debug-log-tp4ep4-swe-timeout.ndjson

## Reproduction Steps
1. Verify four TP4/EP4 instance health endpoints on `host1` and `host2`.
2. Send a minimal completion request to gateway ports `50053/50063/50073/50083`.
3. Send a minimal generate request to backend ports `30000/30010/30020/30030`.
4. Observe timeout behavior and collect pre-fix runtime evidence.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Gateway can answer `/health`, but request forwarding to backend hangs before `/generate` is issued. | High | Low | Pending |
| B | Backend receives `/generate`, but stalls before first token because tokenizer / scheduler / warmup never completes. | High | Med | Partially rejected on host1 single-chat evidence |
| C | Ray Serve replica remains alive for health checks, but inference workers are detached or deadlocked. | Med | Med | Pending |
| D | TP4/EP4 launch parameters or model load state are inconsistent across instances, causing fake-healthy but non-serving replicas. | Med | Med | Pending |

## Log Evidence
- `temp/host1_request_trace.json`: host1 backend health-check request traverses `TokenizerTrace -> send_to_scheduler_done -> SchedulerTrace request_enqueued_waiting_queue`, so backend IPC + scheduler path is alive for health probes.
- `temp/host1_chat_only_rid_trace.json`: chat rid `e277ba614bbc42ed9f56f5ad1a6a0349` reaches `TokenizerTrace dispatch_single_to_scheduler`, then `SchedulerTrace run_batch_*`, `send_output_to_detokenizer`, `response_enqueued`, and finally `OpenAITrace handler_non_streaming_response_created`. This proves host1 backend can finish at least one real non-streaming chat request.
- `temp/host1_single_chat_completion.json`: the same host1 request through gateway `http://127.0.0.1:50053/v1/chat/completions` still times out after 90s with 0 bytes received.
- Working conclusion: the dominant blocker on host1 is no longer "scheduler never runs"; it is "gateway does not return backend completion to the client in time".
- Fresh host1 `inst4` replay with remote debug collector:
  - `http_server.py:generate_request` emits `backend generate handler entered` and `backend generate awaiting tokenizer_manager response`.
  - `tokenizer_manager.py:_send_one_request` emits `tokenizer send_to_scheduler begin` then `tokenizer send_to_scheduler returned` with `Future(done=True)`.
  - `request_receiver.py:_pull_raw_reqs` emits `scheduler received tokenizer request` for the same rid.
  - But there is still **no** `tokenizer received detokenizer payload` / `response_enqueued` before the client times out and the handler later raises `Request is disconnected from the client side`.
- Updated working conclusion: for fresh host1 `inst4`, the real request reaches scheduler input, but does not progress to a detokenizer-visible response path.

## Verification Conclusion
- Added another instrumentation slice in `Backend/CGC/ray_serve_sglang_gateway.py`:
  - forward `X-Request-ID`, `X-CGC-Trace-ID`, `x-cgc-task-type` to backend;
  - split non-streaming upstream observation into `start -> headers received -> body read start -> body read done`.
- This should let the next replay distinguish:
  1. stuck before upstream connects,
  2. stuck after headers but before full body is consumed,
  3. stuck after body is read but before gateway returns response.
- Added focused runtime instrumentation in:
  - `tokenizer_manager.py:_send_one_request`
  - `tokenizer_manager.py:handle_loop`
  - `scheduler_components/request_receiver.py:_pull_raw_reqs`
- Evidence now narrows the live fault domain to the scheduler execution / scheduler-to-detokenizer leg after `recv_from_tokenizer_raw`.

## Latest Update
- Re-checked the newer `completion-max-new-tokens` evidence: the earlier `max_new_tokens -> 0` hypothesis is no longer the dominant path on the currently instrumented host1 runtime.
- On the preserved host1 trace, scheduler sees and keeps the request budget intact:
  - `handle_generate_request_sampling_params ... recv_req_max_new_tokens=8`
  - `init_req_max_new_tokens_after ... after_max_new_tokens=8`
- `prefill_add_one_req_result ... result='NO_TOKEN'` is not a terminal failure by itself in this path, because the same trace still proceeds through:
  - `prefill_selected_can_run_list`
  - `run_batch_begin/run_batch_end`
  - `process_batch_result`
  - `send_output_to_detokenizer`
- Therefore the next highest-value breakpoint is no longer tokenizer->scheduler admission, but the output return path:
  - `scheduler_components/output_streamer.py`
  - `detokenizer_manager.py:event_loop`
  - `detokenizer_manager.py:handle_batch_token_id_out`
  - `tokenizer_manager.py:handle_loop`
- Added new instrumentation on that chain to capture:
  1. scheduler payload type / output token lengths / finished reasons before detokenizer send
  2. detokenizer recv, decode, and send-back events
  3. tokenizer recv payload details on the return leg

## 2026-07-10 Replay: inst2 completion handoff-inst2-route-002
- Synced latest instrumentation to the actual import targets on `host1`:
  - `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/tokenizer_manager.py`
  - `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/detokenizer_manager.py`
  - `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler_components/output_streamer.py`
- Remote `py_compile` passed for all three files.
- `host1 /data` was blocked by stale Ray sessions during relaunch:
  - before cleanup: `/data` `984G/984G`, `/data/ray` `531G`
  - dominant offender: `/data/ray/inst2` `375G`
  - performed instance-scoped cleanup of stale `inst2` Ray data only
  - after cleanup: `/data` `610G/984G`, `/data/ray` `157G`
- Re-launched `inst2` successfully with `validate_single_gateway_host1 --instance-id inst2`.
- Replayed the same `/v1/completions` payload with a fresh trace id to avoid mixing with older logs:
  - `X-Request-ID = handoff-inst2-route-002`
  - `X-CGC-Trace-ID = handoff-inst2-route-002`
  - `prompt = "Reply with only OK."`
  - `max_tokens = 4`
  - result: client-side `TimeoutError` at ~`60072ms`

### Replay Evidence
- Gateway / OpenAI adapter definitely received the request:
  - `OpenAITrace stage='handler_received' trace_id='handoff-inst2-route-002'`
  - `OpenAITrace stage='handler_adapted' trace_id='handoff-inst2-route-002'`
- Tokenizer normalized and dispatched the request correctly:
  - `TokenizerTrace stage='before_normalize' sampling_max_new_tokens=4`
  - `TokenizerTrace stage='after_normalize' rid='f12026723d044824b435be28c3d76655' sampling_max_new_tokens=4`
  - `TokenizerTrace stage='send_to_scheduler_done' rid='f12026723d044824b435be28c3d76655'`
- Tokenizer then stalled waiting for output:
  - repeated `TokenizerTrace stage='wait_timeout_poll' ... out_list_len=0 finished=False`
- Scheduler again collapsed the request into the `NO_TOKEN` loop:
  - `SchedulerTrace stage='prefill_add_one_req_result' rid='f12026723d044824b435be28c3d76655' result='NO_TOKEN' max_new_tokens=0 can_run_count=0`
  - `SchedulerTrace stage='running_batch_marked_full_add_req_no_token' ... running_batch_size=0`
  - `SchedulerTrace stage='prefill_return_none_no_runnable_requests' waiting_rids=['f12026723d044824b435be28c3d76655']`
- New detokenizer / return-path instrumentation did **not** fire for this replay:
  - no `DetokenizerTrace`
  - no `send_output_to_detokenizer`
  - no `recv_from_detokenizer`

### Updated Conclusion
- For the fresh `inst2` replay, the fault domain re-converges **before** the scheduler-to-detokenizer leg.
- The request reaches scheduler admission, but scheduler immediately projects it to:
  - `max_new_tokens=0`
  - `can_run_count=0`
  - `result='NO_TOKEN'`
- Therefore the next highest-value code focus is again:
  - `scheduler.py` around `prefill_add_one_req_result` / `running_batch_marked_full_add_req_no_token`
  - `init_req_max_new_tokens(...)`
  - request budget / pool gate conditions that turn a tokenizer-side `sampling_max_new_tokens=4` request into scheduler-side `max_new_tokens=0`

## Root Cause Identified
- The `max_new_tokens -> 0` transition is now fully explained by `scheduler.py:init_req_max_new_tokens()` itself, not by a later scheduler admission mutation.
- Fresh inst2 trace for `rid='f12026723d044824b435be28c3d76655'` shows:
  - `recv_req_max_new_tokens=4`
  - `req_max_new_tokens=4`
  - `input_len=5`
  - `paged_input_len=256`
  - `page_size=256`
  - `max_total_num_tokens=256`
  - `max_total_tokens_bound=-257`
  - `after_max_new_tokens=0`
- This exactly matches the scheduler clamp formula:
  - `max_total_num_tokens - paged_input_len - page_size - 1`
  - `= 256 - 256 - 256 - 1 = -257`
- Once `init_req_max_new_tokens()` clamps to `0`, the later repeated:
  - `prefill_add_one_req_result ... result='NO_TOKEN'`
  - `running_batch_marked_full_add_req_no_token`
  are downstream symptoms, not the origin.

## Launch Config Mismatch
- The current launcher in `temp/runtime_ops/remote_runtime_ops.py` exported:
  - `CGC_SGLANG_CONTEXT_LENGTH=256`
  - `CGC_SGLANG_MAX_TOTAL_TOKENS=256`
  - `CGC_SGLANG_CHUNKED_PREFILL_SIZE=256`
- Under the observed backend `page_size=256`, this makes any non-empty prompt unschedulable for generation, because even a 5-token prompt already consumes:
  - `ceil_page(input_len)=256`
  - plus another `page_size=256` reserve
  - leaving negative room for generation tokens.

## Local Fix Prepared
- Updated `temp/runtime_ops/remote_runtime_ops.py` to raise:
  - `CGC_SGLANG_MAX_TOTAL_TOKENS` from `256` to `1024`
- This is the minimal launcher-side correction that makes:
  - `ceil_page(input_len) + max_new_tokens + page_size < max_total_num_tokens`
  satisfiable again for the current one-request replay path.

## Verification After Launcher Fix
- Re-brought up `host1 inst2` using the patched local launcher.
- Startup evidence now shows the corrected capacity really took effect:
  - `ModelRunnerTrace event=init_memory_pool_done ... max_total_num_tokens=1024`
  - `SchedulerActorTrace event=get_info_done ... max_total_num_tokens=1024 max_req_input_len=250`
- Replayed the same completion payload with a fresh trace id:
  - `X-Request-ID = handoff-inst2-route-003`
  - `X-CGC-Trace-ID = handoff-inst2-route-003`
  - `prompt = "Reply with only OK."`
  - `max_tokens = 4`
- Result:
  - HTTP `200`
  - response body returned normally
  - `completion_tokens = 4`
  - `finish_reason = "length"`

### Scheduler Proof
- For `rid='4c497701536f4a7db929f1cf1de10679'`, all four scheduler workers now show:
  - `input_len=5`
  - `paged_input_len=256`
  - `page_size=256`
  - `max_total_num_tokens=1024`
  - `max_total_tokens_bound=511`
  - `after_max_new_tokens=4`
- This confirms the previous `0` clamp is removed.

### Important Follow-up Observation
- `prefill_add_one_req_result` still logs:
  - `result='NO_TOKEN'`
  - `max_new_tokens=4`
  - `can_run_count=1`
- But this no longer blocks the request. The request proceeds and eventually reaches:
  - `TokenizerTrace stage='recv_from_detokenizer' ... rids=['4c497701536f4a7db929f1cf1de10679']`
  - `TokenizerTrace stage='generate_finished' ... finish_reason={'type': 'length', 'length': 4}`
- Therefore:
  - the launcher capacity bug was the blocker that forced `max_new_tokens=0` and made the request unschedulable;
  - the remaining `NO_TOKEN + can_run_count=1` trace pattern is a scheduler bookkeeping quirk / non-fatal path, not the current stop-ship fault for single-request inference.

## Four-Instance Replay After Launcher Fix
- Applied the same launcher-side `CGC_SGLANG_MAX_TOTAL_TOKENS=1024` correction to the other three instances via `validate_gateway_instance()`:
  - `inst4` on `host1`
  - `inst1` on `host2`
  - `inst3` on `host2`
- Runtime bring-up results:
  - `inst4`: ready on `50083`
  - `inst1`: ready on `50053`
  - `inst3`: ready on `50073`
- Host2 required extra cleanup during this phase:
  - stale `inst1` Ray residue had to be removed before `inst1` could start
  - `host2 /data2` still remains at ~`99%` due old deleted-open `inst3` session files held by prior Ray processes

### Same-Payload Replay Matrix
- Payload for all four:
  - `prompt = "Reply with only OK."`
  - `max_tokens = 4`
  - `temperature = 0`
  - `stream = false`

### Results
- `inst2` (`host1:50063`):
  - trace: `fourinst-inst2-route-004`
  - HTTP `200`
  - completed in ~`6.3s`
  - `completion_tokens = 4`
- `inst4` (`host1:50083`):
  - trace: `fourinst-inst4-route-004`
  - HTTP `200`
  - completed in ~`10.8s`
  - `completion_tokens = 4`
- `inst1` (`host2:50053`):
  - trace: `fourinst-inst1-route-004`
  - client timeout at ~`90.1s`
- `inst3` (`host2:50073`):
  - trace: `fourinst-inst3-route-004`
  - client timeout at ~`90.1s`

## Host2-Specific New Fault Domain
- Host2 no longer shows the earlier `max_new_tokens -> 0` capacity-collapse signature as the dominant issue.
- For both timed-out requests, scheduler now repeatedly logs:
  - `prefill_skip_empty_can_run_list ... batch_is_full=True running_batch_size=0`
  - `prefill_break_add_req ... add_req_result='AddReqResult.NO_TOKEN' can_run_list_size=0`
- This means host2 has moved to a different stuck path:
  - waiting queue contains the request
  - `running_batch_size` is still `0`
  - but scheduler already considers the batch `full`
- Current best interpretation:
  - the launcher token-capacity fix is effective on the host1 side and likely no longer the primary blocker on host2;
  - the remaining four-instance gap is now **host2-specific scheduler batch-state corruption / stale runtime state**, not the original `max_total_tokens=256` misconfiguration.

## Host2 Batch-Full Path Clarified
- Code inspection now pins down **where** `running_batch.batch_is_full` is being set on the host2 timeout path:
  - `scheduler.py:2913-2920`
  - when `res == AddReqResult.NO_TOKEN`
  - and `enable_hierarchical_cache == False`
  - scheduler unconditionally executes:
    - `self.running_batch.batch_is_full = True`
- This directly explains the observed runtime combination:
  - `running_batch_size=0`
  - `batch_is_full=True`
- `batch_is_full` is **not** coming from an already-populated running batch; it is being set during the current prefill admission pass after the first request already failed admission.

## Why It Sticks With running_batch_size=0
- `batch_is_full` is only force-reset on a few paths:
  - `scheduler.py:2600` (HiSparse path)
  - `scheduler.py:2622` / `2640` (batch filtering removes/finishes running reqs)
  - `scheduler.py:2727` (priority-preemption or hybrid-SWA path)
  - `scheduler.py:3105` (empty batch after `update_running_batch`)
- On the current host2 path:
  - `running_batch` is empty from the start
  - the first request returns `AddReqResult.NO_TOKEN`
  - `batch_is_full` becomes `True`
  - next scheduler iterations hit the early return gate at `scheduler.py:2730-2743`
  - so the scheduler keeps skipping fresh admission attempts and loops on:
    - `prefill_skip_empty_can_run_list`
    - `prefill_break_add_req`
- Therefore the sticky state is real, but it is a **secondary latch** after the first `NO_TOKEN`, not the origin of the failure.

## What NO_TOKEN Depends On Here
- `schedule_policy.py` shows that `AddReqResult.NO_TOKEN` in `add_one_req()` is driven by budget state, not by `running_batch` alone.
- The most relevant sources are:
  - `PrefillAdder.rem_total_tokens`
  - `PrefillAdder.cur_rem_tokens`
  - `PrefillAdder.rem_input_tokens`
- These are derived from:
  - `token_to_kv_pool_allocator.available_size()`
  - `tree_cache.evictable_size()`
  - plus offsets from currently tracked requests
- So the current host2 diagnosis is:
  - first failure = admission budget says `NO_TOKEN`
  - second failure = scheduler latches `batch_is_full=True` and stops making progress

## Residue Check Result
- Host2 does have real stale residue, but it is not yet sufficient to prove it is the direct cause of the first `NO_TOKEN`:
  - `inst3` still has multiple old Ray session directories under `/data2/ray/inst3/`
  - host2 also has three unrelated GPU processes from **2026-07-05**:
    - `python3 test_qwen_rswa.py`
    - each still holds ~`654 MiB`
- However, current bring-up sessions are also clearly healthy enough to initialize scheduler capacity:
  - both `inst1` and `inst3` log:
    - `ModelRunnerTrace event=init_memory_pool_done ... max_total_num_tokens=1024`
    - `SchedulerActorTrace event=get_info_done ... max_total_num_tokens=1024 max_req_input_len=250`
- GPU free memory on host2 also remains very high (~`68-70 GiB` per card), so the stale `test_qwen_rswa.py` processes are evidence of a dirty host, but **not yet a proven explanation** for this scheduler `NO_TOKEN` path.

## Practical Current Read
- We now have three separate facts:
  1. host2 first hits `AddReqResult.NO_TOKEN` during `add_one_req()`
  2. scheduler then self-latches `batch_is_full=True` even with `running_batch_size=0`
  3. host2 also has unrelated stale processes / old session directories, plus massive current worker-log growth (~`4.4-4.5 GiB` per scheduler worker log), which is polluting the machine but is not yet proven to be the first admission failure source

## Host2 Cleanup + Narrower Budget Trace
- Per latest host2-focused request, completed the local-noise cleanup first:
  - killed the three stale `python3 test_qwen_rswa.py` GPU processes on host2
  - truncated the oversized scheduler worker logs on host2 (`worker*.err`, 49 files at the time of cleanup)
- Important runtime-path correction:
  - the first manual sync only patched
    - `/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/managers/scheduler.py`
  - but live host2 workers actually import:
    - `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py`
  - verified with runtime import:
    - `import sglang.srt.managers.scheduler as s; print(s.__file__)`
    - output: `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py`
- After copying the patched scheduler into the live `site-packages` path and scoped-restarting only `inst1` / `inst3`, the new traces finally became authoritative.

## First Host2 NO_TOKEN Budget Source Is Now Pinned
- Replayed one short request per instance after the live-path patch:
  - `inst1 @ 50053` -> new `rid='b67c7c51a7664ca0b812af3e3cba5492'`
  - `inst3 @ 50073` -> new `rid='0f684ba58e704265961f52bddae77b98'`
- Both requests still timed out from the client side, but the scheduler-side evidence changed materially:
  - `handle_generate_request_received`
  - `init_req_max_new_tokens_after ... after_max_new_tokens=4`
  - `request_enqueued_waiting_queue ... waiting_queue_size=1`
  - and then, crucially, the first budget failure is logged as:
    - `[PrefillAdderTrace] stage='no_token' reason='pre_lock_swa_budget_exceeds_budget'`
- Concrete host2 evidence for **both** `inst1` and `inst3`:
  - `extend_input_len=2`
  - `max_new_tokens=4`
  - `rem_total_tokens=1024`
  - `cur_rem_tokens=1024`
  - `rem_input_tokens=16384`
  - `rem_chunk_tokens=256`
  - `rem_swa_tokens=0`
  - `page_size=256`
  - `sliding_window_size=128`
  - `waiting_queue_len=1`
  - `swa_needed=384`

## Interpretation
- This changes the diagnosis materially:
  - the first host2 admission failure is **not** `available_size()/evictable_size()/rem_total_tokens` going to zero
  - the first failure is the **hybrid SWA budget gate**
  - specifically: `swa_needed >= rem_swa_tokens`
- This matches the actual code path in `schedule_policy.py`:
  - `PrefillAdder.rem_swa_tokens = swa_available_size() + swa_evictable_size() - rem_swa_token_offset`
  - `add_one_req()` checks:
    - `if swa_needed >= self.rem_swa_tokens: return AddReqResult.NO_TOKEN`
- For the current replay shape:
  - `_swa_budget_for_req(extend_input_len=2)` computes `swa_needed=384`
    - `max(extend_input_len=2, sliding_window_size=128) + page_size=256`
    - => `128 + 256 = 384`
  - but host2 runtime reports `rem_swa_tokens=0`
  - so the request is rejected **before** the general `rem_total_tokens=1024` budget is exhausted

## Why rem_swa_tokens Is 0
- Host2 startup logs after restart already show the root contributor:
  - `DSV4 pool sizes: full=1024, swa=0, c4=256, c128=8, ...`
  - `Memory pool resolve post-constraint: constrained=1024 ... full=1024 swa=0`
- Therefore the current host2 path is:
  1. launcher-side `max_total_num_tokens=1024` fix removed the old `max_new_tokens -> 0` bug
  2. but with `page_size=256` and the current DSV4 hybrid-SWA pool resolution, the constrained runtime pool ends up with `swa=0`
  3. first request then fails immediately at the **pre-lock SWA budget gate**
  4. scheduler later falls into the already-known `batch_is_full` / `prefill_return_none_early` sticky loop

## Refined Conclusion
- Host2's current first blocker is now pinned much more tightly:
  - **first blocker:** hybrid SWA pool budget collapse (`rem_swa_tokens=0`)
  - **second blocker / latch:** `NO_TOKEN -> batch_is_full=True -> prefill_return_none_early`
- So the next minimal technical direction is no longer "find which of available/evictable/rem_total_tokens hit zero first".
- The first budget source is already identified:
  - `PrefillAdderTrace reason='pre_lock_swa_budget_exceeds_budget'`
  - caused by `rem_swa_tokens=0` while `swa_needed=384`

## Host2 Root Cause for `constrained -> swa=0`
- Follow-up narrowed the question from "which budget hits zero first" to:
  - **why does host2 become `swa=0` immediately after max-token constraint?**
- This is now answered:
  - the live runtime was still importing `/usr/local/lib/python3.12/dist-packages/sglang/srt/model_executor/pool_configurator.py`
  - that live file did **not** include the local `preserve_min_swa_page=True` safeguard added in the workspace version
  - therefore `calculate_pool_sizes_from_max_tokens(max_total_num_tokens=1024, page_size=256)` used the old path:
    - `swa_tokens = int(full_token * swa_ratio) // page_size * page_size`
    - with runtime inputs:
      - `swa_ratio=0.1`
      - `page_size=256`
      - `swa_page_size=128`
      - `full_token=1024`
    - computed result:
      - `int(1024 * 0.1) // 256 * 256 = 0`
- This was confirmed directly in host2 startup logs before the fix:
  - `DSV4 swa sizing inputs: swa_ratio=0.1 page_size=256 swa_page_size=128 full_token=1024 swa_tokens=0`
  - `Memory pool resolve post-constraint: constrained=1024 ... full=1024 swa=0`

## Host2 Fix Applied
- Patched local `temp/runtime_ops/remote_runtime_ops.py` so runtime sync now includes:
  - `.../sglang/srt/model_executor/pool_configurator.py`
  - both to the remote repo tree and live `site-packages`
- Manually copied the patched `pool_configurator.py` into host2:
  - `/usr/local/lib/python3.12/dist-packages/sglang/srt/model_executor/pool_configurator.py`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/model_executor/pool_configurator.py`
  - `/root/flashkv0516/backend/cloud_sglang/python/sglang/srt/model_executor/pool_configurator.py`
- Then scoped-restarted only `inst1` and `inst3`.

## Post-Fix Verification
- After restart, host2 startup logs changed exactly as expected:
  - `DSV4 pool sizes: full=1024, swa=512, c4=256, c128=8, ...`
  - `Memory pool resolve post-constraint: constrained=1024 page_size=256 max_total=1024 full=1024 swa=512`
- This proves the earlier `constrained -> swa=0` root cause is resolved.

## What Changed in Runtime Failure Shape
- Replaying short requests after the pool fix still times out from the client side, but the **first blocker moved**:
  - the old first blocker
    - `reason='pre_lock_swa_budget_exceeds_budget'`
    - is no longer the first failure
  - the new observed host2 admission failure is:
    - `[PrefillAdderTrace] stage='no_token' reason='budget_state'`
    - `rem_total_tokens=508`
    - `cur_rem_tokens=512`
    - `rem_swa_tokens=0`
    - `can_run_list_size=1`
    - `waiting_queue_len=1`
- Interpretation:
  - host2 now **can** reserve the initial SWA pool and admit one candidate into `can_run_list`
  - but immediately after that reservation, the remaining SWA budget drops to `0`
  - so the scheduler no longer fails at "constraint produced `swa=0`", but at the next-stage budget-state exhaustion

## Updated Minimal Next Step
- The "why does constrained host2 become `swa=0`?" question is now closed.
- The new smallest next debugging target is:
  - **why does `rem_swa_tokens` fall from resolved `swa=512` to runtime `0` right after first admission (`can_run_list_size=1`) ?**

## Next Step Closed: Why `rem_swa_tokens` Drops to 0 After First Admission
- This follow-up is now also pinned with runtime evidence.
- After adding narrow traces in `schedule_policy.py`, host2 `inst1` / `inst3` both show the same first-admission sequence:
  - `add_one_req_pre_lock`: `swa_needed=384`, `rem_swa_tokens=512`
  - `add_one_req_post_lock`: still `swa_needed=384`, `rem_swa_tokens=512`
  - `update_prefill_budget_before`: `paged_extend_input_len=256`, `pending_swa_budget=512`
  - `update_prefill_budget_after`: `rem_swa_token_offset=512`, `rem_swa_tokens=0`
  - `budget_state_no_token`: `rem_swa_tokens=0`, `rem_total_tokens=508`, `cur_rem_tokens=512`
- Therefore the first post-admission SWA drain is caused by the scheduler's own reservation logic:
  - `_update_prefill_budget()` page-aligns the prefill length to `256`
  - then reserves SWA with `_swa_budget_for_req(extend_input_len=256)`
  - which becomes `max(256, 128) + 256 = 512`
- So the runtime failure shape is now:
  1. startup pool resolution succeeds (`swa=512`)
  2. first admission passes the pre-lock check (`swa_needed=384 <= 512`)
  3. `_update_prefill_budget()` immediately reserves the full `512`
  4. `rem_swa_tokens` becomes `0`
  5. `budget_state()` returns `NO_TOKEN`

## Host2 Goal Closure: Single-Question Chat Path on inst1/inst3
- After the SWA reservation fix, the remaining functional blocker was narrowed to prompt/template shaping rather than decode headroom:
  - naked `/v1/completions` prompts still produced JS-like garbage
  - manually wrapping prompts as `<｜begin▁of▁sentence｜><|User|>...<|Assistant|>` returned the correct answer `4`
- Productized fix:
  - added a built-in `deepseek-v4` conversation template in `sglang/srt/parser/conversation.py`
  - mapped `model_type=deepseek_v4` to that template
  - synchronized `conversation.py` to both remote repo-tree and live `site-packages`
- Host2 serving verification now shows:
  - gateway logs: `Inferred chat template from model path: deepseek-v4`
  - `inst1` (`50053`) `/v1/chat/completions` returns `"4"` in about `4.51s`
  - `inst3` (`50073`) `/v1/chat/completions` returns `"4"` in about `10.39s`
  - both `/health` endpoints return `status=ok`
- `inst3` was found down during the first re-check after cleanup and was relaunched; final verification was executed only after the gateway returned to ready.
- Evidence artifact:
  - `ComputeGraphCompiler-main/Output/cli_gate_upkg39/host2_chat_single_question_report.json`
- Conclusion:
  - the immediate host2 objective is met for the chosen production path:
    - real single-question generation works on both `inst1` and `inst3`
    - extra SWA headroom is not required for this verified chat path

## FusionRoute Ingress Evidence Closure
- A real upstream ingress request was executed through the local `app.servers.cgc_api_server` path using `/api/generate`.
- Because the local Mac could not reach host2 public gateway directly, the verification used a one-shot SSH tunnel to host2 `inst1` gateway `50053`.
- The ingress path now works end-to-end for the verified request after fixing three HTTP fallback gaps in `app/servers/cgc_api_server.py`:
  - normalize string / prompt-style payloads into OpenAI `messages`
  - strip Ollama-only fields (`options`, `raw`, `use_omlx`, `use_flashmoe`) before calling `/v1/chat/completions`
  - preserve the requested cloud model when `api/generate` and `v1/messages` fall back to the HTTP chat gateway
- Verification result:
  - ingress request prompt: `What is 2+2? Answer with one token.`
  - target gateway: host2 `inst1` `50053`
  - final response: `"4"`
  - local router evidence emitted successfully at `Output/cli_gate_upkg39/fusionroute_ingress_router_runtime.json`
  - structured report emitted at `Output/cli_gate_upkg39/fusionroute_ingress_validation_report.json`
- Boundary:
  - this proves a real ingress request plus local FusionRoute router evidence plus a real landing on one cloud instance
  - it does **not** yet prove dynamic cloud-side selection across all four instance gateways, because the ingress validation used a fixed tunnel target and the current runtime still lacks public gateway-orchestrator evidence

## FusionRoute Cloud Orchestrator Closure
- The missing cloud-side public ingress has now been filled with a minimal runtime component:
  - `app/servers/fusionroute_cloud_orchestrator.py`
  - launched on host1 `:50052`
  - uses `healthy_round_robin` across the four TP4EP4 gateways
  - uses `sshpass + ssh -L` tunnels for host2 `inst1/inst3`, because host1 and host2 cannot directly reach each other's public gateway ports
- Real validation result:
  - one public endpoint: `host1:50052`
  - same request replayed four times
  - landing sequence observed in runtime evidence:
    - request 1 -> `inst2`
    - request 2 -> `inst3`
    - request 3 -> `inst4`
    - request 4 -> `inst1`
  - placement JSON and runtime JSON were emitted successfully
- This closes the exact evidence gap that remained before:
  - a single cloud-side ingress now performs real dynamic selection across `inst1/inst2/inst3/inst4`
  - the landing is no longer manually pinned to one instance
- Remaining runtime quality issue:
  - routing/orchestration is now proven
  - but output quality is still inconsistent across the pool:
    - `inst1` / `inst4` returned `"4"`
    - `inst2` returned `"five"`
    - `inst3` returned `""`
