# Debug Session: chat-disconnect-abort

Status: OPEN

## Symptom

- `50053/v1/chat/completions` and `30000/v1/chat/completions` both time out with `0 bytes received`.
- Server-side traces already show `handler_non_streaming_response_created`.
- `TokenizerTrace` shows `client_disconnect_abort_type1` followed by `send_abort_to_scheduler`.

## Expected

- A single non-streaming chat completion returns a real HTTP response body to the client.

## Current Boundary

- Old DSV4/SWA mapping stall has been pushed through.
- Latest boundary is no longer premature disconnect first.
- Chat requests still enter scheduler waiting queue, but a stale `HEALTH_CHECK_*` request remains ahead of them.
- New code-level suspicion: `/health` may cancel and clean local state before its scheduler abort is actually sent.

## Falsifiable Hypotheses

1. `request.is_disconnected()` is being polled while the upstream transport is already considered closed by Ray Serve / FastAPI, even though the downstream response object is created later.
2. The non-streaming path creates the response object, but an additional await / serialization / flush step still blocks long enough for the disconnect watchdog to fire first.
3. The gateway path on `50053` is not the root cause; the same premature disconnect also happens on direct `30000`, which would implicate the backend HTTP serving stack instead of the forwarding layer.
4. Health checks and chat completions share the same disconnect detection branch, and aggressive timeout polling in `_wait_one_response()` is producing false-positive disconnects under the current serving topology.
5. The request lifecycle state is not being cleared in time after `state.out_list` is populated, so tokenizer still enters the timeout branch and aborts a request that already has a ready response.

## Plan

1. Verify whether the lingering `HEALTH_CHECK_*` rid ever receives `abort_request_received` on scheduler workers.
2. Inspect the tokenizer-side abort guard and health cleanup ordering.
3. Apply a minimal fix so `/health` can force scheduler-side abort even if local request state has already been removed.
4. Re-run chat-only and health probes, then compare queue depth and completion behavior.

## Latest Evidence

- Runtime artifacts show chat requests entering with `waiting_queue_size=2` and being popped back to `1` only after client timeout abort.
- The remaining queued request is `HEALTH_CHECK_1782657354.9846473`, which repeatedly shows `result='NO_TOKEN'` and `running_batch_is_full=True`.
- Recent scheduler grep shows `abort_request_received` for the timed-out chat rid, but no corresponding abort record for the lingering health rid.
- `TokenizerManager.abort_request()` currently returns early when `tokenizer_worker_num == 1` and the rid is no longer present in `rid_to_state`, which can suppress the scheduler abort if `/health` cancelation races with local cleanup.
- Newer post-fix logs prove the `/health` handler now reaches `health_timeout_reached -> health_cleanup_begin -> health_cleanup_done`, and `TokenizerManager` sends `AbortReq` twice for the same health rid (once from `_wait_one_response()` disconnect handling and once from cleanup with `force_send=True`).
- Scheduler workers also prove they receive the health `AbortReq` on the input socket (`recv_from_tokenizer_raw payload_type='AbortReq'`) but still never emit `abort_request_received`.
- Static code review confirms the pre-dispatch health fast-path in `Scheduler.process_input_requests()` matches any object whose rid starts with `HEALTH_CHECK_`, including `AbortReq`, so health aborts are incorrectly skipped before reaching `self.abort_request(...)`.

## Constraints

- Do not rerun `swe verified 500` before one real completion returns successfully.
- Keep instrumentation minimal and evidence-driven.
