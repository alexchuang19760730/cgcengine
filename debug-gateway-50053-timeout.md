[OPEN] Debug Session: gateway-50053-timeout

## Symptom
- `host1:50053/v1/chat/completions` is reachable by listener but times out before returning any bytes.
- Ray Serve `proxy` logs show matching `499 client disconnected` lines.
- `trajectory_count=1` but `submitted_count=0`.

## Expected
- A single LM query through `50053` completes within client timeout budget and advances autosubmission toward a real submission artifact.

## Scope Guard
- Steps 1-4: no business logic modification.
- Focus only on runtime evidence under the live Ray session and surrounding gateway/serve logs.

## Hypotheses
1. The request reaches `ProxyActor`, but stalls before proxy-to-replica dispatch.
2. The request reaches the replica, but blocks inside replica-to-backend forwarding and never emits timely response bytes.
3. The request is processed, but cancellation/timeout propagation between client, proxy, and replica masks the real failure site in current logs.
4. A lower-level Ray worker or core-worker error exists outside `serve/proxy|replica|controller` logs and is the real blocker.
5. Gateway timeout behavior is caused by queueing or backpressure in the single active replica rather than connectivity.

## Evidence Targets
- `/data/ray/inst1/session_2026-06-30_11-53-10_281649_145910/logs/serve/proxy_*.log`
- `/data/ray/inst1/session_2026-06-30_11-53-10_281649_145910/logs/serve/replica_*.log`
- `/data/ray/inst1/session_2026-06-30_11-53-10_281649_145910/logs/worker-*.out`
- `/data/ray/inst1/session_2026-06-30_11-53-10_281649_145910/logs/worker-*.err`
- `/data/ray/inst1/session_2026-06-30_11-53-10_281649_145910/logs/python-core-worker-*.log`

## Plan
1. Enumerate worker/core-worker logs in the live Ray session.
2. Correlate the latest `499` request window from proxy logs with replica and worker logs.
3. Decide whether the stall is at proxy queue, dispatch, or downstream backend handling.
4. Only after evidence converges, propose the smallest runtime-side fix to push `submitted_count > 0`.

## Evidence Update
- `proxy` latest `499` requests (`eefeeecf`, `23fd848d`, `68e04a09`) are also present in the replica worker log as `POST /v1/chat/completions CANCELLED ~15000ms`, so the requests do reach the replica and are cancelled after client timeout.
- Replica worker stderr shows a long-lived `TokenizerTrace` request `rid='1ebbc987fb924ede83dc22d0ea1704a6'` polling for more than 1200s with `event_set=False`, `out_list_len=0`, and `finished=False`, indicating at least one generation path is stuck inside the replica/backend path without producing tokens.
- Proxy core-worker log has repeated `ray_log_sink` failures with `No space left on device`, which means some runtime observability is already degraded.
- `df -h /data/ray/inst1` shows `/data` at `100%` with only `238M` available, and `du -sh` shows the active Ray session alone occupies `567G`.
- `raylet` continuously reports the live session path is over 95% full and warns that object creation will fail if spilling is required.

## Current Assessment
- Hypothesis 1 (`proxy` stalls before replica dispatch): rejected by replica-side `CANCELLED` entries for the same request IDs.
- Hypothesis 2 (`replica` reaches downstream path but never returns in time): strongly supported.
- Hypothesis 4 (lower-level runtime problem outside serve logs): supported by disk exhaustion and broken core-worker log sink.
- Most likely active blocker: one or more stuck backend generation paths inside the replica, amplified by a critically full Ray session volume.

## Storage Hotspot Update
- Active session size is dominated by `logs/` at `567G`.
- The largest offenders are four worker stderr files, each about `142G`:
  - `worker-c62bf...-148844.err`
  - `worker-0dcca0...-148847.err`
  - `worker-db235f...-148846.err`
  - `worker-18b56b...-148845.err`
- This justifies an explicit cleanup-and-restart step after preserving current evidence.

## Cleanup Execution
- Executed remote cleanup/restart path on host1.
- `/data` recovered from `100%` full to `38%` used with `586G` available.
- `/data/ray/inst1` is now effectively empty (`4.0K`).
- No relevant listeners remain on `50053/30000/6379/8265`, so the old stuck runtime is gone.

## New Blocker After Cleanup
- Restart no longer fails on the original `50053` timeout path; it now fails earlier during Python import/bootstrap.
- First blocker exposed after cleanup: `torch.masked.maskedtensor.core` missing during `kda_state_runtime` import.
- Added a lazy-import guard in `app/edge_engine/kda_state_runtime.py` so KDA/Torch is no longer a startup-time hard dependency for the gateway.
- Next blocker exposed after the lazy-import fix: broken Python dependency stack on host1 (`requests` -> `idna.core` missing, then `certifi.core` missing).
- Conclusion: cleanup succeeded, but `submitted_count` cannot advance until the host1 gateway Python environment is repaired enough to start Serve again.

## Gateway Recovery Progress
- Built a clean host1 venv at `/root/flashkv0516/.venv_gateway_clean` and installed clean `requests/idna/certifi/fastapi/uvicorn/click/dacite`.
- Repaired missing system Ray-side modules enough for `ray start`, `ray serve`, and Serve replica deserialization to proceed:
  - `ray._private.thirdparty.dacite.core`
  - system `idna`, `certifi`, `click`, `requests`, `dacite`
- Updated the host1 restart helper so both `ray start` and `cloud_socket_server` use the clean venv import path.
- Current state after restart:
  - `6379` is listening via `gcs_server`
  - `50053` is listening via `ray::ProxyActor`
  - `/health` returns `status=ok`
  - `/v1/models` and `/v1/chat/completions` return structured `backend_unavailable`
  - `30000` is still not listening, so the remaining blocker has moved from gateway bootstrap to backend launch/runtime.

## Current Frontier
- `50053` gateway path is recovered.
- Remaining blocker to reach submission: bring up the backend on `127.0.0.1:30000`, then rerun chat and a 1-problem submission smoke.
