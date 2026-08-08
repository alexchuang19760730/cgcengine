# [OPEN] Debug Session: nccl-invalid-usage

## Goal
- Determine the minimal runtime evidence chain for `NCCL error: invalid usage`.
- Collect only:
  - launch flags / env
  - scheduler actor rank/world info
  - `ncclCommInitRank` pre-call parameters
  - whether `30000` health turns green

## Scope
- Instrumentation and log collection only.
- No business logic changes before runtime evidence is confirmed.
- Single-host focus: `host1`

## Falsifiable Hypotheses
- H1: the latest launch flags or runtime env produce an invalid distributed init shape before NCCL communicator creation.
- H2: scheduler actor rank/world metadata is inconsistent with TP/EP/world sizing, so `ncclCommInitRank` is called with a broken rank topology.
- H3: `ncclCommInitRank` is invoked with invalid pre-call parameters even though the higher-level actor init path appears normal.
- H4: backend `30000` can reach a listening state, but health stays degraded because NCCL init failure leaves the backend only partially alive.

## Evidence Targets
- latest backend launch command / env markers
- scheduler actor init rank/world metadata near the failure
- `ncclCommInitRank` pre-call argument snapshot
- `30000` listen / `/health` status after restart

## Instrumentation Applied
- Remote patch helper:
  - [patch_remote_nccl_invalid_usage_debug.py](file:///Users/alexchuang/Documents/flashkv0516/temp/misc/patch_remote_nccl_invalid_usage_debug.py)
- Remote patch artifact:
  - [host1_nccl_invalid_usage_patch.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_nccl_invalid_usage_patch.json)
- The patch adds:
  - `CA`: scheduler actor before `Scheduler(...)`
  - `CB`: scheduler actor init exception
  - `CC`: scheduler actor init done
  - `CD`: before `ncclCommInitRank`
  - `CE`: `ncclCommInitRank` exception
  - `CF`: `ncclCommInitRank` returned

## Runtime Artifacts
- Debug server startup:
  - [host1_remote_debug_server_nccl_invalid_usage.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_remote_debug_server_nccl_invalid_usage.json)
- Debug log sample:
  - [host1_nccl_invalid_usage_logs.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_nccl_invalid_usage_logs.json)
- Launch/env and health sample:
  - [host1_nccl_invalid_usage_state.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_nccl_invalid_usage_state.json)
- Restart artifact:
  - [host1_swebench_gateway_restart.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_swebench_gateway_restart.json)

## Current Findings
- Launch flags / env for the latest restart are consistent with:
  - `tp_size=4`
  - `ep_size=1`
  - `nnodes=1`
  - `dist_init_addr=127.0.0.1:29500`
  - `context_length=4096`
  - `max_total_tokens=8192`
  - `chunked_prefill_size=2048`
  - `NCCL_DEBUG=WARN`
  - `NCCL_DEBUG_SUBSYS=INIT,ENV`
- Scheduler actor rank/world evidence is collected for all 4 TP ranks:
  - `CA` appears for `tp_rank=0,1,2,3`
  - each actor reports `server_tp_size=4`, `server_ep_size=1`, `server_nnodes=1`
  - each actor is assigned one visible GPU and resolves to logical `cuda:0` inside its per-process `CUDA_VISIBLE_DEVICES`
- `ncclCommInitRank` pre-call parameters are collected for all 4 ranks:
  - `world_size=4`
  - `rank=0,1,2,3`
  - `group_ranks=[0,1,2,3]`
  - `device=cuda:0`
  - identical `unique_id_prefix`
  - `nccl_version=2.30.7`
- In this current restart window:
  - `CD` exists for all ranks
  - `CE` does not appear
  - `CF` appears for all ranks
- This means `ncclCommInitRank` itself returns successfully in the current run.

## 30000 Health
- Despite successful `CF` events, `30000` does not recover to healthy state in the same sampling window:
  - `ss_30000`: empty
  - `backend_health`: `Connection refused`
  - `backend_models`: `Connection refused`
  - `gateway_health`: `200`, but reports backend down
- Therefore `30000 health 是否转正`: no

## Hypothesis Status
- H1: not yet supported as root cause
  - launch flags/env are now captured, but current evidence does not show an obviously invalid distributed shape
- H2: not supported for the current run
  - scheduler actor rank/world metadata is internally consistent with `tp_size=4`
- H3: rejected for the current run
  - `ncclCommInitRank` returns successfully on all observed ranks
- H4: supported
  - backend still fails to become healthy even after NCCL communicator init succeeds

## Interim Conclusion
- The historical `NCCL error: invalid usage` is confirmed in earlier log windows, but it is not reproduced in the latest instrumented restart.
- The active blocker has moved deeper into post-NCCL startup: after `ncclCommInitRank` returns, but before `30000` becomes healthy.
- Because `30000` still does not turn green, the transport frontier cannot yet be resumed.

## Post-NCCL Instrumentation
- Remote patch helper:
  - [patch_remote_post_nccl_startup_debug.py](file:///Users/alexchuang/Documents/flashkv0516/temp/misc/patch_remote_post_nccl_startup_debug.py)
- Remote patch artifact:
  - [host1_post_nccl_startup_patch.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_post_nccl_startup_patch.json)
- Added downstream events:
  - `CG`: waiting for scheduler ready
  - `CH`: scheduler ready
  - `CI`: tokenizer manager ready
  - `CJ`: http server lifespan ready
  - `CK`: `/v1/models` ready
  - `CL/CM/CN`: `/health` status transitions

## Post-NCCL Findings
- In the latest restart window:
  - `CF` still appears for all 4 ranks
  - `CC` now also appears for all 4 TP ranks
  - `CB` does not appear
- This confirms the gate has advanced from:
  - `ncclCommInitRank returned`
  - to `Scheduler init done`
- However, no downstream post-NCCL startup events appear:
  - no `CG`
  - no `CH`
  - no `CI`
  - no `CJ`
  - no `CK`
  - no `CN`
- The current state probe still shows backend not up:
  - `backend_health`: `Connection refused`
  - `backend_models`: `Connection refused`
  - `ss_30000`: empty

## Current Gate Table
- `ncclCommInitRank returned`: yes, confirmed by `CF`
- `scheduler init done/exception`: done, confirmed by `CC`; no `CB`
- `backend /v1/models ready`: no evidence; probe is still `Connection refused`
- `/health 转正`: no evidence; probe is still `Connection refused`
- `当前主 blocker`: after `Scheduler init done`, before backend HTTP startup reaches `/v1/models` or `/health`

## Ray HTTP Startup Correction
- The previous `CG/CH/CI` instrumentation was placed in `entrypoints/engine.py`, but the active ray path is:
  - `sglang/srt/ray/http_server.py`
  - `RayEngine._launch_subprocesses()`
  - `_setup_and_run_http_server(...)`
- Therefore the reliable next boundary for the ray mode is:
  - `CO`: before `RayEngine._launch_subprocesses`
  - `CP`: `_launch_subprocesses` exception
  - `CQ`: `_launch_subprocesses` returned
  - `CR`: before `_setup_and_run_http_server`
  - `CS`: `_setup_and_run_http_server` exception

## Ray HTTP Startup Instrumentation
- Remote patch helper:
  - [patch_remote_ray_http_startup_debug.py](file:///Users/alexchuang/Documents/flashkv0516/temp/misc/patch_remote_ray_http_startup_debug.py)
- Remote patch artifact:
  - [host1_ray_http_startup_patch.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_ray_http_startup_patch.json)

## Latest Findings
- In the latest restart window:
  - `CO` appears
  - `CA` appears for scheduler actors
  - `CD` and `CF` appear
  - but no `CQ`
  - and no `CR/CS`
- This means the latest run does not even return from `RayEngine._launch_subprocesses()`.
- A focused backend log tail shows the deeper cause in the same window:
  - `RuntimeError: NCCL error: invalid usage`
  - raised during `Scheduler.init_model_worker -> init_tp_model_worker`
  - then wrapped as `RuntimeError: Scheduler actor failed to initialize`
- So the current active blocker has moved back earlier than the temporary `CC`-reaching run:
  - `CO -> _launch_subprocesses -> scheduler actor NCCL invalid usage`

## Revised Gate Table
- `CO before RayEngine._launch_subprocesses`: yes
- `CQ _launch_subprocesses returned`: no
- `CR before _setup_and_run_http_server`: no
- `backend /v1/models ready`: no
- `/health 转正`: no
- `当前主 blocker`: inside `RayEngine._launch_subprocesses()`, with the latest focused backend log again pointing to `Scheduler actor failed to initialize -> NCCL error: invalid usage`
