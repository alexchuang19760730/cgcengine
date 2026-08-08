# [OPEN] Debug Session: ray-scheduler-launch

## Goal
- Narrow the active blocker inside `RayEngine._launch_subprocesses()`.
- Collect only:
  - `CO -> _launch_scheduler_processes returned/exception`
  - which `tp_rank` first fails inside `Scheduler(...)`
  - the last successful boundary before `NCCL invalid usage`

## Scope
- Instrumentation and runtime evidence only.
- No business logic changes before evidence is confirmed.
- Single-host focus: `host1`

## Falsifiable Hypotheses
- H1: the active run stalls or fails inside `_launch_scheduler_processes()` before it returns to `_launch_subprocesses()`.
- H2: one specific `tp_rank` consistently becomes the first failing actor inside `Scheduler(...)`, and the rest are secondary fallout.
- H3: there is still a deeper last-success boundary before `invalid usage`, but it is currently unobserved because `_launch_scheduler_processes()` lacks result-collection instrumentation.
- H4: the first failing `tp_rank` is not stable across restart windows, which explains the earlier apparent breakpoint drift.

## Evidence Targets
- `_launch_scheduler_processes` begin/returned/exception
- per-actor result collection order
- first failing `tp_rank`
- last successful runtime marker before `invalid usage`

## Instrumentation Applied
- Remote patch helper:
  - [patch_remote_ray_scheduler_launch_debug.py](file:///Users/alexchuang/Documents/flashkv0516/temp/misc/patch_remote_ray_scheduler_launch_debug.py)
- Remote patch artifact:
  - [host1_ray_scheduler_launch_patch.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_ray_scheduler_launch_patch.json)
- Added launch-side events:
  - `RA`: `_launch_scheduler_processes` begin
  - `RB`: scheduler actors created
  - `RC`: `actor.get_info` dispatched
  - `RD`: `actor.get_info` returned
  - `RE`: `actor.get_info` RayActorError
  - `RF`: get_info collection done
  - `RG`: `_launch_scheduler_processes` returned
- Added scheduler-side events:
  - `SA`: before `TpModelWorker(**worker_kwargs)`
  - `SB`: `TpModelWorker` init returned
  - `SC`: `TpModelWorker` init exception
  - `SD`: before `tp_worker.get_worker_info()`
  - `SE`: `tp_worker.get_worker_info()` returned

## Runtime Artifacts
- Debug server startup:
  - [host1_remote_debug_server_ray_scheduler_launch.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_remote_debug_server_ray_scheduler_launch.json)
- Dedicated session log fetch:
  - [host1_ray_scheduler_launch_logs.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_ray_scheduler_launch_logs.json)
- Focused backend log window:
  - [host1_ray_scheduler_launch_window.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_ray_scheduler_launch_window.json)
- State probe:
  - [host1_nccl_invalid_usage_state.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_nccl_invalid_usage_state.json)

## Findings
- The new remote instrumentation compiles successfully in both:
  - `sglang/srt/ray/engine.py`
  - `sglang/srt/managers/scheduler.py`
- The dedicated `ray-scheduler-launch` collector has continuity issues across restart windows, so the current effective runtime evidence comes from the focused backend log window.
- The latest backend log tail shows:
  - `PostLoadTrace event=launch_scheduler_processes_begin`
  - no corresponding `launch_scheduler_processes_returned` for the failing windows
  - repeated stack frames at `self.tp_worker = TpModelWorker(**worker_kwargs)`
  - repeated `RuntimeError: NCCL error: invalid usage`
  - wrapped as `RuntimeError: Scheduler actor failed to initialize`
- The latest surfaced failing actor is not stable:
  - some windows first surface `tp1`
  - later windows first surface `tp2`
- This supports H4 and weakens the idea of a single permanently failing rank.

## Current Gate Table
- `CO -> _launch_scheduler_processes begin`: yes
- `_launch_scheduler_processes returned/exception`: no successful return in the failing windows
- `哪个 tp_rank 在 Scheduler(...) 内 first fail`: unstable across windows, observed `tp1` and `tp2`
- `invalid usage 前最后一个成功边界`: `init_model_worker_begin` is visible, then execution enters `TpModelWorker(**worker_kwargs)` and fails before any stable `init_model_worker_done`
- `30000 health`: still not recovered

## Interim Conclusion
- The active blocker is now narrowed to the scheduler actor initialization path inside `_launch_scheduler_processes()`.
- In the failing windows, the runtime gets past:
  - actor creation
  - `init_model_worker_begin`
- but fails during:
  - `TpModelWorker(**worker_kwargs)`
- before a stable post-worker success boundary is observed.

## 2026-06-30 Recovery Update
- While pushing one layer deeper, a temporary instrumentation bug was introduced in `ray/engine.py`:
  - `UnboundLocalError: cannot access local variable 'dist_init_addr'`
  - caused by emitting `RA` before `dist_init_addr` was assigned
- This was fixed by moving the `RA` event after `dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"`.
- After the fix and a fresh restart window:
  - [host1_tp_model_worker_latest_window.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_tp_model_worker_latest_window.json) shows a clean new backend start
  - [host1_nccl_invalid_usage_state.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_nccl_invalid_usage_state.json) now shows:
    - `backend_health = 200`
    - `backend_models = 200`
    - `ss_30000` listening on `127.0.0.1:30000`
- Therefore the gating condition for returning to transport has been met:
  - `30000` is healthy again
  - the next active frontier is back to `gateway -> upstream -> client read`
