# [OPEN] Debug Session: scheduler-actor-init

## Goal
- Determine why backend startup fails with `Scheduler actor failed to initialize`.
- Collect only the minimal runtime evidence needed to answer:
  - backend start
  - scheduler actor create
  - actor init fail top frame
  - whether `30000` recovers

## Scope
- Instrumentation or log collection only.
- No business logic changes before runtime evidence is confirmed.
- Single-host focus: `host1`

## Falsifiable Hypotheses
- H1: backend subprocess starts, but Ray scheduler actor creation fails before the HTTP server binds `30000`.
- H2: scheduler actor is created, but `DynamicSchedulerActor.__init__()` crashes inside `Scheduler.init_model_worker()` and exits before readiness.
- H3: the failure is caused by a startup-time environment or packaging mismatch introduced during the latest restart, not by the new gateway/backend route instrumentation itself.
- H4: the scheduler actor failure is a downstream model-worker initialization error, and the visible `Scheduler actor failed to initialize` is only a wrapper symptom.

## Evidence Targets
- backend startup markers around the latest restart window
- scheduler actor creation / init failure stack top frame
- process / port evidence for `30000`
- whether backend ever reaches a ready marker after the failure

## Evidence Collected
- Focused probe helper:
  - [probe_host1_scheduler_actor_failure.py](file:///Users/alexchuang/Documents/flashkv0516/temp/misc/probe_host1_scheduler_actor_failure.py)
- Focused probe artifact:
  - [host1_scheduler_actor_failure_focus.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_scheduler_actor_failure_focus.json)
- Endpoint status artifact:
  - [host1_gateway_endpoint_probe.json](file:///Users/alexchuang/Documents/flashkv0516/temp/host1_gateway_endpoint_probe.json)

## Current Findings
- `backend start` is confirmed by the latest launch marker:
  - `=== backend start 2026-06-30 07:16:43 ===`
- `scheduler actor create` is confirmed by Ray actor creation failure evidence:
  - `ray::sglang_scheduler...:DynamicSchedulerActor.__init__()`
- The failure chain is:
  - `DynamicSchedulerActor.__init__()`
  - `Scheduler.init_model_worker()`
  - `Scheduler.init_tp_model_worker()`
  - `TpModelWorker.__init__()`
  - `ModelRunner.init_torch_distributed()`
  - `initialize_model_parallel()`
  - `GroupCoordinator(...)`
  - `PyNcclCommunicator(...)`
  - `ncclCommInitRank`
- The current deepest confirmed top frame is:
  - `RuntimeError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN for details)`
- This confirms that `Scheduler actor failed to initialize` is a wrapper symptom, not the deepest cause.

## 30000 State
- `30000` is no longer in the earlier `connection refused` state.
- Current focused probe shows:
  - `ss -ltnp` contains `127.0.0.1:30000`
  - `/v1/models` returns `200`
  - `/health` currently returns `503`
- So `30000` has partially recovered to a listening state, but it is not yet healthy enough to restore the gateway path.

## Hypothesis Status
- H1: supported
  - backend startup does progress into scheduler actor creation before full readiness
- H2: supported
  - `DynamicSchedulerActor.__init__()` dies while initializing the model worker path
- H3: not supported by current evidence
  - the deepest failure remains the pre-existing NCCL init path, not the new route instrumentation
- H4: supported
  - `Scheduler actor failed to initialize` is a wrapper over the deeper `NCCL error: invalid usage`
