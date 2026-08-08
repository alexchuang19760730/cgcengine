# Unified Runtime IR v0

> Status: Draft v0
> Scope: Colibri / CGC / Hermes unified control-plane IR for multi-model, multi-backend runtime dispatch

## 1. Goal

`Unified Runtime IR v0` is a control-plane IR, not a kernel IR.

It exists to let Colibri / CGC / Hermes describe one runtime intent and lower it into different backend adapters, such as:

- `gemma4.c`
- `TurboFieldfare`
- `mlx`
- `llama.cpp`
- `sglang`

The target execution chain is:

```text
4D Matrix / TenStepPipeline / ProfileBinding
    -> Unified Runtime IR
    -> Adapter.lower(ir)
    -> BackendRequest
    -> begin_request / prefill / generate / snapshot
```

The design rule is simple:

- unify intent
- do not unify backend internals

## 2. Layering

`Unified Runtime IR v0` sits between Hermes/4DSP control-plane logic and backend-specific execution.

It is not responsible for:

- kernel launch details
- paging algorithm internals
- slot-level cache heuristics
- CUDA / Metal tuning knobs

It is responsible for:

- model identity
- execution mode
- residency intent
- unit placement intent
- prefetch / prime intent
- backend capability negotiation
- adapter lowering boundary

## 3. Canonical Object

```json
{
  "ir_version": "unified_runtime_ir_v0",
  "request_id": "req_xxx",
  "model": {
    "model_id": "gemma4-26b-a4b",
    "model_family": "gemma4",
    "model_format": "safetensors",
    "architecture": "moe_decoder",
    "quantization": "qat_4bit"
  },
  "runtime": {
    "mode": "local_verify_loop",
    "execution_intent": "streaming_decode",
    "backend_family": "auto",
    "backend_hint": "",
    "device_class": "apple_silicon",
    "platform": "macos"
  },
  "decode_strategy": {
    "strategy_family": "standard",
    "speculative_mode": "none",
    "max_tokens": 128,
    "stream": true
  },
  "residency": {
    "policy_family": "tiered_streaming",
    "target_tier": "ram",
    "pin_budget_bytes": 0,
    "resident_budget_bytes": 0,
    "prefetch_semantics": "best_effort",
    "bootstrap_semantics": "decode_preprime"
  },
  "placement": {
    "runtime_unit_plan": {},
    "current": [],
    "next": [],
    "future": []
  },
  "telemetry": {
    "snapshot_level": "standard",
    "emit_runtime_request": true,
    "emit_backend_snapshot": true
  },
  "adapter": {
    "required_capabilities": [],
    "optional_capabilities": []
  }
}
```

## 4. Common Fields

These fields are allowed in unified IR.

### 4.1 Model

- `model.model_id`
- `model.model_family`
- `model.model_format`
- `model.architecture`
- `model.quantization`

These are routing and compatibility fields.

### 4.2 Runtime

- `runtime.mode`
- `runtime.execution_intent`
- `runtime.backend_family`
- `runtime.backend_hint`
- `runtime.device_class`
- `runtime.platform`

These describe where and how the request should run, but not the backend's internal algorithm.

### 4.3 Decode Strategy

- `decode_strategy.strategy_family`
- `decode_strategy.speculative_mode`
- `decode_strategy.max_tokens`
- `decode_strategy.stream`

This layer can align with `SpecDecodeConfig`, but should stay backend-agnostic.

### 4.4 Residency

- `residency.policy_family`
- `residency.target_tier`
- `residency.pin_budget_bytes`
- `residency.resident_budget_bytes`
- `residency.prefetch_semantics`
- `residency.bootstrap_semantics`

This is the key bridge to streaming backends.

### 4.5 Placement

- `placement.runtime_unit_plan`
- `placement.current`
- `placement.next`
- `placement.future`

Each unit may carry:

- `key`
- `unit_kind`
- `layer_id`
- `expert_id`
- `target_tier`
- `routing_heat`
- `pin_priority`
- `resident`
- `pinned`
- `prefetched`
- `available`
- `io_backend`
- `path`
- `offset_bytes`
- `size_bytes`

This layer should map cleanly onto Colibri runtime-unit planning.

### 4.6 Telemetry

- `telemetry.snapshot_level`
- `telemetry.emit_runtime_request`
- `telemetry.emit_backend_snapshot`

This ensures different backends still expose a common observability surface.

### 4.7 Adapter Capability Contract

- `adapter.required_capabilities`
- `adapter.optional_capabilities`

Examples:

- `streaming_expert_units`
- `decode_preprime`
- `safetensors_tensor_slice`
- `kv_cache_persistent`
- `spec_decode_chain`

## 5. Adapter ABI

Every backend adapter should expose the same minimum ABI.

```python
class RuntimeBackendAdapter:
    def lower(self, ir: dict) -> dict:
        ...

    def begin_request(self, request: dict) -> dict:
        ...

    def prefill(self, prompt_or_tokens, **kwargs) -> dict:
        ...

    def generate(self, **kwargs):
        ...

    def snapshot(self) -> dict:
        ...

    def close(self) -> dict:
        ...
```

### 5.1 `lower(ir) -> request`

Purpose:

- validate capability match
- map unified fields into backend request
- attach backend-private config outside the IR boundary

This is the main cross-backend lowering point.

### 5.2 `begin_request(request)`

Purpose:

- create runtime session
- materialize control-plane state
- return a standard runtime snapshot

### 5.3 `prefill(...)`

Purpose:

- optional explicit prefill boundary
- useful for backends that distinguish `prefill` from `generate`

### 5.4 `generate(...)`

Purpose:

- streaming or non-streaming token generation

### 5.5 `snapshot()`

Purpose:

- expose backend state in a common shape

Recommended shared fields:

- `backend`
- `session`
- `runtime_request`
- `capabilities`
- `residency`
- `telemetry`

### 5.6 `close()`

Purpose:

- close session
- release request-scoped state

## 6. Backend-Private Fields

These fields should not enter unified IR.

### 6.1 `gemma4.c` private

- `slot1_protect_window`
- `slot1_adopt_gap`
- `replacement_keep`
- `replacement_sticky_scale_pct`
- `anchor_floor`
- `slot1_floor`
- `sticky_window`

Reason:

These are implementation knobs for one runtime's resident-slot policy.

### 6.2 TurboFieldfare private

- shared-core packing layout
- SSD paging chunk size
- expert residency window internals
- Metal command-buffer batching details
- backend-private decode scheduler knobs

Reason:

These belong to TurboFieldfare's engine design, not to the unified control-plane contract.

### 6.3 MLX / llama.cpp / sglang private

- `n_gpu_layers`
- CUDA graph flags
- Metal graph flags
- server launch flags
- backend-specific thread / batch / graph parameters

Reason:

These are adapter-lowered execution parameters.

## 7. How It Fits Hermes 4D + TenStepPipeline

`Unified Runtime IR v0` should be attached to Hermes / 4DSP as a new abstraction layer between:

- Step `7.6` model dispatch
- Step `8` execution context

Recommended interpretation:

- Step `7.6`: decide what backend family should run
- Step `7.7`: decide draft / verify sync policy
- Step `7.8` internal lowering hook: materialize `Unified Runtime IR`
- Step `8`: build execution context from lowered backend request

Important:

`Step 7.8` is an internal architectural hook, not necessarily a new public CLI step number.

## 8. Why This Enables Cross-Backend Execution

Because backend portability does not come from forcing all runtimes to look the same.

It comes from:

1. one common control-plane intent
2. one adapter ABI
3. many backend-specific lowerings

So:

- Hermes decides intent
- Unified Runtime IR records intent
- Colibri / CGC adapters lower intent
- each backend executes using its own strengths

## 9. Recommended Initial Backend Families

Recommended `backend_family` values for v0:

- `mlx`
- `llama.cpp`
- `sglang`
- `gemma4_native`
- `turbofieldfare`
- `edge_cloud_bridge`

## 10. Non-Goals

`Unified Runtime IR v0` does not try to:

- replace backend-specific engines
- standardize kernel code
- standardize slot/paging internals
- eliminate backend-specific telemetry

It only standardizes the boundary above them.
