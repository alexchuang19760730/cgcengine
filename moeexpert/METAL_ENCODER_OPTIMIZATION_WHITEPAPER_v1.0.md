# Metal Encoder Optimization Technical Whitepaper v1.0

Date: 2026-08-18
Scope: fork of llama.cpp (llama_roadB), ggml-metal backend
Baseline (updated 2026-08-19): §8.79 + CGC_N_CB=8 + OA_ASYNC + build-prod2 (libllama 0.0.5) — **25.07 t/s** (wall-clock, 128 tok), decode 33.13 t/s, graph splits 162, hit rate 98.7%

## 1. Executive Summary

**Updated 2026-08-19**: Production fix achieved **25.07 t/s** wall-clock (+95% from 12.88), breaking through the original 13.7 t/s engine ceiling. Root causes: (1) CGC_N_CB=8 multi-buffer cutting MTL encoding from 60ms to ~30ms, (2) expert-cache hit rate 55%→98.7% eliminating I/O tail, (3) CGC_OA_ASYNC hiding CPU callback sync. Decode-only speed: 33.13 t/s.

The original 12.88→14-16 t/s ceiling was based on a misestimation: the three directions close the gap to 13.7 t/s engine ceiling, but the actual fix came from n_cb tuning + fused-down kernel + expert-cache improvements that were not in the original proposal.

### Remaining ceiling (25.07 → 28.6 t/s kernel floor)
- GPU kernel floor: 35ms/step (weight-read 24.9ms + dispatch bubble 38ms, bandwidth-bound)
- Gap to kernel floor: 25.07→28.6 t/s requires kernel rewrite (IQ3 inline dequant + sequential access pattern), 3-6 week effort
- Prompt eval overhead: 55 tokens × 14.52ms = 799ms, reduces wall-clock from 33.13 to 25.07 t/s

## 2. Background and §8.79 Baseline

### 2.1 Four-segment measurement (§8.79)

qwen36 IQ3_XXS, -ngl 99, 4GiB budget, CGC_OA_ASYNC=1, pool8, n=128, 127 decode steps:
- decode throughput: 12.88 t/s (matches §8.77 chart 12.8)
- eog_sync_wait p50=4.24ms / mean=4.82ms / p99=15.76ms (GPU tail drain, 12.4% of steps >5ms)
- dispatch_to_sync p50=0.01ms (app-side pre-sync, negligible)
- post_sync_to_next p50=0.15ms (sampler + KV + build + set_inputs, boundary serialization already ~0)
- graph_compute_wall p50=67ms / mean=85.53ms (CPU MTL encoding 60ms + GPU compute 35ms overlap, max~60ms)
- per-step wall p50=67ms / mean=79.77ms

### 2.2 Original "~25ms endpoint/boundary" was a misestimation

The §8.77 chart split 83.6ms into "GPU 35 + CPU 24 + endpoint/boundary 25". §8.79 four-segment measurement proved segments 1+2+3 (the endpoint/boundary portion) sum to p50=4.40ms, p99=18.96ms. The 25ms was a misallocation of the 67ms graph_compute_wall internal overlap structure (CPU encoding 60ms + GPU compute 35ms overlap, max~60ms, not 24+35=59).

### 2.3 The real bottleneck: CPU-side MTL encoding

Verified at `ggml-backend.cpp:1856`: OA_ASYNC dispatches Metal splits via `ggml_backend_graph_compute_async` -> `backend->iface.graph_compute` -> `ggml_metal_graph_compute`. The Metal backend's `graph_compute` is CPU-side MTLCommandBuffer encoding (synchronous blocking), not true async. 99 layers x multiple kernels per layer x ~19us per kernel setup (argument binding + threadgroup config + commit fence) consumes ~60ms.

GPU compute (~35ms) overlaps partially with CPU encoding inside graph_compute_wall: after each kernel's commit, GPU starts executing while CPU continues encoding subsequent kernels.

## 3. Current Code Architecture (post-review)

### 3.1 Entry: `ggml_metal_graph_compute` (ggml-metal-context.m:676)

```
n_main = MAX(64, 0.1 * gf->n_nodes)        // nodes encoded by main thread
n_cb = ctx->n_cb                           // command buffer count (1 or 2)
n_nodes_0 = MIN(n_main, gf->n_nodes)       // main thread range
n_nodes_1 = gf->n_nodes - n_nodes_0       // async thread range
n_nodes_per_cb = (n_nodes_1 + n_cb - 1) / n_cb
```

Flow:
1. Main thread: create cmd_bufs[n_cb], encode n_nodes_0 nodes, commit
2. `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)` - parallel encode remaining nodes
3. encode_async (line 1012) per cb: `ggml_metal_op_init` -> loop `ggml_metal_op_encode` -> commit

### 3.2 Per-op encode path: `ggml_metal_op_encode_impl` (ggml-metal-ops.cpp:193)

switch(op) dispatches to per-op handlers. Each handler (e.g. line 606-625):
```
ggml_metal_encoder_set_pipeline(enc, pipeline);              // PSO bind (cached as MTLComputePipelineState)
ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);   // args struct memcpy each call
ggml_metal_encoder_set_buffer(enc, src0, 1);                  // buffer bind per src/dst
ggml_metal_encoder_set_buffer(enc, src1, 2);
ggml_metal_encoder_set_buffer(enc, dst,  3);
ggml_metal_encoder_dispatch_threadgroups(enc, nw0, ne2, ne3, nth, nrptg, 1);
```

### 3.3 Encoder API surface (ggml-metal-device.m:494-510)

```
[encoder->obj setComputePipelineState:pipeline]              // PSO bind
[encoder->obj setBytes:data length:size atIndex:idx]         // args memcpy + internal immutable buffer
[encoder->obj setBuffer:buffer offset:offs atIndex:idx]      // buffer bind
[encoder->obj setThreadgroupMemoryLength:size atIndex:idx]
[encoder->obj dispatchThreadgroups:... threadsPerThreadgroup:...]
```

### 3.4 What is already cached

- PSO (MTLComputePipelineState): cached at `ggml_metal_device_get_pipeline` - pipeline build is one-time
- Graph structure: cached at llama.cpp graph cache (llama_context::graphs_reused) - per-step graph topology is identical for 127 decode steps

### 3.5 What is NOT cached (the optimization surface)

- args struct: `setBytes` does memcpy each call, allocates internal immutable buffer per dispatch
- buffer bindings: 3-4 `setBuffer:offset:atIndex:` calls per op, no reuse across steps
- threadgroup config: `MTLSizeMake` + `dispatchThreadgroups` per op, grid changes per op shape
- command buffer: created per graph (n_cb=1 or 2), committed per buffer

## 4. Direction 1: CPU-side Encoder Overhead Reduction

### 4.1 Design

Two sub-optimizations:

#### 4.1.1 Args pool (replace setBytes with setBuffer + pre-allocated ring)

Current: `setBytes:data length:size atIndex:0` per op - Metal internally memcpy's to an immutable buffer that lives until command buffer commit.

Proposed: pre-allocate a per-encoder ring buffer (e.g. 1MB), per-op args written sequentially via pointer arithmetic, bound via `setBuffer:offset:atIndex:0`. The ring wraps after commit; lifetime is guaranteed by the command buffer's retain cycle.

Risk: if ring overflows mid-graph (large prefill), fallback to setBytes path. Decode steady-state args structs are small (~64-256 bytes per op, 99 layers x ~10 ops/layer x 256B = ~250KB - fits in 1MB).

#### 4.1.2 Argument buffer (replace multiple setBuffer with MTLArgumentEncoder)

Current: 3-4 `setBuffer:offset:atIndex:` per op (src0, src1, dst, optional).

Proposed: use `MTLArgumentEncoder` (macOS 13+, MTLAppleGPUFamilyMetal3+) - bind a single argument buffer per op, set all buffer bindings via `[argumentEncoder setBuffer:offset:atIndex:]` into the argument buffer, then one `setBuffer:argumentBuffer atIndex:0` on the encoder.

Risk: requires macOS 13+ (M4 Max supports it); changes PSO descriptor setup (must declare argument buffer layout via MTLFunctionConstantValues); invasive change to all per-op handlers.

### 4.2 Implementation plan

Phase A (args pool only, low risk):
1. Add `ggml_metal_encoder_args_pool` struct to ggml-metal-impl.h - ring buffer + current offset
2. In `ggml_metal_encoder_init` (ggml-metal-device.m): allocate ring buffer via `ggml_metal_device_alloc_buffer` (1MB)
3. Add `ggml_metal_encoder_set_args_via_pool(enc, data, size, idx)`: write to ring, setBuffer with ring buffer + offset
4. Per-op handlers: replace `set_bytes(enc, &args, sizeof(args), 0)` with `set_args_via_pool(enc, &args, sizeof(args), 0)`
5. On ring overflow: fallback to setBytes, log to stderr (env-gated CGC_ENCODER_DEBUG=1)
6. After commit (in encode_async): reset ring offset to 0

Phase B (argument buffer, medium risk, only if Phase A shows measurable gain):
1. Detect macOS 13+ at device init; fallback to Phase A path if unsupported
2. For each PSO: build `MTLArgumentDescriptor` array describing expected buffers
3. Per-op handler: obtain `MTLArgumentEncoder` for the PSO, bind buffers into argument buffer, then `setBuffer:argBuf atIndex:0`
4. Argument buffer pool: same ring buffer pattern as args pool, but per-encoder

### 4.3 Measurement plan

- CGC_STEP_TIMING=1 - measure graph_compute_wall p50 before/after
- CGC_HOST_GPU_TIMING=1 - measure host_ms (CPU encode) vs gpu_ms before/after
- Expected: graph_compute_wall p50 67ms -> 60-64ms (5-10% reduction from eliminating per-op setBytes memcpy)
- If Phase B succeeds: 67ms -> 55-60ms (additional 5-10% from reducing buffer binding calls)

### 4.4 Risk assessment

- Args pool lifetime: ring buffer must outlive all in-flight command buffers. With n_cb=1 or 2, max in-flight is 2 - ring must be 2x max args size per graph, or use per-cb ring.
- Argument buffer macOS version gate: must runtime-detect, fallback to existing path on macOS <13.
- Per-op handler rewrite scope: Phase A touches all handlers that call `set_bytes` (estimated 50-100 sites in ggml-metal-ops.cpp). Phase B is more invasive (requires MTLArgumentDescriptor per PSO).

## 5. Direction 2: n_cb Tuning (revised from "single uncommitted encoder")

### 5.1 Why the original proposal conflicts with existing mechanism

`ggml-metal-context.m:768-808` already implements multi-command-buffer async encoding:
- Main thread commits cmd_bufs[n_cb] first
- `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)` parallel-encodes remaining buffers
- `ctx->cmd_buf_last` tracks the last queued buffer

Original direction 2 proposed "single uncommitted encoder to avoid per-layer commit fence overhead". But the existing mechanism already avoids per-layer commits - commits happen once per cb (n_cb=1 or 2), not per layer. Switching to a single uncommitted buffer would lose the async encoding overlap (main thread + n_cb background threads), likely making it slower.

### 5.2 Revised direction 2: n_cb tuning (RESULT: VALIDATED — see §8.81)

The comment at `ggml-metal-context.m:692` states: "tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2". M4 Max has 14+ CPU cores and higher memory bandwidth. Question: is n_cb=2 still optimal on M4 Max? Could n_cb=3 or 4 reduce per-thread encode time by splitting work across more cores?

**§8.81 result (2026-08-18): the fork's default was NOT 2 — it was 1** (set to 1 = single command buffer during the earlier per-op timing safety work). The scan showed n_cb=4 recovers ~20-25% (mean 106-126 -> 82-84 ms/run) by eliminating the >100ms encode spikes (14-21/48 steps -> 4/48), all four arms bit-identical. The env is `CGC_N_CB` (not CGC_METAL_N_CB), already present at ggml-metal.cpp:615. The 0-3% projection was wrong because it assumed upstream default 2; the fork had been running at 1.

### 5.3 Implementation plan

1. Expose `n_cb` as env-gated: `CGC_METAL_N_CB=N` (default 2, upstream parity)
2. Add n_cb=3, n_cb=4 benchmark arms
3. Measure graph_compute_wall p50 with CGC_STEP_TIMING=1 at each n_cb setting
4. Guard: n_cb > CPU cores - 1 is counterproductive (context switch overhead)

### 5.4 Risk assessment

- Low risk: only env-gated parameter change, no code logic change
- Expected gain: 0-3%. n_cb=2 likely already near-optimal; n_cb=3+ may help if encode is CPU-bound (which §8.79 proved it is - 60ms of 67ms is CPU encoding), but gains bounded by cross-buffer dependency management overhead.
- Risk: larger n_cb increases cross-buffer data hazard complexity (splits across buffer boundaries require extra synchronization).

## 6. Direction 3: Per-step Dispatch Plan Reuse / Block-encode

### 6.1 Design

Decode steady-state: every step has identical graph topology (graphs reused = 127 in llama_perf output). Currently, the graph structure is cached, but the per-op encoder call sequence (set_pipeline + set_buffer x N + dispatch_threadgroups) is re-executed for every step.

Proposed: cache the per-op dispatch plan across steps. For each op in the cached graph, record:
- PSO pointer (already stable)
- buffer binding slots (1-3: src0, src1, dst) - their buffer ID changes per step only when KV position or input token changes
- args struct template (layout fixed; only a few fields change per step)
- threadgroup config (grid changes only when batch size changes - in decode n_tokens=1, grid is fixed)

Per-step, only patch the fields that change:
- KV cache position offset (advances by 1 each step)
- input token id (from sampler)
- any pointer that shifts due to ring buffer wrap

### 6.2 Implementation plan

This is a multi-phase refactor.

Phase A - dispatch plan capture (read-only):
1. Add `ggml_metal_dispatch_plan` struct: array of per-op entries, each containing {pipeline, buffer_bindings[N], args_template, args_size, threadgroup_config}
2. Add `ggml_metal_encoder_capture_plan(enc)` - records the current dispatch sequence into the plan
3. First decode step: run normal encode path, capture plan via the capture hook
4. Subsequent steps: if plan exists and graph topology matches, skip normal encode, instead iterate plan and patch only changed fields

Phase B - block-encode (medium risk):
1. For decode steady-state (n_tokens=1, graph topology stable), detect plan reusability
2. Group consecutive ops in the plan that share the same PSO family (e.g. all attention Q/K/V/O in a layer) into a block
3. Use `MTLBlitCommandEncoder` or `MTLComputeCommandEncoder` batch APIs to encode the block in one call
4. Per-step: patch only KV offset and input token in the block

Phase C - cross-step plan persistence (high risk):
1. Persist the dispatch plan in `ggml_metal_context` - keyed by graph signature (hash of op types + shapes)
2. Detect graph topology change (prefill vs decode, batch size change) - invalidate plan
3. On plan hit: skip graph visitor + tensor visitor + per-op switch entirely, just iterate cached plan

### 6.3 Measurement plan

- graph_compute_wall p50 67ms -> 50-55ms (Phase A: 15-25% reduction from skipping graph visitor + per-op switch)
- Phase B: additional 5-10% from block-encode
- Phase C: additional 5-10% from cross-step persistence
- Combined ceiling: 67ms -> 40-50ms -> decode t/s 14-16

### 6.4 Risk assessment

- High risk: touches three files (ggml-metal-context.m, ggml-metal-ops.cpp, ggml-metal-device.m), ~1000+ LOC
- Graph topology change detection: if plan is applied to a mismatched graph (e.g. after KV shift, expert cache miss causes different expert set), correctness breaks. Must validate plan signature before reuse.
- Args template patching: must identify which args fields change per-step (KV offset, input token, expert indices) vs stable (weight pointers, layer config). Requires per-op handler audit.
- Expert cache interaction: when expert cache miss occurs, the dispatched expert set changes - plan must invalidate on cache miss. This is the biggest correctness risk for the MoE use case.
- llama.cpp upstream may not accept such a large refactor. Per AGENTS.md: "PR represents a long-term commitment". This may be fork-only.

## 7. Combined Gain Projection

| Scenario | graph_compute_wall p50 | t/s | vs baseline |
|---|---|---|---|
| Baseline (§8.79) | 67ms | 12.88 | - |
| Direction 1 Phase A only | 60-64ms | 13.3-13.6 | +3-5% |
| Direction 1 Phase A + B | 55-60ms | 13.7-14.3 | +6-11% |
| Direction 2 (n_cb=4, actual result §8.81) | ~82-84ms/run mean (spike removal) | ~12-12.5 (clean p50 ~13-14) | **+20-25% mean** |
| Direction 3 Phase A | 50-55ms | 14.3-15.0 | +11-16% |
| Direction 3 Phase A+B+C | 40-50ms | 15.0-16.5 | +16-28% |
| All three combined | 38-48ms | 15.5-17.0 | +20-32% |

Conclusion: even with all three directions fully implemented, the ceiling is ~17 t/s - still below the 20 t/s threshold. The three directions close the gap to the §8.77 engine ceiling (13.7 t/s) and partially close the gap to the Swift/turbo engine line (18-20 t/s), but cannot surpass it. The fundamental limit remains: 60ms of CPU-side MTL encoding is irreducible without a different execution model (Swift/Metal-PSO direct dispatch, which is the turbo engine approach).

## 8. Implementation Sequencing Recommendation

Based on risk/reward ratio:

1. **Direction 2 first** (1-2 days work, low risk, establishes whether n_cb=3+ helps at all)
2. **Direction 1 Phase A** (3-5 days, medium risk, expected 3-5% gain - quick win to validate approach)
3. **Direction 1 Phase B** (5-7 days, medium-high risk, additional 5-10% - only if Phase A shows measurable gain)
4. **Direction 3 Phase A** (7-10 days, high risk, 11-16% gain - the largest single contribution)
5. **Direction 3 Phase B+C** (10-15 days, high risk, additional 5-10% - diminishing returns, evaluate after Phase A)

Total estimated work: 26-39 days for full implementation. Practical milestone: after step 2 (Direction 1 Phase A), evaluate whether to continue or hand off to Swift/turbo engine line.

## 9. Success Criteria

Each direction must meet these criteria to proceed:

- Correctness: bit-identical output vs baseline (verified via CGC bit-identical test suite)
- Performance: graph_compute_wall p50 reduction >= 3ms (measurable above noise floor)
- Stability: 1000-step decode run without crash or output divergence
- Resource: no increase in RSS or GPU memory allocation

If a direction fails any criterion, document the finding and proceed to the next direction.

## 10. Fallback and Abort Conditions

- If Direction 1 Phase A shows <3ms gain: skip Phase B, document, proceed to Direction 3
- If Direction 2 shows no gain at n_cb=3+: document, keep n_cb=2 as default
- If Direction 3 Phase A shows correctness issues with expert cache interaction: abort Phase B+C, keep Phase A as env-gated opt-in (CGC_DISPATCH_PLAN=1) with explicit warning on cache miss paths
- If combined gain <10%: declare the three directions insufficient, formally hand off to Swift/turbo engine line

## 11. Relationship to §8.79 and §8.77 Chart Updates

After each direction's measurement, update:
- §8.80 (new section) in LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md with results
- CGC_TPOT_延遲分解_2026-08-17.html: update the graph_compute_wall segment (currently 80.1% of 83.6ms bar) with the new smaller proportion
- Engine ceiling annotation: if combined gain pushes t/s above 13.7, update "§8.77 已部分達成 12.8/13.7" to reflect the new actual value

## 12. Open Questions

- Does M4 Max support MTLArgumentEncoder with the compute kernel argument buffers we need? (Direction 1 Phase B gate)
- Is the dispatch plan signature hash sufficient to detect all graph topology changes, including expert cache miss-induced expert set changes? (Direction 3 correctness gate)
- Does n_cb=3+ introduce cross-buffer dependency hazards that require extra synchronization, negating the parallel encode gain? (Direction 2 gate)

These must be answered in the first phase of each direction's implementation, before committing to the full scope.
