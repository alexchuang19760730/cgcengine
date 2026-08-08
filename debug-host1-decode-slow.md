# Debug Session: host1-decode-slow [OPEN]

## Goal
- Quantify why `host1` `DeepSeekV4 tp4ep4` decode throughput remains extremely low after `graph on`.
- Split decode cost into falsifiable phases before changing more business logic.

## Symptoms
- `graph off` single-request long decode is about `2.01 tok/s`.
- `graph on` improved only to about `2.85 tok/s`.
- The latest metadata-reuse attempt did not improve throughput and may regress it.

## Falsifiable Hypotheses
1. `PagedIndexerMetadata.refresh_()` dominates per-token decode time, especially DeepGEMM schedule and topk planning.
2. `update_paged_decode_compressor_data_inplace()` dominates per-token decode time because decode plan generation still runs planner kernels every step.
3. The actual attention forward kernel dominates time, so metadata work is not the primary bottleneck.
4. The slowdown comes from host-side sync / graph-boundary behavior around the new replay path, not from any single planner kernel.
5. The apparent regression is measurement noise from warmup / logging position, and the steady-state decode path is not actually slower.

## Plan
1. Add instrumentation only, with phase-level timing around decode metadata refresh and attention forward.
2. Reproduce on `host1` with the same long-decode `graph on` smoke configuration.
3. Compare per-phase evidence and decide whether to keep or revert the metadata-reuse path.

## Status
- Created debug record.
- Added first-round instrumentation.

## Evidence Round 1
- `.dbg/trae-debug-log-host1-decode-slow.ndjson` captured only hypothesis `D` events from `deepseek_v4_backend.py:forward`.
- Observed sampled decode-attention timings during graph capture warmup with `q_tokens=64`, `seq_len_max=1`, `attn_ms≈409-467ms`, then `≈19ms`.
- `baseline_server.log` shows graph capture failed with `cudaErrorStreamCaptureUnsupported` and then `cudaErrorStreamCaptureInvalidated`.

## Hypothesis Status
- A: INCONCLUSIVE. No metadata timing event was emitted before capture failed.
- B: INCONCLUSIVE. No dedicated indexer/compressor timing evidence yet.
- C: INCONCLUSIVE. Attention timing exists, but it was emitted from inside graph capture warmup and contaminated the run.
- D: CONFIRMED for the current failed run. The instrumentation itself crossed the graph-capture boundary and invalidated capture.
- E: REJECTED for this failed run. The regression was not mere measurement noise; the run was invalidated by capture-unsafe instrumentation.

## Next Step
- Keep instrumentation but make decode-attention reporting capture-safe.
- Re-run the same `graph on` smoke and collect phase timing from a valid run.

## Evidence Round 2
- The capture-safe rerun completed graph capture successfully and reached `ready to roll`.
- Real single-token decode attention events were captured with `q_tokens=1` and `attn_ms≈5.35-5.47ms` for the earliest samples, then `≈16.84-17.93ms`.
- Real decode metadata refresh events were captured with `bs=1`, `seq_len_max=2`, `core_ms≈1.26-1.46ms`, `indexer_ms≈0.009-0.011ms`, `compressor_ms≈0.179-0.198ms`, `total_ms≈1.53-1.75ms`.

## Updated Hypothesis Status
- A: REJECTED as primary bottleneck. Metadata refresh is only about `1.5-1.7ms/token`.
- B: REJECTED as primary bottleneck. Indexer and compressor refresh are both sub-millisecond and tiny compared with total decode cost.
- C: CONFIRMED as the dominant direction. Even one layer-0 decode-attention sample is already multiple times larger than the whole metadata refresh path.
- D: CONFIRMED for round 1 only, then mitigated. Capture-unsafe instrumentation invalidated graph in round 1, but the capture-safe rerun succeeded.
- E: REJECTED. The timing gap is stable across repeated samples, not measurement noise.

## Current Conclusion
- The main bottleneck is not decode metadata refresh.
- The problem has narrowed to the decode-attention / per-layer kernel path, plus any surrounding per-layer scheduling cost, rather than `indexer` or `compressor` planning.

## Evidence Round 3
- Third-round instrumentation split `store_cache_ms` and the host-side wall time around `flash_mla_with_kvcache_sm120()`.
- Real single-token decode samples show `store_cache_ms=0.0ms`, `kernel_ms≈0.15-0.45ms`, but `attn_ms≈5.43-17.93ms`.
- Therefore, the expensive portion is not the KV store path, and the current host-side kernel wall time is not a faithful GPU execution time. The missing cost is likely deferred synchronization / stream completion that is being charged later in the sampled path.

## Updated Hypothesis Status
- `store_cache` as the primary bottleneck: REJECTED.
- `flash_mla_with_kvcache_sm120` launch overhead as the primary bottleneck: REJECTED.
- A deferred GPU sync / completion boundary around decode attention: PROMOTED to primary hypothesis for the next round.

## Next Step
- Add capture-safe CUDA event timing around `flash_mla_with_kvcache_sm120`.
- Remove host-side sync contamination from the sampled payload so we can separate:
  - kernel GPU time
  - host launch time
  - deferred sync / completion time

## Evidence Round 4
- Fourth-round instrumentation captured capture-safe CUDA-event timing for real single-token decode samples in `.dbg/trae-debug-log-host1-decode-slow.ndjson`.
- Early samples showed `kernel_launch_ms≈0.47ms`, `kernel_gpu_ms≈0.45ms`, `sync_wait_ms≈0.007ms`, `attn_ms≈0.53ms`, proving the timing path itself works and can observe a mostly-unsynchronized fast case.
- The steady-state slow samples then showed `kernel_launch_ms≈0.17-0.18ms`, `kernel_gpu_ms≈0.067-0.070ms`, but `sync_wait_ms≈17.72-17.74ms` and `attn_ms≈17.92ms`.
- `dbgprobe6` completed successfully, and the benchmark artifact still reports `decode_tps≈2.856 tok/s`, matching the existing slow graph-on baseline rather than showing a new kernel-side regression.

## Updated Hypothesis Status
- A: REJECTED as primary bottleneck. Long-decode metadata refresh remains only about `0.36-0.49ms/token` once decode is in the representative long-sequence region.
- B: REJECTED as primary bottleneck. Indexer and compressor refresh stay far below the observed decode stall.
- C: REJECTED in its narrow form. The `flash_mla_with_kvcache_sm120` kernel itself is not slow; actual GPU execution is only about `0.07ms` in the slow samples.
- D: CONFIRMED as the primary bottleneck. The dominant cost is the completion boundary after launch, where `sync_wait_ms` absorbs essentially the full `attn_ms`.
- E: REJECTED. The slow path is stable across repeated samples and aligns with the unchanged `decode_tps` baseline.

## Current Conclusion
- The root bottleneck is no longer consistent with "decode metadata rebuild is slow" or "the decode kernel itself is slow."
- The evidence now points to a deferred sync / stream completion boundary around decode attention. In the slow path, almost all sampled time is charged to waiting for earlier GPU work to complete, not to executing `flash_mla_with_kvcache_sm120` itself.
- Therefore the next debugging target should move outward from the kernel body and focus on what prior work is draining into this synchronization point: stream ordering, earlier layer work, or another hidden GPU dependency on the same path.

## Next Step
- Keep the current instrumentation in place.
- Add one more round of phase split around the decode-attention caller boundary so we can identify which upstream GPU work is being retired by the `sync_wait_ms` point.
- Prioritize observability around stream/ordering boundaries before attempting any new optimization patch.

## Instrumentation Round 5
- Added caller-boundary CUDA events in `deepseek_v4_backend.py:forward` for the sampled decode path:
  - `entry_event`: first marker on entering the layer-0 decode caller.
  - `post_store_event`: marker immediately after `store_cache()`.
  - `pre_kernel_event`: marker immediately before `_debug_kernel_start.record()`.
- The sampled payload now also reports:
  - `entry_queue_wait_ms`
  - `store_gpu_wait_ms`
  - `pre_kernel_wait_ms`
  - `store_gpu_ms`
  - `pre_kernel_gpu_ms`
- Interpretation goal:
  - If `entry_queue_wait_ms` is dominant, the backlog is already present before this caller begins issuing new decode work.
  - If `store_gpu_wait_ms` is dominant, `store_cache()` or its downstream writes are the hidden queue source.
  - If `pre_kernel_wait_ms` is dominant, the hidden dependency lives between KV-store completion and the flash-MLA launch boundary.

## Evidence Round 5
- `dbgprobe7` completed successfully with graph capture intact and the benchmark remained at `decode_tps≈2.856`, matching prior graph-on runs.
- Fast decode samples still show the clean path: `entry_queue_wait_ms≈0.007ms`, `store_gpu_wait_ms≈0.003ms`, `pre_kernel_wait_ms≈0.002ms`, `kernel_gpu_ms≈0.45ms`, `attn_ms≈0.55ms`.
- Steady-state slow decode samples now split cleanly as:
  - `entry_queue_wait_ms≈17.62-17.66ms`
  - `store_gpu_wait_ms≈0.003ms`
  - `pre_kernel_wait_ms≈0.002ms`
  - `store_gpu_ms≈0.001ms`
  - `pre_kernel_gpu_ms≈0.000-0.001ms`
  - `kernel_gpu_ms≈0.066-0.069ms`
  - `sync_wait_ms≈0.053-0.058ms`
  - `attn_ms≈17.90-17.94ms`
- This means the dominant stall is already queued before the sampled layer-0 decode caller starts issuing its own work on the stream.

## Updated Hypothesis Status
- A: REJECTED as primary bottleneck. Metadata remains sub-millisecond in the representative long-decode region.
- B: REJECTED as primary bottleneck. `store_cache()` and its downstream GPU tail are negligible.
- C: REJECTED. The flash-MLA kernel and the immediate pre-kernel caller region are both tiny.
- D: CONFIRMED, and now localized more precisely. The deferred completion backlog is upstream of the sampled layer-0 caller boundary, not inside the caller's local setup.
- E: REJECTED. The slowdown is stable and reproducible across the new split.

## Current Conclusion
- The stream backlog is not being created by `store_cache()` or by the narrow caller region between KV-store completion and flash-MLA launch.
- The queue source must therefore live earlier in the decode pipeline, such as prior layer work, cross-layer scheduling/order, or another upstream GPU dependency that retires before layer 0 can begin its local decode-attention work.

## Next Step
- Move the instrumentation boundary outward again, above `deepseek_v4_backend.py:forward`, to identify which earlier decode phase is still occupying the stream before layer 0 enters.
- Prioritize per-layer or caller-of-caller phase markers so we can distinguish:
  - earlier attention/MLP work from previous layers
  - cross-layer pipeline ordering
  - another upstream synchronization point outside the local attention backend

## Evidence Round 6
- Round 6 moved the sampled boundary to `DeepseekV4DecoderLayer.forward` in `deepseek_v4.py`, just before `self.self_attn(...)`, and temporarily suppressed the backend-local `D` sample for the same outer sample.
- A first rerun exposed an operational gap rather than a model issue: the remote sync chain was not pushing `deepseek_v4.py`, so the new instrumentation never reached host1. The benchmark still reproduced the usual `decode_tps≈2.856`, but only old `D` events appeared.
- After adding `deepseek_v4.py` to the host1 patch-sync path, `dbgprobe11` successfully emitted the new `L` events.
- For warmup / larger-token shapes (`hidden_tokens=64` then `56`), the sampled layer-0 pre-attn GPU section was large (`pre_attn_gpu_ms≈1040ms` then `≈56-58ms`) and `self_attn_call_ms` was also much larger, while `entry_queue_wait_ms≈0.007ms`; these are not the representative slow steady-state decode samples.
- For the representative slow steady-state samples, the split is now:
  - `hidden_tokens=64` or `56`
  - `entry_queue_wait_ms≈17.96-18.01ms`
  - `pre_attn_wait_ms≈0.045-0.049ms`
  - `pre_attn_gpu_ms≈0.047-0.051ms`
  - `self_attn_call_ms≈0.608-0.622ms`
- Therefore, the dominant stall is definitively upstream of layer-0 decoder entry. Neither the layer-0 pre-attn section nor the layer-0 self-attention call is responsible for the `~18ms` decode delay.

## Updated Hypothesis Status
- A: REJECTED as primary bottleneck. Metadata remains sub-millisecond.
- B: REJECTED as primary bottleneck. `store_cache()` and local pre-attn work are both negligible in the representative slow samples.
- C: REJECTED. Neither the flash-MLA kernel nor the layer-0 self-attention caller path is the dominant cost.
- D: CONFIRMED with a tighter boundary. The backlog is now localized to work that completes before `DeepseekV4DecoderLayer(layer_id=0).forward` begins issuing its own meaningful GPU work.
- E: REJECTED. The slow path remains stable and reproducible.

## Current Conclusion
- The root cause is upstream of the layer-0 decoder's local attention path.
- The next likely sources are:
  - work from an earlier model phase before layer 0 enters,
  - cross-layer / pipeline ordering,
  - or another decode-time dependency outside the local attention backend and outside layer-0 pre-attn.

## Next Step
- Move the sampled boundary above `DeepseekV4DecoderLayer.forward`, ideally to the model-level loop that invokes layer 0.
- Distinguish whether the queue is already present:
  - before entering the layer loop,
  - between model-level loop entry and layer-0 call,
  - or from some earlier decode component outside the decoder-layer path.

## Evidence Round 7
- Round 7 moved the boundary to `DeepseekV4Model.forward` and then into `_forward_prepare_multi_stream()` so we could separate:
  - model-level layer-loop entry
  - layer-0 first-call body
  - the alt-stream join back to `current_stream`
- `M` samples from `DeepseekV4Model.forward` show that the model-level entry itself is not carrying the slow backlog:
  - representative samples keep `entry_queue_wait_ms≈0.003-0.008ms`
  - `pre_layer0_wait_ms≈0.002-0.003ms`
  - but `first_layer_call_ms` remains very large (`≈222-299ms` in representative decode regions, and much larger in warmup/capture-adjacent samples)
- `S` samples from `_forward_prepare_multi_stream()` localize the stall much more tightly. In the representative slow samples:
  - `has_indexer=false`
  - `has_compressor=false`
  - `q_ready_gpu_ms≈0.045-0.049ms`
  - `kv_tail_gpu_ms≈0.034-0.035ms`
  - `join_wait_ms≈17.80-17.98ms`
- Therefore the dominant decode stall is now localized to the multi-stream join boundary itself, specifically the `current_stream.wait_stream(stream_kv)` path in the decode steady state.
- This also explains why earlier samples could show the stall as "already present before layer-0 local work": the sampled layer-entry waits were observing completion debt that is created by the layer-local alt-stream handoff and then retired at the join boundary.

## Updated Hypothesis Status
- A: REJECTED as primary bottleneck. Metadata remains sub-millisecond in representative long-decode samples.
- B: REJECTED in its previous form. The issue is not `store_cache()` and not a generic upstream backlog outside layer 0.
- C: REJECTED for the kernel body. The flash-MLA kernel remains tiny.
- D: CONFIRMED with the tightest boundary so far. The bottleneck is the decode multi-stream join boundary rather than the attention kernel or metadata path.
- E: REJECTED. The stall is stable and reproducible under repeated sampling.

## Current Conclusion
- The decode slowdown is now localized to the `DeepseekV4` decode multi-stream path.
- In the representative slow samples, only the KV alt stream remains active (`has_indexer=false`, `has_compressor=false`), yet the join back to `current_stream` still costs about `17.8ms/token`.
- The next minimal fix should therefore be an A/B that disables multi-stream overlap on the decode path while preserving the rest of the path unchanged.

## Next Step
- Run a minimal decode-only A/B fix: disable `enable_multi_stream` for decode mode, keep graph/capture enabled, and compare `decode_tps` against the current `~2.85 tok/s` baseline.
- If throughput jumps materially, keep the fix path and then decide whether to:
  - retain decode-only single-stream as the stable solution, or
  - reintroduce overlap selectively with a narrower join strategy.
