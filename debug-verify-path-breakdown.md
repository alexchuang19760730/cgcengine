# Debug Session: verify-path-breakdown
- **Status**: [OPEN]
- **Issue**: Determine whether the remaining decode headroom on current `dev` is limited by verify FFN/expert gather, trunk cold/file reads, GPU wait/fill/submit, or MTP head cost.
- **Debug Server**: Pending
- **Log File**: Pending

## Reproduction Steps
1. Use production shell [scripts/run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh) with production GGUF on the current `dev` branch.
2. Focus on long prompt steady MTP runs, especially denseIQ4X production path.
3. Collect per-step evidence for verify wall time, trunk cold/requests/file reads, and GPU wait/fill/submit.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Remaining headroom is mostly limited by verify FFN / expert gather, so verify wall time dominates the per-step profile. | High | Medium | Pending |
| B | Trunk-layer cold requests and file reads still materially slow verify, despite draft-layer `LAYER_CAPS` already being optimized. | High | Medium | Pending |
| C | GPU `wait` remains the dominant bucket while `fill` and `submit` stay secondary, implying a bandwidth/compute floor rather than a shell-level policy problem. | Medium | Medium | Pending |
| D | The visible ceiling is actually driven by MTP head or another non-verify segment, so verify is not the largest contributor once measured directly. | Medium | Medium | Pending |

## Log Evidence
- Debug server session started with `.dbg/verify-path-breakdown.env` and `.dbg/trae-debug-log-verify-path-breakdown.ndjson`.
- Added instrumentation in [src/llama.cpp/src/llama-context.cpp](file:///Users/alexchuang/Documents/flashkv0516/src/llama.cpp/src/llama-context.cpp) to report post-`graph_compute` verify-batch timings and cache deltas when `CGC_VERIFY_PATH_DEBUG=1`.
- Production reproduction command:
  `CGC_VERIFY_PATH_DEBUG=1 LLAMA_EXPERT_CACHE_STEP_DBG=1 ./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady`
- Top-level production result (`/tmp/verify_path_breakdown2.log`): `decoded 1101 tokens in 44.659 seconds, speed 24.653 t/s`, `accept 99.187%`, `hit 88.9%`, `n_accept 732`.
- Verify batch NDJSON evidence (`.dbg/trae-debug-log-verify-path-breakdown.ndjson`):
  - Total captured verify batches: `512`
  - `graph_us` mean `119313.9`, median `95570.5`, p95 `173057`
  - By verify batch size:
    - `n_tokens=8`: median `170909.5 us`, mean `180041.3 us`
    - `n_tokens=3`: median `92264.0 us`, mean `92451.9 us`
  - Early vs late:
    - first 32 verify batches: median `171865.5 us`, mean `225111.0 us`, file-reads nonzero `2/32`
    - last 128 verify batches: median `97283.0 us`, mean `97866.6 us`, file-reads nonzero `0/128`
  - Late steady verify batches (`n_tokens=3`) show `requests_delta=0`, `file_reads_delta=0`, but still `fast_cold_ratio` median `0.566`.
- Stderr evidence (`/tmp/verify_path_breakdown2.err`):
  - Final segment timing line: `CGC-SEG: wait 2612.2 cb 101.6 submit 86.9 us (22080)`
  - Final cache stats: `runtime requests=25401 hits=22591 misses=2810 (hit rate 88.9%) file_reads=6762 pread_usec=11261040 fill_batch_usec=20942`
  - Final MTP fast-path split: `verify: calls=20319 union=723528 cold=427360 (59.1%)`, `draft: calls=738 union=5904 cold=0 (0.0%)`
  - STEPDBG checkpoints confirm verify fast-path cold persists deep into steady state:
    - step 0: `fastuni=64 fastcold=41 ensure_req=6849 ensure_hit=4040`
    - step 160: `fastuni=385800 fastcold=236870 ensure_req=16737 ensure_hit=13927`
    - step 360: `fastuni=576200 fastcold=345467 ensure_req=21537 ensure_hit=18727`
- Decode breakdown script (`scripts/diag/decode_breakdown.py`) over the same stderr shows `wait` overwhelmingly dominates `cb` and `submit`; exact wall-time reconciliation remains approximate because `prefill_hooks` is estimated, but the ranking is stable: `wait >> cb > submit`.
- Metal-side verify kernel census (`CGC_VERIFY_MMV_DBG=1`, `/tmp/verify_kernel_probe.err`):
  - Captured `512` routed MoE FFN `mul_mat_id` dispatches.
  - Path split: `MV=512`, `MM=0` for all captured verify-side routed FFN nodes.
  - Dominant repeating shape per layer:
    - `ffn_moe_gate-*`: `type=iq2_s`, `nei1=2`, `nei0=8`, `ne01=512`, `ne02=71`, `ne00=2048`
    - `ffn_moe_up-*`:   `type=iq2_s`, `nei1=2`, `nei0=8`, `ne01=512`, `ne02=71`, `ne00=2048`
    - `ffn_moe_down-*`: mostly `type=iq3_s`, `nei1=2`, `nei0=8`, `ne01=2048`, `ne02=71`, `ne00=512`
  - Probe run still stays on the same production behavior: `26.143 t/s`, `accept 99.187%`, `hit 88.9%`.
- Verify fuse-hit audit (`N30CACHE_MMV_FUSE=1 CGC_MMV_FUSE_DBG=2`, shortened steady run `-n 96`):
  - Production-shell result: `26.353 t/s`, `accept 96.970%`, `hit 83.8%`.
  - `ggml_metal_op_mul_mat_id_glu_fused` dispatched `7144` fused `gate+up+glu` kernels during the run.
  - Coverage: fused routed verify nodes hit `38` layers (`ffn_moe_gate-1 .. ffn_moe_gate-38`).
  - `ne21` distribution from the fused dispatch log: `{2: 76, 3: 1254, 4: 38, 8: 5776}`.
  - No `fail:` or `REJECT:` lines were emitted from `ggml_metal_op_can_fuse_mmv_glu` during this audit window.
- Verify per-op chunk timing (`CGC_VERIFY_OP_TIMING=1`, shortened steady run `-n 32`):
  - This probe forces scheduler observation around verify `topk + gate/up/down` and logs per-chunk wall time to stderr as `CGC-VERIFY-OP`.
  - Production-shell result under probe: `13.634 t/s`, `accept 91.667%`, `hit 83.3%` (diagnostic overhead expected; used for ranking, not throughput acceptance).
  - Metal-only routed verify chunk stats (`backend=MTL0`, `n=1331/1332`):
    - `ffn_moe_gate`: mean `801.7 us`, median `756 us`, p95 `1151 us`
    - `ffn_moe_up`:   mean `773.9 us`, median `749 us`, p95 `968 us`
    - `ffn_moe_down`: mean `1048.7 us`, median `1019 us`, p95 `1292 us`
  - Important caveat: the `down` chunk still begins at `shared_expert_gate_sigmoid-*` in the current execution order, so it is a conservative upper bound on routed-down cost rather than a pure isolated kernel time.
- Refined post-swiglu / batching probe (`CGC_VERIFY_OP_TIMING=1` with extra boundaries, shortened steady run `-n 32`):
  - Production-shell result under probe: `12.871 t/s`, `accept 91.667%`, `hit 83.3%` (again, diagnostic-only overhead).
  - Routed verify `gate/up` remain single-node chunks on `MTL0`:
    - `gate`: mean `809.5 us`, median `767 us`, `span_mean=1.00`
    - `up`:   mean `785.2 us`, median `761 us`, `span_mean=1.00`
  - Routed verify `down` remains a multi-node post-swiglu micro-subgraph:
    - `down`: mean `1057.4 us`, median `1019 us`, p95 `1287 us`, `span_mean=4.80`, `span_p95=6`
    - dominant variants:
      - `span=6`, start=`conv_states-*`, end=`ffn_moe_down-*`: `933` samples, mean `1046.7 us`
      - `span=2`, start=`ffn_moe_weights_sum-*`, end=`ffn_moe_down-*`: `398` samples, mean `1015.9 us`
  - By token count:
    - `(ntok=8, span=6)`: `877` samples, mean `1056.5 us`, median `1028 us`
    - `(ntok=8, span=2)`: `375` samples, mean `1023.8 us`, median `998 us`
  - Attempting to observe `shared_expert_gate_sigmoid-*` as its own boundary did **not** produce a separate chunk class; the scheduler still coalesced the post-swiglu/down region so the measured chunk starts at `conv_states-*` or `ffn_moe_weights_sum-*`. This is direct evidence that the current ceiling is tied to dispatch granularity over a small post-swiglu subgraph, not just a single routed-down kernel in isolation.
- Down fusion / batching audit (`N30CACHE_MMV_FUSE=1 CGC_MMV_FUSE_DBG=2 CGC_MMV_DOWN_DBG=2`, shortened steady run `-n 64`):
  - Production-shell result under audit: `25.933 t/s`, `accept 95.652%`, `hit 83.6%`.
  - The fused `gate+up+glu` path can already locate a stable routed-`down` consumer for the same ids tensor:
    - main path: `gate/up=iq2_s -> down=iq3_s`, `6408` candidate hits
    - late-layer side path: `gate/up=iq2_s -> down=iq4_xs`, layers `34` and `38`, `356` candidate hits
  - Candidate topology is stable and shallow:
    - dominant gap `5`: `ffn_moe_swiglu-* -> shared_expert_gate_sigmoid-* -> conv_states-* -> ffn_moe_weights_sum-* -> ffn_moe_down-*`
    - secondary gap `4`: `ffn_moe_swiglu-* -> shared_expert_gate_sigmoid-* -> ffn_moe_weights_sum-* -> ffn_moe_down-*`
  - All sampled candidates preserved the same routed ids (`ids_same=1`) and directly consume the fused GLU output (`y_same=1`), so the blocker is no longer graph ambiguity. The remaining work is implementation work: either batch/coalesce that `gap=4/5` post-swiglu strip, or add a real fused-down kernel path for `iq2_s -> iq3_s` (and optionally the `iq4_xs` late-layer side path).
- Implemented cut: main-path early down coalescing (`CGC_GLU_FUSED_DOWN=1`):
  - Scope is intentionally narrow:
    - only `gate/up=iq2_s -> down=iq3_s`
    - late-layer `iq4_xs` (`ffn_moe_down-34`, `ffn_moe_down-38`) stays on stock fallback
    - no revisit of the old standalone `gate+up` experiment
  - Implementation:
    - after a successful fused `gate+up+glu` dispatch, host immediately dispatches the matched `ffn_moe_down-*` `MUL_MAT_ID` from the same encode site and marks it consumed, collapsing the observed `gap=4/5` strip without pretending this is a single-shader fused-down kernel.
    - guarded by:
      - same ids tensor as the fused pair
      - `down->src[1] == fused_glu_output`
      - down weight type exactly `iq3_s`
      - contiguous `F32` down dst
      - galloc overlap rejection across in-between nodes before early dispatch
  - Runtime evidence (`CGC_MMV_DOWN_DBG=2`, shortened steady run `-n 32`):
    - `6012` main-path candidates
    - `6012` early dispatches
    - `0` overlap rejects
  - Production-shell A/B (`N30CACHE_MMV_FUSE=1`, steady `-n 64`):
    - `N30CACHE_GLU_FUSED_DOWN=0`: `26.062 t/s`, `hit 83.6%`, `accept 95.652%`
    - `N30CACHE_GLU_FUSED_DOWN=1`: `26.507 t/s`, `hit 84.9%`, `accept 97.727%`
    - delta: `+0.445 t/s` on this shortened steady window, with no observed `0000` regression in output.

## Verification Conclusion
- A: **Confirmed.** Remaining headroom is primarily limited by verify-side compute/gather time. Late steady verify batches still take about `92-98 ms` even when `requests_delta=0` and `file_reads_delta=0`.
- B: **Partially confirmed.** Trunk cold/file reads matter at the beginning of the run, but they are not the steady-state ceiling. Early verify batches show read activity and much higher `graph_us`; late steady batches have zero new reads yet retain a high verify cost.
- C: **Confirmed.** GPU `wait` is the dominant bucket by a wide margin (`CGC-SEG wait 2612.2 us` vs `cb 101.6 us` vs `submit 86.9 us` per hook at the end of the run), so the current ceiling is not explained by hook fill or submit overhead.
- D: **Rejected as the primary bottleneck.** MTP head may still contribute, but the evidence does not support it as the main remaining limiter on this branch. The dominant residual cost sits on verify-side work after the system has already stopped reading new expert weights.
- E: **Confirmed.** The verify-side routed FFN path is dominated by many small stock `MV` `mul_mat_id` launches, not the `MM` path. This makes the current ceiling look much more like a kernel/dispatched-work granularity problem than a host-side cache-management problem.
- F: **Confirmed.** Under `CGC_MMV_FUSE=1`, verify does hit the existing fused `gate+up+glu` path heavily, so the current bottleneck is not "verify misses the old fuse." Reopening the old gate+up fuse hypothesis is low-value relative to `down` / post-swiglu work or dispatch batching.
- G: **Directional result.** In the current diagnostic split, `down`-ending chunks are clearly larger than `gate` and `up` (`~1.05 ms` vs `~0.80 ms` / `~0.77 ms` mean on `MTL0`). Because the `down` chunk still includes the immediately preceding shared-expert gate signal node, treat this as an upper bound, but it still points the next optimization round toward routed-down or batching that whole post-swiglu region.
- H: **Confirmed.** The post-swiglu/down region is not behaving like a single clean node boundary. On `MTL0` it is repeatedly executed as short `span=2` or `span=6` chunks ending at `ffn_moe_down-*`, with the dominant variant starting at `conv_states-*`. That makes dispatch batching / chunk coalescing the highest-value next lever, with routed-down still the best local kernel candidate inside that region.
- I: **Confirmed.** The routed-down target is now concrete rather than speculative: the actual verify path repeatedly presents `gate/up=iq2_s -> down=iq3_s` candidates with the same ids and a fixed `gap=4/5`. This means the next optimization can focus directly on that post-swiglu strip without reopening the old gate+up question.
- J: **Confirmed.** The main-path early-down coalescing cut is live and profitable: on the current shortened steady production A/B it improves throughput (`26.062 -> 26.507 t/s`) while preserving acceptance/hit behavior and showing no `0000` regression. This is now the correct base to iterate on, instead of reopening the old gate+up-only branch.
