#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

// [CGC-METAL-FAIL 2026-09-15] Error query for the owner of any output tensor.
//
// Metal work is asynchronous: a failing command buffer is only observed at the next
// synchronize(), by which point the output buffer holds stale bytes from an earlier
// compute. Any consumer that syncs and then reads a tensor MUST check this first,
// otherwise it will silently treat stale data as a valid result.
// See docs/PREFILL250_DECODE25_WHITEPAPER_*.html (2026-09-15, P0).
bool         ggml_metal_has_error  (ggml_metal_t ctx);
const char * ggml_metal_error_desc (ggml_metal_t ctx);

// CGC: non-blocking count of graph-compute segments that finished on the GPU
// (polled by the sched pipelined dispatch; see CGC_OA_ASYNC)
int ggml_metal_cgc_done(ggml_metal_t ctx);
int ggml_metal_cgc_bufs(ggml_metal_t ctx);
// [CGC 2026-09-15 GPU-side timing] Drain this segment's GPU busy/start/end samples and reset
// them. `out` holds 5 int64: {busy sum, busy union, earliest start, latest end, unsupported}.
// Returns the number of cmd buffers sampled. All zeros when the segment had no completed buffer.
// Gated by CGC_GPU_TIMING (off => the completions never sample, so take() returns 0).
int ggml_metal_cgc_gpu_take(ggml_metal_t ctx, int64_t * out);

// [CGC 2026-09-18 node-level GPU time] The same timestamps WITHOUT the segment-span collapse: one
// record per command buffer, plus the node index range that buffer encoded. `out` holds max_cb
// records of 5 int64: {start_ns, end_ns, first_node, last_node, ok}. Returns the number of records
// written (one per slot 0..n_cb). See the implementation comment for why the range is derivable and
// for what this can and cannot separate -- it gives a duration per contiguous node RANGE, not per
// node. Needs no sampling and no MTLCounterSampleBuffer; the consumer reads the node names through
// ggml_metal_cgc_node_name() at the same instant.
int          ggml_metal_cgc_gpu_take_cb(ggml_metal_t ctx, int64_t * out, int max_cb);
const char * ggml_metal_cgc_node_name   (ggml_metal_t ctx, int node_idx);
// Same snapshot as ggml_metal_cgc_node_name, returning the ggml_op enum (or -1 when there is no such
// node). Needed because the name-keyed table cannot separate ops that share a name prefix.
int          ggml_metal_cgc_node_op     (ggml_metal_t ctx, int node_idx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
