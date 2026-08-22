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

// CGC: non-blocking count of graph-compute segments that finished on the GPU
// (polled by the sched pipelined dispatch; see CGC_OA_ASYNC)
int ggml_metal_cgc_done(ggml_metal_t ctx);

// CGC: block until at least `target` segments finished, woken by the completion handler's
// semaphore instead of a sched_yield spin (removes the poll latency from the pipelined
// dispatch). Returns the current cgc_done count.
int ggml_metal_wait_cgc_done(ggml_metal_t ctx, int target);

// CGC: number of command buffers created per graph-compute (n_cb chunk buffers + 1 main).
// cgc_done now counts EVERY command buffer's completion (not just the main one), so the sched
// must multiply a segment's completion target by this to wait for the whole segment (the top-k
// node lives in the LAST chunk buffer, which the main buffer does NOT cover).
int ggml_metal_cgc_bufs(ggml_metal_t ctx);

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
