#include "cgc_expert_compute.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void compute_sub_tensor_layout(cgc_expert_compute_bridge_t* bridge) {
    if (!bridge || !bridge->gguf_ctx) return;

    cgc_expert_tensor_info_t tensors[32];
    int count = cgc_find_expert_tensors(bridge->gguf_ctx, 0, tensors, 32);

    for (int i = 0; i < count; i++) {
        cgc_expert_tensor_info_t* t = &tensors[i];
        if (strcmp(t->role, "gate") == 0) {
            bridge->gate_offset = t->offset;
            bridge->gate_size = t->size_bytes;
            bridge->ggml_type = t->ggml_type;
            if (t->n_dims >= 2) {
                bridge->gate_shape[0] = t->dims[0];
                bridge->gate_shape[1] = t->dims[1];
            }
        } else if (strcmp(t->role, "up") == 0) {
            bridge->up_offset = t->offset;
            bridge->up_size = t->size_bytes;
            if (t->n_dims >= 2) {
                bridge->up_shape[0] = t->dims[0];
                bridge->up_shape[1] = t->dims[1];
            }
        } else if (strcmp(t->role, "down") == 0) {
            bridge->down_offset = t->offset;
            bridge->down_size = t->size_bytes;
            if (t->n_dims >= 2) {
                bridge->down_shape[0] = t->dims[0];
                bridge->down_shape[1] = t->dims[1];
            }
        }
    }

    bridge->sub_layout_valid = (bridge->gate_size > 0 && bridge->up_size > 0 && bridge->down_size > 0);
}

void cgc_compute_bridge_init(cgc_expert_compute_bridge_t* bridge,
                               cgc_expert_streamer_t* streamer,
                               const cgc_gguf_lite_ctx_t* gguf_ctx) {
    if (!bridge) return;
    memset(bridge, 0, sizeof(*bridge));

    bridge->streamer = streamer;
    bridge->gguf_ctx = gguf_ctx;

    compute_sub_tensor_layout(bridge);

    if (gguf_ctx) {
        bridge->meta = cgc_parse_layer_gguf_meta(gguf_ctx);
    }

    fprintf(stderr, "[ComputeBridge] initialized: gate=%llu up=%llu down=%llu type=%d\n",
            (unsigned long long)bridge->gate_size,
            (unsigned long long)bridge->up_size,
            (unsigned long long)bridge->down_size,
            bridge->ggml_type);
}

int cgc_compute_bridge_load_weights(cgc_expert_compute_bridge_t* bridge,
                                     const int* expert_ids,
                                     int count,
                                     cgc_expert_weights_view_t* out_views) {
    if (!bridge || !bridge->streamer || !expert_ids || !out_views || count <= 0) return 0;

    cgc_cache_access_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.owner_phase = CGC_CACHE_SLOT_PREFILL_TRANSIENT;
    ctx.control_plane = CGC_CACHE_CONTROL_PREFILL;

    cgc_cache_result_t result = cgc_expert_streamer_load_experts(bridge->streamer, expert_ids, count, &ctx);

    for (int i = 0; i < count && i < (int)result.count; i++) {
        cgc_expert_weights_view_t* view = &out_views[i];
        memset(view, 0, sizeof(*view));

        view->expert_id = expert_ids[i];
        view->raw_buffer = result.buffers[i];
        view->raw_size = result.sizes[i];

        if (view->raw_buffer && bridge->sub_layout_valid) {
            view->gate.data = (char*)view->raw_buffer + bridge->gate_offset;
            view->gate.shape[0] = bridge->gate_shape[0];
            view->gate.shape[1] = bridge->gate_shape[1];
            view->gate.ggml_type = bridge->ggml_type;
            view->gate.offset_in_buffer = bridge->gate_offset;
            view->gate.size_bytes = bridge->gate_size;

            view->up.data = (char*)view->raw_buffer + bridge->up_offset;
            view->up.shape[0] = bridge->up_shape[0];
            view->up.shape[1] = bridge->up_shape[1];
            view->up.ggml_type = bridge->ggml_type;
            view->up.offset_in_buffer = bridge->up_offset;
            view->up.size_bytes = bridge->up_size;

            view->down.data = (char*)view->raw_buffer + bridge->down_offset;
            view->down.shape[0] = bridge->down_shape[0];
            view->down.shape[1] = bridge->down_shape[1];
            view->down.ggml_type = bridge->ggml_type;
            view->down.offset_in_buffer = bridge->down_offset;
            view->down.size_bytes = bridge->down_size;
        }
    }

    return count;
}

int cgc_compute_bridge_get_tensor_info(cgc_expert_compute_bridge_t* bridge,
                                        int expert_id,
                                        cgc_expert_tensor_info_t* out,
                                        int max_out) {
    if (!bridge || !bridge->gguf_ctx) return 0;
    return cgc_find_expert_tensors(bridge->gguf_ctx, expert_id, out, max_out);
}

cgc_layer_gguf_meta_t cgc_compute_bridge_get_meta(cgc_expert_compute_bridge_t* bridge) {
    if (!bridge) {
        cgc_layer_gguf_meta_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return bridge->meta;
}
