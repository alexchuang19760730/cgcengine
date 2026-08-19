#ifndef CGC_EXPERT_COMPUTE_H
#define CGC_EXPERT_COMPUTE_H

#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* data;
    int64_t shape[2];
    int32_t ggml_type;
    uint64_t offset_in_buffer;
    uint64_t size_bytes;
} cgc_sub_tensor_view_t;

typedef struct {
    int expert_id;
    cgc_sub_tensor_view_t gate;
    cgc_sub_tensor_view_t up;
    cgc_sub_tensor_view_t down;
    void* raw_buffer;
    uint64_t raw_size;
} cgc_expert_weights_view_t;

typedef struct {
    cgc_expert_streamer_t* streamer;
    const cgc_gguf_lite_ctx_t* gguf_ctx;

    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_size;
    uint64_t up_size;
    uint64_t down_size;
    int64_t gate_shape[2];
    int64_t up_shape[2];
    int64_t down_shape[2];
    int32_t ggml_type;
    bool sub_layout_valid;
    cgc_layer_gguf_meta_t meta;
} cgc_expert_compute_bridge_t;

void cgc_compute_bridge_init(cgc_expert_compute_bridge_t* bridge,
                               cgc_expert_streamer_t* streamer,
                               const cgc_gguf_lite_ctx_t* gguf_ctx);

int cgc_compute_bridge_load_weights(cgc_expert_compute_bridge_t* bridge,
                                     const int* expert_ids,
                                     int count,
                                     cgc_expert_weights_view_t* out_views);

int cgc_compute_bridge_get_tensor_info(cgc_expert_compute_bridge_t* bridge,
                                        int expert_id,
                                        cgc_expert_tensor_info_t* out,
                                        int max_out);

cgc_layer_gguf_meta_t cgc_compute_bridge_get_meta(cgc_expert_compute_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif
