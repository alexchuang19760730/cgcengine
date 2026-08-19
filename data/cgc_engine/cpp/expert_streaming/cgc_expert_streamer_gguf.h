#ifndef CGC_EXPERT_STREAMER_GGUF_H
#define CGC_EXPERT_STREAMER_GGUF_H

#include "cgc_expert_streamer.h"
#include "cgc_gguf_lite.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int expert_id;
    char role[CGC_MAX_NAME_LEN];
    int32_t ggml_type;
    int64_t dims[4];
    int n_dims;
    uint64_t offset;
    uint64_t size_bytes;
} cgc_expert_tensor_info_t;

typedef struct {
    int layer_index;
    int experts_per_layer;
    uint64_t expert_stride;
    int hidden_size;
    int moe_intermediate_size;
    char quantization[64];
    char imatrix_file[CGC_MAX_PATH_LEN];
} cgc_layer_gguf_meta_t;

cgc_stream_layout_t cgc_load_stream_layout_from_gguf(const char* gguf_path);

int cgc_find_expert_tensors(const cgc_gguf_lite_ctx_t* ctx,
                             int expert_id,
                             cgc_expert_tensor_info_t* out,
                             int max_out);

cgc_layer_gguf_meta_t cgc_parse_layer_gguf_meta(const cgc_gguf_lite_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif
