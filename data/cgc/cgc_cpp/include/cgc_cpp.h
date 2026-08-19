#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum {
    CGC_OK = 0,
    CGC_ERROR = 1,
    CGC_ERROR_NOT_SUPPORTED = 2,
    CGC_ERROR_INVALID_STRATEGY = 3,
} cgc_error_t;

typedef enum {
    CGC_BACKEND_AUTO = 0,
    CGC_BACKEND_CPU = 1,
    CGC_BACKEND_CUDA = 2,
    CGC_BACKEND_METAL = 3,
} cgc_backend_t;

typedef enum {
    CGC_OP_HINT_NONE = 0,
    CGC_OP_HINT_FLASH_ATTENTION = 1,
    CGC_OP_HINT_MOE_ROUTING = 2,
    CGC_OP_HINT_TENSOR_PARALLEL = 3,
    CGC_OP_HINT_VLM_CROSS_ATTENTION = 4,
} cgc_op_hint_t;

typedef struct {
    int32_t tile_m;
    int32_t tile_n;
    int32_t tile_k;
    int32_t attn_block;
    int32_t moe_block;
} cgc_tile_config_t;

typedef struct {
    cgc_backend_t backend;
    cgc_tile_config_t tile_config;
    bool enable_op_fusion;
    int32_t quantization_mode;
    int32_t tp_degree;
    int32_t pp_degree;
    int32_t num_op_hints;
    cgc_op_hint_t op_hints[16];
    char fusion_regions[256];
    char metadata[512];
} cgc_strategy_t;

cgc_error_t cgc_execute_opcode(
    int opcode,
    const float** inputs, const int64_t* input_dims, const int* input_ndims, int num_inputs,
    float** outputs, int64_t* output_dims, int* output_ndims, int num_outputs,
    const void* params
);

bool cgc_has_opcode(int opcode);

cgc_error_t cgc_init(void);
cgc_error_t cgc_destroy(void);

cgc_error_t cgc_inject_strategy(const cgc_strategy_t* strategy);
cgc_error_t cgc_get_strategy(cgc_strategy_t* strategy);
cgc_error_t cgc_reset_strategy(void);

const char* cgc_get_backend_name(cgc_backend_t backend);
bool cgc_set_backend(cgc_backend_t backend);
cgc_backend_t cgc_get_current_backend(void);

bool cgc_has_opcode(int opcode);

// -----------------------------------------------------------------------------
// Hardware Bus Layer / MMAP Zero-Copy API
// -----------------------------------------------------------------------------
void* cgc_mmap_file(const char* filepath, size_t* out_size);
void cgc_munmap_file(void* ptr, size_t size);

// Custom Backend VRAM Interception for PCIe Direct Write
void cgc_install_vram_interception_hook(void);
void* cgc_get_intercepted_kv_cache_ptr(size_t* out_size);

#ifdef __cplusplus
}
#endif