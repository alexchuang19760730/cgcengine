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

cgc_error_t cgc_set_kda_replace_mode(bool enable);
bool cgc_get_kda_replace_mode(void);

const char* cgc_get_backend_name(cgc_backend_t backend);
bool cgc_set_backend(cgc_backend_t backend);
cgc_backend_t cgc_get_current_backend(void);

bool cgc_has_opcode(int opcode);

typedef struct {
    int32_t n_heads;
    int32_t d_state;
    void* state_buffer;
} cgc_kda_state_t;

typedef struct {
    int32_t seq_len;
    int32_t dim;
    int32_t n_heads;
    int32_t n_group;
    int32_t d_state;
    float scale;
    bool is_first_chunk;
    int64_t state_id;
} cgc_kda_params_t;

cgc_error_t cgc_kda_create_state(void** state, int32_t n_heads, int32_t d_state);
cgc_error_t cgc_kda_free_state(void* state);

void cgc_kda_forward(
    const float* q, const float* k, const float* v,
    const float* g, const float* b,
    void* state,
    float* out,
    const cgc_kda_params_t* params
);

// -----------------------------------------------------------------------------
// Hardware Bus Layer / MMAP Zero-Copy API
// -----------------------------------------------------------------------------
void* cgc_mmap_file(const char* filepath, size_t* out_size);
void cgc_munmap_file(void* ptr, size_t size);

// Custom Backend VRAM Interception for PCIe Direct Write
void cgc_install_vram_interception_hook(void);
void* cgc_get_intercepted_kv_cache_ptr(size_t* out_size);

/* -----------------------------------------------------------------------------
 * Unified IR linkage
 * -----------------------------------------------------------------------------
 * 把统一注入入口（inject_unified_ir_for_role）通过统一 IR 头文件一并暴露给引擎
 * 使用者。unified_ir.h 自身会 include 本文件（取 cgc_strategy_t / cgc_error_t），
 * 此处放在 extern "C" 闭合之前，使 cgc_unified_ir_config_t 等类型在本文件作用域
 * 内可见，且无重复声明。
 *
 * 双向联动关系（由 cgc_cpp.cpp / unified_ir.cpp 落地）：
 *   - cgc_inject_unified_ir_for_role 构建 cgc_strategy_t 并回调 cgc_inject_strategy；
 *   - cgc_inject_strategy 检测到 unified_ir 元数据后自动启用 cgc_set_kda_replace_mode，
 *     使 0x10(SDPA) -> 0x11(KDA) 重路由生效。
 * 二者由此真正联动，而非各自为政。
 * -------------------------------------------------------------------------- */
#include "unified_ir.h"

#ifdef __cplusplus
}
#endif