// =============================================================================
// CGC Engine Core (rswaengine) — self-contained linkage layer
// -----------------------------------------------------------------------------
// 本文件是 rswaengine 的引擎核心，实现 cgc_cpp.h 声明的全部 C API，并把
// 统一注入入口（inject_unified_ir_for_role，定义于 unified_ir.cpp）与
// cgc_inject_strategy 真正联动：
//
//   ① cgc_inject_unified_ir_for_role 在构建 cgc_strategy_t 后会回调
//      cgc_inject_strategy（见 unified_ir.cpp），把统一 IR 落地为全局策略；
//   ② cgc_inject_strategy 检测到 metadata 中的 "unified_ir:" 标记后，自动调用
//      cgc_set_kda_replace_mode(true)，使 SDPA(0x10)->KDA(0x11) 重路由激活。
//
// 注意：这是 rswaengine 聚焦的引擎核心子集。重型 opcode kernel（linear /
// attention / quant ...）不在本文件中实现——统一 IR 的注意力计算由
// rswa_manager + window_ortho_kv_compressor 直接提供，并由 Python adapter
// 在 sglang / vllm 侧消费。
// =============================================================================
#include "cgc_cpp.h"
#include "unified_ir.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// -----------------------------------------------------------------------------
// Global state
// -----------------------------------------------------------------------------
static cgc_strategy_t        g_current_strategy;
static bool                  g_strategy_initialized = false;
static cgc_backend_t         g_backend_override    = CGC_BACKEND_AUTO;
static bool                  g_kda_replace_mode    = false;
static std::unordered_map<int64_t, void*> g_kda_states;
static int64_t               g_next_kda_state_id   = 0;

static void cgc_init_default_strategy(void) {
    memset(&g_current_strategy, 0, sizeof(cgc_strategy_t));
    g_current_strategy.backend = CGC_BACKEND_AUTO;
    g_current_strategy.tile_config.tile_m = 128;
    g_current_strategy.tile_config.tile_n = 128;
    g_current_strategy.tile_config.tile_k = 128;
    g_current_strategy.tile_config.attn_block = 128;
    g_current_strategy.tile_config.moe_block = 128;
    g_current_strategy.enable_op_fusion = true;
    g_current_strategy.quantization_mode = 0;
    g_current_strategy.tp_degree = 1;
    g_current_strategy.pp_degree = 1;
    g_current_strategy.num_op_hints = 0;
    g_strategy_initialized = true;
}

// -----------------------------------------------------------------------------
// Backend selection (rswaengine 仅区分 auto/cpu；cuda 由独立 build 提供)
// -----------------------------------------------------------------------------
const char* cgc_get_backend_name(cgc_backend_t backend) {
    switch (backend) {
        case CGC_BACKEND_CPU:  return "cpu";
        case CGC_BACKEND_CUDA: return "cuda";
        case CGC_BACKEND_METAL: return "metal";
        case CGC_BACKEND_AUTO:
        default:               return "auto";
    }
}

bool cgc_set_backend(cgc_backend_t backend) {
    g_backend_override = backend;
    printf("[CGC] backend override -> %s\n", cgc_get_backend_name(backend));
    return true;
}

cgc_backend_t cgc_get_current_backend(void) {
    if (g_backend_override != CGC_BACKEND_AUTO) return g_backend_override;
    return CGC_BACKEND_CPU;
}

// -----------------------------------------------------------------------------
// Strategy injection  + 统一 IR 联动
// -----------------------------------------------------------------------------
cgc_error_t cgc_inject_strategy(const cgc_strategy_t* strategy) {
    if (!strategy) return CGC_ERROR_INVALID_STRATEGY;
    if (!g_strategy_initialized) cgc_init_default_strategy();

    memcpy(&g_current_strategy, strategy, sizeof(cgc_strategy_t));

    printf("[CGC] Strategy injected: backend=%s, fusion=%d, tp=%d, hints=%d\n",
           cgc_get_backend_name(strategy->backend),
           strategy->enable_op_fusion,
           strategy->tp_degree,
           strategy->num_op_hints);

    if (strategy->backend != CGC_BACKEND_AUTO) {
        cgc_set_backend(strategy->backend);
    }

    /* 联动②：统一 IR 注入的 strategy 携带 "unified_ir:" 元数据，
     * 自动激活 KDA replace mode，使 0x10(SDPA) -> 0x11(KDA) 重路由生效。 */
    if (strstr(strategy->metadata, "unified_ir:") != nullptr) {
        cgc_set_kda_replace_mode(true);
    }
    return CGC_OK;
}

cgc_error_t cgc_get_strategy(cgc_strategy_t* strategy) {
    if (!strategy) return CGC_ERROR_INVALID_STRATEGY;
    if (!g_strategy_initialized) cgc_init_default_strategy();
    memcpy(strategy, &g_current_strategy, sizeof(cgc_strategy_t));
    return CGC_OK;
}

cgc_error_t cgc_reset_strategy(void) {
    if (!g_strategy_initialized) cgc_init_default_strategy();
    cgc_init_default_strategy();
    g_kda_replace_mode = false;
    printf("[CGC] Strategy reset to defaults (kda_replace_mode off)\n");
    return CGC_OK;
}

// -----------------------------------------------------------------------------
// KDA replace mode
// -----------------------------------------------------------------------------
cgc_error_t cgc_set_kda_replace_mode(bool enable) {
    g_kda_replace_mode = enable;
    printf("[CGC] KDA Replace Mode: %s\n", enable ? "enabled" : "disabled");
    return CGC_OK;
}

bool cgc_get_kda_replace_mode(void) {
    return g_kda_replace_mode;
}

// -----------------------------------------------------------------------------
// Init / Destroy
// -----------------------------------------------------------------------------
cgc_error_t cgc_init(void) {
    if (!g_strategy_initialized) cgc_init_default_strategy();
    printf("[CGC C++] rswaengine core initialized (backend=%s, kda_replace=%s)\n",
           cgc_get_backend_name(cgc_get_current_backend()),
           g_kda_replace_mode ? "on" : "off");
    return CGC_OK;
}

cgc_error_t cgc_destroy(void) {
    for (auto& p : g_kda_states) free(p.second);
    g_kda_states.clear();
    printf("[CGC C++] rswaengine core destroyed\n");
    return CGC_OK;
}

// -----------------------------------------------------------------------------
// Opcode dispatch (focused subset)
// -----------------------------------------------------------------------------
bool cgc_has_opcode(int opcode) {
    /* rswaengine 核心关注的 opcode：SDPA / KDA；其余交由各 backend adapter。 */
    return opcode == 0x10 || opcode == 0x11 ||
           opcode == 0x12 || opcode == 0x13;
}

cgc_error_t cgc_execute_opcode(
    int opcode,
    const float** inputs, const int64_t* input_dims, const int* input_ndims, int num_inputs,
    float** outputs, int64_t* output_dims, int* output_ndims, int num_outputs,
    const void* params
) {
    int actual = opcode;
    /* 联动③：KDA replace mode 下，SDPA 被重路由到 KDA 路径 */
    if (g_kda_replace_mode && opcode == 0x10) {
        actual = 0x11;
        printf("[CGC C++] KDA Replace: 0x10(SDPA) -> 0x11(KDA)\n");
    }
    printf("[CGC C++] execute opcode 0x%02x\n", actual);

    if (actual == 0x11) {
        /* KDA attention：真实注意力由 rswa_manager / window_ortho_kv_compressor
         * 提供；此处为引擎核心占位，确保重路由链路连通。 */
        if (num_outputs > 0 && outputs[0]) {
            int64_t n = 1;
            for (int i = 0; i < output_ndims[0]; ++i) n *= output_dims[i];
            for (int64_t i = 0; i < n; ++i) outputs[0][i] = 0.0f;
        }
        return CGC_OK;
    }
    (void)inputs; (void)input_dims; (void)input_ndims; (void)num_inputs;
    (void)params; (void)output_dims; (void)output_ndims; (void)num_outputs;
    return CGC_OK;
}

// -----------------------------------------------------------------------------
// KDA state (kda_state_v1 兼容路径的轻量占位；真实 KV 压缩状态由 wokdc 管理)
// -----------------------------------------------------------------------------
cgc_error_t cgc_kda_create_state(void** state, int32_t n_heads, int32_t d_state) {
    size_t sz = (size_t)n_heads * d_state * 2 * sizeof(float) + sizeof(int64_t);
    void* s = malloc(sz);
    if (!s) return CGC_ERROR;
    memset(s, 0, sz);
    int64_t id = g_next_kda_state_id++;
    g_kda_states[id] = s;
    *(int64_t*)s = id;
    *state = s;
    return CGC_OK;
}

cgc_error_t cgc_kda_free_state(void* state) {
    if (!state) return CGC_OK;
    free(state);
    return CGC_OK;
}

void cgc_kda_forward(
    const float* q, const float* k, const float* v,
    const float* g, const float* b,
    void* state,
    float* out,
    const cgc_kda_params_t* params
) {
    (void)q; (void)k; (void)v; (void)g; (void)b; (void)state; (void)out; (void)params;
    /* 占位：真实 KDA forward 走 window_ortho_kv_compressor / TrueOrthoKDA v4。 */
}

// -----------------------------------------------------------------------------
// Hardware Bus Layer / MMAP Zero-Copy
// -----------------------------------------------------------------------------
void* cgc_mmap_file(const char* filepath, size_t* out_size) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[CGC Bus] open failed: %s\n", filepath);
        *out_size = 0; return nullptr;
    }
    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); *out_size = 0; return nullptr; }
    size_t size = (size_t)sb.st_size;
    *out_size = size;
    void* mapped = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { fprintf(stderr, "[CGC Bus] mmap failed\n"); return nullptr; }
    if (mlock(mapped, size) != 0) {
        fprintf(stderr, "[CGC Bus] mlock failed (page faults possible)\n");
    } else {
        printf("[CGC Bus] mlock ok, page-fault delay eliminated\n");
    }
    printf("[CGC Bus] mmap %zu bytes from %s\n", size, filepath);
    return mapped;
}

void cgc_munmap_file(void* ptr, size_t size) {
    if (ptr && ptr != MAP_FAILED) { munlock(ptr, size); munmap(ptr, size); }
}

void cgc_install_vram_interception_hook(void) {
    printf("[CGC Bus] VRAM interception hook: stub (no-op on this build)\n");
}

void* cgc_get_intercepted_kv_cache_ptr(size_t* out_size) {
    *out_size = 0;
    return nullptr;
}
