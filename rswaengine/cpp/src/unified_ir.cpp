#include "unified_ir.h"
#include "kernels/rswa_manager.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace {

struct ir_record {
    cgc_unified_ir_config_t config;
    cgc_strategy_t          strategy;
};

static std::unordered_map<int, ir_record> g_ir_registry;

const char* adapter_name(cgc_backend_adapter_t a) {
    switch (a) {
        case CGC_BACKEND_ADAPTER_SGLANG:       return "sglang";
        case CGC_BACKEND_ADAPTER_VLLM:         return "vllm";
        case CGC_BACKEND_ADAPTER_CLOUD_SGLANG: return "cloud_sglang(M7.5)";
        default:                                return "unknown";
    }
}

const char* role_name(cgc_inject_role_t r) {
    switch (r) {
        case CGC_INJECT_ROLE_PREFILL:     return "prefill";
        case CGC_INJECT_ROLE_DECODE:      return "decode";
        case CGC_INJECT_ROLE_EDGE_RESUME: return "edge_resume";
        default:                          return "unknown";
    }
}

} // namespace

int cgc_build_unified_ir_layers(const cgc_unified_ir_config_t* config,
                                int32_t num_layers,
                                rswa_layer_role_t* out_layers) {
    if (!config || !out_layers || num_layers <= 0) return 0;
    const int hyb = config->rswa.hybrid_every;
    const bool en_ortho = (config->enable_window_ortho != 0);
    for (int32_t i = 0; i < num_layers; ++i) {
        /* 每层窗口侧角色；Reference 区恒为 RSWA_LAYER_REFERENCE_STD（硬约束） */
        bool ortho = en_ortho && (hyb == 0 || (i % hyb) == 0);
        out_layers[i] = ortho ? RSWA_LAYER_WINDOW_ORTHO : RSWA_LAYER_WINDOW_STD;
    }
    return num_layers;
}

cgc_error_t cgc_inject_unified_ir_for_role(cgc_inject_role_t role,
                                          void* model,
                                          const cgc_unified_ir_config_t* config,
                                          cgc_backend_adapter_t adapter) {
    if (!config) return CGC_ERROR_INVALID_STRATEGY;

    /* 1) 构建逐层 IR（Reference 硬约束：恒标准注意力；窗口按混合策略） */
    std::vector<rswa_layer_role_t> layers(config->num_layers);
    cgc_build_unified_ir_layers(config, config->num_layers, layers.data());

    int ortho_layers = 0, std_layers = 0;
    for (int32_t i = 0; i < config->num_layers; ++i) {
        if (layers[i] == RSWA_LAYER_WINDOW_ORTHO) ++ortho_layers; else ++std_layers;
    }

    /* 2) 编码为 cgc_strategy_t（backend 由 HW 决定，adapter 写入 metadata） */
    cgc_strategy_t strat;
    memset(&strat, 0, sizeof(strat));
    strat.backend = CGC_BACKEND_AUTO;
    strat.enable_op_fusion = true;
    strat.tile_config.attn_block = config->rswa.window_size > 0
                                       ? config->rswa.window_size : 128;
    strat.num_op_hints = 0;
    snprintf(strat.metadata, sizeof(strat.metadata),
             "unified_ir:role=%s;adapter=%s;num_layers=%d;ref_std=forced;"
             "window_ortho=%s;ortho_base_dim=%d;window_size=%d;reference_len=%d;"
             "hybrid_every=%d;kda_state_v1=%s",
             role_name(role), adapter_name(adapter),
             (int)config->num_layers,
             config->enable_window_ortho ? "on" : "off",
             (int)config->rswa.ortho_base_dim, (int)config->rswa.window_size,
             (int)config->rswa.reference_len, (int)config->rswa.hybrid_every,
             adapter == CGC_BACKEND_ADAPTER_CLOUD_SGLANG ? "on" : "off");

    /* 3) 登记到统一 IR 注册表 */
    ir_record rec;
    rec.config = *config;
    rec.strategy = strat;
    g_ir_registry[(int)role] = rec;

    /* 4) 联动①：把构建好的 cgc_strategy_t 真正落地到引擎核心的全局策略
     *     （cgc_inject_strategy 会进一步在检测到 unified_ir 元数据时自动启用
     *      KDA replace mode，形成 inject_unified_ir_for_role <-> cgc_inject_strategy
     *      的双向联动）。 */
    cgc_inject_strategy(&strat);

    /* 5) 真实 monkey-patch 由各 backend adapter 在拿到 IR 后执行
     *    （sglang/vllm 侧已有 Python 绑定入口，见 ortho_kda_v4_binding.cpp；
     *     model 为不透明句柄，C/C++ 核心只构建并校验 IR）。 */
    (void)model;

    printf("[CGC UnifiedIR] injected role=%s adapter=%s layers=%d "
           "(window_ortho=%d, window_std=%d) ref_std=forced(always)\n",
           role_name(role), adapter_name(adapter),
           (int)config->num_layers, ortho_layers, std_layers);
    printf("[CGC UnifiedIR] strategy.metadata = %s\n", strat.metadata);
    return CGC_OK;
}

char* cgc_unified_ir_summary(cgc_inject_role_t role) {
    auto it = g_ir_registry.find((int)role);
    if (it == g_ir_registry.end()) {
        char* s = (char*)malloc(64);
        snprintf(s, 64, "no IR registered for role=%d", (int)role);
        return s;
    }
    const ir_record& rec = it->second;
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "role=%s adapter=%s num_layers=%d ref_len=%d win=%d ortho_base=%d "
             "hybrid_every=%d window_ortho=%s",
             role_name(role), adapter_name(rec.config.adapter),
             (int)rec.config.num_layers, (int)rec.config.rswa.reference_len,
             (int)rec.config.rswa.window_size, (int)rec.config.rswa.ortho_base_dim,
             (int)rec.config.rswa.hybrid_every,
             rec.config.enable_window_ortho ? "on" : "off");
    char* out = (char*)malloc(strlen(buf) + 1);
    strcpy(out, buf);
    return out;
}
