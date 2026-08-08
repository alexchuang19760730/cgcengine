#pragma once

#include "cgc_cpp.h"
#include "kernels/rswa_manager.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * inject_unified_ir_for_role — 统一注入入口（C/C++ 版）
 * -----------------------------------------------------------------------------
 * 把「R-SWA（Reference/recency 管理）+ 窗口 OrthoKDA（局部正交 KV 压缩）」
 * 以 backend 无关的中间表示（IR：逐层角色标记）描述，再分别适配到各推理
 * 框架（sglang / vllm / cloud_sglang active path）。避免为每个 backend 重写
 * 一遍注意力逻辑。
 *
 * 与既有 cgc_cpp.h 中 cgc_inject_strategy 的关系：
 *   本函数构建 cgc_strategy_t（通过 op_hints + metadata 编码 RSWA/ORTHO 配置）
 *   并就地登记到统一 IR 注册表；真实的 monkey-patch 由各 backend adapter 在
 *   拿到 IR 后执行（sglang/vllm 侧已有 Python 绑定入口，见 ortho_kda_v4_binding.cpp）。
 * ========================================================================== */

typedef enum {
    CGC_INJECT_ROLE_PREFILL      = 0, /* prefill 阶段（非 cuda-graph，可用 R-SWA 全量） */
    CGC_INJECT_ROLE_DECODE       = 1, /* decode 阶段（cuda-graph，需固定 window_size） */
    CGC_INJECT_ROLE_EDGE_RESUME  = 2, /* edge 端从 kda_state_v1 resume decode（M7.5）   */
} cgc_inject_role_t;

typedef enum {
    CGC_BACKEND_ADAPTER_SGLANG        = 0, /* sglang：替换 RadixAttention              */
    CGC_BACKEND_ADAPTER_VLLM          = 1, /* vllm：替换 attention backend            */
    CGC_BACKEND_ADAPTER_CLOUD_SGLANG  = 2, /* cloud_sglang active path：复用 kda_state_v1 */
} cgc_backend_adapter_t;

typedef struct {
    rswa_config_t             rswa;            /* R-SWA 配置                                  */
    cgc_inject_role_t         role;            /* 注入角色                                    */
    cgc_backend_adapter_t     adapter;         /* 目标 backend adapter                        */
    bool                      enable_window_ortho; /* 窗口层是否启用局部正交压缩              */
    bool                      enable_reference_std;/* Reference 走标准注意力（硬约束，恒 true） */
    int32_t                   num_layers;      /* 模型层数（用于构建 IR 层序列）              */
} cgc_unified_ir_config_t;

/* 统一注入：构建 IR + 登记策略 + 选择 adapter。
 * model 为各 backend 的不透明模型句柄（sglang/vllm 对象）；C/C++ 核心只构建并
 * 校验 IR，真实 patch 由 adapter 在获取 IR 后执行。可传 NULL 仅做 IR 构建。 */
cgc_error_t cgc_inject_unified_ir_for_role(
    cgc_inject_role_t          role,
    void*                      model,
    const cgc_unified_ir_config_t* config,
    cgc_backend_adapter_t      adapter
);

/* 仅构建 IR 层序列（不登记），写入调用方分配的长度为 num_layers 的数组。
 * 返回实际写入层数（= num_layers）。供检视 / 单测使用。 */
int cgc_build_unified_ir_layers(
    const cgc_unified_ir_config_t* config,
    int32_t                        num_layers,
    rswa_layer_role_t*             out_layers
);

/* 取已登记到某角色的 IR 摘要（纯文本，调用方负责 free） */
char* cgc_unified_ir_summary(cgc_inject_role_t role);

#ifdef __cplusplus
}
#endif
