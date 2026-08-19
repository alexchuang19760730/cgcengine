// expert_compute_torch.cpp — 上层集成: ExpertWeightsView → torch → grouped_gemm_silu_bf16_forward
//
// 编译条件: 需要 PyTorch (CGC_HAS_PYTORCH) + cgc_moe_engine (CGC_HAS_MOE_ENGINE)
// CMakeLists.txt 中通过 option 控制:
//   option(CGC_EXPERT_STREAMING_WITH_PYTORCH "Enable PyTorch integration" OFF)
//
// 数据流:
//   ExpertWeightsView (raw buffer, 可能是 IQ3_M 量化)
//   → viewsToGroupedWeights (反量化到 BF16, torch::from_blob)
//   → grouped_gemm_silu_bf16_forward (cgc_moe_engine)
//   → output tensor

#ifdef CGC_HAS_PYTORCH

#include "expert_compute.h"
#include "cgc_gguf_lite.h"

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cstring>
#include <iostream>

namespace cgc_moe {

// ============================================================================
// GGML 量化类型常量 (与 cgc_gguf_lite.h 保持一致)
// ============================================================================
#ifndef CGC_GGML_TYPE_BF16
#define CGC_GGML_TYPE_BF16 30
#endif
#ifndef CGC_GGML_TYPE_F16
#define CGC_GGML_TYPE_F16 1
#endif
#ifndef CGC_GGML_TYPE_F32
#define CGC_GGML_TYPE_F32 0
#endif
#ifndef CGC_GGML_TYPE_IQ3_S
#define CGC_GGML_TYPE_IQ3_S 21
#endif
#ifndef CGC_GGML_TYPE_IQ3_M
#define CGC_GGML_TYPE_IQ3_M 22
#endif
#ifndef CGC_GGML_TYPE_Q4_K
#define CGC_GGML_TYPE_Q4_K 14
#endif
#ifndef CGC_GGML_TYPE_Q8_0
#define CGC_GGML_TYPE_Q8_0 8
#endif

// ============================================================================
// 反量化辅助: 将量化权重反量化到 BF16
// ============================================================================
//
// 当前实现: 占位版本, 仅支持 BF16/F16/F32 (零拷贝路径)
// 量化格式 (IQ3_M 等) 的反量化需要对接 llama.cpp 的 ggml dequantize kernel,
// 或自实现 IQ3_M 反量化逻辑. 后续在 CGC_HAS_GGML 定义时对接.
//
// 对于量化权重的临时处理:
//   - 当前返回全零 tensor (保证 shape 正确, 数值待后续对接)
//   - TODO: 接入 llama.cpp ggml.c 的 dequantize_row_iq3_m 等函数

static torch::Tensor dequantizeToBF16(
    const void* data,
    const int64_t shape[2],
    int32_t ggmlType
) {
    // shape[0] = out_dim, shape[1] = in_dim (GGUF weight 布局)
    int64_t out_dim = shape[0];
    int64_t in_dim = shape[1];
    auto options = torch::TensorOptions().dtype(torch::kBFloat16);

    if (ggmlType == CGC_GGML_TYPE_BF16) {
        // 零拷贝: BF16 直接用 from_blob
        // GGUF weight 布局 [out_dim, in_dim], grouped_gemm 需要 [in_dim, out_dim]
        // 但 grouped_gemm_silu_bf16_forward 内部处理转置, 这里保持 [out_dim, in_dim]
        return torch::from_blob(
            data,
            {out_dim, in_dim},
            options
        ).clone();  // clone 确保内存 owned (raw buffer 生命周期不受控)
    }

    if (ggmlType == CGC_GGML_TYPE_F16) {
        // F16 → BF16: 先 from_blob F16, 再 cast
        auto f16_tensor = torch::from_blob(
            data,
            {out_dim, in_dim},
            torch::TensorOptions().dtype(torch::kFloat16)
        );
        return f16_tensor.to(torch::kBFloat16);
    }

    if (ggmlType == CGC_GGML_TYPE_F32) {
        auto f32_tensor = torch::from_blob(
            data,
            {out_dim, in_dim},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        return f32_tensor.to(torch::kBFloat16);
    }

    // 量化格式 (IQ3_M, IQ3_S, Q4_K, Q8_0 等):
    // TODO: 对接 llama.cpp ggml dequantize kernel
    // 当前占位: 返回零 tensor (shape 正确, 数值待对接)
#ifdef CGC_HAS_GGML
    // 有 ggml 链接时调用真实反量化
    // ggml_type type = ...;
    // ggml_fp16_t * dst = ...;
    // ggml_dequantize_row(type, src, dst, n_elements);
#else
    std::cerr << "[expert_compute_torch] WARNING: 量化类型 " << ggmlType
              << " 未接入反量化 kernel, 返回零 tensor. "
              << "shape=[" << out_dim << "," << in_dim << "]" << std::endl;
#endif
    return torch::zeros({out_dim, in_dim}, options);
}

// ============================================================================
// viewsToGroupedWeights 实现
// ============================================================================
//
// ExpertWeightsView 布局:
//   gate: [out_dim, in_dim] (moe_intermediate_size × hidden_size)
//   up:   [out_dim, in_dim] (moe_intermediate_size × hidden_size)
//   down: [in_dim, out_dim] (hidden_size × moe_intermediate_size)
//
// grouped_gemm_silu_bf16_forward 期望 (全部已转置):
//   gate_weights: [num_experts, in_dim, out_dim]  (GGUF gate [out_dim, in_dim] → 转置)
//   up_weights:   [num_experts, in_dim, out_dim]  (GGUF up [out_dim, in_dim] → 转置)
//   down_weights: [num_experts, out_dim, in_dim]  (GGUF down [in_dim, out_dim] → 转置)

GroupedWeights viewsToGroupedWeights(const std::vector<ExpertWeightsView>& views) {
    GroupedWeights result;
    if (views.empty()) {
        return result;
    }

    size_t num_experts = views.size();

    // 从第一个 view 获取形状信息
    int64_t gate_out = views[0].gate.shape[0];  // moe_intermediate_size
    int64_t gate_in  = views[0].gate.shape[1];  // hidden_size
    int64_t up_out   = views[0].up.shape[0];
    int64_t up_in    = views[0].up.shape[1];
    int64_t down_out = views[0].down.shape[0];  // hidden_size
    int64_t down_in  = views[0].down.shape[1];  // moe_intermediate_size

    // 分配输出 tensors
    // grouped_gemm 期望 [num_experts, in_dim, out_dim]
    auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
    result.gate_weights = torch::empty({(int64_t)num_experts, gate_in, gate_out}, bf16);
    result.up_weights   = torch::empty({(int64_t)num_experts, up_in, up_out}, bf16);
    result.down_weights = torch::empty({(int64_t)num_experts, down_out, down_in}, bf16);

    // 逐 expert 反量化 + 转置
    for (size_t i = 0; i < num_experts; i++) {
        const auto& view = views[i];

        // gate: GGUF [out_dim, in_dim] → 反量化 → 转置 [in_dim, out_dim]
        auto gate_bf16 = dequantizeToBF16(view.gate.data, view.gate.shape, view.gate.ggmlType);
        result.gate_weights[i] = gate_bf16.t().contiguous();

        // up: 同 gate
        auto up_bf16 = dequantizeToBF16(view.up.data, view.up.shape, view.up.ggmlType);
        result.up_weights[i] = up_bf16.t().contiguous();

        // down: GGUF [in_dim, out_dim] = [hidden, inter] → 反量化 → 转置 [out_dim, in_dim] = [inter, hidden]
        auto down_bf16 = dequantizeToBF16(view.down.data, view.down.shape, view.down.ggmlType);
        result.down_weights[i] = down_bf16.t().contiguous();
    }

    return result;
}

// ============================================================================
// moeForward 实现: 端到端 MoE 前向
// ============================================================================
#ifdef CGC_HAS_MOE_ENGINE
#include "cgc_moe_engine.h"

torch::Tensor moeForward(
    const torch::Tensor& tokens,
    const torch::Tensor& gating_logits,
    const std::vector<ExpertWeightsView>& views,
    int64_t num_experts_per_token
) {
    // 1. ExpertWeightsView → GroupedWeights (反量化 + 转置)
    GroupedWeights weights = viewsToGroupedWeights(views);

    // 2. DeepEP dispatch: tokens → dispatched_tokens + indices + weights
    int64_t num_experts = (int64_t)views.size();
    auto dispatch_result = deepep_dispatch_forward(
        tokens,
        gating_logits,
        num_experts,
        num_experts_per_token,
        /*ep_size=*/1,
        /*ep_rank=*/0,
        /*mode=*/"low_latency"
    );

    // 3. grouped_gemm_silu_bf16_forward: dispatched_tokens × weights → expert_output
    //    内部融合: gate_proj GEMM + SiLU + up_proj GEMM + element-wise mul + down_proj GEMM
    torch::Tensor expert_output = grouped_gemm_silu_bf16_forward(
        dispatch_result.dispatched_tokens,
        weights.gate_weights,
        weights.up_weights,
        weights.down_weights,
        dispatch_result.dispatch_indices
    );

    // 4. DeepEP combine: expert_output → final_output (原 token 顺序)
    torch::Tensor final_output = deepep_combine_forward(
        expert_output,
        dispatch_result.handle,
        dispatch_result.weights
    );

    return final_output;
}
#else
// 无 cgc_moe_engine 链接时的占位实现
torch::Tensor moeForward(
    const torch::Tensor& tokens,
    const torch::Tensor& gating_logits,
    const std::vector<ExpertWeightsView>& views,
    int64_t num_experts_per_token
) {
    throw std::runtime_error(
        "moeForward requires CGC_HAS_MOE_ENGINE (link cgc_moe_engine). "
        "Rebuild with -DCGC_HAS_MOE_ENGINE=ON"
    );
}
#endif // CGC_HAS_MOE_ENGINE

} // namespace cgc_moe

#endif // CGC_HAS_PYTORCH
