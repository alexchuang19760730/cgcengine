// CGC MoE Engine - 訓推共用算子標頭
//
// 設計目標：
// 1. 推理算子（DeepEP dispatch, DeepGEMM）與訓練算子共用同一份 C++ 實作
// 2. 訓練路徑透過 torch::autograd::Function 自動獲得 backward
// 3. 推理路徑（SGLang）可透過 extern "C" 直接呼叫，無 autograd 開銷
// 4. 解綁 NeMo / TransformerEngine / Megatron-Core 依賴
//
// 算子清單：
// - DeepEP token dispatch（forward + backward）
// - DeepGEMM grouped GEMM（BF16 / FP8，forward + backward）
// - Expert combine（forward + backward）
//
// 編譯：見 CMakeLists.txt
// Python 綁定：見 bindings.cpp

#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>

namespace cgc_moe {

// ============================================================================
// 1. DeepEP Token Dispatch
// ============================================================================

struct DeepEPDispatchResult {
    torch::Tensor dispatched_tokens;  // [num_tokens_after_dispatch, hidden_dim]
    torch::Tensor dispatch_indices;   // [num_tokens, num_experts_per_token]
    torch::Tensor weights;            // [num_tokens, num_experts_per_token]
    torch::Tensor handle;             // 用於 combine 的 handle
};

// Forward: 把 tokens dispatch 到對應的 expert rank
// - 用於推理（SGLang）和訓練（autograd 的 forward）
DeepEPDispatchResult deepep_dispatch_forward(
    const torch::Tensor& tokens,           // [num_tokens, hidden_dim]
    const torch::Tensor& gating_logits,    // [num_tokens, num_experts]
    int64_t num_experts,
    int64_t num_experts_per_token,
    int64_t ep_size,                       // expert parallel size
    int64_t ep_rank,
    const std::string& mode = "normal"     // "normal" / "low_latency"
);

// Backward: 梯度反向 dispatch（訓練專用，推理不呼叫）
torch::Tensor deepep_dispatch_backward(
    const torch::Tensor& grad_output,      // [num_tokens_after_dispatch, hidden_dim]
    const torch::Tensor& handle,
    int64_t num_tokens,
    int64_t hidden_dim
);

// Combine: 把 expert 處理後的 tokens 合併回原順序
// - 用於推理和訓練的 forward
torch::Tensor deepep_combine_forward(
    const torch::Tensor& expert_output,    // [num_tokens_after_dispatch, hidden_dim]
    const torch::Tensor& handle,
    const torch::Tensor& weights           // [num_tokens, num_experts_per_token]
);

// Combine backward（訓練專用）
torch::Tensor deepep_combine_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& handle,
    const torch::Tensor& weights
);

// ============================================================================
// 2. DeepGEMM Grouped GEMM
// ============================================================================

// Forward: 分組矩陣乘法（每個 expert 一個 GEMM）
// - BF16 版本
// - 用於推理和訓練的 forward
torch::Tensor grouped_gemm_bf16_forward(
    const torch::Tensor& tokens,        // [num_tokens, hidden_dim_in]
    const torch::Tensor& expert_weights,// [num_experts, hidden_dim_in, hidden_dim_out]
    const torch::Tensor& indices,       // [num_tokens, num_experts_per_token]
    bool transposed = false             // 是否轉置權重
);

// Backward（訓練專用）
struct GroupedGEMMBackwardResult {
    torch::Tensor grad_tokens;          // [num_tokens, hidden_dim_in]
    torch::Tensor grad_expert_weights;  // [num_experts, hidden_dim_in, hidden_dim_out]
};

GroupedGEMMBackwardResult grouped_gemm_bf16_backward(
    const torch::Tensor& grad_output,   // [num_tokens, hidden_dim_out]
    const torch::Tensor& tokens,        // [num_tokens, hidden_dim_in]
    const torch::Tensor& expert_weights,// [num_experts, hidden_dim_in, hidden_dim_out]
    const torch::Tensor& indices        // [num_tokens, num_experts_per_token]
);

// FP8 版本（需 Blackwell 架構）
torch::Tensor grouped_gemm_fp8_forward(
    const torch::Tensor& tokens,        // FP8
    const torch::Tensor& expert_weights,// FP8
    const torch::Tensor& token_scale,   // per-token scale
    const torch::Tensor& weight_scale,  // per-expert scale
    const torch::Tensor& indices,
    bool transposed = false
);

GroupedGEMMBackwardResult grouped_gemm_fp8_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tokens,
    const torch::Tensor& expert_weights,
    const torch::Tensor& token_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& indices
);

// ============================================================================
// 3. SiLU Activation（融合到 GEMM，避免額外 kernel launch）
// ============================================================================

torch::Tensor grouped_gemm_silu_bf16_forward(
    const torch::Tensor& tokens,
    const torch::Tensor& gate_weights,  // gate_proj
    const torch::Tensor& up_weights,    // up_proj
    const torch::Tensor& down_weights,  // down_proj
    const torch::Tensor& indices
);

// ============================================================================
// 4. 工具函數
// ============================================================================

// 偵測可用後端（回傳 "cuda" / "metal" / "cpu"）
std::string detect_backend();

// 偵測 DeepEP 是否可用（編譯時連結）
bool deepep_available();

// 偵測 DeepGEMM 是否可用
bool deep_gemm_available();

// 取得版本
std::string version();

}  // namespace cgc_moe
