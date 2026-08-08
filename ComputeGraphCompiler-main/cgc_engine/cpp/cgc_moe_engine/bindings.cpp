// CGC MoE Engine - PyTorch Autograd 綁定
//
// 關鍵設計：
// 把 C++ 算子包裝成 torch::autograd::Function，讓推理算子自動獲得 backward。
// 推理路徑：直接呼叫 forward（torch.no_grad() 下不計算梯度）
// 訓練路徑：呼叫 forward + backward（autograd 自動呼叫）
//
// 這是「推理算子在 C++ engine 讓訓練算子也能共用」的核心實作。

#include <torch/extension.h>
#include "cgc_moe_engine.h"

// ============================================================================
// 1. DeepEP Dispatch Autograd Function
// ============================================================================

class DeepEPDispatchFunction : public torch::autograd::Function<DeepEPDispatchFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor tokens,
        torch::Tensor gating_logits,
        int64_t num_experts,
        int64_t num_experts_per_token,
        int64_t ep_size,
        int64_t ep_rank,
        std::string mode
    ) {
        auto result = cgc_moe::deepep_dispatch_forward(
            tokens, gating_logits, num_experts, num_experts_per_token,
            ep_size, ep_rank, mode
        );
        // 保存給 backward
        ctx->save_for_backward({tokens, result.dispatch_indices, result.weights, result.handle});
        ctx->saved_data["num_tokens"] = tokens.size(0);
        ctx->saved_data["hidden_dim"] = tokens.size(1);
        return result.dispatched_tokens;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto tokens = saved[0];
        auto handle = saved[3];
        int64_t num_tokens = ctx->saved_data["num_tokens"].toInt();
        int64_t hidden_dim = ctx->saved_data["hidden_dim"].toInt();

        auto grad_dispatched = grad_outputs[0];
        auto grad_tokens = cgc_moe::deepep_dispatch_backward(
            grad_dispatched, handle, num_tokens, hidden_dim
        );

        // 梯度：tokens 有梯度，gating_logits 有梯度（透過 softmax）
        // 簡化：返回與 forward 輸入數量相同的梯度
        return {
            grad_tokens,             // tokens 梯度
            torch::Tensor(),         // gating_logits 梯度（需實作 softmax 反向）
            torch::Tensor(),         // num_experts (int, 無梯度)
            torch::Tensor(),         // num_experts_per_token
            torch::Tensor(),         // ep_size
            torch::Tensor(),         // ep_rank
            torch::Tensor(),         // mode
        };
    }
};

// ============================================================================
// 2. DeepEP Combine Autograd Function
// ============================================================================

class DeepEPCombineFunction : public torch::autograd::Function<DeepEPCombineFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor expert_output,
        torch::Tensor handle,
        torch::Tensor weights
    ) {
        ctx->save_for_backward({handle, weights});
        return cgc_moe::deepep_combine_forward(expert_output, handle, weights);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto handle = saved[0];
        auto weights = saved[1];
        auto grad_expert_output = cgc_moe::deepep_combine_backward(
            grad_outputs[0], handle, weights
        );
        return {grad_expert_output, torch::Tensor(), torch::Tensor()};
    }
};

// ============================================================================
// 3. GroupedGEMM BF16 Autograd Function
// ============================================================================

class GroupedGEMMBF16Function : public torch::autograd::Function<GroupedGEMMBF16Function> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor tokens,
        torch::Tensor expert_weights,
        torch::Tensor indices,
        bool transposed
    ) {
        ctx->save_for_backward({tokens, expert_weights, indices});
        ctx->saved_data["transposed"] = transposed;
        return cgc_moe::grouped_gemm_bf16_forward(tokens, expert_weights, indices, transposed);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto tokens = saved[0];
        auto expert_weights = saved[1];
        auto indices = saved[2];

        auto grad_result = cgc_moe::grouped_gemm_bf16_backward(
            grad_outputs[0], tokens, expert_weights, indices
        );
        return {
            grad_result.grad_tokens,
            grad_result.grad_expert_weights,
            torch::Tensor(),  // indices (int, 無梯度)
            torch::Tensor(),  // transposed (bool, 無梯度)
        };
    }
};

// ============================================================================
// 4. GroupedGEMM FP8 Autograd Function
// ============================================================================

class GroupedGEMMFP8Function : public torch::autograd::Function<GroupedGEMMFP8Function> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor tokens,
        torch::Tensor expert_weights,
        torch::Tensor token_scale,
        torch::Tensor weight_scale,
        torch::Tensor indices,
        bool transposed
    ) {
        ctx->save_for_backward({tokens, expert_weights, token_scale, weight_scale, indices});
        return cgc_moe::grouped_gemm_fp8_forward(
            tokens, expert_weights, token_scale, weight_scale, indices, transposed
        );
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto tokens = saved[0];
        auto expert_weights = saved[1];
        auto token_scale = saved[2];
        auto weight_scale = saved[3];
        auto indices = saved[4];

        auto grad_result = cgc_moe::grouped_gemm_fp8_backward(
            grad_outputs[0], tokens, expert_weights, token_scale, weight_scale, indices
        );
        return {
            grad_result.grad_tokens,
            grad_result.grad_expert_weights,
            torch::Tensor(),  // token_scale (無梯度，FP8 量化不反向)
            torch::Tensor(),  // weight_scale
            torch::Tensor(),  // indices
            torch::Tensor(),  // transposed
        };
    }
};

// ============================================================================
// 5. Python 綁定（PYBIND11_MODULE）
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CGC MoE Engine - 訓推共用算子（推理算子可用於訓練）";

    // 工具函數
    m.def("detect_backend", &cgc_moe::detect_backend, "偵測可用後端");
    m.def("deepep_available", &cgc_moe::deepep_available, "DeepEP 是否可用");
    m.def("deep_gemm_available", &cgc_moe::deep_gemm_available, "DeepGEMM 是否可用");
    m.def("version", &cgc_moe::version, "取得版本資訊");

    // === 推理路徑（直接呼叫 forward，無 autograd 開銷）===
    m.def("deepep_dispatch_forward", &cgc_moe::deepep_dispatch_forward,
          "DeepEP token dispatch forward（推理用）");
    m.def("deepep_combine_forward", &cgc_moe::deepep_combine_forward,
          "DeepEP combine forward（推理用）");
    m.def("grouped_gemm_bf16_forward", &cgc_moe::grouped_gemm_bf16_forward,
          "GroupedGEMM BF16 forward（推理用）");
    m.def("grouped_gemm_fp8_forward", &cgc_moe::grouped_gemm_fp8_forward,
          "GroupedGEMM FP8 forward（推理用）");

    // === 訓練路徑（透過 autograd::Function，自動獲得 backward）===
    m.def("deepep_dispatch", [](torch::Tensor tokens, torch::Tensor logits,
                                 int64_t num_experts, int64_t k,
                                 int64_t ep_size, int64_t ep_rank,
                                 std::string mode) {
        return DeepEPDispatchFunction::apply(
            tokens, logits, num_experts, k, ep_size, ep_rank, mode
        );
    }, "DeepEP dispatch（訓練用，自動 backward）");

    m.def("deepep_combine", [](torch::Tensor expert_output, torch::Tensor handle,
                                torch::Tensor weights) {
        return DeepEPCombineFunction::apply(expert_output, handle, weights);
    }, "DeepEP combine（訓練用，自動 backward）");

    m.def("grouped_gemm_bf16", [](torch::Tensor tokens, torch::Tensor weights,
                                   torch::Tensor indices, bool transposed) {
        return GroupedGEMMBF16Function::apply(tokens, weights, indices, transposed);
    }, "GroupedGEMM BF16（訓練用，自動 backward）");

    m.def("grouped_gemm_fp8", [](torch::Tensor tokens, torch::Tensor weights,
                                  torch::Tensor token_scale, torch::Tensor weight_scale,
                                  torch::Tensor indices, bool transposed) {
        return GroupedGEMMFP8Function::apply(
            tokens, weights, token_scale, weight_scale, indices, transposed
        );
    }, "GroupedGEMM FP8（訓練用，自動 backward）");
}
