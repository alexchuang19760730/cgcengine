// expert_compute.h — 桥接 ExpertStreamer 和 cgc_moe_engine
//
// 设计理念:
//   expert_streaming 库本身不依赖 PyTorch (保持轻量)
//   ExpertComputeBridge 提供 raw buffer 指针 + 形状信息
//   上层 (有 PyTorch 的代码) 用 torch::from_blob() 包装,然后调用 grouped_gemm
//
// 数据流:
//   GGUF file → ExpertStreamer (cache/pread) → ExpertWeightsView (raw pointers)
//   → torch::from_blob() → grouped_gemm_silu_bf16_forward() → output
//
// ExpertWeightsView 的内存布局:
//   ExpertStreamer 的 slot buffer 包含一个 expert 的完整数据 (gate+up+down 连续)
//   ExpertComputeBridge 根据 GGUF tensor info 计算 gate/up/down 在 buffer 中的 offset
//   返回的 view 指针直接指向 slot buffer 内部 (零拷贝)
//
// 生命周期注意:
//   ExpertWeightsView 的指针在下次 loadExperts 调用前有效
//   (loadExperts 可能驱逐 slot,导致 buffer 被覆盖)

#pragma once

#include "expert_streamer.h"
#include "expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"
#include <vector>
#include <cstdint>
#include <string>

namespace cgc_moe {

// expert sub-tensor 视图 (零拷贝指针)
struct ExpertSubTensorView {
    void* data = nullptr;           // 指向 slot buffer 内部的指针
    int64_t shape[2] = {0, 0};      // [dim0, dim1] (从 GGUF tensor dims)
    int32_t ggmlType = 0;           // CGC_GGML_TYPE_* (BF16=30, IQ3_S=21)
    uint64_t offsetInBuffer = 0;    // 相对于 expert buffer 起始的偏移
    uint64_t sizeBytes = 0;         // sub-tensor 字节数
};

// expert 完整权重视图 (gate + up + down)
struct ExpertWeightsView {
    int expertId = -1;
    ExpertSubTensorView gate;       // gate_proj weight [out_dim, in_dim]
    ExpertSubTensorView up;         // up_proj weight [out_dim, in_dim]
    ExpertSubTensorView down;       // down_proj weight [in_dim, out_dim]
    void* rawBuffer = nullptr;      // expert 完整 buffer (gate+up+down)
    uint64_t rawSize = 0;           // expertStride
};

// 计算桥接器
class ExpertComputeBridge {
public:
    /// 构造
    /// \param streamer  ExpertStreamer (已用 GGUF layout 初始化)
    /// \param ggufCtx   cgc_gguf_lite_load 返回的 context (提供 tensor info)
    ExpertComputeBridge(ExpertStreamer& streamer, const cgc_gguf_lite_ctx_t* ggufCtx);

    /// 加载 experts 并获取权重视图 (零拷贝)
    ///
    /// \param expertIds  需要加载的 expert IDs
    /// \return           每个 expert 的权重视图
    ///   - view.data 指向 streamer cache 内部 (零拷贝)
    ///   - 有效期到下次 loadExperts 调用
    std::vector<ExpertWeightsView> loadExpertWeights(const std::vector<int>& expertIds);

    /// 获取 expert 的 sub-tensor info (不加载,只查 GGUF 信息)
    std::vector<ExpertTensorInfo> getExpertTensorInfo(int expertId) const;

    /// 获取 layer 元数据
    LayerGGUFMeta layerMeta() const;

    /// 获取 streamer 引用
    ExpertStreamer& streamer() { return streamer_; }

private:
    ExpertStreamer& streamer_;
    const cgc_gguf_lite_ctx_t* ggufCtx_;

    // 缓存: expert 0 的 sub-tensor 在 buffer 中的 offset (所有 expert 布局相同)
    struct SubTensorLayout {
        uint64_t gateOffset = 0;
        uint64_t upOffset = 0;
        uint64_t downOffset = 0;
        uint64_t gateSize = 0;
        uint64_t upSize = 0;
        uint64_t downSize = 0;
        int64_t gateShape[2] = {0, 0};
        int64_t upShape[2] = {0, 0};
        int64_t downShape[2] = {0, 0};
        int32_t ggmlType = 0;
        bool valid = false;
    };
    SubTensorLayout subLayout_;
    LayerGGUFMeta meta_;

    void computeSubTensorLayout();
};

// ============================================================================
// 上层集成辅助 (需要 PyTorch,在 .cpp 中仅声明,实现在有 PyTorch 的编译单元)
// ============================================================================

#ifdef CGC_HAS_PYTORCH
#include <torch/extension.h>

/// 将 ExpertWeightsView 列表转换为 grouped_gemm_silu_bf16_forward 需要的权重张量.
///
/// 输入:
///   views: loadExpertWeights 返回的视图 (每个 expert 一个 view)
/// 输出:
///   gate_weights: [num_experts, in_dim, out_dim] (BF16, 已反量化 + 转置)
///   up_weights:   [num_experts, in_dim, out_dim] (已转置)
///   down_weights: [num_experts, out_dim, in_dim] (已转置)
///
/// 若 expert 权重是 IQ3_S/IQ3_M 等量化格式, 会先反量化到 BF16.
/// 零拷贝路径: 若权重本身就是 BF16, 用 torch::from_blob 包装; 否则反量化到新 tensor.
struct GroupedWeights {
    torch::Tensor gate_weights;  // [num_experts, in_dim, out_dim]
    torch::Tensor up_weights;    // [num_experts, in_dim, out_dim]
    torch::Tensor down_weights;  // [num_experts, out_dim, in_dim]
};

GroupedWeights viewsToGroupedWeights(const std::vector<ExpertWeightsView>& views);

/// 端到端 MoE 前向 (DeepEP dispatch + grouped_gemm_silu + combine).
///
/// 完整流程:
///   1. viewsToGroupedWeights: ExpertWeightsView → GroupedWeights (反量化)
///   2. deepep_dispatch_forward: tokens → dispatched_tokens + indices
///   3. grouped_gemm_silu_bf16_forward: dispatched_tokens × weights → expert_output
///   4. deepep_combine_forward: expert_output + weights → final_output
///
/// \param tokens          [num_tokens, hidden_dim] BF16
/// \param gating_logits   [num_tokens, num_experts] FP32
/// \param views           ExpertWeightsView 列表 (从 ExpertComputeBridge 获取)
/// \param num_experts_per_token  每个 token 选择的 expert 数
/// \return                [num_tokens, hidden_dim] BF16
torch::Tensor moeForward(
    const torch::Tensor& tokens,
    const torch::Tensor& gating_logits,
    const std::vector<ExpertWeightsView>& views,
    int64_t num_experts_per_token = 8
);

#endif // CGC_HAS_PYTORCH

} // namespace cgc_moe
