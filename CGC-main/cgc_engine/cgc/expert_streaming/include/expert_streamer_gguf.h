// expert_streamer_gguf.h — expert_streaming 接入 cgc_gguf
//
// 功能:
// 1. loadStreamLayoutFromGGUF: 解析 repack 输出的 per-layer GGUF,自动构造 StreamLayout
// 2. findExpertTensors: 查找指定 expert 的所有 sub-tensor (gate/up/down)
// 3. getExpertTensorInfo: 获取 expert tensor 的偏移、大小、形状
//
// 解析流程:
//   repack 输出的 GGUF 包含 KV metadata:
//     - gemma4.expert_count   (u32)  — 每层 expert 数
//     - gemma4.expert_stride  (u32)  — 每个 expert 字节步长
//     - gemma4.hidden_size    (i32)
//     - gemma4.moe_intermediate_size (i32)
//   以及 tensor info:
//     - blk.{layer}.expert.{e}.{role}.weight  (role = gate/up/down)
//
//   loadStreamLayoutFromGGUF 优先读 KV metadata,如果 KV 不可用则从 tensor info 推导。

#pragma once

#include "expert_streamer.h"
#include "cgc_gguf_lite.h"
#include <string>
#include <vector>
#include <cstdint>

namespace cgc_moe {

// expert sub-tensor 信息 (一个 expert 的 gate/up/down)
struct ExpertTensorInfo {
    int expertId = -1;
    std::string role;           // "gate", "up", "down"
    int32_t ggmlType = 0;       // CGC_GGML_TYPE_*
    int64_t dims[4] = {0,0,0,0};
    int nDims = 0;
    uint64_t offset = 0;        // 相对于 data_start 的偏移
    uint64_t sizeBytes = 0;     // tensor data 字节数
};

// 从 repack 输出的 per-layer GGUF 文件加载 StreamLayout
//
// \param ggufPath  GGUF 文件路径
// \return StreamLayout (expertsPerLayer, expertStride, streamOffset, streamSize)
//         失败时返回的 StreamLayout 的 expertsPerLayer == 0
StreamLayout loadStreamLayoutFromGGUF(const std::string& ggufPath);

// 查找指定 expert 的所有 sub-tensor
// \param ctx   cgc_gguf_lite_load 返回的 context
// \param expertId  expert ID
// \return sub-tensor 列表 (gate/up/down),按 tensor 在文件中的顺序
std::vector<ExpertTensorInfo> findExpertTensors(const cgc_gguf_lite_ctx_t* ctx, int expertId);

// 获取 layer GGUF 的元数据
struct LayerGGUFMeta {
    int layerIndex = -1;
    int expertsPerLayer = 0;
    uint64_t expertStride = 0;
    int hiddenSize = 0;
    int moeIntermediateSize = 0;
    std::string quantization;   // "IQ3_M" or "BF16"
    std::string imatrixFile;
};

// 解析 layer GGUF 的元数据
LayerGGUFMeta parseLayerGGUFMeta(const cgc_gguf_lite_ctx_t* ctx);

} // namespace cgc_moe
