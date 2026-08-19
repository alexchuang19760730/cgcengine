// expert_streamer_gguf.cpp — expert_streaming 接入 cgc_gguf 实现
//
// 实现 loadStreamLayoutFromGGUF: 解析 repack 输出的 per-layer GGUF,构造 StreamLayout
//
// 核心逻辑:
//   1. cgc_gguf_lite_load(path) 解析 GGUF header
//   2. 优先从 KV metadata 读 expert_count, expert_stride
//   3. 如果 KV 不可用,从 tensor info 推导:
//      - tensor name 格式: blk.{layer}.expert.{e}.{role}.weight
//      - expertsPerLayer = max(expert_id) + 1
//      - expertStride = tensor[expert=1,first_role].offset - tensor[expert=0,first_role].offset
//   4. streamOffset = data_start + tensor[expert=0,first_role].offset
//   5. streamSize = expertsPerLayer * expertStride

#include "expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

namespace cgc_moe {

// ============================================================================
// 内部辅助:从 tensor name 解析 expert ID 和 role
// ============================================================================

// tensor name 格式: blk.{layer}.expert.{e}.{role}.weight
// 返回 true = 解析成功
static bool parseTensorName(const char* name, int* outLayer, int* outExpert, std::string& outRole) {
    if (!name) return false;
    int layer = -1, expert = -1;
    char role[64] = {0};
    // 尝试匹配 blk.%d.expert.%d.%63s.weight
    int n = sscanf(name, "blk.%d.expert.%d.%63[^.].weight", &layer, &expert, role);
    if (n != 3) return false;
    if (outLayer) *outLayer = layer;
    if (outExpert) *outExpert = expert;
    outRole = role;
    return true;
}

// ============================================================================
// loadStreamLayoutFromGGUF
// ============================================================================

StreamLayout loadStreamLayoutFromGGUF(const std::string& ggufPath) {
    StreamLayout layout;
    layout.path = ggufPath;

    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(ggufPath.c_str());
    if (!ctx) {
        fprintf(stderr, "[expert_streamer_gguf] failed to load GGUF: %s\n", ggufPath.c_str());
        return layout;  // expertsPerLayer = 0
    }

    // ---- 优先从 KV metadata 读 expert_count, expert_stride ----
    uint32_t expertCount = 0;
    uint32_t expertStrideU32 = 0;
    bool hasExpertCount = cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expertCount);
    bool hasExpertStride = cgc_gguf_lite_get_u32(ctx, "gemma4.expert_stride", &expertStrideU32);

    int expertsPerLayer = 0;
    uint64_t expertStride = 0;

    if (hasExpertCount && hasExpertStride && expertCount > 0 && expertStrideU32 > 0) {
        // KV metadata 可用
        expertsPerLayer = (int)expertCount;
        expertStride = (uint64_t)expertStrideU32;
    } else {
        // 从 tensor info 推导
        // 遍历所有 tensor,找 max expert_id 和第一个 expert 的 offset
        int maxExpertId = -1;
        uint64_t firstExpertFirstOffset = 0;
        uint64_t secondExpertFirstOffset = 0;
        bool foundFirst = false;
        bool foundSecond = false;

        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer, expert;
            std::string role;
            if (!parseTensorName(ctx->tensor_names[i], &layer, &expert, role)) continue;

            if (expert > maxExpertId) maxExpertId = expert;

            // 第一个 expert (expert=0) 的第一个 tensor
            if (expert == 0 && !foundFirst) {
                firstExpertFirstOffset = ctx->tensors[i].offset;
                foundFirst = true;
            }
            // 第二个 expert (expert=1) 的第一个 tensor (用于计算 stride)
            if (expert == 1 && !foundSecond) {
                secondExpertFirstOffset = ctx->tensors[i].offset;
                foundSecond = true;
            }
        }

        if (maxExpertId < 0) {
            fprintf(stderr, "[expert_streamer_gguf] no expert tensors found in %s\n", ggufPath.c_str());
            cgc_gguf_lite_free(ctx);
            return layout;
        }

        expertsPerLayer = maxExpertId + 1;

        if (foundFirst && foundSecond) {
            expertStride = secondExpertFirstOffset - firstExpertFirstOffset;
        } else {
            // 只有 1 个 expert,从 tensor size 计算总大小作为 stride
            // (这种情况不常见,但避免除零)
            expertStride = 0;
            for (uint64_t i = 0; i < ctx->n_tensors; i++) {
                int layer, expert;
                std::string role;
                if (!parseTensorName(ctx->tensor_names[i], &layer, &expert, role)) continue;
                if (expert == 0) {
                    double bpe = cgc_ggml_type_bytes_per_elem(ctx->tensors[i].type);
                    expertStride += (uint64_t)(bpe * (double)ctx->tensors[i].n_elements);
                }
            }
            if (expertStride == 0) expertStride = 1;  // 避免 0
        }
    }

    // ---- 计算 streamOffset (文件绝对偏移) ----
    // streamOffset = data_start + 第一个 expert 的第一个 tensor 的 offset
    uint64_t streamOffset = ctx->data_start;
    bool foundFirstTensor = false;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        int layer, expert;
        std::string role;
        if (!parseTensorName(ctx->tensor_names[i], &layer, &expert, role)) continue;
        if (expert == 0) {
            streamOffset += ctx->tensors[i].offset;
            foundFirstTensor = true;
            break;
        }
    }
    if (!foundFirstTensor) {
        fprintf(stderr, "[expert_streamer_gguf] no expert-0 tensor found\n");
        cgc_gguf_lite_free(ctx);
        return layout;
    }

    // ---- 填充 StreamLayout ----
    layout.streamOffset = streamOffset;
    layout.streamSize = (uint64_t)expertsPerLayer * expertStride;
    layout.expertsPerLayer = expertsPerLayer;
    layout.expertStride = expertStride;

    cgc_gguf_lite_free(ctx);
    return layout;
}

// ============================================================================
// findExpertTensors
// ============================================================================

std::vector<ExpertTensorInfo> findExpertTensors(const cgc_gguf_lite_ctx_t* ctx, int expertId) {
    std::vector<ExpertTensorInfo> result;
    if (!ctx) return result;

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        int layer, expert;
        std::string role;
        if (!parseTensorName(ctx->tensor_names[i], &layer, &expert, role)) continue;
        if (expert != expertId) continue;

        ExpertTensorInfo info;
        info.expertId = expertId;
        info.role = role;
        info.ggmlType = ctx->tensors[i].type;
        info.nDims = ctx->tensors[i].n_dims;
        for (int j = 0; j < 4 && j < info.nDims; j++) {
            info.dims[j] = ctx->tensors[i].dims[j];
        }
        info.offset = ctx->tensors[i].offset;
        // 计算 sizeBytes
        double bpe = cgc_ggml_type_bytes_per_elem(ctx->tensors[i].type);
        info.sizeBytes = (uint64_t)(bpe * (double)ctx->tensors[i].n_elements);
        result.push_back(info);
    }

    return result;
}

// ============================================================================
// parseLayerGGUFMeta
// ============================================================================

LayerGGUFMeta parseLayerGGUFMeta(const cgc_gguf_lite_ctx_t* ctx) {
    LayerGGUFMeta meta;
    if (!ctx) return meta;

    int32_t layerIdx = 0;
    if (cgc_gguf_lite_get_i32(ctx, "general.layer_index", &layerIdx)) {
        meta.layerIndex = layerIdx;
    }

    uint32_t expertCount = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expertCount)) {
        meta.expertsPerLayer = (int)expertCount;
    }

    uint32_t expertStride = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_stride", &expertStride)) {
        meta.expertStride = (uint64_t)expertStride;
    }

    int32_t hiddenSize = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.hidden_size", &hiddenSize)) {
        meta.hiddenSize = hiddenSize;
    }

    int32_t moeInter = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.moe_intermediate_size", &moeInter)) {
        meta.moeIntermediateSize = moeInter;
    }

    const char* quant = cgc_gguf_lite_get_str(ctx, "gemma4.quantization");
    if (quant) meta.quantization = quant;

    const char* imatrix = cgc_gguf_lite_get_str(ctx, "gemma4.imatrix_file");
    if (imatrix) meta.imatrixFile = imatrix;

    return meta;
}

} // namespace cgc_moe
