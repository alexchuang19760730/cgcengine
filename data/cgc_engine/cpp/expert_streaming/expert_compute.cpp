// expert_compute.cpp — ExpertComputeBridge 实现
//
// 核心逻辑:
//   1. 构造时从 GGUF 解析 expert 0 的 sub-tensor info,计算 gate/up/down 在 buffer 中的 offset
//   2. loadExpertWeights 调用 streamer.loadExperts,获取 buffer 指针
//   3. 根据 offset 从 buffer 中提取 gate/up/down 的 view 指针

#include "expert_compute.h"
#include "cgc_gguf_lite.h"
#include <cstdio>
#include <cstring>

namespace cgc_moe {

ExpertComputeBridge::ExpertComputeBridge(ExpertStreamer& streamer, const cgc_gguf_lite_ctx_t* ggufCtx)
    : streamer_(streamer), ggufCtx_(ggufCtx)
{
    if (ggufCtx_) {
        meta_ = parseLayerGGUFMeta(ggufCtx_);
        computeSubTensorLayout();
    }
}

void ExpertComputeBridge::computeSubTensorLayout() {
    if (!ggufCtx_) return;

    // 获取 expert 0 的所有 sub-tensor (gate/up/down)
    auto tensors = findExpertTensors(ggufCtx_, 0);
    if (tensors.empty()) {
        fprintf(stderr, "[ExpertComputeBridge] no sub-tensors found for expert 0\n");
        return;
    }

    // expert 0 的第一个 tensor (gate) 的 offset 是 expert buffer 的起始 (offset=0)
    // 其他 tensor 的 offset 相对于 gate offset
    uint64_t baseOffset = tensors[0].offset;

    for (const auto& t : tensors) {
        uint64_t relOffset = t.offset - baseOffset;
        if (t.role == "gate") {
            subLayout_.gateOffset = relOffset;
            subLayout_.gateSize = t.sizeBytes;
            subLayout_.gateShape[0] = t.dims[0];
            subLayout_.gateShape[1] = t.dims[1];
            subLayout_.ggmlType = t.ggmlType;
        } else if (t.role == "up") {
            subLayout_.upOffset = relOffset;
            subLayout_.upSize = t.sizeBytes;
            subLayout_.upShape[0] = t.dims[0];
            subLayout_.upShape[1] = t.dims[1];
        } else if (t.role == "down") {
            subLayout_.downOffset = relOffset;
            subLayout_.downSize = t.sizeBytes;
            subLayout_.downShape[0] = t.dims[0];
            subLayout_.downShape[1] = t.dims[1];
        }
    }

    subLayout_.valid = (subLayout_.gateSize > 0 && subLayout_.upSize > 0 && subLayout_.downSize > 0);
    if (!subLayout_.valid) {
        fprintf(stderr, "[ExpertComputeBridge] invalid sub-tensor layout\n");
    }
}

std::vector<ExpertTensorInfo> ExpertComputeBridge::getExpertTensorInfo(int expertId) const {
    if (!ggufCtx_) return {};
    return findExpertTensors(ggufCtx_, expertId);
}

LayerGGUFMeta ExpertComputeBridge::layerMeta() const {
    return meta_;
}

std::vector<ExpertWeightsView> ExpertComputeBridge::loadExpertWeights(const std::vector<int>& expertIds) {
    std::vector<ExpertWeightsView> result;
    if (!subLayout_.valid) {
        fprintf(stderr, "[ExpertComputeBridge] sub-tensor layout not valid\n");
        return result;
    }

    // 调用 streamer 加载 experts
    ExpertCacheResult cacheResult = streamer_.loadExperts(expertIds);
    if (cacheResult.buffers.size() != expertIds.size()) {
        fprintf(stderr, "[ExpertComputeBridge] buffer count mismatch: %zu vs %zu\n",
                cacheResult.buffers.size(), expertIds.size());
        return result;
    }

    // 构造 views (零拷贝)
    for (size_t i = 0; i < expertIds.size(); i++) {
        ExpertWeightsView view;
        view.expertId = expertIds[i];
        view.rawBuffer = cacheResult.buffers[i];
        view.rawSize = streamer_.layout().expertStride;

        // gate
        view.gate.data = (char*)view.rawBuffer + subLayout_.gateOffset;
        view.gate.offsetInBuffer = subLayout_.gateOffset;
        view.gate.sizeBytes = subLayout_.gateSize;
        view.gate.shape[0] = subLayout_.gateShape[0];
        view.gate.shape[1] = subLayout_.gateShape[1];
        view.gate.ggmlType = subLayout_.ggmlType;

        // up
        view.up.data = (char*)view.rawBuffer + subLayout_.upOffset;
        view.up.offsetInBuffer = subLayout_.upOffset;
        view.up.sizeBytes = subLayout_.upSize;
        view.up.shape[0] = subLayout_.upShape[0];
        view.up.shape[1] = subLayout_.upShape[1];
        view.up.ggmlType = subLayout_.ggmlType;

        // down
        view.down.data = (char*)view.rawBuffer + subLayout_.downOffset;
        view.down.offsetInBuffer = subLayout_.downOffset;
        view.down.sizeBytes = subLayout_.downSize;
        view.down.shape[0] = subLayout_.downShape[0];
        view.down.shape[1] = subLayout_.downShape[1];
        view.down.ggmlType = subLayout_.ggmlType;

        result.push_back(view);
    }

    return result;
}

} // namespace cgc_moe
