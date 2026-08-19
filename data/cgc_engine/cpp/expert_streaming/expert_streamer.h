// expert_streamer.h — Expert streaming for Windows (ported from turbo-fieldfare)
//
// 移植自 turbo-fieldfare 的 ExpertStreamer.swift + PreadExpertStreamer.swift
// 平台映射:
//   open()        -> CreateFileW()
//   pread()       -> ReadFile() + OVERLAPPED (异步)
//   mmap()        -> CreateFileMapping() + MapViewOfFile()
//   F_RDADVISE    -> PrefetchVirtualMemory() (Win8+)
//   MTLBuffer     -> void* (CPU buffer, llama.cpp 不需要 GPU buffer)
//
// 核心语义 (与 turbo-fieldfare 1:1 对应):
//   StreamLayout         — 文件布局 (per-layer + per-expert offset + stride)
//   ExpertCachePlan      — slot 分配计划 (hits/misses/assignedSlots)
//   ExpertStreamer       — 核心流式加载器 (pread + LRU cache + prefetch)

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <optional>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cgc_moe {

// ============================================================================
// 1. StreamLayout — 文件布局 (直接移植自 ExpertStreamer.swift:30)
// ============================================================================

struct StreamLayout {
    std::string path;              // layer 文件路径
    uint64_t streamOffset = 0;     // stream 在文件中的起始偏移
    uint64_t streamSize = 0;       // stream 总大小
    int expertsPerLayer = 0;       // 每层 expert 数量
    uint64_t expertStride = 0;     // 每个 expert 的字节步长
    std::vector<uint64_t> expertOffsets; // 可选: per-expert 偏移 (layer 0)

    StreamLayout() = default;

    StreamLayout(std::string p, uint64_t off, uint64_t sz,
                 int epl, uint64_t stride,
                 std::vector<uint64_t> eo = {})
        : path(std::move(p)), streamOffset(off), streamSize(sz),
          expertsPerLayer(epl), expertStride(stride),
          expertOffsets(std::move(eo)) {}

    // 计算 (layer, expert) 在文件中的偏移
    inline uint64_t expertOffset(int layer, int expert) const {
        if (layer == 0 && !expertOffsets.empty() &&
            expert >= 0 && expert < static_cast<int>(expertOffsets.size())) {
            return expertOffsets[expert];
        }
        uint64_t perLayer = static_cast<uint64_t>(expertsPerLayer) * expertStride;
        return static_cast<uint64_t>(layer) * perLayer +
               static_cast<uint64_t>(expert) * expertStride;
    }
};

// ============================================================================
// 2. Cache slot 三态 (直接移植自 PreadExpertStreamer.swift:80)
// ============================================================================

enum class ExpertCacheSlotOwnerPhase : uint8_t {
    Unassigned = 0,
    PrefillTransient,   // prefill 临时占用,可被驱逐
    DecodeProtected,    // decode 受保护,优先保留
    SharedResident      // 共享常驻 (hot pool)
};

enum class ExpertCacheControlPlane : uint8_t {
    Prefill = 0,
    Decode,
    SharedPool
};

struct ExpertCacheAccessContext {
    ExpertCacheSlotOwnerPhase ownerPhase = ExpertCacheSlotOwnerPhase::Unassigned;
    ExpertCacheControlPlane controlPlane = ExpertCacheControlPlane::SharedPool;
    uint64_t requestID = 0;
    int decodeStepIndex = -1;

    ExpertCacheAccessContext() = default;
    ExpertCacheAccessContext(ExpertCacheSlotOwnerPhase ph,
                             ExpertCacheControlPlane cp,
                             uint64_t rid = 0,
                             int dsi = -1)
        : ownerPhase(ph), controlPlane(cp), requestID(rid), decodeStepIndex(dsi) {}

    static ExpertCacheAccessContext phaseOnly(ExpertCacheSlotOwnerPhase ph) {
        ExpertCacheControlPlane cp;
        switch (ph) {
            case ExpertCacheSlotOwnerPhase::PrefillTransient:
                cp = ExpertCacheControlPlane::Prefill; break;
            case ExpertCacheSlotOwnerPhase::DecodeProtected:
                cp = ExpertCacheControlPlane::Decode; break;
            default:
                cp = ExpertCacheControlPlane::SharedPool; break;
        }
        return ExpertCacheAccessContext(ph, cp);
    }
};

// ============================================================================
// 3. ExpertCachePlan — slot 分配计划 (直接移植自 PreadExpertStreamer.swift:22)
// ============================================================================

struct ExpertCachePlan {
    std::vector<int> experts;         // 请求的 expert IDs
    std::vector<int> assignedSlots;   // 每个 expert 分配到的 slot (-1 = miss)
    std::vector<int> misses;          // cache miss 的 expert IDs
    int hits = 0;                     // cache hit 数量
};

// ============================================================================
// 4. ExpertCacheResult — loadExperts 返回值
// ============================================================================

struct ExpertCacheResult {
    // 每个 expert 的 buffer 指针 (miss 的先填充再返回)
    // buffer[i] 指向 slotBuffers_[assignedSlots[i]] + offset
    std::vector<void*> buffers;
    std::vector<uint64_t> offsets;    // expert 在 buffer 中的偏移
    std::vector<uint64_t> sizes;      // expert 数据大小
    int hits = 0;
    int misses = 0;
    uint64_t readWallNanos = 0;       // miss 读取耗时
    uint64_t readBytes = 0;           // miss 读取字节数
};

// ============================================================================
// 5. ExpertCacheTelemetry — 基本遥测 (简化版)
// ============================================================================

struct ExpertCacheTelemetry {
    int slotCount = 0;
    int occupiedSlots = 0;
    uint64_t totalRequests = 0;
    uint64_t totalHits = 0;
    uint64_t totalMisses = 0;
    uint64_t totalLoads = 0;
    uint64_t totalEvictions = 0;
    uint64_t totalReadWallNanos = 0;
    uint64_t totalReadBytes = 0;
};

// ============================================================================
// 6. ExpertStreamer — 核心流式加载器 (Windows 版 PreadExpertStreamer)
// ============================================================================

class ExpertStreamer {
public:
    /// 构造:打开 layer 文件,分配 slot buffer
    ///
    /// \param layout      文件布局
    /// \param slotCount   cache slot 数量 (决定常驻内存)
    /// \param useMmap     是否用 mmap 模式 (zero-copy page-cache)
    /// \param hotPoolExperts  预加载的热门 expert IDs (pin 在 slot 里不驱逐)
    ExpertStreamer(const StreamLayout& layout,
                   int slotCount,
                   bool useMmap = false,
                   std::vector<int> hotPoolExperts = {});

    ~ExpertStreamer();

    // 不可拷贝 (持有 file handle + buffer)
    ExpertStreamer(const ExpertStreamer&) = delete;
    ExpertStreamer& operator=(const ExpertStreamer&) = delete;

    /// 加载 experts:先查 cache,hit 直接返回,miss 用 ReadFile 读取
    ///
    /// \param expertIds  请求的 expert IDs (本 layer 内)
    /// \param ctx        访问上下文 (prefill/decode/sharedPool)
    /// \return           每个 expert 的 buffer 指针 + 统计
    ExpertCacheResult loadExperts(const std::vector<int>& expertIds,
                                  const ExpertCacheAccessContext& ctx = {});

    /// 预取 experts (PrefetchVirtualMemory,不阻塞)
    void prefetch(const std::vector<int>& expertIds);

    /// 获取遥测
    ExpertCacheTelemetry telemetry() const;

    /// 释放 slot (标记为 Unassigned)
    void releaseSlot(int slot);

    /// 获取 layout
    const StreamLayout& layout() const { return layout_; }

    /// 获取 hot pool experts
    const std::vector<int>& hotPoolExperts() const { return hotPoolExperts_; }

private:
    // ---- 配置 ----
    StreamLayout layout_;
    int slotCount_;
    bool useMmap_;
    std::vector<int> hotPoolExperts_;

    // ---- 文件句柄 ----
#ifdef _WIN32
    HANDLE fileHandle_ = INVALID_HANDLE_VALUE;
    HANDLE mappingHandle_ = nullptr;   // mmap 模式
    void* mappedBase_ = nullptr;       // mmap 模式
#else
    int fd_ = -1;
    void* mappedBase_ = nullptr;
#endif

    // ---- slot buffer ----
    std::vector<void*> slotBuffers_;          // 每个 slot 的内存 buffer
    std::vector<int> slotExpert_;             // slot 当前持有的 expert (-1 = 空)
    std::vector<ExpertCacheSlotOwnerPhase> slotOwnerPhase_;
    std::vector<int> slotHitCount_;           // slot 命中次数 (LFU)
    std::vector<int> slotLastUse_;            // slot 最后使用时钟 (LRU)
    std::vector<bool> slotPinned_;            // slot 是否 pin (hot pool)

    // ---- 状态 ----
    std::atomic<int> useClock_{0};
    mutable std::mutex cacheLock_;

    // ---- 遥测计数器 ----
    std::atomic<uint64_t> totalRequests_{0};
    std::atomic<uint64_t> totalHits_{0};
    std::atomic<uint64_t> totalMisses_{0};
    std::atomic<uint64_t> totalLoads_{0};
    std::atomic<uint64_t> totalEvictions_{0};
    std::atomic<uint64_t> totalReadWallNanos_{0};
    std::atomic<uint64_t> totalReadBytes_{0};

    // ---- 内部方法 ----

    /// 查找 expert 在 cache 中的 slot (-1 = not found)
    int findSlot(int expertId);

    /// 驱逐一个 slot (LRU/LFU),返回 slot index
    int evictSlot();

    /// 从文件读取 expert 到 buffer (Windows: ReadFile + OVERLAPPED)
    /// \return 读取耗时 (nanos)
    uint64_t readExpert(int expertId, void* buffer);

    /// Windows PrefetchVirtualMemory 预取
    void prefetchExpert(int expertId);

    /// 分配 slot (cache miss 时调用)
    int allocateSlot(const ExpertCacheAccessContext& ctx);

    /// 对齐分配
    void* allocateAligned(size_t size, size_t alignment = 64);
    void freeAligned(void* ptr);
};

// ============================================================================
// 7. ExpertStreamerPool — 多 layer streamer 管理
// ============================================================================

/// 管理多个 layer 的 ExpertStreamer (一个 layer 一个 streamer)
class ExpertStreamerPool {
public:
    /// 添加一个 layer streamer
    void addStreamer(int layerIdx, std::unique_ptr<ExpertStreamer> streamer);

    /// 获取 layer 的 streamer
    ExpertStreamer* getStreamer(int layerIdx);

    /// 加载指定 layer 的 experts
    ExpertCacheResult loadExperts(int layerIdx,
                                  const std::vector<int>& expertIds,
                                  const ExpertCacheAccessContext& ctx = {});

    /// 预取指定 layer 的 experts
    void prefetch(int layerIdx, const std::vector<int>& expertIds);

    /// 获取所有 streamer 的遥测
    std::vector<std::pair<int, ExpertCacheTelemetry>> allTelemetry() const;

private:
    mutable std::mutex poolLock_;
    std::vector<std::pair<int, std::unique_ptr<ExpertStreamer>>> streamers_;
};

} // namespace cgc_moe
