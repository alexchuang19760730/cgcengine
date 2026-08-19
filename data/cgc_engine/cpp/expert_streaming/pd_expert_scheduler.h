// pd_expert_scheduler.h — PD-Separated Expert Scheduler
//
// 针对 Intel UHD (4GB) + NVIDIA MX250 (2GB) 双 GPU 的 PD 分离式专家调度器.
// 参考 turbo-fieldfare 的 PrefillRoutedTileScheduler + ExpertCachePlan.
//
// 核心功能:
// 1. PD 层分配: prefill 层 → GPU0, decode 层 → GPU1
// 2. Tile 调度: 将 token 路由分组为 tile (每 tile 8 experts)
// 3. 动态专家预取: 基于路由历史预测下一个 token 的专家
// 4. 双 GPU 缓存管理: GPU0 (prefill) + GPU1 (decode) 独立缓存
// 5. 无缝 PD 切换: prefill → decode 时的缓存迁移和失效
//
// 架构:
//   GPU 0 (Intel UHD 4GB)          GPU 1 (NVIDIA MX250 2GB)
//   ┌─────────────────────┐        ┌─────────────────────┐
//   │ Prefill Phase       │        │ Decode Phase        │
//   │ - 处理整个 prompt   │  ─────→ │ - 逐 token 生成     │
//   │ - 加载 prefill 层   │  switch │ - 按需加载专家      │
//   │ - top-K 专家缓存    │        │ - 动态预取          │
//   └─────────────────────┘        └─────────────────────┘

#pragma once

#include "expert_streamer.h"
#include "expert_streamer_gguf.h"

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <functional>

namespace cgc_moe {

// ============================================================================
// 1. PDFase — PD 阶段枚举
// ============================================================================

enum class PDFase : uint8_t {
    Idle = 0,
    Prefill,
    Decode
};

// ============================================================================
// 2. PDExpertRoute — 单个 token 的路由结果
// ============================================================================

struct PDExpertRoute {
    int layer = 0;
    std::vector<int> expertIds;  // top-K expert IDs for this token
};

// ============================================================================
// 3. PDTokenRoutes — 一批 token 的路由结果
// ============================================================================

struct PDTokenRoutes {
    int tokenIndex = 0;
    std::vector<PDExpertRoute> routes;  // per-layer routes for this token
};

// ============================================================================
// 4. PDTile — Tile 调度单元 (参考 PrefillMoETile)
// ============================================================================

struct PDTile {
    int tileIndex = 0;
    int layer = 0;
    std::vector<int> expertIds;  // 本 tile 包含的 expert IDs
    int tokenStart = 0;
    int tokenCount = 0;
};

// ============================================================================
// 5. PDRouteHistory — 路由历史追踪 (用于预取预测)
// ============================================================================

class PDRouteHistory {
public:
    PDRouteHistory(int maxHistory = 2000, double decayFactor = 0.98);
    
    void record(int layer, const std::vector<int>& expertIds);
    std::vector<int> predictNext(int layer, const std::vector<int>& currentExperts, 
                                  int topK = 8) const;
    std::vector<int> getMostFrequent(int layer, int topK = 8) const;
    void clear();

private:
    int maxHistory_;
    double decayFactor_;
    
    // 每层的专家频率: {layer: {expertId: weight}}
    mutable std::mutex mutex_;
    std::unordered_map<int, std::unordered_map<int, double>> freq_;
    
    // 专家共现: {layer: {(e1,e2): count}}
    std::unordered_map<int, std::unordered_map<uint64_t, double>> cooccur_;
    
    uint64_t makePairKey(int e1, int e2) const;
};

// ============================================================================
// 6. PDLayerAssignment — PD 层分配策略
// ============================================================================

struct PDLayerAssignment {
    std::vector<int> prefillLayers;  // 分配到 GPU 0 的层
    std::vector<int> decodeLayers;   // 分配到 GPU 1 的层
    int prefillGPU = 0;              // GPU device index
    int decodeGPU = 1;
    
    PDLayerAssignment() = default;
    
    /// 根据比例分配层 (前 prefillRatio 的层用于 prefill)
    static PDLayerAssignment byRatio(int totalLayers, double prefillRatio = 0.5);
    
    /// 自定义分配
    static PDLayerAssignment custom(std::vector<int> prefill, std::vector<int> decode);
    
    bool isPrefillLayer(int layer) const;
    bool isDecodeLayer(int layer) const;
    int getDeviceForLayer(int layer) const;
};

// ============================================================================
// 7. PDExpertCacheManager — 双 GPU 缓存管理器
// ============================================================================

class PDExpertCacheManager {
public:
    /// 构造
    /// \param gpu0MaxExperts  GPU 0 (prefill) 最大缓存专家数
    /// \param gpu1MaxExperts  GPU 1 (decode) 最大缓存专家数
    PDExpertCacheManager(int gpu0MaxExperts, int gpu1MaxExperts);
    
    // ---- GPU 0 (Prefill) 操作 ----
    bool loadToGPU0(int layer, int expertId, void* data, uint64_t size);
    void* getFromGPU0(int layer, int expertId, uint64_t* outSize = nullptr);
    void clearGPU0();
    int gpu0Count() const;
    int gpu0Max() const;
    
    // ---- GPU 1 (Decode) 操作 ----
    bool loadToGPU1(int layer, int expertId, void* data, uint64_t size);
    void* getFromGPU1(int layer, int expertId, uint64_t* outSize = nullptr);
    void clearGPU1();
    int gpu1Count() const;
    int gpu1Max() const;
    
    // ---- 全局操作 ----
    void evictFromGPU0(int count = 1);
    void evictFromGPU1(int count = 1);
    
    // ---- 统计 ----
    uint64_t gpu0Hits() const;
    uint64_t gpu1Hits() const;
    uint64_t gpu0Misses() const;
    uint64_t gpu1Misses() const;
    double gpu0HitRate() const;
    double gpu1HitRate() const;

private:
    struct CacheEntry {
        int layer = -1;
        int expertId = -1;
        void* data = nullptr;
        uint64_t size = 0;
        uint64_t lastAccess = 0;
        int accessCount = 0;
        bool pinned = false;
    };
    
    int gpu0Max_;
    int gpu1Max_;
    uint64_t clock_;
    
    // {layer_expert_key: CacheEntry}
    std::unordered_map<uint64_t, CacheEntry> gpu0Cache_;
    std::unordered_map<uint64_t, CacheEntry> gpu1Cache_;
    
    mutable std::mutex gpu0Mutex_;
    mutable std::mutex gpu1Mutex_;
    
    // 统计
    std::atomic<uint64_t> gpu0Hits_{0};
    std::atomic<uint64_t> gpu1Hits_{0};
    std::atomic<uint64_t> gpu0Misses_{0};
    std::atomic<uint64_t> gpu1Misses_{0};
    
    uint64_t makeKey(int layer, int expertId) const;
    int evictFromCache(std::unordered_map<uint64_t, CacheEntry>& cache, int maxSize);
};

// ============================================================================
// 8. PDExpertScheduler — 核心 PD 分离式专家调度器
// ============================================================================

class PDExpertScheduler {
public:
    /// 构造
    /// \param streamerPool   已初始化的 ExpertStreamerPool
    /// \param assignment     PD 层分配策略
    /// \param maxExpertsPerLayer  每层最多缓存的专家数
    /// \param tileExperts    每个 tile 包含的专家数 (用于 batching)
    PDExpertScheduler(ExpertStreamerPool& streamerPool,
                      const PDLayerAssignment& assignment,
                      int maxExpertsPerLayer = 8,
                      int tileExperts = 8);
    
    ~PDExpertScheduler();
    
    // ---- 阶段管理 ----
    
    /// 进入 Prefill 阶段 (预加载 prefill 层的专家)
    void enterPrefill();
    
    /// 切换到 Decode 阶段 (释放 GPU0, 预加载 decode 层热门专家到 GPU1)
    void switchToDecode();
    
    /// 获取当前阶段
    PDFase currentPhase() const { return currentPhase_; }
    
    // ---- Prefill 阶段 API ----
    
    /// 处理 prefill 阶段的所有 token 路由
    /// \param tokenRoutes  每个 token 的路由结果
    /// \return 处理后的 tile 列表 (供后续计算使用)
    std::vector<PDTile> processPrefill(const std::vector<PDTokenRoutes>& tokenRoutes);
    
    /// 加载 prefill 阶段指定层的专家到 GPU 0
    /// \param layer  层 ID
    /// \param expertIds  要加载的专家 IDs
    /// \return 加载结果 (包含 buffer 指针)
    ExpertCacheResult loadPrefillExperts(int layer, const std::vector<int>& expertIds);
    
    // ---- Decode 阶段 API ----
    
    /// 处理 decode 阶段的单个 token 路由
    /// \param tokenRoute  当前 token 的路由结果
    /// \return 处理后的 tile 列表
    std::vector<PDTile> processDecode(const PDTokenRoutes& tokenRoute);
    
    /// 加载 decode 阶段指定层的专家到 GPU 1
    /// \param layer  层 ID
    /// \param expertIds  要加载的专家 IDs
    /// \return 加载结果
    ExpertCacheResult loadDecodeExperts(int layer, const std::vector<int>& expertIds);
    
    /// 为下一个 token 触发专家预取
    /// \param currentRoutes  当前 token 的路由 (用于预测)
    /// \return 预取的专家 ID 列表
    std::vector<int> triggerPrefetch(const PDTokenRoutes& currentRoutes);
    
    /// 记录 token 路由 (用于未来预取预测)
    void recordRoutes(const PDTokenRoutes& routes);
    
    // ---- 配置 ----
    
    /// 设置当前阶段的 top-K experts (用于初始预加载)
    void setTopK(int topK) { topK_ = topK; }
    
    /// 获取 tile 大小
    int tileExperts() const { return tileExperts_; }
    
    // ---- 统计 ----
    
    struct PDSchedulerStats {
        PDFase phase = PDFase::Idle;
        int gpu0CacheCount = 0;
        int gpu1CacheCount = 0;
        double gpu0HitRate = 0;
        double gpu1HitRate = 0;
        uint64_t prefillTokens = 0;
        uint64_t decodeTokens = 0;
        uint64_t expertLoads = 0;
        uint64_t prefetchHits = 0;
        uint64_t totalPrefetchTimeNanos = 0;
        uint64_t totalLoadTimeNanos = 0;
    };
    
    PDSchedulerStats getStats() const;
    
    /// 重置统计
    void resetStats();

private:
    // ---- 配置 ----
    ExpertStreamerPool& streamerPool_;
    PDLayerAssignment assignment_;
    int maxExpertsPerLayer_;
    int tileExperts_;
    int topK_ = 8;
    
    // ---- 状态 ----
    PDFase currentPhase_;
    
    // ---- 组件 ----
    PDRouteHistory routeHistory_;
    PDExpertCacheManager cacheManager_;
    
    // ---- 统计 ----
    std::atomic<uint64_t> prefillTokens_{0};
    std::atomic<uint64_t> decodeTokens_{0};
    std::atomic<uint64_t> expertLoads_{0};
    std::atomic<uint64_t> prefetchHits_{0};
    std::atomic<uint64_t> totalPrefetchTimeNanos_{0};
    std::atomic<uint64_t> totalLoadTimeNanos_{0};
    
    // ---- 内部方法 ----
    
    /// 将 token 路由分组为 tiles
    std::vector<PDTile> groupIntoTiles(const PDTokenRoutes& routes) const;
    
    /// 预取单个 layer 的专家到指定 GPU
    void prefetchLayerExperts(int layer, const std::vector<int>& expertIds, bool toGPU1);
    
    /// 获取指定层在当前 GPU 上的缓存专家 IDs
    std::vector<int> getCachedExpertIds(int layer, bool gpu1) const;
};

// ============================================================================
// 9. PDExpertStreamerFactory — 工厂函数
// ============================================================================

/// 从 GGUF 文件集合创建 PDExpertScheduler
/// \param ggufFiles  per-layer GGUF 文件路径列表 (layer index → path)
/// \param assignment  PD 层分配策略
/// \param maxExpertsPerLayer  每层最大缓存专家数
/// \return 初始化好的 scheduler (unique_ptr)
std::unique_ptr<PDExpertScheduler> createPDScheduler(
    const std::vector<std::string>& ggufFiles,
    const PDLayerAssignment& assignment,
    int maxExpertsPerLayer = 8);

} // namespace cgc_moe
