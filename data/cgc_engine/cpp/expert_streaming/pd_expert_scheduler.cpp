// pd_expert_scheduler.cpp — PD-Separated Expert Scheduler Implementation
//
// 实现参考:
//   - turbo-fieldfare/PreadExpertStreamer.swift: 流式加载核心
//   - turbo-fieldfare/PrefillRoutedTileScheduler.swift: tile 调度
//   - turbo-fieldfare/PrefillMoEGrouping.swift: token 路由分组

#include "pd_expert_scheduler.h"
#include "cgc_gguf_lite.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace cgc_moe {

// ============================================================================
// 辅助: 计时
// ============================================================================

static inline uint64_t nowNanos() {
#ifdef _WIN32
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<uint64_t>(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
}

// ============================================================================
// PDRouteHistory
// ============================================================================

PDRouteHistory::PDRouteHistory(int maxHistory, double decayFactor)
    : maxHistory_(maxHistory), decayFactor_(decayFactor) {}

void PDRouteHistory::record(int layer, const std::vector<int>& expertIds) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& layerFreq = freq_[layer];
    
    // 时间衰减
    for (auto& [eid, weight] : layerFreq) {
        weight *= decayFactor_;
    }
    
    // 增加权重
    for (int eid : expertIds) {
        layerFreq[eid] += 1.0;
    }
    
    // 更新共现
    auto& layerCooc = cooccur_[layer];
    for (size_t i = 0; i < expertIds.size(); ++i) {
        for (size_t j = i + 1; j < expertIds.size(); ++j) {
            int e1 = expertIds[i], e2 = expertIds[j];
            uint64_t key = makePairKey(std::min(e1, e2), std::max(e1, e2));
            layerCooc[key] += 1.0;
        }
    }
}

std::vector<int> PDRouteHistory::predictNext(int layer, const std::vector<int>& currentExperts,
                                              int topK) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = freq_.find(layer);
    if (it == freq_.end() || it->second.empty()) {
        // 无历史, 返回均匀分布
        std::vector<int> result(topK);
        for (int i = 0; i < topK; ++i) result[i] = i;
        return result;
    }
    
    const auto& layerFreq = it->second;
    
    // 按频率排序
    std::vector<std::pair<int, double>> sorted(layerFreq.begin(), layerFreq.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<int> predicted;
    predicted.reserve(topK * 2);
    
    // 基础预测: 频率最高的专家
    for (size_t i = 0; i < sorted.size() && predicted.size() < (size_t)(topK * 2); ++i) {
        predicted.push_back(sorted[i].first);
    }
    
    // 共现增强
    if (!currentExperts.empty()) {
        auto coocIt = cooccur_.find(layer);
        if (coocIt != cooccur_.end()) {
            std::unordered_map<int, double> coocScore;
            
            for (int eid : currentExperts) {
                for (const auto& [pairKey, count] : coocIt->second) {
                    int e1 = static_cast<int>(pairKey >> 32);
                    int e2 = static_cast<int>(pairKey & 0xFFFFFFFF);
                    
                    int other = -1;
                    if (e1 == eid) other = e2;
                    else if (e2 == eid) other = e1;
                    
                    if (other >= 0 && std::find(predicted.begin(), predicted.end(), other) == predicted.end()) {
                        coocScore[other] += count;
                    }
                }
            }
            
            // 添加共现频率最高的
            std::vector<std::pair<int, double>> coocSorted(coocScore.begin(), coocScore.end());
            std::sort(coocSorted.begin(), coocSorted.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            int coocAdd = 0;
            for (const auto& [eid, _] : coocSorted) {
                if (coocAdd >= topK / 2) break;
                if (std::find(predicted.begin(), predicted.end(), eid) == predicted.end()) {
                    predicted.push_back(eid);
                    ++coocAdd;
                }
            }
        }
    }
    
    // 截断到 topK
    if ((int)predicted.size() > topK) predicted.resize(topK);
    return predicted;
}

std::vector<int> PDRouteHistory::getMostFrequent(int layer, int topK) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = freq_.find(layer);
    if (it == freq_.end() || it->second.empty()) {
        std::vector<int> result(topK);
        for (int i = 0; i < topK; ++i) result[i] = i;
        return result;
    }
    
    std::vector<std::pair<int, double>> sorted(it->second.begin(), it->second.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<int> result;
    result.reserve(std::min((int)sorted.size(), topK));
    for (int i = 0; i < std::min((int)sorted.size(), topK); ++i) {
        result.push_back(sorted[i].first);
    }
    return result;
}

void PDRouteHistory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    freq_.clear();
    cooccur_.clear();
}

uint64_t PDRouteHistory::makePairKey(int e1, int e2) const {
    return (static_cast<uint64_t>(e1) << 32) | static_cast<uint64_t>(e2);
}

// ============================================================================
// PDLayerAssignment
// ============================================================================

PDLayerAssignment PDLayerAssignment::byRatio(int totalLayers, double prefillRatio) {
    PDLayerAssignment result;
    int prefillCount = static_cast<int>(totalLayers * prefillRatio);
    
    result.prefillLayers.resize(prefillCount);
    for (int i = 0; i < prefillCount; ++i) result.prefillLayers[i] = i;
    
    result.decodeLayers.resize(totalLayers - prefillCount);
    for (int i = 0; i < totalLayers - prefillCount; ++i) {
        result.decodeLayers[i] = prefillCount + i;
    }
    
    return result;
}

PDLayerAssignment PDLayerAssignment::custom(std::vector<int> prefill, std::vector<int> decode) {
    PDLayerAssignment result;
    result.prefillLayers = std::move(prefill);
    result.decodeLayers = std::move(decode);
    return result;
}

bool PDLayerAssignment::isPrefillLayer(int layer) const {
    return std::find(prefillLayers.begin(), prefillLayers.end(), layer) != prefillLayers.end();
}

bool PDLayerAssignment::isDecodeLayer(int layer) const {
    return std::find(decodeLayers.begin(), decodeLayers.end(), layer) != decodeLayers.end();
}

int PDLayerAssignment::getDeviceForLayer(int layer) const {
    if (isPrefillLayer(layer)) return prefillGPU;
    if (isDecodeLayer(layer)) return decodeGPU;
    return -1;
}

// ============================================================================
// PDExpertCacheManager
// ============================================================================

PDExpertCacheManager::PDExpertCacheManager(int gpu0MaxExperts, int gpu1MaxExperts)
    : gpu0Max_(gpu0MaxExperts), gpu1Max_(gpu1MaxExperts) {}

bool PDExpertCacheManager::loadToGPU0(int layer, int expertId, void* data, uint64_t size) {
    std::lock_guard<std::mutex> lock(gpu0Mutex_);
    uint64_t key = makeKey(layer, expertId);
    
    // 淘汰空间
    while ((int)gpu0Cache_.size() >= gpu0Max_ && gpu0Cache_.find(key) == gpu0Cache_.end()) {
        if (!evictFromCache(gpu0Cache_, gpu0Max_)) return false;
    }
    
    CacheEntry entry;
    entry.layer = layer;
    entry.expertId = expertId;
    entry.data = data;
    entry.size = size;
    entry.lastAccess = ++clock_;
    entry.accessCount = 1;
    
    gpu0Cache_[key] = entry;
    return true;
}

void* PDExpertCacheManager::getFromGPU0(int layer, int expertId, uint64_t* outSize) {
    std::lock_guard<std::mutex> lock(gpu0Mutex_);
    uint64_t key = makeKey(layer, expertId);
    
    auto it = gpu0Cache_.find(key);
    if (it == gpu0Cache_.end()) {
        ++gpu0Misses_;
        return nullptr;
    }
    
    ++gpu0Hits_;
    it->second.lastAccess = ++clock_;
    it->second.accessCount++;
    
    if (outSize) *outSize = it->second.size;
    return it->second.data;
}

void PDExpertCacheManager::clearGPU0() {
    std::lock_guard<std::mutex> lock(gpu0Mutex_);
    gpu0Cache_.clear();
}

int PDExpertCacheManager::gpu0Count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(gpu0Mutex_));
    return (int)gpu0Cache_.size();
}

int PDExpertCacheManager::gpu0Max() const { return gpu0Max_; }

bool PDExpertCacheManager::loadToGPU1(int layer, int expertId, void* data, uint64_t size) {
    std::lock_guard<std::mutex> lock(gpu1Mutex_);
    uint64_t key = makeKey(layer, expertId);
    
    while ((int)gpu1Cache_.size() >= gpu1Max_ && gpu1Cache_.find(key) == gpu1Cache_.end()) {
        if (!evictFromCache(gpu1Cache_, gpu1Max_)) return false;
    }
    
    CacheEntry entry;
    entry.layer = layer;
    entry.expertId = expertId;
    entry.data = data;
    entry.size = size;
    entry.lastAccess = ++clock_;
    entry.accessCount = 1;
    
    gpu1Cache_[key] = entry;
    return true;
}

void* PDExpertCacheManager::getFromGPU1(int layer, int expertId, uint64_t* outSize) {
    std::lock_guard<std::mutex> lock(gpu1Mutex_);
    uint64_t key = makeKey(layer, expertId);
    
    auto it = gpu1Cache_.find(key);
    if (it == gpu1Cache_.end()) {
        ++gpu1Misses_;
        return nullptr;
    }
    
    ++gpu1Hits_;
    it->second.lastAccess = ++clock_;
    it->second.accessCount++;
    
    if (outSize) *outSize = it->second.size;
    return it->second.data;
}

void PDExpertCacheManager::clearGPU1() {
    std::lock_guard<std::mutex> lock(gpu1Mutex_);
    gpu1Cache_.clear();
}

int PDExpertCacheManager::gpu1Count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(gpu1Mutex_));
    return (int)gpu1Cache_.size();
}

int PDExpertCacheManager::gpu1Max() const { return gpu1Max_; }

void PDExpertCacheManager::evictFromGPU0(int count) {
    for (int i = 0; i < count; ++i) {
        std::lock_guard<std::mutex> lock(gpu0Mutex_);
        if (!evictFromCache(gpu0Cache_, gpu0Max_)) break;
    }
}

void PDExpertCacheManager::evictFromGPU1(int count) {
    for (int i = 0; i < count; ++i) {
        std::lock_guard<std::mutex> lock(gpu1Mutex_);
        if (!evictFromCache(gpu1Cache_, gpu1Max_)) break;
    }
}

uint64_t PDExpertCacheManager::gpu0Hits() const { return gpu0Hits_; }
uint64_t PDExpertCacheManager::gpu1Hits() const { return gpu1Hits_; }
uint64_t PDExpertCacheManager::gpu0Misses() const { return gpu0Misses_; }
uint64_t PDExpertCacheManager::gpu1Misses() const { return gpu1Misses_; }

double PDExpertCacheManager::gpu0HitRate() const {
    uint64_t total = gpu0Hits_ + gpu0Misses_;
    return total > 0 ? (double)gpu0Hits_ / (double)total * 100.0 : 0.0;
}

double PDExpertCacheManager::gpu1HitRate() const {
    uint64_t total = gpu1Hits_ + gpu1Misses_;
    return total > 0 ? (double)gpu1Hits_ / (double)total * 100.0 : 0.0;
}

uint64_t PDExpertCacheManager::makeKey(int layer, int expertId) const {
    return (static_cast<uint64_t>(layer) << 32) | static_cast<uint64_t>(expertId);
}

int PDExpertCacheManager::evictFromCache(
    std::unordered_map<uint64_t, CacheEntry>& cache, int maxSize) {
    if ((int)cache.size() < maxSize) return true;
    if (cache.empty()) return false;
    
    // 找到 lastAccess 最小的 (LRU)
    uint64_t minAccess = UINT64_MAX;
    uint64_t victimKey = 0;
    bool found = false;
    
    for (const auto& [key, entry] : cache) {
        if (!entry.pinned && entry.lastAccess < minAccess) {
            minAccess = entry.lastAccess;
            victimKey = key;
            found = true;
        }
    }
    
    if (!found) {
        // 如果都是 pinned, 强制驱逐最旧的
        for (const auto& [key, entry] : cache) {
            if (entry.lastAccess < minAccess) {
                minAccess = entry.lastAccess;
                victimKey = key;
                found = true;
            }
        }
    }
    
    if (found) {
        cache.erase(victimKey);
        return true;
    }
    
    return false;
}

// ============================================================================
// PDExpertScheduler
// ============================================================================

PDExpertScheduler::PDExpertScheduler(ExpertStreamerPool& streamerPool,
                                     const PDLayerAssignment& assignment,
                                     int maxExpertsPerLayer,
                                     int tileExperts)
    : streamerPool_(streamerPool)
    , assignment_(assignment)
    , maxExpertsPerLayer_(maxExpertsPerLayer)
    , tileExperts_(tileExperts)
    , currentPhase_(PDFase::Idle)
    , routeHistory_(2000, 0.98)
    , cacheManager_(maxExpertsPerLayer * 8, maxExpertsPerLayer * 8)  // 每层 8 专家 × 多层
{
    fprintf(stderr, "[PDExpertScheduler] Initialized: %d prefill layers, %d decode layers\n",
            (int)assignment_.prefillLayers.size(), (int)assignment_.decodeLayers.size());
}

PDExpertScheduler::~PDExpertScheduler() = default;

void PDExpertScheduler::enterPrefill() {
    fprintf(stderr, "[PDExpertScheduler] Entering PREFILL phase...\n");
    currentPhase_ = PDFase::Prefill;
    
    // 预加载 prefill 层的 top-K 专家到 GPU 0
    int loaded = 0;
    int failed = 0;
    uint64_t t0 = nowNanos();
    
    for (int layer : assignment_.prefillLayers) {
        std::vector<int> expertIds;
        for (int i = 0; i < topK_ && i < maxExpertsPerLayer_; ++i) {
            expertIds.push_back(i);
        }
        
        // 触发异步预取
        streamerPool_.prefetch(layer, expertIds);
        
        // 同步加载前几个
        for (int eid : expertIds) {
            if (loaded >= maxExpertsPerLayer_ * (int)assignment_.prefillLayers.size()) break;
            
            auto* streamer = streamerPool_.getStreamer(layer);
            if (!streamer) { ++failed; continue; }
            
            ExpertCacheAccessContext ctx(
                ExpertCacheSlotOwnerPhase::PrefillTransient,
                ExpertCacheControlPlane::Prefill
            );
            
            auto result = streamer->loadExperts({eid}, ctx);
            if (result.buffers[0]) {
                ++loaded;
                cacheManager_.loadToGPU0(layer, eid, result.buffers[0], result.sizes[0]);
            }
        }
        
        if (failed >= 5) break;
    }
    
    uint64_t elapsed = nowNanos() - t0;
    totalLoadTimeNanos_ += elapsed;
    
    fprintf(stderr, "[PDExpertScheduler] Prefill: loaded %d experts in %.2f ms\n",
            loaded, elapsed / 1e6);
}

void PDExpertScheduler::switchToDecode() {
    fprintf(stderr, "[PDExpertScheduler] Switching to DECODE phase...\n");
    uint64_t t0 = nowNanos();
    
    currentPhase_ = PDFase::Decode;
    
    // 1. 释放 GPU 0 (prefill 完成)
    int gpu0Count = cacheManager_.gpu0Count();
    cacheManager_.clearGPU0();
    fprintf(stderr, "[PDExpertScheduler] Released GPU 0 cache (%d experts)\n", gpu0Count);
    
    // 2. 预加载 decode 层的热门专家到 GPU 1
    int preloaded = 0;
    
    for (int layer : assignment_.decodeLayers) {
        auto hotExperts = routeHistory_.getMostFrequent(layer, maxExpertsPerLayer_ / 2);
        
        if (hotExperts.empty()) {
            for (int i = 0; i < std::min(4, maxExpertsPerLayer_); ++i) {
                hotExperts.push_back(i);
            }
        }
        
        for (int eid : hotExperts) {
            auto* streamer = streamerPool_.getStreamer(layer);
            if (!streamer) continue;
            
            ExpertCacheAccessContext ctx(
                ExpertCacheSlotOwnerPhase::DecodeProtected,
                ExpertCacheControlPlane::Decode
            );
            
            auto result = streamer->loadExperts({eid}, ctx);
            if (result.buffers[0]) {
                cacheManager_.loadToGPU1(layer, eid, result.buffers[0], result.sizes[0]);
                ++preloaded;
            }
        }
    }
    
    // 3. 预取预测的专家
    for (int layer : assignment_.decodeLayers) {
        auto predicted = routeHistory_.predictNext(layer, {}, maxExpertsPerLayer_);
        for (int eid : predicted) {
            auto* streamer = streamerPool_.getStreamer(layer);
            if (streamer) {
                streamer->prefetch({eid});
            }
        }
    }
    
    uint64_t elapsed = nowNanos() - t0;
    totalPrefetchTimeNanos_ += elapsed;
    
    fprintf(stderr, "[PDExpertScheduler] Switch complete: preloaded %d experts in %.2f ms\n",
            preloaded, elapsed / 1e6);
}

std::vector<PDTile> PDExpertScheduler::processPrefill(const std::vector<PDTokenRoutes>& tokenRoutes) {
    if (currentPhase_ != PDFase::Prefill) return {};
    
    std::vector<PDTile> tiles;
    int tileIdx = 0;
    
    for (const auto& routes : tokenRoutes) {
        auto layerTiles = groupIntoTiles(routes);
        for (auto& tile : layerTiles) {
            tile.tileIndex = tileIdx++;
            tiles.push_back(tile);
        }
        
        ++prefillTokens_;
    }
    
    return tiles;
}

ExpertCacheResult PDExpertScheduler::loadPrefillExperts(int layer, const std::vector<int>& expertIds) {
    ExpertCacheResult result;
    
    if (currentPhase_ != PDFase::Prefill || !assignment_.isPrefillLayer(layer)) {
        return result;
    }
    
    auto* streamer = streamerPool_.getStreamer(layer);
    if (!streamer) return result;
    
    ExpertCacheAccessContext ctx(
        ExpertCacheSlotOwnerPhase::PrefillTransient,
        ExpertCacheControlPlane::Prefill
    );
    
    uint64_t t0 = nowNanos();
    result = streamer->loadExperts(expertIds, ctx);
    uint64_t elapsed = nowNanos() - t0;
    
    totalLoadTimeNanos_ += elapsed;
    expertLoads_ += expertIds.size();
    
    // 更新缓存统计
    for (size_t i = 0; i < expertIds.size(); ++i) {
        if (result.buffers[i]) {
            cacheManager_.loadToGPU0(layer, expertIds[i], result.buffers[i], result.sizes[i]);
        }
    }
    
    return result;
}

std::vector<PDTile> PDExpertScheduler::processDecode(const PDTokenRoutes& tokenRoute) {
    if (currentPhase_ != PDFase::Decode) return {};
    
    auto tiles = groupIntoTiles(tokenRoute);
    ++decodeTokens_;
    
    return tiles;
}

ExpertCacheResult PDExpertScheduler::loadDecodeExperts(int layer, const std::vector<int>& expertIds) {
    ExpertCacheResult result;
    
    if (currentPhase_ != PDFase::Decode || !assignment_.isDecodeLayer(layer)) {
        return result;
    }
    
    auto* streamer = streamerPool_.getStreamer(layer);
    if (!streamer) return result;
    
    ExpertCacheAccessContext ctx(
        ExpertCacheSlotOwnerPhase::DecodeProtected,
        ExpertCacheControlPlane::Decode
    );
    
    uint64_t t0 = nowNanos();
    result = streamer->loadExperts(expertIds, ctx);
    uint64_t elapsed = nowNanos() - t0;
    
    totalLoadTimeNanos_ += elapsed;
    expertLoads_ += expertIds.size();
    
    // 更新缓存统计
    for (size_t i = 0; i < expertIds.size(); ++i) {
        if (result.buffers[i]) {
            cacheManager_.loadToGPU1(layer, expertIds[i], result.buffers[i], result.sizes[i]);
        }
    }
    
    return result;
}

std::vector<int> PDExpertScheduler::triggerPrefetch(const PDTokenRoutes& currentRoutes) {
    if (currentPhase_ != PDFase::Decode) return {};
    
    uint64_t t0 = nowNanos();
    std::vector<int> prefetched;
    std::unordered_set<int> prefetchedSet;
    
    for (const auto& route : currentRoutes.routes) {
        if (!assignment_.isDecodeLayer(route.layer)) continue;
        
        auto predicted = routeHistory_.predictNext(route.layer, route.expertIds, maxExpertsPerLayer_);
        
        for (int eid : predicted) {
            if (prefetchedSet.count(eid)) continue;
            
            auto* streamer = streamerPool_.getStreamer(route.layer);
            if (streamer) {
                streamer->prefetch({eid});
                prefetched.push_back(eid);
                prefetchedSet.insert(eid);
                ++prefetchHits_;
            }
        }
    }
    
    uint64_t elapsed = nowNanos() - t0;
    totalPrefetchTimeNanos_ += elapsed;
    
    return prefetched;
}

void PDExpertScheduler::recordRoutes(const PDTokenRoutes& routes) {
    for (const auto& route : routes.routes) {
        routeHistory_.record(route.layer, route.expertIds);
    }
}

std::vector<PDTile> PDExpertScheduler::groupIntoTiles(const PDTokenRoutes& routes) const {
    std::vector<PDTile> tiles;
    
    // 按层分组
    std::unordered_map<int, std::vector<int>> layerExperts;
    for (const auto& route : routes.routes) {
        auto& experts = layerExperts[route.layer];
        for (int eid : route.expertIds) {
            if (std::find(experts.begin(), experts.end(), eid) == experts.end()) {
                experts.push_back(eid);
            }
        }
    }
    
    // 每层的专家分成 tiles
    for (auto& [layer, experts] : layerExperts) {
        // 限制专家数量
        if ((int)experts.size() > maxExpertsPerLayer_) {
            experts.resize(maxExpertsPerLayer_);
        }
        
        // 排序
        std::sort(experts.begin(), experts.end());
        
        // 分成 tiles (每 tile tileExperts 个专家)
        for (size_t i = 0; i < experts.size(); i += tileExperts_) {
            PDTile tile;
            tile.layer = layer;
            size_t end = std::min(i + tileExperts_, experts.size());
            tile.expertIds.assign(experts.begin() + i, experts.begin() + end);
            tile.tokenStart = routes.tokenIndex;
            tile.tokenCount = 1;
            tiles.push_back(tile);
        }
    }
    
    return tiles;
}

void PDExpertScheduler::prefetchLayerExperts(int layer, const std::vector<int>& expertIds, bool toGPU1) {
    auto* streamer = streamerPool_.getStreamer(layer);
    if (!streamer) return;
    
    streamer->prefetch(expertIds);
}

std::vector<int> PDExpertScheduler::getCachedExpertIds(int layer, bool gpu1) const {
    // 这个方法在 cache manager 中实现, 这里简化处理
    return {};
}

PDExpertScheduler::PDSchedulerStats PDExpertScheduler::getStats() const {
    PDSchedulerStats stats;
    stats.phase = currentPhase_;
    stats.gpu0CacheCount = cacheManager_.gpu0Count();
    stats.gpu1CacheCount = cacheManager_.gpu1Count();
    stats.gpu0HitRate = cacheManager_.gpu0HitRate();
    stats.gpu1HitRate = cacheManager_.gpu1HitRate();
    stats.prefillTokens = prefillTokens_;
    stats.decodeTokens = decodeTokens_;
    stats.expertLoads = expertLoads_;
    stats.prefetchHits = prefetchHits_;
    stats.totalPrefetchTimeNanos = totalPrefetchTimeNanos_;
    stats.totalLoadTimeNanos = totalLoadTimeNanos_;
    return stats;
}

void PDExpertScheduler::resetStats() {
    prefillTokens_ = 0;
    decodeTokens_ = 0;
    expertLoads_ = 0;
    prefetchHits_ = 0;
    totalPrefetchTimeNanos_ = 0;
    totalLoadTimeNanos_ = 0;
}

// ============================================================================
// 工厂函数
// ============================================================================

std::unique_ptr<PDExpertScheduler> createPDScheduler(
    const std::vector<std::string>& ggufFiles,
    const PDLayerAssignment& assignment,
    int maxExpertsPerLayer) {
    
    // 创建 streamer pool
    std::shared_ptr<ExpertStreamerPool> pool = std::make_shared<ExpertStreamerPool>();
    
    for (size_t layerIdx = 0; layerIdx < ggufFiles.size(); ++layerIdx) {
        const auto& file = ggufFiles[layerIdx];
        if (file.empty()) continue;
        
        StreamLayout layout = loadStreamLayoutFromGGUF(file);
        if (layout.expertsPerLayer <= 0) {
            fprintf(stderr, "[createPDScheduler] Failed to load GGUF for layer %zu: %s\n",
                    layerIdx, file.c_str());
            continue;
        }
        
        auto streamer = std::make_unique<ExpertStreamer>(
            layout,
            maxExpertsPerLayer * 2,  // 2x 缓存以应对并发
            false,  // useMmap = false (pread mode 更稳定)
            {}  // hotPoolExperts (通过调度器管理)
        );
        
        pool->addStreamer((int)layerIdx, std::move(streamer));
    }
    
    return std::make_unique<PDExpertScheduler>(*pool, assignment, maxExpertsPerLayer);
}

} // namespace cgc_moe
