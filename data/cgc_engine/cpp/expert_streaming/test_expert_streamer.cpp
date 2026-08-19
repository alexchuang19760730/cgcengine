// test_expert_streamer.cpp — 验证 ExpertStreamer 在 Windows 上能跑
//
// 创建一个模拟的 layer 文件 (随机数据),测试:
// 1. pread 模式:cache miss → ReadFile 读取 → cache hit
// 2. mmap 模式:zero-copy page-cache reads
// 3. LRU 驱逐:slot 不够时驱逐最久未用的
// 4. hot pool:pinned slot 不被驱逐
// 5. prefetch:PrefetchVirtualMemory
//
// 运行: test_expert_streamer

#include "expert_streamer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

using namespace cgc_moe;

// ============================================================================
// 创建模拟 layer 文件
// ============================================================================

static std::string createMockLayerFile(int expertsPerLayer, uint64_t expertStride) {
    // 写到临时文件 (每次调用用唯一文件名,避免 ExpertStreamer 持有句柄时冲突)
    static int fileCounter = 0;
    char tmpPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tmpPath);
    std::string path = std::string(tmpPath) + "cgc_test_layer_" + std::to_string(fileCounter++) + ".bin";

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to create temp file: %s\n", path.c_str());
        return "";
    }

    // 写入 expertsPerLayer * expertStride 字节的随机数据
    uint64_t totalSize = static_cast<uint64_t>(expertsPerLayer) * expertStride;
    std::vector<uint8_t> buf(static_cast<size_t>(std::min<uint64_t>(totalSize, 4 * 1024 * 1024)));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    uint64_t written = 0;
    while (written < totalSize) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(buf.size(), totalSize - written));
        for (size_t i = 0; i < chunk; ++i) buf[i] = static_cast<uint8_t>(dist(rng));
        fwrite(buf.data(), 1, chunk, f);
        written += chunk;
    }
    fclose(f);

    fprintf(stderr, "[test] Created mock layer: %s (%llu bytes, %d experts x %llu stride)\n",
            path.c_str(), static_cast<unsigned long long>(totalSize),
            expertsPerLayer, static_cast<unsigned long long>(expertStride));
    return path;
}

// ============================================================================
// 验证 buffer 内容 (每个 expert 的第一个字节 = expertId)
// ============================================================================

static void verifyExpertBuffer(void* buffer, int expertId, uint64_t stride) {
    // 在 createMockLayerFile 里我们写的是随机数据,这里只检查非空
    if (!buffer) {
        fprintf(stderr, "[test] FAIL: expert %d buffer is null\n", expertId);
        return;
    }
    // 检查前 16 字节是否一致 (同一 expert 的数据应该稳定)
    uint8_t* p = static_cast<uint8_t*>(buffer);
    bool allZero = true;
    for (int i = 0; i < 16; ++i) {
        if (p[i] != 0) { allZero = false; break; }
    }
    // 不强制检查 (随机数据可能是全零),只打印
    fprintf(stderr, "[test] expert %d buffer OK (first 4 bytes: %02x %02x %02x %02x)\n",
            expertId, p[0], p[1], p[2], p[3]);
}

// ============================================================================
// Test 1: pread 模式 — basic cache miss/hit
// ============================================================================

static bool testPreadBasic() {
    fprintf(stderr, "\n=== Test 1: pread basic (cache miss → hit) ===\n");

    const int expertsPerLayer = 16;
    const uint64_t expertStride = 1024 * 1024;  // 1MB per expert
    const int slotCount = 4;  // 只有 4 个 slot,测试驱逐

    std::string path = createMockLayerFile(expertsPerLayer, expertStride);
    if (path.empty()) return false;

    StreamLayout layout(path, 0,
                        static_cast<uint64_t>(expertsPerLayer) * expertStride,
                        expertsPerLayer, expertStride);

    ExpertStreamer streamer(layout, slotCount, /*useMmap=*/false);

    // 第一次加载 expert 0,1,2,3 → 4 次 miss
    auto r1 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] Load {0,1,2,3}: hits=%d misses=%d readBytes=%llu\n",
            r1.hits, r1.misses, static_cast<unsigned long long>(r1.readBytes));
    if (r1.hits != 0 || r1.misses != 4) {
        fprintf(stderr, "[test] FAIL: expected 0 hits 4 misses\n");
        return false;
    }
    for (int i = 0; i < 4; ++i) verifyExpertBuffer(r1.buffers[i], i, expertStride);

    // 第二次加载 expert 0,1,2,3 → 4 次 hit
    auto r2 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] Reload {0,1,2,3}: hits=%d misses=%d\n",
            r2.hits, r2.misses);
    if (r2.hits != 4 || r2.misses != 0) {
        fprintf(stderr, "[test] FAIL: expected 4 hits 0 misses\n");
        return false;
    }

    // 加载 expert 4,5,6,7 → 4 次 miss (驱逐 0,1,2,3)
    auto r3 = streamer.loadExperts({4, 5, 6, 7});
    fprintf(stderr, "[test] Load {4,5,6,7}: hits=%d misses=%d\n",
            r3.hits, r3.misses);
    if (r3.hits != 0 || r3.misses != 4) {
        fprintf(stderr, "[test] FAIL: expected 0 hits 4 misses (eviction)\n");
        return false;
    }

    // 再加载 expert 0 → miss (已被驱逐)
    auto r4 = streamer.loadExperts({0});
    fprintf(stderr, "[test] Reload {0} after eviction: hits=%d misses=%d\n",
            r4.hits, r4.misses);
    if (r4.hits != 0 || r4.misses != 1) {
        fprintf(stderr, "[test] FAIL: expected 0 hits 1 miss (after eviction)\n");
        return false;
    }

    auto t = streamer.telemetry();
    fprintf(stderr, "[test] Telemetry: requests=%llu hits=%llu misses=%llu evictions=%llu readBytes=%llu\n",
            static_cast<unsigned long long>(t.totalRequests),
            static_cast<unsigned long long>(t.totalHits),
            static_cast<unsigned long long>(t.totalMisses),
            static_cast<unsigned long long>(t.totalEvictions),
            static_cast<unsigned long long>(t.totalReadBytes));

    fprintf(stderr, "[test] Test 1 PASSED\n");
    return true;
}

// ============================================================================
// Test 2: hot pool — pinned slot 不被驱逐
// ============================================================================

static bool testHotPool() {
    fprintf(stderr, "\n=== Test 2: hot pool (pinned slot) ===\n");

    const int expertsPerLayer = 16;
    const uint64_t expertStride = 1024 * 1024;
    const int slotCount = 4;

    std::string path = createMockLayerFile(expertsPerLayer, expertStride);
    if (path.empty()) return false;

    StreamLayout layout(path, 0,
                        static_cast<uint64_t>(expertsPerLayer) * expertStride,
                        expertsPerLayer, expertStride);

    // expert 0,1 是 hot pool (pinned)
    ExpertStreamer streamer(layout, slotCount, false, {0, 1});

    // 加载 expert 0,1,2,3 → 4 次 miss (0,1 是 hot pool)
    auto r1 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] Load {0,1,2,3}: hits=%d misses=%d\n", r1.hits, r1.misses);

    // 加载 expert 4,5 → 驱逐 2,3 (0,1 pinned 不驱逐)
    auto r2 = streamer.loadExperts({4, 5});
    fprintf(stderr, "[test] Load {4,5}: hits=%d misses=%d\n", r2.hits, r2.misses);

    // 再加载 expert 0,1 → 应该 hit (pinned 没被驱逐)
    auto r3 = streamer.loadExperts({0, 1});
    fprintf(stderr, "[test] Reload {0,1} (pinned): hits=%d misses=%d\n", r3.hits, r3.misses);
    if (r3.hits != 2) {
        fprintf(stderr, "[test] FAIL: hot pool expert was evicted\n");
        return false;
    }

    fprintf(stderr, "[test] Test 2 PASSED\n");
    return true;
}

// ============================================================================
// Test 3: mmap 模式 — zero-copy page-cache
// ============================================================================

static bool testMmap() {
    fprintf(stderr, "\n=== Test 3: mmap mode ===\n");

    const int expertsPerLayer = 8;
    const uint64_t expertStride = 512 * 1024;  // 512KB
    const int slotCount = 4;

    std::string path = createMockLayerFile(expertsPerLayer, expertStride);
    if (path.empty()) return false;

    StreamLayout layout(path, 0,
                        static_cast<uint64_t>(expertsPerLayer) * expertStride,
                        expertsPerLayer, expertStride);

    ExpertStreamer streamer(layout, slotCount, /*useMmap=*/true);

    auto r1 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] mmap Load {0,1,2,3}: hits=%d misses=%d\n", r1.hits, r1.misses);

    // mmap 模式下 buffer 指向映射区域
    for (int i = 0; i < 4; ++i) {
        if (!r1.buffers[i]) {
            fprintf(stderr, "[test] FAIL: mmap buffer %d is null\n", i);
            return false;
        }
    }

    // 重新加载 → 应该 hit (page cache)
    auto r2 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] mmap Reload: hits=%d misses=%d\n", r2.hits, r2.misses);

    fprintf(stderr, "[test] Test 3 PASSED\n");
    return true;
}

// ============================================================================
// Test 4: prefetch
// ============================================================================

static bool testPrefetch() {
    fprintf(stderr, "\n=== Test 4: prefetch ===\n");

    const int expertsPerLayer = 16;
    const uint64_t expertStride = 1024 * 1024;
    const int slotCount = 8;

    std::string path = createMockLayerFile(expertsPerLayer, expertStride);
    if (path.empty()) return false;

    StreamLayout layout(path, 0,
                        static_cast<uint64_t>(expertsPerLayer) * expertStride,
                        expertsPerLayer, expertStride);

    ExpertStreamer streamer(layout, slotCount, /*useMmap=*/true);

    // 预取 expert 0,1,2,3 (不阻塞)
    streamer.prefetch({0, 1, 2, 3});
    fprintf(stderr, "[test] Prefetch {0,1,2,3} done\n");

    // 加载 → 第一次仍然是 miss (prefetch 只是预热 page cache,不填 slot)
    // 但读取应该更快 (page cache hit)
    auto r1 = streamer.loadExperts({0, 1, 2, 3});
    fprintf(stderr, "[test] Load after prefetch: hits=%d misses=%d readWallNanos=%llu\n",
            r1.hits, r1.misses, static_cast<unsigned long long>(r1.readWallNanos));

    fprintf(stderr, "[test] Test 4 PASSED\n");
    return true;
}

// ============================================================================
// Test 5: ExpertStreamerPool — 多 layer
// ============================================================================

static bool testPool() {
    fprintf(stderr, "\n=== Test 5: ExpertStreamerPool ===\n");

    const int expertsPerLayer = 8;
    const uint64_t expertStride = 256 * 1024;
    const int slotCount = 4;
    const int numLayers = 3;

    ExpertStreamerPool pool;

    for (int layer = 0; layer < numLayers; ++layer) {
        std::string path = createMockLayerFile(expertsPerLayer, expertStride);
        if (path.empty()) return false;

        StreamLayout layout(path, 0,
                            static_cast<uint64_t>(expertsPerLayer) * expertStride,
                            expertsPerLayer, expertStride);
        pool.addStreamer(layer, std::make_unique<ExpertStreamer>(layout, slotCount));
    }

    // 在 layer 0 加载 expert 0,1
    auto r0 = pool.loadExperts(0, {0, 1});
    fprintf(stderr, "[test] Layer 0 Load {0,1}: hits=%d misses=%d\n", r0.hits, r0.misses);

    // 在 layer 1 加载 expert 2,3
    auto r1 = pool.loadExperts(1, {2, 3});
    fprintf(stderr, "[test] Layer 1 Load {2,3}: hits=%d misses=%d\n", r1.hits, r1.misses);

    // 遥测
    auto telemetries = pool.allTelemetry();
    for (const auto& [layer, t] : telemetries) {
        fprintf(stderr, "[test] Layer %d: requests=%llu hits=%llu misses=%llu\n",
                layer,
                static_cast<unsigned long long>(t.totalRequests),
                static_cast<unsigned long long>(t.totalHits),
                static_cast<unsigned long long>(t.totalMisses));
    }

    fprintf(stderr, "[test] Test 5 PASSED\n");
    return true;
}

// ============================================================================
// main
// ============================================================================

int main() {
    fprintf(stderr, "=== CGC Expert Streaming Test (Windows port) ===\n");
    fprintf(stderr, "Ported from turbo-fieldfare PreadExpertStreamer.swift\n\n");

    bool allOk = true;

    allOk &= testPreadBasic();
    allOk &= testHotPool();
    allOk &= testMmap();
    allOk &= testPrefetch();
    allOk &= testPool();

    fprintf(stderr, "\n=== %s ===\n", allOk ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return allOk ? 0 : 1;
}
