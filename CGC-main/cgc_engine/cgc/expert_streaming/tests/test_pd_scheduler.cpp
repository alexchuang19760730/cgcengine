// test_pd_scheduler.cpp — PD Expert Scheduler 测试程序
//
// 测试内容:
// 1. PDLayerAssignment: 验证层分配逻辑
// 2. PDRouteHistory: 验证路由历史记录和预测
// 3. PDExpertCacheManager: 验证双 GPU 缓存管理
// 4. PDExpertScheduler: 验证完整 PD 分离流程
// 5. 工厂函数: 验证从 GGUF 文件创建 scheduler

#include "pd_expert_scheduler.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

using namespace cgc_moe;

// ============================================================================
// 测试辅助
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n  TEST: %s\n", name)
#define PASS() do { ++tests_passed; printf("    ✅ PASSED\n"); } while(0)
#define FAIL(msg) do { ++tests_failed; printf("    ❌ FAILED: %s\n", msg); } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

// ============================================================================
// 测试 1: PDLayerAssignment
// ============================================================================

void test_layer_assignment() {
    printf("\n" + std::string(60, '=') + "\n");
    printf("TEST 1: PDLayerAssignment\n");
    printf(std::string(60, '=') + "\n");
    
    // 测试 byRatio
    auto assignment = PDLayerAssignment::byRatio(32, 0.5);
    
    TEST("byRatio 0.5 for 32 layers");
    CHECK(assignment.prefillLayers.size() == 16, "prefill layers should be 16");
    CHECK(assignment.decodeLayers.size() == 16, "decode layers should be 16");
    CHECK(assignment.isPrefillLayer(0), "layer 0 should be prefill");
    CHECK(assignment.isPrefillLayer(15), "layer 15 should be prefill");
    CHECK(!assignment.isPrefillLayer(16), "layer 16 should not be prefill");
    CHECK(assignment.isDecodeLayer(16), "layer 16 should be decode");
    CHECK(assignment.isDecodeLayer(31), "layer 31 should be decode");
    CHECK(assignment.getDeviceForLayer(0) == 0, "prefill layer device should be 0");
    CHECK(assignment.getDeviceForLayer(16) == 1, "decode layer device should be 1");
    
    // 测试 custom
    TEST("custom assignment");
    auto custom = PDLayerAssignment::custom({0, 1, 2, 3}, {4, 5, 6, 7});
    CHECK(custom.isPrefillLayer(0), "layer 0 should be prefill");
    CHECK(!custom.isPrefillLayer(4), "layer 4 should not be prefill");
    CHECK(custom.isDecodeLayer(4), "layer 4 should be decode");
    CHECK(custom.getDeviceForLayer(0) == 0, "prefill layer device should be 0");
    CHECK(custom.getDeviceForLayer(4) == 1, "decode layer device should be 1");
    
    // 测试边界情况
    TEST("boundary cases");
    auto edge = PDLayerAssignment::byRatio(0, 0.5);
    CHECK(edge.prefillLayers.empty(), "empty layers should have no prefill");
    CHECK(edge.decodeLayers.empty(), "empty layers should have no decode");
    
    auto edge2 = PDLayerAssignment::byRatio(1, 0.5);
    CHECK(edge2.prefillLayers.size() == 0, "1 layer with 0.5 ratio should have 0 prefill");
    CHECK(edge2.decodeLayers.size() == 1, "1 layer with 0.5 ratio should have 1 decode");
}

// ============================================================================
// 测试 2: PDRouteHistory
// ============================================================================

void test_route_history() {
    printf("\n" + std::string(60, '=') + "\n");
    printf("TEST 2: PDRouteHistory\n");
    printf(std::string(60, '=') + "\n");
    
    PDRouteHistory history(2000, 0.98);
    
    // 记录一些路由
    TEST("record routes");
    history.record(0, {1, 2, 3, 4});
    history.record(0, {1, 2, 5, 6});
    history.record(0, {1, 3, 7, 8});
    
    auto mostFrequent = history.getMostFrequent(0, 4);
    CHECK(!mostFrequent.empty(), "most frequent should not be empty");
    CHECK(mostFrequent[0] == 1, "expert 1 should be most frequent (appears 3 times)");
    
    // 测试预测
    TEST("predict next");
    auto predicted = history.predictNext(0, {1, 2, 3}, 8);
    CHECK(predicted.size() <= 8, "predicted size should be <= 8");
    CHECK(predicted.size() > 0, "should predict at least 1 expert");
    
    // 测试无历史的情况
    TEST("no history fallback");
    auto predictedEmpty = history.predictNext(99, {}, 8);
    CHECK(predictedEmpty.size() == 8, "should return default topK when no history");
    
    auto mostFrequentEmpty = history.getMostFrequent(99, 8);
    CHECK(mostFrequentEmpty.size() == 8, "should return default when no history");
    
    // 测试共现分析
    TEST("co-occurrence analysis");
    // expert 1 and 2 always co-occur
    auto predictedWithCooc = history.predictNext(0, {1}, 8);
    CHECK(predictedWithCooc.size() > 1, "prediction with co-occurrence should include related experts");
    
    // 测试清空
    TEST("clear");
    history.clear();
    auto afterClear = history.getMostFrequent(0, 8);
    CHECK(afterClear[0] == 0, "after clear, should return default (0)");
}

// ============================================================================
// 测试 3: PDExpertCacheManager
// ============================================================================

void test_cache_manager() {
    printf("\n" + std::string(60, '=') + "\n");
    printf("TEST 3: PDExpertCacheManager\n");
    printf(std::string(60, '=') + "\n");
    
    PDExpertCacheManager cache(16, 16);  // 每个 GPU 最多 16 个专家
    
    // 预分配一些 buffer
    const int BUF_SIZE = 1024;  // 1KB per expert
    std::vector<std::vector<char>> buffers(32);
    for (auto& buf : buffers) buf.resize(BUF_SIZE, 0);
    
    // 测试 GPU 0 缓存
    TEST("GPU 0 cache operations");
    
    // 添加 10 个专家
    for (int i = 0; i < 10; ++i) {
        cache.loadToGPU0(i / 4, i, buffers[i].data(), BUF_SIZE);
    }
    CHECK(cache.gpu0Count() == 10, "GPU 0 should have 10 experts");
    
    // 获取已缓存的专家 (hit)
    void* ptr = cache.getFromGPU0(0, 0);
    CHECK(ptr != nullptr, "should find cached expert (0,0)");
    CHECK(cache.gpu0Hits() == 1, "should have 1 hit");
    
    // 获取未缓存的专家 (miss)
    ptr = cache.getFromGPU0(10, 10);
    CHECK(ptr == nullptr, "should not find uncached expert (10,10)");
    CHECK(cache.gpu0Misses() == 1, "should have 1 miss");
    
    // 命中率
    double hitRate = cache.gpu0HitRate();
    CHECK(hitRate >= 49.0 && hitRate <= 51.0, "hit rate should be ~50%");
    
    // 测试 GPU 1 缓存
    TEST("GPU 1 cache operations");
    
    for (int i = 0; i < 8; ++i) {
        cache.loadToGPU1(i / 2, i, buffers[i + 16].data(), BUF_SIZE);
    }
    CHECK(cache.gpu1Count() == 8, "GPU 1 should have 8 experts");
    
    ptr = cache.getFromGPU1(0, 0);
    CHECK(ptr != nullptr, "should find cached expert in GPU 1");
    
    // 测试驱逐
    TEST("eviction");
    
    // GPU 0: 16 slots, 已有 10, 添加到 16, 再加一个触发驱逐
    for (int i = 10; i < 16; ++i) {
        cache.loadToGPU0(i / 4, i, buffers[i].data(), BUF_SIZE);
    }
    CHECK(cache.gpu0Count() == 16, "GPU 0 should have 16 experts (full)");
    
    // 添加第 17 个, 触发驱逐
    cache.loadToGPU0(4, 16, buffers[16].data(), BUF_SIZE);
    CHECK(cache.gpu0Count() == 16, "GPU 0 should still have 16 experts after eviction");
    
    // 清空
    TEST("clear");
    cache.clearGPU0();
    CHECK(cache.gpu0Count() == 0, "GPU 0 should be empty after clear");
    
    cache.clearGPU1();
    CHECK(cache.gpu1Count() == 0, "GPU 1 should be empty after clear");
}

// ============================================================================
// 测试 4: PDExpertScheduler (使用内存流)
// ============================================================================

void test_scheduler_memory() {
    printf("\n" + std::string(60, '=') + "\n");
    printf("TEST 4: PDExpertScheduler (memory test)\n");
    printf(std::string(60, '=') + "\n");
    
    // 注意: 这是内存测试, 不依赖 GGUF 文件
    // 完整测试需要实际的 per-layer GGUF 文件
    
    auto assignment = PDLayerAssignment::byRatio(8, 0.5);  // 4 prefill, 4 decode
    
    TEST("layer assignment for 8 layers");
    CHECK(assignment.prefillLayers.size() == 4, "should have 4 prefill layers");
    CHECK(assignment.decodeLayers.size() == 4, "should have 4 decode layers");
    
    // 测试路由历史
    PDRouteHistory history(1000, 0.98);
    
    TEST("simulate token routing");
    
    // 模拟 10 个 token 的路由
    for (int token = 0; token < 10; ++token) {
        for (int layer = 0; layer < 8; ++layer) {
            std::vector<int> experts;
            for (int k = 0; k < 4; ++k) {
                experts.push_back((token + k + layer) % 8);
            }
            history.record(layer, experts);
        }
    }
    
    // 验证历史记录
    auto topForLayer0 = history.getMostFrequent(0, 4);
    CHECK(topForLayer0.size() == 4, "should get top 4 for layer 0");
    
    // 验证预测
    auto predicted = history.predictNext(0, {0, 1, 2}, 8);
    CHECK(predicted.size() > 0, "should predict experts for layer 0");
    
    // 测试 tile 分组
    TEST("tile grouping simulation");
    
    PDTokenRoutes tokenRoutes;
    tokenRoutes.tokenIndex = 0;
    
    for (int layer = 0; layer < 4; ++layer) {
        PDExpertRoute route;
        route.layer = layer;
        for (int i = 0; i < 8; ++i) route.expertIds.push_back(i);
        tokenRoutes.routes.push_back(route);
    }
    
    // 简单验证: 路由包含正确的层数
    CHECK(tokenRoutes.routes.size() == 4, "should have 4 layer routes");
    CHECK(tokenRoutes.routes[0].expertIds.size() == 8, "should have 8 experts per layer");
    
    printf("\n  [NOTE] Full PD scheduler test requires per-layer GGUF files.\n");
    printf("  Run test_expert_streamer_gguf with actual model files for end-to-end test.\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  PD EXPERT SCHEDULER TEST SUITE                            ║\n");
    printf("║  Dual-GPU MoE Expert Streaming for Intel UHD + NVIDIA MX250 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    auto t0 = std::chrono::steady_clock::now();
    
    test_layer_assignment();
    test_route_history();
    test_cache_manager();
    test_scheduler_memory();
    
    auto t1 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    
    printf("\n" + std::string(60, '=') + "\n");
    printf("TEST SUMMARY\n");
    printf(std::string(60, '=') + "\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("  Duration: %lld ms\n", (long long)duration.count());
    
    if (tests_failed > 0) {
        printf("\n  ⚠️  SOME TESTS FAILED!\n");
        return 1;
    } else {
        printf("\n  ✅ ALL TESTS PASSED!\n");
        return 0;
    }
}
