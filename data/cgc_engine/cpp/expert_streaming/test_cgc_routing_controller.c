#include "cgc_routing_controller.h"
#include "cgc_expert_streamer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  TEST: %s\n", name)
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("    FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

static int test_create_destroy(void) {
    printf("\n=== Test 1: Create/Destroy ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL, "pool creation");

    cgc_routing_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy = CGC_ROUTING_STRATEGY_LOOKAHEAD;
    cfg.lookahead_tokens = 8;
    cfg.max_prefetch_per_step = 8;
    cfg.async_prefetch = 0;
    cfg.enable_history_predict = 1;
    cfg.history_predict_order = 2;

    cgc_routing_controller_t* ctrl = cgc_routing_controller_create(pool, &cfg);
    CHECK(ctrl != NULL, "controller creation");
    CHECK(ctrl->config.strategy == CGC_ROUTING_STRATEGY_LOOKAHEAD, "strategy set");
    CHECK(ctrl->config.lookahead_tokens == 8, "lookahead tokens");
    CHECK(ctrl->initialized == 1, "initialized");

    cgc_routing_controller_destroy(ctrl);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 1 PASSED\n");
    return 0;
}

static int test_default_config(void) {
    printf("\n=== Test 2: Default Config ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_routing_controller_t* ctrl = cgc_routing_controller_create(pool, NULL);
    CHECK(ctrl != NULL, "create with NULL config");
    CHECK(ctrl->config.strategy == CGC_ROUTING_STRATEGY_LOOKAHEAD, "default lookahead");
    CHECK(ctrl->config.max_prefetch_per_step == 8, "default budget");
    CHECK(ctrl->config.async_prefetch == 1, "default async");

    cgc_routing_controller_destroy(ctrl);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 2 PASSED\n");
    return 0;
}

static int test_on_route_no_streamer(void) {
    printf("\n=== Test 3: On Route (no streamer in pool) ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();

    cgc_routing_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy = CGC_ROUTING_STRATEGY_LOOKAHEAD;
    cfg.max_prefetch_per_step = 8;
    cfg.async_prefetch = 0;

    cgc_routing_controller_t* ctrl = cgc_routing_controller_create(pool, &cfg);
    CHECK(ctrl != NULL, "controller creation");

    // 无 streamer 时 on_route 应安全返回 0
    int expert_ids[4] = {1, 2, 3, 4};
    int n = cgc_routing_controller_on_route(ctrl, 0, expert_ids, 4);
    CHECK(n == 0, "no streamer -> 0 prefetched");

    cgc_routing_stats_t stats = cgc_routing_controller_stats(ctrl);
    CHECK(stats.total_steps == 1, "step counted");
    CHECK(stats.total_route_ids == 4, "route ids counted");

    cgc_routing_controller_reset_stats(ctrl);
    stats = cgc_routing_controller_stats(ctrl);
    CHECK(stats.total_steps == 0, "stats reset");

    cgc_routing_controller_destroy(ctrl);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 3 PASSED\n");
    return 0;
}

static int test_lookahead_predict(void) {
    printf("\n=== Test 4: Lookahead Predict (via streamer) ===\n");

    // 创建一个真实的小 layout 文件流式加载器
    char path[] = "dummy_routing.bin";
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("    SKIP: cannot create dummy file\n");
        return 0;
    }
    // 8 experts x 4KB
    uint8_t buf[4096] = {0};
    for (int i = 0; i < 8; i++) {
        buf[0] = (uint8_t)i;
        fwrite(buf, 1, sizeof(buf), f);
    }
    fclose(f);

    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strncpy(layout.path, path, CGC_MAX_PATH_LEN - 1);
    layout.stream_offset = 0;
    layout.stream_size = 8 * 4096;
    layout.experts_per_layer = 8;
    layout.expert_stride = 4096;

    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 8, false, NULL, 0);
    CHECK(s != NULL, "streamer creation");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(cgc_streamer_pool_add(pool, 0, s), "pool add streamer layer 0");

    cgc_routing_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy = CGC_ROUTING_STRATEGY_LOOKAHEAD;
    cfg.max_prefetch_per_step = 8;
    cfg.async_prefetch = 0;
    cfg.enable_history_predict = 1;
    cfg.history_predict_order = 2;

    cgc_routing_controller_t* ctrl = cgc_routing_controller_create(pool, &cfg);
    CHECK(ctrl != NULL, "controller creation");

    // 喂入有规律的路由序列, 让历史建立关联
    for (int step = 0; step < 20; step++) {
        int e0 = step % 4;
        int e1 = (step + 1) % 4;
        int ids[2] = {e0, e1};
        cgc_routing_controller_on_route(ctrl, 0, ids, 2);
    }

    cgc_routing_stats_t stats = cgc_routing_controller_stats(ctrl);
    CHECK(stats.total_steps == 20, "20 steps recorded");

    cgc_routing_controller_destroy(ctrl);
    cgc_streamer_pool_destroy(pool);
    cgc_expert_streamer_destroy(s);
    remove(path);

    printf("  Test 4 PASSED\n");
    return 0;
}

static int test_async_prefetch(void) {
    printf("\n=== Test 5: Async Prefetch Thread ===\n");

    char path[] = "dummy_async.bin";
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    uint8_t buf[4096] = {0};
    for (int i = 0; i < 8; i++) {
        buf[0] = (uint8_t)i;
        fwrite(buf, 1, sizeof(buf), f);
    }
    fclose(f);

    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strncpy(layout.path, path, CGC_MAX_PATH_LEN - 1);
    layout.stream_offset = 0;
    layout.stream_size = 8 * 4096;
    layout.experts_per_layer = 8;
    layout.expert_stride = 4096;

    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 8, false, NULL, 0);
    CHECK(s != NULL, "streamer creation");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_streamer_pool_add(pool, 0, s);

    cgc_routing_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy = CGC_ROUTING_STRATEGY_CURRENT_ONLY;
    cfg.max_prefetch_per_step = 8;
    cfg.async_prefetch = 1;   // 异步线程

    cgc_routing_controller_t* ctrl = cgc_routing_controller_create(pool, &cfg);
    CHECK(ctrl != NULL, "controller creation with async");

    int ids[4] = {0, 1, 2, 3};
    cgc_routing_controller_on_route(ctrl, 0, ids, 4);

    // 等待异步线程完成
    cgc_routing_controller_drain(ctrl);

    // 验证专家已进入缓存
    cgc_cache_result_t result = cgc_expert_streamer_load_experts(s, ids, 4, NULL);
    CHECK(result.hits >= 1, "some experts should be cached after async prefetch");

    cgc_routing_controller_destroy(ctrl);
    cgc_streamer_pool_destroy(pool);
    cgc_expert_streamer_destroy(s);
    remove(path);

    printf("  Test 5 PASSED\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  C Routing Controller Test Suite\n");
    printf("============================================================\n");

    int failures = 0;
    failures += test_create_destroy();
    failures += test_default_config();
    failures += test_on_route_no_streamer();
    failures += test_lookahead_predict();
    failures += test_async_prefetch();

    printf("\n============================================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TESTS FAILED\n", failures);
    }
    printf("============================================================\n");

    return failures;
}
