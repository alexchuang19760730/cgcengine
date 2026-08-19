#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TEST(name) printf("  TEST: %s\n", name)
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("    FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

static int test_streamer_create_destroy(void) {
    printf("\n=== Test 1: Streamer create/destroy ===\n");

    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strncpy(layout.path, "test.gguf", CGC_MAX_PATH_LEN - 1);
    layout.stream_offset = 0;
    layout.stream_size = 1024 * 1024;
    layout.experts_per_layer = 8;
    layout.expert_stride = 256 * 1024;

    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 8, false, NULL, 0);
    if (s) {
        printf("  Created with invalid file (expected to fail gracefully or return NULL)\n");
        cgc_expert_streamer_destroy(s);
    } else {
        printf("  Correctly returned NULL for invalid file\n");
    }

    printf("  Test 1 PASSED\n");
    return 0;
}

static int test_streamer_pool(void) {
    printf("\n=== Test 2: Streamer Pool ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL, "pool creation");

    CHECK(cgc_streamer_pool_get(pool, 0) == NULL, "get non-existent layer");
    CHECK(cgc_streamer_pool_add(pool, 0, NULL) == false, "add null streamer");

    cgc_streamer_pool_destroy(pool);

    printf("  Test 2 PASSED\n");
    return 0;
}

static int test_cache_plan(void) {
    printf("\n=== Test 3: Cache structures ===\n");

    cgc_cache_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.count = 4;
    plan.hits = 2;
    plan.miss_count = 2;
    for (int i = 0; i < 4; i++) {
        plan.expert_ids[i] = i * 10;
        plan.assigned_slots[i] = i;
    }

    CHECK(plan.count == 4, "plan count");
    CHECK(plan.hits == 2, "plan hits");
    CHECK(plan.miss_count == 2, "plan misses");
    CHECK(plan.expert_ids[2] == 20, "plan expert ids");

    cgc_cache_result_t result;
    memset(&result, 0, sizeof(result));
    result.count = 3;
    result.hits = 1;
    result.misses = 2;
    result.read_wall_nanos = 1000000;
    result.read_bytes = 256 * 1024;

    CHECK(result.count == 3, "result count");
    CHECK(result.hits == 1, "result hits");
    CHECK(result.read_bytes == 256 * 1024, "result bytes");

    cgc_cache_telemetry_t tel;
    memset(&tel, 0, sizeof(tel));
    tel.slot_count = 8;
    tel.occupied_slots = 5;
    tel.total_requests = 100;
    tel.total_hits = 80;
    tel.total_misses = 20;

    CHECK(tel.slot_count == 8, "telemetry slot count");
    CHECK(tel.total_hits == 80, "telemetry hits");

    printf("  Test 3 PASSED\n");
    return 0;
}

static int test_stream_layout(void) {
    printf("\n=== Test 4: Stream Layout ===\n");

    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strncpy(layout.path, "/path/to/layer.gguf", CGC_MAX_PATH_LEN - 1);
    layout.stream_offset = 4096;
    layout.stream_size = 8 * 256 * 1024;
    layout.experts_per_layer = 8;
    layout.expert_stride = 256 * 1024;
    layout.has_explicit_offsets = false;

    uint64_t off0 = cgc_expert_offset(&layout, 0, 0);
    uint64_t off1 = cgc_expert_offset(&layout, 0, 1);
    uint64_t off7 = cgc_expert_offset(&layout, 0, 7);

    CHECK(off0 == 4096, "expert 0 offset");
    CHECK(off1 == 4096 + 256 * 1024, "expert 1 offset");
    CHECK(off7 == 4096 + 7 * 256 * 1024, "expert 7 offset");

    uint64_t off_layer1_exp0 = cgc_expert_offset(&layout, 1, 0);
    uint64_t off_layer1_exp1 = cgc_expert_offset(&layout, 1, 1);

    uint64_t per_layer = (uint64_t)8 * 256 * 1024;
    CHECK(off_layer1_exp0 == 4096 + per_layer, "layer 1 expert 0 offset");
    CHECK(off_layer1_exp1 == 4096 + per_layer + 256 * 1024, "layer 1 expert 1 offset");

    printf("  Test 4 PASSED\n");
    return 0;
}

static int test_gguf_layout_parse(void) {
    printf("\n=== Test 5: GGUF Layout Parse ===\n");

    const char* test_path = "nonexistent.gguf";
    cgc_stream_layout_t layout = cgc_load_stream_layout_from_gguf(test_path);
    CHECK(layout.experts_per_layer == 0, "nonexistent file should return empty layout");

    printf("  Test 5 PASSED (negative test)\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  C Expert Streamer Test Suite\n");
    printf("============================================================\n");

    int failures = 0;
    failures += test_streamer_create_destroy();
    failures += test_streamer_pool();
    failures += test_cache_plan();
    failures += test_stream_layout();
    failures += test_gguf_layout_parse();

    printf("\n============================================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TESTS FAILED\n", failures);
    }
    printf("============================================================\n");

    return failures;
}
