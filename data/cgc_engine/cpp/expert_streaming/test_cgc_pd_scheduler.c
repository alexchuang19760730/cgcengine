#include "cgc_pd_scheduler.h"
#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"

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

static int test_layer_assignment(void) {
    printf("\n=== Test 1: Layer Assignment ===\n");

    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(8, 0.5);

    CHECK(a.prefill_count == 4, "4 prefill layers");
    CHECK(a.decode_count == 4, "4 decode layers");

    for (int i = 0; i < 4; i++) {
        CHECK(a.prefill_layers[i] == i, "prefill layer index");
    }
    for (int i = 0; i < 4; i++) {
        CHECK(a.decode_layers[i] == 4 + i, "decode layer index");
    }

    CHECK(cgc_pd_is_prefill_layer(&a, 0) == true, "layer 0 is prefill");
    CHECK(cgc_pd_is_prefill_layer(&a, 3) == true, "layer 3 is prefill");
    CHECK(cgc_pd_is_decode_layer(&a, 4) == true, "layer 4 is decode");
    CHECK(cgc_pd_is_decode_layer(&a, 7) == true, "layer 7 is decode");
    CHECK(cgc_pd_is_prefill_layer(&a, 4) == false, "layer 4 not prefill");

    CHECK(cgc_pd_get_device_for_layer(&a, 0) == 0, "layer 0 on GPU 0");
    CHECK(cgc_pd_get_device_for_layer(&a, 4) == 1, "layer 4 on GPU 1");

    cgc_pd_layer_assignment_t custom = cgc_pd_layer_assignment_custom(
        (int[]){0, 1, 2}, 3,
        (int[]){3, 4, 5, 6, 7}, 5
    );
    CHECK(custom.prefill_count == 3, "custom prefill count");
    CHECK(custom.decode_count == 5, "custom decode count");

    printf("  Test 1 PASSED\n");
    return 0;
}

static int test_token_routes(void) {
    printf("\n=== Test 2: Token Routes ===\n");

    cgc_pd_token_routes_t routes;
    memset(&routes, 0, sizeof(routes));
    routes.token_index = 0;
    routes.route_count = 2;

    routes.routes[0].layer = 0;
    routes.routes[0].expert_count = 4;
    for (int i = 0; i < 4; i++) {
        routes.routes[0].expert_ids[i] = i;
    }

    routes.routes[1].layer = 4;
    routes.routes[1].expert_count = 3;
    for (int i = 0; i < 3; i++) {
        routes.routes[1].expert_ids[i] = i + 10;
    }

    CHECK(routes.routes[0].layer == 0, "route layer 0");
    CHECK(routes.routes[0].expert_ids[2] == 2, "expert id 2");
    CHECK(routes.routes[1].layer == 4, "route layer 4");
    CHECK(routes.routes[1].expert_ids[0] == 10, "expert id 10");

    printf("  Test 2 PASSED\n");
    return 0;
}

static int test_scheduler_create_destroy(void) {
    printf("\n=== Test 3: Scheduler create/destroy ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(4, 0.5);

    cgc_pd_scheduler_t* sched = cgc_pd_scheduler_create(pool, &a, 8, 8);
    CHECK(sched != NULL, "scheduler creation");
    CHECK(sched->current_phase == CGC_PD_PHASE_IDLE, "initial phase is idle");
    CHECK(sched->assignment.prefill_count == 2, "prefill count");
    CHECK(sched->assignment.decode_count == 2, "decode count");
    CHECK(sched->max_experts_per_layer == 8, "max experts per layer");

    CHECK(cgc_pd_scheduler_current_phase(sched) == CGC_PD_PHASE_IDLE, "phase idle");

    cgc_pd_scheduler_destroy(sched);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 3 PASSED\n");
    return 0;
}

static int test_prefill_phase(void) {
    printf("\n=== Test 4: Prefill Phase ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(4, 0.5);

    cgc_pd_scheduler_t* sched = cgc_pd_scheduler_create(pool, &a, 8, 8);

    cgc_pd_scheduler_enter_prefill(sched);
    CHECK(cgc_pd_scheduler_current_phase(sched) == CGC_PD_PHASE_PREFILL, "phase prefill");

    cgc_pd_token_routes_t routes;
    memset(&routes, 0, sizeof(routes));
    routes.token_index = 0;
    routes.route_count = 1;
    routes.routes[0].layer = 0;
    routes.routes[0].expert_count = 4;
    for (int i = 0; i < 4; i++) {
        routes.routes[0].expert_ids[i] = i;
    }

    cgc_pd_tile_t tiles[8];
    int tile_count = cgc_pd_scheduler_process_prefill(sched, &routes, 1, tiles, 8);
    CHECK(tile_count >= 0, "process prefill returns valid count");

    cgc_pd_scheduler_record_routes(sched, &routes);

    cgc_pd_scheduler_destroy(sched);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 4 PASSED\n");
    return 0;
}

static int test_switch_to_decode(void) {
    printf("\n=== Test 5: Switch to Decode ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(4, 0.5);

    cgc_pd_scheduler_t* sched = cgc_pd_scheduler_create(pool, &a, 8, 8);

    cgc_pd_scheduler_enter_prefill(sched);

    cgc_pd_scheduler_switch_to_decode(sched);
    CHECK(cgc_pd_scheduler_current_phase(sched) == CGC_PD_PHASE_DECODE, "phase decode");

    cgc_pd_token_routes_t routes;
    memset(&routes, 0, sizeof(routes));
    routes.token_index = 1;
    routes.route_count = 1;
    routes.routes[0].layer = 2;
    routes.routes[0].expert_count = 4;
    for (int i = 0; i < 4; i++) {
        routes.routes[0].expert_ids[i] = i + 5;
    }

    cgc_pd_tile_t tiles[8];
    int tile_count = cgc_pd_scheduler_process_decode(sched, &routes, tiles, 8);
    CHECK(tile_count >= 0, "process decode returns valid count");

    cgc_cache_result_t result = cgc_pd_scheduler_load_decode_experts(
        sched, 2, (int[]){5, 6}, 2);
    CHECK(result.count >= 0, "load decode experts returns valid result");

    cgc_pd_scheduler_destroy(sched);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 5 PASSED\n");
    return 0;
}

static int test_stats(void) {
    printf("\n=== Test 6: Stats ===\n");

    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(4, 0.5);

    cgc_pd_scheduler_t* sched = cgc_pd_scheduler_create(pool, &a, 8, 8);

    cgc_pd_scheduler_enter_prefill(sched);

    cgc_pd_scheduler_stats_t stats = cgc_pd_scheduler_get_stats(sched);
    CHECK(stats.phase == CGC_PD_PHASE_PREFILL, "stats phase");
    CHECK(stats.prefill_tokens == 0, "initial prefill tokens");

    cgc_pd_scheduler_reset_stats(sched);
    stats = cgc_pd_scheduler_get_stats(sched);
    CHECK(stats.prefill_tokens == 0, "reset tokens");

    cgc_pd_scheduler_destroy(sched);
    cgc_streamer_pool_destroy(pool);

    printf("  Test 6 PASSED\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  C PD Scheduler Test Suite\n");
    printf("============================================================\n");

    int failures = 0;
    failures += test_layer_assignment();
    failures += test_token_routes();
    failures += test_scheduler_create_destroy();
    failures += test_prefill_phase();
    failures += test_switch_to_decode();
    failures += test_stats();

    printf("\n============================================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TESTS FAILED\n", failures);
    }
    printf("============================================================\n");

    return failures;
}
