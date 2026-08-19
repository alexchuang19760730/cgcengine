#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "cgc_expert_streamer.h"
#include "cgc_gguf_lite.h"
#include "cgc_pd_scheduler.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("TEST: %-50s ", name); \
    fflush(stdout); \
} while(0)

#define CHECK(cond) do { \
    if (cond) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL (line %d)\n", __LINE__); return 1; } \
} while(0)

int test_streamer_create_destroy(void) {
    TEST("streamer create/destroy");
    
    const char* test_file = "test_stream.bin";
    FILE* f = fopen(test_file, "wb");
    if (f) {
        char buf[4096] = {0};
        fwrite(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strcpy(layout.path, test_file);
    layout.stream_offset = 0;
    layout.stream_size = 4096;
    layout.experts_per_layer = 4;
    layout.expert_stride = 1024;
    
    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 2, false, NULL, 0);
    CHECK(s != NULL);
    CHECK(s->slot_count == 2);
    
    cgc_expert_streamer_destroy(s);
    remove(test_file);
    return 0;
}

int test_pool_create_destroy(void) {
    TEST("pool create/destroy");
    
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL);
    CHECK(pool->count == 0);
    
    cgc_streamer_pool_destroy(pool);
    return 0;
}

int test_pool_add_get(void) {
    TEST("pool add/get streamer");
    
    const char* test_file = "test.bin";
    FILE* f = fopen(test_file, "wb");
    if (f) {
        char buf[4096] = {0};
        fwrite(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strcpy(layout.path, test_file);
    layout.stream_offset = 0;
    layout.experts_per_layer = 4;
    layout.expert_stride = 256;
    
    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 2, false, NULL, 0);
    CHECK(s != NULL);
    
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL);
    
    bool ok = cgc_streamer_pool_add(pool, 5, s);
    CHECK(ok == true);
    CHECK(pool->count == 1);
    
    cgc_expert_streamer_t* found = cgc_streamer_pool_get(pool, 5);
    CHECK(found == s);
    
    cgc_expert_streamer_t* not_found = cgc_streamer_pool_get(pool, 99);
    CHECK(not_found == NULL);
    
    cgc_streamer_pool_destroy(pool);
    cgc_expert_streamer_destroy(s);
    remove(test_file);
    return 0;
}

int test_layout_offset(void) {
    TEST("layout offset computation");
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    layout.stream_offset = 1000;
    layout.expert_stride = 256;
    layout.experts_per_layer = 4;
    layout.has_explicit_offsets = false;
    
    uint64_t off0 = cgc_stream_layout_compute_offset(&layout, 0, 0);
    uint64_t off1 = cgc_stream_layout_compute_offset(&layout, 0, 1);
    uint64_t off2 = cgc_stream_layout_compute_offset(&layout, 0, 2);
    
    CHECK(off0 == 1000);
    CHECK(off1 == 1000 + 256);
    CHECK(off2 == 1000 + 512);
    
    return 0;
}

int test_layout_layer_offset(void) {
    TEST("layout layer offset computation");
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    layout.stream_offset = 1000;
    layout.expert_stride = 256;
    layout.experts_per_layer = 4;
    layout.has_explicit_offsets = false;
    
    uint64_t off_l1_e0 = cgc_stream_layout_compute_offset(&layout, 1, 0);
    uint64_t off_l1_e1 = cgc_stream_layout_compute_offset(&layout, 1, 1);
    
    uint64_t per_layer = (uint64_t)4 * 256;
    CHECK(off_l1_e0 == 1000 + per_layer);
    CHECK(off_l1_e1 == 1000 + per_layer + 256);
    
    return 0;
}

int test_layout_explicit_offsets(void) {
    TEST("layout explicit offsets (layer 0)");
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    layout.stream_offset = 0;
    layout.expert_stride = 0;
    layout.experts_per_layer = 3;
    layout.has_explicit_offsets = true;
    layout.expert_offsets[0] = 100;
    layout.expert_offsets[1] = 300;
    layout.expert_offsets[2] = 600;
    
    uint64_t off0 = cgc_stream_layout_compute_offset(&layout, 0, 0);
    uint64_t off1 = cgc_stream_layout_compute_offset(&layout, 0, 1);
    uint64_t off2 = cgc_stream_layout_compute_offset(&layout, 0, 2);
    
    CHECK(off0 == 100);
    CHECK(off1 == 300);
    CHECK(off2 == 600);
    
    return 0;
}

int test_telemetry(void) {
    TEST("streamer telemetry");
    
    const char* test_file = "test_tele.bin";
    FILE* f = fopen(test_file, "wb");
    if (f) {
        char buf[4096] = {0};
        fwrite(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strcpy(layout.path, test_file);
    layout.experts_per_layer = 4;
    layout.expert_stride = 1024;
    
    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 2, false, NULL, 0);
    CHECK(s != NULL);
    
    cgc_cache_telemetry_t t = cgc_expert_streamer_telemetry(s);
    CHECK(t.slot_count == 2);
    CHECK(t.total_hits == 0);
    CHECK(t.total_misses == 0);
    
    cgc_expert_streamer_destroy(s);
    remove(test_file);
    return 0;
}

int test_scheduler_create_destroy(void) {
    TEST("scheduler create/destroy");
    
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL);
    
    cgc_pd_layer_assignment_t assignment;
    memset(&assignment, 0, sizeof(assignment));
    assignment.prefill_count = 6;
    assignment.decode_count = 6;
    assignment.prefill_gpu = 0;
    assignment.decode_gpu = 1;
    for (int i = 0; i < 6; i++) {
        assignment.prefill_layers[i] = i;
        assignment.decode_layers[i] = i + 6;
    }
    
    cgc_pd_scheduler_t* s = cgc_pd_scheduler_create(pool, &assignment, 8, 4);
    CHECK(s != NULL);
    CHECK(s->max_experts_per_layer == 8);
    CHECK(s->tile_experts == 4);
    
    cgc_pd_scheduler_destroy(s);
    cgc_streamer_pool_destroy(pool);
    return 0;
}

int test_scheduler_prefill_decode(void) {
    TEST("scheduler prefill/decode switch");
    
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    CHECK(pool != NULL);
    
    cgc_pd_layer_assignment_t assignment;
    memset(&assignment, 0, sizeof(assignment));
    assignment.prefill_count = 4;
    assignment.decode_count = 4;
    assignment.prefill_gpu = 0;
    assignment.decode_gpu = 1;
    for (int i = 0; i < 4; i++) {
        assignment.prefill_layers[i] = i;
        assignment.decode_layers[i] = i + 4;
    }
    
    cgc_pd_scheduler_t* s = cgc_pd_scheduler_create(pool, &assignment, 8, 4);
    CHECK(s != NULL);
    
    cgc_pd_scheduler_enter_prefill(s);
    CHECK(cgc_pd_scheduler_current_phase(s) == CGC_PD_PHASE_PREFILL);
    
    cgc_pd_scheduler_switch_to_decode(s);
    CHECK(cgc_pd_scheduler_current_phase(s) == CGC_PD_PHASE_DECODE);
    
    cgc_pd_scheduler_destroy(s);
    cgc_streamer_pool_destroy(pool);
    return 0;
}

int test_scheduler_layer_assignment(void) {
    TEST("layer assignment by ratio");
    
    cgc_pd_layer_assignment_t a = cgc_pd_layer_assignment_by_ratio(12, 0.5);
    CHECK(a.prefill_count + a.decode_count == 12);
    CHECK(a.prefill_count >= 5);
    CHECK(a.decode_count >= 5);
    
    bool is_pf_0 = cgc_pd_is_prefill_layer(&a, 0);
    bool is_dec_0 = cgc_pd_is_decode_layer(&a, 0);
    CHECK(is_pf_0 == true);
    CHECK(is_dec_0 == false);
    
    int last_decode_layer = a.prefill_count + a.decode_count - 1;
    bool is_dec_last = cgc_pd_is_decode_layer(&a, last_decode_layer);
    CHECK(is_dec_last == true);
    
    int device = cgc_pd_get_device_for_layer(&a, 0);
    CHECK(device == a.prefill_gpu);
    
    return 0;
}

int test_scheduler_stats(void) {
    TEST("scheduler stats");
    
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();
    cgc_pd_layer_assignment_t assignment;
    memset(&assignment, 0, sizeof(assignment));
    assignment.prefill_count = 2;
    assignment.decode_count = 2;
    for (int i = 0; i < 2; i++) {
        assignment.prefill_layers[i] = i;
        assignment.decode_layers[i] = i + 2;
    }
    
    cgc_pd_scheduler_t* s = cgc_pd_scheduler_create(pool, &assignment, 8, 4);
    CHECK(s != NULL);
    
    cgc_pd_scheduler_stats_t stats = cgc_pd_scheduler_get_stats(s);
    CHECK(stats.prefill_tokens == 0);
    CHECK(stats.decode_tokens == 0);
    
    cgc_pd_scheduler_reset_stats(s);
    stats = cgc_pd_scheduler_get_stats(s);
    CHECK(stats.prefill_tokens == 0);
    
    cgc_pd_scheduler_destroy(s);
    cgc_streamer_pool_destroy(pool);
    return 0;
}

int test_gguf_lite_context(void) {
    TEST("gguf_lite context (no file I/O)");
    
    cgc_gguf_tensor_info_t info;
    memset(&info, 0, sizeof(info));
    info.n_dims = 2;
    info.dims[0] = 1024;
    info.dims[1] = 2048;
    info.type = CGC_GGML_TYPE_F16;
    info.offset = 0;
    info.n_elements = 1024 * 2048;
    
    CHECK(info.n_dims == 2);
    CHECK(info.dims[0] == 1024);
    CHECK(info.type == CGC_GGML_TYPE_F16);
    
    return 0;
}

int test_gguf_type_bytes(void) {
    TEST("ggml type bytes per element");
    
    double f32_bytes = cgc_ggml_type_bytes_per_elem(CGC_GGML_TYPE_F32);
    CHECK(f32_bytes == 4.0);
    
    double f16_bytes = cgc_ggml_type_bytes_per_elem(CGC_GGML_TYPE_F16);
    CHECK(f16_bytes == 2.0);
    
    double bf16_bytes = cgc_ggml_type_bytes_per_elem(CGC_GGML_TYPE_BF16);
    CHECK(bf16_bytes == 2.0);
    
    double q4_bytes = cgc_ggml_type_bytes_per_elem(CGC_GGML_TYPE_Q4_K);
    CHECK(q4_bytes > 0.0);
    
    return 0;
}

int main(void) {
    printf("=== CGC Expert Streaming Minimal Tests ===\n\n");
    
    int failures = 0;
    failures += test_streamer_create_destroy();
    failures += test_pool_create_destroy();
    failures += test_pool_add_get();
    failures += test_layout_offset();
    failures += test_layout_layer_offset();
    failures += test_layout_explicit_offsets();
    failures += test_telemetry();
    failures += test_scheduler_create_destroy();
    failures += test_scheduler_prefill_decode();
    failures += test_scheduler_layer_assignment();
    failures += test_scheduler_stats();
    failures += test_gguf_lite_context();
    failures += test_gguf_type_bytes();
    
    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (failures == 0) printf(" ✅ ===\n");
    else printf(" ❌ (%d failures) ===\n", failures);
    
    return failures;
}