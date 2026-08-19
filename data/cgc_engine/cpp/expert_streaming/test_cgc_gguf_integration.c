#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"
#include "cgc_expert_compute.h"
#include "cgc_gguf_lite.h"

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

static int test_compute_bridge_init(void) {
    printf("\n=== Test 1: Compute Bridge Init ===\n");

    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));
    strncpy(layout.path, "nonexistent.gguf", CGC_MAX_PATH_LEN - 1);
    layout.experts_per_layer = 8;
    layout.expert_stride = 1024;

    cgc_expert_streamer_t* streamer = cgc_expert_streamer_create(&layout, 4, false, NULL, 0);
    if (streamer) {
        cgc_expert_compute_bridge_t bridge;
        cgc_compute_bridge_init(&bridge, streamer, NULL);
        CHECK(bridge.streamer == streamer, "bridge streamer set");
        CHECK(bridge.gguf_ctx == NULL, "bridge gguf ctx is null");

        cgc_expert_weights_view_t views[4];
        int count = cgc_compute_bridge_load_weights(&bridge, (int[]){0}, 1, views);
        CHECK(count == 1, "load weights count");

        cgc_expert_tensor_info_t tensor_info[8];
        int info_count = cgc_compute_bridge_get_tensor_info(&bridge, 0, tensor_info, 8);
        CHECK(info_count == 0, "no tensor info without gguf ctx");

        cgc_layer_gguf_meta_t meta = cgc_compute_bridge_get_meta(&bridge);
        CHECK(meta.layer_index == 0, "meta layer index default");

        cgc_expert_streamer_destroy(streamer);
    }

    printf("  Test 1 PASSED\n");
    return 0;
}

static int test_sub_tensor_layout(void) {
    printf("\n=== Test 2: Sub Tensor Layout ===\n");

    cgc_sub_tensor_view_t view;
    memset(&view, 0, sizeof(view));
    view.data = (void*)0x1000;
    view.shape[0] = 2816;
    view.shape[1] = 704;
    view.ggml_type = 30;
    view.offset_in_buffer = 0;
    view.size_bytes = 2816 * 704 * 2;

    CHECK(view.data == (void*)0x1000, "view data");
    CHECK(view.shape[0] == 2816, "view shape dim0");
    CHECK(view.shape[1] == 704, "view shape dim1");
    CHECK(view.ggml_type == 30, "view type BF16");
    CHECK(view.size_bytes == 2816 * 704 * 2, "view size");

    cgc_expert_weights_view_t weights_view;
    memset(&weights_view, 0, sizeof(weights_view));
    weights_view.expert_id = 0;
    weights_view.gate = view;
    weights_view.up = view;
    weights_view.down = view;
    weights_view.raw_buffer = (void*)0x2000;
    weights_view.raw_size = 2816 * 704 * 2 * 3;

    CHECK(weights_view.expert_id == 0, "weights view expert id");
    CHECK(weights_view.raw_buffer == (void*)0x2000, "weights view raw buffer");
    CHECK(weights_view.gate.size_bytes > 0, "gate size");

    printf("  Test 2 PASSED\n");
    return 0;
}

static int test_layer_meta(void) {
    printf("\n=== Test 3: Layer GGUF Meta ===\n");

    cgc_layer_gguf_meta_t meta;
    memset(&meta, 0, sizeof(meta));
    meta.layer_index = 5;
    meta.experts_per_layer = 128;
    meta.expert_stride = 1073741824;
    meta.hidden_size = 2048;
    meta.moe_intermediate_size = 704;
    strncpy(meta.quantization, "IQ3_M", sizeof(meta.quantization) - 1);

    CHECK(meta.layer_index == 5, "meta layer index");
    CHECK(meta.experts_per_layer == 128, "meta experts per layer");
    CHECK(meta.hidden_size == 2048, "meta hidden size");
    CHECK(meta.moe_intermediate_size == 704, "meta intermediate size");
    CHECK(strcmp(meta.quantization, "IQ3_M") == 0, "meta quantization");

    printf("  Test 3 PASSED\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  C Expert Compute Bridge Test Suite\n");
    printf("============================================================\n");

    int failures = 0;
    failures += test_compute_bridge_init();
    failures += test_sub_tensor_layout();
    failures += test_layer_meta();

    printf("\n============================================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TESTS FAILED\n", failures);
    }
    printf("============================================================\n");

    return failures;
}
