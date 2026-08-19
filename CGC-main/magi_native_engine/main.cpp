#include <stdio.h>
#include <stdlib.h>
#include "metal_runtime.h"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "用法：%s <gguf_path> [iterations]\n", argv[0]);
        fprintf(stderr, "例如：%s qwen2.5-7b-q4_k_m.gguf 10\n", argv[0]);
        return 1;
    }

    int n_iter = (argc > 2) ? atoi(argv[2]) : 10;

    static ModelConfig cfg = {
        .n_layer=28,
        .dim=3584,
        .n_head=14,
        .n_kv_head=2,
        .head_dim=256,
        .vocab_size=151936,
        .max_seq=2048
    };

    metal_device_t* dev = metal_device_create();
    if (!dev) {
        fprintf(stderr, "[ERROR] 無法創建 Metal 設備\n");
        return 1;
    }

    fprintf(stdout, "    [Metal] 設備: %s\n", metal_get_device_name());
    fflush(stdout);

    ModelWeights* weights = metal_load_gguf_weights(dev, argv[1], &cfg);
    if (!weights) {
        fprintf(stderr, "[ERROR] GGUF 權重載入失敗\n");
        metal_device_destroy(dev);
        return 1;
    }

    fprintf(stdout, "    [Metal] ✅ 權重載入成功\n");
    fflush(stdout);

    // Run benchmark
    // (This will be added later when we expose kda benchmark API)
    
    metal_weights_destroy(weights);
    metal_device_destroy(dev);

    return 0;
}