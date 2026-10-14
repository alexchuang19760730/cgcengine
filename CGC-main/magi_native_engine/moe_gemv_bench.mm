// moe_gemv_bench.mm — Metal 真实算量 compute_ms 基准驱动（P2 compute 段真实化）
// 用法: ./moe_gemv_bench [--layers 41] [--N 8] [--experts 256] [--topk 8]
//                       [--dim 2048] [--inter 512] [--rounds 3] [--metal-src moe_router.metal]
// 输出 JSON：compute_ms/layer（真实 Metal 实测，喂 P1b --compute-ms / 白皮书 G1）
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "moe_gemv_metal.h"

static double _now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, const char* argv[]) {
    int layers = MOE_T_LAYERS, N = 8, experts = MOE_T_EXPERTS, topk = MOE_T_TOP_K;
    int dim = MOE_T_DIM, inter = MOE_T_INTER, rounds = 3;
    const char* src = "moe_router.metal";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inter") && i + 1 < argc) inter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--metal-src") && i + 1 < argc) src = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("用法: %s [--layers 41] [--N 8] [--experts 256] [--topk 8] [--dim 2048] [--inter 512] [--rounds 3] [--metal-src moe_router.metal]\n", argv[0]);
            return 0;
        }
    }

    moe_metal_ctx_t* c = moe_metal_create(src, layers, N, experts, topk, dim, inter);
    if (!c) { fprintf(stderr, "[ERROR] Metal 上下文创建失败\n"); return 1; }

    double wall0 = _now_ms();
    moe_metal_result_t avg;
    double mean = moe_metal_bench(c, rounds, &avg);
    double wall1 = _now_ms();
    if (mean < 0) { fprintf(stderr, "[ERROR] bench 失败\n"); moe_metal_destroy(c); return 1; }

    double per_layer_ms = avg.total_ms / layers;
    printf("{\n");
    printf("  \"device\": \"%s\",\n", "Metal");
    printf("  \"layers\": %d, \"n_tokens\": %d, \"experts\": %d, \"topk\": %d, \"dim\": %d, \"inter\": %d,\n",
           layers, N, experts, topk, dim, inter);
    printf("  \"rounds\": %d,\n", rounds);
    printf("  \"wall_ms\": %.3f,\n", wall1 - wall0);
    printf("  \"avg_total_ms_per_round\": %.3f,\n", avg.total_ms);
    printf("  \"avg_router_ms\": %.3f,\n", avg.batch_router_ms);
    printf("  \"avg_topk_select_ms\": %.3f,\n", avg.topk_select_ms);
    printf("  \"avg_gemv_ms\": %.3f,\n", avg.gemv_ms);
    printf("  \"compute_ms_per_layer\": %.4f,\n", per_layer_ms);
    printf("  \"compute_ms_whole_model\": %.4f,\n", avg.total_ms);
    printf("  \"note\": \"真实 Metal kernel 实测（合成权重，算量只依 shape）；喂 P1b --compute-ms\"\n");
    printf("}\n");

    fprintf(stderr, "[bench] device=Metal layers=%d N=%d rounds=%d\n", layers, N, rounds);
    fprintf(stderr, "[bench] avg_total=%.3f ms/round · router=%.3f · topk=%.3f · gemv=%.3f\n",
            avg.total_ms, avg.batch_router_ms, avg.topk_select_ms, avg.gemv_ms);
    fprintf(stderr, "[bench] compute_ms/layer=%.4f ms → P1b --compute-ms %.4f\n", per_layer_ms, per_layer_ms * 41);

    moe_metal_destroy(c);
    return 0;
}
