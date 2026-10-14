/*
 * moe_profile_main.c — P2 四接口桩驱动
 *
 * 跑一轮 MTP 验证（N=8 draft，全部 41 层）：每层 batch_router → Union →
 * async_prefetch → verify_batch，输出四段时间（compute/RAM/SSD/serial）+
 * 页命中率 + 平均 Union，JSON 落盘（喂 P1b --compute-ms / 带宽模型）。
 *
 * 用法：
 *   make moe_profile
 *   ./moe_profile <gguf> <expert_manifest.bin> [--resident 0.3] [--rounds 5] [--N 8]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "moe_inject.h"

#define MAX_N 16
#define DIM 2048

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    const char* gguf = argc > 1 ? argv[1] : NULL;
    const char* manifest = argc > 2 ? argv[2] : NULL;
    double resident = 0.3;
    int rounds = 5, N = 8;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--resident") && i + 1 < argc) resident = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
    }
    if (!gguf || !manifest) {
        fprintf(stderr, "用法: %s <gguf> <manifest.bin> [--resident h] [--rounds R] [--N 8]\n", argv[0]);
        return 2;
    }
    if (N > MAX_N) N = MAX_N;

    moe_pool_cfg_t cfg = {
        .n_layer = 41, .experts_per_layer = 256, .top_k = 8,
        .n_prefetch_threads = 2, .resident_ratio = resident,
        .gguf_path = gguf, .manifest_path = manifest,
    };
    moe_pool_t* pool = moe_pool_create(&cfg);
    if (!pool) return 1;

    float* hidden = malloc(N * DIM * sizeof(float));
    for (int i = 0; i < N * DIM; i++) hidden[i] = ((i * 2654435761u) % 1000) / 1000.0f - 0.5f; /* 确定性 seed=42 风格 */
    float* out_hidden = malloc(N * DIM * sizeof(float));
    int union_set[512];

    printf("== P2 四接口桩 · MTP 验证轮（N=%d, resident=%.2f, rounds=%d）==\n", N, resident, rounds);
    double t_all0 = now_ms();
    for (int r = 0; r < rounds; r++) {
        double t_round = now_ms();
        double compute = 0, ram = 0, ssd_issue = 0;
        int hits = 0, misses = 0, u_sum = 0;
        for (int layer = 0; layer < cfg.n_layer; layer++) {
            int u;
            double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
            moe_batch_router(pool, hidden, N, layer, union_set, NULL, 512, &u, &t1);
            compute += t1; u_sum += u;
            int miss_pages = 0;
            moe_async_prefetch(pool, layer, union_set, u, &t3, &miss_pages);
            ssd_issue += t3; misses += miss_pages;
            /* 等预取完成：simulate serial 等待（worker 队列消化） */
            struct timespec req = {0, 2000000L};
            nanosleep(&req, NULL);
            moe_verify_batch(pool, hidden, N, layer, union_set, u, out_hidden, &t2);
            ram += t2;
            /* 命中 = Union 中 resident 页部分（桩：prefetch 已把 miss 页标记 resident，
             * 故第二轮起全命中——真实命中率以引擎 LRU 为准，桩只测时序） */
        }
        printf("  round %d: wall=%.2f ms · compute=%.2f · RAM=%.2f · SSD issue=%.2f · U_avg=%.1f\n",
               r + 1, now_ms() - t_round, compute, ram, ssd_issue, (double)u_sum / cfg.n_layer);
    }
    printf("总 wall=%.2f ms\n", now_ms() - t_all0);

    moe_profile_print(pool);
    moe_profile_dump_json(pool, "moe_profile_out.json");
    printf("compute_ms 均值（喂 P1b --compute-ms）= %.3f ms\n", moe_compute_ms_get(pool));
    printf("→ moe_profile_out.json 已写\n");

    moe_pool_destroy(pool);
    free(hidden); free(out_hidden);
    return 0;
}
