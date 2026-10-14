// moe_gemv_metal.h — Metal 真实算量 GEMV/Router 计时接口（P2 桩 compute 段真实化）
// 2026-10-14：把 moe_inject.c 的 CPU 确定性伪路由换成真实 Metal kernel
// （moe_router.metal 的 batch_router_kernel + expert_gemv_kernel），
// 得到目标模型算量下的真实 compute_ms（喂 P1b --compute-ms / 白皮书 G1）。
#ifndef MOE_GEMV_METAL_H
#define MOE_GEMV_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moe_metal_ctx moe_metal_ctx_t;

// 目标模型算量常量（GGUF 核准，gguf_verified.json）
#define MOE_T_DIM       2048    /* hidden */
#define MOE_T_INTER     512     /* expert intermediate */
#define MOE_T_EXPERTS   256     /* per-layer experts */
#define MOE_T_TOP_K     8
#define MOE_T_LAYERS    41

// 一次 compute 轮的结果（与 moe_inject 的 compute 段口径对齐）
typedef struct {
    double batch_router_ms;   /* N×256 scores，全层 */
    double topk_select_ms;    /* host top-8 选择 */
    double gemv_ms;           /* top8 × expert_gemv dispatch，全层 */
    double total_ms;          /* 三者之和 */
    int    layers;            /* 实际层数 */
    int    n_tokens;          /* MTP N */
} moe_metal_result_t;

// 创建 Metal 上下文并编译 moe_router.metal 源码。
// metal_src_path 为空时用内置默认源码（同 moe_router.metal 内容）。
moe_metal_ctx_t* moe_metal_create(const char* metal_src_path, int layers,
                                  int n_tokens, int experts, int topk,
                                  int dim, int inter);
void moe_metal_destroy(moe_metal_ctx_t* ctx);

// 跑一轮真实 compute：batch_router(N×E) → host top-k → topk×expert_gemv。
// 权重为随机填充（算量只依 shape），hidden 为随机输入。
int moe_metal_run(moe_metal_ctx_t* ctx, moe_metal_result_t* out);

// 多轮平均；返回平均 total_ms（= compute_ms/轮，喂 P1b --compute-ms）。
double moe_metal_bench(moe_metal_ctx_t* ctx, int rounds, moe_metal_result_t* avg);

#ifdef __cplusplus
}
#endif
#endif /* MOE_GEMV_METAL_H */
