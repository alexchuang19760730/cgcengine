#pragma once
/*
 * moe_inject.h — P2 最小注入接口（CGC MoE Engine · Metal 桩）
 *
 * 四接口语义映射（Union 聚合预取 + Verifier-Side 集合裁剪，白皮书 §2/§3）：
 *   ① moe_load_expert    : 从 GGUF 按 (layer, matrix, expert) 页粒度异步读入
 *                           （对应「Per-Layer LRU + SSD 流式加载」，页 1.7MiB）
 *   ② moe_batch_router   : MTP N 个 draft 每层批量 router → 候选专家集 Union
 *                           （对应「层内批量路由预取」，输出 U 与 top-k 命中）
 *   ③ moe_verify_batch   : 对 Union 专家集批量 GEMV 验证 N 个 token
 *                           （AcceptMoE 式集合裁剪挂点：U_c / P_commit 在此收敛）
 *   ④ moe_async_prefetch : union − resident 差集后台预取，与上层计算重叠
 *                           （隐藏 SSD 延迟；带宽方程仍是硬约束）
 *
 * 桩为最小可编译骨架：CPU 朴素计算 + pthread 异步 pread；Metal kernel
 * （batch_router / expert_gemv）以 moe_router.metal 预留，替换路径见文件注释。
 * 四接口均带纳秒计时，输出 profile JSON 直接喂 P1b（--compute-ms）与带宽模型。
 */
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct MoEExpertPool moe_pool_t;
typedef struct MoEProfile moe_profile_t;

/* 页清单（来自 gen_expert_manifest.py，二进制 packed 32B：<iiiqqi） */
#pragma pack(push, 1)
typedef struct {
    int layer, matrix;      /* matrix: 0=gate 1=up 2=down */
    int page_idx;
    int64_t offset, size;   /* 文件偏移 / 页字节 */
    int n_experts;          /* 页内专家数（近似） */
} moe_page_t;
#pragma pack(pop)

/* 池配置 */
typedef struct {
    int n_layer;
    int experts_per_layer;
    int top_k;
    int n_prefetch_threads;    /* 默认 2 */
    double resident_ratio;     /* 驻留比例 h（0~1），per-layer 预算 = ratio × 层专家字节 */
    const char* gguf_path;     /* 专家权重来源（manifest 页 offset 基于该文件） */
    const char* manifest_path; /* gen_expert_manifest.py 产物 */
} moe_pool_cfg_t;

moe_pool_t* moe_pool_create(const moe_pool_cfg_t* cfg);
void moe_pool_destroy(moe_pool_t* pool);

/* ① 异步读入单个专家（页粒度；已驻留直接命中）。返回 0=命中 1=miss 已入队 */
int moe_load_expert(moe_pool_t* pool, int layer, int matrix, int expert_id,
                    double* t_ms_issue);
/* ② 批量 router：n_tokens 个 hidden → 每层 top-k 专家集（Union 写入 out_union，
 *    返回 Union 大小 U；n_tokens ≤ 16）。router 权重 F32 常驻 2MB/层。 */
int moe_batch_router(moe_pool_t* pool, const float* hidden, int n_tokens,
                     int layer, int* out_union, int* out_probs, int union_cap,
                     int* union_size, double* t_ms);
/* ③ 批量验证：对 expert_set（Union 或裁剪后 U_c）做专家 GEMV，输出累积 hidden */
int moe_verify_batch(moe_pool_t* pool, const float* hidden, int n_tokens,
                     int layer, const int* expert_set, int set_size,
                     float* out_hidden, double* t_ms);
/* ④ 后台预取：union − resident 差集入队，返回本次 miss 页数 */
int moe_async_prefetch(moe_pool_t* pool, int layer, const int* expert_set,
                       int set_size, double* t_ms_issue, int* n_miss_pages);

/* 统计：每层命中率 / 字节 / 四段时间累计；打印 + JSON 落盘 */
moe_profile_t* moe_profile_get(moe_pool_t* pool);
void moe_profile_print(moe_pool_t* pool);
void moe_profile_dump_json(moe_pool_t* pool, const char* path);

/* P1b 注入桥：返回本 pool 实测 compute_ms（router+verify 每轮均值），喂 --compute-ms */
double moe_compute_ms_get(moe_pool_t* pool);

#ifdef __cplusplus
}
#endif
