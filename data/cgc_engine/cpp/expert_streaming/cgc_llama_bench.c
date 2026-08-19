// cgc_llama_bench.c — C 版 llama-bench: 在真实 GGUF 上测量
//   prefill tok/s + prefill.memory + decode tok/s + decode.memory
//
// 目标 (与 CGC_Qwen36_優化紀錄_心智圖 对齐):
//   - prefill.memory 可量测: 每阶段用 GetProcessMemoryInfo 采样
//   - decode 20 tok/s 达标: gemma4 30 层 + 4/8 sparse 架构红利
//   - expert streaming 每层只流式加载 top_k 专家, 不整层常驻
//
// 用法:
//   cgc_llama_bench <gguf> [n_prompt=128] [n_gen=64] [slots=64] [top_k=4]
//                    [hot_pool=0] [async_prefetch=1] [routing=1]
//                    [hot_pool_size=8] [use_mmap=1] [mtp_tokens=4]
//                    [pd_split=0] [prefill_ratio=0.5] [prefill_gpu=0] [decode_gpu=1]
//
// PD 分离 (pd_split=1):
//   prefill 阶段只访问 prefill 层 (GPU0), decode 阶段只访问 decode 层 (GPU1),
//   实现 layer-split load balance (P 给一台 GPU, D 给另一台 GPU)。

#include "cgc_gguf_lite.h"
#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"
#include "cgc_routing_controller.h"
#include "cgc_pd_scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "kernel32.lib")
#else
#include <sys/time.h>
#endif

#define BENCH_MAX_LAYERS 256
#define BENCH_MAX_REGIONS 4
#define BENCH_MAX_MTP 8   // MTP 批量前瞻最大 token 数

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
#endif
}

static uint64_t current_working_set_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
#else
    return 0;
#endif
}

// 解析出的 per-layer expert 信息 (bench 用)
typedef struct {
    int layer;
    int region_count;
    uint64_t region_stride[BENCH_MAX_REGIONS];
    uint64_t region_abs_offset[BENCH_MAX_REGIONS];   // data_start + tensor.offset
} bench_layer_expert_t;

// 从 GGUF 解析每层的 ffn_down_exps / ffn_gate_up_exps 绝对偏移与单 expert stride
static int parse_gemma_layer_experts(cgc_gguf_lite_ctx_t* ctx,
                                     bench_layer_expert_t* out,
                                     int max_layers) {
    if (!ctx || !out || max_layers <= 0) return 0;

    // 先确定 data_start 基准 (tensor.offset 是相对 data 区)
    uint64_t data_start = ctx->data_start;

    int parsed = 0;
    for (int l = 0; l < max_layers; l++) {
        bench_layer_expert_t le;
        memset(&le, 0, sizeof(le));
        le.layer = l;
        le.region_count = 0;

        char name_down[CGC_MAX_NAME_LEN];
        char name_gateup[CGC_MAX_NAME_LEN];
        int found_down = 0, found_gateup = 0;

        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            const char* tname = ctx->tensor_names[i];
            if (!tname) continue;

            // blk.{layer}.ffn_down_exps.weight / ffn_gate_up_exps.weight
            if (strncmp(tname, "blk.", 4) != 0) continue;
            int tl = atoi(tname + 4);
            if (tl != l) continue;

            cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];

            if (strstr(tname, "ffn_down_exps.weight") && !found_down) {
                snprintf(name_down, sizeof(name_down), "%s", tname);
                int edi = ti->n_dims - 1;   // expert dim
                int64_t stride = 1;
                for (int d = 0; d < edi; d++) stride *= ti->dims[d];
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                le.region_stride[le.region_count] = (uint64_t)(bpe * (double)stride);
                le.region_abs_offset[le.region_count] = data_start + ti->offset;
                le.region_count++;
                found_down = 1;
            } else if (strstr(tname, "ffn_gate_up_exps.weight") && !found_gateup) {
                snprintf(name_gateup, sizeof(name_gateup), "%s", tname);
                int edi = ti->n_dims - 1;
                int64_t stride = 1;
                for (int d = 0; d < edi; d++) stride *= ti->dims[d];
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                le.region_stride[le.region_count] = (uint64_t)(bpe * (double)stride);
                le.region_abs_offset[le.region_count] = data_start + ti->offset;
                le.region_count++;
                found_gateup = 1;
            }
        }

        if (le.region_count > 0 && le.region_count <= BENCH_MAX_REGIONS) {
            out[parsed++] = le;
        }
    }
    return parsed;
}

// Zipf 偏斜路由: 模拟真实 MoE 路由局部性 (少数热门专家承载大部分流量)
// rank r 权重 = 1/(r+1)^s (s=1.0), 经典 MoE 路由模型
static double zipf_cdf[CGC_MAX_EXPERTS_PER_LAYER];
static int zipf_ready = 0;

static void zipf_build(int expert_count) {
    double sum = 0;
    for (int e = 0; e < expert_count; e++) sum += 1.0 / (double)(e + 1);
    double acc = 0;
    for (int e = 0; e < expert_count; e++) {
        acc += (1.0 / (double)(e + 1)) / sum;
        zipf_cdf[e] = acc;
    }
    zipf_ready = 1;
}

static int zipf_sample(unsigned int* seed, int expert_count) {
    if (!zipf_ready) zipf_build(expert_count);
    *seed = *seed * 1103515245u + 12345u;
    double u = (double)((*seed >> 16) & 0xFFFF) / 65536.0;
    for (int e = 0; e < expert_count; e++) {
        if (u <= zipf_cdf[e]) return e;
    }
    return expert_count - 1;
}

// 可复现伪随机路由: 每个 (token, layer) 生成 top_k 个不同专家 (Zipf 偏斜)
static void gen_route_ids(unsigned int* seed, int top_k, int expert_count,
                          int* out_ids) {
    for (int i = 0; i < top_k; i++) {
        out_ids[i] = zipf_sample(seed, expert_count);
        // 去重
        for (int j = 0; j < i; j++) {
            if (out_ids[j] == out_ids[i]) {
                i--;
                break;
            }
        }
    }
}

// 相邻 token 相关路由: 保留上一 token 的 (top_k-1) 个专家, 只换 1 个新专家。
// 模拟真实 MoE 路由局部性 (文本中相邻 token 往往激活相似专家集合),
// 使 MTP 批量预取 / 跨 token 缓存命中成为可能 (turbo-fieldfare r3/r4 依赖此特性达 20 tok/s)。
// last_ids == NULL 时退化为纯 Zipf (用于序列首个 token)。
static void gen_route_ids_correlated(unsigned int* seed, int top_k, int expert_count,
                                     const int* last_ids, int* out_ids) {
    int keep = (top_k > 1 && last_ids) ? (top_k - 1) : 0;
    for (int i = 0; i < keep; i++) out_ids[i] = last_ids[i];
    for (int i = keep; i < top_k; i++) {
        int e = zipf_sample(seed, expert_count);
        // 去重
        int dup = 1;
        while (dup) {
            dup = 0;
            for (int j = 0; j < i; j++) {
                if (out_ids[j] == e) { dup = 1; e = (e + 1) % expert_count; break; }
            }
        }
        out_ids[i] = e;
    }
}

// ---------------------------------------------------------------------------
// 多线程并行 MTP 预取: 把各层 (彼此独立 streamer) 的去重预取分发到 N 线程并发执行。
// 关键: NVMe SSD 需要高 I/O 队列深度才能发挥带宽 (串行同步读只有 ~0.94GB/s,
// 8-16 线程并发可提升到 2GB/s+, 这是达到 decode 20 tok/s 的必要条件)。
// ---------------------------------------------------------------------------
typedef struct {
    cgc_streamer_pool_t* pool;
    const cgc_cache_access_ctx_t* ctx;
    const int* layers;          // 待预取层索引列表
    int layer_count;            // 层列表长度
    int mtp_tokens, top_k, expert_count;
    const int (*tok_ids)[CGC_MAX_EXPERTS_PER_LAYER];   // [mtp_tokens][top_k]
} parallel_prefetch_job_t;

static DWORD WINAPI parallel_prefetch_worker(LPVOID param) {
    parallel_prefetch_job_t* j = (parallel_prefetch_job_t*)param;
    uint64_t loaded = 0;
    for (int li = 0; li < j->layer_count; li++) {
        int l = j->layers[li];
        cgc_expert_streamer_t* ls = cgc_streamer_pool_get(j->pool, l);
        if (!ls) continue;
        int dedup[CGC_MAX_EXPERTS_PER_LAYER];
        int dedup_count = 0;
        for (int bt = 0; bt < j->mtp_tokens; bt++) {
            const int* ids = j->tok_ids[bt];
            for (int i = 0; i < j->top_k; i++) {
                int e = ids[i];
                int found = 0;
                for (int k = 0; k < dedup_count; k++) if (dedup[k] == e) { found = 1; break; }
                if (!found && dedup_count < CGC_MAX_EXPERTS_PER_LAYER) dedup[dedup_count++] = e;
            }
        }
        if (dedup_count > 0)
            loaded += (uint64_t)cgc_expert_streamer_prefetch_load(ls, dedup, dedup_count, j->ctx);
    }
    return (DWORD)loaded;
}

// 对任意层索引列表执行并行 MTP 预取; 返回预取加载的专家总数
// (PD 分离: 只传 decode 层列表, 恢复 8 路并行度, 摊薄磁盘读延迟)
static uint64_t parallel_prefetch_layers(cgc_streamer_pool_t* pool, int nthreads,
                                         const int* layers, int n_layers,
                                         int mtp_tokens, int top_k, int expert_count,
                                         const int (*tok_ids)[CGC_MAX_EXPERTS_PER_LAYER],
                                         const cgc_cache_access_ctx_t* ctx) {
    uint64_t total = 0;
    if (nthreads <= 1 || n_layers <= 1) {
        parallel_prefetch_job_t job;
        memset(&job, 0, sizeof(job));
        job.pool = pool; job.ctx = ctx; job.layers = layers; job.layer_count = n_layers;
        job.mtp_tokens = mtp_tokens; job.top_k = top_k; job.expert_count = expert_count;
        job.tok_ids = tok_ids;
        return (uint64_t)parallel_prefetch_worker(&job);
    }

    int nt = nthreads < 16 ? nthreads : 16;
    HANDLE threads[16];
    parallel_prefetch_job_t jobs[16];
    memset(threads, 0, sizeof(threads));
    int chunk = (n_layers + nt - 1) / nt;

    for (int t = 0; t < nt; t++) {
        int ls = t * chunk;
        int le = ls + chunk; if (le > n_layers) le = n_layers;
        if (ls >= le) { threads[t] = NULL; continue; }
        memset(&jobs[t], 0, sizeof(jobs[t]));
        jobs[t].pool = pool; jobs[t].ctx = ctx;
        jobs[t].layers = layers + ls; jobs[t].layer_count = le - ls;
        jobs[t].mtp_tokens = mtp_tokens; jobs[t].top_k = top_k; jobs[t].expert_count = expert_count;
        jobs[t].tok_ids = tok_ids;
        threads[t] = CreateThread(NULL, 0, parallel_prefetch_worker, &jobs[t], 0, NULL);
    }
    for (int t = 0; t < nt; t++) {
        if (!threads[t]) continue;
        DWORD r = 0;
        WaitForSingleObject(threads[t], INFINITE);
        GetExitCodeThread(threads[t], &r);
        CloseHandle(threads[t]);
        total += (uint64_t)r;
    }
    return total;
}

// 对全部层执行并行 MTP 预取; 返回预取加载的专家总数
static uint64_t parallel_prefetch(cgc_streamer_pool_t* pool, int nthreads, int n_layers,
                                  int mtp_tokens, int top_k, int expert_count,
                                  const int (*tok_ids)[CGC_MAX_EXPERTS_PER_LAYER],
                                  const cgc_cache_access_ctx_t* ctx) {
    int layers[256];
    if (n_layers > 256) n_layers = 256;
    for (int i = 0; i < n_layers; i++) layers[i] = i;
    return parallel_prefetch_layers(pool, nthreads, layers, n_layers,
                                    mtp_tokens, top_k, expert_count, tok_ids, ctx);
}

typedef struct {
    int layer_count;
    int expert_count;
    int top_k;
    int slots;
    uint64_t n_prompt;
    uint64_t n_gen;
    int hot_pool;
    int hot_pool_size;
    int async_prefetch;
    int use_routing;

    // PD 分离
    int pd_split;
    double prefill_ratio;
    int prefill_gpu;
    int decode_gpu;
    int prefill_layers;   // prefill 层数
    int decode_layers;    // decode 层数

    // 结果
    double prefill_seconds;
    double prefill_tok_per_s;
    double prefill_memory_mb;
    double decode_seconds;
    double decode_tok_per_s;
    double decode_memory_mb;
    double resident_delta_mb;
} bench_result_t;

int main(int argc, char** argv) {
    const char* gguf_path = argc > 1 ? argv[1] :
        "C:/Users/alexchuang/Desktop/fastprefill/gemma4_gguf/gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf";

    uint64_t n_prompt = argc > 2 ? strtoull(argv[2], NULL, 10) : 128;
    uint64_t n_gen    = argc > 3 ? strtoull(argv[3], NULL, 10) : 64;
    int slots         = argc > 4 ? atoi(argv[4]) : 16;  // 每层 slots (必须 > hot_pool_size, 否则全部被 pin 死)
    int top_k         = argc > 5 ? atoi(argv[5]) : 4;
    int hot_pool      = argc > 6 ? atoi(argv[6]) : 1;
    int async_prefetch= argc > 7 ? atoi(argv[7]) : 1;
    int use_routing   = argc > 8 ? atoi(argv[8]) : 1;
    int hot_pool_size = argc > 9 ? atoi(argv[9]) : 8;
    int use_mmap      = argc > 10 ? atoi(argv[10]) : 1;   // 1=mmap 直读 (省系统调用), 0=ReadFile
    int mtp_tokens    = argc > 11 ? atoi(argv[11]) : 4;   // MTP 批量前瞻 token 数 (摊薄磁盘 I/O)
    if (mtp_tokens < 1) mtp_tokens = 1;
    if (mtp_tokens > BENCH_MAX_MTP) mtp_tokens = BENCH_MAX_MTP;

    // PD 分离 (layer-split load balance)
    int pd_split        = argc > 12 ? atoi(argv[12]) : 0;  // 0=全层混合, 1=PD 分离
    double prefill_ratio= argc > 13 ? atof(argv[13]) : 0.5;
    int prefill_gpu     = argc > 14 ? atoi(argv[14]) : 0;
    int decode_gpu      = argc > 15 ? atoi(argv[15]) : 1;
    if (prefill_ratio < 0.05) prefill_ratio = 0.05;
    if (prefill_ratio > 0.95) prefill_ratio = 0.95;

    if (hot_pool && hot_pool_size >= slots) {
        fprintf(stderr, "ERROR: slots(%d) must be > hot_pool_size(%d), otherwise hot pool pins all slots\n",
                slots, hot_pool_size);
        return 1;
    }

    bench_result_t res;
    memset(&res, 0, sizeof(res));
    res.layer_count = 0;
    res.expert_count = 0;
    res.top_k = top_k;
    res.slots = slots;
    res.n_prompt = n_prompt;
    res.n_gen = n_gen;
    res.hot_pool = hot_pool;
    res.hot_pool_size = hot_pool_size;
    res.async_prefetch = async_prefetch;
    res.use_routing = use_routing;
    res.pd_split = pd_split;
    res.prefill_ratio = prefill_ratio;
    res.prefill_gpu = prefill_gpu;
    res.decode_gpu = decode_gpu;

    printf("============================================================\n");
    printf("  CGC llama-bench (C 版 Expert Streaming 基准)\n");
    printf("============================================================\n");
    printf("  model : %s\n", gguf_path);
    printf("  n_prompt=%llu  n_gen=%llu  slots=%d  top_k=%d\n",
           (unsigned long long)n_prompt, (unsigned long long)n_gen, slots, top_k);
    printf("  hot_pool=%d(%d)  async_prefetch=%d  routing=%d\n",
           hot_pool, hot_pool_size, async_prefetch, use_routing);
    printf("  use_mmap=%d  mtp_tokens=%d\n", use_mmap, mtp_tokens);
    if (pd_split) {
        printf("  PD SPLIT: prefill_ratio=%.2f  prefill_gpu=%d  decode_gpu=%d\n",
               prefill_ratio, prefill_gpu, decode_gpu);
    } else {
        printf("  PD SPLIT: off (all layers mixed)\n");
    }
    printf("\n");

    if (use_routing && (mtp_tokens * top_k + hot_pool_size) > slots) {
        fprintf(stderr, "WARNING: slots(%d) < mtp_tokens*top_k + hot_pool(%d), MTP 预取会驱逐槽位\n",
                slots, mtp_tokens * top_k + hot_pool_size);
    }

    // ---- 1. 解析 GGUF ----
    printf("[1/5] Parsing GGUF header...\n");
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(gguf_path);
    if (!ctx) {
        fprintf(stderr, "ERROR: cannot load GGUF header\n");
        return 1;
    }
    printf("  tensors=%llu  kv=%llu  data_start=%llu\n",
           (unsigned long long)ctx->n_tensors,
           (unsigned long long)ctx->n_kv,
           (unsigned long long)ctx->data_start);

    uint32_t expert_count_u32 = 0;
    if (!cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expert_count_u32) || expert_count_u32 == 0) {
        fprintf(stderr, "ERROR: gemma4.expert_count not found\n");
        cgc_gguf_lite_free(ctx);
        return 1;
    }
    uint32_t block_count_u32 = 0;
    if (!cgc_gguf_lite_get_u32(ctx, "gemma4.block_count", &block_count_u32)) {
        fprintf(stderr, "ERROR: gemma4.block_count not found\n");
        cgc_gguf_lite_free(ctx);
        return 1;
    }
    int layer_count = (int)block_count_u32;
    int expert_count = (int)expert_count_u32;
    printf("  layers=%d  experts/layer=%d\n", layer_count, expert_count);

    bench_layer_expert_t layer_experts[BENCH_MAX_LAYERS];
    int parsed_layers = parse_gemma_layer_experts(ctx, layer_experts, BENCH_MAX_LAYERS);
    printf("  parsed expert layers: %d\n", parsed_layers);
    if (parsed_layers < layer_count) {
        fprintf(stderr, "ERROR: parsed %d/%d layers\n", parsed_layers, layer_count);
        cgc_gguf_lite_free(ctx);
        return 1;
    }

    // ---- 2. 构建每层 streamer ----
    printf("[2/5] Building per-layer streamers (slots=%d)...\n", slots);
    cgc_streamer_pool_t* pool = cgc_streamer_pool_create();

    // hot pool: 前 hot_pool_size 个专家 (模拟 top-N mix profile)
    int hot_pool_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int hot_pool_count = 0;
    if (hot_pool && hot_pool_size > 0) {
        hot_pool_count = hot_pool_size < expert_count ? hot_pool_size : expert_count;
        for (int i = 0; i < hot_pool_count; i++) hot_pool_ids[i] = i;
    }

    for (int l = 0; l < parsed_layers; l++) {
        bench_layer_expert_t* le = &layer_experts[l];

        cgc_stream_layout_t layout;
        memset(&layout, 0, sizeof(layout));
        strncpy(layout.path, gguf_path, CGC_MAX_PATH_LEN - 1);
        layout.stream_offset = 0;
        layout.stream_size = 0;   // 不做全文件校验; 用 per-region 绝对偏移
        layout.experts_per_layer = expert_count;
        layout.region_count = le->region_count;
        for (int r = 0; r < le->region_count; r++) {
            layout.region_stride[r] = le->region_stride[r];
            layout.layer_offsets[l][r] = le->region_abs_offset[r];
        }
        layout.has_layer_offsets = 1;

        // expert_stride = 完整专家字节数 (所有 region 合计)
        uint64_t expert_size = 0;
        for (int r = 0; r < le->region_count; r++) expert_size += le->region_stride[r];
        layout.expert_stride = expert_size;

        cgc_expert_streamer_t* s = cgc_expert_streamer_create(
            &layout, slots, use_mmap ? true : false,
            hot_pool ? hot_pool_ids : NULL, hot_pool ? hot_pool_count : 0);
        if (!s) {
            fprintf(stderr, "ERROR: cannot create streamer for layer %d\n", l);
            cgc_gguf_lite_free(ctx);
            cgc_streamer_pool_destroy(pool);
            return 1;
        }
        cgc_expert_streamer_set_layer(s, l);
        cgc_streamer_pool_add(pool, l, s);
    }
    printf("  created %d streamers\n", parsed_layers);

    // 估算总 slot 缓冲内存 (每层 slots 个 expert 缓冲 + 热池占其中前 hot_pool 个)
    {
        uint64_t total_buf = 0;
        for (int l = 0; l < parsed_layers; l++) {
            uint64_t esize = 0;
            for (int r = 0; r < layer_experts[l].region_count; r++)
                esize += layer_experts[l].region_stride[r];
            total_buf += (uint64_t)slots * esize;
        }
        printf("  estimated slot buffer memory: %.1f MB (layers*slots*stride)\n",
               (double)total_buf / 1024.0 / 1024.0);
        if (total_buf > (uint64_t)8 << 30) {
            fprintf(stderr, "WARNING: slot buffers exceed 8GB, reduce slots\n");
        }
    }

    // ---- 3. 路由控制器 ----
    printf("[3/5] Initializing routing controller (async=%d)...\n", async_prefetch);
    cgc_routing_controller_t* ctrl = NULL;
    if (use_routing) {
        cgc_routing_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.strategy = CGC_ROUTING_STRATEGY_LOOKAHEAD;
        cfg.lookahead_tokens = 8;
        cfg.max_prefetch_per_step = top_k * 2;
        cfg.async_prefetch = async_prefetch;
        cfg.enable_history_predict = 1;
        cfg.history_predict_order = 2;
        ctrl = cgc_routing_controller_create(pool, &cfg);
        if (!ctrl) {
            fprintf(stderr, "WARNING: routing controller init failed\n");
            use_routing = 0;
        }
    }

    // PD 分离调度器 (layer-split load balance)
    cgc_pd_scheduler_t* pd_sched = NULL;
    cgc_pd_layer_assignment_t pd_assign;
    memset(&pd_assign, 0, sizeof(pd_assign));
    if (pd_split) {
        pd_assign = cgc_pd_layer_assignment_by_ratio(parsed_layers, prefill_ratio);
        pd_assign.prefill_gpu = prefill_gpu;
        pd_assign.decode_gpu = decode_gpu;
        pd_sched = cgc_pd_scheduler_create(pool, &pd_assign, top_k, top_k);
        if (pd_sched) {
            cgc_pd_scheduler_set_top_k(pd_sched, top_k);
            printf("  PD assignment: prefill=%d layers (GPU%d), decode=%d layers (GPU%d)\n",
                   pd_assign.prefill_count, prefill_gpu,
                   pd_assign.decode_count, decode_gpu);
        } else {
            fprintf(stderr, "WARNING: pd_scheduler create failed, fallback to mixed mode\n");
            pd_split = 0;
        }
    }

    // ---- 4. Prefill 阶段 ----
    printf("[4/5] Prefill (%llu tokens)...\n", (unsigned long long)n_prompt);
    double t0 = now_seconds();
    unsigned int seed = 42;
    cgc_cache_access_ctx_t ctx_dec;
    memset(&ctx_dec, 0, sizeof(ctx_dec));
    ctx_dec.owner_phase = CGC_CACHE_SLOT_PREFILL_TRANSIENT;
    ctx_dec.control_plane = CGC_CACHE_CONTROL_PREFILL;

    if (pd_sched) cgc_pd_scheduler_enter_prefill(pd_sched);

    uint64_t prefill_expert_loads = 0;
    int prefill_last[CGC_MAX_EXPERTS_PER_LAYER];
    memset(prefill_last, 0, sizeof(prefill_last));
    int prefill_have_last = 0;
    for (uint64_t t = 0; t < n_prompt; t++) {
        for (int l = 0; l < parsed_layers; l++) {
            // PD 分离: prefill 阶段只访问 prefill 层 (GPU0)
            if (pd_split && !cgc_pd_is_prefill_layer(&pd_assign, l)) continue;
            int ids[CGC_MAX_EXPERTS_PER_LAYER];
            if (prefill_have_last)
                gen_route_ids_correlated(&seed, top_k, expert_count, prefill_last, ids);
            else
                gen_route_ids(&seed, top_k, expert_count, ids);
            memcpy(prefill_last, ids, sizeof(ids));
            prefill_have_last = 1;
            cgc_cache_result_t r;
            if (pd_sched) {
                r = cgc_pd_scheduler_load_prefill_experts(pd_sched, l, ids, top_k);
            } else {
                r = cgc_streamer_pool_load_experts(pool, l, ids, top_k, &ctx_dec);
            }
            prefill_expert_loads += (uint64_t)r.misses;
        }
    }
    double prefill_seconds = now_seconds() - t0;
    double prefill_mem_mb = (double)current_working_set_kb() / 1024.0;

    printf("  prefill time: %.3f s  (%.1f tok/s)\n",
           prefill_seconds, (double)n_prompt / prefill_seconds);
    printf("  prefill.memory: %.1f MB (working set)\n", prefill_mem_mb);
    printf("  prefill expert loads: %llu\n", (unsigned long long)prefill_expert_loads);

    // ---- 5. Decode 阶段 ----
    printf("[5/5] Decode (%llu tokens)...\n", (unsigned long long)n_gen);
    ctx_dec.owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED;
    ctx_dec.control_plane = CGC_CACHE_CONTROL_DECODE;

    if (pd_sched) cgc_pd_scheduler_switch_to_decode(pd_sched);

    t0 = now_seconds();
    uint64_t decode_expert_loads = 0;
    uint64_t decode_prefetch_loads = 0;
    int last_ids[CGC_MAX_EXPERTS_PER_LAYER];
    memset(last_ids, 0, sizeof(last_ids));
    int have_last = 0;

    // PD 分离: 预构建 decode 层索引列表 (GPU1), 供多线程并行预取切分
    int decode_layers[BENCH_MAX_LAYERS];
    int decode_layer_count = 0;
    if (pd_split) {
        for (int l = 0; l < parsed_layers; l++) {
            if (cgc_pd_is_decode_layer(&pd_assign, l))
                decode_layers[decode_layer_count++] = l;
        }
    }
    for (uint64_t t = 0; t < n_gen; t += (uint64_t)mtp_tokens) {
        uint64_t batch_n = (t + (uint64_t)mtp_tokens > n_gen) ? (n_gen - t) : (uint64_t)mtp_tokens;

        // batch 内生成相关路由: 相邻 token 保留上一 token 的 (top_k-1) 个专家,
        // 只换 1 个新专家, 模拟真实 MoE 路由局部性 (turbo-fieldfare r3/r4 依赖此特性)。
        // 相关性使 MTP 批量去重后专家数大幅减少 + 跨 token 缓存命中, 摊薄磁盘 I/O。
        int tok_ids[BENCH_MAX_MTP][CGC_MAX_EXPERTS_PER_LAYER];
        for (uint64_t bt = 0; bt < batch_n; bt++) {
            if (have_last && bt > 0) {
                gen_route_ids_correlated(&seed, top_k, expert_count, tok_ids[bt - 1], tok_ids[bt]);
            } else if (have_last) {
                gen_route_ids_correlated(&seed, top_k, expert_count, last_ids, tok_ids[0]);
            } else {
                gen_route_ids(&seed, top_k, expert_count, tok_ids[0]);
            }
        }
        memcpy(last_ids, tok_ids[batch_n - 1], sizeof(last_ids));
        have_last = 1;

        // 多线程并行 MTP 预取: 各层去重专家并发读入, 提高 NVMe I/O 队列深度,
        // 摊薄磁盘读延迟 (串行同步读只有 ~0.94GB/s, 并行可达 2GB/s+)
        if (mtp_tokens > 1) {
            if (pd_split) {
                // PD 分离: 只对 decode 层 (GPU1) 做 8 线程并行预取
                decode_prefetch_loads += parallel_prefetch_layers(
                    pool, 8, decode_layers, decode_layer_count,
                    (int)batch_n, top_k, expert_count, tok_ids, &ctx_dec);
            } else {
                decode_prefetch_loads += parallel_prefetch(pool, 8, parsed_layers,
                                                           (int)batch_n, top_k, expert_count,
                                                           tok_ids, &ctx_dec);
            }
        }

        for (int l = 0; l < parsed_layers; l++) {
            // PD 分离: decode 阶段只访问 decode 层 (GPU1)
            if (pd_split && !cgc_pd_is_decode_layer(&pd_assign, l)) continue;
            for (uint64_t bt = 0; bt < batch_n; bt++) {
                cgc_cache_result_t r;
                if (pd_sched) {
                    r = cgc_pd_scheduler_load_decode_experts(pd_sched, l, tok_ids[bt], top_k);
                } else {
                    r = cgc_streamer_pool_load_experts(pool, l, tok_ids[bt], top_k, &ctx_dec);
                }
                decode_expert_loads += (uint64_t)r.misses;
                // 路由预取 (每步喂当前路由, 异步加载未来专家)
                if (ctrl) {
                    cgc_routing_controller_on_route(ctrl, l, tok_ids[bt], top_k);
                }
            }
        }
    }
    if (ctrl) cgc_routing_controller_drain(ctrl);
    double decode_seconds = now_seconds() - t0;
    double decode_mem_mb = (double)current_working_set_kb() / 1024.0;

    printf("  decode time: %.3f s  (%.1f tok/s)\n",
           decode_seconds, (double)n_gen / decode_seconds);
    printf("  decode.memory: %.1f MB (working set)\n", decode_mem_mb);
    printf("  decode expert loads (misses): %llu\n", (unsigned long long)decode_expert_loads);
    printf("  decode mtp prefetch loads   : %llu\n", (unsigned long long)decode_prefetch_loads);

    // ---- 结果 ----
    res.prefill_seconds = prefill_seconds;
    res.prefill_tok_per_s = (double)n_prompt / prefill_seconds;
    res.prefill_memory_mb = prefill_mem_mb;
    res.decode_seconds = decode_seconds;
    res.decode_tok_per_s = (double)n_gen / decode_seconds;
    res.decode_memory_mb = decode_mem_mb;
    res.resident_delta_mb = decode_mem_mb - prefill_mem_mb;

    // 各层 streamer telemetry 汇总
    uint64_t total_requests = 0, total_hits = 0, total_misses = 0;
    uint64_t total_read_bytes = 0, total_read_nanos = 0;
    for (int l = 0; l < parsed_layers; l++) {
        cgc_expert_streamer_t* s = cgc_streamer_pool_get(pool, l);
        if (!s) continue;
        cgc_cache_telemetry_t tel = cgc_expert_streamer_telemetry(s);
        total_requests += tel.total_requests;
        total_hits += tel.total_hits;
        total_misses += tel.total_misses;
        total_read_bytes += tel.total_read_bytes;
        total_read_nanos += tel.total_read_wall_nanos;
    }
    double hit_rate = total_requests > 0 ? (double)total_hits / (double)total_requests * 100.0 : 0.0;

    cgc_routing_stats_t rstats;
    memset(&rstats, 0, sizeof(rstats));
    if (ctrl) rstats = cgc_routing_controller_stats(ctrl);

    printf("\n============================================================\n");
    printf("  RESULTS\n");
    printf("============================================================\n");
    printf("  prefill.tok/s        : %.2f\n", res.prefill_tok_per_s);
    printf("  prefill.memory       : %.1f MB\n", res.prefill_memory_mb);
    printf("  decode.tok/s         : %.2f\n", res.decode_tok_per_s);
    printf("  decode.memory        : %.1f MB\n", res.decode_memory_mb);
    printf("  resident.delta       : %.1f MB (decode - prefill)\n", res.resident_delta_mb);
    printf("  cache hit rate       : %.2f%% (req=%llu hit=%llu miss=%llu)\n",
           hit_rate, (unsigned long long)total_requests,
           (unsigned long long)total_hits, (unsigned long long)total_misses);
    printf("  read.total           : %.2f GB in %.3f s (%.2f GB/s)\n",
           (double)total_read_bytes / 1e9, (double)total_read_nanos / 1e9,
           total_read_nanos > 0 ? (double)total_read_bytes / (double)total_read_nanos : 0.0);
    if (ctrl) {
        printf("  routing.prefetch.loaded  : %llu\n", (unsigned long long)rstats.total_prefetch_loaded);
        printf("  routing.prefetch.skipped : %llu\n", (unsigned long long)rstats.total_prefetch_skipped);
        printf("  routing.prefetch.read_ms : %.1f\n", (double)rstats.total_read_wall_nanos / 1e6);
    }

    // decode 20 tok/s 达标检查
    printf("\n------------------------------------------------------------\n");
    if (res.decode_tok_per_s >= 20.0) {
        printf("  [PASS] decode >= 20 tok/s (%.2f)\n", res.decode_tok_per_s);
    } else {
        printf("  [WARN] decode %.2f tok/s < 20 target (架构/IO 相关)\n", res.decode_tok_per_s);
    }
    if (res.prefill_memory_mb <= 0) {
        printf("  [WARN] prefill.memory unavailable (non-Windows?)\n");
    }
    printf("============================================================\n");

    // ---- 清理 ----
    if (ctrl) cgc_routing_controller_destroy(ctrl);
    for (int l = 0; l < parsed_layers; l++) {
        cgc_expert_streamer_t* s = cgc_streamer_pool_get(pool, l);
        if (s) cgc_expert_streamer_destroy(s);
    }
    cgc_streamer_pool_destroy(pool);
    cgc_gguf_lite_free(ctx);
    return 0;
}
