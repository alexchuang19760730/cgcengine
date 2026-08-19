#include "cgc_routing_controller.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32
#pragma comment(lib, "kernel32.lib")
#else
#include <pthread.h>
#include <time.h>
#endif

// ---- 平台线程抽象 (仅 Windows 用 CRITICAL_SECTION + 简单 worker) ----
#ifdef _WIN32

typedef CRITICAL_SECTION cgc_mutex_t;
#define CGC_MUTEX_INIT(m) InitializeCriticalSection(m)
#define CGC_MUTEX_LOCK(m) EnterCriticalSection(m)
#define CGC_MUTEX_UNLOCK(m) LeaveCriticalSection(m)

typedef struct {
    HANDLE thread;
    volatile LONG running;
    volatile LONG wake;
} cgc_worker_t;

static DWORD WINAPI worker_proc(LPVOID arg);

#else

typedef pthread_mutex_t cgc_mutex_t;
#define CGC_MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define CGC_MUTEX_LOCK(m) pthread_mutex_lock(m)
#define CGC_MUTEX_UNLOCK(m) pthread_mutex_unlock(m)

typedef struct {
    pthread_t thread;
    volatile int running;
    volatile int wake;
} cgc_worker_t;

static void* worker_proc(void* arg);

#endif

static uint64_t now_nanos(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int freq_init = 0;
    if (!freq_init) {
        QueryPerformanceFrequency(&freq);
        freq_init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// ---- 控制器扩展 (私有: 附加在工作线程上的状态) ----
typedef struct {
    cgc_routing_controller_t* ctrl;
    cgc_worker_t worker;
    cgc_mutex_t queue_lock;
    cgc_mutex_t history_lock;
    volatile int stop;
} cgc_routing_controller_impl_t;

// 预取任务队列 (每层一组的 pending experts)
typedef struct {
    int layer;
    int expert_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int count;
    int in_flight;
} cgc_prefetch_job_t;

#define CGC_ROUTING_MAX_JOBS 8
static cgc_prefetch_job_t g_jobs[CGC_ROUTING_MAX_JOBS];
static cgc_routing_controller_impl_t* g_impl = NULL;   // 单实例 (测试用)

// ---- 异步 worker ----
static void routing_worker_loop(cgc_routing_controller_impl_t* impl) {
    cgc_routing_controller_t* ctrl = impl->ctrl;
    if (!ctrl) return;

    while (!impl->stop) {
        // 等待唤醒 (简单 busy-wait + sleep, 避免平台差异)
        if (!impl->worker.wake) {
#ifdef _WIN32
            Sleep(2);
#else
            struct timespec ts = {0, 2000000};
            nanosleep(&ts, NULL);
#endif
            continue;
        }
        InterlockedExchange((volatile LONG*)&impl->worker.wake, 0);

        // 处理队列中所有 in_flight job
        cgc_mutex_t* qlock = &impl->queue_lock;
        (void)qlock;

        for (;;) {
            cgc_prefetch_job_t job;
            memset(&job, 0, sizeof(job));
            int found = 0;

            CGC_MUTEX_LOCK(&impl->queue_lock);
            for (int i = 0; i < CGC_ROUTING_MAX_JOBS; i++) {
                if (g_jobs[i].in_flight && g_jobs[i].count > 0) {
                    job = g_jobs[i];
                    g_jobs[i].in_flight = 0;
                    g_jobs[i].count = 0;
                    found = 1;
                    break;
                }
            }
            CGC_MUTEX_UNLOCK(&impl->queue_lock);

            if (!found) break;

            // 真正执行预取 (miss-only + 热池过滤由 streamer 内部处理)
            cgc_expert_streamer_t* s = cgc_streamer_pool_get(ctrl->pool, job.layer);
            if (s) {
                cgc_cache_access_ctx_t ctx;
                memset(&ctx, 0, sizeof(ctx));
                ctx.owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED;
                ctx.control_plane = CGC_CACHE_CONTROL_DECODE;
                uint64_t t0 = now_nanos();
                int n = cgc_expert_streamer_prefetch_load(s, job.expert_ids, job.count, &ctx);
                uint64_t dt = now_nanos() - t0;
                CGC_MUTEX_LOCK(&impl->history_lock);
                ctrl->stats.total_prefetch_loaded += (uint64_t)n;
                ctrl->stats.total_prefetch_skipped += (uint64_t)(job.count - n);
                ctrl->stats.total_read_wall_nanos += dt;
                CGC_MUTEX_UNLOCK(&impl->history_lock);
            }
        }
    }
}

#ifdef _WIN32
static DWORD WINAPI worker_proc(LPVOID arg) {
    routing_worker_loop((cgc_routing_controller_impl_t*)arg);
    return 0;
}
#else
static void* worker_proc(void* arg) {
    routing_worker_loop((cgc_routing_controller_impl_t*)arg);
    return NULL;
}
#endif

// ---- 查找历史序列中的 (layer, expert) ----
static int history_contains(const cgc_routing_controller_t* ctrl, int layer, int expert) {
    for (int i = 0; i < ctrl->history_count; i++) {
        if (ctrl->history_layers[i] == layer && ctrl->history_experts[i] == expert) {
            return 1;
        }
    }
    return 0;
}

// ---- N-gram 前瞻预测: 找出历史中最常跟在当前 (layer, expert) 之后的专家 ----
static void predict_lookahead(cgc_routing_controller_t* ctrl,
                              int layer,
                              int current_expert,
                              int* out_experts,
                              int* out_count) {
    int counts[CGC_MAX_EXPERTS_PER_LAYER];
    memset(counts, 0, sizeof(counts));

    // 统计: 历史中 current_expert 出现后紧跟的 expert 频率
    for (int i = 0; i + 1 < ctrl->history_count; i++) {
        if (ctrl->history_layers[i] == layer && ctrl->history_experts[i] == current_expert) {
            int next_exp = ctrl->history_experts[i + 1];
            if (next_exp >= 0 && next_exp < CGC_MAX_EXPERTS_PER_LAYER) {
                counts[next_exp]++;
            }
        }
    }

    // 取频率最高的 top-N (排除自身)
    int order = ctrl->config.history_predict_order > 0 ? ctrl->config.history_predict_order : 2;
    int max_out = order;
    if (max_out > CGC_ROUTING_MAX_LOOKAHEAD) max_out = CGC_ROUTING_MAX_LOOKAHEAD;

    *out_count = 0;
    for (int iter = 0; iter < max_out; iter++) {
        int best = -1;
        int best_count = -1;
        for (int e = 0; e < CGC_MAX_EXPERTS_PER_LAYER; e++) {
            if (e == current_expert) continue;
            if (counts[e] > best_count) {
                best_count = counts[e];
                best = e;
            }
        }
        if (best < 0 || best_count <= 0) break;
        out_experts[(*out_count)++] = best;
        counts[best] = -1;   // 已选
    }
}

// ---- 记录历史 ----
static void record_history(cgc_routing_controller_t* ctrl, int layer, int expert) {
    if (ctrl->history_count >= CGC_ROUTING_HISTORY_MAX) {
        // 环形覆盖最旧一半
        int keep = ctrl->history_count / 2;
        memmove(ctrl->history_layers, ctrl->history_layers + keep,
                sizeof(int) * (ctrl->history_count - keep));
        memmove(ctrl->history_experts, ctrl->history_experts + keep,
                sizeof(int) * (ctrl->history_count - keep));
        ctrl->history_count -= keep;
    }
    ctrl->history_layers[ctrl->history_count] = layer;
    ctrl->history_experts[ctrl->history_count] = expert;
    ctrl->history_count++;
}

// ---- 实现 ----

cgc_routing_controller_t* cgc_routing_controller_create(cgc_streamer_pool_t* pool,
                                                         const cgc_routing_config_t* config) {
    if (!pool) return NULL;

    cgc_routing_controller_t* ctrl = (cgc_routing_controller_t*)calloc(1, sizeof(cgc_routing_controller_t));
    if (!ctrl) return NULL;

    ctrl->pool = pool;
    if (config) {
        ctrl->config = *config;
    } else {
        ctrl->config.strategy = CGC_ROUTING_STRATEGY_LOOKAHEAD;
        ctrl->config.lookahead_tokens = 8;
        ctrl->config.max_prefetch_per_step = 8;
        ctrl->config.async_prefetch = 1;
        ctrl->config.enable_history_predict = 1;
        ctrl->config.history_predict_order = 2;
        ctrl->config.min_refresh_interval_nanos = 0;
    }

    if (ctrl->config.lookahead_tokens <= 0) ctrl->config.lookahead_tokens = 1;
    if (ctrl->config.lookahead_tokens > CGC_ROUTING_MAX_LOOKAHEAD) ctrl->config.lookahead_tokens = CGC_ROUTING_MAX_LOOKAHEAD;
    if (ctrl->config.max_prefetch_per_step <= 0) ctrl->config.max_prefetch_per_step = 8;

    ctrl->initialized = 1;

    if (ctrl->config.async_prefetch) {
        cgc_routing_controller_impl_t* impl =
            (cgc_routing_controller_impl_t*)calloc(1, sizeof(cgc_routing_controller_impl_t));
        if (!impl) { free(ctrl); return NULL; }
        impl->ctrl = ctrl;
        CGC_MUTEX_INIT(&impl->queue_lock);
        CGC_MUTEX_INIT(&impl->history_lock);
        impl->stop = 0;

#ifdef _WIN32
        impl->worker.thread = CreateThread(NULL, 0, worker_proc, impl, 0, NULL);
        if (!impl->worker.thread) {
            free(impl); free(ctrl); return NULL;
        }
#else
        if (pthread_create(&impl->worker.thread, NULL, worker_proc, impl) != 0) {
            free(impl); free(ctrl); return NULL;
        }
#endif
        // 单实例全局 (测试用)
        g_impl = impl;
    }

    return ctrl;
}

void cgc_routing_controller_destroy(cgc_routing_controller_t* ctrl) {
    if (!ctrl) return;

    if (g_impl && g_impl->ctrl == ctrl) {
        cgc_routing_controller_impl_t* impl = g_impl;
        impl->stop = 1;
#ifdef _WIN32
        WaitForSingleObject(impl->worker.thread, 100);
        CloseHandle(impl->worker.thread);
#else
        pthread_join(impl->worker.thread, NULL);
#endif
        free(impl);
        g_impl = NULL;
    }

    free(ctrl);
}

// 提交一个预取 job (去重 + 上限)
static int submit_prefetch_job(cgc_routing_controller_t* ctrl, int layer,
                               const int* expert_ids, int count) {
    if (!ctrl || !expert_ids || count <= 0) return 0;

    // 与热池/已缓存去重: 让 streamer 的 find_slot 过滤, 这里只做层内去重
    int dedup[CGC_MAX_EXPERTS_PER_LAYER];
    int dedup_count = 0;
    int budget = ctrl->config.max_prefetch_per_step;

    for (int i = 0; i < count && dedup_count < budget; i++) {
        int e = expert_ids[i];
        if (e < 0) continue;

        // 去重
        int dup = 0;
        for (int j = 0; j < dedup_count; j++) {
            if (dedup[j] == e) { dup = 1; break; }
        }
        if (dup) continue;
        dedup[dedup_count++] = e;
    }

    if (dedup_count == 0) return 0;

    if (ctrl->config.async_prefetch && g_impl) {
        cgc_routing_controller_impl_t* impl = g_impl;
        CGC_MUTEX_LOCK(&impl->queue_lock);
        // 替换同层旧 job (只保留最新)
        int slot = -1;
        for (int i = 0; i < CGC_ROUTING_MAX_JOBS; i++) {
            if (g_jobs[i].in_flight && g_jobs[i].layer == layer) { slot = i; break; }
        }
        if (slot < 0) {
            for (int i = 0; i < CGC_ROUTING_MAX_JOBS; i++) {
                if (!g_jobs[i].in_flight) { slot = i; break; }
            }
        }
        if (slot < 0) {   // 队列满: 丢最旧的
            slot = 0;
            for (int i = 1; i < CGC_ROUTING_MAX_JOBS; i++) {
                if (!g_jobs[i].in_flight) { slot = i; break; }
            }
        }
        g_jobs[slot].layer = layer;
        memcpy(g_jobs[slot].expert_ids, dedup, sizeof(int) * dedup_count);
        g_jobs[slot].count = dedup_count;
        g_jobs[slot].in_flight = 1;
        CGC_MUTEX_UNLOCK(&impl->queue_lock);

        ctrl->stats.total_prefetch_attempts += (uint64_t)dedup_count;
        InterlockedExchange((volatile LONG*)&impl->worker.wake, 1);
        return dedup_count;
    }

    // 同步模式: 直接执行
    cgc_expert_streamer_t* s = cgc_streamer_pool_get(ctrl->pool, layer);
    if (s) {
        cgc_cache_access_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED;
        ctx.control_plane = CGC_CACHE_CONTROL_DECODE;
        uint64_t t0 = now_nanos();
        int n = cgc_expert_streamer_prefetch_load(s, dedup, dedup_count, &ctx);
        uint64_t dt = now_nanos() - t0;
        ctrl->stats.total_prefetch_attempts += (uint64_t)dedup_count;
        ctrl->stats.total_prefetch_loaded += (uint64_t)n;
        ctrl->stats.total_prefetch_skipped += (uint64_t)(dedup_count - n);
        ctrl->stats.total_read_wall_nanos += dt;
        return n;
    }
    return 0;
}

int cgc_routing_controller_on_route(cgc_routing_controller_t* ctrl,
                                    int layer,
                                    const int* expert_ids,
                                    int count) {
    if (!ctrl || !expert_ids || count <= 0) return 0;
    if (!ctrl->initialized) return 0;

    ctrl->stats.total_steps++;
    ctrl->stats.total_route_ids += (uint64_t)count;

    // 记录本层最近路由 (供 predict 用)
    if (layer >= 0 && layer < CGC_MAX_LAYERS) {
        int n = count < CGC_MAX_EXPERTS_PER_LAYER ? count : CGC_MAX_EXPERTS_PER_LAYER;
        memcpy(ctrl->layer_route_experts[layer], expert_ids, sizeof(int) * n);
        ctrl->layer_route_count[layer] = n;
        ctrl->layer_route_layer[layer] = layer;
    }

    // 记录历史
    for (int i = 0; i < count; i++) {
        if (expert_ids[i] >= 0) {
            record_history(ctrl, layer, expert_ids[i]);
        }
    }

    // 策略: none 只记录
    if (ctrl->config.strategy == CGC_ROUTING_STRATEGY_NONE) return 0;

    // 构造预取候选: 当前 miss + 前瞻
    int prefetch_candidates[CGC_MAX_EXPERTS_PER_LAYER * 2];
    int pc_count = 0;

    // 1) 当前 token 的 miss
    for (int i = 0; i < count && pc_count < ctrl->config.max_prefetch_per_step; i++) {
        prefetch_candidates[pc_count++] = expert_ids[i];
    }

    // 2) 前瞻预测 (lookahead strategy + enable_history_predict)
    if (ctrl->config.strategy == CGC_ROUTING_STRATEGY_LOOKAHEAD &&
        ctrl->config.enable_history_predict) {
        // 基于当前层的每个路由专家预测后续
        for (int i = 0; i < count && pc_count < CGC_MAX_EXPERTS_PER_LAYER * 2; i++) {
            int pred[CGC_ROUTING_MAX_LOOKAHEAD];
            int pred_count = 0;
            predict_lookahead(ctrl, layer, expert_ids[i], pred, &pred_count);
            for (int p = 0; p < pred_count && pc_count < CGC_MAX_EXPERTS_PER_LAYER * 2; p++) {
                prefetch_candidates[pc_count++] = pred[p];
            }
        }
    }

    int submitted = submit_prefetch_job(ctrl, layer, prefetch_candidates, pc_count);
    return submitted;
}

int cgc_routing_controller_on_route_batch(cgc_routing_controller_t* ctrl,
                                          const int* layers,
                                          const int* const* expert_ids_per_layer,
                                          const int* counts,
                                          int layer_count) {
    if (!ctrl || !layers || layer_count <= 0) return 0;
    int total = 0;
    for (int i = 0; i < layer_count; i++) {
        total += cgc_routing_controller_on_route(ctrl, layers[i],
                                                 expert_ids_per_layer[i],
                                                 counts ? counts[i] : 0);
    }
    return total;
}

int cgc_routing_controller_prefetch_experts(cgc_routing_controller_t* ctrl,
                                            int layer,
                                            const int* expert_ids,
                                            int count) {
    if (!ctrl || !expert_ids || count <= 0) return 0;
    return submit_prefetch_job(ctrl, layer, expert_ids, count);
}

int cgc_routing_controller_refresh_lookahead(cgc_routing_controller_t* ctrl) {
    if (!ctrl) return 0;
    // 前瞻预测在 on_route 内实时计算, 这里仅更新 refresh 计数
    ctrl->stats.last_refresh_step = ctrl->stats.total_steps;
    return 1;
}

void cgc_routing_controller_drain(cgc_routing_controller_t* ctrl) {
    if (!ctrl || !g_impl || g_impl->ctrl != ctrl) return;
    // 等待当前队列处理完 (简单等待)
    for (int i = 0; i < 50; i++) {
        int pending = 0;
        CGC_MUTEX_LOCK(&g_impl->queue_lock);
        for (int j = 0; j < CGC_ROUTING_MAX_JOBS; j++) {
            if (g_jobs[j].in_flight) { pending = 1; break; }
        }
        CGC_MUTEX_UNLOCK(&g_impl->queue_lock);
        if (!pending) return;
#ifdef _WIN32
        Sleep(2);
#else
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
#endif
    }
}

cgc_routing_stats_t cgc_routing_controller_stats(const cgc_routing_controller_t* ctrl) {
    cgc_routing_stats_t s;
    memset(&s, 0, sizeof(s));
    if (ctrl) s = ctrl->stats;
    return s;
}

void cgc_routing_controller_reset_stats(cgc_routing_controller_t* ctrl) {
    if (!ctrl) return;
    memset(&ctrl->stats, 0, sizeof(ctrl->stats));
}
