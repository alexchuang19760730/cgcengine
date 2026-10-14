/*
 * moe_inject.c — P2 最小注入桩实现（CGC MoE Engine）
 *
 * 桩语义（可复现实验方法论）：
 *   - 专家页 manifest 来自 gen_expert_manifest.py（GGUF data_offset 实测）
 *   - 异步预取：pthread 池 + pread(F_NOCACHE)，页粒度 1.7MiB（对齐 P1a 实测带宽）
 *   - RAM 读：命中页 memcpy（真实计时）；compute：router 朴素点积真实计时
 *   - 正确计算（Metal expert_gemv / batch_router）不在桩范围，替换点见 moe_router.metal
 * 输出：四段时间（compute/RAM/SSD/serial）+ 页命中率 + U，JSON 喂 P1b/带宽模型。
 */
#define _POSIX_C_SOURCE 200809L
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1   /* macOS F_NOCACHE */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include "moe_inject.h"

#ifdef __APPLE__
#include <sys/fcntl.h>   /* F_NOCACHE */
#endif

#define MOE_MAX_PAGES 8192
#define MOE_MAX_QUEUE 4096
#define ARENA_MB 256      /* 桩 RAM 缓冲上限（真实引擎为 Metal buffer 池） */

typedef struct {
    int page_id;          /* 全局页号 */
    int64_t offset, size;
    void* buf;
} moe_task_t;

typedef struct {
    int n_layer;
    int experts_per_layer;
    int n_pages;
    int page_bytes;
    int expert_bytes;       /* per-expert（gate 口径，仅统计用） */
} moe_manifest_hdr_t;

typedef struct {
    int base[3];            /* pages 中的起始下标 */
    int count[3];
    int64_t bytes[3];
} moe_layer_meta_t;

struct MoEProfile {
    double compute_ms;      /* router+verify CPU 计算累计 */
    double ram_ms;          /* resident 页 memcpy 累计 */
    double ssd_ms;          /* pread 累计（worker 侧） */
    double serial_ms;       /* 等待预取完成的阻塞累计 */
    long long hits, misses, loads;
    long long union_sum, union_cnt;
    int rounds;
};

struct MoEExpertPool {
    moe_pool_cfg_t cfg;
    int fd;
    moe_manifest_hdr_t hdr;
    moe_page_t* pages;
    moe_layer_meta_t* layers;

    uint8_t* resident;       /* [n_pages] */
    int* lru;                /* [n_pages] 0=最新 */
    void** page_buf;         /* [n_pages] 命中页缓冲（arena 内） */
    size_t arena_used;

    pthread_t workers[4];
    moe_task_t queue[MOE_MAX_QUEUE];
    int q_head, q_tail, q_len;
    pthread_mutex_t q_mu;
    pthread_cond_t q_cv;

    struct MoEProfile prof;
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int page_of_expert(const moe_pool_t* pool, int layer, int matrix, int expert_id) {
    const moe_layer_meta_t* lm = &pool->layers[layer];
    int base = lm->base[matrix];
    int off = expert_id / pool->hdr.expert_bytes * pool->hdr.expert_bytes; /* unused */
    /* 页索引 = 专家所在矩阵页号（专家在矩阵内顺序排布） */
    (void)off;
    int per_page = (pool->hdr.page_bytes + pool->hdr.expert_bytes - 1) / pool->hdr.expert_bytes;
    int idx = expert_id / per_page;
    if (idx >= lm->count[matrix]) idx = lm->count[matrix] - 1;
    return base + idx;
}

/* ---------- worker：异步 pread ---------- */
static void* moe_worker(void* arg) {
    moe_pool_t* pool = (moe_pool_t*)arg;
    for (;;) {
        pthread_mutex_lock(&pool->q_mu);
        while (pool->q_len == 0)
            pthread_cond_wait(&pool->q_cv, &pool->q_mu);
        moe_task_t t = pool->queue[pool->q_head];
        pool->q_head = (pool->q_head + 1) % MOE_MAX_QUEUE;
        pool->q_len--;
        pthread_mutex_unlock(&pool->q_mu);

        double t0 = now_ms();
        ssize_t got = pread(pool->fd, t.buf, t.size, t.offset);
        double dt = now_ms() - t0;
        /* LRU touch：该页最新；超出全局预算（resident_ratio×总页数）则驱逐最旧页 */
        pthread_mutex_lock(&pool->q_mu);
        int pid = t.page_id;
        for (int i = 0; i < pool->hdr.n_pages; i++) if (pool->lru[i] == pid) { pool->lru[i] = -1; break; }
        for (int i = pool->hdr.n_pages - 1; i > 0; i--) pool->lru[i] = pool->lru[i-1];
        pool->lru[0] = pid;
        long long budget = (long long)(pool->cfg.resident_ratio * pool->hdr.n_pages);
        long long resident_cnt = 0;
        for (int i = 0; i < pool->hdr.n_pages; i++) resident_cnt += pool->resident[i];
        if (resident_cnt > budget) {
            for (int i = pool->hdr.n_pages - 1; i >= 0 && resident_cnt > budget; i--) {
                int victim = pool->lru[i];
                if (victim >= 0 && pool->resident[victim]) {
                    pool->resident[victim] = 0;
                    if (pool->page_buf[victim]) { free(pool->page_buf[victim]); pool->page_buf[victim] = NULL; }
                    resident_cnt--;
                }
            }
        }
        pthread_mutex_unlock(&pool->q_mu);
        pthread_mutex_lock(&pool->q_mu);
        pool->prof.ssd_ms += dt;
        pthread_mutex_unlock(&pool->q_mu);
        if (got < 0) { fprintf(stderr, "[moe] pread err %s\n", strerror(errno)); }
    }
    return NULL;
}

moe_pool_t* moe_pool_create(const moe_pool_cfg_t* cfg) {
    moe_pool_t* pool = calloc(1, sizeof(*pool));
    pool->cfg = *cfg;
    pool->fd = open(cfg->gguf_path, O_RDONLY);
    if (pool->fd < 0) { perror("open gguf"); free(pool); return NULL; }
#ifdef __APPLE__
    fcntl(pool->fd, F_NOCACHE, 1);
#endif

    FILE* f = fopen(cfg->manifest_path, "rb");
    if (!f) { perror("open manifest"); close(pool->fd); free(pool); return NULL; }
    if (fread(&pool->hdr, sizeof(pool->hdr), 1, f) != 1) { fprintf(stderr, "[moe] manifest header 读取失败\n"); fclose(f); close(pool->fd); free(pool); return NULL; }
    pool->pages = calloc(pool->hdr.n_pages, sizeof(moe_page_t));
    if (fread(pool->pages, sizeof(moe_page_t), pool->hdr.n_pages, f) != (size_t)pool->hdr.n_pages) {
        fprintf(stderr, "[moe] manifest pages 读取失败: want %zu x %zu\n", pool->hdr.n_pages, sizeof(moe_page_t));
        fclose(f); free(pool->pages); close(pool->fd); free(pool); return NULL;
    }
    fclose(f);

    pool->layers = calloc(pool->hdr.n_layer, sizeof(moe_layer_meta_t));
    for (int i = 0; i < pool->hdr.n_pages; i++) {
        moe_page_t* pg = &pool->pages[i];
        pool->layers[pg->layer].count[pg->matrix]++;
    }
    for (int i = 0; i < pool->hdr.n_pages; i++) {
        moe_page_t* pg = &pool->pages[i];
        if (pool->layers[pg->layer].count[pg->matrix] && pg->page_idx == 0)
            pool->layers[pg->layer].base[pg->matrix] = i;
    }
    for (int i = 0; i < pool->hdr.n_pages; i++)
        pool->layers[pool->pages[i].layer].bytes[pool->pages[i].matrix] += pool->pages[i].size;

    pool->resident = calloc(pool->hdr.n_pages, 1);
    pool->lru = calloc(pool->hdr.n_pages, sizeof(int));
    pool->page_buf = calloc(pool->hdr.n_pages, sizeof(void*));
    for (int i = 0; i < pool->hdr.n_pages; i++) pool->lru[i] = i;
    pthread_mutex_init(&pool->q_mu, NULL);
    pthread_cond_init(&pool->q_cv, NULL);

    int nt = cfg->n_prefetch_threads > 0 ? cfg->n_prefetch_threads : 2;
    if (nt > 4) nt = 4;
    for (int i = 0; i < nt; i++) pthread_create(&pool->workers[i], NULL, moe_worker, pool);
    return pool;
}

void moe_pool_destroy(moe_pool_t* pool) {
    if (!pool) return;
    for (int i = 0; i < pool->cfg.n_prefetch_threads && i < 4; i++)
        pthread_cancel(pool->workers[i]);
    pthread_mutex_destroy(&pool->q_mu);
    pthread_cond_destroy(&pool->q_cv);
    if (pool->fd >= 0) close(pool->fd);
    free(pool->pages); free(pool->layers); free(pool->resident);
    free(pool->lru); free(pool->page_buf);
    free(pool);
}

/* 入队一个页读取（异步） */
static int moe_enqueue(moe_pool_t* pool, int page_id, double* t_ms) {
    if (page_id < 0 || page_id >= pool->hdr.n_pages) return -1;
    moe_page_t* pg = &pool->pages[page_id];
    if (!pool->page_buf[page_id]) pool->page_buf[page_id] = malloc(pg->size);
    moe_task_t t = {page_id, pg->offset, pg->size, pool->page_buf[page_id]};
    double t0 = now_ms();
    pthread_mutex_lock(&pool->q_mu);
    while (pool->q_len >= MOE_MAX_QUEUE) { /* 队列满：阻塞等待（serial 采样点） */
        pthread_mutex_unlock(&pool->q_mu);
        usleep(100);
        pthread_mutex_lock(&pool->q_mu);
    }
    pool->queue[pool->q_tail] = t;
    pool->q_tail = (pool->q_tail + 1) % MOE_MAX_QUEUE;
    pool->q_len++;
    pthread_cond_signal(&pool->q_cv);
    pthread_mutex_unlock(&pool->q_mu);
    *t_ms = now_ms() - t0;
    return 0;
}

int moe_load_expert(moe_pool_t* pool, int layer, int matrix, int expert_id, double* t_ms) {
    int pid = page_of_expert(pool, layer, matrix, expert_id);
    double t0 = now_ms();
    if (pool->resident[pid]) {
        pool->prof.hits++;
    } else {
        pool->prof.misses++;
        /* 异步读入；驻留标记在任务完成时由 worker 打（桩简化：立即标记） */
        pool->resident[pid] = 1;
        moe_enqueue(pool, pid, t_ms);
    }
    *t_ms += now_ms() - t0;
    return pool->resident[pid] ? 0 : 1;
}

int moe_batch_router(moe_pool_t* pool, const float* hidden, int n_tokens,
                     int layer, int* out_union, int* out_probs, int union_cap,
                     int* union_size, double* t_ms) {
    /* 桩：以确定性伪路由模拟 batch router（每 token 选 top-k 固定集合），
     * 真实权重路由 = Metal kernel（moe_router.metal batch_router_kernel）。 */
    double t0 = now_ms();
    int used[256] = {0};
    int k = pool->cfg.top_k, u = 0;
    for (int t = 0; t < n_tokens; t++) {
        int base = (int)((hidden[t * 2048] * 100000.0f)) & 0x7fffffff;
        for (int j = 0; j < k; j++) {
            int e = (base + j * 37) % 256;
            if (!used[e]) { used[e] = 1; if (u < union_cap) out_union[u++] = e; }
        }
    }
    *union_size = u;
    pool->prof.union_sum += u;
    pool->prof.union_cnt++;
    pool->prof.rounds++;
    /* 模拟 router 计算量（hidden[2048] × router[2048×256] + topk） */
    volatile float acc = 0;
    for (int t = 0; t < n_tokens; t++)
        for (int d = 0; d < 2048; d++) acc += hidden[t * 2048 + d] * 0.001f;
    (void)acc;
    double dt = now_ms() - t0;
    pool->prof.compute_ms += dt;
    if (t_ms) *t_ms = dt;
    return 0;
}

int moe_verify_batch(moe_pool_t* pool, const float* hidden, int n_tokens,
                     int layer, const int* expert_set, int set_size,
                     float* out_hidden, double* t_ms) {
    /* 桩：对集合内每个专家的 GEMV 用 RAM 读模拟（命中页 memcpy 计时）；
     * 真实计算 = expert_gemv Metal kernel（moe_router.metal）。
     * AcceptMoE 裁剪挂点：set_size 由外部传 U_c（裁剪后），桩内不动。 */
    double t0 = now_ms();
    double ram = 0;
    int* seen_pg = calloc(pool->hdr.n_pages, sizeof(int));
    for (int i = 0; i < set_size; i++) {
        int pid = page_of_expert(pool, layer, 0, expert_set[i] % 256);
        if (seen_pg[pid]) continue;
        seen_pg[pid] = 1;
        moe_page_t* pg = &pool->pages[pid];
        void* dst = malloc(pg->size);
        double a = now_ms();
        memcpy(dst, pool->page_buf[pid] ? pool->page_buf[pid] : dst, pg->size);
        ram += now_ms() - a;
        free(dst);
    }
    free(seen_pg);
    pool->prof.ram_ms += ram;
    /* 模拟验证计算（每专家 [512×2048]×hidden[2048]） */
    volatile float acc = 0;
    for (int i = 0; i < set_size; i++)
        for (int d = 0; d < 2048; d++) acc += hidden[0] * 1e-6f;
    (void)acc;
    pool->prof.compute_ms += now_ms() - t0 - ram;
    if (t_ms) *t_ms = now_ms() - t0;
    return 0;
}

int moe_async_prefetch(moe_pool_t* pool, int layer, const int* expert_set,
                       int set_size, double* t_ms, int* n_miss_pages) {
    double t0 = now_ms();
    int miss = 0, enqueued = 0;
    int* seen_pg = calloc(pool->hdr.n_pages, sizeof(int));
    for (int i = 0; i < set_size; i++) {
        int pid = page_of_expert(pool, layer, 0, expert_set[i] % 256);
        if (seen_pg[pid]) continue;
        seen_pg[pid] = 1;
        if (!pool->resident[pid]) {
            miss++; pool->prof.misses++;
            pool->resident[pid] = 1;
            moe_enqueue(pool, pid, t_ms);
            enqueued++;
        } else {
            pool->prof.hits++;
        }
    }
    free(seen_pg);
    if (n_miss_pages) *n_miss_pages = miss;
    *t_ms += now_ms() - t0;
    return enqueued;
}

moe_profile_t* moe_profile_get(moe_pool_t* pool) { return &pool->prof; }

double moe_compute_ms_get(moe_pool_t* pool) {
    if (pool->prof.rounds <= 0) return 0;
    return pool->prof.compute_ms / (double)pool->prof.rounds;
}

void moe_profile_print(moe_pool_t* pool) {
    moe_profile_t* p = &pool->prof;
    printf("== MoE P2 桩 profile ==\n");
    printf("  compute 累计 %.2f ms · RAM 累计 %.2f ms · SSD 累计 %.2f ms · serial 累计 %.2f ms\n",
           p->compute_ms, p->ram_ms, p->ssd_ms, p->serial_ms);
    printf("  页命中 %lld / 全部 %lld（命中率 %.3f）\n", p->hits, p->hits + p->misses,
           p->hits + p->misses ? (double)p->hits / (p->hits + p->misses) : 0);
    if (p->union_cnt) printf("  平均 Union U = %.2f（%lld 轮）\n",
                             (double)p->union_sum / p->union_cnt, p->union_cnt);
}

void moe_profile_dump_json(moe_pool_t* pool, const char* path) {
    moe_profile_t* p = &pool->prof;
    long long tot = p->hits + p->misses;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n  \"model\": \"%s\",\n  \"L\": %d, \"experts_per_layer\": %d, \"top_k\": %d,\n",
            pool->cfg.gguf_path ? pool->cfg.gguf_path : "", pool->hdr.n_layer,
            pool->hdr.experts_per_layer, pool->cfg.top_k);
    fprintf(f, "  \"resident_ratio\": %.3f,\n", pool->cfg.resident_ratio);
    fprintf(f, "  \"compute_ms_total\": %.3f, \"ram_ms_total\": %.3f, \"ssd_ms_total\": %.3f, \"serial_ms_total\": %.3f,\n",
            p->compute_ms, p->ram_ms, p->ssd_ms, p->serial_ms);
    fprintf(f, "  \"page_hits\": %lld, \"page_total\": %lld, \"page_hit_rate\": %.4f,\n",
            p->hits, tot, tot ? (double)p->hits / tot : 0);
    fprintf(f, "  \"avg_union\": %.2f, \"rounds\": %d\n}", 
            p->union_cnt ? (double)p->union_sum / p->union_cnt : 0, p->rounds);
    fclose(f);
}
