#ifndef CGC_EXPERT_STREAMER_H
#define CGC_EXPERT_STREAMER_H

#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CGC_MAX_PATH_LEN     512
#define CGC_MAX_EXPERTS_PER_LAYER 256
#define CGC_MAX_LAYERS       256
#define CGC_DEFAULT_ALIGN    64
#define CGC_MAX_SLOT_COUNT   1024
#define CGC_MAX_NAME_LEN     256
#define CGC_MAX_EXPERT_REGIONS 8

typedef enum {
    CGC_CACHE_SLOT_UNASSIGNED = 0,
    CGC_CACHE_SLOT_PREFILL_TRANSIENT,
    CGC_CACHE_SLOT_DECODE_PROTECTED,
    CGC_CACHE_SLOT_SHARED_RESIDENT
} cgc_cache_slot_phase_t;

typedef enum {
    CGC_CACHE_CONTROL_PREFILL = 0,
    CGC_CACHE_CONTROL_DECODE,
    CGC_CACHE_CONTROL_SHARED_POOL
} cgc_cache_control_plane_t;

typedef struct {
    cgc_cache_slot_phase_t owner_phase;
    cgc_cache_control_plane_t control_plane;
    uint64_t request_id;
    int decode_step_index;
} cgc_cache_access_ctx_t;

// 流式布局: 支持两种专家权重存储方式
//  1) per-expert  (Qwen3.6 repack): 每层独立文件, 每 expert 一个连续 blob
//     - stream_offset + layer*per_layer + expert*expert_stride  (单一 region)
//  2) per-layer   (Gemma4 GGUF):   所有层共用一个文件, 每层含多个 region
//     (ffn_down_exps + ffn_gate_up_exps), 每个 region 把全部 expert 沿末维打包
//     - 使用 has_layer_offsets + region_count + layer_offsets 定位 (绝对偏移)
typedef struct {
    char path[CGC_MAX_PATH_LEN];
    uint64_t stream_offset;
    uint64_t stream_size;
    int experts_per_layer;
    uint64_t expert_stride;
    uint64_t expert_offsets[CGC_MAX_EXPERTS_PER_LAYER];
    int has_explicit_offsets;

    // per-layer 多 region 布局 (Gemma4 per-layer GGUF)
    int region_count;                                        // region 数量 (0 = 未启用)
    uint64_t region_stride[CGC_MAX_EXPERT_REGIONS];          // 每个 region 单 expert 字节数
    uint64_t layer_offsets[CGC_MAX_LAYERS][CGC_MAX_EXPERT_REGIONS]; // 每层每 region 文件绝对偏移
    int has_layer_offsets;
} cgc_stream_layout_t;

// 计算单个 expert 的完整字节数 (所有 region 合计)
static inline uint64_t cgc_stream_layout_expert_size(const cgc_stream_layout_t* l) {
    if (l && l->has_layer_offsets && l->region_count > 0) {
        uint64_t t = 0;
        for (int r = 0; r < l->region_count && r < CGC_MAX_EXPERT_REGIONS; r++) {
            t += l->region_stride[r];
        }
        return t;
    }
    return l ? l->expert_stride : 0;
}

// region 内第 expert 个专家的文件绝对偏移
static inline uint64_t cgc_expert_offset(const cgc_stream_layout_t* layout,
                                          int layer, int expert) {
    if (layout->has_layer_offsets && layout->region_count > 0 &&
        layer >= 0 && layer < CGC_MAX_LAYERS &&
        expert >= 0 && expert < layout->experts_per_layer) {
        // per-layer 布局: region 0 为基准 (完整 expert = region0+region1 拼接)
        return layout->layer_offsets[layer][0] + (uint64_t)expert * layout->region_stride[0];
    }
    if (layer == 0 && layout->has_explicit_offsets &&
        expert >= 0 && expert < layout->experts_per_layer) {
        return layout->expert_offsets[expert];
    }
    uint64_t per_layer = (uint64_t)layout->experts_per_layer * layout->expert_stride;
    return layout->stream_offset + (uint64_t)layer * per_layer + (uint64_t)expert * layout->expert_stride;
}

static inline uint64_t cgc_stream_region_offset(const cgc_stream_layout_t* l,
                                                 int layer, int region, int expert) {
    if (l->has_layer_offsets && l->region_count > 0 &&
        layer >= 0 && layer < CGC_MAX_LAYERS &&
        region >= 0 && region < l->region_count &&
        expert >= 0 && expert < l->experts_per_layer) {
        return l->layer_offsets[layer][region] + (uint64_t)expert * l->region_stride[region];
    }
    // 兼容: 单一 region (per-expert 布局) 退化为 cgc_expert_offset
    if (region == 0) {
        return cgc_expert_offset(l, layer, expert);
    }
    return 0;
}

typedef struct {
    int expert_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int assigned_slots[CGC_MAX_EXPERTS_PER_LAYER];
    int misses[CGC_MAX_EXPERTS_PER_LAYER];
    int count;
    int hits;
    int miss_count;
} cgc_cache_plan_t;

typedef struct {
    void* buffers[CGC_MAX_EXPERTS_PER_LAYER];
    uint64_t offsets[CGC_MAX_EXPERTS_PER_LAYER];
    uint64_t sizes[CGC_MAX_EXPERTS_PER_LAYER];
    int count;
    int hits;
    int misses;
    uint64_t read_wall_nanos;
    uint64_t read_bytes;
} cgc_cache_result_t;

typedef struct {
    int slot_count;
    int occupied_slots;
    uint64_t total_requests;
    uint64_t total_hits;
    uint64_t total_misses;
    uint64_t total_loads;
    uint64_t total_evictions;
    uint64_t total_read_wall_nanos;
    uint64_t total_read_bytes;
    uint64_t total_prefetch_wall_nanos;   // 主动预取加载 (prefetch_load) 累计耗时
} cgc_cache_telemetry_t;

typedef struct {
    cgc_stream_layout_t layout;
    int slot_count;
    bool use_mmap;
    int hot_pool_experts[CGC_MAX_EXPERTS_PER_LAYER];
    int hot_pool_count;
    int layer_index;            // 该 streamer 负责的层号 (per-layer 布局用)

#ifdef _WIN32
    HANDLE file_handle;
    HANDLE mapping_handle;
    void* mapped_base;
#else
    int fd;
    void* mapped_base;
#endif

    void* slot_buffers[CGC_MAX_SLOT_COUNT];
    int slot_expert[CGC_MAX_SLOT_COUNT];
    cgc_cache_slot_phase_t slot_owner_phase[CGC_MAX_SLOT_COUNT];
    int slot_hit_count[CGC_MAX_SLOT_COUNT];
    int slot_last_use[CGC_MAX_SLOT_COUNT];
    bool slot_pinned[CGC_MAX_SLOT_COUNT];

    // 紧凑 slot 连续区域模式: 每 region 一块连续内存, 槽数据按
    //   region_bases[r] + slot * region_stride[r]  存放,
    // 使 ggml_mul_mat_id 能以 src0->data = region_bases[r], ids 重映射为
    // slot 局部索引后直接读取 (kernel 只做 src0->data + cur_a*nb02 偏移)。
    // 启用时 slot_buffers 为 NULL; slot_expert[i] 仍表示"槽 i 存的全局专家号"。
    void* region_bases[CGC_MAX_EXPERT_REGIONS];
    bool compact_regions;

    int use_clock;
    uint64_t total_requests;
    uint64_t total_hits;
    uint64_t total_misses;
    uint64_t total_loads;
    uint64_t total_evictions;
    uint64_t total_read_wall_nanos;
    uint64_t total_read_bytes;
    uint64_t total_prefetch_wall_nanos;   // 主动预取加载 (prefetch_load) 累计耗时

    int initialized;
    char error_msg[256];
} cgc_expert_streamer_t;

typedef struct {
    cgc_expert_streamer_t* streamers[1024];
    int layer_indices[1024];
    int count;
} cgc_streamer_pool_t;

cgc_expert_streamer_t* cgc_expert_streamer_create(const cgc_stream_layout_t* layout,
                                                   int slot_count,
                                                   bool use_mmap,
                                                   const int* hot_pool_experts,
                                                   int hot_pool_count);

// create 的扩展版: compact_regions=true 时启用紧凑 slot 连续区域模式
// (region_bases 连续分配, 供 ggml_mul_mat_id 零拷贝直接读取)。
cgc_expert_streamer_t* cgc_expert_streamer_create_ex(const cgc_stream_layout_t* layout,
                                                      int slot_count,
                                                      bool use_mmap,
                                                      const int* hot_pool_experts,
                                                      int hot_pool_count,
                                                      bool compact_regions);

void cgc_expert_streamer_destroy(cgc_expert_streamer_t* streamer);

// 紧凑模式下返回指定 region 的连续基址 (未启用返回 NULL)。
// region 数据按 base + slot*region_stride 排布, slot 局部索引来自
// cgc_expert_streamer_build_region_ids。
void* cgc_expert_streamer_get_region_base(cgc_expert_streamer_t* streamer, int region);

// 紧凑模式下返回指定 region 的单槽字节跨度 (region_stride)。
uint64_t cgc_expert_streamer_get_region_stride(cgc_expert_streamer_t* streamer, int region);

// 紧凑模式: 确保 expert_ids 全部已入槽 (miss 则同步加载), 输出全局专家号 ->
// 槽局部索引映射到 local_ids_out (与 expert_ids 等长, 逐元素对应)。
// n_as_out 返回本次映射实际占用的最大槽数 (用于把 mul_mat_id 的 src0 ne[2] 收敛)。
// 返回 count 表示全部映射成功; 返回 -1 表示参数非法或非紧凑模式。
int cgc_expert_streamer_build_region_ids(cgc_expert_streamer_t* streamer,
                                         const int* expert_ids,
                                         int count,
                                         const cgc_cache_access_ctx_t* ctx,
                                         int* local_ids_out,
                                         int* n_as_out);

cgc_cache_result_t cgc_expert_streamer_load_experts(cgc_expert_streamer_t* streamer,
                                                     const int* expert_ids,
                                                     int count,
                                                     const cgc_cache_access_ctx_t* ctx);

void cgc_expert_streamer_prefetch(cgc_expert_streamer_t* streamer,
                                   const int* expert_ids,
                                   int count);

// 主动预取加载 (miss-only): 把未缓存的专家真正读入缓存槽
int cgc_expert_streamer_prefetch_load(cgc_expert_streamer_t* streamer,
                                      const int* expert_ids,
                                      int count,
                                      const cgc_cache_access_ctx_t* ctx);

cgc_cache_telemetry_t cgc_expert_streamer_telemetry(const cgc_expert_streamer_t* streamer);

void cgc_expert_streamer_release_slot(cgc_expert_streamer_t* streamer, int slot);

// 设置 streamer 负责的层号 (per-layer 单文件多 region 布局必调)
void cgc_expert_streamer_set_layer(cgc_expert_streamer_t* streamer, int layer);

cgc_streamer_pool_t* cgc_streamer_pool_create(void);

void cgc_streamer_pool_destroy(cgc_streamer_pool_t* pool);

bool cgc_streamer_pool_add(cgc_streamer_pool_t* pool,
                            int layer_idx,
                            cgc_expert_streamer_t* streamer);

cgc_expert_streamer_t* cgc_streamer_pool_get(cgc_streamer_pool_t* pool, int layer_idx);

cgc_cache_result_t cgc_streamer_pool_load_experts(cgc_streamer_pool_t* pool,
                                                   int layer_idx,
                                                   const int* expert_ids,
                                                   int count,
                                                   const cgc_cache_access_ctx_t* ctx);

uint64_t cgc_stream_layout_compute_offset(const cgc_stream_layout_t* layout,
                                            int layer, int expert);

void cgc_streamer_pool_prefetch(cgc_streamer_pool_t* pool,
                                  int layer_idx,
                                  const int* expert_ids,
                                  int count);

#ifdef __cplusplus
}
#endif

#endif
