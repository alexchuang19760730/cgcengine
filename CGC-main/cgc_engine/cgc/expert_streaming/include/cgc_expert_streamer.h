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
#define CGC_DEFAULT_ALIGN    64
#define CGC_MAX_SLOT_COUNT   1024
#define CGC_MAX_NAME_LEN     256

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

typedef struct {
    char path[CGC_MAX_PATH_LEN];
    uint64_t stream_offset;
    uint64_t stream_size;
    int experts_per_layer;
    uint64_t expert_stride;
    uint64_t expert_offsets[CGC_MAX_EXPERTS_PER_LAYER];
    int has_explicit_offsets;

    // llama.cpp 標準 GGUF（qwen35moe / gemma4 3-tensor 佈局）:
    // 每層 gate/up/down 是三個獨立 packed 張量（各 [out, in, n_experts]），
    // 專家權重分散在三個非連續檔案區段。has_segments=1 時，
    // seg_base[0..2] = gate/up/down 張量資料起點，seg_size[0..2] = 單 expert 各段 bytes
    // （stride = seg_size 之和，slot buffer 內依 gate|up|down 順序拼接）。
    int has_segments;
    uint64_t seg_base[3];
    uint64_t seg_size[3];
} cgc_stream_layout_t;

static inline uint64_t cgc_expert_offset(const cgc_stream_layout_t* layout,
                                          int layer, int expert) {
    if (layer == 0 && layout->has_explicit_offsets &&
        expert >= 0 && expert < layout->experts_per_layer) {
        return layout->expert_offsets[expert];
    }
    uint64_t per_layer = (uint64_t)layout->experts_per_layer * layout->expert_stride;
    return layout->stream_offset + (uint64_t)layer * per_layer + (uint64_t)expert * layout->expert_stride;
}

// 回傳 expert 的載入區段（offset+size）。has_segments 時為 3 段（gate/up/down），
// 否則為單一段（沿用 cgc_expert_offset / expert_stride）。回傳段數（0 = 無效）。
static inline int cgc_expert_segments(const cgc_stream_layout_t* layout,
                                       int layer, int expert,
                                       uint64_t out_offsets[3], uint64_t out_sizes[3]) {
    if (!layout || !out_offsets || !out_sizes) return 0;
    if (layout->has_segments) {
        if (expert < 0 || expert >= layout->experts_per_layer) return 0;
        for (int k = 0; k < 3; k++) {
            out_offsets[k] = layout->seg_base[k] + (uint64_t)expert * layout->seg_size[k];
            out_sizes[k] = layout->seg_size[k];
        }
        return 3;
    }
    out_offsets[0] = cgc_expert_offset(layout, layer, expert);
    out_sizes[0] = layout->expert_stride;
    return 1;
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
} cgc_cache_telemetry_t;

typedef struct {
    cgc_stream_layout_t layout;
    int slot_count;
    bool use_mmap;
    int hot_pool_experts[CGC_MAX_EXPERTS_PER_LAYER];
    int hot_pool_count;

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

    int use_clock;
    uint64_t total_requests;
    uint64_t total_hits;
    uint64_t total_misses;
    uint64_t total_loads;
    uint64_t total_evictions;
    uint64_t total_read_wall_nanos;
    uint64_t total_read_bytes;

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

void cgc_expert_streamer_destroy(cgc_expert_streamer_t* streamer);

cgc_cache_result_t cgc_expert_streamer_load_experts(cgc_expert_streamer_t* streamer,
                                                     const int* expert_ids,
                                                     int count,
                                                     const cgc_cache_access_ctx_t* ctx);

void cgc_expert_streamer_prefetch(cgc_expert_streamer_t* streamer,
                                   const int* expert_ids,
                                   int count);

cgc_cache_telemetry_t cgc_expert_streamer_telemetry(const cgc_expert_streamer_t* streamer);

void cgc_expert_streamer_release_slot(cgc_expert_streamer_t* streamer, int slot);

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
