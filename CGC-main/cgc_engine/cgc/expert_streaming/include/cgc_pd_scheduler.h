#ifndef CGC_PD_SCHEDULER_H
#define CGC_PD_SCHEDULER_H

#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CGC_MAX_PD_LAYERS   256
#define CGC_MAX_TILES       1024
#define CGC_MAX_ROUTE_HISTORY 4096

typedef enum {
    CGC_PD_PHASE_IDLE = 0,
    CGC_PD_PHASE_PREFILL,
    CGC_PD_PHASE_DECODE
} cgc_pd_phase_t;

typedef struct {
    int layer;
    int expert_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int expert_count;
} cgc_pd_expert_route_t;

typedef struct {
    int token_index;
    cgc_pd_expert_route_t routes[CGC_MAX_PD_LAYERS];
    int route_count;
} cgc_pd_token_routes_t;

typedef struct {
    int tile_index;
    int layer;
    int expert_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int expert_count;
    int token_start;
    int token_count;
} cgc_pd_tile_t;

typedef struct {
    int prefill_layers[CGC_MAX_PD_LAYERS];
    int prefill_count;
    int decode_layers[CGC_MAX_PD_LAYERS];
    int decode_count;
    int prefill_gpu;
    int decode_gpu;
} cgc_pd_layer_assignment_t;

typedef struct {
    int layer;
    int expert_id;
    void* data;
    uint64_t size;
    uint64_t last_access;
    int access_count;
    bool pinned;
} cgc_pd_cache_entry_t;

typedef struct {
    cgc_pd_cache_entry_t entries[CGC_MAX_ROUTE_HISTORY];
    int count;
    int max_count;
    int gpu_id;
    uint64_t hits;
    uint64_t misses;
} cgc_pd_gpu_cache_t;

typedef struct {
    int layer;
    int expert_id;
    double weight;
} cgc_pd_freq_entry_t;

typedef struct {
    cgc_pd_freq_entry_t entries[CGC_MAX_ROUTE_HISTORY];
    int count;
    int max_history;
    double decay_factor;
    int current_layer;
} cgc_pd_route_history_t;

typedef struct {
    cgc_pd_phase_t phase;
    int gpu0_cache_count;
    int gpu1_cache_count;
    double gpu0_hit_rate;
    double gpu1_hit_rate;
    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    uint64_t expert_loads;
    uint64_t prefetch_hits;
    uint64_t total_prefetch_time_nanos;
    uint64_t total_load_time_nanos;
} cgc_pd_scheduler_stats_t;

typedef struct {
    cgc_streamer_pool_t* streamer_pool;
    cgc_pd_layer_assignment_t assignment;
    int max_experts_per_layer;
    int tile_experts;
    int top_k;

    cgc_pd_phase_t current_phase;

    cgc_pd_route_history_t route_history;
    cgc_pd_gpu_cache_t gpu0_cache;
    cgc_pd_gpu_cache_t gpu1_cache;

    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    uint64_t expert_loads;
    uint64_t prefetch_hits;
    uint64_t total_prefetch_time_nanos;
    uint64_t total_load_time_nanos;

    int initialized;
    char error_msg[256];
} cgc_pd_scheduler_t;

cgc_pd_scheduler_t* cgc_pd_scheduler_create(cgc_streamer_pool_t* pool,
                                               const cgc_pd_layer_assignment_t* assignment,
                                               int max_experts_per_layer,
                                               int tile_experts);

void cgc_pd_scheduler_destroy(cgc_pd_scheduler_t* sched);

void cgc_pd_scheduler_enter_prefill(cgc_pd_scheduler_t* sched);

void cgc_pd_scheduler_switch_to_decode(cgc_pd_scheduler_t* sched);

cgc_pd_phase_t cgc_pd_scheduler_current_phase(const cgc_pd_scheduler_t* sched);

int cgc_pd_scheduler_process_prefill(cgc_pd_scheduler_t* sched,
                                      const cgc_pd_token_routes_t* token_routes,
                                      int route_count,
                                      cgc_pd_tile_t* out_tiles,
                                      int max_tiles);

cgc_cache_result_t cgc_pd_scheduler_load_prefill_experts(cgc_pd_scheduler_t* sched,
                                                          int layer,
                                                          const int* expert_ids,
                                                          int count);

int cgc_pd_scheduler_process_decode(cgc_pd_scheduler_t* sched,
                                     const cgc_pd_token_routes_t* token_route,
                                     cgc_pd_tile_t* out_tiles,
                                     int max_tiles);

cgc_cache_result_t cgc_pd_scheduler_load_decode_experts(cgc_pd_scheduler_t* sched,
                                                        int layer,
                                                        const int* expert_ids,
                                                        int count);

int cgc_pd_scheduler_trigger_prefetch(cgc_pd_scheduler_t* sched,
                                       const cgc_pd_token_routes_t* current_routes,
                                       int* out_expert_ids,
                                       int max_ids);

void cgc_pd_scheduler_record_routes(cgc_pd_scheduler_t* sched,
                                      const cgc_pd_token_routes_t* routes);

void cgc_pd_scheduler_set_top_k(cgc_pd_scheduler_t* sched, int top_k);

cgc_pd_scheduler_stats_t cgc_pd_scheduler_get_stats(const cgc_pd_scheduler_t* sched);

void cgc_pd_scheduler_reset_stats(cgc_pd_scheduler_t* sched);

cgc_pd_layer_assignment_t cgc_pd_layer_assignment_by_ratio(int total_layers, double prefill_ratio);

cgc_pd_layer_assignment_t cgc_pd_layer_assignment_custom(const int* prefill, int prefill_count,
                                                          const int* decode, int decode_count);

bool cgc_pd_is_prefill_layer(const cgc_pd_layer_assignment_t* a, int layer);
bool cgc_pd_is_decode_layer(const cgc_pd_layer_assignment_t* a, int layer);
int cgc_pd_get_device_for_layer(const cgc_pd_layer_assignment_t* a, int layer);

#ifdef __cplusplus
}
#endif

#endif
