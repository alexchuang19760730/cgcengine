#pragma once

#include "cgc_cpp.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CGC_ENGINE_ORCHESTRATOR = 0,
    CGC_ENGINE_SIMD = 1,
    CGC_ENGINE_DEVICE = 2,
} cgc_engine_type_t;

typedef enum {
    BATCH_STRATEGY_STATIC = 0,
    BATCH_STRATEGY_CONTINUOUS = 1,
    BATCH_STRATEGY_DYNAMIC = 2,
    BATCH_STRATEGY_SPECULATIVE = 3,
} cgc_batch_strategy_t;

typedef enum {
    PD_PHASE_PREFILL = 0,
    PD_PHASE_DECODE = 1,
    PD_PHASE_HYBRID = 2,
    PD_PHASE_UNKNOWN = 3,
} cgc_pd_phase_t;

typedef struct {
    cgc_engine_type_t type;
    cgc_backend_t backend;
    char name[64];
    bool initialized;
    int num_ops;
} cgc_engine_info_t;

typedef struct {
    int32_t batch_size;
    int32_t num_batches;
    float batch_utilization;
    cgc_batch_strategy_t strategy;
    int32_t max_batch_size;
    int32_t max_tokens;
} cgc_orchestrator_config_t;

typedef struct {
    int32_t prefill_chunk_size;
    int32_t prefill_batch_size;
    int32_t decode_batch_size;
    int32_t hybrid_threshold;
    cgc_pd_phase_t current_phase;
} cgc_pd_config_t;

typedef struct {
    int32_t prefix_cache_hits;
    int32_t prefix_cache_misses;
    float prefix_cache_hit_rate;
    int32_t num_prefix_reuses;
    bool prefix_cache_enabled;
} cgc_prefix_cache_stats_t;

typedef struct {
    int32_t tokens_processed;
    float tokens_per_second;
    float avg_waiting_time_ms;
    float max_waiting_time_ms;
    float prefill_latency_ms;
    float decode_latency_ms;
    float total_latency_ms;
    float ttft_ms;
    float tpot_ms;
} cgc_throughput_stats_t;

typedef struct {
    int32_t total_blocks;
    int32_t free_blocks;
    int32_t used_blocks;
    float block_utilization;
    int64_t cache_memory_bytes;
    int64_t max_cache_memory_bytes;
} cgc_device_cache_config_t;

typedef struct {
    bool jit_enabled;
    bool gds_enabled;
    bool spdk_enabled;
    int64_t io_bandwidth_mbps;
    int32_t prefetch_depth;
} cgc_device_io_config_t;

typedef struct {
    cgc_device_io_config_t io;
    cgc_device_cache_config_t cache;
    int64_t total_device_memory_bytes;
    int64_t used_device_memory_bytes;
    float device_memory_utilization;
} cgc_device_config_t;

typedef struct {
    cgc_orchestrator_config_t orchestrator;
    cgc_pd_config_t pd_config;
    cgc_prefix_cache_stats_t prefix_cache;
    cgc_throughput_stats_t throughput;
    cgc_engine_info_t engine_info;
} cgc_runtime_stats_t;

cgc_error_t cgc_get_engine_info(cgc_engine_type_t engine_type, cgc_engine_info_t* info);
cgc_error_t cgc_init_engine(cgc_engine_type_t engine_type, cgc_backend_t backend);
cgc_error_t cgc_destroy_engine(cgc_engine_type_t engine_type);
cgc_error_t cgc_get_runtime_stats(cgc_runtime_stats_t* stats);
cgc_error_t cgc_reset_stats(void);

cgc_error_t cgc_orchestrator_set_batch_strategy(cgc_batch_strategy_t strategy);
cgc_error_t cgc_orchestrator_set_pd_config(const cgc_pd_config_t* config);
cgc_error_t cgc_orchestrator_schedule(void);

cgc_error_t cgc_device_set_io_config(const cgc_device_io_config_t* config);
cgc_error_t cgc_device_set_cache_config(const cgc_device_cache_config_t* config);
cgc_error_t cgc_device_allocate(int32_t num_blocks);
cgc_error_t cgc_device_free(void);
cgc_error_t cgc_device_get_stats(cgc_device_config_t* stats);

#ifdef __cplusplus
}
#endif
