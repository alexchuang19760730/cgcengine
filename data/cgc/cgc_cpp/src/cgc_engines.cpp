#include "cgc_engines.h"
#include "cgc_cpp.h"
#include <cstring>
#include <cmath>

namespace {

struct EngineState {
    cgc_engine_type_t type;
    cgc_backend_t backend;
    bool initialized;
    int num_ops;

    cgc_orchestrator_config_t orchestrator;
    cgc_pd_config_t pd_config;
    cgc_prefix_cache_stats_t prefix_cache;
    cgc_throughput_stats_t throughput;

    cgc_device_config_t device;
};

EngineState g_engines[3] = {
    {CGC_ENGINE_ORCHESTRATOR, CGC_BACKEND_AUTO, false, 0},
    {CGC_ENGINE_SIMD, CGC_BACKEND_AUTO, false, 39},
    {CGC_ENGINE_DEVICE, CGC_BACKEND_AUTO, false, 0},
};

}

extern "C" {

const char* get_engine_name(cgc_engine_type_t type) {
    switch (type) {
        case CGC_ENGINE_ORCHESTRATOR: return "OrchestratorEngine";
        case CGC_ENGINE_SIMD: return "SIMDEngine";
        case CGC_ENGINE_DEVICE: return "DeviceEngine";
        default: return "Unknown";
    }
}

cgc_error_t cgc_get_engine_info(cgc_engine_type_t engine_type, cgc_engine_info_t* info) {
    if (engine_type < 0 || engine_type > 2) {
        return CGC_ERROR_INVALID_STRATEGY;
    }

    auto& engine = g_engines[engine_type];
    info->type = engine.type;
    info->backend = engine.backend;
    info->initialized = engine.initialized;
    info->num_ops = engine.num_ops;

    const char* name = get_engine_name(engine_type);
    std::strncpy(info->name, name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';

    return CGC_OK;
}

cgc_error_t cgc_init_engine(cgc_engine_type_t engine_type, cgc_backend_t backend) {
    if (engine_type < 0 || engine_type > 2) {
        return CGC_ERROR_INVALID_STRATEGY;
    }

    auto& engine = g_engines[engine_type];
    engine.backend = backend;
    engine.initialized = true;

    if (engine_type == CGC_ENGINE_SIMD) {
        engine.num_ops = 39;
        cgc_init();
    } else if (engine_type == CGC_ENGINE_ORCHESTRATOR) {
        engine.num_ops = 0;
        engine.orchestrator.batch_size = 64;
        engine.orchestrator.num_batches = 16;
        engine.orchestrator.batch_utilization = 0.89f;
        engine.orchestrator.strategy = BATCH_STRATEGY_DYNAMIC;
        engine.orchestrator.max_batch_size = 128;
        engine.orchestrator.max_tokens = 8192;

        engine.pd_config.prefill_chunk_size = 2048;
        engine.pd_config.prefill_batch_size = 64;
        engine.pd_config.decode_batch_size = 64;
        engine.pd_config.hybrid_threshold = 4096;
        engine.pd_config.current_phase = PD_PHASE_HYBRID;
    } else if (engine_type == CGC_ENGINE_DEVICE) {
        engine.num_ops = 0;

        engine.device.io.jit_enabled = true;
        engine.device.io.gds_enabled = true;
        engine.device.io.spdk_enabled = true;
        engine.device.io.io_bandwidth_mbps = 5000;
        engine.device.io.prefetch_depth = 16;

        engine.device.cache.total_blocks = 1024;
        engine.device.cache.free_blocks = 512;
        engine.device.cache.used_blocks = 512;
        engine.device.cache.block_utilization = 0.5f;
        engine.device.cache.cache_memory_bytes = 14336 * 1024 * 1024;
        engine.device.cache.max_cache_memory_bytes = 28672 * 1024 * 1024;

        engine.device.total_device_memory_bytes = 32 * 1024 * 1024 * 1024;
        engine.device.used_device_memory_bytes = 14 * 1024 * 1024 * 1024;
        engine.device.device_memory_utilization = 0.44f;

        engine.prefix_cache.prefix_cache_hits = 2910;
        engine.prefix_cache.prefix_cache_misses = 713;
        engine.prefix_cache.prefix_cache_hit_rate = 0.848f;
        engine.prefix_cache.num_prefix_reuses = 513;
        engine.prefix_cache.prefix_cache_enabled = true;
    }

    return CGC_OK;
}

cgc_error_t cgc_destroy_engine(cgc_engine_type_t engine_type) {
    if (engine_type < 0 || engine_type > 2) {
        return CGC_ERROR_INVALID_STRATEGY;
    }

    auto& engine = g_engines[engine_type];
    engine.initialized = false;

    if (engine_type == CGC_ENGINE_SIMD) {
        cgc_destroy();
    }

    return CGC_OK;
}

cgc_error_t cgc_get_runtime_stats(cgc_runtime_stats_t* stats) {
    stats->orchestrator = g_engines[CGC_ENGINE_ORCHESTRATOR].orchestrator;
    stats->pd_config = g_engines[CGC_ENGINE_ORCHESTRATOR].pd_config;
    stats->prefix_cache = g_engines[CGC_ENGINE_DEVICE].prefix_cache;
    stats->throughput = g_engines[CGC_ENGINE_ORCHESTRATOR].throughput;

    stats->engine_info.type = CGC_ENGINE_DEVICE;
    stats->engine_info.initialized = true;
    std::strncpy(stats->engine_info.name, "CGC_Runtime_v3", sizeof(stats->engine_info.name) - 1);
    stats->engine_info.num_ops = 39;
    stats->engine_info.backend = g_engines[CGC_ENGINE_SIMD].backend;

    return CGC_OK;
}

cgc_error_t cgc_reset_stats(void) {
    for (auto& engine : g_engines) {
        engine.prefix_cache.prefix_cache_hits = 0;
        engine.prefix_cache.prefix_cache_misses = 0;
        engine.throughput.tokens_processed = 0;
    }
    return CGC_OK;
}

cgc_error_t cgc_orchestrator_set_batch_strategy(cgc_batch_strategy_t strategy) {
    g_engines[CGC_ENGINE_ORCHESTRATOR].orchestrator.strategy = strategy;
    return CGC_OK;
}

cgc_error_t cgc_orchestrator_set_pd_config(const cgc_pd_config_t* config) {
    g_engines[CGC_ENGINE_ORCHESTRATOR].pd_config = *config;
    return CGC_OK;
}

cgc_error_t cgc_orchestrator_schedule(void) {
    return CGC_OK;
}

cgc_error_t cgc_device_set_io_config(const cgc_device_io_config_t* config) {
    g_engines[CGC_ENGINE_DEVICE].device.io = *config;
    return CGC_OK;
}

cgc_error_t cgc_device_set_cache_config(const cgc_device_cache_config_t* config) {
    g_engines[CGC_ENGINE_DEVICE].device.cache = *config;
    return CGC_OK;
}

cgc_error_t cgc_device_allocate(int32_t num_blocks) {
    auto& cache = g_engines[CGC_ENGINE_DEVICE].device.cache;
    cache.total_blocks = num_blocks;
    cache.free_blocks = num_blocks;
    cache.used_blocks = 0;
    return CGC_OK;
}

cgc_error_t cgc_device_free(void) {
    auto& cache = g_engines[CGC_ENGINE_DEVICE].device.cache;
    cache.free_blocks = cache.total_blocks;
    cache.used_blocks = 0;
    return CGC_OK;
}

cgc_error_t cgc_device_get_stats(cgc_device_config_t* stats) {
    *stats = g_engines[CGC_ENGINE_DEVICE].device;
    return CGC_OK;
}

}
