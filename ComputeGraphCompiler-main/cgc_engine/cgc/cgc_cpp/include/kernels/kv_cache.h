#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void kv_cache_load(
    const float* cache,
    float* out,
    const int64_t* indices,
    int64_t cache_size,
    int64_t num_indices,
    int64_t elem_size
);

void kv_cache_store(
    float* cache,
    const float* values,
    const int64_t* indices,
    int64_t cache_size,
    int64_t num_indices,
    int64_t elem_size
);

void kv_cache_update(
    float* cache,
    const float* new_values,
    const int64_t* positions,
    int64_t cache_size,
    int64_t num_updates,
    int64_t elem_size
);

#ifdef __cplusplus
}
#endif