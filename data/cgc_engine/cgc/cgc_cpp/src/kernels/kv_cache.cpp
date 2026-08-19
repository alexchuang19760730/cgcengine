#include "kernels/kv_cache.h"
#include <cstring>

void kv_cache_load(
    const float* cache,
    float* out,
    const int64_t* indices,
    int64_t cache_size,
    int64_t num_indices,
    int64_t elem_size
) {
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = indices[i];
        if (idx >= 0 && idx < cache_size) {
            memcpy(out + i * elem_size, cache + idx * elem_size, elem_size * sizeof(float));
        }
    }
}

void kv_cache_store(
    float* cache,
    const float* values,
    const int64_t* indices,
    int64_t cache_size,
    int64_t num_indices,
    int64_t elem_size
) {
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = indices[i];
        if (idx >= 0 && idx < cache_size) {
            memcpy(cache + idx * elem_size, values + i * elem_size, elem_size * sizeof(float));
        }
    }
}

void kv_cache_update(
    float* cache,
    const float* new_values,
    const int64_t* positions,
    int64_t cache_size,
    int64_t num_updates,
    int64_t elem_size
) {
    for (int64_t i = 0; i < num_updates; ++i) {
        int64_t pos = positions[i];
        if (pos >= 0 && pos < cache_size) {
            memcpy(cache + pos * elem_size, new_values + i * elem_size, elem_size * sizeof(float));
        }
    }
}