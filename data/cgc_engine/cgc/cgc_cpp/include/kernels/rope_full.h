#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void rope_hf(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* freq_base
);

void rope_yarn(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    float base,
    float scale,
    const float* freq_base
);

void rope_long(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    float long_factor,
    const float* freq_base
);

void rope_gptj(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* freq_base
);

void rope_fast(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* cos_cache,
    const float* sin_cache
);

#ifdef __cplusplus
}
#endif