#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void layer_norm(
    const float* x, 
    float* out, 
    int64_t batch,
    int64_t seqlen,
    int64_t dim,
    const float* weight,
    const float* bias,
    float eps
);

void group_norm(
    const float* x, 
    float* out, 
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_groups,
    const float* weight,
    const float* bias,
    float eps
);

#ifdef __cplusplus
}
#endif