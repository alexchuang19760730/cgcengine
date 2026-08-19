#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void activation_silu(
    const float* x, float* out, int64_t size
);

void activation_gelu(
    const float* x, float* out, int64_t size
);

void activation_sigmoid(
    const float* x, float* out, int64_t size
);

void softmax(
    const float* x, float* out, int64_t batch, int64_t seqlen, int64_t dim
);

#ifdef __cplusplus
}
#endif