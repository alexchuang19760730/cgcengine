#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void rms_norm(
    const float* x, const float* weight, float* out,
    float eps, int64_t batch, int64_t seqlen, int64_t d
);

#ifdef __cplusplus
}
#endif