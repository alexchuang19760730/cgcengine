#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void rope_apply(
    const float* x, const float* pos, float* out,
    int64_t batch, int64_t seqlen, int64_t d
);

#ifdef __cplusplus
}
#endif