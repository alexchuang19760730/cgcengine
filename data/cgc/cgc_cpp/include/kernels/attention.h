
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void attention_sdpa(
    const float* q, const float* k, const float* v, float* out,
    int64_t batch, int64_t heads, int64_t seqlen, int64_t d
);

#ifdef __cplusplus
}
#endif

