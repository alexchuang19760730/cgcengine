
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void linear_gemm(
    const float* a, const float* b, float* c,
    int64_t m, int64_t n, int64_t k
);

#ifdef __cplusplus
}
#endif

