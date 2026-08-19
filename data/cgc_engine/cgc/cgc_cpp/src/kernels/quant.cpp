#include "kernels/quant.h"
#include <stdio.h>
#include <cmath>

void quant_w8a16(
    const float* x, float* out,
    int64_t m, int64_t n
) {
    printf("[CGC C++] quant_w8a16 (fallback) called: %ldx%ld\n", m, n);

    for (int64_t i = 0; i < m * n; i++) {
        out[i] = x[i];
    }
}

void quant_gguf_q4(
    const float* x, float* out,
    int64_t m, int64_t n
) {
    printf("[CGC C++] quant_gguf_q4 (fallback) called: %ldx%ld\n", m, n);

    for (int64_t i = 0; i < m * n; i++) {
        out[i] = x[i];
    }
}