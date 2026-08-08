#include "kernels/linear.h"
#include <stdio.h>

void linear_gemm(
    const float* a, const float* b, float* c,
    int64_t m, int64_t n, int64_t k
) {
    printf("[CGC C++] linear_gemm (fallback) called: %ldx%ldx%ld)\n", m, n, k);

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}