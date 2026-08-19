#include "kernels/gemm_cpu.h"
#include <string.h>

void gemm_cpu(const float* A, const float* B, float* C, int m, int n, int k) {
    memset(C, 0, m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void gemm_batched_cpu(const float* A, const float* B, float* C,
                      int batch_size, int m, int n, int k) {
    for (int b = 0; b < batch_size; b++) {
        const float* A_batch = A + b * m * k;
        const float* B_batch = B + b * k * n;
        float* C_batch = C + b * m * n;
        gemm_cpu(A_batch, B_batch, C_batch, m, n, k);
    }
}