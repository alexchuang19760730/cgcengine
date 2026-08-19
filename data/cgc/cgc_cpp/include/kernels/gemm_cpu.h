#ifndef GEMM_CPU_H
#define GEMM_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gemm_cpu(const float* A, const float* B, float* C, int m, int n, int k);

void gemm_batched_cpu(const float* A, const float* B, float* C, 
                      int batch_size, int m, int n, int k);

#ifdef __cplusplus
}
#endif

#endif // GEMM_CPU_H