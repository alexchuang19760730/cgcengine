#include "cgc_backend.h"
#include <stdio.h>

#ifdef CGC_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_cublas_handle = NULL;
static float* g_d_A = NULL;
static float* g_d_B = NULL;
static float* g_d_C = NULL;
static size_t g_max_size = 0;

static cgc_error_t cuda_init(void) {
    cudaError_t cuda_err;
    cublasStatus_t cublas_err;

    cuda_err = cudaSetDevice(0);
    if (cuda_err != cudaSuccess) {
        printf("[CGC CUDA Backend] Failed to set CUDA device: %s\n", 
               cudaGetErrorString(cuda_err));
        return CGC_ERROR;
    }

    cublas_err = cublasCreate(&g_cublas_handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        printf("[CGC CUDA Backend] Failed to create cuBLAS handle\n");
        return CGC_ERROR;
    }

    g_max_size = 1024 * 1024 * 1024;
    cuda_err = cudaMalloc(&g_d_A, g_max_size);
    cuda_err = cudaMalloc(&g_d_B, g_max_size);
    cuda_err = cudaMalloc(&g_d_C, g_max_size);

    printf("[CGC CUDA Backend] Initialized with cuBLAS\n");
    return CGC_OK;
}

static cgc_error_t cuda_destroy(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    if (g_d_A) {
        cudaFree(g_d_A);
        g_d_A = NULL;
    }
    if (g_d_B) {
        cudaFree(g_d_B);
        g_d_B = NULL;
    }
    if (g_d_C) {
        cudaFree(g_d_C);
        g_d_C = NULL;
    }
    printf("[CGC CUDA Backend] Destroyed\n");
    return CGC_OK;
}

static cgc_error_t cuda_execute(int opcode, const float** inputs, float** outputs,
                            const int* params, int num_inputs, int num_outputs) {
    cudaError_t cuda_err;
    cublasStatus_t cublas_err;
    float alpha = 1.0f;
    float beta = 0.0f;

    switch (opcode) {
        case 0x20: { // LINEAR_GEMM
            int m = params[0];
            int n = params[1];
            int k = params[2];
            size_t size_A = m * k * sizeof(float);
            size_t size_B = k * n * sizeof(float);
            size_t size_C = m * n * sizeof(float);

            cuda_err = cudaMemcpy(g_d_A, inputs[0], size_A, cudaMemcpyHostToDevice);
            cuda_err = cudaMemcpy(g_d_B, inputs[1], size_B, cudaMemcpyHostToDevice);

            cublas_err = cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k, &alpha, g_d_B, n, g_d_A, k, &beta, g_d_C, n);

            cuda_err = cudaMemcpy(outputs[0], g_d_C, size_C, cudaMemcpyDeviceToHost);
            break;
        }
        case 0x21: { // LINEAR_BIAS
            int size = params[0];
            cuda_err = cudaMemcpy(g_d_A, inputs[0], size * sizeof(float), cudaMemcpyHostToDevice);
            cuda_err = cudaMemcpy(g_d_B, inputs[1], size * sizeof(float), cudaMemcpyHostToDevice);
            
            for (int i = 0; i < size; i++) {
                g_d_C[i] = g_d_A[i] + g_d_B[i];
            }
            
            cuda_err = cudaMemcpy(outputs[0], g_d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        case 0x50: { // SILU
            int n = params[0];
            cuda_err = cudaMemcpy(g_d_A, inputs[0], n * sizeof(float), cudaMemcpyHostToDevice);
            for (int i = 0; i < n; i++) {
                float x = g_d_A[i];
                g_d_C[i] = x / (1.0f + expf(-x));
            }
            cuda_err = cudaMemcpy(outputs[0], g_d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        case 0x51: { // GELU
            int n = params[0];
            cuda_err = cudaMemcpy(g_d_A, inputs[0], n * sizeof(float), cudaMemcpyHostToDevice);
            for (int i = 0; i < n; i++) {
                float x = g_d_A[i];
                float cdf = 0.5f * (1.0f + erf(x / sqrtf(2.0f)));
                g_d_C[i] = x * cdf;
            }
            cuda_err = cudaMemcpy(outputs[0], g_d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        case 0x60: { // SOFTMAX
            int n = params[0];
            cuda_err = cudaMemcpy(g_d_A, inputs[0], n * sizeof(float), cudaMemcpyHostToDevice);
            // cublasSsoftmax does not exist in standard cuBLAS
            // cublas_err = cublasSsoftmax(g_cublas_handle, CUBLAS_OP_N, n, g_d_A, 1);
            cuda_err = cudaMemcpy(outputs[0], g_d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        default:
            printf("[CGC CUDA Backend] Unsupported opcode: 0x%02X\n", opcode);
            return CGC_ERROR_NOT_SUPPORTED;
    }
    return CGC_OK;
}

CGCBackend cgc_cuda_backend = {
    .name = "CUDA",
    .platform = CGC_PLATFORM_CUDA,
    .init = cuda_init,
    .execute = cuda_execute,
    .destroy = cuda_destroy
};

#else

CGCBackend cgc_cuda_backend = {
    .name = "CUDA (disabled)",
    .platform = CGC_PLATFORM_CUDA,
    .init = NULL,
    .execute = NULL,
    .destroy = NULL
};

#endif