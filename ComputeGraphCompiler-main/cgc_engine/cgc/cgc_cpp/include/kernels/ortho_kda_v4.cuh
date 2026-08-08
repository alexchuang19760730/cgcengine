#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

constexpr int ORTHO_KDA_V4_N_BASE = 128;
constexpr int ORTHO_KDA_V4_HEAD_DIM = 512;
constexpr float ORTHO_KDA_V4_DEFAULT_DECAY = 0.01f;
constexpr float ORTHO_KDA_V4_EPS = 1e-8f;

struct OrthoKDAKV_v4 {
    float K[ORTHO_KDA_V4_N_BASE][ORTHO_KDA_V4_HEAD_DIM];
    float V[ORTHO_KDA_V4_N_BASE][ORTHO_KDA_V4_HEAD_DIM];
    float decay[ORTHO_KDA_V4_N_BASE];
    int idx;
    int num_heads;
    int head_dim;
    int ortho_base_dim;
};

#ifndef __CUDACC__
typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#endif

cudaError_t ortho_kda_v4_alloc_kv(
    OrthoKDAKV_v4** kv,
    int num_heads,
    int head_dim,
    int ortho_base_dim,
    cudaStream_t stream
);

cudaError_t ortho_kda_v4_free_kv(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
);

cudaError_t ortho_kda_v4_reset(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
);

cudaError_t ortho_kda_v4_update(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    cudaStream_t stream
);

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
);

cudaError_t ortho_kda_v4_get_state(
    OrthoKDAKV_v4* kv,
    float* K_out,
    float* V_out,
    float* decay_out,
    int* idx_out,
    cudaStream_t stream
);

#ifdef __CUDACC__
__device__ __forceinline__ void ortho_kda_v4_gram_schmidt(
    float* v,
    const float (*basis)[ORTHO_KDA_V4_HEAD_DIM],
    int n,
    int head_dim
) {
    for (int i = 0; i < n; i++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += v[d] * basis[i][d];
        }
        for (int d = 0; d < head_dim; d++) {
            v[d] -= dot * basis[i][d];
        }
    }

    float norm = ORTHO_KDA_V4_EPS;
    for (int d = 0; d < head_dim; d++) {
        norm += v[d] * v[d];
    }
    norm = rsqrtf(norm);
    for (int d = 0; d < head_dim; d++) {
        v[d] *= norm;
    }
}

__global__ void ortho_kda_v4_update_kernel(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    int batch_size,
    int num_heads,
    int head_dim
);

__global__ void ortho_kda_v4_forward_kernel(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    int batch_size,
    int num_heads,
    int head_dim
);

__device__ void ortho_kda_v4_gram_schmidt_single(
    float* v,
    const float (*basis)[ORTHO_KDA_V4_HEAD_DIM],
    int n,
    int head_dim
);
#endif
