#include "kernels/ortho_kda_v4.cuh"
#include "cgc_cpp.h"
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace {

#ifdef __CUDACC__
constexpr int ORTHO_KDA_V4_THREADS_PER_BLOCK = 128;

__global__ void ortho_kda_v4_reset_kernel(OrthoKDAKV_v4* kv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv->ortho_base_dim) return;

    for (int d = 0; d < ORTHO_KDA_V4_HEAD_DIM; d++) {
        kv->K[i][d] = 0.0f;
        kv->V[i][d] = 0.0f;
    }
    kv->decay[i] = 0.0f;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        kv->idx = 0;
    }
}

__global__ void ortho_kda_v4_get_state_kernel(
    const OrthoKDAKV_v4* kv,
    float* K_out,
    float* V_out,
    float* decay_out,
    int* idx_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv->ortho_base_dim) return;

    int head_dim = kv->head_dim;

    for (int d = 0; d < head_dim; d++) {
        K_out[i * head_dim + d] = kv->K[i][d];
        V_out[i * head_dim + d] = kv->V[i][d];
    }
    decay_out[i] = kv->decay[i];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *idx_out = kv->idx;
    }
}
#endif

} // anonymous namespace

#ifdef __CUDACC__
cudaError_t ortho_kda_v4_alloc_kv(
    OrthoKDAKV_v4** kv,
    int num_heads,
    int head_dim,
    int ortho_base_dim,
    cudaStream_t stream
) {
    cudaError_t err;

    size_t kv_size = sizeof(OrthoKDAKV_v4);
    OrthoKDAKV_v4* kv_host = new OrthoKDAKV_v4();
    kv_host->num_heads = num_heads;
    kv_host->head_dim = head_dim;
    kv_host->ortho_base_dim = ortho_base_dim;
    kv_host->idx = 0;

    for (int i = 0; i < ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            kv_host->K[i][d] = 0.0f;
            kv_host->V[i][d] = 0.0f;
        }
        kv_host->decay[i] = 0.0f;
    }

    err = cudaMalloc(kv, kv_size);
    if (err != cudaSuccess) {
        delete kv_host;
        return err;
    }

    err = cudaMemcpy(*kv, kv_host, kv_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*kv);
        delete kv_host;
        return err;
    }

    delete kv_host;
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_free_kv(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    if (kv != nullptr) {
        return cudaFree(kv);
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_reset(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    int blocks = (kv->ortho_base_dim + ORTHO_KDA_V4_THREADS_PER_BLOCK - 1) / ORTHO_KDA_V4_THREADS_PER_BLOCK;
    ortho_kda_v4_reset_kernel<<<blocks, ORTHO_KDA_V4_THREADS_PER_BLOCK, 0, stream>>>(kv);
    return cudaGetLastError();
}

cudaError_t ortho_kda_v4_update(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    cudaStream_t stream
) {
    ortho_kda_v4_update_single_kernel<<<1, kv->ortho_base_dim, 0, stream>>>(
        kv, key, value
    );

    if (kv->idx < kv->ortho_base_dim - 1) {
        kv->idx++;
    }

    return cudaGetLastError();
}

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = kv->num_heads;

    ortho_kda_v4_forward_kernel<<<blocks, threads, 0, stream>>>(
        kv, query, output, kv->num_heads, kv->head_dim
    );

    return cudaGetLastError();
}

cudaError_t ortho_kda_v4_get_state(
    OrthoKDAKV_v4* kv,
    float* K_out,
    float* V_out,
    float* decay_out,
    int* idx_out,
    cudaStream_t stream
) {
    int blocks = (kv->ortho_base_dim + ORTHO_KDA_V4_THREADS_PER_BLOCK - 1) / ORTHO_KDA_V4_THREADS_PER_BLOCK;

    ortho_kda_v4_get_state_kernel<<<blocks, ORTHO_KDA_V4_THREADS_PER_BLOCK, 0, stream>>>(
        kv, K_out, V_out, decay_out, idx_out
    );

    return cudaGetLastError();
}

__global__ void ortho_kda_v4_update_single_kernel(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value
) {
    int i = threadIdx.x;
    if (i >= ORTHO_KDA_V4_N_BASE) return;

    float k[ORTHO_KDA_V4_HEAD_DIM];
    for (int d = 0; d < ORTHO_KDA_V4_HEAD_DIM; d++) {
        k[d] = key[d];
    }

    ortho_kda_v4_gram_schmidt(k, kv->K, i, ORTHO_KDA_V4_HEAD_DIM);

    for (int d = 0; d < ORTHO_KDA_V4_HEAD_DIM; d++) {
        atomicAdd(&kv->K[i][d], k[d]);
        atomicAdd(&kv->V[i][d], value[d]);
    }

    kv->decay[i] = expf(-ORTHO_KDA_V4_DEFAULT_DECAY * static_cast<float>(i));
}

__global__ void ortho_kda_v4_forward_kernel(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    int batch_size,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    int tid = threadIdx.x;
    float sum[ORTHO_KDA_V4_HEAD_DIM];

    for (int d = 0; d < head_dim; d++) {
        sum[d] = 0.0f;
    }

    for (int i = 0; i < kv->idx; i++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int idx = head_idx * head_dim + d;
            score += query[idx] * kv->K[i][d];
        }

        float weighted_score = score * kv->decay[i];

        for (int d = 0; d < head_dim; d++) {
            int idx = head_idx * head_dim + d;
            sum[d] += weighted_score * kv->V[i][d];
        }
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        int idx = head_idx * head_dim + d;
        output[idx] = sum[d];
    }
}

#else

struct OrthoKDAKV_v4;
typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaGetLastError() cudaSuccess

static OrthoKDAKV_v4* g_kv_placeholder = nullptr;

cudaError_t ortho_kda_v4_alloc_kv(
    OrthoKDAKV_v4** kv,
    int num_heads,
    int head_dim,
    int ortho_base_dim,
    cudaStream_t stream
) {
    *kv = nullptr;
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_free_kv(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_reset(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_update(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    cudaStream_t stream
) {
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
) {
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_get_state(
    OrthoKDAKV_v4* kv,
    float* K_out,
    float* V_out,
    float* decay_out,
    int* idx_out,
    cudaStream_t stream
) {
    return cudaSuccess;
}

#endif
