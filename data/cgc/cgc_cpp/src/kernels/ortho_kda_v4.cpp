#include "kernels/ortho_kda_v4.cuh"
#include "cgc_cpp.h"
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#ifdef __CUDACC__
__global__ void ortho_kda_v4_update_single_kernel(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value
);

__global__ void ortho_kda_v4_forward_kernel(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    int num_heads,
    int head_dim
);
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
    // Use cudaMallocManaged so both CPU and GPU can access kv-> fields
    cudaError_t err;

    size_t kv_size = sizeof(OrthoKDAKV_v4);
    err = cudaMallocManaged(kv, kv_size);
    if (err != cudaSuccess) {
        return err;
    }

    (*kv)->num_heads = num_heads;
    (*kv)->head_dim = head_dim;
    (*kv)->ortho_base_dim = ortho_base_dim;
    (*kv)->idx = 0;

    for (int i = 0; i < ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            (*kv)->K[i][d] = 0.0f;
            (*kv)->V[i][d] = 0.0f;
        }
        (*kv)->decay[i] = 0.0f;
    }

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
    // Copy key/value from CPU to GPU temp buffers
    size_t vec_size = ORTHO_KDA_V4_HEAD_DIM * sizeof(float);
    float* d_key = nullptr;
    float* d_value = nullptr;
    cudaMalloc(&d_key, vec_size);
    cudaMalloc(&d_value, vec_size);
    cudaMemcpy(d_key, key, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, vec_size, cudaMemcpyHostToDevice);

    ortho_kda_v4_update_single_kernel<<<1, kv->ortho_base_dim, 0, stream>>>(
        kv, d_key, d_value
    );

    cudaDeviceSynchronize();

    // kv is unified memory, so CPU can access idx
    if (kv->idx < kv->ortho_base_dim - 1) {
        kv->idx++;
    }

    cudaFree(d_key);
    cudaFree(d_value);

    return cudaGetLastError();
}

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
) {
    // kv is unified memory, CPU can read num_heads/head_dim
    int num_heads = kv->num_heads;
    int head_dim = kv->head_dim;

    // Copy query from CPU to GPU
    size_t q_size = num_heads * head_dim * sizeof(float);
    float* d_query = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_query, q_size);
    cudaMalloc(&d_output, q_size);
    cudaMemcpy(d_query, query, q_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = num_heads;

    ortho_kda_v4_forward_kernel<<<blocks, threads, 0, stream>>>(
        kv, d_query, d_output, num_heads, head_dim
    );

    cudaDeviceSynchronize();

    // Copy output from GPU to CPU
    cudaMemcpy(output, d_output, q_size, cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_output);

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
    // kv is unified memory, CPU can read fields directly
    int ortho_base_dim = kv->ortho_base_dim;
    int head_dim = kv->head_dim;

    // Allocate GPU buffers for kernel output
    size_t kv_size = ortho_base_dim * head_dim * sizeof(float);
    size_t decay_size = ortho_base_dim * sizeof(float);
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_decay = nullptr;
    int* d_idx = nullptr;
    cudaMalloc(&d_K, kv_size);
    cudaMalloc(&d_V, kv_size);
    cudaMalloc(&d_decay, decay_size);
    cudaMalloc(&d_idx, sizeof(int));

    int blocks = (ortho_base_dim + ORTHO_KDA_V4_THREADS_PER_BLOCK - 1) / ORTHO_KDA_V4_THREADS_PER_BLOCK;

    ortho_kda_v4_get_state_kernel<<<blocks, ORTHO_KDA_V4_THREADS_PER_BLOCK, 0, stream>>>(
        kv, d_K, d_V, d_decay, d_idx
    );

    cudaDeviceSynchronize();

    // Copy results to CPU
    cudaMemcpy(K_out, d_K, kv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V_out, d_V, kv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(decay_out, d_decay, decay_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(idx_out, d_idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_decay);
    cudaFree(d_idx);

    return cudaSuccess;
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

#include <cmath>

void ortho_kda_v4_gram_schmidt_cpu(
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
    norm = 1.0f / std::sqrt(norm);
    for (int d = 0; d < head_dim; d++) {
        v[d] *= norm;
    }
}

cudaError_t ortho_kda_v4_alloc_kv(
    OrthoKDAKV_v4** kv,
    int num_heads,
    int head_dim,
    int ortho_base_dim,
    cudaStream_t stream
) {
    *kv = new OrthoKDAKV_v4();
    (*kv)->num_heads = num_heads;
    (*kv)->head_dim = head_dim;
    (*kv)->ortho_base_dim = ortho_base_dim;
    (*kv)->idx = 0;

    for (int i = 0; i < ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            (*kv)->K[i][d] = 0.0f;
            (*kv)->V[i][d] = 0.0f;
        }
        (*kv)->decay[i] = 0.0f;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_free_kv(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    if (kv != nullptr) {
        delete kv;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_reset(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;
    kv->idx = 0;
    for (int i = 0; i < kv->ortho_base_dim; i++) {
        for (int d = 0; d < kv->head_dim; d++) {
            kv->K[i][d] = 0.0f;
            kv->V[i][d] = 0.0f;
        }
        kv->decay[i] = 0.0f;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_update(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;

    for (int i = 0; i < kv->ortho_base_dim; i++) {
        float k[ORTHO_KDA_V4_HEAD_DIM];
        for (int d = 0; d < kv->head_dim; d++) {
            k[d] = key[d];
        }

        ortho_kda_v4_gram_schmidt_cpu(k, kv->K, i, kv->head_dim);

        for (int d = 0; d < kv->head_dim; d++) {
            kv->K[i][d] += k[d];
            kv->V[i][d] += value[d];
        }
        kv->decay[i] = std::exp(-ORTHO_KDA_V4_DEFAULT_DECAY * static_cast<float>(i));
    }

    if (kv->idx < kv->ortho_base_dim - 1) {
        kv->idx++;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;

    int num_heads = kv->num_heads;
    int head_dim = kv->head_dim;

    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        float sum[ORTHO_KDA_V4_HEAD_DIM] = {0.0f};

        for (int i = 0; i < kv->idx; i++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int idx = head_idx * head_dim + d;
                score += query[idx] * kv->K[i][d];
            }

            float weighted_score = score * kv->decay[i];

            for (int d = 0; d < head_dim; d++) {
                sum[d] += weighted_score * kv->V[i][d];
            }
        }

        for (int d = 0; d < head_dim; d++) {
            int idx = head_idx * head_dim + d;
            output[idx] = sum[d];
        }
    }
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
    if (!kv) return cudaSuccess;

    int head_dim = kv->head_dim;
    for (int i = 0; i < kv->ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            K_out[i * head_dim + d] = kv->K[i][d];
            V_out[i * head_dim + d] = kv->V[i][d];
        }
        decay_out[i] = kv->decay[i];
    }
    *idx_out = kv->idx;
    return cudaSuccess;
}

#endif
