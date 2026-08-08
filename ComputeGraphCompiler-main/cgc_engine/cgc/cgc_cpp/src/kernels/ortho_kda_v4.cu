#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>
#include <cmath>

constexpr int N_BASE = 128;
constexpr int HEAD_DIM = 128;

struct OrthoKDAKV {
    float K[N_BASE][HEAD_DIM];
    float V[N_BASE][HEAD_DIM];
    float decay[N_BASE];
    int idx;
};

__device__ void gram_schmidt(float* v, const float (*basis)[HEAD_DIM], int n) {
    for (int i = 0; i < n; i++) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) dot += v[d] * basis[i][d];
        for (int d = 0; d < HEAD_DIM; d++) v[d] -= dot * basis[i][d];
    }
    float norm = 1e-8f;
    for (int d = 0; d < HEAD_DIM; d++) norm += v[d] * v[d];
    norm = rsqrtf(norm);
    for (int d = 0; d < HEAD_DIM; d++) v[d] *= norm;
}

__global__ void ortho_kda_v4_update(
    OrthoKDAKV* kv,
    const float* __restrict__ key,
    const float* __restrict__ value
) {
    const int i = threadIdx.x;
    if (i >= N_BASE) return;

    float k[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) k[d] = key[d];
    gram_schmidt(k, kv->K, i);

    for (int d = 0; d < HEAD_DIM; d++) {
        kv->K[i][d] = k[d];
        kv->V[i][d] += value[d];
    }
    kv->decay[i] = expf(-0.01f * (float)i);
}

__global__ void ortho_kda_v4_forward(
    const OrthoKDAKV* __restrict__ kv,
    const float* __restrict__ Q,
    float* __restrict__ out,
    const int num_heads
) {
    const int head_idx = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= N_BASE || head_idx >= num_heads) return;

    const float* q = Q + head_idx * HEAD_DIM;
    float* out_head = out + head_idx * HEAD_DIM;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) score += q[d] * kv->K[i][d];
    const float attn = score * kv->decay[i];

    for (int d = 0; d < HEAD_DIM; d++) out_head[d] += attn * kv->V[i][d];
}

extern "C" void call_ortho_kda_forward(
    OrthoKDAKV* kv, const float* Q, float* out, int num_heads) {
    dim3 grid(num_heads);
    dim3 block(N_BASE);
    ortho_kda_v4_forward<<<grid, block>>>(kv, Q, out, num_heads);
}

extern "C" void call_ortho_kda_update(
    OrthoKDAKV* kv, const float* key, const float* value) {
    dim3 block(N_BASE);
    ortho_kda_v4_update<<<1, block>>>(kv, key, value);
}

extern "C" void init_ortho_kda_kv(OrthoKDAKV* kv) {
    for (int i = 0; i < N_BASE; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            kv->K[i][d] = 0.0f;
            kv->V[i][d] = 0.0f;
        }
        kv->decay[i] = 1.0f;
    }
    kv->idx = 0;
}

torch::Tensor ortho_kda_forward(
    torch::Tensor kv_cache,
    torch::Tensor q,
    int num_heads
) {
    auto out = torch::zeros_like(q);
    call_ortho_kda_forward(
        (OrthoKDAKV*)kv_cache.data_ptr(),
        q.data_ptr<float>(),
        out.data_ptr<float>(),
        num_heads
    );
    return out;
}

torch::Tensor ortho_kda_update(
    torch::Tensor kv_cache,
    torch::Tensor key,
    torch::Tensor value
) {
    call_ortho_kda_update(
        (OrthoKDAKV*)kv_cache.data_ptr(),
        key.data_ptr<float>(),
        value.data_ptr<float>()
    );
    return kv_cache;
}

torch::Tensor create_kv_cache() {
    return torch::zeros(
        {N_BASE, HEAD_DIM * 2 + N_BASE + 1},
        torch::dtype(torch::kFloat32).device(torch::kCUDA)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ortho_kda_forward, "OrthoKDA v4 forward");
    m.def("update", &ortho_kda_update, "OrthoKDA v4 update");
    m.def("create_kv_cache", &create_kv_cache, "Create O(1) KV cache");
}