// =============================================================================
// WindowOrthoKVCompressor — CUDA kernel (aligned with ortho_kda_v4.cu style)
// -----------------------------------------------------------------------------
// 与 CPU 版 window_ortho_kv_compressor.cpp 严格对应：窗口作用域正交 KV 压缩器。
// 设计约束（白皮书 §4）：
//   1. 只接收 R-SWA 窗口内 K/V，绝不接收 Reference；
//   2. 淘汰由 R-SWA 窗口推进触发，命中后从「当前窗口原始 K/V」重建局部正交基
//      （re-orthogonalize），抑制 ring 漂移（风险 1）；
//   3. 保留 get/set_state，兼容 kda_state_v1 跨节点传输。
//
// 风格对齐 ortho_kda_v4.cu：
//   - 同样的 device Gram-Schmidt / kernel 命名 / host `extern "C"` 包装约定；
//   - 状态以「宿主侧镜像 struct（含 device 指针）+ 每步回拷标量」的方式管理，
//     与 v4 的 OrthoKDAKV 设备对象思路一致。
//
// 注意：本文件仅由 nvcc 编译（见 build_unified_cuda.sh）。
// =============================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

// -----------------------------------------------------------------------------
// Device state (mirrors window_ortho_kv_compressor struct; pointers are device)
// -----------------------------------------------------------------------------
struct WindowOrthoState {
    int    num_heads;
    int    head_dim;
    int    ortho_base_dim;   // B
    int    window_capacity;  // W
    float  decay_rate;

    float* win_k;            // [window_capacity][num_heads][head_dim]
    float* win_v;            // [window_capacity][num_heads][head_dim]
    int    win_head;
    int    win_count;

    float* K_basis;          // [num_heads][ortho_base_dim][head_dim]
    float* V_basis;          // [num_heads][ortho_base_dim][head_dim]
    float* decay;            // [ortho_base_dim]
    int    current_dim;
};

// Host-mirror -> device-struct copy mapping (kept in sync each call).
#include <unordered_map>
#include <cstdint>
static std::unordered_map<uintptr_t, WindowOrthoState*> host_to_device;

// Device Gram-Schmidt: subtract projection onto first n basis vectors.
__device__ void wok_gram_schmidt(float* v, const float* basis, int n, int head_dim) {
    for (int i = 0; i < n; ++i) {
        const float* bi = basis + i * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += v[d] * bi[d];
        for (int d = 0; d < head_dim; ++d) v[d] -= dot * bi[d];
    }
    float norm = 1e-8f;
    for (int d = 0; d < head_dim; ++d) norm += v[d] * v[d];
    norm = rsqrtf(norm);
    for (int d = 0; d < head_dim; ++d) v[d] *= norm;
}

// -----------------------------------------------------------------------------
// Feed + rebuild kernel (single block: ring append then re-orthogonalize)
// -----------------------------------------------------------------------------
__global__ void window_ortho_feed_kernel(
    WindowOrthoState* st,
    const float* __restrict__ k,   // [num_heads][head_dim]
    const float* __restrict__ v
) {
    const int nh  = st->num_heads;
    const int hd  = st->head_dim;
    const int cap = st->window_capacity;

    // --- ring append (mutated by the only writer) ---
    int phys;
    if (st->win_count < cap) {
        phys = (st->win_head + st->win_count) % cap;
        st->win_count++;
    } else {
        phys = st->win_head;
        st->win_head = (st->win_head + 1) % cap;
    }
    for (int h = 0; h < nh; ++h) {
        const float* kk = k + h * hd;
        const float* vv = v + h * hd;
        float* wk = st->win_k + ((size_t)phys * nh + h) * hd;
        float* wv = st->win_v + ((size_t)phys * nh + h) * hd;
        for (int d = 0; d < hd; ++d) { wk[d] = kk[d]; wv[d] = vv[d]; }
    }

    // --- rebuild local orthogonal basis over current window (re-orthogonalize) ---
    const int Wc = st->win_count;
    const int B  = (Wc < st->ortho_base_dim) ? Wc : st->ortho_base_dim;

    for (int h = 0; h < nh; ++h) {
        // basis buffer (B * hd) reused via K_basis working area is not thread-safe
        // across heads, but each head writes its own slice of K_basis/V_basis.
        float* basis = st->K_basis + (size_t)h * st->ortho_base_dim * hd; // owns B*hd
        for (int i = 0; i < B; ++i) {
            int phys_i = (st->win_head + i) % cap;
            const float* kw = st->win_k + ((size_t)phys_i * nh + h) * hd;
            float tmp[256];
            for (int d = 0; d < hd; ++d) tmp[d] = kw[d];
            wok_gram_schmidt(tmp, basis, i, hd);

            // record orthonormal vector e_i into K_basis[h][i]
            float* e_i = basis + i * hd;
            for (int d = 0; d < hd; ++d) e_i[d] = tmp[d];

            // V_basis[h][i] = sum_j (k_w[j] . e_i) * v_w[j]  (low-rank projection)
            float vb[256];
            for (int d = 0; d < hd; ++d) vb[d] = 0.0f;
            for (int j = 0; j < Wc; ++j) {
                int pj = (st->win_head + j) % cap;
                const float* kj = st->win_k + ((size_t)pj * nh + h) * hd;
                const float* vj = st->win_v + ((size_t)pj * nh + h) * hd;
                float proj = 0.0f;
                for (int d = 0; d < hd; ++d) proj += kj[d] * tmp[d];
                for (int d = 0; d < hd; ++d) vb[d] += proj * vj[d];
            }
            float* vb_out = st->V_basis + ((size_t)h * st->ortho_base_dim + i) * hd;
            for (int d = 0; d < hd; ++d) vb_out[d] = vb[d];

            // decay by age: newest token age 0 (highest weight)
            st->decay[i] = expf(-(float)(Wc - 1 - i) * st->decay_rate);
        }
    }
    st->current_dim = B;
}

// -----------------------------------------------------------------------------
// Attention kernel: Q . K_basis[i] * decay[i] -> accumulate V_basis[i]
// -----------------------------------------------------------------------------
__global__ void window_ortho_attention_kernel(
    const WindowOrthoState* __restrict__ st,
    const float* __restrict__ Q,   // [num_heads][head_dim]
    float* __restrict__ out         // [num_heads][head_dim]
) {
    const int h  = blockIdx.x;
    if (h >= st->num_heads) return;
    const int hd = st->head_dim;
    const int B  = st->ortho_base_dim;
    const float* qh = Q + h * hd;
    float* oh = out + h * hd;
    for (int d = 0; d < hd; ++d) oh[d] = 0.0f;

    for (int i = 0; i < st->current_dim; ++i) {
        const float* kb = st->K_basis + ((size_t)h * B + i) * hd;
        const float* vb = st->V_basis + ((size_t)h * B + i) * hd;
        float score = 0.0f;
        for (int d = 0; d < hd; ++d) score += qh[d] * kb[d];
        float w = score * st->decay[i];
        for (int d = 0; d < hd; ++d) oh[d] += w * vb[d];
    }
}

// -----------------------------------------------------------------------------
// Host wrappers (extern "C") — mirror v4's call_ortho_kda_* convention
// -----------------------------------------------------------------------------
extern "C" cudaError_t window_ortho_create_cuda(
    int num_heads, int head_dim, int ortho_base_dim, int window_capacity,
    float decay_rate, WindowOrthoState** out_state
) {
    WindowOrthoState* s = (WindowOrthoState*)malloc(sizeof(WindowOrthoState));
    s->num_heads = num_heads;
    s->head_dim = head_dim;
    s->ortho_base_dim = ortho_base_dim;
    s->window_capacity = window_capacity > 0 ? window_capacity : ortho_base_dim;
    s->decay_rate = decay_rate;
    s->win_head = 0;
    s->win_count = 0;
    s->current_dim = 0;

    size_t win_bytes = (size_t)s->window_capacity * num_heads * head_dim * sizeof(float);
    size_t basis_bytes = (size_t)num_heads * ortho_base_dim * head_dim * sizeof(float);
    size_t decay_bytes = (size_t)ortho_base_dim * sizeof(float);

    cudaMalloc(&s->win_k, win_bytes);
    cudaMalloc(&s->win_v, win_bytes);
    cudaMalloc(&s->K_basis, basis_bytes);
    cudaMalloc(&s->V_basis, basis_bytes);
    cudaMalloc(&s->decay, decay_bytes);

    // zero device buffers
    cudaMemset(s->win_k, 0, win_bytes);
    cudaMemset(s->win_v, 0, win_bytes);
    cudaMemset(s->K_basis, 0, basis_bytes);
    cudaMemset(s->V_basis, 0, basis_bytes);
    cudaMemset(s->decay, 0, decay_bytes);

    // device copy of the state struct (kept in sync each call)
    WindowOrthoState* d_s;
    cudaMalloc(&d_s, sizeof(WindowOrthoState));
    cudaMemcpy(d_s, s, sizeof(WindowOrthoState), cudaMemcpyHostToDevice);
    s->win_head = 0; // (kept on host mirror as well)

    // stash device-struct pointer in an unused slot? We instead return the host
    // mirror and re-copy each call. Store d_s in a side channel via current_dim? No.
    // Simpler: caller passes host mirror; we keep d_s in *out_state is the HOST mirror,
    // and we stash d_s separately by over-allocating: store d_s in win_head? Not portable.
    // => We store the device struct pointer inside the host struct's trailing ints is
    //    impossible. So we keep d_s in a file-static map keyed by host pointer.
    host_to_device[(uintptr_t)s] = d_s;
    *out_state = s;
    return cudaGetLastError();
}

// file-static map host-mirror -> device struct copy (declared above the wrappers)

extern "C" cudaError_t window_ortho_feed_cuda(
    WindowOrthoState* s, const float* k, const float* v
) {
    auto it = host_to_device.find((uintptr_t)s);
    if (it == host_to_device.end()) return cudaErrorInvalidDevicePointer;
    WindowOrthoState* d_s = it->second;

    size_t kv_bytes = (size_t)s->num_heads * s->head_dim * sizeof(float);
    float *d_k, *d_v;
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMemcpy(d_k, k, kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, kv_bytes, cudaMemcpyHostToDevice);

    // re-sync mirror scalars to device struct
    cudaMemcpy(d_s, s, sizeof(WindowOrthoState), cudaMemcpyHostToDevice);

    window_ortho_feed_kernel<<<1, 1>>>(d_s, d_k, d_v);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();

    // read back updated scalars (win_head/win_count/current_dim)
    WindowOrthoState tmp;
    cudaMemcpy(&tmp, d_s, sizeof(WindowOrthoState), cudaMemcpyDeviceToHost);
    s->win_head = tmp.win_head;
    s->win_count = tmp.win_count;
    s->current_dim = tmp.current_dim;

    cudaFree(d_k); cudaFree(d_v);
    return err;
}

extern "C" cudaError_t window_ortho_attention_cuda(
    WindowOrthoState* s, const float* Q, float* out
) {
    auto it = host_to_device.find((uintptr_t)s);
    if (it == host_to_device.end()) return cudaErrorInvalidDevicePointer;
    WindowOrthoState* d_s = it->second;

    size_t q_bytes = (size_t)s->num_heads * s->head_dim * sizeof(float);
    float *d_q, *d_out;
    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_out, q_bytes);
    cudaMemcpy(d_q, Q, q_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, q_bytes);
    cudaMemcpy(d_s, s, sizeof(WindowOrthoState), cudaMemcpyHostToDevice);

    int blocks = s->num_heads;
    window_ortho_attention_kernel<<<blocks, 1>>>(d_s, d_q, d_out);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, q_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_q); cudaFree(d_out);
    return err;
}

extern "C" cudaError_t window_ortho_get_state_cuda(
    WindowOrthoState* s, float* K, float* V, float* decay, int* current_dim
) {
    auto it = host_to_device.find((uintptr_t)s);
    if (it == host_to_device.end()) return cudaErrorInvalidDevicePointer;
    WindowOrthoState* d_s = it->second;

    size_t basis_bytes = (size_t)s->num_heads * s->ortho_base_dim * s->head_dim * sizeof(float);
    cudaMemcpy(K, d_s->K_basis, basis_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_s->V_basis, basis_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(decay, d_s->decay, (size_t)s->ortho_base_dim * sizeof(float), cudaMemcpyDeviceToHost);
    int cd; cudaMemcpy(&cd, &d_s->current_dim, sizeof(int), cudaMemcpyDeviceToHost);
    *current_dim = cd;
    return cudaGetLastError();
}

extern "C" cudaError_t window_ortho_set_state_cuda(
    WindowOrthoState* s, const float* K, const float* V, const float* decay, int current_dim
) {
    auto it = host_to_device.find((uintptr_t)s);
    if (it == host_to_device.end()) return cudaErrorInvalidDevicePointer;
    WindowOrthoState* d_s = it->second;

    size_t basis_bytes = (size_t)s->num_heads * s->ortho_base_dim * s->head_dim * sizeof(float);
    cudaMemcpy(d_s->K_basis, K, basis_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s->V_basis, V, basis_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s->decay, decay, (size_t)s->ortho_base_dim * sizeof(float), cudaMemcpyHostToDevice);
    s->current_dim = current_dim;
    cudaMemcpy(&d_s->current_dim, &current_dim, sizeof(int), cudaMemcpyHostToDevice);
    return cudaGetLastError();
}

extern "C" cudaError_t window_ortho_destroy_cuda(WindowOrthoState* s) {
    auto it = host_to_device.find((uintptr_t)s);
    if (it != host_to_device.end()) {
        WindowOrthoState* d_s = it->second;
        cudaFree(d_s->win_k); cudaFree(d_s->win_v);
        cudaFree(d_s->K_basis); cudaFree(d_s->V_basis); cudaFree(d_s->decay);
        cudaFree(d_s);
        host_to_device.erase(it);
    }
    free(s);
    return cudaGetLastError();
}
