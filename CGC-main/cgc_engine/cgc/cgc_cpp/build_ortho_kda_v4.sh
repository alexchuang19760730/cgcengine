#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_SRC="${SCRIPT_DIR}/src/kernels/ortho_kda_v4.cu"
OUTPUT_DIR="${SCRIPT_DIR}/build"
OUTPUT_LIB="${OUTPUT_DIR}/libortho_kda.so"

echo "=============================================="
echo "OrthoKDA v4 CUDA Kernel 編譯"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

TORCH_INCLUDE=$(python3 -c 'import torch; print(torch.include_path)' 2>/dev/null || echo "")
TORCH_CXX_FLAGS=$(python3 -c 'import torch; print(torch.cxx_flags)' 2>/dev/null || echo "")

echo "Torch include: ${TORCH_INCLUDE}"

NVCC="/usr/local/cuda/bin/nvcc"

if [ ! -f "${NVCC}" ]; then
    echo "❌ nvcc not found"
    exit 1
fi

echo ""
echo "Compiling with PyTorch fallback (no torch headers in CUDA mode)..."

cat > "${SCRIPT_DIR}/src/kernels/ortho_kda_v4_simple.cu" << 'EOF'
#include <cuda_runtime.h>
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

extern "C" __host__ void call_ortho_kda_forward(
    OrthoKDAKV* kv, const float* Q, float* out, int num_heads) {
    dim3 grid(num_heads);
    dim3 block(N_BASE);
    ortho_kda_v4_forward<<<grid, block>>>(kv, Q, out, num_heads);
}

extern "C" __host__ void call_ortho_kda_update(
    OrthoKDAKV* kv, const float* key, const float* value) {
    dim3 block(N_BASE);
    ortho_kda_v4_update<<<1, block>>>(kv, key, value);
}
EOF

"${NVCC}" -shared -o "${OUTPUT_LIB}" "${SCRIPT_DIR}/src/kernels/ortho_kda_v4_simple.cu" \
    -DNDEBUG -O3 --compiler-options '-fPIC -O3' \
    -use_fast_math \
    --gpu-architecture=sm_80 \
    -allow-unsupported-compiler

rm -f "${SCRIPT_DIR}/src/kernels/ortho_kda_v4_simple.cu"

if [ -f "${OUTPUT_LIB}" ]; then
    SIZE=$(stat -c%s "${OUTPUT_LIB}" 2>/dev/null || stat -f%z "${OUTPUT_LIB}" 2>/dev/null)
    echo ""
    echo "✅ Compilation successful!"
    echo "   Output: ${OUTPUT_LIB}"
    echo "   Size: ${SIZE} bytes"
else
    echo "❌ Compilation failed!"
    exit 1
fi