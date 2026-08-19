#include "magi_compiler_integration.h"

// 显式实例化
template class FixedKVCache;
template class MagiCompiler;

// ==============================
// 算子实现
// ==============================
__global__ void token_to_ortho_coeff(
    const float* key_token,
    const float* ortho_basis,
    float* coeff,
    int head_dim
) {
    int i = threadIdx.x;
    if (i >= ORTHO_BASE_DIM) return;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += key_token[d] * ortho_basis[i * head_dim + d];
    }
    coeff[i] = sum;
}

__global__ void update_fixed_kv(
    float* fixed_K,
    float* fixed_V,
    const float* coeff,
    const float* value_token,
    int head_dim
) {
    int i = threadIdx.x;
    if (i >= ORTHO_BASE_DIM) return;

    float c = coeff[i];
    for (int d = 0; d < head_dim; d++) {
        fixed_K[i * head_dim + d] += c * ortho_basis[i * head_dim + d];
        fixed_V[i * head_dim + d] += c * value_token[d];
    }
}

__global__ void fixed_kv_attention(
    const float* query,
    const float* fixed_K,
    const float* fixed_V,
    float* output,
    int head_dim
) {
    int i = threadIdx.x;
    if (i >= ORTHO_BASE_DIM) return;

    float attn = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        attn += query[d] * fixed_K[i * head_dim + d];
    }

    for (int d = 0; d < head_dim; d++) {
        output[d] += attn * fixed_V[i * head_dim + d];
    }
}

__global__ void kda_v4_fixed_kv_forward(
    const float* Q,
    const float* fixed_K,
    const float* fixed_V,
    const float* time_decay,
    float* out,
    int head_dim
) {
    int i = threadIdx.x;
    if (i >= ORTHO_BASE_DIM) return;

    float decay = time_decay[i];
    
    float attn = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        attn += Q[d] * fixed_K[i * head_dim + d];
    }
    
    attn *= decay;

    for (int d = 0; d < head_dim; d++) {
        out[d] += attn * fixed_V[i * head_dim + d];
    }
}