#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device float* a        [[buffer(0)]],
    device float* b        [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant int& M        [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& K        [[buffer(5)]],
    uint gid              [[thread_position_in_grid]]
) {
    const int row = gid / N;
    const int col = gid % N;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    out[row * N + col] = sum;
}

kernel void embedding_kernel(
    device int* tokens     [[buffer(0)]],
    device float* embed   [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant int& dim     [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    int token = tokens[gid];
    for (int d = 0; d < dim; d++) {
        out[d] = embed[token * dim + d];
    }
}

kernel void lm_head_kernel(
    device float* hidden   [[buffer(0)]],
    device float* weight   [[buffer(1)]],
    device float* logits   [[buffer(2)]],
    constant int& dim     [[buffer(3)]],
    constant int& vocab    [[buffer(4)]],
    uint gid              [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum += hidden[d] * weight[gid * dim + d];
    }
    logits[gid] = sum;
}

kernel void argmax_kernel(
    device float* logits   [[buffer(0)]],
    device int* out        [[buffer(1)]],
    constant int& size     [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    float max_val = logits[0];
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    out[0] = max_idx;
}
