// moe_router.metal — P2 预留 Metal kernel（batch_router + expert_gemv）
// 桩阶段默认走 moe_inject.c 的 CPU 朴素实现；此文件是替换路径的编译目标，
// 接入时：metal_kernel_create(dev, source, "batch_router_kernel") 即可替换 moe_batch_router。
#include <metal_stdlib>
using namespace metal;

// ② batch_router：hidden[N,2048] × router_w[2048,256] → scores[N,256]，取 top-k
// 每层 router 权重常驻（2048×256×2B BF16 → 1MB/层）
kernel void batch_router_kernel(
    device const float* hidden   [[buffer(0)]],   // N×2048
    device const float* router_w [[buffer(1)]],   // 2048×256
    device float* scores          [[buffer(2)]],  // N×256
    constant int& dim             [[buffer(3)]],  // 2048
    constant int& n_experts       [[buffer(4)]],  // 256
    uint gid                      [[thread_position_in_grid]]
) {
    const int tok = gid / n_experts;
    const int e   = gid % n_experts;
    if (tok >= 64) return;
    float s = 0.0f;
    for (int d = 0; d < dim; d++) s += hidden[tok * dim + d] * router_w[d * n_experts + e];
    scores[tok * n_experts + e] = s;
}

// ③ expert_gemv：hidden[N,intermediate] × expert_gate[intermediate,2048] → out[N,2048]
// 每个专家一个 dispatch；集合裁剪（U_c/P_commit）在 host 侧决定调用哪些专家。
kernel void expert_gemv_kernel(
    device const float* hidden  [[buffer(0)]],   // N×inter
    device const float* w       [[buffer(1)]],   // inter×2048（单专家 gate/up/down 权重）
    device float* out           [[buffer(2)]],   // N×2048
    constant int& inter         [[buffer(3)]],
    constant int& dim           [[buffer(4)]],
    constant int& N             [[buffer(5)]],
    uint gid                    [[thread_position_in_grid]]
) {
    const int tok = gid / dim;
    const int d   = gid % dim;
    if (tok >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < inter; k++) acc += hidden[tok * inter + k] * w[k * dim + d];
    out[tok * dim + d] = acc;
}
