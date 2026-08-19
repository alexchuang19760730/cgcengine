#include <metal_stdlib>
using namespace metal;

kernel void swiglu_up_kernel(
    device float* up       [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant int& size     [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    float x = up[gid];
    out[gid] = x / (1.0f + expf(-x));
}

kernel void swiglu_gate_kernel(
    device float* gate     [[buffer(0)]],
    device float* up      [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant int& size     [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    float g = gate[gid];
    float u = up[gid];
    float silu = g / (1.0f + expf(-g));
    out[gid] = u * silu;
}

kernel void swiglu_down_kernel(
    device float* down     [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant int& size     [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    out[gid] = down[gid];
}
