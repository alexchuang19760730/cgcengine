#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_kernel(
    device float* x        [[buffer(0)]],
    device float* weight   [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant int& dim     [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[gid * dim + i];
        sum += val * val;
    }
    float rms = sqrt(sum / (float)dim + 1e-6f);

    for (int i = 0; i < dim; i++) {
        out[gid * dim + i] = x[gid * dim + i] * weight[i] / rms;
    }
}

kernel void final_norm_kernel(
    device float* x        [[buffer(0)]],
    device float* weight   [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant int& dim     [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[i];
        sum += val * val;
    }
    float rms = sqrt(sum / (float)dim + 1e-6f);

    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * weight[i] / rms;
    }
}
