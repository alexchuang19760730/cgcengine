#include <metal_stdlib>
using namespace metal;

kernel void rope_kernel(
    device float* q        [[buffer(0)]],
    device float* k        [[buffer(1)]],
    constant int& dim      [[buffer(2)]],
    constant int& pos      [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    const int head_dim = dim / 2;
    const int i = gid % head_dim;
    const float theta = powf(10000.0f, -2.0f * (float)i / (float)head_dim);
    const float freq = pos * theta;
    const float cos_freq = cosf(freq);
    const float sin_freq = sinf(freq);

    const int base = gid - (gid % head_dim);
    const float q0 = q[base + i];
    const float q1 = q[base + i + head_dim];
    q[base + i] = q0 * cos_freq - q1 * sin_freq;
    q[base + i + head_dim] = q0 * sin_freq + q1 * cos_freq;

    if (k != q) {
        const float k0 = k[base + i];
        const float k1 = k[base + i + head_dim];
        k[base + i] = k0 * cos_freq - k1 * sin_freq;
        k[base + i + head_dim] = k0 * sin_freq + k1 * cos_freq;
    }
}
