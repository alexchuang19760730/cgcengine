#include "kernels/activation_full.h"
#include <math.h>

void activation_swiglu(
    const float* x, 
    const float* gate, 
    float* out, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        float gate_val = gate[i];
        float sigmoid = 1.0f / (1.0f + expf(-gate_val));
        out[i] = x[i] * gate_val * sigmoid;
    }
}

void activation_relu(
    const float* x, 
    float* out, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

void activation_relu2(
    const float* x, 
    float* out, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        float val = x[i];
        out[i] = (val > 0.0f) ? (val * val) : 0.0f;
    }
}

void activation_tanh(
    const float* x, 
    float* out, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        out[i] = tanhf(x[i]);
    }
}