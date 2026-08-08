#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void activation_swiglu(
    const float* x, 
    const float* gate, 
    float* out, 
    int64_t size
);

void activation_relu(
    const float* x, 
    float* out, 
    int64_t size
);

void activation_relu2(
    const float* x, 
    float* out, 
    int64_t size
);

void activation_tanh(
    const float* x, 
    float* out, 
    int64_t size
);

void activation_sigmoid(
    const float* x, 
    float* out, 
    int64_t size
);

#ifdef __cplusplus
}
#endif