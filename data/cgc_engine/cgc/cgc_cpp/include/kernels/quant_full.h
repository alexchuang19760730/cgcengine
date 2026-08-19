#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void quantize_q4(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    int64_t group_size
);

void quantize_q8(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    int64_t group_size
);

void quantize_nf4(
    const float* input, 
    uint8_t* output, 
    int64_t size
);

void quantize_iq4(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    float* zeros,
    int64_t group_size
);

void dequantize_q4(
    const uint8_t* input, 
    float* output, 
    int64_t size,
    const float* scales,
    int64_t group_size
);

void dequantize_q8(
    const uint8_t* input, 
    float* output, 
    int64_t size,
    const float* scales,
    int64_t group_size
);

void dequantize_nf4(
    const uint8_t* input, 
    float* output, 
    int64_t size
);

void quantize_rowwise(
    const float* input, 
    uint8_t* output, 
    int64_t rows,
    int64_t cols,
    float* scales,
    float* zeros
);

#ifdef __cplusplus
}
#endif