#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void sample_greedy(
    const float* logits, 
    int64_t* output, 
    int64_t batch_size,
    int64_t vocab_size
);

void sample_multinomial(
    const float* logits, 
    int64_t* output, 
    int64_t batch_size,
    int64_t vocab_size
);

void sample_temperature(
    const float* logits, 
    float* output, 
    int64_t batch_size,
    int64_t vocab_size,
    float temperature
);

void sample_topk(
    const float* logits, 
    float* output_logits, 
    int64_t* indices,
    int64_t batch_size,
    int64_t vocab_size,
    int64_t k
);

void sample_topp(
    const float* logits, 
    float* output_logits, 
    int64_t batch_size,
    int64_t vocab_size,
    float p
);

void softmax_sample(
    const float* logits, 
    float* output, 
    int64_t batch_size,
    int64_t vocab_size
);

#ifdef __cplusplus
}
#endif