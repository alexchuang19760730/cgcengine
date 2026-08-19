#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void quant_w8a16(
    const float* x, float* out,
    int64_t m, int64_t n
);

void quant_gguf_q4(
    const float* x, float* out,
    int64_t m, int64_t n
);

#ifdef __cplusplus
}
#endif