#include "kernels/attention.h"
#include <stdio.h>
#include <cmath>

void attention_sdpa(
    const float* q, const float* k, const float* v, float* out,
    int64_t batch, int64_t heads, int64_t seqlen, int64_t d
) {
    printf("[CGC C++] attention_sdpa (fallback) called: B=%ld, H=%ld, S=%ld, D=%ld\n", batch, heads, seqlen, d);

    int64_t total = batch * heads * seqlen * d;
    for (int64_t i = 0; i < total; i++) {
        out[i] = q[i];
    }
}