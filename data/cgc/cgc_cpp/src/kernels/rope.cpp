#include "kernels/rope.h"
#include <stdio.h>

void rope_apply(
    const float* x, const float* pos, float* out,
    int64_t batch, int64_t seqlen, int64_t d
) {
    printf("[CGC C++] rope_apply (fallback) called: B=%ld, S=%ld, D=%ld\n", batch, seqlen, d);

    int64_t total = batch * seqlen * d;
    for (int64_t i = 0; i < total; i++) {
        out[i] = x[i];
    }
}