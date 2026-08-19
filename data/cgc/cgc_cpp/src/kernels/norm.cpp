#include "kernels/norm.h"
#include <stdio.h>
#include <cmath>

void rms_norm(
    const float* x, const float* weight, float* out,
    float eps, int64_t batch, int64_t seqlen, int64_t d
) {
    printf("[CGC C++] rms_norm (fallback) called: B=%ld, S=%ld, D=%ld\n", batch, seqlen, d);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seqlen; s++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < d; i++) {
                float val = x[(b * seqlen + s) * d + i];
                sum += val * val;
            }
            float rms = sqrtf(sum / d + eps);
            for (int64_t i = 0; i < d; i++) {
                float val = x[(b * seqlen + s) * d + i];
                out[(b * seqlen + s) * d + i] = val / rms * weight[i];
            }
        }
    }
}