#include "kernels/activation.h"
#include <stdio.h>
#include <cmath>

void activation_silu(
    const float* x, float* out, int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void activation_gelu(
    const float* x, float* out, int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (xi + 0.044715f * xi * xi * xi)));
        out[i] = xi * cdf;
    }
}

void activation_sigmoid(
    const float* x, float* out, int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void softmax(
    const float* x, float* out, int64_t batch, int64_t seqlen, int64_t dim
) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seqlen; s++) {
            const float* x_row = x + (b * seqlen + s) * dim;
            float* out_row = out + (b * seqlen + s) * dim;

            float max_val = -INFINITY;
            for (int64_t d = 0; d < dim; d++) {
                if (x_row[d] > max_val) max_val = x_row[d];
            }

            float sum = 0.0f;
            for (int64_t d = 0; d < dim; d++) {
                sum += expf(x_row[d] - max_val);
            }

            for (int64_t d = 0; d < dim; d++) {
                out_row[d] = expf(x_row[d] - max_val) / sum;
            }
        }
    }
}