#include "kernels/rope_full.h"
#include <math.h>

void rope_hf(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* freq_base
) {
    int64_t half_dim = dim / 2;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < head; h++) {
            for (int64_t s = 0; s < seqlen; s++) {
                int64_t pos = s + offset;
                int64_t base_idx = b * head * seqlen * dim + h * seqlen * dim + s * dim;
                
                for (int64_t i = 0; i < half_dim; i++) {
                    float freq = freq_base[i];
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    
                    float x0 = x[base_idx + i];
                    float x1 = x[base_idx + i + half_dim];
                    
                    x[base_idx + i] = x0 * cos_val - x1 * sin_val;
                    x[base_idx + i + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void rope_yarn(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    float rope_base,
    float rope_scale,
    const float* freq_base
) {
    int64_t half_dim = dim / 2;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < head; h++) {
            for (int64_t s = 0; s < seqlen; s++) {
                int64_t pos = s + offset;
                int64_t base_idx = b * head * seqlen * dim + h * seqlen * dim + s * dim;
                
                float yarn_pos = (float)pos / rope_scale;
                
                for (int64_t i = 0; i < half_dim; i++) {
                    float freq = powf(rope_base, (float)(2 * i) / (float)dim);
                    float cos_val = cosf(yarn_pos / freq);
                    float sin_val = sinf(yarn_pos / freq);
                    
                    float x0 = x[base_idx + i];
                    float x1 = x[base_idx + i + half_dim];
                    
                    x[base_idx + i] = x0 * cos_val - x1 * sin_val;
                    x[base_idx + i + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void rope_long(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    float long_factor,
    const float* freq_base
) {
    int64_t half_dim = dim / 2;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < head; h++) {
            for (int64_t s = 0; s < seqlen; s++) {
                int64_t pos = s + offset;
                int64_t base_idx = b * head * seqlen * dim + h * seqlen * dim + s * dim;
                
                float long_pos = (float)pos / long_factor;
                
                for (int64_t i = 0; i < half_dim; i++) {
                    float freq = freq_base[i];
                    float cos_val = cosf(long_pos * freq);
                    float sin_val = sinf(long_pos * freq);
                    
                    float x0 = x[base_idx + i];
                    float x1 = x[base_idx + i + half_dim];
                    
                    x[base_idx + i] = x0 * cos_val - x1 * sin_val;
                    x[base_idx + i + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void rope_gptj(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* freq_base
) {
    int64_t half_dim = dim / 2;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < head; h++) {
            for (int64_t s = 0; s < seqlen; s++) {
                int64_t pos = s + offset;
                int64_t base_idx = b * head * seqlen * dim + h * seqlen * dim + s * dim;
                
                for (int64_t i = 0; i < half_dim; i++) {
                    float freq = freq_base[i];
                    float cos_val = cosf(pos * freq);
                    float sin_val = sinf(pos * freq);
                    
                    float x0 = x[base_idx + 2 * i];
                    float x1 = x[base_idx + 2 * i + 1];
                    
                    x[base_idx + 2 * i] = x0 * cos_val - x1 * sin_val;
                    x[base_idx + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void rope_fast(
    float* x, 
    int64_t batch,
    int64_t head,
    int64_t seqlen,
    int64_t dim,
    int64_t offset,
    const float* cos_cache,
    const float* sin_cache
) {
    int64_t half_dim = dim / 2;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < head; h++) {
            for (int64_t s = 0; s < seqlen; s++) {
                int64_t pos = s + offset;
                int64_t base_idx = b * head * seqlen * dim + h * seqlen * dim + s * dim;
                
                for (int64_t i = 0; i < half_dim; i++) {
                    float cos_val = cos_cache[pos * half_dim + i];
                    float sin_val = sin_cache[pos * half_dim + i];
                    
                    float x0 = x[base_idx + i];
                    float x1 = x[base_idx + i + half_dim];
                    
                    x[base_idx + i] = x0 * cos_val - x1 * sin_val;
                    x[base_idx + i + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}