#include "kernels/norm_full.h"
#include <math.h>

void layer_norm(
    const float* x, 
    float* out, 
    int64_t batch,
    int64_t seqlen,
    int64_t dim,
    const float* weight,
    const float* bias,
    float eps
) {
    int64_t total = batch * seqlen;
    
    for (int64_t i = 0; i < total; i++) {
        const float* in = x + i * dim;
        float* out_row = out + i * dim;
        
        float mean = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            mean += in[j];
        }
        mean /= (float)dim;
        
        float var = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            float diff = in[j] - mean;
            var += diff * diff;
        }
        var /= (float)dim;
        
        float inv_std = 1.0f / sqrtf(var + eps);
        
        for (int64_t j = 0; j < dim; j++) {
            float normalized = (in[j] - mean) * inv_std;
            if (weight != nullptr && bias != nullptr) {
                out_row[j] = normalized * weight[j] + bias[j];
            } else {
                out_row[j] = normalized;
            }
        }
    }
}

void group_norm(
    const float* x, 
    float* out, 
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t num_groups,
    const float* weight,
    const float* bias,
    float eps
) {
    int64_t group_size = channels / num_groups;
    int64_t spatial_size = height * width;
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t group_offset = b * channels * spatial_size + g * group_size * spatial_size;
            
            float mean = 0.0f;
            int64_t group_elem_count = group_size * spatial_size;
            
            for (int64_t i = 0; i < group_elem_count; i++) {
                mean += x[group_offset + i];
            }
            mean /= (float)group_elem_count;
            
            float var = 0.0f;
            for (int64_t i = 0; i < group_elem_count; i++) {
                float diff = x[group_offset + i] - mean;
                var += diff * diff;
            }
            var /= (float)group_elem_count;
            
            float inv_std = 1.0f / sqrtf(var + eps);
            
            for (int64_t c = 0; c < group_size; c++) {
                float w = (weight != nullptr) ? weight[g * group_size + c] : 1.0f;
                float b_val = (bias != nullptr) ? bias[g * group_size + c] : 0.0f;
                
                for (int64_t s = 0; s < spatial_size; s++) {
                    int64_t idx = group_offset + c * spatial_size + s;
                    float normalized = (x[idx] - mean) * inv_std;
                    out[idx] = normalized * w + b_val;
                }
            }
        }
    }
}