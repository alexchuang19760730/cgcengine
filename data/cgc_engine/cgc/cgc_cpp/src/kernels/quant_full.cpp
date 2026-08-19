#include "kernels/quant_full.h"
#include <math.h>
#include <string.h>

void quantize_q4(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    int64_t group_size
) {
    int64_t num_groups = size / group_size;
    for (int64_t g = 0; g < num_groups; g++) {
        const float* in = input + g * group_size;
        uint8_t* out = output + g * group_size / 2;
        
        float max_val = 0.0f;
        for (int64_t i = 0; i < group_size; i++) {
            float abs_val = fabsf(in[i]);
            if (abs_val > max_val) max_val = abs_val;
        }
        
        float scale = max_val / 7.0f;
        scales[g] = scale;
        
        for (int64_t i = 0; i < group_size; i += 2) {
            int q0 = (int)(in[i] / scale + 0.5f);
            int q1 = (int)(in[i+1] / scale + 0.5f);
            
            q0 = (q0 > 7) ? 7 : (q0 < -8) ? -8 : q0;
            q1 = (q1 > 7) ? 7 : (q1 < -8) ? -8 : q1;
            
            out[i/2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

void quantize_q8(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    int64_t group_size
) {
    int64_t num_groups = size / group_size;
    for (int64_t g = 0; g < num_groups; g++) {
        const float* in = input + g * group_size;
        uint8_t* out = output + g * group_size;
        
        float max_val = 0.0f;
        for (int64_t i = 0; i < group_size; i++) {
            float abs_val = fabsf(in[i]);
            if (abs_val > max_val) max_val = abs_val;
        }
        
        float scale = max_val / 127.0f;
        scales[g] = scale;
        
        for (int64_t i = 0; i < group_size; i++) {
            int q = (int)(in[i] / scale + 0.5f);
            q = (q > 127) ? 127 : (q < -128) ? -128 : q;
            out[i] = (uint8_t)(q + 128);
        }
    }
}

static const float nf4_table[] = {
    -1.0f, -0.6961917f, -0.5250738f, -0.3949166f,
    -0.2844418f, -0.1848865f, -0.0910463f, 0.0f,
    0.0795834f, 0.1609340f, 0.2461262f, 0.3379128f,
    0.4407066f, 0.5626183f, 0.7229567f, 1.0f
};

void quantize_nf4(
    const float* input, 
    uint8_t* output, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        float val = input[i];
        float min_dist = 1e10f;
        int idx = 0;
        
        for (int j = 0; j < 16; j++) {
            float dist = fabsf(val - nf4_table[j]);
            if (dist < min_dist) {
                min_dist = dist;
                idx = j;
            }
        }
        
        output[i] = (uint8_t)idx;
    }
}

void quantize_iq4(
    const float* input, 
    uint8_t* output, 
    int64_t size,
    float* scales,
    float* zeros,
    int64_t group_size
) {
    int64_t num_groups = size / group_size;
    for (int64_t g = 0; g < num_groups; g++) {
        const float* in = input + g * group_size;
        uint8_t* out = output + g * group_size / 2;
        
        float min_val = 1e10f, max_val = -1e10f;
        for (int64_t i = 0; i < group_size; i++) {
            if (in[i] < min_val) min_val = in[i];
            if (in[i] > max_val) max_val = in[i];
        }
        
        float range = max_val - min_val;
        float scale = range / 15.0f;
        float zero = -min_val / scale;
        
        scales[g] = scale;
        zeros[g] = zero;
        
        for (int64_t i = 0; i < group_size; i += 2) {
            int q0 = (int)((in[i] / scale) + zero + 0.5f);
            int q1 = (int)((in[i+1] / scale) + zero + 0.5f);
            
            q0 = (q0 > 15) ? 15 : (q0 < 0) ? 0 : q0;
            q1 = (q1 > 15) ? 15 : (q1 < 0) ? 0 : q1;
            
            out[i/2] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void dequantize_q4(
    const uint8_t* input, 
    float* output, 
    int64_t size,
    const float* scales,
    int64_t group_size
) {
    int64_t num_groups = size / group_size;
    for (int64_t g = 0; g < num_groups; g++) {
        const uint8_t* in = input + g * group_size / 2;
        float* out = output + g * group_size;
        float scale = scales[g];
        
        for (int64_t i = 0; i < group_size; i += 2) {
            uint8_t packed = in[i/2];
            int q0 = (int8_t)((packed & 0xF) << 4) >> 4;
            int q1 = (int8_t)((packed >> 4) << 4) >> 4;
            
            out[i] = (float)q0 * scale;
            out[i+1] = (float)q1 * scale;
        }
    }
}

void dequantize_q8(
    const uint8_t* input, 
    float* output, 
    int64_t size,
    const float* scales,
    int64_t group_size
) {
    int64_t num_groups = size / group_size;
    for (int64_t g = 0; g < num_groups; g++) {
        const uint8_t* in = input + g * group_size;
        float* out = output + g * group_size;
        float scale = scales[g];
        
        for (int64_t i = 0; i < group_size; i++) {
            int q = (int8_t)(in[i] - 128);
            out[i] = (float)q * scale;
        }
    }
}

void dequantize_nf4(
    const uint8_t* input, 
    float* output, 
    int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        int idx = input[i] & 0xF;
        output[i] = nf4_table[idx];
    }
}

void quantize_rowwise(
    const float* input, 
    uint8_t* output, 
    int64_t rows,
    int64_t cols,
    float* scales,
    float* zeros
) {
    for (int64_t r = 0; r < rows; r++) {
        const float* in = input + r * cols;
        uint8_t* out = output + r * cols;
        
        float min_val = 1e10f, max_val = -1e10f;
        for (int64_t c = 0; c < cols; c++) {
            if (in[c] < min_val) min_val = in[c];
            if (in[c] > max_val) max_val = in[c];
        }
        
        float range = max_val - min_val;
        float scale = range / 255.0f;
        float zero = -min_val / scale;
        
        scales[r] = scale;
        zeros[r] = zero;
        
        for (int64_t c = 0; c < cols; c++) {
            int q = (int)(in[c] / scale + zero + 0.5f);
            q = (q > 255) ? 255 : (q < 0) ? 0 : q;
            out[c] = (uint8_t)q;
        }
    }
}