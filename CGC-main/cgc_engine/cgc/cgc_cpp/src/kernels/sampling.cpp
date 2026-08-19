#include "kernels/sampling.h"
#include <math.h>
#include <stdlib.h>

void sample_greedy(
    const float* logits, 
    int64_t* output, 
    int64_t batch_size,
    int64_t vocab_size
) {
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        float max_val = -INFINITY;
        int64_t max_idx = 0;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            if (row[i] > max_val) {
                max_val = row[i];
                max_idx = i;
            }
        }
        
        output[b] = max_idx;
    }
}

void softmax_sample(
    const float* logits, 
    float* output, 
    int64_t batch_size,
    int64_t vocab_size
) {
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        float* out_row = output + b * vocab_size;
        
        float max_val = -INFINITY;
        for (int64_t i = 0; i < vocab_size; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        
        float sum = 0.0f;
        for (int64_t i = 0; i < vocab_size; i++) {
            out_row[i] = expf(row[i] - max_val);
            sum += out_row[i];
        }
        
        for (int64_t i = 0; i < vocab_size; i++) {
            out_row[i] /= sum;
        }
    }
}

void sample_multinomial(
    const float* logits, 
    int64_t* output, 
    int64_t batch_size,
    int64_t vocab_size
) {
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        
        float max_val = -INFINITY;
        for (int64_t i = 0; i < vocab_size; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        
        float sum = 0.0f;
        for (int64_t i = 0; i < vocab_size; i++) {
            probs[i] = expf(row[i] - max_val);
            sum += probs[i];
        }
        
        for (int64_t i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
        
        float r = (float)rand() / RAND_MAX;
        float cdf = 0.0f;
        int64_t idx = 0;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            cdf += probs[i];
            if (r <= cdf) {
                idx = i;
                break;
            }
        }
        
        output[b] = idx;
    }
    
    free(probs);
}

void sample_temperature(
    const float* logits, 
    float* output, 
    int64_t batch_size,
    int64_t vocab_size,
    float temperature
) {
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        float* out_row = output + b * vocab_size;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            out_row[i] = row[i] / temperature;
        }
    }
}

void sample_topk(
    const float* logits, 
    float* output_logits, 
    int64_t* indices,
    int64_t batch_size,
    int64_t vocab_size,
    int64_t k
) {
    int64_t* temp_indices = (int64_t*)malloc(vocab_size * sizeof(int64_t));
    
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        float* out_row = output_logits + b * k;
        int64_t* idx_row = indices + b * k;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            temp_indices[i] = i;
        }
        
        for (int64_t i = 0; i < k; i++) {
            int64_t max_idx = i;
            float max_val = row[temp_indices[i]];
            
            for (int64_t j = i + 1; j < vocab_size; j++) {
                if (row[temp_indices[j]] > max_val) {
                    max_val = row[temp_indices[j]];
                    max_idx = j;
                }
            }
            
            int64_t temp = temp_indices[i];
            temp_indices[i] = temp_indices[max_idx];
            temp_indices[max_idx] = temp;
            
            idx_row[i] = temp_indices[i];
            out_row[i] = row[temp_indices[i]];
        }
    }
    
    free(temp_indices);
}

void sample_topp(
    const float* logits, 
    float* output_logits, 
    int64_t batch_size,
    int64_t vocab_size,
    float p
) {
    int64_t* indices = (int64_t*)malloc(vocab_size * sizeof(int64_t));
    float* sorted_logits = (float*)malloc(vocab_size * sizeof(float));
    
    for (int64_t b = 0; b < batch_size; b++) {
        const float* row = logits + b * vocab_size;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            indices[i] = i;
            sorted_logits[i] = row[i];
        }
        
        for (int64_t i = 0; i < vocab_size - 1; i++) {
            for (int64_t j = i + 1; j < vocab_size; j++) {
                if (sorted_logits[j] > sorted_logits[i]) {
                    float temp_logit = sorted_logits[i];
                    sorted_logits[i] = sorted_logits[j];
                    sorted_logits[j] = temp_logit;
                    
                    int64_t temp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp_idx;
                }
            }
        }
        
        float max_val = sorted_logits[0];
        float* exp_vals = (float*)malloc(vocab_size * sizeof(float));
        float sum = 0.0f;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            exp_vals[i] = expf(sorted_logits[i] - max_val);
            sum += exp_vals[i];
        }
        
        float cdf = 0.0f;
        int64_t cutoff = vocab_size;
        
        for (int64_t i = 0; i < vocab_size; i++) {
            cdf += exp_vals[i] / sum;
            if (cdf >= p) {
                cutoff = i + 1;
                break;
            }
        }
        
        float* out_row = output_logits + b * vocab_size;
        for (int64_t i = 0; i < vocab_size; i++) {
            out_row[i] = -INFINITY;
        }
        
        for (int64_t i = 0; i < cutoff; i++) {
            out_row[indices[i]] = sorted_logits[i];
        }
        
        free(exp_vals);
    }
    
    free(indices);
    free(sorted_logits);
}