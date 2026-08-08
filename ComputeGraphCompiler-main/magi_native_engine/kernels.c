#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void embedding_forward(float* output, int token_id) {
    printf("    [Kernel] embedding_forward: token=%d (CPU fallback)\n", token_id);
}

void rms_norm_forward(float* x, float* weight, int size) {
    double t0 = get_time_ms();

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    float rms = sqrtf(sum / size + 1e-6f);
    for (int i = 0; i < size; i++) {
        x[i] = x[i] * weight[i] / rms;
    }

    printf("    [RMS Norm] size=%d (%.2f ms)\n", size, get_time_ms() - t0);
}

void qkv_matmul_forward(float* output, float* input, float* weight, int out_dim, int in_dim) {
    double t0 = get_time_ms();

    for (int i = 0; i < out_dim; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            output[i] += input[j] * weight[i * in_dim + j];
        }
    }

    printf("    [QKV Matmul] %dx%d (%.2f ms)\n", out_dim, in_dim, get_time_ms() - t0);
}

void rope_forward(float* q, float* k, float* rope_cos, float* rope_sin, int seq_len) {
    double t0 = get_time_ms();
    int dim = 0;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < dim / 2; i++) {
            float q0 = q[pos * dim + i];
            float q1 = q[pos * dim + i + dim / 2];
            float cos_val = rope_cos[pos * dim + i];
            float sin_val = rope_sin[pos * dim + i];
            q[pos * dim + i] = q0 * cos_val - q1 * sin_val;
            q[pos * dim + i + dim / 2] = q0 * sin_val + q1 * cos_val;

            float k0 = k[pos * dim + i];
            float k1 = k[pos * dim + i + dim / 2];
            k[pos * dim + i] = k0 * cos_val - k1 * sin_val;
            k[pos * dim + i + dim / 2] = k0 * sin_val + k1 * cos_val;
        }
    }

    printf("    [RoPE] seq=%d (%.2f ms)\n", seq_len, get_time_ms() - t0);
}

void kda_attention_forward(float* q, float* k, float* v, float* output, int seq_len) {
    printf("    [KERNEL] ★ KDA Attention (O(N) 线性注意力!)\n");
    double t0 = get_time_ms();

    float scale = 1.0f / sqrtf(64.0f);
    float state = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < 64; i++) {
            state += k[t * 64 + i] * v[t * 64 + i];
        }
    }

    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < 64; i++) {
            output[t * 64 + i] = q[t * 64 + i] * state * scale;
        }
    }

    printf("      KDA 完成: %.2f ms (seq_len=%d)\n", get_time_ms() - t0, seq_len);
}

void o_matmul_forward(float* output, float* input, float* weight, int out_dim, int in_dim) {
    double t0 = get_time_ms();

    for (int i = 0; i < out_dim; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            output[i] += input[j] * weight[i * in_dim + j];
        }
    }

    printf("    [O Matmul] %dx%d (%.2f ms)\n", out_dim, in_dim, get_time_ms() - t0);
}

void residual_forward(float* x, float* residual) {
    for (int i = 0; i < 896; i++) {
        x[i] = x[i] + residual[i];
    }
}

void swiglu_up_forward(float* output, float* input, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void swiglu_gate_forward(float* output, float* up, float* gate, int size) {
    for (int i = 0; i < size; i++) {
        float silu = gate[i] / (1.0f + expf(-gate[i]));
        output[i] = up[i] * silu;
    }
}

void swiglu_down_forward(float* output, float* input, float* weight, int out_dim, int in_dim) {
    double t0 = get_time_ms();

    for (int i = 0; i < out_dim; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            output[i] += input[j] * weight[i * in_dim + j];
        }
    }

    printf("    [SwiGLU Down] %dx%d (%.2f ms)\n", out_dim, in_dim, get_time_ms() - t0);
}

void final_norm_forward(float* x, float* weight, int size) {
    double t0 = get_time_ms();

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    float rms = sqrtf(sum / size + 1e-6f);
    for (int i = 0; i < size; i++) {
        x[i] = x[i] * weight[i] / rms;
    }

    printf("    [Final Norm] size=%d (%.2f ms)\n", size, get_time_ms() - t0);
}

void lm_head_forward(float* logits, float* hidden, float* weight, int vocab, int dim) {
    double t0 = get_time_ms();

    for (int i = 0; i < vocab; i++) {
        logits[i] = 0.0f;
        for (int j = 0; j < dim; j++) {
            logits[i] += hidden[j] * weight[i * dim + j];
        }
    }

    printf("    [LM Head] vocab=%d dim=%d (%.2f ms)\n", vocab, dim, get_time_ms() - t0);
}

int sample_token(float* logits, int vocab) {
    float max_logit = logits[0];
    int max_idx = 0;
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}