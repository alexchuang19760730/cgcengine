#pragma once

void embedding_forward(float* output, int token_id);
void rms_norm_forward(float* x, float* weight, int size);
void qkv_matmul_forward(float* output, float* input, float* weight, int out_dim, int in_dim);
void rope_forward(float* q, float* k, float* rope_cos, float* rope_sin, int seq_len);
void kda_attention_forward(float* q, float* k, float* v, float* output, int seq_len);
void o_matmul_forward(float* output, float* input, float* weight, int out_dim, int in_dim);
void residual_forward(float* x, float* residual);
void swiglu_up_forward(float* output, float* input, int size);
void swiglu_gate_forward(float* output, float* up, float* gate, int size);
void swiglu_down_forward(float* output, float* input, float* weight, int out_dim, int in_dim);
void final_norm_forward(float* x, float* weight, int size);
void lm_head_forward(float* logits, float* hidden, float* weight, int vocab, int dim);
int sample_token(float* logits, int vocab);
