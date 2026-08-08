#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

typedef struct MetalDevice metal_device_t;
typedef struct MetalBuffer metal_buffer_t;
typedef struct MetalKernel metal_kernel_t;
typedef struct MetalKDAState metal_kda_state_t;

typedef struct {
    int n_layer;
    int dim;
    int n_head;
    int n_kv_head;
    int head_dim;
    int vocab_size;
    int max_seq;
} ModelConfig;

typedef struct {
    metal_buffer_t* embedding;
    metal_buffer_t* layer_attn_norm;
    metal_buffer_t* layer_qkv;
    metal_buffer_t* layer_o;
    metal_buffer_t* layer_ffn_norm;
    metal_buffer_t* layer_ffn_up;
    metal_buffer_t* layer_ffn_gate;
    metal_buffer_t* layer_ffn_down;
    metal_buffer_t* final_norm;
    metal_buffer_t* lm_head;
    metal_buffer_t* rope_cos;
    metal_buffer_t* rope_sin;
} ModelWeights;

metal_device_t* metal_device_create(void);
void metal_device_destroy(metal_device_t* dev);
bool metal_is_available(void);
const char* metal_get_device_name(void);

metal_buffer_t* metal_buffer_create(metal_device_t* dev, size_t size);
metal_buffer_t* metal_buffer_create_from_data(metal_device_t* dev, const void* data, size_t size);
void metal_buffer_destroy(metal_buffer_t* buf);
float* metal_buffer_get_host_ptr(metal_buffer_t* buf);
void metal_buffer_copy_to_device(metal_buffer_t* dest, const void* src, size_t size);
void metal_buffer_copy_from_device(void* dest, metal_buffer_t* src, size_t size);

metal_kernel_t* metal_kernel_create(metal_device_t* dev, const char* source, const char* kernel_name);
void metal_kernel_destroy(metal_kernel_t* kernel);

void metal_kernel_set_buffer(metal_kernel_t* kernel, int index, metal_buffer_t* buf);
void metal_kernel_set_value(metal_kernel_t* kernel, int index, const void* value, size_t size);

void metal_kernel_execute(metal_kernel_t* kernel, int threads_x, int threads_y, int threads_z);

void metal_synchronize(metal_device_t* dev);

ModelWeights* metal_load_gguf_weights(metal_device_t* dev, const char* gguf_path, ModelConfig* config);
void metal_weights_destroy(ModelWeights* weights);

void metal_gemm(metal_device_t* dev,
                metal_buffer_t* a, metal_buffer_t* b, metal_buffer_t* c,
                int M, int N, int K, bool transpose_a, bool transpose_b);

void metal_rms_norm(metal_device_t* dev, metal_buffer_t* x, metal_buffer_t* weight,
                    metal_buffer_t* output, int size);

void metal_kda_attention(metal_device_t* dev,
                         metal_buffer_t* q, metal_buffer_t* k, metal_buffer_t* v,
                         metal_buffer_t* output, metal_buffer_t* state,
                         int n_head, int n_kv_head, int head_dim, int seq_len, float beta);

metal_kda_state_t* metal_kda_state_create(metal_device_t* dev, int n_head, int head_dim, int seq_len);
void metal_kda_state_destroy(metal_kda_state_t* state);
metal_buffer_t* metal_kda_state_get_buffer(metal_kda_state_t* state);

void metal_rope(metal_device_t* dev,
                metal_buffer_t* q, metal_buffer_t* k,
                metal_buffer_t* rope_cos, metal_buffer_t* rope_sin,
                int seq_len, int dim);

void metal_embedding(metal_device_t* dev,
                     metal_buffer_t* tokens, metal_buffer_t* embedding,
                     metal_buffer_t* output, int batch_size, int dim);

void metal_lm_head(metal_device_t* dev,
                   metal_buffer_t* hidden, metal_buffer_t* weight,
                   metal_buffer_t* logits, int dim, int vocab_size);

#ifdef __cplusplus
}
#endif