#include "cgc_cpp.h"
#include "cgc_platform.h"
#include "cgc_backend.h"
#include "kernels/linear.h"
#include "kernels/attention.h"
#include "kernels/quant.h"
#include "kernels/quant_full.h"
#include "kernels/rope.h"
#include "kernels/rope_full.h"
#include "kernels/norm.h"
#include "kernels/norm_full.h"
#include "kernels/ortho_kda_v4.cuh"
#include "kernels/activation.h"
#include "kernels/activation_full.h"
#include "kernels/kv_cache.h"
#include "kernels/sampling.h"
#include <stdio.h>
#include <stdbool.h>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <cmath>

static const int SUPPORTED_OPCODES[] = {
    // Attention
    0x10, // ATTENTION_SDPA
    0x11, // ATTENTION_KDA
    0x12, // ATTENTION_PAGED
    0x13, // ATTENTION_FLASH
    
    // Linear/GEMM
    0x20, // LINEAR_GEMM
    0x21, // LINEAR_BIAS
    0x22, // GEMM_BATCHED
    
    // Norm
    0x30, // LAYER_NORM
    0x31, // RMS_NORM
    0x32, // GROUP_NORM
    
    // RoPE
    0x40, // ROPE
    0x41, // ROPE_FUSED
    0x42, // YARN_ROPE
    
    // Activation
    0x50, // SILU
    0x51, // GELU
    0x52, // GELU_TANH
    0x53, // RELU
    0x54, // SIGMOID
    
    // Sampling
    0x60, // SOFTMAX
    0x61, // LOG_SOFTMAX
    0x62, // TOP_K
    0x63, // TOP_P
    0x64, // TEMPERATURE
    
    // Memory/KV Cache
    0x70, // KV_CACHE_LOAD
    0x71, // KV_CACHE_STORE
    0x72, // KV_CACHE_UPDATE
    0x73, // EMBEDDING_LOOKUP
    0x74, // KV_CACHE_STATIC_LAYOUT
    0x75, // KV_CACHE_COMMIT
    
    // Quantization
    0xA0, // QUANTIZE_W8A16
    0xA1, // QUANTIZE_W4A16
    0xA2, // DEQUANTIZE
    0xA3, // GPTQ_KERNEL
    0xA4, // AWQ_KERNEL
    
    // LLAMA.CPP
    0xC3, // LLAMA_Q4_K_MATMUL
};
static const int NUM_SUPPORTED_OPCODES = 39;

static std::unordered_map<int64_t, void*> g_kda_states;
static int64_t g_next_kda_state_id = 0;
static CGCBackend* g_active_backend = nullptr;
static CGCPlatform g_detected_platform = CGC_PLATFORM_UNKNOWN;
static cgc_strategy_t g_current_strategy;
static bool g_strategy_initialized = false;
static cgc_backend_t g_current_backend_override = CGC_BACKEND_AUTO;
static bool g_kda_replace_mode = false;

static void cgc_init_default_strategy(void) {
    memset(&g_current_strategy, 0, sizeof(cgc_strategy_t));
    g_current_strategy.backend = CGC_BACKEND_AUTO;
    g_current_strategy.tile_config.tile_m = 128;
    g_current_strategy.tile_config.tile_n = 128;
    g_current_strategy.tile_config.tile_k = 128;
    g_current_strategy.tile_config.attn_block = 128;
    g_current_strategy.tile_config.moe_block = 128;
    g_current_strategy.enable_op_fusion = true;
    g_current_strategy.quantization_mode = 0;
    g_current_strategy.tp_degree = 1;
    g_current_strategy.pp_degree = 1;
    g_current_strategy.num_op_hints = 0;
    g_strategy_initialized = true;
}

const char* cgc_get_backend_name(cgc_backend_t backend) {
    switch (backend) {
        case CGC_BACKEND_CPU: return "cpu";
        case CGC_BACKEND_CUDA: return "cuda";
        case CGC_BACKEND_METAL: return "metal";
        case CGC_BACKEND_AUTO:
        default: return "auto";
    }
}

bool cgc_set_backend(cgc_backend_t backend) {
    if (backend == CGC_BACKEND_AUTO) {
        g_current_backend_override = CGC_BACKEND_AUTO;
        return true;
    }

    if (!g_strategy_initialized) {
        cgc_init_default_strategy();
    }

    g_current_backend_override = backend;

    switch (backend) {
#ifdef CGC_CUDA_ENABLED
        case CGC_BACKEND_CUDA:
            if (g_active_backend != &cgc_cuda_backend) {
                if (g_active_backend && g_active_backend->destroy) {
                    g_active_backend->destroy();
                }
                g_active_backend = &cgc_cuda_backend;
                if (g_active_backend->init) {
                    g_active_backend->init();
                }
            }
            break;
#endif
#ifdef CGC_METAL_ENABLED
        case CGC_BACKEND_METAL:
            if (g_active_backend != &cgc_metal_backend) {
                if (g_active_backend && g_active_backend->destroy) {
                    g_active_backend->destroy();
                }
                g_active_backend = &cgc_metal_backend;
                if (g_active_backend->init) {
                    g_active_backend->init();
                }
            }
            break;
#endif
        case CGC_BACKEND_CPU:
            if (g_active_backend != &cgc_cpu_backend) {
                if (g_active_backend && g_active_backend->destroy) {
                    g_active_backend->destroy();
                }
                g_active_backend = &cgc_cpu_backend;
                if (g_active_backend->init) {
                    g_active_backend->init();
                }
            }
            break;
        default:
            return false;
    }
    return true;
}

cgc_backend_t cgc_get_current_backend(void) {
    if (g_current_backend_override != CGC_BACKEND_AUTO) {
        return g_current_backend_override;
    }
    switch (g_detected_platform) {
        case CGC_PLATFORM_CUDA: return CGC_BACKEND_CUDA;
        case CGC_PLATFORM_METAL: return CGC_BACKEND_METAL;
        default: return CGC_BACKEND_CPU;
    }
}

cgc_error_t cgc_inject_strategy(const cgc_strategy_t* strategy) {
    if (!strategy) {
        return CGC_ERROR_INVALID_STRATEGY;
    }

    if (!g_strategy_initialized) {
        cgc_init_default_strategy();
    }

    memcpy(&g_current_strategy, strategy, sizeof(cgc_strategy_t));

    printf("[CGC] Strategy injected: backend=%s, fusion=%d, tp=%d, hints=%d\n",
           cgc_get_backend_name(strategy->backend),
           strategy->enable_op_fusion,
           strategy->tp_degree,
           strategy->num_op_hints);

    if (strategy->backend != CGC_BACKEND_AUTO) {
        cgc_set_backend(strategy->backend);
    }

    return CGC_OK;
}

cgc_error_t cgc_get_strategy(cgc_strategy_t* strategy) {
    if (!strategy) {
        return CGC_ERROR_INVALID_STRATEGY;
    }
    if (!g_strategy_initialized) {
        cgc_init_default_strategy();
    }
    memcpy(strategy, &g_current_strategy, sizeof(cgc_strategy_t));
    return CGC_OK;
}

cgc_error_t cgc_reset_strategy(void) {
    if (!g_strategy_initialized) {
        cgc_init_default_strategy();
    }
    cgc_init_default_strategy();
    printf("[CGC] Strategy reset to defaults\n");
    return CGC_OK;
}

cgc_error_t cgc_set_kda_replace_mode(bool enable) {
    g_kda_replace_mode = enable;
    printf("[CGC] KDA Replace Mode: %s\n", enable ? "enabled" : "disabled");
    return CGC_OK;
}

bool cgc_get_kda_replace_mode(void) {
    return g_kda_replace_mode;
}

bool cgc_has_opcode(int opcode) {
    for (int i = 0; i < NUM_SUPPORTED_OPCODES; i++) {
        if (SUPPORTED_OPCODES[i] == opcode) {
            return true;
        }
    }
    return false;
}

CGCPlatform cgc_get_platform(void) {
    return g_detected_platform;
}

const char* cgc_get_platform_name(void) {
    return cgc_platform_name(g_detected_platform);
}

cgc_error_t cgc_init(void) {
    g_detected_platform = cgc_detect_platform();
    printf("[CGC C++] Detected platform: %s\n", cgc_platform_name(g_detected_platform));

    switch (g_detected_platform) {
#ifdef CGC_CUDA_ENABLED
        case CGC_PLATFORM_CUDA:
            g_active_backend = &cgc_cuda_backend;
            break;
#endif
#ifdef CGC_METAL_ENABLED
        case CGC_PLATFORM_METAL:
            g_active_backend = &cgc_metal_backend;
            break;
#endif
        case CGC_PLATFORM_CPU:
        default:
            g_active_backend = &cgc_cpu_backend;
            break;
    }

    if (g_active_backend && g_active_backend->init) {
        cgc_error_t err = g_active_backend->init();
        if (err != CGC_OK) {
            printf("[CGC C++] Failed to initialize %s backend, falling back to CPU\n", 
                   g_active_backend->name);
            g_active_backend = &cgc_cpu_backend;
            g_active_backend->init();
        }
    }

    printf("[CGC C++] Engine initialized with %d opcodes using %s backend!\n", 
           NUM_SUPPORTED_OPCODES, g_active_backend->name);
    return CGC_OK;
}

cgc_error_t cgc_destroy(void) {
    if (g_active_backend && g_active_backend->destroy) {
        g_active_backend->destroy();
    }
    
    // g_kda_states is no longer used, kept for backward compatibility if needed
    g_kda_states.clear();
    
    printf("[CGC C++] Engine destroyed!\n");
    return CGC_OK;
}

static cgc_error_t cgc_execute_cpu_fallback(
    int opcode,
    const float** inputs, const int64_t* input_dims, const int* input_ndims, int num_inputs,
    float** outputs, int64_t* output_dims, int* output_ndims, int num_outputs,
    const void* params
) {
    switch (opcode) {
        case 0x20: { // LINEAR_GEMM
            if (num_inputs < 2) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t k = input_dims[1];
            int64_t n = input_dims[2];
            linear_gemm(inputs[0], inputs[1], outputs[0], m, n, k);
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x21: { // LINEAR_BIAS
            if (num_inputs < 3) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t n = input_dims[1];
            linear_gemm(inputs[0], inputs[1], outputs[0], m, n, input_dims[2]);
            for (int64_t i = 0; i < m * n; i++) {
                outputs[0][i] += inputs[2][i % n];
            }
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x22: { // GEMM_BATCHED
            if (num_inputs < 2) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t m = input_dims[1];
            int64_t k = input_dims[2];
            int64_t n = input_dims[3];
            for (int64_t b = 0; b < batch; b++) {
                const float* a = inputs[0] + b * m * k;
                const float* b_mat = inputs[1] + b * k * n;
                float* c = outputs[0] + b * m * n;
                linear_gemm(a, b_mat, c, m, n, k);
            }
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = m;
                output_dims[2] = n;
                output_ndims[0] = 3;
            }
            break;
        }
        case 0x10: { // ATTENTION_SDPA
            if (num_inputs < 3) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t heads = input_dims[1];
            int64_t seqlen = input_dims[2];
            int64_t d = input_dims[3];
            attention_sdpa(inputs[0], inputs[1], inputs[2], outputs[0], batch, heads, seqlen, d);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = heads;
                output_dims[2] = seqlen;
                output_dims[3] = d;
                output_ndims[0] = 4;
            }
            break;
        }
        case 0x11: { // KDA Attention (Legacy, stubbed out)
            break;
        }
        case 0x12: case 0x13: { // ATTENTION_PAGED / ATTENTION_FLASH
            if (num_inputs < 3) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t heads = input_dims[1];
            int64_t seqlen = input_dims[2];
            int64_t d = input_dims[3];
            attention_sdpa(inputs[0], inputs[1], inputs[2], outputs[0], batch, heads, seqlen, d);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = heads;
                output_dims[2] = seqlen;
                output_dims[3] = d;
                output_ndims[0] = 4;
            }
            break;
        }
        case 0x30: { // LAYER_NORM
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t seqlen = input_dims[1];
            int64_t dim = input_dims[2];
            float eps = 1e-6f;
            const float* weight = (num_inputs > 1) ? inputs[1] : nullptr;
            const float* bias = (num_inputs > 2) ? inputs[2] : nullptr;
            layer_norm(inputs[0], outputs[0], batch, seqlen, dim, weight, bias, eps);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = seqlen;
                output_dims[2] = dim;
                output_ndims[0] = 3;
            }
            break;
        }
        case 0x31: { // RMS_NORM
            if (num_inputs < 2) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t seqlen = input_dims[1];
            int64_t d = input_dims[2];
            float eps = 1e-6f;
            rms_norm(inputs[0], inputs[1], outputs[0], eps, batch, seqlen, d);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = seqlen;
                output_dims[2] = d;
                output_ndims[0] = 3;
            }
            break;
        }
        case 0x32: { // GROUP_NORM
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t channels = input_dims[1];
            int64_t height = input_dims[2];
            int64_t width = input_dims[3];
            int64_t num_groups = 32;
            float eps = 1e-6f;
            const float* weight = (num_inputs > 1) ? inputs[1] : nullptr;
            const float* bias = (num_inputs > 2) ? inputs[2] : nullptr;
            group_norm(inputs[0], outputs[0], batch, channels, height, width, num_groups, weight, bias, eps);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = channels;
                output_dims[2] = height;
                output_dims[3] = width;
                output_ndims[0] = 4;
            }
            break;
        }
        case 0x40: { // ROPE
            if (num_inputs < 2) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t seqlen = input_dims[1];
            int64_t d = input_dims[2];
            rope_apply(inputs[0], inputs[1], outputs[0], batch, seqlen, d);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x41: { // ROPE_FUSED
            if (num_inputs < 2) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t head = input_dims[1];
            int64_t seqlen = input_dims[2];
            int64_t dim = input_dims[3];
            rope_fast(outputs[0], batch, head, seqlen, dim, 0, inputs[0], inputs[1]);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x42: { // YARN_ROPE
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t head = input_dims[1];
            int64_t seqlen = input_dims[2];
            int64_t dim = input_dims[3];
            float base = 10000.0f;
            float scale = 1.0f;
            rope_yarn(outputs[0], batch, head, seqlen, dim, 0, base, scale, nullptr);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x50: { // SILU
            if (num_inputs < 1) return CGC_ERROR;
            int64_t size = 1;
            for (int i = 0; i < input_ndims[0]; i++) size *= input_dims[i];
            activation_silu(inputs[0], outputs[0], size);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x51: case 0x52: { // GELU / GELU_TANH
            if (num_inputs < 1) return CGC_ERROR;
            int64_t size = 1;
            for (int i = 0; i < input_ndims[0]; i++) size *= input_dims[i];
            activation_gelu(inputs[0], outputs[0], size);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x53: { // RELU
            if (num_inputs < 1) return CGC_ERROR;
            int64_t size = 1;
            for (int i = 0; i < input_ndims[0]; i++) size *= input_dims[i];
            activation_relu(inputs[0], outputs[0], size);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x54: { // SIGMOID
            if (num_inputs < 1) return CGC_ERROR;
            int64_t size = 1;
            for (int i = 0; i < input_ndims[0]; i++) size *= input_dims[i];
            activation_sigmoid(inputs[0], outputs[0], size);
            if (num_outputs > 0) {
                for (int i = 0; i < input_ndims[0]; i++) output_dims[i] = input_dims[i];
                output_ndims[0] = input_ndims[0];
            }
            break;
        }
        case 0x60: { // SOFTMAX
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t seqlen = input_dims[1];
            int64_t dim = input_dims[2];
            softmax(inputs[0], outputs[0], batch, seqlen, dim);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = seqlen;
                output_dims[2] = dim;
                output_ndims[0] = 3;
            }
            break;
        }
        case 0x61: { // LOG_SOFTMAX
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t seqlen = input_dims[1];
            int64_t dim = input_dims[2];
            softmax(inputs[0], outputs[0], batch, seqlen, dim);
            for (int64_t i = 0; i < batch * seqlen * dim; i++) {
                outputs[0][i] = logf(outputs[0][i] + 1e-10f);
            }
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = seqlen;
                output_dims[2] = dim;
                output_ndims[0] = 3;
            }
            break;
        }
        case 0x62: { // TOP_K
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t vocab_size = input_dims[1];
            int64_t k = 10;
            sample_topk(inputs[0], outputs[0], nullptr, batch, vocab_size, k);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = k;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x63: { // TOP_P
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t vocab_size = input_dims[1];
            float p = 0.9f;
            sample_topp(inputs[0], outputs[0], batch, vocab_size, p);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = vocab_size;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x64: { // TEMPERATURE
            if (num_inputs < 1) return CGC_ERROR;
            int64_t batch = input_dims[0];
            int64_t vocab_size = input_dims[1];
            float temperature = 1.0f;
            sample_temperature(inputs[0], outputs[0], batch, vocab_size, temperature);
            if (num_outputs > 0) {
                output_dims[0] = batch;
                output_dims[1] = vocab_size;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x70: { // KV_CACHE_LOAD
            if (num_inputs < 2) return CGC_ERROR;
            int64_t cache_size = input_dims[0];
            int64_t num_indices = input_dims[1];
            int64_t elem_size = input_dims[2];
            kv_cache_load(inputs[0], outputs[0], (const int64_t*)inputs[1], cache_size, num_indices, elem_size);
            if (num_outputs > 0) {
                output_dims[0] = num_indices;
                output_dims[1] = elem_size;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x71: { // KV_CACHE_STORE
            if (num_inputs < 3) return CGC_ERROR;
            int64_t cache_size = input_dims[0];
            int64_t num_indices = input_dims[1];
            int64_t elem_size = input_dims[2];
            kv_cache_store((float*)inputs[0], inputs[1], (const int64_t*)inputs[2], cache_size, num_indices, elem_size);
            break;
        }
        case 0x72: { // KV_CACHE_UPDATE
            if (num_inputs < 3) return CGC_ERROR;
            int64_t cache_size = input_dims[0];
            int64_t num_updates = input_dims[1];
            int64_t elem_size = input_dims[2];
            kv_cache_update((float*)inputs[0], inputs[1], (const int64_t*)inputs[2], cache_size, num_updates, elem_size);
            if (num_outputs > 0) {
                output_dims[0] = cache_size;
                output_dims[1] = elem_size;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x73: { // EMBEDDING_LOOKUP
            if (num_inputs < 2) return CGC_ERROR;
            int64_t vocab_size = input_dims[0];
            int64_t embed_dim = input_dims[1];
            int64_t num_indices = input_dims[2];
            for (int64_t i = 0; i < num_indices; i++) {
                int64_t idx = (int64_t)inputs[1][i];
                const float* embed = inputs[0] + idx * embed_dim;
                float* out = outputs[0] + i * embed_dim;
                for (int64_t j = 0; j < embed_dim; j++) out[j] = embed[j];
            }
            if (num_outputs > 0) {
                output_dims[0] = num_indices;
                output_dims[1] = embed_dim;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0x74: case 0x75: { // KV_CACHE_STATIC_LAYOUT / KV_CACHE_COMMIT
            break;
        }
        case 0xA0: { // QUANTIZE_W8A16
            if (num_inputs < 1) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t n = input_dims[1];
            quant_w8a16(inputs[0], outputs[0], m, n);
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0xA1: { // QUANTIZE_W4A16
            if (num_inputs < 1) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t n = input_dims[1];
            int64_t size = m * n;
            float* scales = (float*)malloc((size / 32) * sizeof(float));
            quantize_q4(inputs[0], (uint8_t*)outputs[0], size, scales, 32);
            free(scales);
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n / 2;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0xA2: { // DEQUANTIZE
            if (num_inputs < 2) return CGC_ERROR;
            int64_t size = input_dims[0];
            dequantize_q4((const uint8_t*)inputs[0], outputs[0], size, inputs[1], 32);
            if (num_outputs > 0) {
                output_dims[0] = size;
                output_ndims[0] = 1;
            }
            break;
        }
        case 0xA3: case 0xA4: { // GPTQ_KERNEL / AWQ_KERNEL
            if (num_inputs < 2) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t n = input_dims[1];
            quant_gguf_q4(inputs[0], outputs[0], m, n);
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n;
                output_ndims[0] = 2;
            }
            break;
        }
        case 0xC3: { // LLAMA_Q4_K_MATMUL
            if (num_inputs < 2) return CGC_ERROR;
            int64_t m = input_dims[0];
            int64_t k = input_dims[1];
            int64_t n = input_dims[2];
            quant_gguf_q4(inputs[0], outputs[0], m, n);
            if (num_outputs > 0) {
                output_dims[0] = m;
                output_dims[1] = n;
                output_ndims[0] = 2;
            }
            break;
        }
        default:
            fprintf(stderr, "[CGC C++] Unsupported opcode: 0x%02x\n", opcode);
            return CGC_ERROR;
    }
    return CGC_OK;
}

cgc_error_t cgc_execute_opcode(
    int opcode,
    const float** inputs, const int64_t* input_dims, const int* input_ndims, int num_inputs,
    float** outputs, int64_t* output_dims, int* output_ndims, int num_outputs,
    const void* params
) {
    int actual_opcode = opcode;

    if (g_kda_replace_mode && opcode == 0x10) {
        actual_opcode = 0x11;
        printf("[CGC C++] KDA Replace Mode: 0x10 (SDPA) -> 0x11 (KDA)\n");
    }

    printf("[CGC C++] Executing opcode: 0x%02x on %s backend\n", actual_opcode, g_active_backend->name);

    if (g_active_backend && g_active_backend->execute) {
        int* params_int = const_cast<int*>(static_cast<const int*>(params));
        cgc_error_t err = g_active_backend->execute(actual_opcode, inputs, outputs, params_int, num_inputs, num_outputs);

        if (err == CGC_ERROR_NOT_SUPPORTED) {
            printf("[CGC C++] Opcode 0x%02x not supported by %s backend, falling back to CPU\n",
                   actual_opcode, g_active_backend->name);
            return cgc_execute_cpu_fallback(actual_opcode, inputs, input_dims, input_ndims, num_inputs,
                                           outputs, output_dims, output_ndims, num_outputs, params);
        }
        return err;
    }

    return cgc_execute_cpu_fallback(actual_opcode, inputs, input_dims, input_ndims, num_inputs,
                                   outputs, output_dims, output_ndims, num_outputs, params);
}

// -----------------------------------------------------------------------------
// Hardware Bus Layer / MMAP Zero-Copy API
// -----------------------------------------------------------------------------

void* cgc_mmap_file(const char* filepath, size_t* out_size) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[CGC Hardware Bus] Failed to open file for mmap: %s\n", filepath);
        *out_size = 0;
        return nullptr;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        fprintf(stderr, "[CGC Hardware Bus] Failed to stat file\n");
        close(fd);
        *out_size = 0;
        return nullptr;
    }
    
    size_t size = sb.st_size;
    *out_size = size;
    
    void* mapped = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); // fd can be closed after mmap
    
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[CGC Hardware Bus] mmap failed\n");
        return nullptr;
    }
    
    // 強制 OS 立即載入分頁，避免 Decode 階段產生 Page Fault 延遲
    if (mlock(mapped, size) != 0) {
        fprintf(stderr, "[CGC Hardware Bus] mlock failed, page faults may occur during decode\n");
    } else {
        printf("[CGC Hardware Bus] mlock successful, eliminated Page Fault delay\n");
    }
    
    printf("[CGC Hardware Bus] Successfully memory-mapped %zu bytes from %s\n", size, filepath);
    return mapped;
}

void cgc_munmap_file(void* ptr, size_t size) {
    if (ptr && ptr != MAP_FAILED) {
        munlock(ptr, size); // 解除鎖定
        munmap(ptr, size);
    }
}
