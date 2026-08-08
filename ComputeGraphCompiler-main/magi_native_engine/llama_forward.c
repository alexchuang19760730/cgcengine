#include "llama_forward.h"
#include "metal_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static metal_device_t* g_dev = NULL;
static ModelConfig g_config;
static ModelWeights* g_weights = NULL;

static metal_buffer_t* buf_hidden = NULL;
static metal_buffer_t* buf_qkv = NULL;
static metal_buffer_t* buf_attn_out = NULL;
static metal_buffer_t* buf_ffn_up = NULL;
static metal_buffer_t* buf_ffn_gate = NULL;
static metal_buffer_t* buf_ffn_out = NULL;
static metal_buffer_t* buf_logits = NULL;
static metal_buffer_t* buf_residual = NULL;
static metal_buffer_t* buf_tokens = NULL;

static int sample_token(metal_buffer_t* logits, int vocab) {
    float* logits_ptr = metal_buffer_get_host_ptr(logits);
    float max_logit = logits_ptr[0];
    int max_idx = 0;
    float sum = max_logit;
    for (int i = 1; i < vocab; i++) {
        if (logits_ptr[i] > max_logit) {
            max_logit = logits_ptr[i];
            max_idx = i;
        }
        sum += logits_ptr[i];
    }

    static int last_idx = 0;
    static int call_count = 0;
    call_count++;

    if (call_count <= 3) {
        printf("    [Sample] step=%d max_logit=%.4f max_idx=%d sum=%.2f\n",
               call_count, max_logit, max_idx, sum);
    }

    float r = (float)rand() / RAND_MAX;
    float cumsum = 0;
    float temperature = 0.8f;

    float* probs = (float*)malloc(vocab * sizeof(float));
    if (!probs) return max_idx;

    float max_l = logits_ptr[0];
    for (int i = 1; i < vocab; i++) {
        if (logits_ptr[i] > max_l) max_l = logits_ptr[i];
    }

    float exp_sum = 0;
    for (int i = 0; i < vocab; i++) {
        float exp_val = expf((logits_ptr[i] - max_l) / temperature);
        probs[i] = exp_val;
        exp_sum += exp_val;
    }

    for (int i = 0; i < vocab; i++) {
        probs[i] /= exp_sum;
    }

    for (int i = 0; i < vocab; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            free(probs);
            last_idx = i;
            return i;
        }
    }

    free(probs);
    last_idx = max_idx;
    return max_idx;
}

static void residual_add(metal_buffer_t* x, metal_buffer_t* residual, int size) {
    float* x_ptr = metal_buffer_get_host_ptr(x);
    float* res_ptr = metal_buffer_get_host_ptr(residual);
    for (int i = 0; i < size; i++) {
        x_ptr[i] = x_ptr[i] + res_ptr[i];
    }
}

static void swiglu_inplace(metal_buffer_t* x, metal_buffer_t* gate, int size) {
    float* x_ptr = metal_buffer_get_host_ptr(x);
    float* gate_ptr = metal_buffer_get_host_ptr(gate);
    for (int i = 0; i < size; i++) {
        float silu = gate_ptr[i] / (1.0f + expf(-gate_ptr[i]));
        x_ptr[i] = x_ptr[i] * silu;
    }
}

void llama_forward(
    int n_layer_p,
    int dim_p,
    int n_head_p,
    int n_kv_head_p,
    int head_dim_p,
    int vocab_size_p,
    int* tokens,
    int n_tokens,
    int* out_tokens,
    int max_gen
) {
    g_config.n_layer = n_layer_p;
    g_config.dim = dim_p;
    g_config.n_head = n_head_p;
    g_config.n_kv_head = n_kv_head_p;
    g_config.head_dim = head_dim_p;
    g_config.vocab_size = vocab_size_p;
    g_config.max_seq = 2048;

    printf("\n");
    printf("======================================\n");
    printf("🔥 MagiCompiler KDA 原生推理引擎\n");
    printf("🧠 模型配置:\n");
    printf("   • 层数: %d\n", g_config.n_layer);
    printf("   • 维度: %d\n", g_config.dim);
    printf("   • 头数: %d (KV: %d)\n", g_config.n_head, g_config.n_kv_head);
    printf("   • Head Dim: %d\n", g_config.head_dim);
    printf("   • 词表: %d\n", g_config.vocab_size);
    printf("======================================\n\n");

    double init_start = get_time_ms();

    g_dev = metal_device_create();
    if (!g_dev) {
        printf("    [错误] 无法创建 Metal 设备\n");
        return;
    }

    printf("    [Metal] 设备名: %s\n", metal_get_device_name());
    printf("    [Metal] 加速: %s\n", metal_is_available() ? "启用" : "禁用 (CPU回退)");

    g_weights = metal_load_gguf_weights(g_dev, NULL, &g_config);
    if (!g_weights) {
        printf("    [错误] 权重加载失败\n");
        metal_device_destroy(g_dev);
        return;
    }

    size_t hidden_size = (size_t)g_config.dim * sizeof(float);
    size_t qkv_size = (size_t)g_config.dim * 3 * sizeof(float);
    size_t logits_size = (size_t)g_config.vocab_size * sizeof(float);
    size_t token_size = sizeof(int);

    buf_hidden = metal_buffer_create(g_dev, hidden_size);
    buf_qkv = metal_buffer_create(g_dev, qkv_size);
    buf_attn_out = metal_buffer_create(g_dev, hidden_size);
    buf_ffn_up = metal_buffer_create(g_dev, hidden_size);
    buf_ffn_gate = metal_buffer_create(g_dev, hidden_size);
    buf_ffn_out = metal_buffer_create(g_dev, hidden_size);
    buf_logits = metal_buffer_create(g_dev, logits_size);
    buf_residual = metal_buffer_create(g_dev, hidden_size);
    buf_tokens = metal_buffer_create(g_dev, token_size);

    float init_time = get_time_ms() - init_start;
    printf("\n    [计时] 初始化: %.2f ms\n\n", init_time);

    int* tok_ptr = (int*)metal_buffer_get_host_ptr(buf_tokens);
    tok_ptr[0] = tokens[0];

    float* hidden_ptr = metal_buffer_get_host_ptr(buf_hidden);
    float* embed_ptr = metal_buffer_get_host_ptr(g_weights->embedding);
    for (int i = 0; i < g_config.dim; i++) {
        hidden_ptr[i] = embed_ptr[tokens[0] * g_config.dim + i];
    }
    for (int i = 0; i < g_config.dim; i++) {
        float* res_ptr = metal_buffer_get_host_ptr(buf_residual);
        res_ptr[i] = hidden_ptr[i];
    }

    double total_start = get_time_ms();

    for (int step = 0; step < max_gen; step++) {
        double step_start = get_time_ms();
        printf("【Step %d】\n", step + 1);

        for (int layer = 0; layer < g_config.n_layer; layer++) {
            if (layer == 0 || layer == g_config.n_layer - 1 || layer == g_config.n_layer / 2) {
                printf("  ── Layer %d ──\n", layer);
            }

            metal_rms_norm(g_dev, buf_hidden,
                          g_weights->layer_attn_norm ?
                          metal_buffer_create_from_data(g_dev,
                              metal_buffer_get_host_ptr(g_weights->layer_attn_norm) + layer * g_config.dim,
                              g_config.dim * sizeof(float)) : NULL,
                          buf_hidden, g_config.dim);

            float* qkv_w = metal_buffer_get_host_ptr(g_weights->layer_qkv) + layer * g_config.dim * g_config.dim * 3;
            metal_buffer_t* qkv_weight_buf = metal_buffer_create_from_data(g_dev, qkv_w, g_config.dim * g_config.dim * 3 * sizeof(float));
            metal_gemm(g_dev, buf_hidden, qkv_weight_buf, buf_qkv,
                      1, g_config.dim * 3, g_config.dim, false, false);
            metal_buffer_destroy(qkv_weight_buf);

            float* q_ptr = metal_buffer_get_host_ptr(buf_qkv);
            float* k_ptr = q_ptr + g_config.dim;
            float* v_ptr = k_ptr + g_config.dim;

            metal_rope(g_dev, buf_qkv, metal_buffer_create_from_data(g_dev, k_ptr, g_config.dim * sizeof(float)),
                      g_weights->rope_cos, g_weights->rope_sin, 1, g_config.dim);

            if (layer == 0 || layer == g_config.n_layer - 1) {
                metal_buffer_t* q_buf = metal_buffer_create_from_data(g_dev, q_ptr, g_config.dim * sizeof(float));
                metal_buffer_t* k_buf = metal_buffer_create_from_data(g_dev, k_ptr, g_config.dim * sizeof(float));
                metal_buffer_t* v_buf = metal_buffer_create_from_data(g_dev, v_ptr, g_config.dim * sizeof(float));

                metal_kda_attention(g_dev, q_buf, k_buf, v_buf, buf_attn_out,
                                   g_config.n_head, g_config.n_kv_head, g_config.head_dim, 1, 1.0f);

                metal_buffer_destroy(q_buf);
                metal_buffer_destroy(k_buf);
                metal_buffer_destroy(v_buf);
            }

            float* o_w = metal_buffer_get_host_ptr(g_weights->layer_o) + layer * g_config.dim * g_config.dim;
            metal_buffer_t* o_weight_buf = metal_buffer_create_from_data(g_dev, o_w, g_config.dim * g_config.dim * sizeof(float));
            metal_gemm(g_dev, buf_attn_out, o_weight_buf, buf_hidden,
                      1, g_config.dim, g_config.dim, false, false);
            metal_buffer_destroy(o_weight_buf);

            float* res_ptr = metal_buffer_get_host_ptr(buf_residual);
            float* hid_ptr = metal_buffer_get_host_ptr(buf_hidden);
            for (int i = 0; i < g_config.dim; i++) {
                res_ptr[i] = hid_ptr[i];
            }

            metal_rms_norm(g_dev, buf_hidden,
                          g_weights->layer_ffn_norm ?
                          metal_buffer_create_from_data(g_dev,
                              metal_buffer_get_host_ptr(g_weights->layer_ffn_norm) + layer * g_config.dim,
                              g_config.dim * sizeof(float)) : NULL,
                          buf_hidden, g_config.dim);

            float* ffn_up_w = metal_buffer_get_host_ptr(g_weights->layer_ffn_up) + layer * g_config.dim * g_config.dim;
            float* ffn_gate_w = metal_buffer_get_host_ptr(g_weights->layer_ffn_gate) + layer * g_config.dim * g_config.dim;

            metal_buffer_t* ffn_up_w_buf = metal_buffer_create_from_data(g_dev, ffn_up_w, g_config.dim * g_config.dim * sizeof(float));
            metal_buffer_t* ffn_gate_w_buf = metal_buffer_create_from_data(g_dev, ffn_gate_w, g_config.dim * g_config.dim * sizeof(float));

            metal_gemm(g_dev, buf_hidden, ffn_up_w_buf, buf_ffn_up, 1, g_config.dim, g_config.dim, false, false);
            metal_gemm(g_dev, buf_hidden, ffn_gate_w_buf, buf_ffn_gate, 1, g_config.dim, g_config.dim, false, false);

            metal_buffer_destroy(ffn_up_w_buf);
            metal_buffer_destroy(ffn_gate_w_buf);

            float* up_ptr = metal_buffer_get_host_ptr(buf_ffn_up);
            float* gate_ptr = metal_buffer_get_host_ptr(buf_ffn_gate);
            for (int i = 0; i < g_config.dim; i++) {
                gate_ptr[i] = gate_ptr[i] / (1.0f + expf(-gate_ptr[i]));
                up_ptr[i] = up_ptr[i] * gate_ptr[i];
            }

            float* ffn_down_w = metal_buffer_get_host_ptr(g_weights->layer_ffn_down) + layer * g_config.dim * g_config.dim;
            metal_buffer_t* ffn_down_w_buf = metal_buffer_create_from_data(g_dev, ffn_down_w, g_config.dim * g_config.dim * sizeof(float));
            metal_gemm(g_dev, buf_ffn_up, ffn_down_w_buf, buf_ffn_out, 1, g_config.dim, g_config.dim, false, false);
            metal_buffer_destroy(ffn_down_w_buf);

            float* ffn_out_ptr = metal_buffer_get_host_ptr(buf_ffn_out);
            float* hid_ptr2 = metal_buffer_get_host_ptr(buf_hidden);
            float* res_ptr2 = metal_buffer_get_host_ptr(buf_residual);
            for (int i = 0; i < g_config.dim; i++) {
                res_ptr2[i] = hid_ptr2[i] + ffn_out_ptr[i];
                hid_ptr2[i] = res_ptr2[i];
            }
        }

        metal_rms_norm(g_dev, buf_hidden, g_weights->final_norm, buf_hidden, g_config.dim);

        metal_lm_head(g_dev, buf_hidden, g_weights->lm_head, buf_logits, g_config.dim, g_config.vocab_size);

        int next_tok = sample_token(buf_logits, g_config.vocab_size);
        out_tokens[step] = next_tok;

        tok_ptr[0] = next_tok;

        float* embed_ptr2 = metal_buffer_get_host_ptr(g_weights->embedding);
        float* hid_ptr3 = metal_buffer_get_host_ptr(buf_hidden);
        for (int i = 0; i < g_config.dim; i++) {
            hid_ptr3[i] = embed_ptr2[next_tok * g_config.dim + i];
        }

        float* res_ptr3 = metal_buffer_get_host_ptr(buf_residual);
        for (int i = 0; i < g_config.dim; i++) {
            res_ptr3[i] = hid_ptr3[i];
        }

        double step_time = get_time_ms() - step_start;
        printf("  ✓ 完成: %.2f ms\n", step_time);
    }

    double total_time = get_time_ms() - total_start;

    printf("\n======================================\n");
    printf("✅ 推理完成!\n");
    printf("======================================\n");
    printf("📊 总时间: %.2f ms\n", total_time);
    printf("📊 平均: %.2f ms/step\n", total_time / max_gen);
    printf("📝 输出: ");
    for (int i = 0; i < max_gen; i++) {
        printf("%d ", out_tokens[i]);
    }
    printf("\n");

    if (buf_hidden) metal_buffer_destroy(buf_hidden);
    if (buf_qkv) metal_buffer_destroy(buf_qkv);
    if (buf_attn_out) metal_buffer_destroy(buf_attn_out);
    if (buf_ffn_up) metal_buffer_destroy(buf_ffn_up);
    if (buf_ffn_gate) metal_buffer_destroy(buf_ffn_gate);
    if (buf_ffn_out) metal_buffer_destroy(buf_ffn_out);
    if (buf_logits) metal_buffer_destroy(buf_logits);
    if (buf_residual) metal_buffer_destroy(buf_residual);
    if (buf_tokens) metal_buffer_destroy(buf_tokens);

    if (g_weights) metal_weights_destroy(g_weights);
    if (g_dev) metal_device_destroy(g_dev);
}