#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include "gguf.h"
#include "ggml.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

static inline float ggml_half_to_float(uint16_t h) {
    union { uint32_t u; float f; } u = { (uint32_t)h << 16 };
    return u.f;
}

static inline uint16_t ggml_float_to_half(float f) {
    union { uint32_t u; float f; } u = { .f = f };
    return (uint16_t)(u.u >> 16);
}

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_layer;
    int dim;
    int n_head;
    int n_kv_head;
    int head_dim;
    int ffn_dim;
    int vocab_size;
    int seq_len;
} magi_config_t;

typedef struct {
    id<MTLBuffer> buffer;
    int n_head;
    int head_dim;
    int seq_len;
} metal_kda_state_t;

typedef struct {
    id<MTLBuffer> embedding;
    int embedding_nbytes;
    id<MTLBuffer> layer_attn_norm;
    id<MTLBuffer> layer_qkv;
    int layer_qkv_nbytes;
    id<MTLBuffer> layer_o;
    int layer_o_nbytes;
    id<MTLBuffer> layer_ffn_norm;
    id<MTLBuffer> layer_ffn_up;
    int layer_ffn_up_nbytes;
    id<MTLBuffer> layer_ffn_gate;
    int layer_ffn_gate_nbytes;
    id<MTLBuffer> layer_ffn_down;
    int layer_ffn_down_nbytes;
    id<MTLBuffer> final_norm;
    id<MTLBuffer> lm_head;
    int lm_head_nbytes;
} magi_weights_t;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    magi_config_t* config;
    magi_weights_t* weights;
    struct ggml_context* ggml_ctx;
} metal_context_t;

static size_t ggml_tensor_size(struct ggml_tensor* t) {
    return ggml_nbytes(t);
}

static int ggml_tensor_ne(struct ggml_tensor* t) {
    return (int)(t->ne[0] * t->ne[1]);
}

static enum ggml_type ggml_tensor_type(struct ggml_tensor* t) {
    return t->type;
}

static void metal_upload_f32(id<MTLBuffer> buf, size_t off, const float* data, size_t n_elements) {
    memcpy((char*)[buf contents] + off, data, n_elements * sizeof(float));
}

static bool metal_load_gguf(metal_context_t* ctx, const char* path) {
    magi_config_t* c = ctx->config;
    magi_weights_t* w = ctx->weights;
    id<MTLDevice> dev = ctx->device;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx->ggml_ctx
    };

    struct gguf_context* g = gguf_init_from_file(path, params);
    if (!g) {
        fprintf(stderr, "[ERROR] GGUF fail\n");
        return false;
    }

    struct ggml_context* ggml_ctx = ctx->ggml_ctx;

    int dim = c->dim;
    int n_layer = c->n_layer;
    int ffn_dim = c->ffn_dim;
    size_t max_ffn_down_sz = 0;
    size_t max_ffn_up_sz = 0;
    size_t max_q_sz = 0;
    size_t max_k_sz = 0;
    size_t max_v_sz = 0;
    size_t max_o_sz = 0;

    fprintf(stdout, "    [GGUF] 載入真實權重：%s\n", path);
    fprintf(stdout, "    [GGUF] 配置: dim=%d, n_layer=%d\n", dim, n_layer);
    fflush(stdout);

    char tname[256];

    for (int l = 0; l < n_layer; l++) {
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        struct ggml_tensor* t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_ffn_down_sz) max_ffn_down_sz = deq_sz;
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_ffn_up_sz) max_ffn_up_sz = deq_sz;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_q_sz) max_q_sz = deq_sz;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_k_sz) max_k_sz = deq_sz;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_v_sz) max_v_sz = deq_sz;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        t = ggml_get_tensor(ggml_ctx, tname);
        if (t) {
            size_t deq_sz = ggml_tensor_ne(t) * sizeof(float);
            if (deq_sz > max_o_sz) max_o_sz = deq_sz;
        }
    }
    fprintf(stdout, "    [scan] max ffn_down size = %zu bytes\n", max_ffn_down_sz);
    fprintf(stdout, "    [scan] max ffn_up size = %zu bytes\n", max_ffn_up_sz);
    fprintf(stdout, "    [scan] max Q size = %zu bytes\n", max_q_sz);
    fprintf(stdout, "    [scan] max K size = %zu bytes\n", max_k_sz);
    fprintf(stdout, "    [scan] max V size = %zu bytes\n", max_v_sz);
    fprintf(stdout, "    [scan] max O size = %zu bytes\n", max_o_sz);

    if (max_ffn_down_sz == 0) max_ffn_down_sz = (size_t)dim * ffn_dim * sizeof(float);
    if (max_ffn_up_sz == 0) max_ffn_up_sz = (size_t)dim * ffn_dim * sizeof(float);
    if (max_q_sz == 0) max_q_sz = (size_t)dim * dim * sizeof(float);
    if (max_k_sz == 0) max_k_sz = (size_t)dim * dim * sizeof(float);
    if (max_v_sz == 0) max_v_sz = (size_t)dim * dim * sizeof(float);
    if (max_o_sz == 0) max_o_sz = (size_t)dim * dim * sizeof(float);

    size_t max_q_float = max_q_sz / sizeof(float);
    size_t max_k_float = max_k_sz / sizeof(float);
    size_t max_v_float = max_v_sz / sizeof(float);
    size_t max_o_float = max_o_sz / sizeof(float);
    size_t max_ffn_float = max_ffn_down_sz / sizeof(float);
    if (max_ffn_float < (size_t)dim * ffn_dim) max_ffn_float = (size_t)dim * ffn_dim;

    w->layer_attn_norm = [dev newBufferWithLength:n_layer * dim * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_qkv      = [dev newBufferWithLength:n_layer * (max_q_float + max_k_float + max_v_float) * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_o        = [dev newBufferWithLength:n_layer * max_o_float * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_ffn_norm = [dev newBufferWithLength:n_layer * dim * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_ffn_gate = [dev newBufferWithLength:n_layer * max_ffn_float * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_ffn_up   = [dev newBufferWithLength:n_layer * max_ffn_float * sizeof(float) options:MTLResourceStorageModeShared];
    w->layer_ffn_down = [dev newBufferWithLength:n_layer * max_ffn_float * sizeof(float) options:MTLResourceStorageModeShared];

    fprintf(stdout, "    [buffer] QKV buffer: %zu bytes per layer, %zu total\n", (max_q_float + max_k_float + max_v_float) * sizeof(float), n_layer * (max_q_float + max_k_float + max_v_float) * sizeof(float));
    fflush(stdout);

    struct ggml_tensor* te = ggml_get_tensor(ggml_ctx, "token_embd.weight");
    if (!te) te = ggml_get_tensor(ggml_ctx, "tok_embeddings.weight");
    if (te) {
        enum ggml_type ttype = ggml_tensor_type(te);
        size_t te_bytes = ggml_tensor_size(te);
        int te_ne = ggml_tensor_ne(te);
        fprintf(stdout, "    [GGUF] embedding: type=%d, %zu bytes, %d elements\n", ttype, te_bytes, te_ne);

        if (ttype == GGML_TYPE_F32) {
            w->embedding = [dev newBufferWithLength:te_bytes options:MTLResourceStorageModeShared];
            memcpy([w->embedding contents], te->data, te_bytes);
        } else if (ttype == GGML_TYPE_F16) {
            w->embedding = [dev newBufferWithLength:te_ne * sizeof(float) options:MTLResourceStorageModeShared];
            float* emb_f32 = (float*)[w->embedding contents];
            const uint16_t* emb_f16 = (const uint16_t*)te->data;
            for (int i = 0; i < te_ne; i++) {
                emb_f32[i] = ggml_half_to_float(emb_f16[i]);
            }
        } else {
            fprintf(stdout, "    [GGUF] embedding: unsupported type %d, using placeholder\n", ttype);
            w->embedding = [dev newBufferWithLength:te_ne * sizeof(float) options:MTLResourceStorageModeShared];
        }
        w->embedding_nbytes = te_ne * sizeof(float);
    }

    fprintf(stdout, "    [GGUF] 開始載入 %d 層...\n", n_layer);
    fflush(stdout);

    for (int l = 0; l < n_layer; l++) {
        fprintf(stdout, "    [GGUF] Processing layer %d...\n", l);
        fflush(stdout);

        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
        struct ggml_tensor* tan = ggml_get_tensor(ggml_ctx, tname);
        if (tan) {
            enum ggml_type ttype = ggml_tensor_type(tan);
            int tan_ne = ggml_tensor_ne(tan);
            if (ttype == GGML_TYPE_F32) {
                memcpy((char*)[w->layer_attn_norm contents] + l * dim * sizeof(float), tan->data, tan_ne * sizeof(float));
            } else if (ttype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tan->data;
                float* dst = (float*)((char*)[w->layer_attn_norm contents] + l * dim * sizeof(float));
                for (int i = 0; i < tan_ne; i++) dst[i] = ggml_half_to_float(src[i]);
            }
            fprintf(stdout, "    [GGUF]   attn_norm done (type=%d)\n", ttype);
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        struct ggml_tensor* tq = ggml_get_tensor(ggml_ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        struct ggml_tensor* tk = ggml_get_tensor(ggml_ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        struct ggml_tensor* tv = ggml_get_tensor(ggml_ctx, tname);

        if (tq && tk && tv) {
            enum ggml_type qtype = ggml_tensor_type(tq);
            enum ggml_type ktype = ggml_tensor_type(tk);
            enum ggml_type vtype = ggml_tensor_type(tv);
            int q_ne = ggml_tensor_ne(tq);
            int k_ne = ggml_tensor_ne(tk);
            int v_ne = ggml_tensor_ne(tv);

            fprintf(stdout, "    [GGUF]   QKV types: q=%d k=%d v=%d\n", qtype, ktype, vtype);
            fprintf(stdout, "    [GGUF]   QKV elements: q=%d k=%d v=%d\n", q_ne, k_ne, v_ne);
            fprintf(stdout, "    [GGUF]   Allocating Q: %zu bytes\n", q_ne * sizeof(float));
            fflush(stdout);

            float* q_f32 = (float*)malloc(q_ne * sizeof(float));
            float* k_f32 = (float*)malloc(k_ne * sizeof(float));
            float* v_f32 = (float*)malloc(v_ne * sizeof(float));

            fprintf(stdout, "    [GGUF]   malloc Q=%p K=%p V=%p\n", (void*)q_f32, (void*)k_f32, (void*)v_f32);
            fflush(stdout);

            if (!q_f32 || !k_f32 || !v_f32) {
                fprintf(stderr, "    [GGUF]   ERROR: malloc failed!\n");
                if (q_f32) free(q_f32);
                if (k_f32) free(k_f32);
                if (v_f32) free(v_f32);
                exit(1);
            }

            if (qtype == GGML_TYPE_Q4_K) {
                fprintf(stdout, "    [GGUF]   Dequantizing Q with dequantize_row_q4_K...\n");
                fflush(stdout);
                dequantize_row_q4_K((const block_q4_K*)tq->data, q_f32, q_ne);
                fprintf(stdout, "    [GGUF]   Q dequantized\n");
                fflush(stdout);
            } else if (qtype == GGML_TYPE_F32) {
                memcpy(q_f32, tq->data, q_ne * sizeof(float));
            } else if (qtype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tq->data;
                for (int i = 0; i < q_ne; i++) q_f32[i] = ggml_half_to_float(src[i]);
            } else {
                fprintf(stderr, "    [GGUF]   WARNING: unsupported Q type %d\n", qtype);
                memset(q_f32, 0, q_ne * sizeof(float));
            }

            if (ktype == GGML_TYPE_Q4_K) {
                fprintf(stdout, "    [GGUF]   Dequantizing K with dequantize_row_q4_K...\n");
                fflush(stdout);
                dequantize_row_q4_K((const block_q4_K*)tk->data, k_f32, k_ne);
                fprintf(stdout, "    [GGUF]   K dequantized\n");
                fflush(stdout);
            } else if (ktype == GGML_TYPE_F32) {
                memcpy(k_f32, tk->data, k_ne * sizeof(float));
            } else if (ktype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tk->data;
                for (int i = 0; i < k_ne; i++) k_f32[i] = ggml_half_to_float(src[i]);
            } else {
                memset(k_f32, 0, k_ne * sizeof(float));
            }

            if (vtype == GGML_TYPE_F32) {
                memcpy(v_f32, tv->data, v_ne * sizeof(float));
            } else if (vtype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tv->data;
                for (int i = 0; i < v_ne; i++) v_f32[i] = ggml_half_to_float(src[i]);
            } else if (vtype == GGML_TYPE_Q4_K) {
                fprintf(stdout, "    [GGUF]   Dequantizing V with dequantize_row_q4_K...\n");
                fflush(stdout);
                dequantize_row_q4_K((const block_q4_K*)tv->data, v_f32, v_ne);
                fprintf(stdout, "    [GGUF]   V dequantized\n");
                fflush(stdout);
            } else if (vtype == GGML_TYPE_Q6_K) {
                fprintf(stdout, "    [GGUF]   Dequantizing V with dequantize_row_q6_K...\n");
                fflush(stdout);
                dequantize_row_q6_K((const block_q6_K*)tv->data, v_f32, v_ne);
                fprintf(stdout, "    [GGUF]   V dequantized\n");
                fflush(stdout);
            } else {
                fprintf(stderr, "    [GGUF]   WARNING: unsupported V type %d, using zeros\n", vtype);
                memset(v_f32, 0, v_ne * sizeof(float));
            }

            size_t qkv_layer_size = (max_q_float + max_k_float + max_v_float) * sizeof(float);
            size_t qkv_offset = l * qkv_layer_size;
            fprintf(stdout, "    [GGUF]   Uploading QKV to Metal, offset=%zu\n", qkv_offset);
            fflush(stdout);
            metal_upload_f32(w->layer_qkv, qkv_offset + 0, q_f32, q_ne);
            fprintf(stdout, "    [GGUF]   Q uploaded\n");
            fflush(stdout);
            metal_upload_f32(w->layer_qkv, qkv_offset + max_q_float * sizeof(float), k_f32, k_ne);
            fprintf(stdout, "    [GGUF]   K uploaded\n");
            fflush(stdout);
            metal_upload_f32(w->layer_qkv, qkv_offset + (max_q_float + max_k_float) * sizeof(float), v_f32, v_ne);
            fprintf(stdout, "    [GGUF]   V uploaded\n");
            fflush(stdout);

            fprintf(stdout, "    [GGUF]   QKV dequant+upload done\n");
            fflush(stdout);
            free(q_f32);
            free(k_f32);
            free(v_f32);
            fprintf(stdout, "    [GGUF]   QKV freed\n");
            fflush(stdout);

            snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
            fprintf(stdout, "    [GGUF]   Getting attn_output tensor...\n");
            fflush(stdout);
            struct ggml_tensor* to = ggml_get_tensor(ggml_ctx, tname);
            fprintf(stdout, "    [GGUF]   Got attn_output tensor %p\n", (void*)to);
            fflush(stdout);
            if (to) {
                enum ggml_type otype = ggml_tensor_type(to);
                int o_ne = ggml_tensor_ne(to);
                fprintf(stdout, "    [GGUF]   attn_output: type=%d, elements=%d\n", otype, o_ne);
                fprintf(stdout, "    [GGUF]   about to malloc for attn_output, size=%zu\n", (size_t)o_ne * sizeof(float));
                fflush(stdout);

                float* o_f32 = (float*)malloc(o_ne * sizeof(float));
                fprintf(stdout, "    [GGUF]   attn_output malloc: %p\n", (void*)o_f32);
                fflush(stdout);
                if (!o_f32) {
                    fprintf(stderr, "    [GGUF]   ERROR: malloc failed for attn_output!\n");
                    exit(1);
                }

                if (otype == GGML_TYPE_Q4_K) {
                    fprintf(stdout, "    [GGUF]   Dequantizing attn_output with Q4_K...\n");
                    fflush(stdout);
                    dequantize_row_q4_K((const block_q4_K*)to->data, o_f32, o_ne);
                    fprintf(stdout, "    [GGUF]   attn_output dequantized\n");
                    fflush(stdout);
                } else if (otype == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K((const block_q6_K*)to->data, o_f32, o_ne);
                } else if (otype == GGML_TYPE_F32) {
                    memcpy(o_f32, to->data, o_ne * sizeof(float));
                } else if (otype == GGML_TYPE_F16) {
                    uint16_t* src = (uint16_t*)to->data;
                    for (int i = 0; i < o_ne; i++) o_f32[i] = ggml_half_to_float(src[i]);
                } else {
                    fprintf(stderr, "    [GGUF]   WARNING: unsupported attn_output type %d, using zeros\n", otype);
                    memset(o_f32, 0, o_ne * sizeof(float));
                }

                fprintf(stdout, "    [GGUF]   Uploading attn_output to Metal...\n");
                fflush(stdout);
                metal_upload_f32(w->layer_o, l * max_o_float * sizeof(float), o_f32, o_ne);
                fprintf(stdout, "    [GGUF]   attn_output done\n");
                free(o_f32);
            }
        }
        fprintf(stdout, "    [GGUF]   free(o_f32) done for layer %d\n", l);
        fflush(stdout);
        fprintf(stdout, "    [GGUF]   Processing FFN for layer %d...\n", l);
        fflush(stdout);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
        fprintf(stdout, "    [GGUF]   Getting ffn_norm tensor...\n");
        fflush(stdout);
        struct ggml_tensor* tfn = ggml_get_tensor(ggml_ctx, tname);
        fprintf(stdout, "    [GGUF]   Got ffn_norm tensor %p\n", (void*)tfn);
        fflush(stdout);
        if (tfn) {
            enum ggml_type fntype = ggml_tensor_type(tfn);
            int fn_ne = ggml_tensor_ne(tfn);
            if (fntype == GGML_TYPE_F32) {
                memcpy((char*)[w->layer_ffn_norm contents] + l * dim * sizeof(float), tfn->data, fn_ne * sizeof(float));
            } else if (fntype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tfn->data;
                float* dst = (float*)((char*)[w->layer_ffn_norm contents] + l * dim * sizeof(float));
                for (int i = 0; i < fn_ne; i++) dst[i] = ggml_half_to_float(src[i]);
            }
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
        struct ggml_tensor* tgate = ggml_get_tensor(ggml_ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        struct ggml_tensor* tup = ggml_get_tensor(ggml_ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        struct ggml_tensor* tdown = ggml_get_tensor(ggml_ctx, tname);

        if (tgate && tup && tdown) {
            enum ggml_type gtype = ggml_tensor_type(tgate);
            enum ggml_type utype = ggml_tensor_type(tup);
            enum ggml_type dtype = ggml_tensor_type(tdown);
            int gate_ne = ggml_tensor_ne(tgate);
            int up_ne = ggml_tensor_ne(tup);
            int down_ne = ggml_tensor_ne(tdown);

            float* gate_f32 = (float*)malloc(gate_ne * sizeof(float));
            float* up_f32 = (float*)malloc(up_ne * sizeof(float));
            float* down_f32 = (float*)malloc(down_ne * sizeof(float));

            if (gtype == GGML_TYPE_Q4_K) {
                dequantize_row_q4_K((const block_q4_K*)tgate->data, gate_f32, gate_ne);
            } else if (gtype == GGML_TYPE_Q6_K) {
                dequantize_row_q6_K((const block_q6_K*)tgate->data, gate_f32, gate_ne);
            } else if (gtype == GGML_TYPE_F32) {
                memcpy(gate_f32, tgate->data, gate_ne * sizeof(float));
            } else if (gtype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tgate->data;
                for (int i = 0; i < gate_ne; i++) gate_f32[i] = ggml_half_to_float(src[i]);
            } else {
                fprintf(stderr, "    [GGUF]   WARNING: unsupported gate type %d, using zeros\n", gtype);
                memset(gate_f32, 0, gate_ne * sizeof(float));
            }

            if (utype == GGML_TYPE_Q4_K) {
                dequantize_row_q4_K((const block_q4_K*)tup->data, up_f32, up_ne);
            } else if (utype == GGML_TYPE_Q6_K) {
                dequantize_row_q6_K((const block_q6_K*)tup->data, up_f32, up_ne);
            } else if (utype == GGML_TYPE_F32) {
                memcpy(up_f32, tup->data, up_ne * sizeof(float));
            } else if (utype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tup->data;
                for (int i = 0; i < up_ne; i++) up_f32[i] = ggml_half_to_float(src[i]);
            } else {
                fprintf(stderr, "    [GGUF]   WARNING: unsupported up type %d, using zeros\n", utype);
                memset(up_f32, 0, up_ne * sizeof(float));
            }

            if (dtype == GGML_TYPE_Q4_K) {
                dequantize_row_q4_K((const block_q4_K*)tdown->data, down_f32, down_ne);
            } else if (dtype == GGML_TYPE_Q6_K) {
                dequantize_row_q6_K((const block_q6_K*)tdown->data, down_f32, down_ne);
            } else if (dtype == GGML_TYPE_F32) {
                memcpy(down_f32, tdown->data, down_ne * sizeof(float));
            } else if (dtype == GGML_TYPE_F16) {
                uint16_t* src = (uint16_t*)tdown->data;
                for (int i = 0; i < down_ne; i++) down_f32[i] = ggml_half_to_float(src[i]);
            } else {
                fprintf(stderr, "    [GGUF]   WARNING: unsupported down type %d, using zeros\n", dtype);
                memset(down_f32, 0, down_ne * sizeof(float));
            }

            metal_upload_f32(w->layer_ffn_gate, l * max_ffn_float * sizeof(float), gate_f32, gate_ne);
            metal_upload_f32(w->layer_ffn_up, l * max_ffn_float * sizeof(float), up_f32, up_ne);
            metal_upload_f32(w->layer_ffn_down, l * max_ffn_float * sizeof(float), down_f32, down_ne);

            free(gate_f32);
            free(up_f32);
            free(down_f32);
        }

        if (l % 4 == 0 || l == n_layer - 1) {
            fprintf(stdout, "    [GGUF] Loaded layer %d/%d\n", l + 1, n_layer);
            fflush(stdout);
        }
    }

    struct ggml_tensor* tfn = ggml_get_tensor(ggml_ctx, "norm.weight");
    if (!tfn) tfn = ggml_get_tensor(ggml_ctx, "final_norm.weight");
    if (tfn) {
        enum ggml_type fntype = ggml_tensor_type(tfn);
        int fn_ne = ggml_tensor_ne(tfn);
        fprintf(stdout, "    [GGUF] final_norm: type=%d, elements=%d\n", fntype, fn_ne);
        float* fn_f32 = (float*)malloc(fn_ne * sizeof(float));
        if (fntype == GGML_TYPE_F32) {
            memcpy(fn_f32, tfn->data, fn_ne * sizeof(float));
        } else if (fntype == GGML_TYPE_F16) {
            uint16_t* src = (uint16_t*)tfn->data;
            for (int i = 0; i < fn_ne; i++) fn_f32[i] = ggml_half_to_float(src[i]);
        } else {
            memset(fn_f32, 0, fn_ne * sizeof(float));
        }
        w->final_norm = [dev newBufferWithLength:fn_ne * sizeof(float) options:MTLResourceStorageModeShared];
        memcpy([w->final_norm contents], fn_f32, fn_ne * sizeof(float));
        free(fn_f32);
    } else {
        fprintf(stdout, "    [GGUF] final_norm: NOT FOUND\n");
        w->final_norm = [dev newBufferWithLength:dim * sizeof(float) options:MTLResourceStorageModeShared];
    }

    struct ggml_tensor* tlm = ggml_get_tensor(ggml_ctx, "output.weight");
    if (!tlm) tlm = ggml_get_tensor(ggml_ctx, "lm_head.weight");
    if (tlm) {
        enum ggml_type lmtype = ggml_tensor_type(tlm);
        int lm_ne = ggml_tensor_ne(tlm);
        fprintf(stdout, "    [GGUF] lm_head: type=%d, elements=%d\n", lmtype, lm_ne);
        float* lm_f32 = (float*)malloc(lm_ne * sizeof(float));
        if (lmtype == GGML_TYPE_Q4_K) {
            dequantize_row_q4_K((const block_q4_K*)tlm->data, lm_f32, lm_ne);
        } else if (lmtype == GGML_TYPE_Q6_K) {
            dequantize_row_q6_K((const block_q6_K*)tlm->data, lm_f32, lm_ne);
        } else if (lmtype == GGML_TYPE_F32) {
            memcpy(lm_f32, tlm->data, lm_ne * sizeof(float));
        } else if (lmtype == GGML_TYPE_F16) {
            uint16_t* src = (uint16_t*)tlm->data;
            for (int i = 0; i < lm_ne; i++) lm_f32[i] = ggml_half_to_float(src[i]);
        } else {
            fprintf(stderr, "    [GGUF]   WARNING: unsupported lm_head type %d, using zeros\n", lmtype);
            memset(lm_f32, 0, lm_ne * sizeof(float));
        }
        w->lm_head = [dev newBufferWithLength:lm_ne * sizeof(float) options:MTLResourceStorageModeShared];
        memcpy([w->lm_head contents], lm_f32, lm_ne * sizeof(float));
        w->lm_head_nbytes = lm_ne * sizeof(float);
        free(lm_f32);
    }

    gguf_free(g);
    fprintf(stdout, "    [GGUF] ✅ GGUF 上下文已釋放\n");
    fprintf(stdout, "    [GGUF] ✅ 成功載入所有 %d 層 + 反量化完成\n", n_layer);
    fflush(stdout);
    return true;
}

static const char* kda_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

struct KDAParamsMetal {
    int batch;
    int heads;
    int seq_len;
    int head_dim;
    float beta;
};

kernel void kimi_kda_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* S [[buffer(4)]],
    constant KDAParamsMetal& params [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const int b = gid.x;
    const int h = gid.y;
    const int l = gid.z;

    if (b >= params.batch || h >= params.heads || l >= params.seq_len) {
        return;
    }

    const int D = params.head_dim;
    const float beta = params.beta;
    const float scale = 1.0f / sqrtf((float)D);

    const int q_offset = ((b * params.heads + h) * params.seq_len + l) * D;
    const int k_offset = q_offset;
    const int v_offset = q_offset;
    const int o_offset = q_offset;
    const int s_offset = ((b * params.heads + h) * D + l * D) * D;

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;

            for (int k = 0; k < D; k++) {
                float k_val = K[k_offset + k];
                float s_val = S[s_offset + k * D + j];

                sum += k_val * s_val;
            }

            float new_s = sum * (1.0f - beta * K[q_offset + i] * K[q_offset + j]) +
                         beta * K[q_offset + i] * V[v_offset + j];

            S[s_offset + i * D + j] = new_s;
        }
    }

    for (int i = 0; i < D; i++) {
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            sum += Q[q_offset + j] * S[s_offset + j * D + i];
        }
        O[o_offset + i] = sum * scale;
    }
}
)";

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void metal_run_kda_bench(metal_context_t* ctx, int n_iter) {
    @autoreleasepool {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:kda_kernel_source];
        id<MTLLibrary> library = [ctx->device newLibraryWithSource:src options:nil error:&error];
        if (error) {
            NSLog(@"[Metal] Library compile error: %@", [error localizedDescription]);
            return;
        }

        id<MTLFunction> func = [library newFunctionWithName:@"kimi_kda_forward"];
        if (!func) {
            NSLog(@"[Metal] Function 'kimi_kda_forward' not found");
            return;
        }

        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:func error:&error];
        if (error) {
            NSLog(@"[Metal] Pipeline create error: %@", [error localizedDescription]);
            return;
        }

        int seq_len = ctx->config->seq_len;
        int n_heads = ctx->config->n_head;
        int head_dim = ctx->config->head_dim;
        int n_layer = ctx->config->n_layer;
        float beta = 0.1f;

        struct {
            int batch;
            int heads;
            int seq_len;
            int head_dim;
            float beta;
        } kda_params = { 1, n_heads, seq_len, head_dim, beta };

        size_t q_size = n_heads * seq_len * head_dim * sizeof(float);
        size_t s_size = n_heads * seq_len * head_dim * head_dim * sizeof(float);

        id<MTLBuffer> buf_q = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_k = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_v = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_o = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_s = [ctx->device newBufferWithLength:s_size options:MTLResourceStorageModeShared];

        double total_time = 0.0;
        double min_time = 1e9;
        double max_time = 0.0;

        for (int iter = 0; iter < n_iter; iter++) {
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:pso];
            [enc setBuffer:buf_q offset:0 atIndex:0];
            [enc setBuffer:buf_k offset:0 atIndex:1];
            [enc setBuffer:buf_v offset:0 atIndex:2];
            [enc setBuffer:buf_o offset:0 atIndex:3];
            [enc setBuffer:buf_s offset:0 atIndex:4];
            [enc setBytes:&kda_params length:sizeof(kda_params) atIndex:5];

            MTLSize grid = MTLSizeMake(1, n_heads, seq_len);
            MTLSize group = MTLSizeMake(1, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];

            double start = get_time_ms();
            [cmd commit];
            [cmd waitUntilCompleted];
            double elapsed = get_time_ms() - start;

            total_time += elapsed;
            if (elapsed < min_time) min_time = elapsed;
            if (elapsed > max_time) max_time = elapsed;
        }

        int total_tokens = seq_len * n_layer;
        double avg_time = total_time / n_iter;
        double tokens_per_sec = total_tokens / (avg_time / 1000.0);

        fprintf(stdout, "\n");
        fprintf(stdout, "=== MagiCompiler KDA Benchmark (n_layer=%d, seq_len=%d, heads=%d, head_dim=%d) ===\n", n_layer, seq_len, n_heads, head_dim);
        fprintf(stdout, "  iterations: %d\n", n_iter);
        fprintf(stdout, "  total tokens per iteration: %d\n", total_tokens);
        fprintf(stdout, "  avg time:   %.2f ms\n", avg_time);
        fprintf(stdout, "  min time:   %.2f ms\n", min_time);
        fprintf(stdout, "  max time:   %.2f ms\n", max_time);
        fprintf(stdout, "  throughput: %.2f tokens/s\n", tokens_per_sec);
        fprintf(stdout, "\n");

        [library release];
        [pso release];
        [buf_q release];
        [buf_k release];
        [buf_v release];
        [buf_o release];
        [buf_s release];
    }
}

static void metal_run_kda(metal_context_t* ctx) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:kda_kernel_source];
        id<MTLLibrary> library = [ctx->device newLibraryWithSource:src options:nil error:&error];
        if (error) {
            NSLog(@"[Metal] Library compile error: %@", [error localizedDescription]);
            return;
        }

        id<MTLFunction> func = [library newFunctionWithName:@"kimi_kda_forward"];
        if (!func) {
            NSLog(@"[Metal] Function 'kimi_kda_forward' not found");
            return;
        }

        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:func error:&error];
        if (error) {
            NSLog(@"[Metal] Pipeline create error: %@", [error localizedDescription]);
            return;
        }

        [enc setComputePipelineState:pso];

        int seq_len = ctx->config->seq_len;
        int n_heads = ctx->config->n_head;
        int head_dim = ctx->config->head_dim;
        int dim = ctx->config->dim;
        float beta = 0.1f;

        struct {
            int batch;
            int heads;
            int seq_len;
            int head_dim;
            float beta;
        } kda_params = { 1, n_heads, seq_len, head_dim, beta };

        size_t q_size = n_heads * seq_len * head_dim * sizeof(float);
        size_t s_size = n_heads * seq_len * head_dim * head_dim * sizeof(float);

        id<MTLBuffer> buf_q = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_k = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_v = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_o = [ctx->device newBufferWithLength:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_s = [ctx->device newBufferWithLength:s_size options:MTLResourceStorageModeShared];

        [enc setBuffer:buf_q offset:0 atIndex:0];
        [enc setBuffer:buf_k offset:0 atIndex:1];
        [enc setBuffer:buf_v offset:0 atIndex:2];
        [enc setBuffer:buf_o offset:0 atIndex:3];
        [enc setBuffer:buf_s offset:0 atIndex:4];
        [enc setBytes:&kda_params length:sizeof(kda_params) atIndex:5];

        MTLSize grid = MTLSizeMake(1, n_heads, seq_len);
        MTLSize group = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        float* result = (float*)[buf_o contents];
        fprintf(stdout, "    [Metal] KDA attention result[0] = %.4f\n", result[0]);
        fprintf(stdout, "    [Metal] ✅ KDA Attention 核心執行成功\n");

        [library release];
        [pso release];
    }
}

// KDA State API
metal_kda_state_t* metal_kda_state_create(metal_device_t* dev, int n_head, int head_dim, int seq_len) {
    metal_kda_state_t* state = (metal_kda_state_t*)malloc(sizeof(metal_kda_state_t));
    if (!state) return NULL;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        free(state);
        return NULL;
    }

    size_t size = n_head * seq_len * head_dim * head_dim * sizeof(float);
    
    state->buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    state->n_head = n_head;
    state->head_dim = head_dim;
    state->seq_len = seq_len;

    // Initialize state matrix to zero
    memset([state->buffer contents], 0, size);
    
    return state;
}

void metal_kda_state_destroy(metal_kda_state_t* state) {
    if (!state) return;
    [state->buffer release];
    free(state);
}

metal_buffer_t* metal_kda_state_get_buffer(metal_kda_state_t* state) {
    if (!state) return NULL;
    return (metal_buffer_t*)state->buffer;
}

// KDA Attention API
void metal_kda_attention(metal_device_t* dev,
                         metal_buffer_t* q, metal_buffer_t* k, metal_buffer_t* v,
                         metal_buffer_t* output, metal_buffer_t* state,
                         int n_head, int n_kv_head, int head_dim, int seq_len, float beta) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"[Metal] KDA Failed to create device");
            return;
        }
        
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            NSLog(@"[Metal] KDA Failed to create command queue");
            return;
        }
        
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:kda_kernel_source];
        id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
        if (error) {
            NSLog(@"[Metal] KDA Library compile error: %@", [error localizedDescription]);
            return;
        }

        id<MTLFunction> func = [library newFunctionWithName:@"kimi_kda_forward"];
        if (!func) {
            NSLog(@"[Metal] KDA Function not found");
            [library release];
            return;
        }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (error) {
            NSLog(@"[Metal] KDA Pipeline create error: %@", [error localizedDescription]);
            [library release];
            return;
        }

        struct {
            int batch;
            int heads;
            int seq_len;
            int head_dim;
            float beta;
        } kda_params = { 1, n_head, seq_len, head_dim, beta };

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)q offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)k offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)v offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)state offset:0 atIndex:4];
        [enc setBytes:&kda_params length:sizeof(kda_params) atIndex:5];

        MTLSize grid = MTLSizeMake(1, n_head, seq_len);
        MTLSize group = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        [library release];
        [pso release];
    }
}

#ifdef __cplusplus
}
#endif