// cgc_arch_info.c — 读 config.json,提取架构信息 (C 实现)
//
// 移植自 turbo-fieldfare 的 ArchInfo.swift
// Gemma 4 MoE config.json 结构:
//   {
//     "text_config": {
//       "hidden_size": ...,
//       "moe_intermediate_size": ...,
//       "num_experts": ...,
//       "top_k_experts": ...,
//       "num_hidden_layers": ...,
//       ...
//     }
//   }

#include "cgc_repack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 简易 JSON 路径访问
// ============================================================================

typedef struct {
    char* buf;
    size_t len;
} json_doc_t;

static json_doc_t json_load(const char* path) {
    json_doc_t doc = { NULL, 0 };
    FILE* f = fopen(path, "rb");
    if (!f) return doc;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    doc.buf = (char*)malloc(sz + 1);
    if (!doc.buf) { fclose(f); return doc; }
    doc.len = fread(doc.buf, 1, sz, f);
    doc.buf[doc.len] = '\0';
    fclose(f);
    return doc;
}

// 在 json 中查找 "key": value,返回 value 的起始位置
static const char* json_find_key(const char* json, size_t len, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    size_t plen = strlen(pattern);
    for (size_t i = 0; i + plen < len; ++i) {
        if (strncmp(json + i, pattern, plen) == 0) {
            size_t j = i + plen;
            while (j < len && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n')) j++;
            if (j < len && json[j] == ':') {
                j++;
                while (j < len && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n')) j++;
                return json + j;
            }
        }
    }
    return NULL;
}

static bool json_read_number(const char* p, size_t max_len, double* out) {
    size_t i = 0;
    while (i < max_len && p[i] == ' ') i++;
    size_t start = i;
    while (i < max_len) {
        char c = p[i];
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') i++;
        else break;
    }
    if (i == start) return false;
    char buf[64];
    size_t n = i - start;
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, p + start, n);
    buf[n] = '\0';
    *out = strtod(buf, NULL);
    return true;
}

static bool json_read_int(const char* p, size_t max_len, int* out) {
    double d;
    if (!json_read_number(p, max_len, &d)) return false;
    *out = (int)d;
    return true;
}

static bool json_read_string(const char* p, size_t max_len, char* out, size_t out_size) {
    size_t i = 0;
    while (i < max_len && p[i] == ' ') i++;
    if (i >= max_len || p[i] != '"') return false;
    i++;
    size_t j = 0;
    while (i < max_len && p[i] != '"') {
        if (j + 1 < out_size) out[j++] = p[i];
        i++;
    }
    out[j] = '\0';
    return true;
}

static bool json_read_bool(const char* p, size_t max_len, bool* out) {
    if (max_len >= 4 && strncmp(p, "true", 4) == 0) { *out = true; return true; }
    if (max_len >= 5 && strncmp(p, "false", 5) == 0) { *out = false; return true; }
    return false;
}

// 辅助宏:在 text_config 范围内读 int
#define TC_READ_INT(key, out) do { \
    const char* v = json_find_key(tc, tc_len, key); \
    if (v) json_read_int(v, tc_len - (size_t)(v - tc), out); \
} while(0)

#define TC_READ_DOUBLE(key, out) do { \
    const char* v = json_find_key(tc, tc_len, key); \
    if (v) json_read_number(v, tc_len - (size_t)(v - tc), out); \
} while(0)

#define TC_READ_STR(key, out, n) do { \
    const char* v = json_find_key(tc, tc_len, key); \
    if (v) json_read_string(v, tc_len - (size_t)(v - tc), out, n); \
} while(0)

#define TC_READ_BOOL(key, out) do { \
    const char* v = json_find_key(tc, tc_len, key); \
    if (v) json_read_bool(v, tc_len - (size_t)(v - tc), out); \
} while(0)

// ============================================================================
// cgc_arch_info_load
// ============================================================================

int cgc_arch_info_load(const char* config_path, cgc_arch_info_t* arch) {
    memset(arch, 0, sizeof(*arch));
    arch->rope_theta = 10000.0;
    arch->full_rope_theta = 1000000.0;
    arch->partial_rotary_factor = 0.25;
    arch->final_logit_softcap = 0.0;
    strcpy(arch->hidden_activation, "gelu_pytorch_tanh");

    json_doc_t doc = json_load(config_path);
    if (!doc.buf) {
        fprintf(stderr, "[arch_info] cannot load %s\n", config_path);
        return -1;
    }

    // 找 text_config
    const char* tc = json_find_key(doc.buf, doc.len, "text_config");
    if (!tc) tc = doc.buf;
    size_t tc_len = doc.len - (size_t)(tc - doc.buf);

    TC_READ_INT("hidden_size",         &arch->hidden_size);
    TC_READ_INT("intermediate_size",   &arch->intermediate_size);
    TC_READ_INT("moe_intermediate_size", &arch->moe_intermediate_size);
    TC_READ_INT("num_attention_heads", &arch->num_heads);
    TC_READ_INT("num_key_value_heads", &arch->num_kv_heads);
    TC_READ_INT("num_global_key_value_heads", &arch->num_full_kv_heads);
    TC_READ_INT("head_dim",            &arch->head_dim);
    TC_READ_INT("global_head_dim",     &arch->full_head_dim);
    TC_READ_INT("vocab_size",          &arch->vocab_size);
    TC_READ_INT("sliding_window",      &arch->sliding_window);
    TC_READ_INT("num_hidden_layers",   &arch->num_layers);
    TC_READ_INT("num_experts",         &arch->num_experts);
    TC_READ_INT("top_k_experts",       &arch->top_k_experts);

    TC_READ_DOUBLE("final_logit_softcapping", &arch->final_logit_softcap);
    TC_READ_DOUBLE("rope_theta",               &arch->rope_theta);
    TC_READ_STR("hidden_activation", arch->hidden_activation, sizeof(arch->hidden_activation));

    // rope_parameters 嵌套
    const char* rope = json_find_key(tc, tc_len, "rope_parameters");
    if (rope) {
        size_t rope_len = tc_len - (size_t)(rope - tc);
        const char* full = json_find_key(rope, rope_len, "full_attention");
        if (full) {
            size_t full_len = rope_len - (size_t)(full - rope);
            const char* v;
            double d;
            v = json_find_key(full, full_len, "rope_theta");
            if (v && json_read_number(v, full_len - (size_t)(v - full), &d)) arch->full_rope_theta = d;
            v = json_find_key(full, full_len, "partial_rotary_factor");
            if (v && json_read_number(v, full_len - (size_t)(v - full), &d)) arch->partial_rotary_factor = d;
        }
        const char* swa = json_find_key(rope, rope_len, "sliding_attention");
        if (swa) {
            size_t swa_len = rope_len - (size_t)(swa - rope);
            const char* v = json_find_key(swa, swa_len, "rope_theta");
            double d;
            if (v && json_read_number(v, swa_len - (size_t)(v - swa), &d)) arch->rope_theta = d;
        }
    }

    TC_READ_BOOL("tie_word_embeddings", &arch->tie_word_embeddings);
    TC_READ_BOOL("attention_k_eq_v",    &arch->attention_k_eq_v);

    free(doc.buf);

    fprintf(stderr, "[arch_info] %s\n", config_path);
    fprintf(stderr, "[arch_info]   hidden=%d moe_int=%d layers=%d experts=%d topk=%d\n",
            arch->hidden_size, arch->moe_intermediate_size,
            arch->num_layers, arch->num_experts, arch->top_k_experts);
    fprintf(stderr, "[arch_info]   heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
            arch->num_heads, arch->num_kv_heads, arch->head_dim, arch->vocab_size);

    return 0;
}

// ============================================================================
// tensor 分类 (移植 RepackPlanner.classify)
// ============================================================================

cgc_bucket_t cgc_classify_tensor(const char* name, int num_layers) {
    cgc_bucket_t b;
    memset(&b, 0, sizeof(b));
    b.kind = CGC_BUCKET_UNKNOWN;
    b.layer = -1;

    // turbo-fieldfare: "language_model.layers.<N>.experts.switch_glu.{gate,up,down}_proj.weight"
    if (strstr(name, ".experts.switch_glu.") != NULL ||
        strstr(name, ".experts.glu.") != NULL ||
        strstr(name, ".block_sparse_moe.experts.") != NULL) {
        const char* p = strstr(name, ".layers.");
        if (!p) p = strstr(name, ".blocks.");
        if (p) {
            p = strchr(p + 1, '.') + 1; // skip "layers" / "blocks"
            int layer = 0;
            while (*p >= '0' && *p <= '9') {
                layer = layer * 10 + (*p - '0');
                p++;
            }
            if (layer >= 0 && layer < num_layers) {
                b.kind = CGC_BUCKET_ROUTED_EXPERT;
                b.layer = layer;
                if (strstr(name, ".gate_proj.")) strcpy(b.role, "gate");
                else if (strstr(name, ".up_proj.")) strcpy(b.role, "up");
                else if (strstr(name, ".down_proj.")) strcpy(b.role, "down");
                else if (strstr(name, ".w1.")) strcpy(b.role, "gate");
                else if (strstr(name, ".w3.")) strcpy(b.role, "up");
                else if (strstr(name, ".w2.")) strcpy(b.role, "down");
                return b;
            }
        }
    }

    if (strstr(name, "language_model.") != NULL || strstr(name, "model.") != NULL) {
        b.kind = CGC_BUCKET_LM_RESIDENT;
        return b;
    }

    if (strstr(name, "vision_") != NULL || strstr(name, "multi_modal_") != NULL) {
        b.kind = CGC_BUCKET_EXCLUDED_MULTIMODAL;
        return b;
    }

    return b;
}
