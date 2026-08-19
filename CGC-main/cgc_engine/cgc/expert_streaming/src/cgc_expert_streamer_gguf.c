#include "cgc_expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define CGC_MAX_EXPERTS_PER_LAYER 256

typedef enum {
    CGC_LAYOUT_PER_EXPERT = 0,
    CGC_LAYOUT_PER_LAYER = 1,
} cgc_layout_type_t;

typedef struct {
    cgc_layout_type_t layout_type;

    int layer_index;
    int expert_count;
    int experts_per_layer;
    int active_experts;

    int hidden_size;
    int moe_intermediate_size;

    char ffn_down_exps_name[CGC_MAX_NAME_LEN];
    char ffn_gate_up_exps_name[CGC_MAX_NAME_LEN];
    char ffn_down_exps_scale_name[CGC_MAX_NAME_LEN];

    uint64_t ffn_down_exps_offset;
    uint64_t ffn_gate_up_exps_offset;
    uint64_t ffn_down_exps_scale_offset;

    int32_t ffn_down_exps_type;
    int32_t ffn_gate_up_exps_type;
    int32_t ffn_down_exps_scale_type;

    int64_t ffn_down_dims[4];
    int64_t ffn_gate_up_dims[4];

    uint64_t ffn_down_exps_size;
    uint64_t ffn_gate_up_exps_size;

    // llama.cpp qwen35moe: gate/up 是分開的張量（ffn_gate_exps / ffn_up_exps），
    // 各 [inter, hidden, n_experts] packed。記 offset/type/size 以便組 blob。
    uint64_t ffn_gate_exps_offset;
    uint64_t ffn_up_exps_offset;
    int32_t ffn_gate_exps_type;
    int32_t ffn_up_exps_type;
    uint64_t ffn_gate_exps_size;
    uint64_t ffn_up_exps_size;
} cgc_layer_expert_info_t;

static bool parse_per_expert_name(const char* name, int* out_layer, int* out_expert, char* out_role) {
    if (!name || !out_layer || !out_expert || !out_role) return false;

    int layer = -1, expert = -1;
    char role[CGC_MAX_NAME_LEN] = {0};
    int n = sscanf(name, "blk.%d.expert.%d.%255[^.].weight", &layer, &expert, role);
    if (n != 3) return false;

    *out_layer = layer;
    *out_expert = expert;
    strncpy(out_role, role, CGC_MAX_NAME_LEN - 1);
    out_role[CGC_MAX_NAME_LEN - 1] = '\0';
    return true;
}

// Architecture-aware KV lookup: try qwen35moe.* (llama.cpp Qwen3.5/3.6 MoE),
// then gemma4.* (legacy cgc repack format). Values may be UINT32 or INT32.
static bool get_arch_u32(const cgc_gguf_lite_ctx_t* ctx, const char* suffix, uint32_t* out) {
    if (!ctx || !suffix || !out) return false;
    char key[128];
    snprintf(key, sizeof(key), "qwen35moe.%s", suffix);
    if (cgc_gguf_lite_get_u32(ctx, key, out)) return true;
    snprintf(key, sizeof(key), "gemma4.%s", suffix);
    if (cgc_gguf_lite_get_u32(ctx, key, out)) return true;
    snprintf(key, sizeof(key), "qwen35moe.%s", suffix);
    int32_t v32 = 0;
    if (cgc_gguf_lite_get_i32(ctx, key, &v32)) { *out = (uint32_t)v32; return true; }
    snprintf(key, sizeof(key), "gemma4.%s", suffix);
    if (cgc_gguf_lite_get_i32(ctx, key, &v32)) { *out = (uint32_t)v32; return true; }
    return false;
}

static bool parse_per_layer_name(const char* name, int* out_layer, char* out_exps_type) {
    if (!name || !out_layer || !out_exps_type) return false;

    int layer = -1;
    char exps_type[CGC_MAX_NAME_LEN] = {0};

    if (strstr(name, "_exps") == NULL) return false;

    if (strncmp(name, "blk.", 4) != 0) return false;

    const char* p = name + 4;
    layer = atoi(p);

    const char* dot = strchr(p, '.');
    if (!dot) return false;

    if (strstr(name, "ffn_down_exps")) {
        strncpy(exps_type, "ffn_down", CGC_MAX_NAME_LEN - 1);
    } else if (strstr(name, "ffn_gate_up_exps")) {
        strncpy(exps_type, "ffn_gate_up", CGC_MAX_NAME_LEN - 1);
    } else if (strstr(name, "ffn_gate_exps")) {
        strncpy(exps_type, "ffn_gate", CGC_MAX_NAME_LEN - 1);
    } else if (strstr(name, "ffn_up_exps")) {
        strncpy(exps_type, "ffn_up", CGC_MAX_NAME_LEN - 1);
    } else if (strstr(name, "ffn_gate_inp_exps")) {
        strncpy(exps_type, "ffn_gate_inp", CGC_MAX_NAME_LEN - 1);
    } else {
        return false;
    }

    *out_layer = layer;
    strncpy(out_exps_type, exps_type, CGC_MAX_NAME_LEN - 1);
    out_exps_type[CGC_MAX_NAME_LEN - 1] = '\0';
    return true;
}

static int count_expert_tensors(const cgc_gguf_lite_ctx_t* ctx, cgc_layout_type_t* out_type) {
    int per_expert_count = 0;
    int per_layer_count = 0;

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        int layer, expert;
        char role[CGC_MAX_NAME_LEN];
        if (parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) {
            per_expert_count++;
        }

        int layer2;
        char exps_type[CGC_MAX_NAME_LEN];
        if (parse_per_layer_name(ctx->tensor_names[i], &layer2, exps_type)) {
            per_layer_count++;
        }
    }

    if (per_expert_count > 0 && per_layer_count == 0) {
        *out_type = CGC_LAYOUT_PER_EXPERT;
        return per_expert_count;
    } else if (per_layer_count > 0 && per_expert_count == 0) {
        *out_type = CGC_LAYOUT_PER_LAYER;
        return per_layer_count;
    } else if (per_layer_count > per_expert_count) {
        *out_type = CGC_LAYOUT_PER_LAYER;
        return per_layer_count;
    } else {
        *out_type = CGC_LAYOUT_PER_EXPERT;
        return per_expert_count;
    }
}

static cgc_layer_expert_info_t* find_all_layers(const cgc_gguf_lite_ctx_t* ctx,
                                                  int* out_layer_count,
                                                  int* out_experts_per_layer) {
    cgc_layout_type_t layout_type;
    int total_count = count_expert_tensors(ctx, &layout_type);

    if (total_count == 0) {
        *out_layer_count = 0;
        *out_experts_per_layer = 0;
        return NULL;
    }

    int max_layer = 0;
    int max_expert = 0;

    if (layout_type == CGC_LAYOUT_PER_EXPERT) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer, expert;
            char role[CGC_MAX_NAME_LEN];
            if (parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) {
                if (layer > max_layer) max_layer = layer;
                if (expert > max_expert) max_expert = expert;
            }
        }
    } else {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer;
            char exps_type[CGC_MAX_NAME_LEN];
            if (parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) {
                if (layer > max_layer) max_layer = layer;
            }
        }

        uint32_t expert_count_u32 = 0;
        if (get_arch_u32(ctx, "expert_count", &expert_count_u32)) {
            max_expert = (int)expert_count_u32;
        } else {
            // Per-layer packed tensors are shaped [out, in, num_experts]; the
            // expert count is the LAST dim, not the max dim (that would pick the
            // intermediate width and overcount by ~8x).
            for (uint64_t i = 0; i < ctx->n_tensors; i++) {
                int layer;
                char exps_type[CGC_MAX_NAME_LEN];
                if (parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) {
                    if (ctx->tensors[i].n_dims >= 3) {
                        int64_t ne = ctx->tensors[i].dims[ctx->tensors[i].n_dims - 1];
                        if (ne > max_expert) max_expert = (int)ne;
                    }
                }
            }
        }
    }

    int layer_count = max_layer + 1;
    // PER_EXPERT layout: max_expert is the highest expert *index* → count = max+1.
    // PER_LAYER layout: max_expert already holds the *count* from metadata or
    // the last tensor dim — do not add 1.
    int experts_per_layer = (layout_type == CGC_LAYOUT_PER_EXPERT) ? max_expert + 1 : max_expert;

    *out_layer_count = layer_count;
    *out_experts_per_layer = experts_per_layer;

    cgc_layer_expert_info_t* layers = (cgc_layer_expert_info_t*)calloc(layer_count, sizeof(cgc_layer_expert_info_t));
    if (!layers) return NULL;

    for (int l = 0; l < layer_count; l++) {
        layers[l].layout_type = layout_type;
        layers[l].layer_index = l;
        layers[l].expert_count = experts_per_layer;
        layers[l].experts_per_layer = experts_per_layer;

        uint32_t active_experts = 0;
        if (get_arch_u32(ctx, "expert_used_count", &active_experts)) {
            layers[l].active_experts = (int)active_experts;
        } else {
            layers[l].active_experts = 8;
        }

        uint32_t hidden = 0;
        if (get_arch_u32(ctx, "embedding_length", &hidden)) {
            layers[l].hidden_size = (int)hidden;
        }

        uint32_t ffn_len = 0;
        if (get_arch_u32(ctx, "expert_feed_forward_length", &ffn_len)) {
            layers[l].moe_intermediate_size = (int)ffn_len;
        } else if (get_arch_u32(ctx, "feed_forward_length", &ffn_len)) {
            layers[l].moe_intermediate_size = (int)ffn_len;
        }
    }

    if (layout_type == CGC_LAYOUT_PER_LAYER) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer;
            char exps_type[CGC_MAX_NAME_LEN];
            if (!parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) continue;
            if (layer >= layer_count) continue;

            const char* tname = ctx->tensor_names[i];
            cgc_layer_expert_info_t* li = &layers[layer];
            cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];

            if (strstr(tname, "ffn_down_exps.scale") && li->ffn_down_exps_scale_name[0] == '\0') {
                strncpy(li->ffn_down_exps_scale_name, tname, CGC_MAX_NAME_LEN - 1);
                li->ffn_down_exps_scale_offset = ti->offset;
                li->ffn_down_exps_scale_type = ti->type;
            } else if (strstr(tname, "ffn_down_exps.weight") && li->ffn_down_exps_name[0] == '\0') {
                strncpy(li->ffn_down_exps_name, tname, CGC_MAX_NAME_LEN - 1);
                li->ffn_down_exps_offset = ti->offset;
                li->ffn_down_exps_type = ti->type;
                memcpy(li->ffn_down_dims, ti->dims, sizeof(int64_t) * ti->n_dims);

                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_down_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            } else if (strstr(tname, "ffn_gate_up_exps.weight") && li->ffn_gate_up_exps_name[0] == '\0') {
                strncpy(li->ffn_gate_up_exps_name, tname, CGC_MAX_NAME_LEN - 1);
                li->ffn_gate_up_exps_offset = ti->offset;
                li->ffn_gate_up_exps_type = ti->type;
                memcpy(li->ffn_gate_up_dims, ti->dims, sizeof(int64_t) * ti->n_dims);

                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_gate_up_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            } else if (strstr(tname, "ffn_gate_exps.weight") && li->ffn_gate_exps_offset == 0) {
                li->ffn_gate_exps_offset = ti->offset;
                li->ffn_gate_exps_type = ti->type;
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_gate_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            } else if (strstr(tname, "ffn_up_exps.weight") && li->ffn_up_exps_offset == 0) {
                li->ffn_up_exps_offset = ti->offset;
                li->ffn_up_exps_type = ti->type;
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_up_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            }
        }
    } else {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer, expert;
            char role[CGC_MAX_NAME_LEN];
            if (!parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) continue;
            if (layer >= layer_count) continue;

            cgc_layer_expert_info_t* li = &layers[layer];
            cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];

            if (strcmp(role, "ffn_down") == 0 && li->ffn_down_exps_name[0] == '\0') {
                strncpy(li->ffn_down_exps_name, ctx->tensor_names[i], CGC_MAX_NAME_LEN - 1);
                li->ffn_down_exps_offset = ti->offset;
                li->ffn_down_exps_type = ti->type;
                memcpy(li->ffn_down_dims, ti->dims, sizeof(int64_t) * ti->n_dims);

                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_down_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            } else if (strcmp(role, "ffn_gate_up") == 0 && li->ffn_gate_up_exps_name[0] == '\0') {
                strncpy(li->ffn_gate_up_exps_name, ctx->tensor_names[i], CGC_MAX_NAME_LEN - 1);
                li->ffn_gate_up_exps_offset = ti->offset;
                li->ffn_gate_up_exps_type = ti->type;
                memcpy(li->ffn_gate_up_dims, ti->dims, sizeof(int64_t) * ti->n_dims);

                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                li->ffn_gate_up_exps_size = (uint64_t)(bpe * (double)ti->n_elements);
            }
        }
    }

    return layers;
}

cgc_stream_layout_t cgc_load_stream_layout_from_gguf(const char* gguf_path) {
    cgc_stream_layout_t layout;
    memset(&layout, 0, sizeof(layout));

    if (!gguf_path) return layout;
    strncpy(layout.path, gguf_path, CGC_MAX_PATH_LEN - 1);

    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(gguf_path);
    if (!ctx) {
        fprintf(stderr, "[cgc_expert_streamer_gguf] failed to load GGUF: %s\n", gguf_path);
        return layout;
    }

    int layer_count = 0;
    int experts_per_layer = 0;
    cgc_layer_expert_info_t* layers = find_all_layers(ctx, &layer_count, &experts_per_layer);

    if (!layers || layer_count == 0) {
        fprintf(stderr, "[cgc_expert_streamer_gguf] no expert layers found in %s\n", gguf_path);
        cgc_gguf_lite_free(ctx);
        free(layers);
        return layout;
    }

    cgc_layout_type_t layout_type = layers[0].layout_type;

    // GGUF v3 的 tensor offset 是「相對於 data_start」的（見 cgc_gguf_lite.h:75），
    // pread/mmap 需要檔案絕對位置 = data_start + offset。
    // 所有放進 layout 的 offset（seg_base / stream_offset / expert_offsets）都必須加。
    const uint64_t DS = ctx->data_start;

    if (layout_type == CGC_LAYOUT_PER_LAYER) {
        // llama.cpp 佈局（qwen35moe / gemma4）:
        //   每層 gate/up/down 是分開的 packed 張量（各 [out, in, n_experts]），
        //   專家 e 的權重分散在 3 個張量內，不相鄰。
        // 對策: 組「專家 blob」虛擬佈局——stride = gate+up+down 各一專家 bytes 之和；
        //       每個 blob 起始 offset = 三張量各自 e*per-expert-size 的和（layer 0）。
        //       cgc 只使用 layer 0（decode 每步逐層 load），has_explicit_offsets 只對 layer 0 有效。
        uint64_t g0 = layers[0].ffn_gate_exps_size;
        uint64_t u0 = layers[0].ffn_up_exps_size;
        uint64_t d0 = layers[0].ffn_down_exps_size;
        uint64_t gu0 = layers[0].ffn_gate_up_exps_size;

        uint64_t per_expert_gate_up = 0;
        if (g0 > 0 && u0 > 0) {
            per_expert_gate_up = g0 / (uint64_t)experts_per_layer + u0 / (uint64_t)experts_per_layer;
        } else if (gu0 > 0) {
            per_expert_gate_up = gu0 / (uint64_t)experts_per_layer;
        }
        uint64_t per_expert_down = d0 > 0 ? d0 / (uint64_t)experts_per_layer : 0;

        layout.expert_stride = per_expert_gate_up + per_expert_down;
        layout.experts_per_layer = experts_per_layer;

        if (g0 > 0 && u0 > 0 && d0 > 0) {
            // llama.cpp qwen35moe 三張量佈局（ffn_gate_exps / ffn_up_exps / ffn_down_exps）:
            // 專家權重分散在三個非連續張量區段。用 segment 定址：
            //   seg_base  = 各張量資料起點（絕對位置 = data_start + raw），
            //   seg_size = 單 expert 各段 bytes。
            // 消費端（cgc_expert_segments）依 gate|up|down 順序拼進 slot buffer。
            // mmap 路徑不適用（三段非連續），streamer 建立時會強制走 pread。
            layout.has_segments = 1;
            layout.seg_base[0] = DS + layers[0].ffn_gate_exps_offset;
            layout.seg_base[1] = DS + layers[0].ffn_up_exps_offset;
            layout.seg_base[2] = DS + layers[0].ffn_down_exps_offset;
            layout.seg_size[0] = g0 / (uint64_t)experts_per_layer;
            layout.seg_size[1] = u0 / (uint64_t)experts_per_layer;
            layout.seg_size[2] = d0 / (uint64_t)experts_per_layer;
            layout.stream_offset = DS + layers[0].ffn_gate_exps_offset;
            layout.stream_size = d0 + g0 + u0;
            layout.has_explicit_offsets = 0;
        } else if (gu0 > 0 && d0 > 0) {
            // llama.cpp gemma4 合併佈局（ffn_gate_up_exps + ffn_down_exps）:
            // 專家權重分散在兩個非連續張量區段。segment 定址：
            //   seg_base[0] = gate_up 張量起點，seg_base[1] = down 張量起點，
            //   seg_base[2] 未使用（size=0，消費端會 skip）。
            layout.has_segments = 1;
            layout.seg_base[0] = DS + layers[0].ffn_gate_up_exps_offset;
            layout.seg_base[1] = DS + layers[0].ffn_down_exps_offset;
            layout.seg_base[2] = 0;
            layout.seg_size[0] = gu0 / (uint64_t)experts_per_layer;
            layout.seg_size[1] = d0 / (uint64_t)experts_per_layer;
            layout.seg_size[2] = 0;
            layout.stream_offset = DS + layers[0].ffn_gate_up_exps_offset;
            layout.stream_size = gu0 + d0;
            layout.has_explicit_offsets = 0;
        } else {
            // 單張量或未知: 退回層間 gap
            layout.stream_offset = DS + layers[0].ffn_down_exps_offset;
            layout.stream_size = d0;
            layout.has_explicit_offsets = 0;
            if (layer_count > 1) {
                // 兩者都是 raw，差值正確；但 stream_offset 已是絕對位置，不能直接相減
                layout.expert_stride = layers[1].ffn_down_exps_offset - layers[0].ffn_down_exps_offset;
            }
        }
    } else {
        uint64_t first_offset = layers[0].ffn_down_exps_offset;
        uint64_t second_offset = 0;

        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer, expert;
            char role[CGC_MAX_NAME_LEN];
            if (parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) {
                if (layer == 0 && expert == 1) {
                    second_offset = ctx->tensors[i].offset;
                    break;
                }
            }
        }

        layout.stream_offset = DS + first_offset;
        if (second_offset > first_offset) {
            layout.expert_stride = second_offset - first_offset;
            layout.stream_size = (uint64_t)experts_per_layer * layout.expert_stride;
        } else {
            uint64_t total_size = 0;
            for (int l = 0; l < layer_count; l++) {
                total_size += layers[l].ffn_down_exps_size;
                total_size += layers[l].ffn_gate_up_exps_size;
            }
            layout.stream_size = total_size;
            layout.expert_stride = total_size / (uint64_t)experts_per_layer;
        }
        layout.experts_per_layer = experts_per_layer;
    }

    for (int l = 0; l < layer_count; l++) {
        if (l < CGC_MAX_EXPERTS_PER_LAYER) {
            layout.expert_offsets[l] = DS + layers[l].ffn_down_exps_offset;
        }
    }
    layout.has_explicit_offsets = true;

    free(layers);
    cgc_gguf_lite_free(ctx);
    return layout;
}

int cgc_find_expert_tensors(const cgc_gguf_lite_ctx_t* ctx,
                             int expert_id,
                             cgc_expert_tensor_info_t* out,
                             int max_out) {
    if (!ctx || !out || max_out <= 0) return 0;

    cgc_layout_type_t layout_type;
    int total_count = count_expert_tensors(ctx, &layout_type);

    if (layout_type == CGC_LAYOUT_PER_EXPERT) {
        int count = 0;
        for (uint64_t i = 0; i < ctx->n_tensors && count < max_out; i++) {
            int layer, expert;
            char role[CGC_MAX_NAME_LEN];
            if (!parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) continue;
            if (expert != expert_id) continue;

            cgc_expert_tensor_info_t* info = &out[count];
            memset(info, 0, sizeof(*info));
            info->expert_id = expert_id;
            strncpy(info->role, role, CGC_MAX_NAME_LEN - 1);
            info->ggml_type = ctx->tensors[i].type;
            info->n_dims = ctx->tensors[i].n_dims;
            for (int j = 0; j < 4 && j < info->n_dims; j++) {
                info->dims[j] = ctx->tensors[i].dims[j];
            }
            info->offset = ctx->tensors[i].offset;
            double bpe = cgc_ggml_type_bytes_per_elem(ctx->tensors[i].type);
            info->size_bytes = (uint64_t)(bpe * (double)ctx->tensors[i].n_elements);
            count++;
        }
        return count;
    } else {
        int count = 0;
        for (uint64_t i = 0; i < ctx->n_tensors && count < max_out; i++) {
            int layer;
            char exps_type[CGC_MAX_NAME_LEN];
            if (!parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) continue;

            const char* tname = ctx->tensor_names[i];
            cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];

            if (strstr(tname, "ffn_down_exps.scale")) {
                continue;
            }

            cgc_expert_tensor_info_t* info = &out[count];
            memset(info, 0, sizeof(*info));
            info->expert_id = expert_id;

            if (strstr(tname, "ffn_down_exps.weight")) {
                strncpy(info->role, "ffn_down", CGC_MAX_NAME_LEN - 1);
            } else if (strstr(tname, "ffn_gate_up_exps.weight")) {
                strncpy(info->role, "ffn_gate_up", CGC_MAX_NAME_LEN - 1);
            } else {
                continue;
            }

            info->ggml_type = ti->type;
            info->n_dims = ti->n_dims;

            int expert_dims_idx = ti->n_dims - 1;
            for (int j = 0; j < ti->n_dims; j++) {
                if (j == expert_dims_idx) {
                    info->dims[j] = 1;
                } else {
                    info->dims[j] = ti->dims[j];
                }
            }
            if (info->n_dims < 4) {
                for (int j = info->n_dims; j < 4; j++) {
                    info->dims[j] = 0;
                }
            }

            int64_t stride_elems = 1;
            for (int j = 0; j < expert_dims_idx; j++) {
                stride_elems *= ti->dims[j];
            }

            double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
            uint64_t expert_size = (uint64_t)(bpe * (double)stride_elems);

            info->offset = ti->offset + (uint64_t)expert_id * expert_size;
            info->size_bytes = expert_size;

            count++;
        }
        return count;
    }
}

cgc_layer_gguf_meta_t cgc_parse_layer_gguf_meta(const cgc_gguf_lite_ctx_t* ctx) {
    cgc_layer_gguf_meta_t meta;
    memset(&meta, 0, sizeof(meta));

    if (!ctx) return meta;

    uint32_t expert_count = 0;
    if (get_arch_u32(ctx, "expert_count", &expert_count)) {
        meta.experts_per_layer = (int)expert_count;
    }

    uint32_t block_count = 0;
    if (get_arch_u32(ctx, "block_count", &block_count)) {
        meta.layer_index = (int)block_count;
    }

    uint32_t hidden_size = 0;
    if (get_arch_u32(ctx, "embedding_length", &hidden_size)) {
        meta.hidden_size = (int)hidden_size;
    }

    uint32_t ffn_len = 0;
    if (get_arch_u32(ctx, "expert_feed_forward_length", &ffn_len)) {
        meta.moe_intermediate_size = (int)ffn_len;
    } else if (get_arch_u32(ctx, "feed_forward_length", &ffn_len)) {
        meta.moe_intermediate_size = (int)ffn_len;
    }

    return meta;
}
