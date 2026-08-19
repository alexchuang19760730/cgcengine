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
    int experts_per_layer = 0;   // 真实专家数量 (与 KV 元数据一致)

    if (layout_type == CGC_LAYOUT_PER_EXPERT) {
        int max_expert = 0;
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer, expert;
            char role[CGC_MAX_NAME_LEN];
            if (parse_per_expert_name(ctx->tensor_names[i], &layer, &expert, role)) {
                if (layer > max_layer) max_layer = layer;
                if (expert > max_expert) max_expert = expert;
            }
        }
        // per-expert 布局: expert 索引从 0 开始, 数量 = max_index + 1
        experts_per_layer = max_expert + 1;
    } else {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            int layer;
            char exps_type[CGC_MAX_NAME_LEN];
            if (parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) {
                if (layer > max_layer) max_layer = layer;
            }
        }

        // per-layer 布局: gemma4.expert_count 是"数量" (如 128), 直接使用
        uint32_t expert_count_u32 = 0;
        if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expert_count_u32) && expert_count_u32 > 0) {
            experts_per_layer = (int)expert_count_u32;
        } else {
            // 回退: 从 tensor 末维推断 (dim 是 expert 数量或 max_index+1, 取最大值保守)
            int max_dim = 0;
            for (uint64_t i = 0; i < ctx->n_tensors; i++) {
                int layer;
                char exps_type[CGC_MAX_NAME_LEN];
                if (!parse_per_layer_name(ctx->tensor_names[i], &layer, exps_type)) continue;
                for (int d = 0; d < ctx->tensors[i].n_dims; d++) {
                    if (ctx->tensors[i].dims[d] > max_dim) {
                        max_dim = (int)ctx->tensors[i].dims[d];
                    }
                }
            }
            // GGUF 中 ffn_down_exps 形状是 [hidden, ffn, experts],
            // 末维即 expert 数量 (无 +1, 因为它描述的是"层内专家个数")
            experts_per_layer = max_dim;
        }
    }

    int layer_count = max_layer + 1;

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
        if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_used_count", &active_experts)) {
            layers[l].active_experts = (int)active_experts;
        } else {
            layers[l].active_experts = 8;
        }

        int32_t hidden = 0;
        if (cgc_gguf_lite_get_i32(ctx, "gemma4.embedding_length", &hidden)) {
            layers[l].hidden_size = hidden;
        }

        int32_t ffn_len = 0;
        if (cgc_gguf_lite_get_i32(ctx, "gemma4.expert_feed_forward_length", &ffn_len)) {
            layers[l].moe_intermediate_size = ffn_len;
        } else {
            if (cgc_gguf_lite_get_i32(ctx, "gemma4.feed_forward_length", &ffn_len)) {
                layers[l].moe_intermediate_size = ffn_len;
            }
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

    if (layout_type == CGC_LAYOUT_PER_LAYER) {
        uint64_t total_size = 0;
        for (int l = 0; l < layer_count; l++) {
            total_size += layers[l].ffn_down_exps_size;
            total_size += layers[l].ffn_gate_up_exps_size;
            if (layers[l].ffn_down_exps_scale_name[0] != '\0') {
                total_size += 512;
            }
        }

        uint64_t first_offset = layers[0].ffn_down_exps_offset;
        uint64_t last_offset = 0;
        uint64_t last_size = 0;

        for (int l = layer_count - 1; l >= 0; l--) {
            if (layers[l].ffn_gate_up_exps_offset > last_offset) {
                last_offset = layers[l].ffn_gate_up_exps_offset;
                last_size = layers[l].ffn_gate_up_exps_size;
            }
            if (layers[l].ffn_down_exps_offset > last_offset) {
                last_offset = layers[l].ffn_down_exps_offset;
                last_size = layers[l].ffn_down_exps_size;
            }
        }

        layout.stream_offset = first_offset;
        layout.stream_size = last_offset + last_size - first_offset;
        layout.experts_per_layer = experts_per_layer;

        if (layer_count > 1) {
            uint64_t second_offset = layers[1].ffn_down_exps_offset;
            layout.expert_stride = second_offset - first_offset;
        } else {
            layout.expert_stride = total_size;
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

        layout.stream_offset = first_offset;
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
            layout.expert_offsets[l] = layers[l].ffn_down_exps_offset;
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
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expert_count)) {
        meta.experts_per_layer = (int)expert_count;
    }

    uint32_t block_count = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.block_count", &block_count)) {
        meta.layer_index = (int)block_count;
    }

    int32_t hidden_size = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.embedding_length", &hidden_size)) {
        meta.hidden_size = hidden_size;
    }

    int32_t ffn_len = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.expert_feed_forward_length", &ffn_len)) {
        meta.moe_intermediate_size = ffn_len;
    } else {
        if (cgc_gguf_lite_get_i32(ctx, "gemma4.feed_forward_length", &ffn_len)) {
            meta.moe_intermediate_size = ffn_len;
        }
    }

    return meta;
}
