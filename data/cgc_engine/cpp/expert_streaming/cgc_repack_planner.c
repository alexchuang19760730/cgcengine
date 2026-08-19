// cgc_repack_planner.c — 生成 per-layer 布局 (C 实现)
//
// 移植自 turbo-fieldfare 的 RepackPlanner.swift
// 对每个 layer,找到 gate/up/down 权重,计算 per-expert stride,生成 LayerFilePlan

#include "cgc_repack.h"
#include "cgc_quantize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 在 tensors 中找指定 layer + role 的 expert weight tensor
// ============================================================================

static const cgc_source_tensor_t* find_expert_tensor(
    const cgc_source_tensor_t* tensors, int n_tensors,
    int layer, const char* role)
{
    for (int i = 0; i < n_tensors; ++i) {
        cgc_bucket_t b = cgc_classify_tensor(tensors[i].name, layer + 1);
        if (b.kind == CGC_BUCKET_ROUTED_EXPERT && b.layer == layer &&
            strcmp(b.role, role) == 0) {
            return &tensors[i];
        }
    }
    return NULL;
}

// ============================================================================
// 生成 repack 计划
// ============================================================================

int cgc_repack_plan_create(const cgc_source_tensor_t* tensors, int n_tensors,
                            const cgc_arch_info_t* arch,
                            const char* output_dir,
                            cgc_repack_plan_t* out_plan)
{
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->arch = *arch;
    out_plan->n_layers = arch->num_layers;
    strncpy(out_plan->output_dir, output_dir, sizeof(out_plan->output_dir) - 1);

    out_plan->layers = (cgc_layer_file_plan_t*)calloc(arch->num_layers, sizeof(cgc_layer_file_plan_t));
    if (!out_plan->layers) return -1;

    // 对每个 layer 生成 plan
    for (int layer = 0; layer < arch->num_layers; ++layer) {
        cgc_layer_file_plan_t* lp = &out_plan->layers[layer];
        lp->layer_index = layer;
        snprintf(lp->path, sizeof(lp->path), "%s\\layer_%d.gguf", output_dir, layer);
        lp->experts_per_layer = arch->num_experts;

        // 默认 physical order = identity (0,1,2,...,N-1)
        for (int e = 0; e < arch->num_experts && e < 256; ++e) {
            lp->physical_order[e] = e;
        }

        // 找 gate/up/down 权重
        // Gemma 4 expert tensor 通常是 2D: [moe_intermediate_size, hidden_size] (gate/up)
        //                                            [hidden_size, moe_intermediate_size] (down)
        // 所有 expert 打包在一个大 tensor 里:shape = [num_experts, out, in]
        const cgc_source_tensor_t* gate = find_expert_tensor(tensors, n_tensors, layer, "gate");
        const cgc_source_tensor_t* up   = find_expert_tensor(tensors, n_tensors, layer, "up");
        const cgc_source_tensor_t* down = find_expert_tensor(tensors, n_tensors, layer, "down");

        if (!gate || !up || !down) {
            fprintf(stderr, "[planner] layer %d: missing expert tensors (gate=%p up=%p down=%p)\n",
                    layer, gate, up, down);
            // 跳过这个 layer (可能是 dense layer)
            continue;
        }

        // per-expert 大小 = gate_size + up_size + down_size (所有 expert 打包在一个 tensor)
        // shape[0] = num_experts, 其余 = per-expert shape
        uint64_t gate_per_expert = gate->size_bytes / arch->num_experts;
        uint64_t up_per_expert   = up->size_bytes   / arch->num_experts;
        uint64_t down_per_expert = down->size_bytes / arch->num_experts;

        // 构建 sub_tensors (9 个: gate/up/down × {weights, scales, biases})
        // 简化版:只有 weights (无 scales/biases,IQ3_M 量化后 scales 内嵌在 block 里)
        uint32_t n_sub = 0;
        cgc_per_expert_tensor_slice_t* s;

        // gate weights
        s = &lp->sub_tensors[n_sub++];
        strcpy(s->role, "gate"); strcpy(s->component, "weights");
        s->dtype = gate->dtype;
        s->n_dims = gate->n_dims > 0 ? gate->n_dims - 1 : 1; // 去掉 expert 维
        for (uint32_t i = 0; i < s->n_dims && i + 1 < 4; ++i) s->logical_shape[i] = gate->shape[i + 1];
        s->size_in_expert_blob = gate_per_expert;
        s->bits_for_weights = 3; // IQ3_M

        // up weights
        s = &lp->sub_tensors[n_sub++];
        strcpy(s->role, "up"); strcpy(s->component, "weights");
        s->dtype = up->dtype;
        s->n_dims = up->n_dims > 0 ? up->n_dims - 1 : 1;
        for (uint32_t i = 0; i < s->n_dims && i + 1 < 4; ++i) s->logical_shape[i] = up->shape[i + 1];
        s->size_in_expert_blob = up_per_expert;
        s->bits_for_weights = 3;

        // down weights
        s = &lp->sub_tensors[n_sub++];
        strcpy(s->role, "down"); strcpy(s->component, "weights");
        s->dtype = down->dtype;
        s->n_dims = down->n_dims > 0 ? down->n_dims - 1 : 1;
        for (uint32_t i = 0; i < s->n_dims && i + 1 < 4; ++i) s->logical_shape[i] = down->shape[i + 1];
        s->size_in_expert_blob = down_per_expert;
        s->bits_for_weights = 3;

        lp->n_sub_tensors = n_sub;

        // 计算 expert stride 和各 sub_tensor 在 expert blob 内的 offset
        // IQ3_S: block_iq3_s = 80 bytes / 256 elements = 0.3125 * 8 = 2.5 bpw
        // 实际从 ggml: type_traits[GGML_TYPE_IQ3_S].type_size = 80, blck_size = 256
        uint64_t offset = 0;
        for (uint32_t i = 0; i < n_sub; ++i) {
            lp->sub_tensors[i].offset_in_expert_blob = offset;
            uint64_t elem_bytes = cgc_dtype_element_bytes(lp->sub_tensors[i].dtype);
            uint64_t n_elems = lp->sub_tensors[i].size_in_expert_blob / elem_bytes;
            // IQ3_S 量化后大小 = (n_elems / 256) * 80
            uint64_t n_blocks = (n_elems + 255) / 256;
            uint64_t quantized_size = n_blocks * 80; // IQ3_S block = 80 bytes
            lp->sub_tensors[i].size_in_expert_blob = quantized_size;
            offset += quantized_size;
        }
        lp->expert_stride = offset;

        if (layer < 3 || layer == arch->num_layers - 1) {
            fprintf(stderr, "[planner] layer %d: experts=%d stride=%llu gate=%llu up=%llu down=%llu\n",
                    layer, lp->experts_per_layer,
                    (unsigned long long)lp->expert_stride,
                    (unsigned long long)lp->sub_tensors[0].size_in_expert_blob,
                    (unsigned long long)lp->sub_tensors[1].size_in_expert_blob,
                    (unsigned long long)lp->sub_tensors[2].size_in_expert_blob);
        }
    }

    snprintf(out_plan->manifest_path, sizeof(out_plan->manifest_path),
             "%s\\manifest.json", output_dir);

    fprintf(stderr, "[planner] plan created: %d layers, manifest=%s\n",
            out_plan->n_layers, out_plan->manifest_path);
    return 0;
}

void cgc_repack_plan_free(cgc_repack_plan_t* plan) {
    if (plan->layers) {
        free(plan->layers);
        plan->layers = NULL;
    }
}
