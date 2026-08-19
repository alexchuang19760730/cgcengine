#include "cgc_gguf_lite.h"
#include "cgc_expert_streamer_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* gguf_path = argc > 1 ? argv[1] : 
        "C:/Users/alexchuang/Desktop/fastprefill/gemma4_gguf/gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf";
    
    printf("=== C GGUF Parser Verification for Gemma 4 ===\n\n");
    
    printf("Loading GGUF file: %s\n", gguf_path);
    
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(gguf_path);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to load GGUF file!\n");
        return 1;
    }
    
    printf("SUCCESS: Loaded GGUF header\n");
    printf("  Version: %u\n", ctx->version);
    printf("  Num tensors: %llu\n", (unsigned long long)ctx->n_tensors);
    printf("  Num KV pairs: %llu\n", (unsigned long long)ctx->n_kv);
    printf("  Data start offset: %llu\n", (unsigned long long)ctx->data_start);
    
    printf("\n--- Key KV Metadata ---\n");
    
    const char* arch = cgc_gguf_lite_get_str(ctx, "general.architecture");
    printf("  Architecture: %s\n", arch ? arch : "N/A");
    
    uint32_t expert_count = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expert_count)) {
        printf("  Expert count: %u\n", expert_count);
    } else {
        printf("  Expert count: NOT FOUND\n");
    }
    
    uint32_t active_experts = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_used_count", &active_experts)) {
        printf("  Active experts: %u\n", active_experts);
    }
    
    uint32_t block_count = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.block_count", &block_count)) {
        printf("  Block count (layers): %u\n", block_count);
    }
    
    int32_t hidden_size = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.embedding_length", &hidden_size)) {
        printf("  Hidden size: %d\n", hidden_size);
    }
    
    int32_t ffn_len = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.expert_feed_forward_length", &ffn_len)) {
        printf("  Expert FFN length: %d\n", ffn_len);
    }
    
    uint32_t file_type = 0;
    if (cgc_gguf_lite_get_u32(ctx, "general.file_type", &file_type)) {
        printf("  File type: %u (30=IQ4_XS)\n", file_type);
    }
    
    printf("\n--- Expert Tensor Analysis ---\n");
    
    int per_layer_count = 0;
    int per_expert_count = 0;
    
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const char* name = ctx->tensor_names[i];
        if (!name) continue;
        
        if (strstr(name, "_exps")) {
            per_layer_count++;
            if (per_layer_count <= 10) {
                cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];
                printf("  [PER_LAYER] %s:\n", name);
                printf("    dims: [%lld, %lld, %lld", 
                       (long long)ti->dims[0], 
                       (long long)ti->dims[1],
                       (long long)ti->dims[2]);
                if (ti->n_dims > 3) printf(", %lld", (long long)ti->dims[3]);
                printf("]\n");
                printf("    type: %d\n", ti->type);
                printf("    offset: %llu\n", (unsigned long long)ti->offset);
                printf("    n_elements: %llu\n", (unsigned long long)ti->n_elements);
                
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                uint64_t size_bytes = (uint64_t)(bpe * (double)ti->n_elements);
                printf("    size: %llu bytes (%.1f MB)\n", 
                       (unsigned long long)size_bytes,
                       (double)size_bytes / 1024.0 / 1024.0);
                
                int expert_dims_idx = ti->n_dims - 1;
                int64_t stride_elems = 1;
                for (int d = 0; d < expert_dims_idx; d++) {
                    stride_elems *= ti->dims[d];
                }
                uint64_t expert_stride_bytes = (uint64_t)(bpe * (double)stride_elems);
                printf("    expert_stride: %llu bytes (%.1f MB)\n",
                       (unsigned long long)expert_stride_bytes,
                       (double)expert_stride_bytes / 1024.0 / 1024.0);
            }
        }
        
        if (strstr(name, "expert")) {
            per_expert_count++;
        }
    }
    
    printf("\n  Per-layer expert tensors (_exps): %d\n", per_layer_count);
    printf("  Per-expert tensors (expert.X): %d\n", per_expert_count);
    
    if (per_layer_count > 0) {
        printf("\n  Layout: PER_LAYER (Gemma 4 style, all experts packed)\n");
    } else if (per_expert_count > 0) {
        printf("\n  Layout: PER_EXPERT (Qwen3.6 style, each expert separate)\n");
    }
    
    printf("\n--- Testing Stream Layout Creation ---\n");
    
    cgc_stream_layout_t layout = cgc_load_stream_layout_from_gguf(gguf_path);
    
    printf("  Layout path: %s\n", layout.path);
    printf("  Stream offset: %llu\n", (unsigned long long)layout.stream_offset);
    printf("  Stream size: %llu bytes (%.1f MB)\n", 
           (unsigned long long)layout.stream_size,
           (double)layout.stream_size / 1024.0 / 1024.0);
    printf("  Experts per layer: %d\n", layout.experts_per_layer);
    printf("  Expert stride: %llu bytes (%.1f MB)\n",
           (unsigned long long)layout.expert_stride,
           (double)layout.expert_stride / 1024.0 / 1024.0);
    printf("  Has explicit offsets: %d\n", layout.has_explicit_offsets);
    
    if (layout.has_explicit_offsets) {
        printf("  First 5 layer offsets:\n");
        for (int i = 0; i < 5 && i < CGC_MAX_EXPERTS_PER_LAYER; i++) {
            if (layout.expert_offsets[i] > 0) {
                printf("    Layer %d: offset=%llu\n", i, (unsigned long long)layout.expert_offsets[i]);
            }
        }
    }
    
    printf("\n--- Testing Layer Meta Parsing ---\n");
    
    cgc_layer_gguf_meta_t meta = cgc_parse_layer_gguf_meta(ctx);
    printf("  Layer index: %d\n", meta.layer_index);
    printf("  Experts per layer: %d\n", meta.experts_per_layer);
    printf("  Hidden size: %d\n", meta.hidden_size);
    printf("  MoE intermediate: %d\n", meta.moe_intermediate_size);
    
    printf("\n--- Verification Complete ---\n");
    
    cgc_gguf_lite_free(ctx);
    return 0;
}
