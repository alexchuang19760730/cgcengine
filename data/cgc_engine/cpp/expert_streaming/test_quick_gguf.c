#include "cgc_gguf_lite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("=== Quick GGUF Header Verification ===\n\n");
    
    const char* gguf_path = "C:/Users/alexchuang/Desktop/fastprefill/gemma4_gguf/gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf";
    
    printf("Loading GGUF file (header only): %s\n", gguf_path);
    
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(gguf_path);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to load GGUF file!\n");
        return 1;
    }
    
    printf("SUCCESS: GGUF header loaded\n\n");
    printf("--- Basic Info ---\n");
    printf("  Version: %u\n", ctx->version);
    printf("  Num tensors: %llu\n", (unsigned long long)ctx->n_tensors);
    printf("  Num KV pairs: %llu\n", (unsigned long long)ctx->n_kv);
    printf("  Data start: %llu\n", (unsigned long long)ctx->data_start);
    
    printf("\n--- Metadata ---\n");
    
    const char* arch = cgc_gguf_lite_get_str(ctx, "general.architecture");
    printf("  Architecture: %s\n", arch ? arch : "N/A");
    
    uint32_t expert_count = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expert_count)) {
        printf("  Expert count: %u\n", expert_count);
    }
    
    uint32_t block_count = 0;
    if (cgc_gguf_lite_get_u32(ctx, "gemma4.block_count", &block_count)) {
        printf("  Block count: %u\n", block_count);
    }
    
    int32_t hidden_size = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.embedding_length", &hidden_size)) {
        printf("  Hidden size: %d\n", hidden_size);
    }
    
    int32_t ffn_len = 0;
    if (cgc_gguf_lite_get_i32(ctx, "gemma4.expert_feed_forward_length", &ffn_len)) {
        printf("  Expert FFN length: %d\n", ffn_len);
    }
    
    printf("\n--- Searching for _exps tensors ---\n");
    
    int exps_count = 0;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const char* name = ctx->tensor_names[i];
        if (!name) continue;
        
        if (strstr(name, "_exps")) {
            exps_count++;
            
            if (exps_count <= 6) {
                cgc_gguf_tensor_info_t* ti = &ctx->tensors[i];
                printf("  [%llu] %s\n", (unsigned long long)i, name);
                printf("    dims: [%lld, %lld, %lld]\n",
                       (long long)ti->dims[0], (long long)ti->dims[1], (long long)ti->dims[2]);
                printf("    type: %d\n", ti->type);
                printf("    offset: %llu\n", (unsigned long long)ti->offset);
                printf("    n_elements: %llu\n", (unsigned long long)ti->n_elements);
                
                double bpe = cgc_ggml_type_bytes_per_elem(ti->type);
                uint64_t size = (uint64_t)(bpe * (double)ti->n_elements);
                printf("    size: %llu bytes (%.1f MB)\n", (unsigned long long)size, (double)size/1024/1024);
                
                int expert_dims = ti->n_dims - 1;
                int64_t stride = 1;
                for (int d = 0; d < expert_dims; d++) stride *= ti->dims[d];
                uint64_t expert_size = (uint64_t)(bpe * (double)stride);
                printf("    per_expert: %llu bytes (%.1f MB)\n\n", (unsigned long long)expert_size, (double)expert_size/1024/1024);
            }
        }
    }
    
    printf("  Total _exps tensors: %d\n", exps_count);
    
    if (exps_count > 0) {
        printf("\n--- Layout Summary ---\n");
        printf("  Layout: PER_LAYER (Gemma 4 style)\n");
        printf("  Experts per layer: %u\n", expert_count);
        printf("  Layers: %u\n", block_count);
        printf("  Total expert tensors: %d (3 per layer: scale, ffn_down, ffn_gate_up)\n", exps_count);
    }
    
    cgc_gguf_lite_free(ctx);
    printf("\nDone.\n");
    return 0;
}
