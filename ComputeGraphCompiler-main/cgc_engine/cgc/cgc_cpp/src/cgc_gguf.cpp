#include "cgc_gguf.h"
#include "cgc_cpp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations for GGUF implementation
extern "C" {
    gguf_context_t* gguf_load(const char* filename, const char* key);
    void gguf_free(gguf_context_t* ctx);
    int gguf_get_n_tensors(const gguf_context_t* ctx);
    const char* gguf_get_tensor_name(const gguf_context_t* ctx, int index);
    gguf_tensor_info_t gguf_get_tensor_info(const gguf_context_t* ctx, int index);
    void* gguf_get_tensor_data(gguf_context_t* ctx, int index);
    int64_t gguf_get_val_i64(const gguf_context_t* ctx, const char* key);
    bool gguf_has_key(const gguf_context_t* ctx, const char* key);
}

CGCModel* cgc_model_load_from_gguf(const char* filename) {
    printf("[CGC-GGUF] Loading model: %s\n", filename);

    // Load GGUF file
    gguf_context_t* ctx = gguf_load(filename, NULL);
    if (!ctx) {
        printf("[CGC-GGUF] Failed to load GGUF file\n");
        return NULL;
    }

    // Allocate model
    CGCModel* model = (CGCModel*)malloc(sizeof(CGCModel));
    memset(model, 0, sizeof(CGCModel));

    // Read architecture metadata
    if (!cgc_gguf_get_metadata(ctx, model)) {
        printf("[CGC-GGUF] Failed to read metadata\n");
        gguf_free(ctx);
        free(model);
        return NULL;
    }

    // Load all tensors
    model->n_tensors = gguf_get_n_tensors(ctx);
    model->tensors = (void**)malloc(model->n_tensors * sizeof(void*));
    model->tensor_names = (char**)malloc(model->n_tensors * sizeof(char*));
    model->tensor_shapes = (int64_t*)malloc(model->n_tensors * 4 * sizeof(int64_t));

    for (int i = 0; i < model->n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        gguf_tensor_info_t info = gguf_get_tensor_info(ctx, i);
        void* data = gguf_get_tensor_data(ctx, i);

        model->tensor_names[i] = strdup(name);
        model->tensors[i] = data;
        for (int j = 0; j < info.n_dims && j < 4; j++) {
            model->tensor_shapes[i * 4 + j] = info.dims[j];
        }
    }

    gguf_free(ctx);
    printf("[CGC-GGUF] Loaded %d tensors, hidden_dim=%d, layers=%d, heads=%d\n", 
           model->n_tensors, model->hidden_dim, model->n_layers, model->n_heads);
    
    return model;
}

bool cgc_gguf_get_metadata(gguf_context_t* ctx, CGCModel* model) {
    // Try Qwen2.5 specific fields first
    if (gguf_has_key(ctx, "qwen2.embedding_length")) {
        model->hidden_dim = (int32_t)gguf_get_val_i64(ctx, "qwen2.embedding_length");
        model->n_layers = (int32_t)gguf_get_val_i64(ctx, "qwen2.block_count");
        model->n_heads = (int32_t)gguf_get_val_i64(ctx, "qwen2.attention.head_count");
        model->n_kv_heads = (int32_t)gguf_get_val_i64(ctx, "qwen2.attention.head_count_kv");
        model->vocab_size = (int32_t)gguf_get_val_i64(ctx, "qwen2.vocabulary_size");
    }
    // Fallback to llama fields
    else if (gguf_has_key(ctx, "llama.embedding_length")) {
        model->hidden_dim = (int32_t)gguf_get_val_i64(ctx, "llama.embedding_length");
        model->n_layers = (int32_t)gguf_get_val_i64(ctx, "llama.block_count");
        model->n_heads = (int32_t)gguf_get_val_i64(ctx, "llama.attention.head_count");
        model->n_kv_heads = (int32_t)gguf_get_val_i64(ctx, "llama.attention.head_count_kv");
        model->vocab_size = (int32_t)gguf_get_val_i64(ctx, "llama.vocabulary_size");
    }
    // Generic fallback
    else {
        model->hidden_dim = (int32_t)gguf_get_val_i64(ctx, "n_embd");
        model->n_layers = (int32_t)gguf_get_val_i64(ctx, "n_layer");
        model->n_heads = (int32_t)gguf_get_val_i64(ctx, "n_head");
        model->n_kv_heads = (int32_t)gguf_get_val_i64(ctx, "n_head_kv");
        model->vocab_size = (int32_t)gguf_get_val_i64(ctx, "vocab_size");
    }

    // Set defaults if not found
    if (model->hidden_dim == 0) model->hidden_dim = 4096;
    if (model->n_layers == 0) model->n_layers = 32;
    if (model->n_heads == 0) model->n_heads = 32;
    if (model->n_kv_heads == 0) model->n_kv_heads = model->n_heads;
    if (model->vocab_size == 0) model->vocab_size = 32000;
    
    // Calculate head_dim
    model->head_dim = model->hidden_dim / model->n_heads;

    return true;
}

CGCTensor cgc_tensor_from_gguf(gguf_context_t* ctx, int tensor_index) {
    CGCTensor t = {0};
    
    const char* name = gguf_get_tensor_name(ctx, tensor_index);
    gguf_tensor_info_t info = gguf_get_tensor_info(ctx, tensor_index);
    void* data = gguf_get_tensor_data(ctx, tensor_index);

    t.data = (float*)data;
    for (int i = 0; i < info.n_dims && i < 4; i++) {
        t.shape[i] = info.dims[i];
    }
    t.ndim = info.n_dims;
    
    return t;
}