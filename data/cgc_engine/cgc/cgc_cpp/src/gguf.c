#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// GGUF Magic number
#define GGUF_MAGIC 0x46554747

// GGUF Version
#define GGUF_VERSION 3

// Quantization types
typedef enum {
    GGUF_TYPE_F32  = 0,
    GGUF_TYPE_F16  = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q5_0 = 6,
    GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8,
    GGUF_TYPE_Q8_1 = 9,
    GGUF_TYPE_Q2_K = 10,
    GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12,
    GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14,
    GGUF_TYPE_Q8_K = 15,
} gguf_type_t;

struct gguf_context {
    FILE* file;
    char* filename;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    void* kv_data;
    uint64_t tensor_offset;
    uint64_t tensor_data_offset;
    
    // Parsed tensors
    char** tensor_names;
    gguf_tensor_info_t* tensor_infos;
    void** tensor_data;
};

// Helper functions
static uint64_t fread_le64(FILE* f) {
    uint64_t v;
    fread(&v, sizeof(v), 1, f);
    return v;
}

static uint32_t fread_le32(FILE* f) {
    uint32_t v;
    fread(&v, sizeof(v), 1, f);
    return v;
}

static float fread_f32(FILE* f) {
    float v;
    fread(&v, sizeof(v), 1, f);
    return v;
}

static char* fread_string(FILE* f) {
    uint64_t len = fread_le64(f);
    char* s = (char*)malloc(len + 1);
    fread(s, 1, len, f);
    s[len] = 0;
    return s;
}

// Dequantization helpers (simplified)
static float q4_k_dequantize(uint8_t* data, int idx, float* scales, uint8_t* zeros, int block_size) {
    int block_idx = idx / block_size;
    int in_block_idx = idx % block_size;
    uint8_t byte = data[block_idx * 8 + (in_block_idx >> 1)];
    uint8_t nibble = (in_block_idx & 1) ? (byte >> 4) : (byte & 0xF);
    float scale = scales[block_idx];
    float zero = zeros[block_idx];
    return scale * ((int)nibble - 8) + zero;
}

gguf_context_t* gguf_load(const char* filename, const char* key) {
    (void)key; // Unused for now
    
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    gguf_context_t* ctx = (gguf_context_t*)calloc(1, sizeof(gguf_context_t));
    ctx->file = f;
    ctx->filename = strdup(filename);
    
    // Read magic
    uint32_t magic = fread_le32(f);
    if (magic != GGUF_MAGIC) {
        fclose(f);
        free(ctx);
        return NULL;
    }
    
    // Read version
    ctx->version = fread_le32(f);
    
    // Read header
    ctx->n_tensors = fread_le64(f);
    ctx->n_kv = fread_le64(f);
    ctx->tensor_offset = fread_le64(f);
    ctx->tensor_data_offset = fread_le64(f);
    
    // Read KV pairs (metadata)
    ctx->kv_data = malloc(4096); // Buffer for metadata
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        char* key = fread_string(f);
        uint32_t type = fread_le32(f);
        switch (type) {
            case 0: fread_f32(f); break;    // f32
            case 1: fread_le32(f); break;   // i32
            case 2: fread_le64(f); break;   // i64
            case 3: {                        // string
                char* val = fread_string(f);
                free(val);
                break;
            }
            case 4: {                        // array
                uint64_t n = fread_le64(f);
                uint32_t etype = fread_le32(f);
                for (uint64_t j = 0; j < n; j++) {
                    if (etype == 0) fread_f32(f);
                    else if (etype == 1) fread_le32(f);
                    else if (etype == 2) fread_le64(f);
                }
                break;
            }
            default:
                // Skip unknown types
                fseek(f, 4, SEEK_CUR);
                break;
        }
        free(key);
    }
    
    // Read tensor info
    fseek(f, ctx->tensor_offset, SEEK_SET);
    ctx->tensor_names = (char**)malloc(ctx->n_tensors * sizeof(char*));
    ctx->tensor_infos = (gguf_tensor_info_t*)malloc(ctx->n_tensors * sizeof(gguf_tensor_info_t));
    ctx->tensor_data = (void**)malloc(ctx->n_tensors * sizeof(void*));
    
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        ctx->tensor_names[i] = fread_string(f);
        gguf_tensor_info_t* info = &ctx->tensor_infos[i];
        info->n_dims = fread_le32(f);
        for (int j = 0; j < info->n_dims; j++) {
            info->dims[j] = fread_le64(f);
        }
        info->type = fread_le32(f);
        info->offset = fread_le64(f);
        info->n_elements = 1;
        for (int j = 0; j < info->n_dims; j++) {
            info->n_elements *= info->dims[j];
        }
    }
    
    return ctx;
}

void gguf_free(gguf_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->tensor_names) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            free(ctx->tensor_names[i]);
            if (ctx->tensor_data[i]) free(ctx->tensor_data[i]);
        }
        free(ctx->tensor_names);
        free(ctx->tensor_infos);
        free(ctx->tensor_data);
    }
    
    if (ctx->file) fclose(ctx->file);
    if (ctx->filename) free(ctx->filename);
    if (ctx->kv_data) free(ctx->kv_data);
    free(ctx);
}

int gguf_get_n_tensors(const gguf_context_t* ctx) {
    return (int)ctx->n_tensors;
}

const char* gguf_get_tensor_name(const gguf_context_t* ctx, int index) {
    if (index < 0 || (uint64_t)index >= ctx->n_tensors) return NULL;
    return ctx->tensor_names[index];
}

gguf_tensor_info_t gguf_get_tensor_info(const gguf_context_t* ctx, int index) {
    gguf_tensor_info_t empty = {0};
    if (index < 0 || (uint64_t)index >= ctx->n_tensors) return empty;
    return ctx->tensor_infos[index];
}

void* gguf_get_tensor_data(gguf_context_t* ctx, int index) {
    if (index < 0 || (uint64_t)index >= ctx->n_tensors) return NULL;
    
    // Return cached data if already loaded
    if (ctx->tensor_data[index]) return ctx->tensor_data[index];
    
    gguf_tensor_info_t* info = &ctx->tensor_infos[index];
    
    // Allocate output buffer
    float* output = (float*)malloc(info->n_elements * sizeof(float));
    ctx->tensor_data[index] = output;
    
    // Seek to tensor data
    fseek(ctx->file, ctx->tensor_data_offset + info->offset, SEEK_SET);
    
    if (info->type == GGUF_TYPE_F32) {
        // Already float32
        fread(output, sizeof(float), info->n_elements, ctx->file);
    } else if (info->type == GGUF_TYPE_Q4_K) {
        // Q4_K quantization
        int block_size = 32;
        int n_blocks = (info->n_elements + block_size - 1) / block_size;
        
        // Read scales (f16)
        float* scales = (float*)malloc(n_blocks * sizeof(float));
        uint16_t* scales_f16 = (uint16_t*)malloc(n_blocks * sizeof(uint16_t));
        fread(scales_f16, sizeof(uint16_t), n_blocks, ctx->file);
        for (int i = 0; i < n_blocks; i++) {
            scales[i] = (float)*(uint16_t*)&scales_f16[i];
        }
        free(scales_f16);
        
        // Read zeros (Q4)
        uint8_t* zeros = (uint8_t*)malloc(n_blocks * sizeof(uint8_t));
        fread(zeros, 1, n_blocks * 2, ctx->file); // Q4 uses 2 bytes per block but we only need 1
        
        // Read quantized data
        int data_size = (info->n_elements + 1) / 2;
        uint8_t* data = (uint8_t*)malloc(data_size);
        fread(data, 1, data_size, ctx->file);
        
        // Dequantize
        for (int i = 0; i < (int)info->n_elements; i++) {
            output[i] = q4_k_dequantize(data, i, scales, zeros, block_size);
        }
        
        free(scales);
        free(zeros);
        free(data);
    } else {
        // For other types, just read as bytes and return empty
        memset(output, 0, info->n_elements * sizeof(float));
    }
    
    return output;
}

int64_t gguf_get_val_i64(const gguf_context_t* ctx, const char* key) {
    (void)ctx; (void)key;
    // Simplified: return 0 for now (real implementation would parse KV pairs)
    return 0;
}

int gguf_get_val_i32(const gguf_context_t* ctx, const char* key) {
    return (int)gguf_get_val_i64(ctx, key);
}

float gguf_get_val_f32(const gguf_context_t* ctx, const char* key) {
    (void)ctx; (void)key;
    return 0.0f;
}

const char* gguf_get_val_str(const gguf_context_t* ctx, const char* key) {
    (void)ctx; (void)key;
    return "";
}

bool gguf_has_key(const gguf_context_t* ctx, const char* key) {
    (void)ctx; (void)key;
    // Simplified: always return false
    return false;
}