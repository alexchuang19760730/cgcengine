#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGUF context
typedef struct gguf_context gguf_context_t;

// Tensor info
typedef struct {
    int64_t dims[4];
    int n_dims;
    int type;
    size_t offset;
    size_t n_elements;
} gguf_tensor_info_t;

// Load GGUF file
gguf_context_t* gguf_load(const char* filename, const char* key);

// Free GGUF context
void gguf_free(gguf_context_t* ctx);

// Get number of tensors
int gguf_get_n_tensors(const gguf_context_t* ctx);

// Get tensor name
const char* gguf_get_tensor_name(const gguf_context_t* ctx, int index);

// Get tensor info
gguf_tensor_info_t gguf_get_tensor_info(const gguf_context_t* ctx, int index);

// Get tensor data (returns pointer to dequantized data)
void* gguf_get_tensor_data(gguf_context_t* ctx, int index);

// Get metadata values
int64_t gguf_get_val_i64(const gguf_context_t* ctx, const char* key);
int gguf_get_val_i32(const gguf_context_t* ctx, const char* key);
float gguf_get_val_f32(const gguf_context_t* ctx, const char* key);
const char* gguf_get_val_str(const gguf_context_t* ctx, const char* key);
bool gguf_has_key(const gguf_context_t* ctx, const char* key);

#ifdef __cplusplus
}
#endif