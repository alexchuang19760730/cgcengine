#pragma once
#include "gguf.h"
#include "cgc_cpp.h"

#ifdef __cplusplus
extern "C" {
#endif

// CGC Model structure (from gguf)
typedef struct {
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t n_tensors;
    void** tensors;
    char** tensor_names;
    int64_t* tensor_shapes;
} CGCModel;

// CGC Tensor structure
typedef struct {
    const char* name;
    void* data;
    int32_t ndim;
    int32_t type;
    int32_t shape[4];
} CGCTensor;

// 🔥 核心：直接从 GGUF 加载模型到 CGC C++ 引擎
CGCModel* cgc_model_load_from_gguf(const char* filename);

// 工具：GGUF tensor → CGC tensor
CGCTensor cgc_tensor_from_gguf(gguf_context_t* ctx, int tensor_index);

// 获取模型元数据
bool cgc_gguf_get_metadata(gguf_context_t* ctx, CGCModel* model);

#ifdef __cplusplus
}
#endif