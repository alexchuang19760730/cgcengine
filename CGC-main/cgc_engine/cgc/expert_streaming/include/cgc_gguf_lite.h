// cgc_gguf_lite.h — 轻量 GGUF v3 header 解析器 (只读 header,不加载 data)
//
// 专为 expert_streaming 设计:
// - 遵循标准 GGUF v3 格式 (与 cgc_repack.c 输出匹配)
// - 只读 header + tensor info (几 KB),不加载 12GB tensor data
// - 提供 KV metadata 查询 (gemma4.expert_count, gemma4.expert_stride 等)
// - 提供 tensor info 查询 (name, dims, type, offset)
//
// 与 cgc_cpp/gguf.c 的区别:
// - gguf.c 的 header 解析不标准 (多读 tensor_offset/tensor_data_offset),无法解析 repack 输出
// - cgc_gguf_lite 严格遵循标准 GGUF v3,且不加载 data 到内存

#ifndef CGC_GGUF_LITE_H
#define CGC_GGUF_LITE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CGC_GGUF_LITE_MAGIC 0x46554747u  // "GGUF" little-endian

// GGUF value types (标准 GGUF v3)
enum {
    CGC_GGUF_TYPE_UINT8   = 0,
    CGC_GGUF_TYPE_INT8    = 1,
    CGC_GGUF_TYPE_UINT16  = 2,
    CGC_GGUF_TYPE_INT16   = 3,
    CGC_GGUF_TYPE_UINT32  = 4,
    CGC_GGUF_TYPE_INT32   = 5,
    CGC_GGUF_TYPE_FLOAT32 = 6,
    CGC_GGUF_TYPE_BOOL    = 7,
    CGC_GGUF_TYPE_STRING  = 8,
    CGC_GGUF_TYPE_ARRAY   = 9,
    CGC_GGUF_TYPE_UINT64  = 10,
    CGC_GGUF_TYPE_INT64   = 11,
    CGC_GGUF_TYPE_FLOAT64 = 12,
};

// ggml tensor dtype (与 ggml.h 一致,只列需要的)
enum {
    CGC_GGML_TYPE_F32    = 0,
    CGC_GGML_TYPE_F16    = 1,
    CGC_GGML_TYPE_Q4_0   = 2,
    CGC_GGML_TYPE_Q4_1   = 3,
    CGC_GGML_TYPE_Q5_0   = 6,
    CGC_GGML_TYPE_Q5_1   = 7,
    CGC_GGML_TYPE_Q8_0   = 8,
    CGC_GGML_TYPE_Q2_K   = 10,
    CGC_GGML_TYPE_Q3_K   = 11,
    CGC_GGML_TYPE_Q4_K   = 12,
    CGC_GGML_TYPE_Q5_K   = 13,
    CGC_GGML_TYPE_Q6_K   = 14,
    CGC_GGML_TYPE_Q8_K   = 15,
    CGC_GGML_TYPE_IQ2_XXS = 16,
    CGC_GGML_TYPE_IQ2_XS  = 17,
    CGC_GGML_TYPE_IQ3_XXS = 18,
    CGC_GGML_TYPE_IQ1_S   = 19,
    CGC_GGML_TYPE_IQ4_NL  = 20,
    CGC_GGML_TYPE_IQ3_S   = 21,
    CGC_GGML_TYPE_IQ2_S   = 22,
    CGC_GGML_TYPE_IQ4_XS  = 23,
    CGC_GGML_TYPE_IQ1_M   = 29,
    CGC_GGML_TYPE_BF16   = 30,
};

// tensor info
typedef struct {
    int64_t dims[4];
    int32_t n_dims;
    int32_t type;         // CGC_GGML_TYPE_*
    uint64_t offset;      // 相对于 data_start 的偏移
    uint64_t n_elements;
} cgc_gguf_tensor_info_t;

// KV metadata entry
typedef struct {
    char key[128];
    int32_t value_type;   // CGC_GGUF_TYPE_*
    // 联合值 (简化:只存 string 和数值)
    char str_val[512];
    int64_t i64_val;
    double f64_val;
} cgc_gguf_kv_t;

// 解析上下文
typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t data_start;        // tensor data 区的文件绝对偏移
    cgc_gguf_tensor_info_t* tensors;
    char** tensor_names;        // 每个 tensor 的 name (堆分配)
    cgc_gguf_kv_t* kvs;
} cgc_gguf_lite_ctx_t;

// 加载 GGUF header (只读 header + tensor info,不加载 data)
// 返回 NULL = 失败
cgc_gguf_lite_ctx_t* cgc_gguf_lite_load(const char* filename);

// 释放
void cgc_gguf_lite_free(cgc_gguf_lite_ctx_t* ctx);

// 查询 KV metadata
const char* cgc_gguf_lite_get_str(const cgc_gguf_lite_ctx_t* ctx, const char* key);
bool cgc_gguf_lite_get_i32(const cgc_gguf_lite_ctx_t* ctx, const char* key, int32_t* out);
bool cgc_gguf_lite_get_u32(const cgc_gguf_lite_ctx_t* ctx, const char* key, uint32_t* out);
bool cgc_gguf_lite_get_i64(const cgc_gguf_lite_ctx_t* ctx, const char* key, int64_t* out);

// 查找 tensor by name,返回 index (-1 = not found)
int cgc_gguf_lite_find_tensor(const cgc_gguf_lite_ctx_t* ctx, const char* name);

// 计算 ggml type 的每元素字节数 (block-quantized 类型按 block 折算)
// 返回 0 = 未知类型
double cgc_ggml_type_bytes_per_elem(int32_t ggml_type);

#ifdef __cplusplus
}
#endif

#endif // CGC_GGUF_LITE_H
