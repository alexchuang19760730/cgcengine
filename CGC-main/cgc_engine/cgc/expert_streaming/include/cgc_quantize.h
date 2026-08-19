// cgc_quantize.h — IQ3_M 量化包装层
//
// 封装 llama.cpp 的 ggml_quantize_chunk,提供简洁的 IQ3_M 量化接口。
// IQ3_M 在 ggml 层面 = GGML_TYPE_IQ3_S (llama ftype 27 = IQ3_M,底层 iq3_s)
//
// 依赖:llama.cpp 的 ggml 库 (ggml.c + ggml-quants.c)

#ifndef CGC_QUANTIZE_H
#define CGC_QUANTIZE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// IQ3_M (底层 IQ3_S) 的 block size = 256
#define CGC_IQ3_M_BLOCK_SIZE 256
// 每个 block 的字节数 = sizeof(block_iq3_s) = 32 + 4 + 2 = 不对
// 实际:block_iq3_s { uint8_t qs[QK_K/4]; uint8_t qh; float d; } = 64 + 1 + ... 
// 从 ggml-common.h: block_iq3_s = 64 bytes (qs[QK_K/4=64]) + 1 (qh) ... 实际看定义
// 简化:用 cgc_iq3_m_row_size 计算每行字节数

// 计算量化后每行的字节数
// n_per_row: 每行元素数 (必须是 256 的倍数)
// 返回: 量化后字节数
size_t cgc_iq3_m_row_size(int64_t n_per_row);

// 执行 IQ3_M 量化
// src:       输入 float 数组 (nrow * n_per_row 个元素)
// dst:       输出缓冲区 (大小 = nrow * cgc_iq3_m_row_size(n_per_row))
// nrow:      行数
// n_per_row: 每行元素数 (必须是 256 的倍数)
// imatrix:   重要性矩阵 (NULL = 不用 imatrix,均匀量化;非 NULL = 每 per_row 个 float)
// 返回:      量化后总字节数,0 = 错误
size_t cgc_quantize_iq3_m(const float* src, void* dst,
                           int64_t nrow, int64_t n_per_row,
                           const float* imatrix);

// 把 BF16 数据转成 float (ggml 量化需要 float 输入)
// bf16:  输入 BF16 数据
// out:   输出 float 数组 (调用方分配)
// count: 元素数
void cgc_bf16_to_float(const void* bf16, float* out, size_t count);

// 把 FP16 数据转成 float
void cgc_fp16_to_float(const void* fp16, float* out, size_t count);

// 检查 ggml 是否可用 (编译时是否链接了 ggml)
// 返回: 1 = 可用, 0 = 不可用
int cgc_quantize_available(void);

#ifdef __cplusplus
}
#endif

#endif // CGC_QUANTIZE_H
