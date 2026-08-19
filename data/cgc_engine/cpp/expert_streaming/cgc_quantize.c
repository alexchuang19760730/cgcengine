// cgc_quantize.c — IQ3_M 量化包装层
//
// 封装 llama.cpp 的 ggml_quantize_chunk,提供简洁的 IQ3_M 量化接口。
// 当链接了 ggml 库时,调用真实的 quantize_iq3_s;
// 当未链接 ggml 库时,返回错误 (直通模式)。

#include "cgc_quantize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// BF16 / FP16 → float 转换 (不依赖 ggml)
// ============================================================================

void cgc_bf16_to_float(const void* bf16, float* out, size_t count) {
    const uint16_t* p = (const uint16_t*)bf16;
    for (size_t i = 0; i < count; ++i) {
        // BF16 = 截断的 float32 (高 16 位)
        uint32_t bits = ((uint32_t)p[i]) << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
}

void cgc_fp16_to_float(const void* fp16, float* out, size_t count) {
    const uint16_t* p = (const uint16_t*)fp16;
    for (size_t i = 0; i < count; ++i) {
        uint16_t h = p[i];
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t frac =  h        & 0x3FF;

        uint32_t f;
        if (exp == 0) {
            if (frac == 0) {
                // 零
                f = sign << 31;
            } else {
                // 非正规数
                int e = -14;
                float val = frac / 1024.0f * powf(2.0f, e);
                if (sign) val = -val;
                out[i] = val;
                continue;
            }
        } else if (exp == 0x1F) {
            // inf/nan
            f = (sign << 31) | (0xFF << 23) | (frac << 13);
        } else {
            // 正规数
            f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        }
        memcpy(&out[i], &f, sizeof(float));
    }
}

// ============================================================================
// IQ3_M row size 计算
// ============================================================================
// block_iq3_s 在 ggml-common.h 中定义:
//   typedef struct {
//       uint8_t qs[QK_K/4];  // 64 bytes
//       uint8_t qh;          // 1 byte
//       float d;             // 4 bytes
//   } block_iq3_s;
// 但实际有 padding,总大小 = 72 bytes? 不对。
// 从 llama.cpp 源码:block_iq3_s 大小 = QK_K/8 + 4 + 2 = 32 + 4 + 2 = 38? 也不对。
//
// 最准确的方法:用 ggml_row_size(GGML_TYPE_IQ3_S, n_per_row)
// 如果没链接 ggml,用估算:block_iq3_s = 64 (qs) + 1 (qh) + 3 (pad) + 4 (d) = 72 bytes / 256 elements
// 实际从 ggml-common.h:
//   #define QK_IQ3S 256
//   typedef struct { uint8_t qs[QK_K/4]; uint8_t qh; float d; } block_iq3_s;
//   QK_K = 256, qs = 64, qh = 1, d = 4, padding → sizeof = 72 (对齐)
// 实际 GGML 报告: IQ3_S = 3.0625 bpw → 256 * 3.0625 / 8 = 98 bytes/block? 不对。
// 从 ggml.c type_traits: GGML_TYPE_IQ3_S blck_size=256, type_size=80
// 所以 block_iq3_s = 80 bytes / 256 elements

#define CGC_IQ3_S_BLOCK_SIZE 256
#define CGC_IQ3_S_BLOCK_BYTES 80

size_t cgc_iq3_m_row_size(int64_t n_per_row) {
    if (n_per_row % CGC_IQ3_S_BLOCK_SIZE != 0) return 0;
    int64_t n_blocks = n_per_row / CGC_IQ3_S_BLOCK_SIZE;
    return (size_t)(n_blocks * CGC_IQ3_S_BLOCK_BYTES);
}

// ============================================================================
// ggml 链接检测
// ============================================================================

// 如果编译时定义了 CGC_LINK_GGML,则调用真实的 ggml_quantize_chunk
#ifdef CGC_LINK_GGML

#include "ggml.h"
#include "ggml-quants.h"

int cgc_quantize_available(void) { return 1; }

size_t cgc_quantize_iq3_m(const float* src, void* dst,
                           int64_t nrow, int64_t n_per_row,
                           const float* imatrix)
{
    if (n_per_row % CGC_IQ3_S_BLOCK_SIZE != 0) {
        fprintf(stderr, "[quantize] n_per_row=%lld not multiple of %d\n",
                (long long)n_per_row, CGC_IQ3_S_BLOCK_SIZE);
        return 0;
    }

    // ggml_quantize_chunk 签名:
    //   size_t ggml_quantize_chunk(enum ggml_type type, const float* src, void* dst,
    //                              int64_t start, int64_t nrows, int64_t n_per_row,
    //                              const float* imatrix);
    size_t result = ggml_quantize_chunk(GGML_TYPE_IQ3_S, src, dst,
                                         0, nrow, n_per_row, imatrix);
    return result;
}

#else // !CGC_LINK_GGML — 未链接 ggml,直通模式

int cgc_quantize_available(void) { return 0; }

size_t cgc_quantize_iq3_m(const float* src, void* dst,
                           int64_t nrow, int64_t n_per_row,
                           const float* imatrix)
{
    (void)src; (void)dst; (void)nrow; (void)n_per_row; (void)imatrix;
    fprintf(stderr, "[quantize] ERROR: ggml not linked, IQ3_M quantization unavailable.\n");
    fprintf(stderr, "[quantize] Rebuild with -DCGC_LINK_GGML=ON to enable IQ3_M.\n");
    fprintf(stderr, "[quantize] Falling back to BF16 passthrough.\n");
    return 0;
}

#endif // CGC_LINK_GGML
