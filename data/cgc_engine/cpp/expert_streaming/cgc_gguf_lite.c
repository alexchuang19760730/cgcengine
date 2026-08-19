// cgc_gguf_lite.c — 轻量 GGUF v3 header 解析器实现
//
// 严格遵循标准 GGUF v3 格式:
//   magic(4) + version(4) + n_tensors(8) + n_kv(8)
//   + KV pairs (key:string + type:u32 + value)
//   + tensor info (name:string + n_dims:u32 + dims:u64[] + type:u32 + offset:u64)
//   + padding to 32-byte alignment
//   + tensor data (不加载)

#include "cgc_gguf_lite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 内部:文件读取辅助
// ============================================================================

static uint32_t read_u32(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64(FILE* f) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (i * 8);
    return v;
}

static int32_t read_i32(FILE* f) { return (int32_t)read_u32(f); }
static int64_t read_i64(FILE* f) { return (int64_t)read_u64(f); }

static float read_f32(FILE* f) {
    uint32_t u = read_u32(f);
    float f_val;
    memcpy(&f_val, &u, sizeof(float));
    return f_val;
}

static double read_f64(FILE* f) {
    uint64_t u = read_u64(f);
    double d_val;
    memcpy(&d_val, &u, sizeof(double));
    return d_val;
}

// 读 GGUF string: u64 length + bytes (无 null terminator)
// 返回堆分配的 null-terminated string (调用方负责 free)
static char* read_string(FILE* f) {
    uint64_t len = read_u64(f);
    if (len > 65536) {
        // 异常长 string,跳过
        fseek(f, (long)len, SEEK_CUR);
        return NULL;
    }
    char* s = (char*)malloc((size_t)len + 1);
    if (!s) return NULL;
    if (len > 0) {
        if (fread(s, 1, (size_t)len, f) != (size_t)len) {
            free(s);
            return NULL;
        }
    }
    s[len] = '\0';
    return s;
}

// ============================================================================
// KV 解析
// ============================================================================

// 解析单个 KV pair,写入 kv struct
// 返回 0 = 成功, 非 0 = 错误
static int parse_kv(FILE* f, cgc_gguf_kv_t* kv) {
    long pos_before = ftell(f);
    
    char* key = read_string(f);
    if (!key) {
        fprintf(stderr, "[gguf_lite] read_string failed at pos %ld\n", pos_before);
        return 1;
    }

    // 截断 key 到 127 字符
    strncpy(kv->key, key, 127);
    kv->key[127] = '\0';
    free(key);

    uint32_t type = read_u32(f);
    kv->value_type = (int32_t)type;
    
    fprintf(stderr, "[gguf_lite] KV: key=%s type=%u pos=%ld\n", kv->key, type, pos_before);

    switch (type) {
        case CGC_GGUF_TYPE_UINT8:
        case CGC_GGUF_TYPE_INT8: {
            uint8_t b;
            if (fread(&b, 1, 1, f) != 1) return 4;
            kv->i64_val = (int64_t)b;
            kv->str_val[0] = '\0';
            break;
        }
        case CGC_GGUF_TYPE_UINT16:
        case CGC_GGUF_TYPE_INT16: {
            uint8_t b[2];
            if (fread(b, 1, 2, f) != 2) return 4;
            kv->i64_val = (int64_t)(b[0] | ((uint16_t)b[1] << 8));
            kv->str_val[0] = '\0';
            break;
        }
        case CGC_GGUF_TYPE_UINT32:
        case CGC_GGUF_TYPE_INT32: {
            kv->i64_val = (int64_t)read_i32(f);
            kv->str_val[0] = '\0';
            break;
        }
        case CGC_GGUF_TYPE_BOOL: {
            // GGUF bool 值占 1 字节 (0 或 1), 无 padding
            uint8_t b;
            if (fread(&b, 1, 1, f) != 1) return 4;
            kv->i64_val = (int64_t)b;
            kv->str_val[0] = '\0';
            break;
        }
        case CGC_GGUF_TYPE_FLOAT32:
            kv->f64_val = (double)read_f32(f);
            kv->str_val[0] = '\0';
            break;
        case CGC_GGUF_TYPE_UINT64:
        case CGC_GGUF_TYPE_INT64:
            kv->i64_val = read_i64(f);
            kv->str_val[0] = '\0';
            break;
        case CGC_GGUF_TYPE_FLOAT64:
            kv->f64_val = read_f64(f);
            kv->str_val[0] = '\0';
            break;
        case CGC_GGUF_TYPE_STRING: {
            char* val = read_string(f);
            if (val) {
                strncpy(kv->str_val, val, 511);
                kv->str_val[511] = '\0';
                free(val);
            } else {
                kv->str_val[0] = '\0';
            }
            break;
        }
        case CGC_GGUF_TYPE_ARRAY: {
            // GGUF 中 ARRAY 的存储格式: elem_type (u32) 在前, count (u64) 在后
            // 与 llama.cpp gguf.h 的 gguf_get_arr_type/gguf_get_arr_n 一致
            // 字符串数组元素之间是紧凑排列的, 无 4 字节对齐
            uint32_t elem_type = read_u32(f);
            uint64_t n = read_u64(f);

            if (n > 10000) {
                // 大数组: 快速跳过
                size_t elem_size;
                switch (elem_type) {
                    case CGC_GGUF_TYPE_UINT8:
                    case CGC_GGUF_TYPE_INT8:
                    case CGC_GGUF_TYPE_BOOL:
                        elem_size = 1; break;
                    case CGC_GGUF_TYPE_UINT16:
                    case CGC_GGUF_TYPE_INT16:
                        elem_size = 2; break;
                    case CGC_GGUF_TYPE_UINT32:
                    case CGC_GGUF_TYPE_INT32:
                    case CGC_GGUF_TYPE_FLOAT32:
                        elem_size = 4; break;
                    case CGC_GGUF_TYPE_UINT64:
                    case CGC_GGUF_TYPE_INT64:
                    case CGC_GGUF_TYPE_FLOAT64:
                        elem_size = 8; break;
                    default:
                        elem_size = 4; break;
                }

                if (elem_type == CGC_GGUF_TYPE_STRING) {
                    for (uint64_t i = 0; i < n; i++) {
                        char* s = read_string(f);
                        if (s) free(s);
                    }
                } else {
                    fseek(f, (long)(n * elem_size), SEEK_CUR);
                }

                kv->str_val[0] = '\0';
                break;
            }

            for (uint64_t i = 0; i < n; i++) {
                switch (elem_type) {
                    case CGC_GGUF_TYPE_UINT8:
                    case CGC_GGUF_TYPE_INT8:
                    case CGC_GGUF_TYPE_BOOL:
                        fseek(f, 1, SEEK_CUR); break;
                    case CGC_GGUF_TYPE_UINT16:
                    case CGC_GGUF_TYPE_INT16:
                        fseek(f, 2, SEEK_CUR); break;
                    case CGC_GGUF_TYPE_UINT32:
                    case CGC_GGUF_TYPE_INT32:
                    case CGC_GGUF_TYPE_FLOAT32:
                        fseek(f, 4, SEEK_CUR); break;
                    case CGC_GGUF_TYPE_UINT64:
                    case CGC_GGUF_TYPE_INT64:
                    case CGC_GGUF_TYPE_FLOAT64:
                        fseek(f, 8, SEEK_CUR); break;
                    case CGC_GGUF_TYPE_STRING: {
                        char* s = read_string(f);
                        if (s) free(s);
                        break;
                    }
                    default:
                        return 2;
                }
            }
            kv->str_val[0] = '\0';
            break;
        }
        default:
            // 未知类型
            return 3;
    }
    return 0;
}

// ============================================================================
// API 实现
// ============================================================================

cgc_gguf_lite_ctx_t* cgc_gguf_lite_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    cgc_gguf_lite_ctx_t* ctx = (cgc_gguf_lite_ctx_t*)calloc(1, sizeof(cgc_gguf_lite_ctx_t));
    if (!ctx) { fclose(f); return NULL; }

    // magic
    uint32_t magic = read_u32(f);
    if (magic != CGC_GGUF_LITE_MAGIC) {
        fprintf(stderr, "[gguf_lite] bad magic: 0x%08X (expected 0x%08X)\n", magic, CGC_GGUF_LITE_MAGIC);
        fclose(f);
        free(ctx);
        return NULL;
    }

    // version
    ctx->version = read_u32(f);
    if (ctx->version != 3) {
        fprintf(stderr, "[gguf_lite] unsupported version: %u (expected 3)\n", ctx->version);
        // 继续尝试 (version 2 格式类似)
    }

    // n_tensors, n_kv
    ctx->n_tensors = read_u64(f);
    ctx->n_kv = read_u64(f);

    if (ctx->n_kv > 1024) {
        fprintf(stderr, "[gguf_lite] too many KV pairs: %llu\n", (unsigned long long)ctx->n_kv);
        fclose(f);
        free(ctx);
        return NULL;
    }

    // 读 KV pairs
    if (ctx->n_kv > 0) {
        ctx->kvs = (cgc_gguf_kv_t*)calloc(ctx->n_kv, sizeof(cgc_gguf_kv_t));
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            if (parse_kv(f, &ctx->kvs[i]) != 0) {
                fprintf(stderr, "[gguf_lite] failed to parse KV #%llu\n", (unsigned long long)i);
                cgc_gguf_lite_free(ctx);
                fclose(f);
                return NULL;
            }
        }
    }

    // 记录 tensor info 开始位置
    long tensor_info_start = ftell(f);

    // 读 tensor info
    if (ctx->n_tensors > 0) {
        ctx->tensors = (cgc_gguf_tensor_info_t*)calloc(ctx->n_tensors, sizeof(cgc_gguf_tensor_info_t));
        ctx->tensor_names = (char**)calloc(ctx->n_tensors, sizeof(char*));

        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            char* name = read_string(f);
            ctx->tensor_names[i] = name;  // 可能为 NULL

            cgc_gguf_tensor_info_t* info = &ctx->tensors[i];
            info->n_dims = (int32_t)read_u32(f);
            if (info->n_dims < 0 || info->n_dims > 4) {
                fprintf(stderr, "[gguf_lite] tensor %llu: bad n_dims=%d\n",
                        (unsigned long long)i, info->n_dims);
                cgc_gguf_lite_free(ctx);
                fclose(f);
                return NULL;
            }
            info->n_elements = 1;
            for (int j = 0; j < info->n_dims; j++) {
                info->dims[j] = (int64_t)read_u64(f);
                if (info->dims[j] > 0) info->n_elements *= (uint64_t)info->dims[j];
            }
            info->type = (int32_t)read_u32(f);
            info->offset = read_u64(f);
        }
    }

    // 计算 data_start: tensor info 结束位置,对齐到 32 字节
    long after_tensor_info = ftell(f);
    long aligned = (after_tensor_info + 31) & ~31L;
    ctx->data_start = (uint64_t)aligned;

    // 不关闭 file (保持打开供后续读 data,虽然 lite 模式通常不需要)
    // 实际上 lite 模式不需要 file handle,关闭它
    fclose(f);

    (void)tensor_info_start;  // 调试用
    return ctx;
}

void cgc_gguf_lite_free(cgc_gguf_lite_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->tensors) free(ctx->tensors);
    if (ctx->tensor_names) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            if (ctx->tensor_names[i]) free(ctx->tensor_names[i]);
        }
        free(ctx->tensor_names);
    }
    if (ctx->kvs) free(ctx->kvs);
    free(ctx);
}

// ============================================================================
// KV 查询
// ============================================================================

static const cgc_gguf_kv_t* find_kv(const cgc_gguf_lite_ctx_t* ctx, const char* key) {
    if (!ctx || !ctx->kvs) return NULL;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0) return &ctx->kvs[i];
    }
    return NULL;
}

const char* cgc_gguf_lite_get_str(const cgc_gguf_lite_ctx_t* ctx, const char* key) {
    const cgc_gguf_kv_t* kv = find_kv(ctx, key);
    if (!kv || kv->value_type != CGC_GGUF_TYPE_STRING) return NULL;
    return kv->str_val;
}

bool cgc_gguf_lite_get_i32(const cgc_gguf_lite_ctx_t* ctx, const char* key, int32_t* out) {
    const cgc_gguf_kv_t* kv = find_kv(ctx, key);
    if (!kv) return false;
    switch (kv->value_type) {
        case CGC_GGUF_TYPE_INT32:
        case CGC_GGUF_TYPE_UINT32:
        case CGC_GGUF_TYPE_BOOL:
        case CGC_GGUF_TYPE_INT16:
        case CGC_GGUF_TYPE_UINT16:
        case CGC_GGUF_TYPE_INT8:
        case CGC_GGUF_TYPE_UINT8:
            *out = (int32_t)kv->i64_val;
            return true;
        default:
            return false;
    }
}

bool cgc_gguf_lite_get_u32(const cgc_gguf_lite_ctx_t* ctx, const char* key, uint32_t* out) {
    int32_t v;
    if (!cgc_gguf_lite_get_i32(ctx, key, &v)) return false;
    *out = (uint32_t)v;
    return true;
}

bool cgc_gguf_lite_get_i64(const cgc_gguf_lite_ctx_t* ctx, const char* key, int64_t* out) {
    const cgc_gguf_kv_t* kv = find_kv(ctx, key);
    if (!kv) return false;
    switch (kv->value_type) {
        case CGC_GGUF_TYPE_INT64:
        case CGC_GGUF_TYPE_UINT64:
        case CGC_GGUF_TYPE_INT32:
        case CGC_GGUF_TYPE_UINT32:
        case CGC_GGUF_TYPE_BOOL:
        case CGC_GGUF_TYPE_INT16:
        case CGC_GGUF_TYPE_UINT16:
        case CGC_GGUF_TYPE_INT8:
        case CGC_GGUF_TYPE_UINT8:
            *out = kv->i64_val;
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Tensor 查询
// ============================================================================

int cgc_gguf_lite_find_tensor(const cgc_gguf_lite_ctx_t* ctx, const char* name) {
    if (!ctx || !ctx->tensor_names) return -1;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensor_names[i] && strcmp(ctx->tensor_names[i], name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

// ============================================================================
// ggml type → bytes per element
// ============================================================================

double cgc_ggml_type_bytes_per_elem(int32_t ggml_type) {
    switch (ggml_type) {
        case CGC_GGML_TYPE_F32:  return 4.0;
        case CGC_GGML_TYPE_F16:  return 2.0;
        case CGC_GGML_TYPE_BF16: return 2.0;

        case CGC_GGML_TYPE_Q2_K: return 84.0 / 256.0;
        case CGC_GGML_TYPE_Q3_K: return 110.0 / 256.0;
        case CGC_GGML_TYPE_Q4_K: return 144.0 / 256.0;
        case CGC_GGML_TYPE_Q5_K: return 176.0 / 256.0;
        case CGC_GGML_TYPE_Q6_K: return 210.0 / 256.0;
        case CGC_GGML_TYPE_Q8_K: return 292.0 / 256.0;

        case CGC_GGML_TYPE_Q4_0: return 18.0 / 32.0;
        case CGC_GGML_TYPE_Q4_1: return 20.0 / 32.0;
        case CGC_GGML_TYPE_Q5_0: return 22.0 / 32.0;
        case CGC_GGML_TYPE_Q5_1: return 24.0 / 32.0;
        case CGC_GGML_TYPE_Q8_0: return 34.0 / 32.0;

        case CGC_GGML_TYPE_IQ3_S:   return 110.0 / 256.0;
        case CGC_GGML_TYPE_IQ2_XXS: return 66.0 / 256.0;
        case CGC_GGML_TYPE_IQ2_XS:  return 74.0 / 256.0;
        case CGC_GGML_TYPE_IQ3_XXS: return 98.0 / 256.0;
        case CGC_GGML_TYPE_IQ1_S:   return 50.0 / 256.0;
        case CGC_GGML_TYPE_IQ4_NL:  return 18.0 / 32.0;
        case CGC_GGML_TYPE_IQ2_S:   return 82.0 / 256.0;
        case CGC_GGML_TYPE_IQ4_XS:  return 136.0 / 256.0;
        case CGC_GGML_TYPE_IQ1_M:   return 56.0 / 256.0;

        default: return 0.0;
    }
}
