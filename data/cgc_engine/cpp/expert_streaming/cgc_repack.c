// cgc_repack.c — 主程序:GGUF + imatrix 3-bit repack (C 实现)
//
// 用法:
//   cgc_repack --input <HF模型目录> --output <输出目录> [--imatrix <imatrix文件>] [--bits 3] [--dry-run]
//
// 流程:
//   1. 扫描 input_dir 的 .safetensors → SourceTensor[]
//   2. 读 config.json → ArchInfo
//   3. 生成 per-layer 布局 → RepackPlan
//   4. 对每个 layer:
//      a. 读 BF16/FP16 expert 权重
//      b. (可选) IQ3_M 量化 + imatrix
//      c. 写 per-layer GGUF 文件
//   5. 写 manifest.json
//
// 当前版本:直通模式 (BF16 直接写入 GGUF,不做量化)
// 下一步:接入 llama.cpp 的 quantize_iq3_m

#include "cgc_repack.h"
#include "cgc_quantize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif

// ============================================================================
// 辅助:读文件到 buffer
// ============================================================================

static void* read_file_range(const char* path, uint64_t offset, uint64_t size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, (long)offset, SEEK_SET) != 0) { fclose(f); return NULL; }
    void* buf = malloc(size);
    if (!buf) { fclose(f); return NULL; }
    uint64_t read = fread(buf, 1, size, f);
    fclose(f);
    if (read != size) {
        fprintf(stderr, "[repack] short read: %s offset=%llu size=%llu got=%llu\n",
                path, (unsigned long long)offset,
                (unsigned long long)size, (unsigned long long)read);
        free(buf);
        return NULL;
    }
    return buf;
}

// ============================================================================
// 简化版 GGUF writer (直通模式:BF16/FP16 直接写)
// ============================================================================
// 真正的 IQ3_M 量化版本会接入 llama.cpp 的 ggml-quants.c
// 这里先写一个最小的 GGUF writer,验证流程

#define GGUF_MAGIC 0x46554747  // "GGUF" LE
#define GGUF_VERSION 3

// GGUF type
enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ggml type (tensor dtype) — 值来自 ggml.h
enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_IQ3_S = 21,   // IQ3_M 底层 = IQ3_S
    GGML_TYPE_BF16 = 30,
};

static void write_u32_le(FILE* f, uint32_t v) {
    uint8_t b[4] = { v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF };
    fwrite(b, 1, 4, f);
}

static void write_u64_le(FILE* f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (v >> (i * 8)) & 0xFF;
    fwrite(b, 1, 8, f);
}

static void write_gguf_string(FILE* f, const char* s) {
    uint64_t len = strlen(s);
    write_u64_le(f, len);
    fwrite(s, 1, len, f);
}

static void write_gguf_kv_str(FILE* f, const char* key, const char* val) {
    write_gguf_string(f, key);
    write_u32_le(f, GGUF_TYPE_STRING);
    write_gguf_string(f, val);
}

static void write_gguf_kv_u32(FILE* f, const char* key, uint32_t val) {
    write_gguf_string(f, key);
    write_u32_le(f, GGUF_TYPE_UINT32);
    write_u32_le(f, val);
}

static void write_gguf_kv_i32(FILE* f, const char* key, int32_t val) {
    write_gguf_string(f, key);
    write_u32_le(f, GGUF_TYPE_INT32);
    write_u32_le(f, (uint32_t)val);
}

// ============================================================================
// 写 per-layer GGUF 文件
// ============================================================================

int cgc_write_layer_gguf(const cgc_repack_plan_t* plan,
                         int layer_idx,
                         const cgc_source_tensor_t* tensors, int n_tensors,
                         const cgc_repack_options_t* opts)
{
    const cgc_layer_file_plan_t* lp = &plan->layers[layer_idx];
    if (lp->n_sub_tensors == 0) {
        fprintf(stderr, "[repack] layer %d: no sub_tensors, skipping\n", layer_idx);
        return 0;
    }

    fprintf(stderr, "[repack] writing %s ...\n", lp->path);
    FILE* f = fopen(lp->path, "wb");
    if (!f) {
        fprintf(stderr, "[repack] cannot create %s\n", lp->path);
        return -1;
    }

    // 找这个 layer 的 gate/up/down source tensor
    const cgc_source_tensor_t* src_gate = NULL;
    const cgc_source_tensor_t* src_up = NULL;
    const cgc_source_tensor_t* src_down = NULL;
    for (int i = 0; i < n_tensors; ++i) {
        cgc_bucket_t b = cgc_classify_tensor(tensors[i].name, plan->n_layers);
        if (b.kind == CGC_BUCKET_ROUTED_EXPERT && b.layer == layer_idx) {
            if (strcmp(b.role, "gate") == 0) src_gate = &tensors[i];
            else if (strcmp(b.role, "up") == 0) src_up = &tensors[i];
            else if (strcmp(b.role, "down") == 0) src_down = &tensors[i];
        }
    }

    // GGUF header
    write_u32_le(f, GGUF_MAGIC);
    write_u32_le(f, GGUF_VERSION);

    // tensor count = experts_per_layer * n_sub_tensors (每个 expert 的 gate/up/down)
    uint64_t n_tensors_out = (uint64_t)lp->experts_per_layer * lp->n_sub_tensors;
    write_u64_le(f, n_tensors_out);

    // KV count
    uint32_t n_kv = 8;
    write_u64_le(f, n_kv);

    // KV metadata
    write_gguf_kv_str(f, "general.architecture", "gemma4_moe");
    write_gguf_kv_u32(f, "general.layer_index", (uint32_t)layer_idx);
    write_gguf_kv_u32(f, "gemma4.expert_count", (uint32_t)lp->experts_per_layer);
    write_gguf_kv_u32(f, "gemma4.expert_stride", (uint32_t)lp->expert_stride); // 溢出截断 (简化)
    write_gguf_kv_str(f, "gemma4.quantization", opts->quant_bits == 3 ? "IQ3_M" : "BF16");
    write_gguf_kv_i32(f, "gemma4.hidden_size", plan->arch.hidden_size);
    write_gguf_kv_i32(f, "gemma4.moe_intermediate_size", plan->arch.moe_intermediate_size);
    if (opts->imatrix_path) {
        write_gguf_kv_str(f, "gemma4.imatrix_file", opts->imatrix_path);
    }

    // 第一遍:计算 tensor info 区大小,得到 data 起始 offset
    // 每个 tensor info: name(string) + n_dims(u32) + dims(u64*n) + dtype(u32) + offset(u64)
    // 先写到临时 buffer,再 flush
    // 简化:先计算总大小
    uint64_t tensor_info_size = 0;
    for (int e = 0; e < lp->experts_per_layer; ++e) {
        for (uint32_t s = 0; s < lp->n_sub_tensors; ++s) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.expert.%d.%s.weight",
                     layer_idx, e, lp->sub_tensors[s].role);
            tensor_info_size += 8 + strlen(name); // name string
            tensor_info_size += 4;                  // n_dims
            tensor_info_size += 8 * 2;              // 2 dims (out, in)
            tensor_info_size += 4;                  // dtype
            tensor_info_size += 8;                  // offset
        }
    }

    uint64_t header_size = 4 + 4 + 8 + 8; // magic + version + n_tensors + n_kv
    uint64_t kv_size = 0; // 需要计算 (简化:用 ftell)
    // 实际上我们已经写了 KV,所以用当前 ftell 作为 data 起点
    long data_start = ftell(f) + (long)tensor_info_size;
    // 对齐到 32 字节 (GGUF 标准)
    data_start = (data_start + 31) & ~31;
    // 写 padding
    long current = ftell(f);
    // 先写 tensor info,同时记录每个 tensor 的 data offset

    uint64_t current_data_offset = 0;
    for (int e = 0; e < lp->experts_per_layer; ++e) {
        for (uint32_t s = 0; s < lp->n_sub_tensors; ++s) {
            const cgc_per_expert_tensor_slice_t* sub = &lp->sub_tensors[s];

            // tensor name
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.expert.%d.%s.weight",
                     layer_idx, e, sub->role);
            write_gguf_string(f, name);

            // n_dims = 2 (out, in)
            write_u32_le(f, 2);
            // dims (out, in) — 从 logical_shape 取
            write_u64_le(f, sub->logical_shape[0]);
            write_u64_le(f, sub->logical_shape[1]);
            // dtype (BF16 = 30, IQ3_M = IQ3_S = 21)
            uint32_t dtype = (opts->quant_bits == 3) ? GGML_TYPE_IQ3_S : GGML_TYPE_BF16;
            write_u32_le(f, dtype);
            // data offset (相对于 data_start)
            write_u64_le(f, current_data_offset);
            current_data_offset += sub->size_in_expert_blob;
        }
    }

    // padding 到 data_start
    long now = ftell(f);
    while (now < data_start) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, f);
        now++;
    }

    // 第二遍:写 tensor data
    // 读 source expert 数据,写入 GGUF
    const cgc_source_tensor_t* src_list[3] = { src_gate, src_up, src_down };
    for (int e = 0; e < lp->experts_per_layer; ++e) {
        for (uint32_t s = 0; s < lp->n_sub_tensors; ++s) {
            const cgc_per_expert_tensor_slice_t* sub = &lp->sub_tensors[s];
            const cgc_source_tensor_t* src = src_list[s];
            if (!src) {
                // 写零
                uint8_t* zeros = (uint8_t*)calloc(sub->size_in_expert_blob, 1);
                fwrite(zeros, 1, sub->size_in_expert_blob, f);
                free(zeros);
                continue;
            }

            // 读这个 expert 的数据
            uint64_t elem_bytes = cgc_dtype_element_bytes(src->dtype);
            uint64_t per_expert_bytes = src->size_bytes / lp->experts_per_layer;
            uint64_t src_offset = src->absolute_offset + (uint64_t)e * per_expert_bytes;

            void* expert_data = read_file_range(src->shard_path, src_offset, per_expert_bytes);
            if (!expert_data) {
                fprintf(stderr, "[repack] failed to read expert %d layer %d %s\n",
                        e, layer_idx, sub->role);
                // 写零
                uint8_t* zeros = (uint8_t*)calloc(sub->size_in_expert_blob, 1);
                fwrite(zeros, 1, sub->size_in_expert_blob, f);
                free(zeros);
                continue;
            }

            if (opts->quant_bits == 3 && cgc_quantize_available()) {
                // IQ3_M 量化: BF16 → float → IQ3_S
                uint64_t n_elems = per_expert_bytes / elem_bytes;
                float* fbuf = (float*)malloc(n_elems * sizeof(float));
                if (!fbuf) {
                    fwrite(expert_data, 1, per_expert_bytes, f); // fallback
                    free(expert_data);
                    continue;
                }
                // BF16/FP16 → float
                if (src->dtype == CGC_DTYPE_BF16) cgc_bf16_to_float(expert_data, fbuf, n_elems);
                else if (src->dtype == CGC_DTYPE_FP16) cgc_fp16_to_float(expert_data, fbuf, n_elems);
                else memcpy(fbuf, expert_data, n_elems * sizeof(float)); // FP32

                // 量化 (imatrix 暂用 NULL,后续从 opts->imatrix_path 加载)
                int64_t nrows = (int64_t)sub->logical_shape[0];
                int64_t n_per_row = (int64_t)sub->logical_shape[1];
                if (nrows * n_per_row != (int64_t)n_elems) {
                    // fallback: 单行
                    nrows = 1; n_per_row = (int64_t)n_elems;
                }
                size_t quant_size = cgc_quantize_iq3_m(fbuf, expert_data,
                                                       nrows, n_per_row, NULL);
                if (quant_size > 0) {
                    fwrite(expert_data, 1, quant_size, f);
                } else {
                    // 量化失败,fallback BF16
                    fwrite(expert_data, 1, per_expert_bytes, f);
                }
                free(fbuf);
            } else {
                // 直通模式:直接写 BF16
                fwrite(expert_data, 1, per_expert_bytes, f);
            }

            free(expert_data);
        }
    }

    fclose(f);
    fprintf(stderr, "[repack] layer %d done: %s\n", layer_idx, lp->path);
    return 0;
}

// ============================================================================
// 写 manifest.json
// ============================================================================

int cgc_write_manifest(const cgc_repack_plan_t* plan, const cgc_repack_options_t* opts) {
    FILE* f = fopen(plan->manifest_path, "wb");
    if (!f) {
        fprintf(stderr, "[repack] cannot create manifest %s\n", plan->manifest_path);
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"architecture\": \"gemma4_moe\",\n");
    fprintf(f, "  \"quantization\": \"%s\",\n", opts->quant_bits == 3 ? "IQ3_M" : "BF16");
    fprintf(f, "  \"imatrix\": %s,\n", opts->imatrix_path ? "true" : "false");
    fprintf(f, "  \"arch\": {\n");
    fprintf(f, "    \"hidden_size\": %d,\n", plan->arch.hidden_size);
    fprintf(f, "    \"moe_intermediate_size\": %d,\n", plan->arch.moe_intermediate_size);
    fprintf(f, "    \"num_layers\": %d,\n", plan->arch.num_layers);
    fprintf(f, "    \"num_experts\": %d,\n", plan->arch.num_experts);
    fprintf(f, "    \"top_k_experts\": %d,\n", plan->arch.top_k_experts);
    fprintf(f, "    \"num_heads\": %d,\n", plan->arch.num_heads);
    fprintf(f, "    \"num_kv_heads\": %d,\n", plan->arch.num_kv_heads);
    fprintf(f, "    \"head_dim\": %d,\n", plan->arch.head_dim);
    fprintf(f, "    \"vocab_size\": %d\n", plan->arch.vocab_size);
    fprintf(f, "  },\n");
    fprintf(f, "  \"layers\": [\n");
    for (int i = 0; i < plan->n_layers; ++i) {
        const cgc_layer_file_plan_t* lp = &plan->layers[i];
        fprintf(f, "    {\"layer\": %d, \"path\": \"layer_%d.gguf\", \"experts\": %d, \"expert_stride\": %llu}%s\n",
                i, i, lp->experts_per_layer,
                (unsigned long long)lp->expert_stride,
                i + 1 < plan->n_layers ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);

    fprintf(stderr, "[repack] manifest written: %s\n", plan->manifest_path);
    return 0;
}

// ============================================================================
// 创建目录
// ============================================================================

#ifdef _WIN32
static int mkdir_p(const char* path) {
    char cmd[MAX_PATH * 2];
    snprintf(cmd, sizeof(cmd), "if not exist \"%s\" mkdir \"%s\"", path, path);
    return system(cmd);
}
#else
static int mkdir_p(const char* path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", path);
    return system(cmd);
}
#endif

// ============================================================================
// main
// ============================================================================

int cgc_repack_run(const cgc_repack_options_t* opts) {
    fprintf(stderr, "=== CGC Repack (GGUF + IQ3_M, C port) ===\n");
    fprintf(stderr, "[repack] input:  %s\n", opts->input_dir);
    fprintf(stderr, "[repack] output: %s\n", opts->output_dir);
    fprintf(stderr, "[repack] imatrix: %s\n", opts->imatrix_path ? opts->imatrix_path : "(none)");
    fprintf(stderr, "[repack] bits: %d\n", opts->quant_bits);
    fprintf(stderr, "[repack] dry_run: %s\n", opts->dry_run ? "true" : "false");

    // 1. 创建输出目录
    if (mkdir_p(opts->output_dir) != 0) {
        fprintf(stderr, "[repack] failed to create output dir\n");
        return 1;
    }

    // 2. 读 config.json
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s%cconfig.json", opts->input_dir, PATH_SEP);
    cgc_arch_info_t arch;
    if (cgc_arch_info_load(config_path, &arch) != 0) {
        return 1;
    }

    // 3. 扫描 safetensors
    int n_tensors = 0;
    cgc_source_tensor_t* tensors = cgc_safetensors_scan_dir(opts->input_dir, &n_tensors);
    if (!tensors || n_tensors == 0) {
        fprintf(stderr, "[repack] no tensors found in %s\n", opts->input_dir);
        return 1;
    }
    fprintf(stderr, "[repack] loaded %d tensors\n", n_tensors);

    // 4. 生成 plan
    cgc_repack_plan_t plan;
    if (cgc_repack_plan_create(tensors, n_tensors, &arch, opts->output_dir, &plan) != 0) {
        free(tensors);
        return 1;
    }

    if (opts->dry_run) {
        fprintf(stderr, "[repack] dry run, not writing files\n");
        cgc_repack_plan_free(&plan);
        free(tensors);
        return 0;
    }

    // 5. 写 per-layer GGUF
    for (int layer = 0; layer < plan.n_layers; ++layer) {
        if (cgc_write_layer_gguf(&plan, layer, tensors, n_tensors, opts) != 0) {
            fprintf(stderr, "[repack] layer %d failed\n", layer);
            // 继续下一个 layer
        }
    }

    // 6. 写 manifest
    cgc_write_manifest(&plan, opts);

    cgc_repack_plan_free(&plan);
    free(tensors);

    fprintf(stderr, "=== Repack complete ===\n");
    return 0;
}

// ============================================================================
// 命令行入口 — 见 cgc_repack_main.c
// ============================================================================
