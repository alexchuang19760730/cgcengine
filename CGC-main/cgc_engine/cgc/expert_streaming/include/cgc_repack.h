// cgc_repack.h — GGUF + imatrix 3-bit repack (C 实现)
//
// 移植自 turbo-fieldfare 的 RepackPlanner + ResidentWriter + LayerFilePlan,
// 改用 GGUF 格式 + IQ3_M 量化,生成 per-layer 文件供 expert streamer 使用。
//
// 流程:
//   1. 读 safetensors header  → SourceTensor[]
//   2. 读 config.json          → ArchInfo
//   3. 分类 tensor             → lmResident / routedExpert(layer, role)
//   4. 生成 per-layer 布局     → LayerFilePlan (per-expert offset/stride)
//   5. IQ3_M 量化 (可选 imatrix) → 量化后字节
//   6. 写 per-layer GGUF 文件   → layer_N.gguf
//   7. 写 manifest.json         → 文件列表 + arch info

#ifndef CGC_REPACK_H
#define CGC_REPACK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 基础类型
// ============================================================================

typedef enum {
    CGC_DTYPE_U32  = 0,
    CGC_DTYPE_BF16 = 1,
    CGC_DTYPE_FP16 = 2,
    CGC_DTYPE_FP32 = 3,
} cgc_dtype_t;

static inline uint32_t cgc_dtype_element_bytes(cgc_dtype_t dt) {
    switch (dt) {
        case CGC_DTYPE_U32:  return 4;
        case CGC_DTYPE_BF16: return 2;
        case CGC_DTYPE_FP16: return 2;
        case CGC_DTYPE_FP32: return 4;
        default: return 0;
    }
}

// ============================================================================
// SourceTensor — 从 safetensors header 解析出来的一个 tensor
// ============================================================================

typedef struct {
    char name[256];          // tensor 名 (如 "language_model.layers.0.experts.switch_glu.gate_proj.weight")
    char shard_path[512];    // 所在 safetensors 文件路径
    cgc_dtype_t dtype;       // BF16 / FP16 / FP32
    uint32_t n_dims;         // 维度数 (1-4)
    uint64_t shape[4];       // shape (n_dims 个有效)
    uint64_t absolute_offset; // 在 shard 文件中的绝对字节偏移
    uint64_t size_bytes;     // 字节大小
} cgc_source_tensor_t;

// ============================================================================
// ArchInfo — 从 config.json 解析出来的架构信息
// ============================================================================

typedef struct {
    int hidden_size;
    int intermediate_size;        // shared expert FFN
    int moe_intermediate_size;    // per-expert FFN
    int num_heads;
    int num_kv_heads;
    int num_full_kv_heads;
    int head_dim;
    int full_head_dim;
    int vocab_size;
    int sliding_window;
    double final_logit_softcap;
    double rope_theta;
    double full_rope_theta;
    double partial_rotary_factor;
    int num_layers;
    int num_experts;
    int top_k_experts;
    bool tie_word_embeddings;
    bool attention_k_eq_v;
    char hidden_activation[64];   // "gelu_pytorch_tanh" 等
} cgc_arch_info_t;

// ============================================================================
// Bucket — tensor 分类
// ============================================================================

typedef enum {
    CGC_BUCKET_LM_RESIDENT = 0,       // 非 expert 的 LM 权重
    CGC_BUCKET_ROUTED_EXPERT = 1,     // expert 权重 (gate/up/down)
    CGC_BUCKET_EXCLUDED_MULTIMODAL = 2,
    CGC_BUCKET_UNKNOWN = 3,
} cgc_bucket_kind_t;

typedef struct {
    cgc_bucket_kind_t kind;
    char role[16];   // "gate" / "up" / "down" (routed expert)
    int layer;       // layer index (routed expert)
} cgc_bucket_t;

// ============================================================================
// LayerFilePlan — per-layer 文件布局
// ============================================================================

typedef struct {
    char role[16];          // "gate" / "up" / "down"
    char component[16];     // "weights" / "scales" / "biases"
    cgc_dtype_t dtype;
    uint32_t n_dims;
    uint64_t logical_shape[4];     // per-expert logical shape
    uint64_t offset_in_expert_blob; // 在每个 expert blob 内的偏移
    uint64_t size_in_expert_blob;  // 在每个 expert blob 内的大小
    uint64_t source_offset_per_expert; // source 中每个 expert 的 stride
    int bits_for_weights;          // 3 for IQ3_M; 0 for scales/biases
} cgc_per_expert_tensor_slice_t;

typedef struct {
    int layer_index;
    char path[512];         // 输出 GGUF 文件路径 (如 "layer_0.gguf")
    int experts_per_layer;
    uint64_t expert_stride; // 每个 expert 的字节步长
    uint32_t n_sub_tensors;
    cgc_per_expert_tensor_slice_t sub_tensors[16]; // 9 个: gate/up/down × {weights, scales, biases}
    int physical_order[256]; // logical → physical 映射 (可重排)
} cgc_layer_file_plan_t;

// ============================================================================
// RepackPlan — 完整的 repack 计划
// ============================================================================

typedef struct {
    cgc_arch_info_t arch;
    int n_layers;
    cgc_layer_file_plan_t* layers;  // n_layers 个
    char manifest_path[512];
    char output_dir[512];
} cgc_repack_plan_t;

// ============================================================================
// RepackOptions
// ============================================================================

typedef struct {
    const char* input_dir;       // HF 模型目录 (含 safetensors + config.json)
    const char* output_dir;      // 输出目录
    const char* imatrix_path;    // 可选 imatrix 文件路径 (NULL = 不用 imatrix)
    int quant_bits;              // 3 = IQ3_M
    bool dry_run;                // 只打印计划,不写文件
    bool overwrite;              // 覆盖已有输出
    int n_threads;               // 量化线程数
} cgc_repack_options_t;

// ============================================================================
// API
// ============================================================================

// 主入口:执行完整 repack
// 返回 0 = 成功, 非 0 = 错误码
int cgc_repack_run(const cgc_repack_options_t* opts);

// 解析 safetensors header
// 返回 tensor 数组 (调用方负责 free),*out_count = tensor 数量
cgc_source_tensor_t* cgc_safetensors_parse(const char* shard_path,
                                            int* out_count);

// 扫描目录下所有 safetensors 文件并合并解析
cgc_source_tensor_t* cgc_safetensors_scan_dir(const char* dir,
                                               int* out_count);

// 解析 config.json
int cgc_arch_info_load(const char* config_path, cgc_arch_info_t* out_arch);

// 分类 tensor
cgc_bucket_t cgc_classify_tensor(const char* name, int num_layers);

// 生成 repack 计划
int cgc_repack_plan_create(const cgc_source_tensor_t* tensors, int n_tensors,
                            const cgc_arch_info_t* arch,
                            const char* output_dir,
                            cgc_repack_plan_t* out_plan);

void cgc_repack_plan_free(cgc_repack_plan_t* plan);

// 写 per-layer GGUF 文件
int cgc_write_layer_gguf(const cgc_repack_plan_t* plan,
                         int layer_idx,
                         const cgc_source_tensor_t* tensors, int n_tensors,
                         const cgc_repack_options_t* opts);

// 写 manifest.json
int cgc_write_manifest(const cgc_repack_plan_t* plan,
                       const cgc_repack_options_t* opts);

#ifdef __cplusplus
}
#endif

#endif // CGC_REPACK_H
