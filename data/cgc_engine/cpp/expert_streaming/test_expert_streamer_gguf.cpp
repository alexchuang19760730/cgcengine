// test_expert_streamer_gguf.cpp — 测试 GGUF layout 解析
//
// 测试策略:
//   1. 手动构造一个最小的 per-layer GGUF 文件 (2 experts × 3 sub-tensors = 6 tensors)
//   2. 用 loadStreamLayoutFromGGUF 解析,验证 StreamLayout 正确
//   3. 用 cgc_gguf_lite 验证 KV metadata 和 tensor info
//   4. 用 findExpertTensors 验证 expert tensor 查找
//
// GGUF 格式 (与 cgc_repack.c 输出匹配):
//   magic(4) + version(4) + n_tensors(8) + n_kv(8)
//   + KV pairs
//   + tensor info
//   + padding to 32-byte align
//   + tensor data

#include "expert_streamer_gguf.h"
#include "expert_compute.h"
#include "cgc_gguf_lite.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <windows.h>

// 模拟 cgc_repack.c 的 GGUF writer (最小化)
static void write_u32(FILE* f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
                     (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF) };
    fwrite(b, 1, 4, f);
}

static void write_u64(FILE* f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)((v >> (i * 8)) & 0xFF);
    fwrite(b, 1, 8, f);
}

static void write_i32(FILE* f, int32_t v) { write_u32(f, (uint32_t)v); }

static void write_string(FILE* f, const char* s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, (size_t)len, f);
}

static void write_kv_str(FILE* f, const char* key, const char* val) {
    write_string(f, key);
    write_u32(f, 8);  // CGC_GGUF_TYPE_STRING
    write_string(f, val);
}

static void write_kv_u32(FILE* f, const char* key, uint32_t val) {
    write_string(f, key);
    write_u32(f, 4);  // CGC_GGUF_TYPE_UINT32
    write_u32(f, val);
}

static void write_kv_i32(FILE* f, const char* key, int32_t val) {
    write_string(f, key);
    write_u32(f, 5);  // CGC_GGUF_TYPE_INT32
    write_i32(f, val);
}

// 创建测试 GGUF 文件
// 2 experts × 3 sub-tensors (gate/up/down),每个 tensor 4 bytes (F32, 1 element)
static std::string createTestGGUF() {
    char tmpPath[MAX_PATH];
    char tempDir[MAX_PATH];
    GetTempPathA(MAX_PATH, tempDir);
    GetTempFileNameA(tempDir, "cgc_gguf_test_", 0, tmpPath);
    std::string path = tmpPath;

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Failed to create temp file\n"); return ""; }

    const int n_experts = 2;
    const int n_sub = 3;
    const char* roles[] = { "gate", "up", "down" };
    const int tensor_data_size = 4;  // F32, 1 element = 4 bytes
    const int expert_stride = n_sub * tensor_data_size;  // 12 bytes per expert

    // Header
    write_u32(f, 0x46554747);  // magic "GGUF"
    write_u32(f, 3);           // version
    write_u64(f, (uint64_t)n_experts * n_sub);  // n_tensors = 6
    write_u64(f, 7);           // n_kv = 7

    // KV metadata (与 cgc_repack.c 一致)
    write_kv_str(f, "general.architecture", "gemma4_moe");
    write_kv_u32(f, "general.layer_index", 5);          // layer 5
    write_kv_u32(f, "gemma4.expert_count", n_experts);
    write_kv_u32(f, "gemma4.expert_stride", (uint32_t)expert_stride);
    write_kv_str(f, "gemma4.quantization", "BF16");
    write_kv_i32(f, "gemma4.hidden_size", 256);
    write_kv_i32(f, "gemma4.moe_intermediate_size", 512);

    // 计算 tensor info 区大小,得到 data_start
    // 每个 tensor info: name(string) + n_dims(u32) + dims(u64*n) + dtype(u32) + offset(u64)
    // name: "blk.5.expert.0.gate.weight" = 28 chars + 8 (length) = 36
    // n_dims: 4, dims: 2*8=16, dtype: 4, offset: 8 → 36 + 4 + 16 + 4 + 8 = 68
    // 6 tensors × 68 = 408 bytes (approximate, names vary slightly)

    // 先记录当前 offset (header + KV end)
    long after_kv = ftell(f);

    // 写 tensor info,同时记录 offset
    uint64_t current_data_offset = 0;
    for (int e = 0; e < n_experts; e++) {
        for (int s = 0; s < n_sub; s++) {
            char name[128];
            snprintf(name, sizeof(name), "blk.5.expert.%d.%s.weight", e, roles[s]);
            write_string(f, name);
            write_u32(f, 2);                 // n_dims = 2
            write_u64(f, 1);                 // dim[0] = 1 (out_dim)
            write_u64(f, 1);                 // dim[1] = 1 (in_dim)
            write_u32(f, 0);                 // dtype = F32
            write_u64(f, current_data_offset);  // offset
            current_data_offset += tensor_data_size;
        }
    }

    long after_tensor_info = ftell(f);
    // 对齐到 32 字节
    long aligned = (after_tensor_info + 31) & ~31L;
    long padding = aligned - after_tensor_info;
    for (long i = 0; i < padding; i++) fputc(0, f);

    // 写 tensor data (6 tensors × 4 bytes = 24 bytes,全零)
    for (int i = 0; i < n_experts * n_sub * tensor_data_size; i++) {
        fputc(0, f);
    }

    fclose(f);
    return path;
}

// ============================================================================
// Tests
// ============================================================================

static int test_gguf_lite_load(const std::string& path) {
    printf("\n=== Test 1: cgc_gguf_lite_load ===\n");
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(path.c_str());
    if (!ctx) { printf("FAIL: load returned NULL\n"); return 1; }

    printf("  version: %u\n", ctx->version);
    printf("  n_tensors: %llu\n", (unsigned long long)ctx->n_tensors);
    printf("  n_kv: %llu\n", (unsigned long long)ctx->n_kv);
    printf("  data_start: %llu\n", (unsigned long long)ctx->data_start);

    if (ctx->n_tensors != 6) { printf("FAIL: expected 6 tensors, got %llu\n", (unsigned long long)ctx->n_tensors); cgc_gguf_lite_free(ctx); return 1; }
    if (ctx->n_kv != 7) { printf("FAIL: expected 7 KV pairs, got %llu\n", (unsigned long long)ctx->n_kv); cgc_gguf_lite_free(ctx); return 1; }

    // 验证 KV
    uint32_t expertCount = 0;
    if (!cgc_gguf_lite_get_u32(ctx, "gemma4.expert_count", &expertCount)) {
        printf("FAIL: gemma4.expert_count not found\n"); cgc_gguf_lite_free(ctx); return 1;
    }
    printf("  gemma4.expert_count: %u\n", expertCount);
    if (expertCount != 2) { printf("FAIL: expected 2, got %u\n", expertCount); cgc_gguf_lite_free(ctx); return 1; }

    const char* quant = cgc_gguf_lite_get_str(ctx, "gemma4.quantization");
    printf("  gemma4.quantization: %s\n", quant ? quant : "(null)");
    if (!quant || strcmp(quant, "BF16") != 0) { printf("FAIL: expected BF16\n"); cgc_gguf_lite_free(ctx); return 1; }

    // 验证 tensor names
    printf("  tensors:\n");
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        printf("    [%llu] %s  type=%d offset=%llu\n",
               (unsigned long long)i,
               ctx->tensor_names[i] ? ctx->tensor_names[i] : "(null)",
               ctx->tensors[i].type,
               (unsigned long long)ctx->tensors[i].offset);
    }

    cgc_gguf_lite_free(ctx);
    printf("PASS\n");
    return 0;
}

static int test_loadStreamLayout(const std::string& path) {
    printf("\n=== Test 2: loadStreamLayoutFromGGUF ===\n");
    cgc_moe::StreamLayout layout = cgc_moe::loadStreamLayoutFromGGUF(path);

    printf("  path: %s\n", layout.path.c_str());
    printf("  streamOffset: %llu\n", (unsigned long long)layout.streamOffset);
    printf("  streamSize: %llu\n", (unsigned long long)layout.streamSize);
    printf("  expertsPerLayer: %d\n", layout.expertsPerLayer);
    printf("  expertStride: %llu\n", (unsigned long long)layout.expertStride);

    if (layout.expertsPerLayer != 2) { printf("FAIL: expected 2 experts, got %d\n", layout.expertsPerLayer); return 1; }
    if (layout.expertStride != 12) { printf("FAIL: expected stride 12, got %llu\n", (unsigned long long)layout.expertStride); return 1; }
    if (layout.streamSize != 24) { printf("FAIL: expected streamSize 24, got %llu\n", (unsigned long long)layout.streamSize); return 1; }
    if (layout.streamOffset == 0) { printf("FAIL: streamOffset is 0\n"); return 1; }

    printf("PASS\n");
    return 0;
}

static int test_findExpertTensors(const std::string& path) {
    printf("\n=== Test 3: findExpertTensors ===\n");
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(path.c_str());
    if (!ctx) { printf("FAIL: load failed\n"); return 1; }

    // expert 0 应有 3 个 sub-tensor (gate/up/down)
    auto tensors = cgc_moe::findExpertTensors(ctx, 0);
    if (tensors.size() != 3) { printf("FAIL: expected 3 tensors for expert 0, got %zu\n", tensors.size()); cgc_gguf_lite_free(ctx); return 1; }

    printf("  expert 0 tensors:\n");
    for (const auto& t : tensors) {
        printf("    role=%s type=%d offset=%llu sizeBytes=%llu\n",
               t.role.c_str(), t.ggmlType,
               (unsigned long long)t.offset, (unsigned long long)t.sizeBytes);
    }

    // 验证 role 顺序
    if (tensors[0].role != "gate") { printf("FAIL: expected gate, got %s\n", tensors[0].role.c_str()); cgc_gguf_lite_free(ctx); return 1; }
    if (tensors[1].role != "up") { printf("FAIL: expected up, got %s\n", tensors[1].role.c_str()); cgc_gguf_lite_free(ctx); return 1; }
    if (tensors[2].role != "down") { printf("FAIL: expected down, got %s\n", tensors[2].role.c_str()); cgc_gguf_lite_free(ctx); return 1; }

    // 验证 offset
    if (tensors[0].offset != 0) { printf("FAIL: gate offset should be 0\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (tensors[1].offset != 4) { printf("FAIL: up offset should be 4\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (tensors[2].offset != 8) { printf("FAIL: down offset should be 8\n"); cgc_gguf_lite_free(ctx); return 1; }

    // expert 1
    auto tensors1 = cgc_moe::findExpertTensors(ctx, 1);
    if (tensors1.size() != 3) { printf("FAIL: expected 3 tensors for expert 1, got %zu\n", tensors1.size()); cgc_gguf_lite_free(ctx); return 1; }
    if (tensors1[0].offset != 12) { printf("FAIL: expert 1 gate offset should be 12, got %llu\n", (unsigned long long)tensors1[0].offset); cgc_gguf_lite_free(ctx); return 1; }

    cgc_gguf_lite_free(ctx);
    printf("PASS\n");
    return 0;
}

static int test_parseLayerGGUFMeta(const std::string& path) {
    printf("\n=== Test 4: parseLayerGGUFMeta ===\n");
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(path.c_str());
    if (!ctx) { printf("FAIL: load failed\n"); return 1; }

    cgc_moe::LayerGGUFMeta meta = cgc_moe::parseLayerGGUFMeta(ctx);
    printf("  layerIndex: %d\n", meta.layerIndex);
    printf("  expertsPerLayer: %d\n", meta.expertsPerLayer);
    printf("  expertStride: %llu\n", (unsigned long long)meta.expertStride);
    printf("  hiddenSize: %d\n", meta.hiddenSize);
    printf("  moeIntermediateSize: %d\n", meta.moeIntermediateSize);
    printf("  quantization: %s\n", meta.quantization.c_str());

    if (meta.layerIndex != 5) { printf("FAIL: expected layer 5, got %d\n", meta.layerIndex); cgc_gguf_lite_free(ctx); return 1; }
    if (meta.expertsPerLayer != 2) { printf("FAIL: expected 2 experts\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (meta.hiddenSize != 256) { printf("FAIL: expected hidden 256\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (meta.moeIntermediateSize != 512) { printf("FAIL: expected moe_inter 512\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (meta.quantization != "BF16") { printf("FAIL: expected BF16\n"); cgc_gguf_lite_free(ctx); return 1; }

    cgc_gguf_lite_free(ctx);
    printf("PASS\n");
    return 0;
}

// 测试: loadStreamLayoutFromGGUF + ExpertStreamer 联合验证
static int test_streamer_with_gguf(const std::string& path) {
    printf("\n=== Test 5: ExpertStreamer with GGUF layout ===\n");
    cgc_moe::StreamLayout layout = cgc_moe::loadStreamLayoutFromGGUF(path);
    if (layout.expertsPerLayer == 0) { printf("FAIL: layout parse failed\n"); return 1; }

    // 创建 streamer (4 slots,不用 mmap)
    cgc_moe::ExpertStreamer streamer(layout, 4, false);

    // 加载 expert 0 和 1
    auto result = streamer.loadExperts({0, 1});
    printf("  loaded experts: hits=%d misses=%d  (expected hits=0 misses=2)\n", result.hits, result.misses);
    if (result.misses != 2) { printf("FAIL: expected 2 misses on first load\n"); return 1; }
    if (result.buffers.size() != 2) { printf("FAIL: expected 2 buffers\n"); return 1; }

    // 再次加载 expert 0 (应命中 cache)
    auto result2 = streamer.loadExperts({0});
    printf("  reload expert 0: hits=%d misses=%d  (expected hits=1 misses=0)\n", result2.hits, result2.misses);
    if (result2.hits != 1) { printf("FAIL: expected 1 hit on reload\n"); return 1; }

    auto telemetry = streamer.telemetry();
    printf("  telemetry: requests=%llu hits=%llu misses=%llu\n",
           (unsigned long long)telemetry.totalRequests,
           (unsigned long long)telemetry.totalHits,
           (unsigned long long)telemetry.totalMisses);

    printf("PASS\n");
    return 0;
}

// 测试: ExpertComputeBridge (桥接 ExpertStreamer → cgc_moe_engine)
static int test_compute_bridge(const std::string& path) {
    printf("\n=== Test 6: ExpertComputeBridge ===\n");
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(path.c_str());
    if (!ctx) { printf("FAIL: GGUF load failed\n"); return 1; }

    cgc_moe::StreamLayout layout = cgc_moe::loadStreamLayoutFromGGUF(path);
    if (layout.expertsPerLayer == 0) { printf("FAIL: layout parse failed\n"); cgc_gguf_lite_free(ctx); return 1; }

    // 创建 streamer + bridge
    cgc_moe::ExpertStreamer streamer(layout, 4, false);
    cgc_moe::ExpertComputeBridge bridge(streamer, ctx);

    // 验证 layer meta
    auto meta = bridge.layerMeta();
    printf("  meta: layer=%d experts=%d hidden=%d moeInter=%d quant=%s\n",
           meta.layerIndex, meta.expertsPerLayer, meta.hiddenSize,
           meta.moeIntermediateSize, meta.quantization.c_str());
    if (meta.hiddenSize != 256) { printf("FAIL: hidden size mismatch\n"); cgc_gguf_lite_free(ctx); return 1; }

    // 加载 expert 0, 1
    auto views = bridge.loadExpertWeights({0, 1});
    if (views.size() != 2) { printf("FAIL: expected 2 views, got %zu\n", views.size()); cgc_gguf_lite_free(ctx); return 1; }

    for (const auto& v : views) {
        printf("  expert %d: gate=[%lld,%lld]@off=%llu up=[%lld,%lld]@off=%llu down=[%lld,%lld]@off=%llu\n",
               v.expertId,
               (long long)v.gate.shape[0], (long long)v.gate.shape[1], (unsigned long long)v.gate.offsetInBuffer,
               (long long)v.up.shape[0], (long long)v.up.shape[1], (unsigned long long)v.up.offsetInBuffer,
               (long long)v.down.shape[0], (long long)v.down.shape[1], (unsigned long long)v.down.offsetInBuffer);

        // 验证指针非空
        if (!v.gate.data || !v.up.data || !v.down.data) {
            printf("FAIL: expert %d has null pointer\n", v.expertId);
            cgc_gguf_lite_free(ctx); return 1;
        }
        // 验证 rawBuffer 非空
        if (!v.rawBuffer) { printf("FAIL: expert %d rawBuffer null\n", v.expertId); cgc_gguf_lite_free(ctx); return 1; }
        // 验证 gate offset = 0 (gate 是第一个 sub-tensor)
        if (v.gate.offsetInBuffer != 0) { printf("FAIL: gate offset should be 0, got %llu\n", (unsigned long long)v.gate.offsetInBuffer); cgc_gguf_lite_free(ctx); return 1; }
        // 验证 up offset = 4 (gate size = 4 bytes for F32×1×1)
        if (v.up.offsetInBuffer != 4) { printf("FAIL: up offset should be 4, got %llu\n", (unsigned long long)v.up.offsetInBuffer); cgc_gguf_lite_free(ctx); return 1; }
        // 验证 down offset = 8
        if (v.down.offsetInBuffer != 8) { printf("FAIL: down offset should be 8, got %llu\n", (unsigned long long)v.down.offsetInBuffer); cgc_gguf_lite_free(ctx); return 1; }
        // 验证形状 [1, 1] (测试 GGUF 的 tensor dims)
        if (v.gate.shape[0] != 1 || v.gate.shape[1] != 1) { printf("FAIL: gate shape mismatch\n"); cgc_gguf_lite_free(ctx); return 1; }
        // 验证 ggmlType = F32 (0)
        if (v.gate.ggmlType != 0) { printf("FAIL: expected F32 type, got %d\n", v.gate.ggmlType); cgc_gguf_lite_free(ctx); return 1; }
    }

    // 验证 gate/up/down 指针在 buffer 中的相对位置
    char* base = (char*)views[0].rawBuffer;
    if (views[0].gate.data != base + 0) { printf("FAIL: gate pointer mismatch\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (views[0].up.data != base + 4) { printf("FAIL: up pointer mismatch\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (views[0].down.data != base + 8) { printf("FAIL: down pointer mismatch\n"); cgc_gguf_lite_free(ctx); return 1; }

    // 再次加载 expert 0 (验证 cache hit + 指针一致性)
    auto views2 = bridge.loadExpertWeights({0});
    if (views2.size() != 1) { printf("FAIL: expected 1 view on reload\n"); cgc_gguf_lite_free(ctx); return 1; }
    if (views2[0].gate.data != views[0].gate.data) {
        printf("FAIL: gate pointer changed after reload (cache should return same buffer)\n");
        cgc_gguf_lite_free(ctx); return 1;
    }
    printf("  reload expert 0: gate pointer consistent (cache hit)\n");

    cgc_gguf_lite_free(ctx);
    printf("PASS\n");
    return 0;
}

int main() {
    printf("========================================\n");
    printf("test_expert_streamer_gguf\n");
    printf("Testing GGUF layout parsing + expert_streaming integration\n");
    printf("========================================\n");

    std::string ggufPath = createTestGGUF();
    if (ggufPath.empty()) {
        printf("FAIL: failed to create test GGUF\n");
        return 1;
    }
    printf("Created test GGUF: %s\n", ggufPath.c_str());

    int failures = 0;
    failures += test_gguf_lite_load(ggufPath);
    failures += test_loadStreamLayout(ggufPath);
    failures += test_findExpertTensors(ggufPath);
    failures += test_parseLayerGGUFMeta(ggufPath);
    failures += test_streamer_with_gguf(ggufPath);
    failures += test_compute_bridge(ggufPath);

    // 清理
    remove(ggufPath.c_str());

    printf("\n========================================\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", failures);
    }
    printf("========================================\n");
    return failures;
}
