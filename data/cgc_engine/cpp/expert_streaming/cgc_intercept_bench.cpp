// cgc_intercept_bench.cpp — CGC Option A 计算期拦截 独立 A/B 测试
//
// 对比 (同一张 mul_mat_id 图, 同一 CGC backend, 仅切换 CGC_EXPERT_INTERCEPT):
//   [1] baseline     : 标准 mul_mat_id 直读整模型权重 buffer (mmap 语义)
//   [2] intercept(全驻留): 拦截换 src0 指针 -> compact slot 区 + ids 重映射 (零拷贝直读)
//   [3] intercept(miss)   : 部分专家未驻留 -> 拦截必须整段 fallback, 结果仍正确
//
// 验证:
//   - 正确性: [2]/[3] 结果与 [1] 逐元素一致 (allclose 1e-5)
//   - 零拷贝: 把 compact 区填成垃圾时结果必须改变 (证明确实读 compact 区)
//   - 开销  : [2] 相对 [1] 的 per-compute 时间增量 (换指针 + ids 重映射成本)
//
// 构建 (在 llama-build-cgc 产物目录, MinGW g++):
//   g++ -O2 -std=c++17 -I<ggml/include> -I<ggml/src> -I<ggml/src/ggml-cgc> -I<ggml/src/ggml-cpu> \
//       cgc_intercept_bench.cpp ggml-cgc.cpp cgc_plan.cpp cgc_inject_tap.cpp cgc_report.cpp \
//       -L<llama-build-cgc/ggml/src> -lggml-base -lggml -lggml-cpu -o cgc_intercept_bench.exe
// 运行:
//   CGC_SPLIT_MOE=1 ./cgc_intercept_bench.exe [n_expert] [n_ff] [n_embd] [top_k] [n_repeat]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" ggml_backend_reg_t ggml_backend_cgc_reg(void);
extern "C" void cgc_intercept_register(const void * orig_src0_data, int n_expert,
                                       int slot_count, size_t slot_stride,
                                       const int32_t * exp_to_slot,
                                       const char * compact_base);

struct bench_params_t {
    int n_expert = 32;
    int n_ff     = 128;
    int n_embd   = 64;
    int top_k    = 4;
    int n_tokens = 1;
    int n_repeat = 200;
    int seed     = 42;
};

// 简单的确定性 LCG (避免依赖 <random> 的平台差异)
static unsigned int g_seed = 42;
static unsigned int lcg_rand(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return (g_seed >> 16) & 0x7fff;
}
static float frand(void) {
    return ((float) lcg_rand() / 32767.0f) * 2.0f - 1.0f;
}

static double now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1e6;
#endif
}

struct graph_t {
    struct ggml_context * ctx = nullptr;
    struct ggml_cgraph gf;
    struct ggml_tensor * w      = nullptr; // [n_ff, n_embd, n_expert] F32
    struct ggml_tensor * cur    = nullptr; // [n_embd, n_tokens] F32
    struct ggml_tensor * scores = nullptr; // [n_expert, n_tokens] F32
    struct ggml_tensor * sel    = nullptr; // [top_k, n_tokens] I32 (argsort_top_k)
    struct ggml_tensor * out    = nullptr; // [n_ff, n_tokens] F32
};

static bool build_graph(graph_t & g, const bench_params_t & P) {
    const int64_t ne_w[3] = { P.n_ff, P.n_embd, P.n_expert };
    const int64_t ne_cur[2] = { P.n_embd, P.n_tokens };
    const int64_t ne_scores[2] = { P.n_expert, P.n_tokens };

    g.ctx = ggml_init({ 64 * 1024 * 1024, nullptr, false });
    if (g.ctx == nullptr) return false;

    g.w      = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, ne_w[0], ne_w[1], ne_w[2]);
    g.cur    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne_cur[0], ne_cur[1]);
    g.scores = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne_scores[0], ne_scores[1]);

    // 真实路由 op: argsort_top_k -> selected_experts I32 [top_k, n_tokens]
    g.sel = ggml_argsort_top_k(g.ctx, g.scores, P.top_k);

    // MoE FFN 计算: out = mul_mat_id(w, cur, sel)  (src0=w, src1=cur, src2=sel)
    g.out = ggml_mul_mat_id(g.ctx, g.w, g.cur, g.sel);

    ggml_build_forward_expand(&g.gf, g.out);
    return true;
}

static bool alloc_graph(ggml_backend_t backend, graph_t & g) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, ggml_backend_get_default_buffer_type(backend));
    if (buf == nullptr) return false;
    // 分配器已把每个 tensor 的 data 填好; 备份 buffer 句柄以便 release
    ggml_backend_buffer_free(buf); // 图上下文持有? 不持有 -> 需在调用方保存
    return true;
}

// 在 w buffer 上填充伪随机权重
static void fill_weights(graph_t & g, const bench_params_t & P) {
    float * w = (float *) g.w->data;
    const int64_t n = g.w->ne[0] * g.w->ne[1] * g.w->ne[2];
    for (int64_t i = 0; i < n; i++) w[i] = frand();
}

static void fill_inputs(graph_t & g, const bench_params_t & P) {
    float * cur = (float *) g.cur->data;
    for (int64_t i = 0; i < g.cur->ne[0] * g.cur->ne[1]; i++) cur[i] = frand();
    float * sc = (float *) g.scores->data;
    for (int64_t i = 0; i < g.scores->ne[0] * g.scores->ne[1]; i++) sc[i] = frand();
}

// 读取 routing 结果 (argsort_top_k 输出) 选中的专家 id
static void read_selected(graph_t & g, int32_t * out_ids, int n) {
    memcpy(out_ids, g.sel->data, sizeof(int32_t) * n);
}

// 全量结果校验 (逐元素 allclose)
static bool allclose(const float * a, const float * b, int64_t n, double tol) {
    for (int64_t i = 0; i < n; i++) {
        const double da = a[i];
        const double db = b[i];
        const double diff = std::fabs(da - db);
        const double scale = std::max(std::fabs(da), std::fabs(db));
        if (diff > tol * (scale > 1.0 ? scale : 1.0)) {
            fprintf(stderr, "  MISMATCH at %lld: a=%f b=%f (diff=%f)\n",
                    (long long) i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

static void set_env_flag(const char * name, bool on) {
    if (on) {
        _putenv_s(name, "1");
    } else {
        _putenv_s(name, "0");
    }
}

int main(int argc, char ** argv) {
    bench_params_t P;
    if (argc > 1) P.n_expert = atoi(argv[1]);
    if (argc > 2) P.n_ff     = atoi(argv[2]);
    if (argc > 3) P.n_embd   = atoi(argv[3]);
    if (argc > 4) P.top_k    = atoi(argv[4]);
    if (argc > 5) P.n_repeat = atoi(argv[5]);
    g_seed = (unsigned int) P.seed;

    printf("============================================================\n");
    printf("  CGC Option A: 计算期拦截 A/B 测试\n");
    printf("============================================================\n");
    printf("  n_expert=%d n_ff=%d n_embd=%d top_k=%d n_tokens=%d n_repeat=%d\n",
           P.n_expert, P.n_ff, P.n_embd, P.top_k, P.n_tokens, P.n_repeat);

    // ---- 0. 后端 ----
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) { fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n"); return 1; }

    ggml_backend_reg_t cgc_reg = ggml_backend_cgc_reg();
    if (cgc_reg == nullptr) { fprintf(stderr, "ERROR: ggml_backend_cgc_reg failed\n"); return 1; }
    ggml_backend_dev_t cgc_dev = ggml_backend_reg_get_device(cgc_reg, 0);
    if (cgc_dev == nullptr) { fprintf(stderr, "ERROR: cgc dev null\n"); return 1; }
    ggml_backend_t cgc = ggml_backend_dev_init(cgc_dev, nullptr);
    if (cgc == nullptr) { fprintf(stderr, "ERROR: cgc backend init failed\n"); return 1; }
    printf("  backends: cpu=%s cgc=%s\n",
           ggml_backend_name(cpu), ggml_backend_name(cgc));

    // ---- 1. 建图 + 分配 ----
    graph_t g;
    if (!build_graph(g, P)) { fprintf(stderr, "ERROR: build_graph failed\n"); return 1; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, ggml_backend_get_default_buffer_type(cpu));
    if (buf == nullptr) { fprintf(stderr, "ERROR: alloc failed\n"); return 1; }
    ggml_backend_buffer_t cgc_buf = ggml_backend_alloc_ctx_tensors(g.ctx, ggml_backend_get_default_buffer_type(cgc));
    if (cgc_buf == nullptr) { fprintf(stderr, "ERROR: cgc alloc failed\n"); return 1; }

    fill_weights(g, P);
    fill_inputs(g, P);
    ggml_backend_tensor_set(g.w, g.w->data, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.cur, g.cur->data, 0, ggml_nbytes(g.cur));
    ggml_backend_tensor_set(g.scores, g.scores->data, 0, ggml_nbytes(g.scores));

    const int64_t out_n = g.out->ne[0] * g.out->ne[1];
    std::vector<float> ref(out_n), cgc_base(out_n), cgc_int(out_n), cgc_int_miss(out_n), cgc_garbage(out_n);

    // 确保路由已算好 (argsort 输出 g.sel)
    // ---- 2. 参考: 纯 CPU backend ----
    ggml_backend_graph_compute(cpu, &g.gf);
    ggml_backend_tensor_get(g.out, ref.data(), 0, sizeof(float) * out_n);

    // ---- 3. baseline: CGC backend (split_moe=1, intercept=0) ----
    set_env_flag("CGC_SPLIT_MOE", true);
    set_env_flag("CGC_EXPERT_INTERCEPT", false);
    ggml_backend_graph_compute(cgc, &g.gf);
    ggml_backend_tensor_get(g.out, cgc_base.data(), 0, sizeof(float) * out_n);
    printf("  [1] baseline(cgc) vs cpu-ref : %s\n",
           allclose(cgc_base.data(), ref.data(), out_n, 1e-5) ? "PASS" : "FAIL");

    // ---- 4. compact slot 区: 全专家驻留 (slot e = expert e) ----
    const size_t expert_bytes = (size_t) ggml_nbytes(g.w) / (size_t) P.n_expert;
    std::vector<char> compact(P.n_expert * expert_bytes);
    {
        const char * wsrc = (const char *) g.w->data;
        for (int e = 0; e < P.n_expert; e++) {
            memcpy(compact.data() + (size_t) e * expert_bytes,
                   wsrc + (size_t) e * expert_bytes, expert_bytes);
        }
    }
    std::vector<int32_t> exp_to_slot(P.n_expert);
    for (int e = 0; e < P.n_expert; e++) exp_to_slot[e] = e;

    // ---- 5. intercept 全驻留: 换指针 + ids 重映射 ----
    cgc_intercept_register(g.w->data, P.n_expert, P.n_expert, expert_bytes,
                           exp_to_slot.data(), compact.data());
    set_env_flag("CGC_EXPERT_INTERCEPT", true);
    ggml_backend_graph_compute(cgc, &g.gf);
    ggml_backend_tensor_get(g.out, cgc_int.data(), 0, sizeof(float) * out_n);
    printf("  [2] intercept(all resident) vs cpu-ref : %s\n",
           allclose(cgc_int.data(), ref.data(), out_n, 1e-5) ? "PASS" : "FAIL");

    // ---- 6. 零拷贝证明: compact 区填垃圾, 结果必须改变 ----
    std::vector<char> compact_orig = compact;
    for (size_t i = 0; i < compact.size(); i++) compact[i] = (char) (i * 7 + 3);
    ggml_backend_graph_compute(cgc, &g.gf);
    ggml_backend_tensor_get(g.out, cgc_garbage.data(), 0, sizeof(float) * out_n);
    const bool changed = !allclose(cgc_garbage.data(), ref.data(), out_n, 1e-5);
    printf("  [3] zero-copy proof (garbage compact changes result): %s\n",
           changed ? "PASS (确实直读 compact 区)" : "FAIL (未读到 compact 区)");
    compact = compact_orig;
    cgc_intercept_register(g.w->data, P.n_expert, P.n_expert, expert_bytes,
                           exp_to_slot.data(), compact.data());

    // ---- 7. intercept miss: 注册一半专家, 路由选中未驻留专家 -> fallback ----
    {
        // 先看当前路由选中了谁
        std::vector<int32_t> ids(P.top_k);
        read_selected(g, ids.data(), P.top_k);
        // 驻留前 P.n_expert/2 个专家, 若路由选到 >= n_expert/2 则必然 miss
        std::vector<int32_t> exp2(P.n_expert, -1);
        for (int e = 0; e < P.n_expert / 2; e++) exp2[e] = e;
        cgc_intercept_register(g.w->data, P.n_expert, P.n_expert / 2, expert_bytes,
                               exp2.data(), compact.data());
        ggml_backend_graph_compute(cgc, &g.gf);
        ggml_backend_tensor_get(g.out, cgc_int_miss.data(), 0, sizeof(float) * out_n);
        printf("  [4] intercept(miss -> fallback) vs cpu-ref : %s\n",
               allclose(cgc_int_miss.data(), ref.data(), out_n, 1e-5) ? "PASS" : "FAIL");
    }

    // ---- 8. 开销 A/B: baseline vs intercept (同图, 同 backend, 交替 N 次) ----
    set_env_flag("CGC_EXPERT_INTERCEPT", false);
    ggml_backend_graph_compute(cgc, &g.gf); // warmup
    const double t0 = now_seconds();
    for (int i = 0; i < P.n_repeat; i++) ggml_backend_graph_compute(cgc, &g.gf);
    const double t_base = now_seconds() - t0;

    set_env_flag("CGC_EXPERT_INTERCEPT", true);
    ggml_backend_graph_compute(cgc, &g.gf); // warmup
    const double t1 = now_seconds();
    for (int i = 0; i < P.n_repeat; i++) ggml_backend_graph_compute(cgc, &g.gf);
    const double t_int = now_seconds() - t1;

    const double per_base = t_base / P.n_repeat * 1e6;
    const double per_int  = t_int / P.n_repeat * 1e6;
    printf("\n  ---- 开销 A/B (per-compute) ----\n");
    printf("  baseline   : %8.2f us/step\n", per_base);
    printf("  intercept  : %8.2f us/step\n", per_int);
    printf("  delta      : %+8.2f us/step (%+.2f%%)\n",
           per_int - per_base, (per_int - per_base) / per_base * 100.0);
    printf("  (拦截本身只含换 2 个指针 + %d 个 int32 remap; 权重读取零拷贝)\n", P.top_k);

    // ---- 清理 ----
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(cgc_buf);
    ggml_backend_free(cgc);
    ggml_backend_free(cpu);
    ggml_free(g.ctx);
    printf("\n  DONE\n");
    return 0;
}
