// moe_gemv_metal.mm — Metal 真实算量实现（ObjC++）
// batch_router_kernel + expert_gemv_kernel 的 dispatch 与计时。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "moe_gemv_metal.h"

// 内置 kernel 源码（与 moe_router.metal 一致；外部传入文件时覆盖）
static const char* kRouterKernelSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void batch_router_kernel(\n"
    "    device const float* hidden   [[buffer(0)]],\n"
    "    device const float* router_w [[buffer(1)]],\n"
    "    device float* scores          [[buffer(2)]],\n"
    "    constant int& dim             [[buffer(3)]],\n"
    "    constant int& n_experts       [[buffer(4)]],\n"
    "    uint gid                      [[thread_position_in_grid]]\n"
    ") {\n"
    "    const int tok = gid / n_experts;\n"
    "    const int e   = gid % n_experts;\n"
    "    if (tok >= 64) return;\n"
    "    float s = 0.0f;\n"
    "    for (int d = 0; d < dim; d++) s += hidden[tok * dim + d] * router_w[d * n_experts + e];\n"
    "    scores[tok * n_experts + e] = s;\n"
    "}\n"
    "kernel void expert_gemv_kernel(\n"
    "    device const float* hidden  [[buffer(0)]],\n"
    "    device const float* w       [[buffer(1)]],\n"
    "    device float* out           [[buffer(2)]],\n"
    "    constant int& inter         [[buffer(3)]],\n"
    "    constant int& dim           [[buffer(4)]],\n"
    "    constant int& N             [[buffer(5)]],\n"
    "    uint gid                    [[thread_position_in_grid]]\n"
    ") {\n"
    "    const int tok = gid / dim;\n"
    "    const int d   = gid % dim;\n"
    "    if (tok >= N) return;\n"
    "    float acc = 0.0f;\n"
    "    for (int k = 0; k < inter; k++) acc += hidden[tok * inter + k] * w[k * dim + d];\n"
    "    out[tok * dim + d] = acc;\n"
    "}\n";

struct moe_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> router_pso;
    id<MTLComputePipelineState> gemv_pso;
    int layers, n_tokens, experts, topk, dim, inter;

    // buffers
    id<MTLBuffer> router_w;    // layers × dim×experts × 4B（router 权重常驻）
    id<MTLBuffer> hidden;      // N×inter × 4B（输入；同时充当 N×dim 布局下的 read）
    id<MTLBuffer> hidden_dim;  // N×dim × 4B（router 输入布局）
    id<MTLBuffer> scores;      // N×experts × 4B
    id<MTLBuffer> expert_w;    // topk × inter×dim × 4B（单层 topk 专家权重）
    id<MTLBuffer> out;         // N×dim × 4B
};

static double _now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

moe_metal_ctx_t* moe_metal_create(const char* metal_src_path, int layers,
                                  int n_tokens, int experts, int topk,
                                  int dim, int inter) {
    @autoreleasepool {
        moe_metal_ctx_t* c = (moe_metal_ctx_t*)calloc(1, sizeof(moe_metal_ctx_t));
        if (!c) return NULL;
        c->device = MTLCreateSystemDefaultDevice();
        if (!c->device) { free(c); return NULL; }
        c->queue = [c->device newCommandQueue];
        c->layers = layers; c->n_tokens = n_tokens; c->experts = experts;
        c->topk = topk; c->dim = dim; c->inter = inter;

        // 编译 kernel 源码（优先读外部文件，否则内置）
        NSString* src;
        if (metal_src_path) {
            NSError* ferr = nil;
            src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_src_path]
                                            encoding:NSUTF8StringEncoding error:&ferr];
            if (ferr || !src) {
                fprintf(stderr, "[Metal] 读 kernel 源码失败 %s，用内置\n", metal_src_path);
                src = [NSString stringWithUTF8String:kRouterKernelSrc];
            }
        } else {
            src = [NSString stringWithUTF8String:kRouterKernelSrc];
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [c->device newLibraryWithSource:src options:nil error:&err];
        if (err || !lib) {
            fprintf(stderr, "[Metal] Library 编译失败: %s\n",
                    [[err localizedDescription] UTF8String] ?: "unknown");
            moe_metal_destroy(c);
            return NULL;
        }
        id<MTLFunction> fr = [lib newFunctionWithName:@"batch_router_kernel"];
        id<MTLFunction> fg = [lib newFunctionWithName:@"expert_gemv_kernel"];
        if (!fr || !fg) {
            fprintf(stderr, "[Metal] kernel 函数缺失\n");
            moe_metal_destroy(c);
            return NULL;
        }
        NSError* perr = nil;
        c->router_pso = [c->device newComputePipelineStateWithFunction:fr error:&perr];
        c->gemv_pso   = [c->device newComputePipelineStateWithFunction:fg error:&perr];
        if (!c->router_pso || !c->gemv_pso) {
            fprintf(stderr, "[Metal] PSO 创建失败\n");
            moe_metal_destroy(c);
            return NULL;
        }

        // buffers（随机填充权重——算量只依 shape）
        size_t router_bytes = (size_t)layers * dim * experts * sizeof(float);
        size_t hidden_bytes = (size_t)n_tokens * dim * sizeof(float);
        size_t h_i_bytes    = (size_t)n_tokens * inter * sizeof(float);
        size_t scores_bytes = (size_t)n_tokens * experts * sizeof(float);
        size_t ew_bytes     = (size_t)topk * inter * dim * sizeof(float);
        size_t out_bytes    = (size_t)n_tokens * dim * sizeof(float);

        c->router_w   = [c->device newBufferWithLength:router_bytes options:MTLResourceStorageModeShared];
        c->hidden_dim = [c->device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
        c->hidden     = [c->device newBufferWithLength:h_i_bytes    options:MTLResourceStorageModeShared];
        c->scores     = [c->device newBufferWithLength:scores_bytes options:MTLResourceStorageModeShared];
        c->expert_w   = [c->device newBufferWithLength:ew_bytes     options:MTLResourceStorageModeShared];
        c->out        = [c->device newBufferWithLength:out_bytes    options:MTLResourceStorageModeShared];
        if (!c->router_w || !c->hidden || !c->hidden_dim || !c->scores || !c->expert_w || !c->out) {
            fprintf(stderr, "[Metal] buffer 分配失败\n");
            moe_metal_destroy(c);
            return NULL;
        }
        // 随机填充（种子固定，可复现）
        srand(42);
        float* rw = (float*)[c->router_w contents];
        for (size_t i = 0; i < router_bytes / 4; i++) rw[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        float* hd = (float*)[c->hidden_dim contents];
        for (size_t i = 0; i < hidden_bytes / 4; i++) hd[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        float* hi = (float*)[c->hidden contents];
        for (size_t i = 0; i < h_i_bytes / 4; i++) hi[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        float* ew = (float*)[c->expert_w contents];
        for (size_t i = 0; i < ew_bytes / 4; i++) ew[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        return c;
    }
}

void moe_metal_destroy(moe_metal_ctx_t* c) {
    if (!c) return;
    @autoreleasepool {
        [c->router_pso release]; [c->gemv_pso release];
        [c->router_w release]; [c->hidden release]; [c->hidden_dim release];
        [c->scores release]; [c->expert_w release]; [c->out release];
        [c->queue release]; [c->device release];
    }
    free(c);
}

// host top-k 选择：scores[N×E] → topk 索引（简单冒泡选择）
static void _topk(float* scores, int n_tokens, int experts, int topk, int* idx_out) {
    for (int t = 0; t < n_tokens; t++) {
        float* s = scores + (size_t)t * experts;
        for (int k = 0; k < topk; k++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < experts; e++) {
                if (s[e] > bv) { bv = s[e]; best = e; }
            }
            idx_out[t * topk + k] = best;
            s[best] = -1e30f;  // 标记已选
        }
    }
}

int moe_metal_run(moe_metal_ctx_t* c, moe_metal_result_t* out) {
    if (!c || !out) return -1;
    @autoreleasepool {
        id<MTLCommandQueue> q = c->queue;
        int N = c->n_tokens, E = c->experts, D = c->dim, I = c->inter, K = c->topk, L = c->layers;

        double t0 = _now_ms();
        for (int l = 0; l < L; l++) {
            // 每层单个 command buffer：router + topk×gemv 全部合流，一次 commit/wait
            // （对齐 llama.cpp Metal 后端做法，消除 per-kernel wait 的调度 overhead）
            id<MTLCommandBuffer> cb = [q commandBuffer];

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:c->router_pso];
            [enc setBuffer:c->hidden_dim offset:0 atIndex:0];
            [enc setBuffer:c->router_w offset:(size_t)l * D * E * 4 atIndex:1];
            [enc setBuffer:c->scores offset:0 atIndex:2];
            int rdim = D, rexp = E;
            [enc setBytes:&rdim length:4 atIndex:3];
            [enc setBytes:&rexp length:4 atIndex:4];
            MTLSize grid = MTLSizeMake(N * E, 1, 1);
            MTLSize grp  = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:grp];
            [enc endEncoding];

            for (int k = 0; k < K; k++) {
                id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
                [enc2 setComputePipelineState:c->gemv_pso];
                [enc2 setBuffer:c->hidden offset:0 atIndex:0];
                [enc2 setBuffer:c->expert_w offset:(size_t)k * I * D * 4 atIndex:1];
                [enc2 setBuffer:c->out offset:0 atIndex:2];
                int rI = I, rD = D, rN = N;
                [enc2 setBytes:&rI length:4 atIndex:3];
                [enc2 setBytes:&rD length:4 atIndex:4];
                [enc2 setBytes:&rN length:4 atIndex:5];
                MTLSize grid2 = MTLSizeMake(N * D, 1, 1);
                MTLSize grp2  = MTLSizeMake(256, 1, 1);
                [enc2 dispatchThreads:grid2 threadsPerThreadgroup:grp2];
                [enc2 endEncoding];
            }

            [cb commit];
            [cb waitUntilCompleted];
        }
        double t1 = _now_ms();

        // 2) host top-k
        float* scores = (float*)[c->scores contents];
        int* idx = (int*)malloc((size_t)N * K * sizeof(int));
        _topk(scores, N, E, K, idx);
        double t2 = _now_ms();

        // 3) expert_gemv：hidden[N×I] × expert_w[k][I×D] → out[N×D]，每层 topk 个 dispatch
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                (void)idx[k];  // 本轮用固定 expert_w 布局（合成权重，专家索引仅占位）
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:c->gemv_pso];
                [enc setBuffer:c->hidden offset:0 atIndex:0];
                [enc setBuffer:c->expert_w offset:(size_t)k * I * D * 4 atIndex:1];
                [enc setBuffer:c->out offset:0 atIndex:2];
                int rI = I, rD = D, rN = N;
                [enc setBytes:&rI length:4 atIndex:3];
                [enc setBytes:&rD length:4 atIndex:4];
                [enc setBytes:&rN length:4 atIndex:5];
                MTLSize grid = MTLSizeMake(N * D, 1, 1);
                MTLSize grp  = MTLSizeMake(256, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:grp];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        double t3 = _now_ms();
        free(idx);

        out->batch_router_ms = t1 - t0;
        out->topk_select_ms  = t2 - t1;
        out->gemv_ms         = t3 - t2;
        out->total_ms        = t3 - t0;
        out->layers          = L;
        out->n_tokens        = N;
        return 0;
    }
}

double moe_metal_bench(moe_metal_ctx_t* c, int rounds, moe_metal_result_t* avg) {
    double sum = 0;
    if (avg) memset(avg, 0, sizeof(*avg));
    for (int r = 0; r < rounds; r++) {
        moe_metal_result_t res;
        if (moe_metal_run(c, &res) != 0) return -1;
        sum += res.total_ms;
        if (avg) {
            avg->batch_router_ms += res.batch_router_ms;
            avg->topk_select_ms  += res.topk_select_ms;
            avg->gemv_ms         += res.gemv_ms;
            avg->total_ms        += res.total_ms;
            avg->layers = res.layers;
            avg->n_tokens = res.n_tokens;
        }
    }
    if (rounds > 0 && avg) {
        avg->batch_router_ms /= rounds;
        avg->topk_select_ms  /= rounds;
        avg->gemv_ms         /= rounds;
        avg->total_ms        /= rounds;
    }
    return rounds > 0 ? sum / rounds : 0;
}
