// bench_gguf_fill.c — measure real pread fill latency for one decode step
// from a llama.cpp GGUF file, using the segment-aware addressing.
//
// Usage: bench_gguf_fill <model.gguf> <trace.txt> [tokens=N]
// Replays the routing trace (groups of n_layers lines = 1 token), preads each
// token's 8 experts x segments with cgc_expert_segments addressing.
// Prints per-token fill ms + aggregate. Run twice: pass 1 ~cold,
// pass 2 ~warm page cache.
//
// Dual-family support:
//   qwen35moe: blk.N.ffn_gate_exps / ffn_up_exps / ffn_down_exps (3 segments)
//   gemma4:    blk.N.ffn_gate_up_exps / ffn_down_exps             (2 segments, merged gate_up)
// Layer count / expert count are derived from the GGUF itself.
// Layer-0 segments are validated against cgc_load_stream_layout_from_gguf
// (the fixed parser) — a mismatch aborts.

#include "cgc_gguf_lite.h"
#include "cgc_expert_streamer_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_LAYERS 64
#define MAX_TRACE_LINES 20000

typedef struct {
    int n_seg;
    uint64_t seg_base[3];   // absolute file offsets (data_start + raw)
    uint64_t seg_size[3];   // per-expert bytes per segment
} layer_seg_t;

static uint64_t now_nanos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// deterministic pseudo-random expert id
static uint32_t rng_state = 0x9e3779b9u;
static uint32_t next_rand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <trace.txt|-for-synth> [tokens=N]\n", argv[0]);
        return 2;
    }
    int synth_tokens = 0;
    for (int a = 3; a < argc; a++) {
        if (sscanf(argv[a], "tokens=%d", &synth_tokens) == 1) { /* ok */ }
    }

    // ---- layout via the fixed parser (layer-0 authoritative) ----
    cgc_stream_layout_t layout = cgc_load_stream_layout_from_gguf(argv[1]);
    if (layout.experts_per_layer == 0) { fprintf(stderr, "layout parse failed\n"); return 1; }
    const int n_experts = layout.experts_per_layer;

    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(argv[1]);
    if (!ctx) { fprintf(stderr, "GGUF load failed\n"); return 1; }
    const uint64_t DS = ctx->data_start;

    // ---- per-layer segment table from tensor names (absolute offsets) ----
    // 先偵測佈局型態：合併（ffn_gate_up_exps，gemma4，2 段：gate_up→0, down→1）
    // 或分開（ffn_gate_exps/ffn_up_exps，qwen35moe，3 段：gate→0, up→1, down→2）
    int merged = 0;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const char* nm = ctx->tensor_names[i];
        if (nm && strstr(nm, "ffn_gate_up_exps")) { merged = 1; break; }
    }
    layer_seg_t L[MAX_LAYERS];
    memset(L, 0, sizeof(L));
    int n_layers = 0;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const char* nm = ctx->tensor_names[i];
        if (!nm || strncmp(nm, "blk.", 4) != 0) continue;
        int layer = -1;
        if (sscanf(nm, "blk.%d.", &layer) != 1 || layer < 0 || layer >= MAX_LAYERS) continue;
        if (layer + 1 > n_layers) n_layers = layer + 1;
        if (strstr(nm, ".scale")) continue;
        uint64_t abs_off = DS + ctx->tensors[i].offset;
        double bpe = cgc_ggml_type_bytes_per_elem(ctx->tensors[i].type);
        uint64_t total = (uint64_t)(bpe * (double)ctx->tensors[i].n_elements);
        uint64_t per_exp = total / (uint64_t)n_experts;
        if (merged) {
            if (strstr(nm, "ffn_gate_up_exps")) { L[layer].seg_base[0] = abs_off; L[layer].seg_size[0] = per_exp; }
            else if (strstr(nm, "ffn_down_exps")) { L[layer].seg_base[1] = abs_off; L[layer].seg_size[1] = per_exp; }
        } else {
            if (strstr(nm, "ffn_gate_exps"))  { L[layer].seg_base[0] = abs_off; L[layer].seg_size[0] = per_exp; }
            else if (strstr(nm, "ffn_up_exps"))    { L[layer].seg_base[1] = abs_off; L[layer].seg_size[1] = per_exp; }
            else if (strstr(nm, "ffn_down_exps"))  { L[layer].seg_base[2] = abs_off; L[layer].seg_size[2] = per_exp; }
        }
    }
    if (n_layers == 0) { fprintf(stderr, "no blk.N layers found\n"); return 1; }

    // determine per-layer segment count: 3 = gate+up+down, 2 = gate_up+down, else 1
    int n_seg0 = merged ? 2 : 3;
    for (int l = 0; l < n_layers; l++) L[l].n_seg = n_seg0;
    for (int l = 0; l < n_layers; l++) {
        int have = (L[l].seg_size[0] ? 1 : 0) + (L[l].seg_size[1] ? 1 : 0) + (L[l].seg_size[2] ? 1 : 0);
        if (have != n_seg0) { fprintf(stderr, "inconsistent seg count at layer %d\n", l); return 1; }
    }
    uint64_t per_exp_bytes = 0;
    for (int s = 0; s < n_seg0; s++) per_exp_bytes += L[0].seg_size[s];
    printf("GGUF: layers=%d experts=%d per_expert=%llu bytes (%d segments) data_start=%llu\n",
           n_layers, n_experts, (unsigned long long)per_exp_bytes, n_seg0,
           (unsigned long long)DS);

    // ---- validate layer-0 segments against the fixed parser ----
    int l0_ok = 1;
    if (layout.has_segments) {
        for (int s = 0; s < n_seg0; s++) {
            if (layout.seg_base[s] != L[0].seg_base[s] || layout.seg_size[s] != L[0].seg_size[s]) {
                fprintf(stderr, "[FAIL] layer-0 seg%d: parser(%llu,%llu) != table(%llu,%llu)\n",
                        s, (unsigned long long)layout.seg_base[s], (unsigned long long)layout.seg_size[s],
                        (unsigned long long)L[0].seg_base[s], (unsigned long long)L[0].seg_size[s]);
                l0_ok = 0;
            }
        }
    } else {
        fprintf(stderr, "[FAIL] expected has_segments=1\n");
        l0_ok = 0;
    }
    if (!l0_ok) { fprintf(stderr, "aborting: segment addressing mismatch\n"); return 1; }
    printf("layer-0 segment table == parser layout ✓\n");

    // ---- read trace (or synthesize) ----
    static int trace_exp[MAX_TRACE_LINES][8];
    static int trace_layer[MAX_TRACE_LINES];
    int nlines = 0;
    FILE* tf = fopen(argv[2], "r");
    if (tf) {
        char buf[512];
        while (fgets(buf, sizeof(buf), tf) && nlines < MAX_TRACE_LINES) {
            char* p = strchr(buf, ',');
            if (!p) continue;
            char* phase_start = p + 1;        // phase
            p = strchr(phase_start, ',');
            if (!p) continue;
            // 只回放 decode 相位（prefill 是不同調度，fill 成本不可比）
            int phase_is_decode = (p - phase_start == 15 && memcmp(phase_start, "decodeProtected", 15) == 0);
            p = strchr(p + 1, ',');           // step
            if (!p) continue;
            p = strchr(p + 1, ',');           // hits
            if (!p) continue;
            char* exp = p + 1;
            int layer = -1;
            sscanf(buf, "layer_%d.", &layer);
            if (layer < 0 || layer >= n_layers) continue;
            if (!phase_is_decode) continue;
            int n = 0;
            char* tok = strtok(exp, " \n");
            while (tok && n < 8) { trace_exp[nlines][n++] = atoi(tok) % n_experts; tok = strtok(NULL, " \n"); }
            trace_layer[nlines] = layer;
            nlines++;
        }
        fclose(tf);
    }
    int n_tokens = nlines / n_layers;
    if (synth_tokens > 0 && n_tokens < synth_tokens) {
        // extend with synthetic tokens (deterministic), clamped to n_experts
        printf("trace gave %d tokens, synthesizing to %d\n", n_tokens, synth_tokens);
        for (int t = n_tokens; t < synth_tokens; t++) {
            for (int li = 0; li < n_layers; li++) {
                int idx = t * n_layers + li;
                if (idx >= MAX_TRACE_LINES) { fprintf(stderr, "trace buffer full\n"); return 1; }
                trace_layer[idx] = li;
                for (int k = 0; k < 8; k++)
                    trace_exp[idx][k] = (int)(next_rand() % (uint32_t)n_experts);
            }
        }
        nlines = synth_tokens * n_layers;
        n_tokens = synth_tokens;
    }
    if (n_tokens == 0) { fprintf(stderr, "no tokens (trace empty)\n"); return 1; }
    printf("trace: %d lines = %d tokens\n", nlines, n_tokens);

    // ---- replay: one pread per (layer, expert, segment) ----
    uint8_t* buf = malloc((size_t)per_exp_bytes);
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    uint64_t total_ns = 0;
    uint64_t total_bytes = 0;
    for (int t = 0; t < n_tokens; t++) {
        uint64_t t0 = now_nanos();
        for (int li = 0; li < n_layers; li++) {
            int idx = t * n_layers + li;
            int layer = trace_layer[idx];
            for (int k = 0; k < 8; k++) {
                int e = trace_exp[idx][k];
                for (int s = 0; s < L[layer].n_seg; s++) {
                    uint64_t off = L[layer].seg_base[s] + (uint64_t)e * L[layer].seg_size[s];
                    size_t sz = (size_t)L[layer].seg_size[s];
                    if (pread(fd, buf, sz, (off_t)off) != (ssize_t)sz) {
                        fprintf(stderr, "pread fail layer=%d exp=%d seg=%d\n", layer, e, s);
                        return 1;
                    }
                }
            }
        }
        uint64_t dt = now_nanos() - t0;
        total_ns += dt;
        total_bytes += (uint64_t)n_layers * 8 * per_exp_bytes;
    }
    double avg_ms = (double)total_ns / n_tokens / 1e6;
    double avg_bytes = (double)total_bytes / n_tokens;
    // bytes/ns 數值上即 GB/s（1 B/ns = 1e9 B/s = 1 GB/s）
    double eff_gbs = avg_bytes / ((double)total_ns / n_tokens);
    printf("\nRESULT: %d steps | avg fill %6.2f ms/step | avg %.1f MB/step | "
           "eff %.2f GB/s\n",
           n_tokens, avg_ms, avg_bytes / 1e6, eff_gbs);
    free(buf);
    close(fd);
    cgc_gguf_lite_free(ctx);
    return 0;
}
