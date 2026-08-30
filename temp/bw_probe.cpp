// CGC bandwidth probe: isolate ggml_mul_mat_id on the Metal backend at real
// qwen36 MoE expert shapes, measure effective weight-read bandwidth.
//
//   gate/up : IQ2_S   m=512 (N), k=2048 (K), 256 experts, top 8
//   down    : IQ3_XXS m=2048,      k=512,       256 experts, top 8
//
// Effective bandwidth = bytes_read / GPU_time (bytes = n_used slabs).
// Compare against M4 Max peak (~546 GB/s) to get the access-pattern efficiency.

#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static const int N_MATS = 256;
static const int N_USED = 8;

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

// build, alloc, init, then time N reps; returns min/mean ms
static void probe(ggml_backend_t backend, ggml_type wq, int m, int k, int n_tokens, int n_used = N_USED) {
    ggml_init_params ip = { /* mem_size */ 1024ull*1024*1024, /* mem_buffer */ nullptr, /* no_alloc */ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as = ggml_new_tensor_3d(ctx, wq, k, m, N_MATS);
    ggml_set_name(as, "probe_weights");
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(ids, "probe_ids");
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_set_name(x, "probe_x");
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, x, ids);
    ggml_set_name(out, "probe_out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // init: weights = garbage bytes (in-bounds, timing-only), ids = 0..n_used-1, x = ones
    std::vector<uint8_t> wdata(ggml_nbytes(as));
    for (auto & b : wdata) b = 0xA5;
    ggml_backend_tensor_set(as, wdata.data(), 0, wdata.size());

    std::vector<int32_t> idata(n_used * n_tokens);
    for (int i = 0; i < n_used * n_tokens; i++) idata[i] = i % n_used;
    ggml_backend_tensor_set(ids, idata.data(), 0, idata.size() * sizeof(int32_t));

    std::vector<float> xdata((size_t) k * n_used * n_tokens, 1.0f);
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size() * sizeof(float));

    // warmup
    for (int i = 0; i < 10; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }

    // (A) isolated: sync every rep -> per-graph round-trip cost (encode+exec+sync)
    const int n_rep = 50;
    double t_min = 1e30, t_sum = 0;
    for (int i = 0; i < n_rep; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double dt = now_ms() - t0;
        if (dt < t_min) t_min = dt;
        t_sum += dt;
    }

    // (B) sustained: pipeline K graphs, sync every G -> true streaming rate
    const int K = 200, G = 20;
    double s0 = now_ms();
    for (int i = 0; i < K; i++) {
        ggml_backend_graph_compute(backend, gf);
        if ((i + 1) % G == 0) {
            ggml_backend_synchronize(backend);
        }
    }
    ggml_backend_synchronize(backend);
    double s_ms = now_ms() - s0;

    // bytes per graph = n_tokens * n_used slabs (ids = [n_used, n_tokens])
    double bytes = (double) ggml_nbytes(as) / N_MATS * n_used * n_tokens;
    double gbps_iso  = bytes / 1e9 / (t_min / 1e3);
    double gbps_sust = bytes / 1e9 / (s_ms / K / 1e3);

    printf("%-9s m=%-5d k=%-5d used=%-2d tok=%-2d | %8.2f MB/graph | iso(min) %7.3f ms  sust %8.3f ms/graph | iso %6.1f GB/s   sust %6.1f GB/s\n",
           ggml_type_name(wq), m, k, n_used, n_tokens, bytes/1e6, t_min, s_ms/K, gbps_iso, gbps_sust);

    ggml_free(ctx);
}

// (C) realistic in-graph cost: NL layers x (gate/up + down) mmid ops in ONE graph_compute
static void probe_layers(ggml_backend_t backend, int n_layers, int n_tokens, int n_used) {
    ggml_init_params ip = { /* mem_size */ 2048ull*1024*1024, /* mem_buffer */ nullptr, /* no_alloc */ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    double bytes_total = 0;
    std::vector<ggml_tensor *> outs;

    // per layer: gate/up (iq2_s m=512 k=2048) + down (iq3_xxs m=2048 k=512), n_used distinct experts
    for (int l = 0; l < n_layers; l++) {
        ggml_tensor * as_gu = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_S,   2048, 512,  N_MATS);
        ggml_tensor * as_dn = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_XXS, 512,  2048, N_MATS);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_tensor * x_gu  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, n_used, n_tokens);
        ggml_tensor * x_dn  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512,  n_used, n_tokens);
        ggml_tensor * o_gu  = ggml_mul_mat_id(ctx, as_gu, x_gu, ids);
        ggml_tensor * o_dn  = ggml_mul_mat_id(ctx, as_dn, x_dn, ids);
        ggml_build_forward_expand(gf, o_gu);
        ggml_build_forward_expand(gf, o_dn);
        outs.push_back(o_gu);
        outs.push_back(o_dn);
        bytes_total += (double) ggml_nbytes(as_gu) / N_MATS * n_used * n_tokens;
        bytes_total += (double) ggml_nbytes(as_dn) / N_MATS * n_used * n_tokens;
    }

    ggml_backend_alloc_ctx_tensors(ctx, backend);
    // init tensors (garbage weights, ids 0..23, x ones)
    for (size_t i = 0; i < outs.size(); i++) {
        ggml_tensor * o = outs[i];
        std::vector<uint8_t> w(ggml_nbytes(o->src[0]), 0xA5);
        ggml_backend_tensor_set(o->src[0], w.data(), 0, w.size());
    }
    std::vector<int32_t> idata(n_used * n_tokens);
    for (int i = 0; i < n_used * n_tokens; i++) idata[i] = i % n_used;
    for (size_t i = 0; i < outs.size(); i++) {
        ggml_tensor * o = outs[i];
        size_t kcols = (size_t) o->src[0]->ne[0]; // gate/up=2048, down=512
        std::vector<float> xd(kcols * n_used * n_tokens, 1.0f);
        ggml_backend_tensor_set(o->src[1], xd.data(), 0, xd.size() * sizeof(float));
        ggml_backend_tensor_set(o->src[2], idata.data(), 0, idata.size() * sizeof(int32_t));
    }

    for (int i = 0; i < 10; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }
    const int n_rep = 50;
    double t_min = 1e30, t_sum = 0;
    for (int i = 0; i < n_rep; i++) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double dt = now_ms() - t0;
        if (dt < t_min) t_min = dt;
        t_sum += dt;
    }
    int n_mmid = n_layers * 2;
    printf("%-9s %2d layers = %3d mmid ops, tok=%-2d | %9.2f MB | graph min %7.3f ms | per-mmid %7.4f ms | eff %6.1f GB/s\n",
           "in-graph", n_layers, n_mmid, n_tokens, bytes_total/1e6, t_min, t_min/n_mmid, bytes_total/1e9/(t_min/1e3));

    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    int tok_max = argc > 1 ? atoi(argv[1]) : 3; // max tokens (MTP verify batch)

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "failed to init Metal backend\n");
        return 1;
    }

    printf("== CGC mmid bandwidth probe (Metal, 256E top8) ==\n");
    printf("M4 Max peak reference ~546 GB/s\n\n");

    for (int t = 1; t <= tok_max; t++) {
        probe(backend, GGML_TYPE_IQ2_S,   512, 2048, t); // gate/up, 8 experts shared
        probe(backend, GGML_TYPE_IQ3_XXS, 2048, 512, t); // down, 8 experts shared
    }

    // decisive: same total bytes, different work/occupancy packing
    // tok=3/used=8 : 8 experts x 3 tokens = 24 (expert,token) pairs, 24 slabs
    // tok=1/used=24: 24 distinct experts = 24 slabs  -> same bytes, fewer tokens
    probe(backend, GGML_TYPE_IQ2_S,   512, 2048, 1, 24); // 24 distinct experts
    probe(backend, GGML_TYPE_IQ3_XXS, 2048, 512, 1, 24);
    probe(backend, GGML_TYPE_IQ2_S,   512, 2048, 3, 24); // real 3-token routing, distinct experts

    printf("\n-- in-graph (40 layers, one graph_compute, per-mmid cost) --\n");
    probe_layers(backend, 40, 3, 16);   // 3-token verify, ~16 distinct experts/layer (real sharing)
    probe_layers(backend, 40, 3, 24);   // worst case: no sharing
    probe_layers(backend, 40, 1, 8);    // single-token reference (8 experts)

    ggml_backend_free(backend);
    return 0;
}
