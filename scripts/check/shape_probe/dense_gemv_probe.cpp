// dense_gemv_probe.cpp -- per-shape GEMV achievable-bandwidth probe.
//
// WHY THIS EXISTS
//   On this fork there is NO instrument that can quote an achieved GB/s for the
//   dense GEMV shapes that dominate the decode step:
//     - CGC-GPUOPS runs with instrumentation on (step total 139-210 ms vs
//       79-98 ms in production) and yields absurd numbers (the router comes out
//       at 258 GB/s against a measured peak of 108.8 GB/s) -> unusable.
//     - test-backend-ops registers MUL_MAT only at 4096x14336, so no -p regex
//       can reach this model's shapes (-p 'type_a=iq4_xs,m=8192,...' -> 0 hits).
//   So we allocate ONE weight tensor of a given type/shape on the Metal device,
//   feed an r-column input, and time repeated MUL_MAT. No model load, no llama
//   graph, no CGC_* instrumentation: the only thing measured is the op.
//
// WHY TWO TIMING MODES -- AND WHY THEIR DIFFERENCE IS THE SIGNAL
//     sync : one dispatch per iteration + explicit wait => includes per-dispatch cost
//     batch: `batch` copies inside ONE graph, amortized => kernel-only cost
//   If sync >> batch the shape is launch-bound, not bandwidth-bound. That is
//   exactly the "launch-limited vs bandwidth-limited" question that could not
//   be answered before, and it decides whether ANY tile knob can help.
//
// WHY A STANDALONE BINARY INSTEAD OF cmake --build
//   The build products in this repo are version-controlled and shared: every
//   A/B here assumes "same build". Rebuilding would overwrite the dylib another
//   session is mapping. So: compile THIS file only, link the existing libs.
//
// WHY MUL_MAT_ID (MoE) IS A SEPARATE OP HERE
//   The expert bank is the one family that scales with BOTH the layer count and
//   the token count: each token reads n_expert_used slices of EVERY layer's bank,
//   so its bytes/step grow with n_tokens while the dense weights do not. That is
//   exactly the term MTP's verify batch pays, and pricing it needs ids -- a bank
//   indexed by expert is not a MUL_MAT, and pricing it as one would under-count
//   the read by 1/n_used.
//
//   One bank is allocated and the COPIES differ by their ids: copy i selects
//   experts [(i*n_used + j) % n_expert], so a batch of n_expert/n_used copies
//   reads the whole bank exactly once, with no expert read twice inside a graph.
//   That is the same DRAM traffic a batch of n copies over n distinct banks would
//   cost, in 1/n the device memory -- and unlike the dense case it needs no
//   approximation. --shared-weight instead repeats the first n_used experts in
//   every copy, which is the (unrealistic) upper bound.
//
// Usage:
//   ./cgc_shape_probe --type iq4_xs --k 2048 --out 8192 --r 1 \
//                     --mode both --iters 60 --batch 32 --json out.json
//   ./cgc_shape_probe --op mul_mat_id --type iq2_s --k 2048 --out 512 \
//                     --experts 256 --used 8 --tokens 4 --mode batch --batch 32
//
// Exit code 0 on success; non-zero if the shape cannot be built (e.g. the Metal
// backend refuses the type, or the allocation does not fit).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

const std::map<std::string, ggml_type> & types() {
    static const std::map<std::string, ggml_type> t = {
        {"f32",      GGML_TYPE_F32},
        {"f16",      GGML_TYPE_F16},
        {"bf16",     GGML_TYPE_BF16},
        {"q6_k",     GGML_TYPE_Q6_K},
        {"iq2_s",    GGML_TYPE_IQ2_S},
        {"iq3_s",    GGML_TYPE_IQ3_S},
        {"iq3_xxs",  GGML_TYPE_IQ3_XXS},
        {"iq4_xs",   GGML_TYPE_IQ4_XS},
        {"q8_0",    GGML_TYPE_Q8_0},
        {"q4_0",    GGML_TYPE_Q4_0},
        {"q4_k",    GGML_TYPE_Q4_K},
        {"q5_k",    GGML_TYPE_Q5_K},
        {"q2_k",    GGML_TYPE_Q2_K},
        {"q3_k",    GGML_TYPE_Q3_K},
        {"iq4_nl",  GGML_TYPE_IQ4_NL},
    };
    return t;
}

size_t align_up(size_t x, size_t a) { return a ? ((x + a - 1) / a) * a : x; }

// A host-side view of one device buffer, with a bump allocator for tensors.
struct Pool {
    ggml_backend_buffer_t buf = nullptr;
    char * base   = nullptr;
    size_t size   = 0;
    size_t used   = 0;
    size_t align  = 0;

    Pool(ggml_backend_t backend, size_t bytes) {
        buf   = ggml_backend_alloc_buffer(backend, bytes);
        if (!buf) { fprintf(stderr, "FATAL: alloc_buffer(%zu) failed\n", bytes); exit(2); }
        base  = (char *) ggml_backend_buffer_get_base(buf);
        size  = ggml_backend_buffer_get_size(buf);
        align = ggml_backend_buffer_get_alignment(buf);
        used  = 0;
    }
    ~Pool() { if (buf) { ggml_backend_buffer_free(buf); } }

    size_t place(struct ggml_tensor * t) {
        const size_t need = ggml_backend_buffer_get_alloc_size(buf, t);
        used = align_up(used, align);
        if (used + need > size) {
            fprintf(stderr, "FATAL: pool exhausted (%zu + %zu > %zu) -- raise --pool-mib\n",
                    used, need, size);
            exit(2);
        }
        if (ggml_backend_tensor_alloc(buf, t, base + used) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: tensor_alloc failed\n");
            exit(2);
        }
        used += need;
        return need;
    }
};

// Deterministic filler. Values do not matter for timing, but they must be the
// SAME every run: a different seed changes the memory footprint not at all yet
// makes "before/after" comparisons irreproducible to anyone repeating them.
void fill(struct ggml_tensor * t, const std::string & type, unsigned seed) {
    const size_t nbytes = ggml_nbytes(t);
    std::mt19937 rng(seed);
    std::vector<uint8_t> tmp(nbytes);
    if (type == "f32" || type == "f16" || type == "bf16") {
        std::uniform_real_distribution<float> d(-0.05f, 0.05f);
        if (type == "f32") {
            float * p = (float *) tmp.data();
            for (size_t i = 0; i < nbytes / 4; i++) { p[i] = d(rng); }
        } else {
            uint16_t * p = (uint16_t *) tmp.data();
            for (size_t i = 0; i < nbytes / 2; i++) {
                p[i] = ggml_fp32_to_fp16(d(rng));
            }
        }
    } else {
        // Quantized weights: any bit pattern is a legal, finite tensor. We keep
        // away from 0x00/0xFF saturation only to avoid trivially degenerate rows.
        for (size_t i = 0; i < nbytes; i++) { tmp[i] = (uint8_t) (rng() & 0xFFu); }
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
}

double median(std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string arg(int argc, char ** argv, const char * key, const std::string & def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], key) == 0) { return argv[i + 1]; }
    }
    return def;
}
bool has(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) { if (strcmp(argv[i], key) == 0) { return true; } }
    return false;
}
std::vector<std::string> all(int argc, char ** argv, const char * key) {
    std::vector<std::string> out;
    for (int i = 1; i < argc - 1; i++) { if (strcmp(argv[i], key) == 0) { out.push_back(argv[i + 1]); } }
    return out;
}

double now_us() { return (double) ggml_time_us(); }

}  // namespace

int main(int argc, char ** argv) {
    if (has(argc, argv, "--help")) {
        fprintf(stderr,
            "usage: cgc_shape_probe --type NAME --k K --out N [--r 1] [--mode sync|batch|both]\n"
            "                       [--iters 60] [--batch 32] [--target-ms 800] [--pool-mib 512]\n"
            "                       [--set K=V]*\n"
            "                       [--json FILE]\n"
            "  --k    reduction dim (rows of the input vector)   == ggml ne00\n"
            "  --out  output dim (number of experts/features)    == ggml ne01\n"
            "  --r    input columns (n in test-backend-ops vars) == decode: 1, MTP K=4\n"
            "  --op mul_mat | mul_mat_id (MoE expert bank; then --experts/--used/--tokens apply)\n"
            "  --experts  experts in the bank        == ggml ne02 of `as`   (model: 256)\n"
            "  --used     experts per token          == n_expert_used     (model: 8)\n"
            "  --tokens   tokens per op (the MTP axis: 1 = decode, up to 4 = verify batch)\n");
        return 0;
    }

    for (const auto & kv : all(argc, argv, "--set")) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) { fprintf(stderr, "FATAL: --set needs K=V (got %s)\n", kv.c_str()); return 2; }
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        setenv(k.c_str(), v.c_str(), 1);
        fprintf(stderr, "env      %s=%s\n", k.c_str(), v.c_str());
    }

    const std::string tname = arg(argc, argv, "--type", "iq4_xs");
    const int64_t K         = std::stoll(arg(argc, argv, "--k",   "2048"));
    const int64_t N         = std::stoll(arg(argc, argv, "--out", "8192"));
    const int64_t R         = std::stoll(arg(argc, argv, "--r",   "1"));
    const std::string mode  = arg(argc, argv, "--mode", "both");
    const int      iters    = std::stoi(arg(argc, argv, "--iters", "60"));
    const int      batch    = std::stoi(arg(argc, argv, "--batch", "32"));
    // How long the batch mode keeps looping. The marginal model takes a DIFFERENCE of two of these
    // runs, so its noise is the noise of two timings added -- raising the target is the cheapest way
    // to buy accuracy in the number every conclusion rests on.
    const double   target_ms = std::stod(arg(argc, argv, "--target-ms", "800"));
    const size_t   pool_mib = std::stoull(arg(argc, argv, "--pool-mib", "512"));
    const std::string jpath = arg(argc, argv, "--json", "");
    const std::string opname = arg(argc, argv, "--op", "mul_mat");
    const int64_t E           = std::stoll(arg(argc, argv, "--experts", "256"));
    const int64_t U           = std::stoll(arg(argc, argv, "--used",    "8"));
    const int64_t T           = std::stoll(arg(argc, argv, "--tokens",  "1"));
    if (opname != "mul_mat" && opname != "mul_mat_id") {
        fprintf(stderr, "FATAL: --op must be mul_mat or mul_mat_id (got %s)\n", opname.c_str());
        return 2;
    }
    const bool mmid = (opname == "mul_mat_id");
    if (mmid && (E < 1 || U < 1 || U > E || T < 1)) {
        fprintf(stderr, "FATAL: mul_mat_id needs 1 <= used <= experts and tokens >= 1\n");
        return 2;
    }

    const auto & tbl = types();
    if (!tbl.count(tname)) {
        fprintf(stderr, "FATAL: unknown --type %s (probe knows %zu types)\n", tname.c_str(), tbl.size());
        return 2;
    }
    const ggml_type wtype = tbl.at(tname);

    // ---- backend ----------------------------------------------------------
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "FATAL: Metal backend unavailable\n"); return 2; }
    const char * bname = ggml_backend_name(backend);

    // ---- tensors: ctx0 is no_alloc, every tensor is placed by hand --------
    struct ggml_init_params p0 = { /*mem_size*/ 64u * 1024u * 1024u, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx0 = ggml_init(p0);
    if (!ctx0) { fprintf(stderr, "FATAL: ggml_init failed\n"); return 2; }

    struct ggml_tensor * x = mmid ? ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K, 1, T)
                                 : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, K, R);

    // ---- how many copies, and do they read the SAME weight? -----------------
    //
    // This is the difference between a number and a hallucination:
    //   SHARED weight  : copy 2..N find the weights already warmed in cache, so
    //                    the graph reads them ~once and every later op is nearly
    //                    free. That OVERSTATES bandwidth (measured: 73 GB/s).
    //   DISTINCT weight: each op reads its own ([K,N] tensor) exactly as a real
    //                    decode does -- every layer touches different weights.
    // Default is DISTINCT, because that is what the step actually pays.
    const bool distinct = !has(argc, argv, "--shared-weight");
    const int  nc       = (mode == "sync") ? 1 : batch;
    std::vector<struct ggml_tensor *> ws;
    std::vector<struct ggml_tensor *> outs;
    std::vector<struct ggml_tensor *> ids;
    if (mmid) {
        // ONE bank; the copies differ by their ids (see the header note). With --shared-weight every
        // copy reuses the first `used` experts, which is the unrealistic upper bound.
        ws.push_back(ggml_new_tensor_3d(ctx0, wtype, K, N, E));
        for (int i = 0; i < nc; i++) {
            struct ggml_tensor * id = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, U, T);
            ids.push_back(id);
            outs.push_back(ggml_mul_mat_id(ctx0, ws[0], x, id));
        }
    } else {
        for (int i = 0; i < nc; i++) {
            // DISTINCT -> one fresh weight per op; SHARED -> reuse ws[0] every time.
            if (ws.empty() || distinct) { ws.push_back(ggml_new_tensor_2d(ctx0, wtype, K, N)); }
            outs.push_back(ggml_mul_mat(ctx0, ws.back(), x));
        }
    }

    Pool pool(backend, pool_mib * 1024ull * 1024ull);
    const size_t w_bytes = pool.place(ws[0]);
    pool.place(x);
    for (size_t i = 1; i < ws.size(); i++) { pool.place(ws[i]); }
    for (auto * t : ids) { pool.place(t); }
    for (auto * t : outs) { pool.place(t); }
    if (pool.used > pool.size) { return 2; }

    for (size_t i = 0; i < ws.size(); i++) { fill(ws[i], tname, (unsigned) (0xC0FFEEu + i * 7919u)); }
    fill(x, "f32", 0xBEEFu);
    for (size_t i = 0; i < ids.size(); i++) {
        std::vector<int32_t> v((size_t) U * (size_t) T);
        for (int64_t t = 0; t < T; t++) {
            for (int64_t j = 0; j < U; j++) {
                const int64_t e = distinct ? ((int64_t) i * U + j) % E : j;
                v[(size_t) (t * U + j)] = (int32_t) e;
            }
        }
        ggml_backend_tensor_set(ids[i], v.data(), 0, v.size() * sizeof(int32_t));
    }

    // ---- graph ------------------------------------------------------------
    struct ggml_init_params pg = { /*mem_size*/ 16u * 1024u * 1024u, nullptr, false };
    struct ggml_context *  ctxg = ggml_init(pg);
    struct ggml_cgraph *   gf   = ggml_new_graph(ctxg);
    for (auto * t : outs) { ggml_build_forward_expand(gf, t); }

    // Weight bytes one op must read. MUL_MAT_ID reads only the selected experts; MUL_MAT reads its
    // own weight. Everything else (src1, ids, dst) is <2% of this and is not counted.
    const double per_expert_bytes = mmid ? (double) w_bytes / (double) E : (double) w_bytes;
    const double op_bytes         = mmid ? per_expert_bytes * (double) U * (double) T
                                         : (double) w_bytes;

    fprintf(stderr, "backend  %s\n", bname);
    if (mmid) {
        fprintf(stderr, "shape    MUL_MAT_ID type=%s k=%lld out=%lld experts=%lld used=%lld tokens=%lld\n",
                tname.c_str(), (long long) K, (long long) N, (long long) E, (long long) U, (long long) T);
        fprintf(stderr, "         bank %.2f MiB (%.3f MiB/expert), %zu op(s), reads %.3f MiB/op %s\n",
                (double) w_bytes / 1048576.0, per_expert_bytes / 1048576.0, outs.size(),
                op_bytes / 1048576.0,
                distinct ? "(ids cycle the bank: no expert read twice per graph)"
                         : "(SHARED ids: first experts only -- upper bound)");
    } else {
        fprintf(stderr, "shape    MUL_MAT type=%s k=%lld out=%lld r=%lld  weight %.2f MiB x%zu %s\n",
                tname.c_str(), (long long) K, (long long) N, (long long) R,
                (double) w_bytes / 1048576.0, ws.size(),
                distinct ? "(distinct, cold)" : "(SHARED weight -- upper bound only)");
    }
    if (!ggml_backend_supports_op(backend, outs[0])) {
        fprintf(stderr, "WARN: backend reports it does not support this op\n");
    }

    // warmup: first compute also compiles the pipeline, so exclude it from every statistic
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }

    // ---- mode "sync": one dispatch per timed iteration --------------------
    std::vector<double> sync_us;
    if (mode == "sync" || mode == "both") {
        sync_us.reserve(iters);
        for (int i = 0; i < iters; i++) {
            const double t0 = now_us();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "FATAL: graph_compute failed\n"); return 2;
            }
            ggml_backend_synchronize(backend);
            sync_us.push_back(now_us() - t0);
        }
    }

    // ---- mode "batch": one graph of n_copy dispatch(es) per timed iteration
    std::vector<double> per_op_us;
    if (mode == "batch" || mode == "both") {
        const int64_t target_us = (int64_t) (target_ms * 1000.0);   // default 0.8 s, like test-backend-ops
        int loops = 0;
        double total_us = 0.0;
        while (total_us < target_us) {
            const double t0 = now_us();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "FATAL: graph_compute failed (batch)\n"); return 2;
            }
            ggml_backend_synchronize(backend);
            total_us += (now_us() - t0);
            loops++;
        }
        per_op_us.push_back(total_us / (double) (loops * outs.size()));
        fprintf(stderr, "         batch: %d loops x %zu ops\n", loops, outs.size());
    }

    const double sync_med = median(sync_us);
    const double op_med   = median(per_op_us);
    // achieved bandwidth counts the weight read only; src1 + dst are <1% here.
    const double bytes = op_bytes;
    auto gbs = [&](double us) -> double {
        return (us > 0) ? (bytes / (us * 1e-6)) / (1024.0 * 1024.0 * 1024.0) : 0.0;
    };

    fprintf(stderr, "RESULT   sync  median %8.2f us/op  %7.2f GB/s\n", sync_med, gbs(sync_med));
    fprintf(stderr, "RESULT   batch median %8.2f us/op  %7.2f GB/s\n", op_med,  gbs(op_med));
    if (sync_us.size() >= 2) {
        fprintf(stderr, "         sync min %.2f max %.2f (n=%zu)\n",
                *std::min_element(sync_us.begin(), sync_us.end()),
                *std::max_element(sync_us.begin(), sync_us.end()), sync_us.size());
    }
    if (sync_med > 0 && op_med > 0) {
        fprintf(stderr, "         sync/batch = %.2f  (>1.2 => launch-bound, not bandwidth-bound)\n",
                sync_med / op_med);
    }

    if (!jpath.empty()) {
        FILE * f = fopen(jpath.c_str(), "w");
        if (f) {
            fprintf(f, "{\"op\":\"%s\",\"type\":\"%s\",\"k\":%lld,\"out\":%lld,\"r\":%lld,"
                       "\"experts\":%lld,\"used\":%lld,\"tokens\":%lld,"
                       "\"weight_bytes\":%zu,\"op_bytes\":%.1f,\"copies\":%zu,\"distinct\":%d,"
                       "\"backend\":\"%s\",\"sync_med_us\":%.3f,\"batch_med_us\":%.3f,"
                       "\"sync_gbs\":%.3f,\"batch_gbs\":%.3f,\"sync_samples\":[",
                    opname.c_str(), tname.c_str(), (long long) K, (long long) N, (long long) R,
                    (long long) E, (long long) U, (long long) T, w_bytes, op_bytes,
                    outs.size(), distinct ? 1 : 0, bname, sync_med, op_med, gbs(sync_med), gbs(op_med));
            for (size_t i = 0; i < sync_us.size(); i++) {
                fprintf(f, "%s%.3f", i ? "," : "", sync_us[i]);
            }
            fprintf(f, "]}\n");
            fclose(f);
        }
    }

    ggml_free(ctxg);
    ggml_free(ctx0);
    ggml_backend_free(backend);
    return 0;
}
