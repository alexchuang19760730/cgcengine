// CGC verify decomposition probe v4 — CHAINED graphs (real verify is a sequential
// layer dependency chain, so independent-branch probes under-measure via GPU overlap).
// Each class builds a realistic sequential chain at real qwen36 shapes/types
// (extracted from Nail Qwen3.6-35B-A3B-MTP GGUF), 3-token MTP verify batch.
//
// v4: share read-only weight tensors across layers (latency measurement unaffected —
//     only used slabs are read per layer; memory stays ~1 copy per weight).
//     graph sized 16384 to avoid node overflow on COMBINED.
//
//   FULL_ATT   : 10 chained full-attn layers, Q6_K wq[2048->8192] wk/wv[2048->512]
//                wo[4096->2048] + rms_norm + flash_attn(n_head16, kv2, hd256, N_KV)
//   LINEAR_ATT : 30 chained linear-attn layers, Q6_K wqkv[2048->8192] gate[2048->4096]
//                ssm_out[4096->2048] (+ small F32 conv/gdn EXCLUDED)
//   ATT_MIX    : 40 chained layers, every 4th = full-attn else linear-attn (real topology)
//   FFN        : 40 chained FFN layers, gate IQ2_S + up IQ2_S + down IQ3_S (256E top-8)
//                + gate_inp F32 + shared-expert Q6_K  (3 mmid / layer)
//   NORM_HEAD  : 80 rms_norm chain + output_norm + head Q6_K[2048->248320]
//   COMBINED   : 40 chained blocks mixing attn+ffn + head (approximation of real verify)
//
// Usage: decomp_probe [N_KV]   (N_KV = attention KV cache length, default 512)

#include "ggml.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

static const int N_EXP = 256;
static const int N_USED = 8;    // real qwen35moe.expert_used_count (=8); ids are [8, n_tokens]
static const int N_TOK = 3;
static int N_KV = 512;          // attention KV cache length (argv[1])

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

struct G {
    ggml_context * ctx;
    ggml_cgraph  * gf;
    std::vector<ggml_tensor*> leaves;   // activation roots needing data
    std::vector<ggml_tensor*> extras;   // other inputs (ids, mask, mmid x) needing data
    double bytes = 0;                   // bytes READ per graph_compute (counted by caller)

    G() {
        ggml_init_params ip = { 4096ull*1024*1024, nullptr, true };
        ctx = ggml_init(ip);
        gf  = ggml_new_graph_custom(ctx, 16384, false);
    }
    ~G() { ggml_free(ctx); }

    ggml_tensor * norm(ggml_tensor * x, int64_t d) {
        return ggml_rms_norm(ctx, x, 1e-6f);
    }
    ggml_tensor * leaf(int64_t k, int64_t tok) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tok);
        leaves.push_back(x);
        return x;
    }
    ggml_tensor * extra(ggml_type t, int64_t k, int64_t m) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, t, k, m);
        extras.push_back(x);
        return x;
    }
    ggml_tensor * extra3(ggml_type t, int64_t k, int64_t m, int64_t n) {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, t, k, m, n);
        extras.push_back(x);
        return x;
    }
    ggml_tensor * extra4(ggml_type t, int64_t a, int64_t b, int64_t c, int64_t d) {
        ggml_tensor * x = ggml_new_tensor_4d(ctx, t, a, b, c, d);
        extras.push_back(x);
        return x;
    }
    void expand(ggml_tensor * t) { ggml_build_forward_expand(gf, t); }

    double run(ggml_backend_t backend, int n_rep = 50) {
        ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (auto * x : leaves) {
            std::vector<float> f(ggml_nelements(x), 1.0f);
            ggml_backend_tensor_set(x, f.data(), 0, f.size()*sizeof(float));
        }
        for (auto * x : extras) {
            if (x->type == GGML_TYPE_I32) {
                std::vector<int32_t> f(ggml_nelements(x));
                for (size_t i = 0; i < f.size(); i++) f[i] = (int32_t)(i % N_USED);
                ggml_backend_tensor_set(x, f.data(), 0, f.size()*sizeof(int32_t));
            } else if (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16) {
                std::vector<uint8_t> b(ggml_nbytes(x), 0x00);
                ggml_backend_tensor_set(x, b.data(), 0, b.size());
            } else {
                std::vector<uint8_t> b(ggml_nbytes(x), 0xA5);
                ggml_backend_tensor_set(x, b.data(), 0, b.size());
            }
        }
        for (int i = 0; i < 10; i++) {
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
        }
        double t_min = 1e30;
        for (int i = 0; i < n_rep; i++) {
            double t0 = now_ms();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            double dt = now_ms() - t0;
            if (dt < t_min) t_min = dt;
        }
        return t_min;
    }
};

// ---- shared weight bundles (one copy, reused across all layers of a test) ----
struct FAW { ggml_tensor * wqg, * wk, * wv, * wo; };   // full attn
struct LAW { ggml_tensor * wqkv, * wz, * wo; };        // linear attn
struct LAW2 {                                          // linear attn + real SSM glue
    ggml_tensor * wqkv, * wz, * wo;                    // Q6_K [2048->8192], [2048->4096], [4096->2048]
    ggml_tensor * w_beta, * w_alpha, * w_a, * w_dt;    // F32 [2048->32], dt/a [32]
    ggml_tensor * conv_kernel;                         // F32 [4, 8192]
    ggml_tensor * ssm_norm_w;                          // F32 [128, 32]
};
struct FFNW {                                          // moe ffn
    ggml_tensor * w_gate, * w_up, * w_down;            // 3D [k, m, N_EXP]
    ggml_tensor * w_su, * w_sg, * w_sd;                // 2D Q6_K shared expert
};

static double bytes_of(ggml_tensor * t) { return (double) ggml_nbytes(t); }

static LAW2 make_law2(G & g) {
    LAW2 w;
    w.wqkv = g.extra(GGML_TYPE_Q6_K, 2048, 8192);
    w.wz   = g.extra(GGML_TYPE_Q6_K, 2048, 4096);
    w.wo   = g.extra(GGML_TYPE_Q6_K, 4096, 2048);
    w.w_beta   = g.extra(GGML_TYPE_F32, 2048, 32);
    w.w_alpha  = g.extra(GGML_TYPE_F32, 2048, 32);
    w.w_a      = g.extra(GGML_TYPE_F32, 32, 1);
    w.w_dt     = g.extra(GGML_TYPE_F32, 32, 1);
    w.conv_kernel = g.extra(GGML_TYPE_F32, 4, 8192);
    w.ssm_norm_w = g.extra(GGML_TYPE_F32, 128, 32);
    return w;
}

// one full-attn block: cur[2048,tok] -> out[2048,tok]  (real types Q6_K)
static ggml_tensor * block_full_attn(G & g, ggml_tensor * cur, const FAW & w) {
    ggml_tensor * h = g.norm(cur, 2048);
    g.expand(h);
    ggml_tensor * qg = ggml_mul_mat(g.ctx, w.wqg, h);   // QG joint [8192, tok]
    ggml_tensor * kt = ggml_mul_mat(g.ctx, w.wk,  h);
    ggml_tensor * v  = ggml_mul_mat(g.ctx, w.wv,  h);
    g.expand(qg); g.expand(kt); g.expand(v);
    // QG joint [8192,tok] -> Q part [4096,tok] (head-major) + gate part [4096,tok]
    const size_t esz = 4;
    ggml_tensor * qv = ggml_view_3d(g.ctx, qg, 256, 16, N_TOK, esz*256*2, esz*256*2*16, 0);
    qv = ggml_cont_2d(g.ctx, qv, 4096, N_TOK);
    ggml_tensor * q4 = ggml_reshape_4d(g.ctx, qv, 256, 16, N_TOK, 1);   // (hd, h, tok, seq)
    // K/V over the cached sequence (N_KV positions) like the real KV-cache view
    ggml_tensor * k4 = ggml_reshape_4d(g.ctx, kt, 256, 2,  N_TOK, 1);
    ggml_tensor * v4 = ggml_reshape_4d(g.ctx, v,  256, 2,  N_TOK, 1);
    ggml_tensor * kcache = g.extra3(GGML_TYPE_F16, 256, 2, N_KV);
    ggml_tensor * vcache = g.extra3(GGML_TYPE_F16, 256, 2, N_KV);
    // permute(0,2,1,3) like build_attn_mha, then flash attn over N_KV
    ggml_tensor * qp = ggml_permute(g.ctx, q4, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(g.ctx, kcache, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(g.ctx, vcache, 0, 2, 1, 3);
    ggml_tensor * mask = g.extra(GGML_TYPE_F16, N_TOK, N_KV);
    ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qp, kp, vp, mask, 1.0f/sqrtf(256.0f), 0.0f, 0.0f);
    g.expand(fa);                                          // [256,16,tok,1]
    ggml_tensor * fa2 = ggml_reshape_2d(g.ctx, fa, 4096, N_TOK); // [4096, tok]
    // gate (second 4096 of QG) -> sigmoid -> broadcast-mul over fa
    ggml_tensor * gv = ggml_view_3d(g.ctx, qg, 256, 16, N_TOK, esz*256*2, esz*256*2*16, esz*256);
    gv = ggml_cont_2d(g.ctx, gv, 4096, N_TOK);
    ggml_tensor * gs = ggml_sigmoid(g.ctx, gv);
    g.expand(gs);
    ggml_tensor * attn = ggml_mul(g.ctx, fa2, gs);         // [4096, tok]
    ggml_tensor * o = ggml_mul_mat(g.ctx, w.wo, attn);     // wo 4096->2048
    g.expand(o);
    ggml_tensor * r = ggml_add(g.ctx, cur, ggml_reshape_2d(g.ctx, o, 2048, N_TOK));
    g.expand(r);
    return r;
}

// one linear-attn block (matmul-dominated; small F32 conv/gdn/norm-gated excluded)
static ggml_tensor * block_linear_attn(G & g, ggml_tensor * cur, const LAW & w) {
    ggml_tensor * h = g.norm(cur, 2048);
    g.expand(h);
    ggml_tensor * qkv = ggml_mul_mat(g.ctx, w.wqkv, h);
    ggml_tensor * z   = ggml_mul_mat(g.ctx, w.wz,   h);
    g.expand(qkv); g.expand(z);
    ggml_tensor * attn = ggml_mul_mat(g.ctx, w.wo, ggml_reshape_2d(g.ctx, z, 4096, N_TOK));
    g.expand(attn);
    ggml_tensor * r = ggml_add(g.ctx, cur, attn);
    g.expand(r);
    return r;
}

// REAL linear-attn block: matmuls + beta/alpha activations + ssm_conv + l2_norm x2
// + fused gated_delta_net + gated rms-norm. Shapes from qwen35moe:
//   d_state=128 (S_v), n_group=16 (H_k), dt_rank=32 (H_v), d_inner=4096, conv_k=4
static ggml_tensor * block_linear_attn2(G & g, ggml_tensor * cur, const LAW2 & w) {
    ggml_tensor * h = g.norm(cur, 2048);
    g.expand(h);
    // joint QKV + z projections (matmul-dominated, Q6_K)
    ggml_tensor * qkv = ggml_mul_mat(g.ctx, w.wqkv, h); // [8192, tok]
    ggml_tensor * z   = ggml_mul_mat(g.ctx, w.wz,   h); // [4096, tok]
    g.expand(qkv); g.expand(z);
    // beta: mm [2048->32] -> sigmoid -> [1,32,tok,1]
    ggml_tensor * beta = ggml_mul_mat(g.ctx, w.w_beta, h);
    g.expand(beta);
    ggml_tensor * b4 = ggml_sigmoid(g.ctx, ggml_reshape_4d(g.ctx, beta, 1, 32, N_TOK, 1));
    g.expand(b4);
    // alpha: mm [2048->32] + dt, softplus, *A -> gate [1,32,tok,1]
    ggml_tensor * alpha = ggml_mul_mat(g.ctx, w.w_alpha, h);
    g.expand(alpha);
    ggml_tensor * a_sp = ggml_softplus(g.ctx, ggml_add(g.ctx, alpha, w.w_dt));
    g.expand(a_sp);
    ggml_tensor * g4 = ggml_reshape_4d(g.ctx, ggml_mul(g.ctx, a_sp, w.w_a), 1, 32, N_TOK, 1);
    g.expand(g4);
    // SSM conv1d: conv_in [6,8192,1] (3 state + 3 tok) x kernel [4,8192] -> silu -> [8192,3,1]
    ggml_tensor * conv_in = g.extra3(GGML_TYPE_F32, 6, 8192, 1);
    ggml_tensor * cq = ggml_silu(g.ctx, ggml_ssm_conv(g.ctx, conv_in, w.conv_kernel));
    g.expand(cq);
    // extract q/k/v slabs from conv output (per-token slab = row of 8192 floats)
    const size_t e     = ggml_element_size(cq);
    const size_t slab  = ggml_row_size(cq->type, 8192);
    ggml_tensor * q_raw = ggml_view_3d(g.ctx, cq, 128, 16, N_TOK, ggml_row_size(cq->type,128), slab, 0);
    ggml_tensor * k_raw = ggml_view_3d(g.ctx, cq, 128, 16, N_TOK, ggml_row_size(cq->type,128), slab, 128*16*e);
    ggml_tensor * v_raw = ggml_view_3d(g.ctx, cq, 128, 32, N_TOK, ggml_row_size(cq->type,128), slab, 128*16*2*e);
    // l2-norm q and k (real model normalizes before GDN)
    ggml_tensor * qn = ggml_l2_norm(g.ctx, q_raw, 1e-5f);
    ggml_tensor * kn = ggml_l2_norm(g.ctx, k_raw, 1e-5f);
    g.expand(qn); g.expand(kn);
    // recurrent state s [128,128,32,1]
    ggml_tensor * s = g.extra4(GGML_TYPE_F32, 128, 128, 32, 1);
    // fused Gated Delta Net (K=1): result [4096, tok*seq + state_rows, 1, 1]
    ggml_tensor * gdn = ggml_gated_delta_net(g.ctx, qn, kn, v_raw, g4, b4, s, 1);
    g.expand(gdn);
    // view the per-token attn output [128,32,tok,1] out of the flat result
    ggml_tensor * gdn_out = ggml_view_4d(g.ctx, gdn, 128, 32, N_TOK, 1,
            ggml_row_size(gdn->type, 128),
            ggml_row_size(gdn->type, 128*32),
            ggml_row_size(gdn->type, 128*32*N_TOK), 0);
    g.expand(gdn_out);
    // gated norm: rms_norm(gdn) * silu(z)
    ggml_tensor * z4 = ggml_reshape_4d(g.ctx, z, 128, 32, N_TOK, 1);
    ggml_tensor * zn = ggml_rms_norm(g.ctx, gdn_out, 1e-6f);
    g.expand(zn);
    ggml_tensor * zs = ggml_silu(g.ctx, z4);
    g.expand(zs);
    ggml_tensor * gated = ggml_mul(g.ctx, zn, zs);
    g.expand(gated);
    // output projection
    ggml_tensor * attn = ggml_mul_mat(g.ctx, w.wo, ggml_reshape_2d(g.ctx, gated, 4096, N_TOK));
    g.expand(attn);
    ggml_tensor * r = ggml_add(g.ctx, cur, attn);
    g.expand(r);
    return r;
}

// linear-attn SSM glue ONLY: ssm_conv + l2_norm x2 + gated_delta_net + gated norm
// (no qkv/z/wo matmuls). Isolates the small-op overhead from the matmul cost.
static ggml_tensor * block_linear_glue(G & g, ggml_tensor * cur, const LAW2 & w) {
    ggml_tensor * conv_in = g.extra3(GGML_TYPE_F32, 6, 8192, 1);
    ggml_tensor * cq = ggml_silu(g.ctx, ggml_ssm_conv(g.ctx, conv_in, w.conv_kernel));
    g.expand(cq);
    const size_t e     = ggml_element_size(cq);
    const size_t slab  = ggml_row_size(cq->type, 8192);
    ggml_tensor * q_raw = ggml_view_3d(g.ctx, cq, 128, 16, N_TOK, ggml_row_size(cq->type,128), slab, 0);
    ggml_tensor * k_raw = ggml_view_3d(g.ctx, cq, 128, 16, N_TOK, ggml_row_size(cq->type,128), slab, 128*16*e);
    ggml_tensor * v_raw = ggml_view_3d(g.ctx, cq, 128, 32, N_TOK, ggml_row_size(cq->type,128), slab, 128*16*2*e);
    ggml_tensor * qn = ggml_l2_norm(g.ctx, q_raw, 1e-5f);
    ggml_tensor * kn = ggml_l2_norm(g.ctx, k_raw, 1e-5f);
    g.expand(qn); g.expand(kn);
    ggml_tensor * g4 = g.extra4(GGML_TYPE_F32, 1, 32, N_TOK, 1);
    ggml_tensor * b4 = g.extra4(GGML_TYPE_F32, 1, 32, N_TOK, 1);
    ggml_tensor * s  = g.extra4(GGML_TYPE_F32, 128, 128, 32, 1);
    ggml_tensor * gdn = ggml_gated_delta_net(g.ctx, qn, kn, v_raw, g4, b4, s, 1);
    g.expand(gdn);
    ggml_tensor * gdn_out = ggml_view_4d(g.ctx, gdn, 128, 32, N_TOK, 1,
            ggml_row_size(gdn->type, 128),
            ggml_row_size(gdn->type, 128*32),
            ggml_row_size(gdn->type, 128*32*N_TOK), 0);
    g.expand(gdn_out);
    ggml_tensor * zn = ggml_rms_norm(g.ctx, gdn_out, 1e-6f);
    g.expand(zn);
    ggml_tensor * zs = g.extra4(GGML_TYPE_F32, 128, 32, N_TOK, 1);
    g.expand(zs);
    ggml_tensor * gated = ggml_mul(g.ctx, zn, zs);
    g.expand(gated);
    // artificial chain to keep the sequence dependency (glue alone has no data dep)
    ggml_tensor * r = ggml_add(g.ctx, cur, ggml_view_2d(g.ctx, gated, 2048, N_TOK, gated->nb[2], 0));
    g.expand(r);
    return r;
}

// one FFN block with REAL routing: gate_inp -> softmax -> argsort_top_k(8) -> get_rows
// -> softmax-weight + sum_rows normalization -> 3 mmid + swiglu + weighted expert sum
// + shared expert Q6_K. Uses GPU-derived ids (argsort) instead of a fixed ids leaf.
static ggml_tensor * block_ffn_route(G & g, ggml_tensor * cur, const FFNW & w) {
    ggml_tensor * h = g.norm(cur, 2048);
    g.expand(h);
    // routing proj (F32) [n_expert, tok]
    ggml_tensor * gi = ggml_mul_mat(g.ctx, g.extra(GGML_TYPE_F32, 2048, N_EXP), h);
    g.expand(gi);
    ggml_tensor * probs = ggml_soft_max(g.ctx, gi);                 // [256, tok]
    g.expand(probs);
    ggml_tensor * ids = ggml_argsort_top_k(g.ctx, probs, N_USED);   // [8, tok] I32
    g.expand(ids);
    // weights = get_rows(probs, ids) -> softmax-weight -> sum-rows normalize
    ggml_tensor * probs3 = ggml_reshape_3d(g.ctx, probs, 1, N_EXP, N_TOK);
    ggml_tensor * wts0 = ggml_get_rows(g.ctx, probs3, ids);          // [1, 8, tok]
    g.expand(wts0);
    ggml_tensor * wts2 = ggml_reshape_2d(g.ctx, wts0, N_USED, N_TOK);
    ggml_tensor * wts_sm = ggml_soft_max(g.ctx, wts2);               // [8, tok]
    g.expand(wts_sm);
    ggml_tensor * wsum = ggml_sum_rows(g.ctx, wts_sm);               // [1, tok]
    g.expand(wsum);
    ggml_tensor * wnorm = ggml_div(g.ctx, wts_sm, ggml_clamp(g.ctx, wsum, 6.1e-5f, INFINITY));
    g.expand(wnorm);
    ggml_tensor * wts = ggml_reshape_3d(g.ctx, wnorm, 1, N_USED, N_TOK); // [1, 8, tok]
    // expert inputs b=[k, 1, tok] broadcast over N_USED via ids (like real build_moe_ffn)
    ggml_tensor * xg = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * xu = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, w.w_gate, xg, ids); // [512, 8, tok]
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, w.w_up,   xu, ids); // [512, 8, tok]
    g.expand(gate); g.expand(up);
    ggml_tensor * glu = ggml_swiglu_split(g.ctx, gate, up);
    g.expand(glu);
    ggml_tensor * xd = g.extra3(GGML_TYPE_F32, 512, 1, N_TOK);
    ggml_tensor * dn = ggml_mul_mat_id(g.ctx, w.w_down, xd, ids); // [2048, 8, tok]
    g.expand(dn);
    // weight by routing probs then sum over N_USED expert slices
    ggml_tensor * experts = ggml_mul(g.ctx, dn, wts);       // [2048, 8, tok]
    g.expand(experts);
    ggml_tensor * sum = ggml_view_2d(g.ctx, experts, 2048, N_TOK, experts->nb[2], 0*experts->nb[1]);
    g.expand(sum);
    for (int i = 1; i < N_USED; i++) {
        ggml_tensor * sl = ggml_view_2d(g.ctx, experts, 2048, N_TOK, experts->nb[2], i*experts->nb[1]);
        g.expand(sl);
        sum = ggml_add(g.ctx, sum, sl);
        g.expand(sum);
    }
    // shared expert Q6_K: up/gate/down + sigmoid gate
    ggml_tensor * su = ggml_mul_mat(g.ctx, w.w_su, h);
    ggml_tensor * sg = ggml_mul_mat(g.ctx, w.w_sg, h);
    ggml_tensor * sd = ggml_mul_mat(g.ctx, w.w_sd, ggml_silu(g.ctx, ggml_add(g.ctx, su, sg)));
    g.expand(sd);
    ggml_tensor * moe = ggml_add(g.ctx, sum, sd);
    ggml_tensor * r = ggml_add(g.ctx, cur, moe);
    g.expand(r);
    return r;
}

// mmid-only FFN, merged gate+up layout: 1 mmid [1024,8,tok] + down mmid = 2 mmid/layer
// (gemma4-style fused gate_up tensor; qwen35moe uses 3 separate mmid today).
static ggml_tensor * block_ffn_mmid_gu(G & g, ggml_tensor * cur, const FFNW & w, ggml_tensor * ids, ggml_tensor * w_gu) {
    ggml_tensor * xg = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * gu = ggml_mul_mat_id(g.ctx, w_gu, xg, ids); // [1024, 8, tok]
    g.expand(gu);
    ggml_tensor * gv = ggml_view_3d(g.ctx, gu, 512, N_USED, N_TOK, gu->nb[1], gu->nb[2], 0);
    ggml_tensor * uv = ggml_view_3d(g.ctx, gu, 512, N_USED, N_TOK, gu->nb[1], gu->nb[2], 512*ggml_element_size(gu));
    ggml_tensor * glu = ggml_swiglu_split(g.ctx, gv, uv);
    g.expand(glu);
    ggml_tensor * xd = g.extra3(GGML_TYPE_F32, 512, 1, N_TOK);
    ggml_tensor * dn = ggml_mul_mat_id(g.ctx, w.w_down, xd, ids); // [2048, 8, tok]
    g.expand(dn);
    return dn;
}

// ---- FFN_PACK (turbo probe): pack G layers' experts into ONE mmid dispatch ----
// Stack G layer-sets into a single [k, m, G*256] weight; input batch = G*3 (layers x tokens).
// grid z = ne20*ne21 = 8 * G*3 = 24G per dispatch. 40 layers => 2*ceil(40/G) dispatches.
// G=1 reproduces FFN_GU1 (2 mmid/layer, 80 dispatches). G>1 = fewer, bigger dispatches.
static double ffn_pack_time(ggml_backend_t backend, int GP) {
    G g;
    ggml_tensor * w_gu = g.extra3(GGML_TYPE_IQ2_S, 2048, 1024, GP*N_EXP); // [in, gate|up, G*E]
    ggml_tensor * w_dn = g.extra3(GGML_TYPE_IQ3_S, 512,  2048, GP*N_EXP); // [in, out,   G*E]
    ggml_tensor * ids  = g.extra(GGML_TYPE_I32, N_USED, GP*N_TOK);
    ggml_tensor * xg   = g.extra3(GGML_TYPE_F32, 2048, 1, GP*N_TOK);
    ggml_tensor * xd   = g.extra3(GGML_TYPE_F32, 512,  1, GP*N_TOK);
    const int n_groups = (40 + GP - 1) / GP;
    for (int grp = 0; grp < n_groups; grp++) {
        ggml_tensor * gu = ggml_mul_mat_id(g.ctx, w_gu, xg, ids); // [1024, 8, G*3]
        ggml_tensor * dn = ggml_mul_mat_id(g.ctx, w_dn, xd, ids); // [2048, 8, G*3]
        g.expand(gu); g.expand(dn);
    }
    return g.run(backend);
}

// one FFN block (MoE, real 3 mmid/layer): gate_inp F32 -> gate/up/down mmid
// (routing skipped, fixed ids + weights) + shared expert Q6_K
static ggml_tensor * block_ffn(G & g, ggml_tensor * cur, const FFNW & w, ggml_tensor * ids, ggml_tensor * wts) {
    ggml_tensor * h = g.norm(cur, 2048);
    g.expand(h);
    // gate_inp (routing proj, F32) [n_expert, tok]
    ggml_tensor * gi = ggml_mul_mat(g.ctx, g.extra(GGML_TYPE_F32, 2048, N_EXP), h);
    g.expand(gi);
    // expert inputs b=[k, 1, tok] broadcast over N_USED via ids (like real build_moe_ffn)
    ggml_tensor * xg = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * xu = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, w.w_gate, xg, ids); // [512, N_USED, tok]
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, w.w_up,   xu, ids); // [512, N_USED, tok]
    g.expand(gate); g.expand(up);
    // swiglu: fused silu(gate)*up (real model uses ggml_swiglu_split)
    ggml_tensor * glu = ggml_swiglu_split(g.ctx, gate, up);
    g.expand(glu);
    ggml_tensor * xd = g.extra3(GGML_TYPE_F32, 512, 1, N_TOK);
    ggml_tensor * dn = ggml_mul_mat_id(g.ctx, w.w_down, xd, ids); // [2048, N_USED, tok]
    g.expand(dn);
    // weight by routing probs [1, N_USED, tok] then sum over N_USED expert slices
    ggml_tensor * experts = ggml_mul(g.ctx, dn, wts);       // [2048, N_USED, tok]
    g.expand(experts);
    ggml_tensor * sum = ggml_view_2d(g.ctx, experts, 2048, N_TOK, experts->nb[2], 0*experts->nb[1]);
    g.expand(sum);
    for (int i = 1; i < N_USED; i++) {
        ggml_tensor * sl = ggml_view_2d(g.ctx, experts, 2048, N_TOK, experts->nb[2], i*experts->nb[1]);
        g.expand(sl);
        sum = ggml_add(g.ctx, sum, sl);
        g.expand(sum);
    }
    // shared expert Q6_K: up/gate/down + sigmoid gate
    ggml_tensor * su = ggml_mul_mat(g.ctx, w.w_su, h);
    ggml_tensor * sg = ggml_mul_mat(g.ctx, w.w_sg, h);
    ggml_tensor * sd = ggml_mul_mat(g.ctx, w.w_sd, ggml_silu(g.ctx, ggml_add(g.ctx, su, sg)));
    g.expand(sd);
    ggml_tensor * moe = ggml_add(g.ctx, sum, sd);
    ggml_tensor * r = ggml_add(g.ctx, cur, moe);
    g.expand(r);
    return r;
}

// mmid-only FFN: just the 3 matmul-id ops + fused swiglu (no norm/gate_inp/sum/shared)
// isolates the pure mmid cost from the glue ops (routing, expert-sum, shared expert).
static ggml_tensor * block_ffn_mmid(G & g, ggml_tensor * cur, const FFNW & w, ggml_tensor * ids) {
    ggml_tensor * xg = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * xu = g.extra3(GGML_TYPE_F32, 2048, 1, N_TOK);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, w.w_gate, xg, ids); // [512, N_USED, tok]
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, w.w_up,   xu, ids); // [512, N_USED, tok]
    g.expand(gate); g.expand(up);
    ggml_tensor * glu = ggml_swiglu_split(g.ctx, gate, up);
    g.expand(glu);
    ggml_tensor * xd = g.extra3(GGML_TYPE_F32, 512, 1, N_TOK);
    ggml_tensor * dn = ggml_mul_mat_id(g.ctx, w.w_down, xd, ids); // [2048, N_USED, tok]
    g.expand(dn);
    return dn;
}

int main(int argc, char ** argv) {
    if (argc > 1) N_KV = atoi(argv[1]);
    setvbuf(stdout, nullptr, _IONBF, 0); // unbuffered: results survive the teardown assert

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "no metal backend\n"); return 1; }
    printf("== CGC verify decomposition (chained, tok=%d, N_KV=%d, Metal) ==\n\n", N_TOK, N_KV);

    // ---- FULL_ATT x10 chained ----
    {
        G g;
        FAW w = { g.extra(GGML_TYPE_Q6_K, 2048, 8192),
                  g.extra(GGML_TYPE_Q6_K, 2048, 512),
                  g.extra(GGML_TYPE_Q6_K, 2048, 512),
                  g.extra(GGML_TYPE_Q6_K, 4096, 2048) };
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 10; l++) {
            cur = block_full_attn(g, cur, w);
            g.bytes += bytes_of(w.wqg) + bytes_of(w.wk) + bytes_of(w.wv) + bytes_of(w.wo);
        }
        double t = g.run(backend);
        printf("FULL_ATT   10 chained layers        | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- LIN_ATT_MM x30 chained (3 matmuls only, no SSM glue) ----
    {
        G g;
        LAW w = { g.extra(GGML_TYPE_Q6_K, 2048, 8192),
                  g.extra(GGML_TYPE_Q6_K, 2048, 4096),
                  g.extra(GGML_TYPE_Q6_K, 4096, 2048) };
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 30; l++) {
            cur = block_linear_attn(g, cur, w);
            g.bytes += bytes_of(w.wqkv) + bytes_of(w.wz) + bytes_of(w.wo);
        }
        double t = g.run(backend);
        printf("LIN_ATT_MM 30 layers, 3 mm only     | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- LIN_GLUE x30 chained (SSM glue only: conv+l2norm+gdn+normgated) ----
    {
        G g;
        LAW2 w = make_law2(g);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 30; l++) {
            cur = block_linear_glue(g, cur, w);
        }
        double t = g.run(backend);
        printf("LIN_GLUE   30 layers, ssm only      | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- LINEAR_ATT x30 chained (FULL real: matmuls + SSM glue) ----
    {
        G g;
        LAW2 w = make_law2(g);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 30; l++) {
            cur = block_linear_attn2(g, cur, w);
            g.bytes += bytes_of(w.wqkv) + bytes_of(w.wz) + bytes_of(w.wo)
                     + bytes_of(w.w_beta) + bytes_of(w.w_alpha) + bytes_of(w.w_a) + bytes_of(w.w_dt)
                     + bytes_of(w.conv_kernel);
        }
        double t = g.run(backend);
        printf("LINEAR_ATT 30 chained layers (full) | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- ATT_MIX x40 (every 4th full) ----
    {
        G g;
        FAW fa = { g.extra(GGML_TYPE_Q6_K, 2048, 8192),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 4096, 2048) };
        LAW2 la = make_law2(g);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 40; l++) {
            if (l % 4 == 3) {
                cur = block_full_attn(g, cur, fa);
                g.bytes += bytes_of(fa.wqg) + bytes_of(fa.wk) + bytes_of(fa.wv) + bytes_of(fa.wo);
            } else {
                cur = block_linear_attn2(g, cur, la);
                g.bytes += bytes_of(la.wqkv) + bytes_of(la.wz) + bytes_of(la.wo)
                         + bytes_of(la.w_beta) + bytes_of(la.w_alpha) + bytes_of(la.w_a) + bytes_of(la.w_dt)
                         + bytes_of(la.conv_kernel);
            }
        }
        double t = g.run(backend);
        printf("ATT_MIX    40 chained layers (10F+30L)| %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- FFN_MMID x40 (3 mmid + swiglu only, no glue) ----
    {
        G g;
        FFNW w = { g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ3_S, 512,  2048, N_EXP),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 512,  2048) };
        ggml_tensor * ids = g.extra(GGML_TYPE_I32, N_USED, N_TOK);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        const double per_layer =
            (bytes_of(w.w_gate) + bytes_of(w.w_up)) / N_EXP * N_USED +
             bytes_of(w.w_down) / N_EXP * N_USED;
        for (int l = 0; l < 40; l++) {
            cur = block_ffn_mmid(g, cur, w, ids);
            g.bytes += per_layer;
        }
        double t = g.run(backend);
        printf("FFN_MMID  40 layers, 3 mmid only    | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- FFN_GU1 x40 (merged gate+up mmid + down mmid = 2 mmid/layer) ----
    {
        G g;
        ggml_tensor * w_gu = g.extra3(GGML_TYPE_IQ2_S, 2048, 1024, N_EXP); // [in, gate|up, E] fused
        FFNW w = { nullptr, nullptr,
                   g.extra3(GGML_TYPE_IQ3_S, 512, 2048, N_EXP),
                   nullptr, nullptr, nullptr };
        ggml_tensor * ids = g.extra(GGML_TYPE_I32, N_USED, N_TOK);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        const double per_layer =
            bytes_of(w_gu) / N_EXP * N_USED +
            bytes_of(w.w_down) / N_EXP * N_USED;
        for (int l = 0; l < 40; l++) {
            cur = block_ffn_mmid_gu(g, cur, w, ids, w_gu);
            g.bytes += per_layer;
        }
        double t = g.run(backend);
        printf("FFN_GU1   40 layers, 2 mmid (GU+dn) | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- FFN_PACK turbo sweep (G layers per dispatch) ----
    {
        printf("FFN_PACK  turbo sweep: G layers packed into 1 mmid dispatch (grid z=24G)\n");
        const int Gs[] = { 1, 2, 4, 8, 12 };
        double t_base = 0.0;
        for (int G : Gs) {
            double t = ffn_pack_time(backend, G);
            if (G == 1) t_base = t;
            const int disp = 2*((40 + G - 1)/G);
            printf("  G=%2d  %3d dispatches  z=%3d/token | min %7.3f ms | per-layer %6.3f ms | speedup %6.2f x\n",
                   G, disp, 24*G, t, t/40, t_base/t);
        }
    }
    // ---- FFN x40 chained ----
    {
        G g;
        FFNW w = { g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ3_S, 512,  2048, N_EXP),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 512,  2048) };
        ggml_tensor * ids = g.extra(GGML_TYPE_I32, N_USED, N_TOK);
        ggml_tensor * wts = g.extra3(GGML_TYPE_F32, 1, N_USED, N_TOK);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        const double per_layer =
            (bytes_of(w.w_gate) + bytes_of(w.w_up)) / N_EXP * N_USED +
             bytes_of(w.w_down) / N_EXP * N_USED +
             bytes_of(w.w_su) + bytes_of(w.w_sg) + bytes_of(w.w_sd);
        for (int l = 0; l < 40; l++) {
            cur = block_ffn(g, cur, w, ids, wts);
            g.bytes += per_layer;
        }
        double t = g.run(backend);
        printf("FFN        40 chained layers (3 mmid)| %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- FFN_ROUTE x40 chained (3 mmid + REAL routing: softmax/argsort/get_rows/weight-norm) ----
    {
        G g;
        FFNW w = { g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ3_S, 512,  2048, N_EXP),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 512,  2048) };
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        const double per_layer =
            (bytes_of(w.w_gate) + bytes_of(w.w_up)) / N_EXP * N_USED +
             bytes_of(w.w_down) / N_EXP * N_USED +
             bytes_of(w.w_su) + bytes_of(w.w_sg) + bytes_of(w.w_sd);
        for (int l = 0; l < 40; l++) {
            cur = block_ffn_route(g, cur, w);
            g.bytes += per_layer;
        }
        double t = g.run(backend);
        printf("FFN_ROUTE 40 layers (3 mmid+routing)| %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- NORM_HEAD ----
    {
        G g;
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        for (int l = 0; l < 80; l++) { cur = g.norm(cur, 2048); g.expand(cur); }
        cur = g.norm(cur, 2048); g.expand(cur);
        ggml_tensor * head = g.extra(GGML_TYPE_Q6_K, 2048, 248320);
        ggml_tensor * logits = ggml_mul_mat(g.ctx, head, cur);
        g.expand(logits);
        g.bytes = bytes_of(head);
        double t = g.run(backend);
        printf("NORM_HEAD  80 norm + head Q6_K      | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }
    // ---- COMBINED (40 attn-mix + ffn + head) ----
    {
        G g;
        FAW fa = { g.extra(GGML_TYPE_Q6_K, 2048, 8192),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 4096, 2048) };
        LAW2 la = make_law2(g);
        FFNW w = { g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ2_S, 2048, 512, N_EXP),
                   g.extra3(GGML_TYPE_IQ3_S, 512,  2048, N_EXP),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 2048, 512),
                   g.extra(GGML_TYPE_Q6_K, 512,  2048) };
        ggml_tensor * ids = g.extra(GGML_TYPE_I32, N_USED, N_TOK);
        ggml_tensor * wts = g.extra3(GGML_TYPE_F32, 1, N_USED, N_TOK);
        ggml_tensor * cur = g.leaf(2048, N_TOK);
        const double ffn_layer =
            (bytes_of(w.w_gate) + bytes_of(w.w_up)) / N_EXP * N_USED +
             bytes_of(w.w_down) / N_EXP * N_USED +
             bytes_of(w.w_su) + bytes_of(w.w_sg) + bytes_of(w.w_sd);
        const double attn_layer =
            bytes_of(fa.wqg) + bytes_of(fa.wk) + bytes_of(fa.wv) + bytes_of(fa.wo);
        const double lattn_layer =
            bytes_of(la.wqkv) + bytes_of(la.wz) + bytes_of(la.wo)
          + bytes_of(la.w_beta) + bytes_of(la.w_alpha) + bytes_of(la.w_a) + bytes_of(la.w_dt)
          + bytes_of(la.conv_kernel);
        for (int l = 0; l < 40; l++) {
            if (l % 4 == 3) {
                cur = block_full_attn(g, cur, fa);
                g.bytes += attn_layer;
            } else {
                cur = block_linear_attn2(g, cur, la);
                g.bytes += lattn_layer;
            }
            cur = block_ffn(g, cur, w, ids, wts);
            g.bytes += ffn_layer;
        }
        cur = g.norm(cur, 2048); g.expand(cur);
        ggml_tensor * head = g.extra(GGML_TYPE_Q6_K, 2048, 248320);
        ggml_tensor * logits = ggml_mul_mat(g.ctx, head, cur);
        g.expand(logits);
        g.bytes += bytes_of(head);
        double t = g.run(backend);
        printf("COMBINED   40 attn+ffn + head      | %9.2f MB | min %7.3f ms | eff %6.1f GB/s\n", g.bytes/1e6, t, g.bytes/1e9/(t/1e3));
    }

    ggml_backend_free(backend);
    return 0;
}
