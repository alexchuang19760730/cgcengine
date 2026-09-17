// M1 work item 4 (canonical gather order): offline test A.
//
// Spec: docs/M1_WORKITEM4_CANONICAL_GATHER_ORDER_2026-09-17.md §3 test A --
//   "same set of 8 (expert, weight) pairs, fed in two DIFFERENT input position orders; with
//    canonical order on, the ffn_moe_out summary must be bit-identical".
//
// This is the part of work item 4 that needs no GPU, no server and no build of the project: the
// permutation lives in src/llama.cpp/src/llama-cgc-canon.h (header-only on purpose) and everything
// below is the arithmetic the engine performs with it.
//
// Build & run (seconds, safe while another session owns the GPU):
//     clang++ -std=c++17 -O2 -o /tmp/canon_order_selftest scripts/check/canon_order_selftest.cpp
//     /tmp/canon_order_selftest
//
// The controls are the point of the file. A permutation-invariance test that cannot fail proves
// nothing, so for every trial the harness also computes the sum WITHOUT canonicalisation and
// requires that it actually differs between the two input orders -- otherwise the fp32 chain would
// be associative here and the test would pass for the wrong reason.

#include "../../src/llama.cpp/src/llama-cgc-canon.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::printf("FAIL  %s\n", what);
        g_fail++;
    } else {
        std::printf("ok    %s\n", what);
    }
}

// The engine's aggregation: a left-to-right fp32 chain over the k positions (llama-graph.cpp
// "aggregate experts"). Written as a loop with a volatile-free but non-foldable form: the compiler
// must not reassociate it, which -O2 without -ffast-math does not.
static float sum_chain(const float * term, int64_t k) {
    float acc = term[0];
    for (int64_t i = 1; i < k; ++i) {
        acc = acc + term[i];
    }
    return acc;
}

// One trial: k (expert, weight) pairs arrive in some input position order. Returns the chain sum
// with and without canonicalisation, plus the canonical id sequence (for the ordering assertions).
struct trial_out {
    float sum_raw      = 0.0f;
    float sum_canon    = 0.0f;
    float sum_identity = 0.0f;
    std::vector<int32_t> canon_ids;
    std::vector<int32_t> canon_weights_perm; // which input position supplied canonical slot i
};

static trial_out run_trial(const std::vector<int32_t> & ids_in,
                           const std::vector<float>   & w_in,
                           const std::vector<float>   & expert_value) {
    const int64_t k = (int64_t) ids_in.size();
    trial_out o;

    std::vector<int32_t> perm((size_t) k), ids_c((size_t) k);
    cgc_canon_build_perm(ids_in.data(), perm.data(), 1, k, 1);
    cgc_canon_apply(ids_in.data(), perm.data(), ids_c.data(), k);
    o.canon_ids = ids_c;
    o.canon_weights_perm.resize((size_t) k);
    for (int64_t i = 0; i < k; ++i) {
        o.canon_weights_perm[(size_t) i] = perm[(size_t) i];
    }

    // term[i] = weight[i] * expert_value[ids[i]] -- exactly what the graph computes positionally.
    std::vector<float> term_raw((size_t) k), term_c((size_t) k), term_id((size_t) k);
    for (int64_t i = 0; i < k; ++i) {
        term_raw[(size_t) i] = w_in[(size_t) i] * expert_value[(size_t) ids_in[(size_t) i]];
    }
    for (int64_t i = 0; i < k; ++i) {
        term_c[(size_t) i]  = term_raw[(size_t) perm[(size_t) i]];
    }
    // identity mode: perm must be 0..k-1, so the terms are the input terms in input order.
    std::vector<int32_t> perm_id((size_t) k);
    cgc_canon_build_perm(ids_in.data(), perm_id.data(), 1, k, 2);
    for (int64_t i = 0; i < k; ++i) {
        term_id[(size_t) i] = term_raw[(size_t) perm_id[(size_t) i]];
    }

    o.sum_raw      = sum_chain(term_raw.data(), k);
    o.sum_canon    = sum_chain(term_c.data(),   k);
    o.sum_identity = sum_chain(term_id.data(),  k);
    return o;
}

int main() {
    std::mt19937 rng(20260917); // fixed seed: this is a gate, not a fuzz run
    std::uniform_int_distribution<int32_t> id_dist(0, 255);
    std::uniform_real_distribution<float>   w_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float>   v_dist(-2.0f, 2.0f);

    const int64_t k       = 8;
    const int     n_trial = 2000;

    int raw_differs = 0, identity_differs = 0, canon_differs = 0;
    int per_token_checked = 0;

    // --- assertion 1..4 over many trials: perm is a bijection and produces ascending ids ---------
    bool bijection = true, ascending = true, stable_ties = true, identity_is_identity = true;
    for (int t = 0; t < n_trial; ++t) {
        std::vector<int32_t> ids((size_t) k);
        for (int64_t i = 0; i < k; ++i) {
            ids[(size_t) i] = id_dist(rng);
        }
        // force some ties so the tie-break rule is actually exercised
        if (t % 3 == 0) {
            ids[3] = ids[1];
        }

        std::vector<int32_t> perm((size_t) k);
        cgc_canon_build_perm(ids.data(), perm.data(), 1, k, 1);

        std::vector<int32_t> seen((size_t) k, 0);
        for (int64_t i = 0; i < k; ++i) {
            if (perm[(size_t) i] < 0 || perm[(size_t) i] >= k) {
                bijection = false;
            } else if (seen[(size_t) perm[(size_t) i]]++) {
                bijection = false;
            }
        }
        for (int64_t i = 1; i < k; ++i) {
            if (ids[(size_t) perm[(size_t) i - 1]] > ids[(size_t) perm[(size_t) i]]) {
                ascending = false;
            }
        }
        // stable ties: equal ids must appear in their original relative order
        for (int64_t i = 1; i < k; ++i) {
            const int32_t a = ids[(size_t) perm[(size_t) i - 1]];
            const int32_t b = ids[(size_t) perm[(size_t) i]];
            if (a == b && perm[(size_t) i - 1] > perm[(size_t) i]) {
                stable_ties = false;
            }
        }
        std::vector<int32_t> perm_id((size_t) k);
        cgc_canon_build_perm(ids.data(), perm_id.data(), 1, k, 2);
        for (int64_t i = 0; i < k; ++i) {
            if (perm_id[(size_t) i] != i) {
                identity_is_identity = false;
            }
        }
        per_token_checked++;
    }

    check(bijection,          "perm is a bijection of positions (no id lost, none duplicated)");
    check(ascending,          "canonical ids are non-decreasing by expert id");
    check(stable_ties,        "equal expert ids keep their original relative order (order is unique)");
    check(identity_is_identity, "mode 2 writes the identity, so the control cannot silently reorder");
    std::printf("      (%d single-token permutations checked)\n", per_token_checked);

    // --- test A: two different input orders, same 8 pairs --------------------------------------
    for (int t = 0; t < n_trial; ++t) {
        // build k distinct expert ids so the two orders are the same SET in different POSITIONS
        std::vector<int32_t> ids((size_t) k);
        for (int64_t i = 0; i < k; ++i) {
            bool dup = true;
            while (dup) {
                dup = false;
                ids[(size_t) i] = id_dist(rng);
                for (int64_t j = 0; j < i; ++j) {
                    if (ids[(size_t) j] == ids[(size_t) i]) {
                        dup = true;
                    }
                }
            }
        }
        std::vector<float> w((size_t) k);
        for (int64_t i = 0; i < k; ++i) {
            w[(size_t) i] = w_dist(rng);
        }
        std::vector<float> expert_value(256);
        for (auto & v : expert_value) {
            v = v_dist(rng);
        }

        // order B: a rotation of the same (id, weight) pairs -- the pairs stay together, only their
        // positions change, which is exactly what a different pool layout does to the ids array.
        const int64_t rot = 1 + (t % (k - 1));
        std::vector<int32_t> ids_b((size_t) k);
        std::vector<float>   w_b((size_t) k);
        for (int64_t i = 0; i < k; ++i) {
            // pair p = (ids[i], w[i]); map i -> i+rot so pair p lands at a different position
            const int64_t src = (i + k - rot) % k;
            ids_b[(size_t) i] = ids[(size_t) src];
            w_b[(size_t) i]   = w[(size_t) src];
        }

        const trial_out a = run_trial(ids,   w,   expert_value);
        const trial_out b = run_trial(ids_b, w_b, expert_value);

        if (a.sum_raw != b.sum_raw) {
            raw_differs++;
        }
        if (a.sum_identity != b.sum_identity) {
            identity_differs++;
        }
        if (a.sum_canon != b.sum_canon) {
            canon_differs++;
        }
        if (a.canon_ids != b.canon_ids) {
            std::printf("FAIL  canonical id sequence differs between two input orders (trial %d)\n", t);
            g_fail++;
            break;
        }
    }

    check(raw_differs > 0,
          "CONTROL: without canonicalisation the chain sum really does depend on the input order");
    check(identity_differs > 0,
          "CONTROL: the identity mode keeps that dependence, so mode 2 is not accidentally canonical");
    check(canon_differs == 0,
          "TEST A: with canonical order the chain sum is bit-identical across input orders");

    // --- test A, multi-token: the shape the engine actually builds ---------------------------------
    //
    // The leaf is ONE tensor of k*n_tokens ints and the hook loops tokens, so a per-token bug (a perm
    // computed from the wrong row, or a perm applied across the whole block) is invisible to the
    // single-token case above -- and T >= 2 is precisely the shape whose routing this project already
    // got wrong once (the linear top-k read). Every token gets its own permutation, and the two input
    // orders are solved per token.
    {
        bool mt_ids_ok = true, mt_bij_ok = true;
        int  mt_raw_differs = 0, mt_canon_differs = 0, mt_tokens = 0;
        for (int t = 0; t < 400; ++t) {
            const int64_t T = 2 + (t % 7); // 2..8, the pool-path range (cgc_pool_max_tokens default 8)
            std::vector<int32_t> ids((size_t) (T * k)), ids_b((size_t) (T * k));
            std::vector<float>   w((size_t) (T * k)),   w_b((size_t) (T * k));
            std::vector<float>   expert_value(256);
            for (auto & v : expert_value) {
                v = v_dist(rng);
            }
            for (int64_t tt = 0; tt < T; ++tt) {
                for (int64_t i = 0; i < k; ++i) {
                    bool dup = true;
                    int32_t e = 0;
                    while (dup) {
                        dup = false;
                        e = id_dist(rng);
                        for (int64_t j = 0; j < i; ++j) {
                            if (ids[(size_t) (tt * k + j)] == e) {
                                dup = true;
                            }
                        }
                    }
                    ids[(size_t) (tt * k + i)] = e;
                    w[(size_t) (tt * k + i)]   = w_dist(rng);
                }
                // each token gets a DIFFERENT rotation, so a perm that ignored the token index
                // would be caught here rather than passing by luck
                const int64_t rot = 1 + ((t + tt) % (k - 1));
                for (int64_t i = 0; i < k; ++i) {
                    const int64_t src = (i + k - rot) % k;
                    ids_b[(size_t) (tt * k + i)] = ids[(size_t) (tt * k + src)];
                    w_b[(size_t) (tt * k + i)]   = w[(size_t) (tt * k + src)];
                }
            }

            std::vector<int32_t> perm((size_t) (T * k)), perm_b((size_t) (T * k));
            std::vector<int32_t> cids((size_t) (T * k)), cids_b((size_t) (T * k));
            cgc_canon_build_perm(ids.data(),   perm.data(),   T, k, 1);
            cgc_canon_build_perm(ids_b.data(), perm_b.data(), T, k, 1);
            cgc_canon_apply(ids.data(),   perm.data(),   cids.data(),   T * k);
            cgc_canon_apply(ids_b.data(), perm_b.data(), cids_b.data(), T * k);

            if (cids != cids_b) {
                mt_ids_ok = false;
                static int dbg = 0;
                if (dbg++ < 2) {
                    for (int64_t tt = 0; tt < T; ++tt) {
                        std::printf("      DEBUG t=%d tt=%lld rot=%lld inA=", t, (long long) tt,
                                    (long long) (1 + ((t + tt) % (k - 1))));
                        for (int64_t i = 0; i < k; ++i) { std::printf(" %d", ids[(size_t) (tt * k + i)]); }
                        std::printf(" | cA=");
                        for (int64_t i = 0; i < k; ++i) { std::printf(" %d", cids[(size_t) (tt * k + i)]); }
                        std::printf(" | cB=");
                        for (int64_t i = 0; i < k; ++i) { std::printf(" %d", cids_b[(size_t) (tt * k + i)]); }
                        std::printf("\n");
                    }
                }
            }
            for (int64_t tt = 0; tt < T; ++tt) {
                std::vector<int32_t> seen((size_t) k, 0);
                for (int64_t i = 0; i < k; ++i) {
                    const int32_t p = perm[(size_t) (tt * k + i)];
                    // ABSOLUTE range, and confined to THIS token's row. A row-local perm (0..k-1)
                    // fails the first half; a perm that leaked into a neighbouring row fails the
                    // second. Both are silent in production: every value stays a legal id, and
                    // GET_ROWS over the flattened block would just read another token's experts.
                    if (p < tt * k || p >= (tt + 1) * k) {
                        mt_bij_ok = false;
                    } else if (seen[(size_t) (p - tt * k)]++) {
                        mt_bij_ok = false;
                    }
                }
            }

            // per token: chain sum of the terms, with and without canonicalisation
            for (int64_t tt = 0; tt < T; ++tt) {
                std::vector<float> ta((size_t) k), tb((size_t) k), ca((size_t) k), cb((size_t) k);
                for (int64_t i = 0; i < k; ++i) {
                    ta[(size_t) i] = w[(size_t) (tt * k + i)]   * expert_value[(size_t) ids[(size_t) (tt * k + i)]];
                    tb[(size_t) i] = w_b[(size_t) (tt * k + i)] * expert_value[(size_t) ids_b[(size_t) (tt * k + i)]];
                }
                for (int64_t i = 0; i < k; ++i) {
                    // perm is ABSOLUTE (see build_perm); ta/tb are this row's k terms, so map back.
                    // Indexing ta with the absolute value is the same bug one level up from the one
                    // the id-block assertion catches, and it is why the two spaces are named here.
                    const int64_t la = perm[(size_t) (tt * k + i)] - tt * k;
                    const int64_t lb = perm_b[(size_t) (tt * k + i)] - tt * k;
                    ca[(size_t) i] = ta[(size_t) la];
                    cb[(size_t) i] = tb[(size_t) lb];
                }
                if (sum_chain(ta.data(), k) != sum_chain(tb.data(), k)) {
                    mt_raw_differs++;
                }
                if (sum_chain(ca.data(), k) != sum_chain(cb.data(), k)) {
                    mt_canon_differs++;
                }
                mt_tokens++;
            }
        }
        check(mt_ids_ok,   "multi-token: canonical id block is identical across two input orders");
        check(mt_bij_ok,   "multi-token: each token's perm is its own bijection (no leak across rows)");
        check(mt_raw_differs > 0, "CONTROL: multi-token raw chain sums do depend on the input order");
        check(mt_canon_differs == 0, "TEST A: multi-token canonical chain sums are bit-identical");
        std::printf("      (%d token rows checked; order-sensitive rows raw=%d, canonical=%d)\n",
                    mt_tokens, mt_raw_differs, mt_canon_differs);
    }

    std::printf("\n  order-sensitive trials: raw=%d/%d  identity=%d/%d  canonical=%d/%d\n",
                raw_differs, n_trial, identity_differs, n_trial, canon_differs, n_trial);

    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
