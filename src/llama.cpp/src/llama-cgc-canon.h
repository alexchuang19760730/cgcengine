#pragma once

// CGC M1 work item 4: canonical gather order.
//
// Why this exists (roadmap §M1, work item 4; design doc
// docs/M1_WORKITEM4_CANONICAL_GATHER_ORDER_2026-09-17.md):
//
// The MoE FFN aggregation is a left-to-right fp32 chain over the k expert contributions:
//
//     moe_out = (((c0 + c1) + c2) + ... )
//
// and position i holds the expert the *pool* happened to assign to that position (the ids
// operand is a slot index on the pool path). fp32 addition is not associative, so the rounding
// of that sum is a function of the pool layout -- which is exactly what M1 (bit-identical across
// pool sizes) must not be. Measured necessity: CGC_ADD_ORDER=rev moved the anchor arm's md5 from
// 2d4e5099 to f6718d44, i.e. this aggregation is order-sensitive enough to change generated text.
//
// The fix is to fix the ORDER, not to pick a different order: sort the k positions by expert id
// (ascending), ties broken by original position, and then apply that ONE permutation to both the
// ids and the routing weights. Permuting the ids alone is a silent wrong answer -- the weights are
// gathered positionally (`weights = get_rows(probs, sel)`) and consumed positionally by `ggml_mul`,
// so expert A's output would be paired with expert B's weight. See cgc_canon_apply()'s callers.
//
// Modes (parsed from CGC_CANON_ORDER, an env knob so the default path is untouched):
//
//   0 (default)  no permutation nodes at all -- the graph is the pre-work-item-4 graph
//   1            canonical: ascending expert id, stable ties by original position
//   2            IDENTITY control: the same nodes are built (same buffer layout, same
//                scheduler pressure) but the permutation is the identity, so mode 2 vs mode 1
//                isolates "the reordering changed the numbers" from "the extra nodes changed the
//                numbers". Mode 2 must reproduce the baseline's text; if it does not, the extra
//                nodes -- not the ordering -- are the carrier.
//
// Mode 1 and mode 2 change model output by construction, so no run under either may be quoted as
// quality or D5-gate evidence for a reference produced in a different mode.

#include <cstdint>
#include <cstdlib>

// Parsed once per process: every reader must agree, or the graph would permute the device ids
// while the hook left the host-written remap leaf in the old order (a silent mis-pairing).
static inline int cgc_canon_order_mode() {
    static const int mode = [] {
        const char * e = getenv("CGC_CANON_ORDER");
        return e != nullptr ? atoi(e) : 0;
    }();
    return mode;
}

// Build the canonical permutation of a row-major [n_tokens, k] id block.
//
//   ids  : n_tokens * k expert ids, token t's row at ids[t*k .. t*k+k-1]
//   perm : n_tokens * k out; perm[p] = the ORIGINAL ABSOLUTE POSITION whose id belongs at
//          canonical position p. So the canonical id vector is ids[perm[p]], and the canonical
//          weight vector for the same canonical position p is weights[perm[p]].
//
// ABSOLUTE, not row-local, and that distinction is not cosmetic: the consumer of this array in the
// graph is a GET_ROWS whose src0 is the FLATTENED [k*n_tokens] id vector, so a row-local index
// would look up another token's row instead of this token's k-th position. That is the same shape
// of defect this project already paid for once -- the pre-fix top-k snapshot read `t->data` linearly
// on a strided view, so every token >= 1 of every T >= 2 step was routed by another token's ids,
// and it stayed invisible because every arm made the same mistake (see the r32 note in
// expert_cache_on_topk). A permutation is exactly the kind of index array where that failure is
// silent: every value stays in range, so nothing asserts. scripts/check/canon_order_selftest.cpp
// asserts the absolute-range property per token for this reason.
//
// Insertion sort, not std::sort: k <= 8 here, and the stable-tie property (equal ids keep their
// original relative order) must be part of the definition rather than an emergent property of the
// implementation -- two positions with the same expert id are otherwise not ordered, and a
// canonical order that is not unique is not a canonical order. Ties break on the ORIGINAL POSITION
// (the index), which is what makes equal ids deterministic as well; for equal ids the pairing is
// indistinguishable anyway, so uniqueness is what matters, not which order is chosen.
//
// Mode 2 writes the identity, which is what makes the control meaningful: same permutation size,
// same node structure, no reordering.
static inline void cgc_canon_build_perm(const int32_t * ids, int32_t * perm,
                                        int64_t n_tokens, int64_t k, int mode) {
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int32_t * row  = ids  + t * k;
        int32_t       * prow = perm + t * k;
        // local scratch: positions 0..k-1 of THIS token, sorted stably by its id
        int32_t loc[64];
        for (int64_t j = 0; j < k; ++j) {
            loc[j] = (int32_t) j;
        }
        if (mode == 1) {
            // stable insertion sort of the positions, keyed by (row[pos], pos) ascending
            for (int64_t j = 1; j < k; ++j) {
                const int32_t p   = loc[j];
                const int32_t key = row[p];
                int64_t       i   = j - 1;
                while (i >= 0 && row[loc[i]] > key) {
                    loc[i + 1] = loc[i];
                    --i;
                }
                loc[i + 1] = p;
            }
        }
        // publish ABSOLUTE positions (mode 1 and mode 2 alike, so the two differ only in the order)
        for (int64_t j = 0; j < k; ++j) {
            prow[j] = (int32_t) (t * k) + loc[j];
        }
    }
}

// Apply a permutation: out[i] = ids[perm[i]], perm holding ABSOLUTE positions (see build_perm).
// Kept separate from build_perm because the id and the weight are permuted by the SAME perm, and
// callers must be able to see that they use one perm and not two independently computed ones.
static inline void cgc_canon_apply(const int32_t * ids, const int32_t * perm, int32_t * out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = ids[perm[i]];
    }
}
