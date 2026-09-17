#pragma once

// CGC M1 work item 2: phase split, decided from the REQUEST's token count.
//
// Roadmap §M1 work item 2 (docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md):
//
//     graph 按 phase 分流，在建圖前決定（依 request 的 token 數，不看常駐）：
//       T >= T_prefill（先用 512 當起點）-> PREFILL GRAPH：整層 slab、256 experts
//       T <= T_decode（16 = floor(slots/top_k) 的上界）-> DECODE GRAPH：compacted gather
//
// and work item 3: "cap 不再是從 slots 推導的常數：它退化成 decode graph 的寬度上限".
//
// WHY THIS FILE EXISTS AT ALL
//
// Before it, the phase was an emergent property of one expression -- `n_tokens <=
// cgc_pool_max_tokens()` -- evaluated independently at six sites:
//
//   llama-graph.cpp    :2017  the CGC_RN_ROUTING mask leaf        (pool-path feature)
//   llama-graph.cpp    :2182  the remap leaf / S1 slot-table block (the decode graph itself)
//   llama-context.cpp  :1862  hot prewarm                          (pool-path feature)
//   llama-context.cpp  :1963  prefetch / SPAC refresh              (pool-path feature)
//   llama-context.cpp  :5468  the hook's large-batch (prefill) branch
//   llama-context.cpp  :288   the n_batch clamp in the constructor
//
// Six copies of one predicate is six chances to disagree, and the failure mode is not a
// crash: the graph builds one phase while the hook serves the other, which means either
// `tensor buffer is nil` (raw ids against shrunk tensors) or a silently un-filled remap leaf
// (pool regions read with the previous step's ids). Both were measured during M1 work item 1.
// So the predicate lives here, both sides call the same function, and the raw env var is read
// in exactly one place.
//
// The two phases, in terms of what already exists in this engine:
//
//   DECODE  : the pool path. The remap leaf (or the S1 slot table) is built, the eval hook maps
//             expert ids to pool slot indices for the whole step, and the FFN reads the pool
//             region by slot. Requires the step's expert UNION to be routable, i.e.
//             n_tokens * top_k <= usable_slots -- that is where the decode bound comes from.
//   PREFILL : the whole-layer slab path (M2, 2026-09-14): the layer's experts are streamed into
//             a transient slab with ne[2] = n_expert and mul_mat_id consumes RAW expert ids, so
//             the pool is not involved and the width is not bounded by it.
//
// The phase bound is therefore `min(cap, floor(usable_slots / top_k))`, NOT `cap`:
// `slots/top_k` is the widest step whose union can still be routed, so a step wider than that
// cannot legitimately take the decode path whatever cap says. This is the cap demotion of work
// item 3: cap stops being the phase switch and becomes only the decode graph's ceiling.

#include <cstdint>
#include <cstdlib>

// Routable upper bound of the decode graph, in tokens, for a pool that can route `slots` experts
// per layer with `top_k` experts per token. Soundness property (asserted by
// scripts/check/phase_split_selftest.cpp): for every n_tokens <= the returned bound,
// n_tokens * top_k <= slots, i.e. the worst-case union still fits.
//
// `cap` is the operator's ceiling (CGC_POOL_MAX_TOKENS, default 8, clamped to [2, 64] by
// cgc_pool_max_tokens()); it can only lower this bound, never raise it.
// `slots == 0` means "no pool" -> the caller is not in the pool regime at all and the bound is
// not meaningful; the function returns cap so that the value stays a pure ceiling.
static inline uint32_t cgc_decode_bound(uint64_t slots, uint32_t top_k, uint32_t cap) {
    if (top_k == 0) {
        return cap;
    }
    if (slots == 0) {
        return cap;
    }
    const uint64_t by_slots = slots / (uint64_t) top_k; // floor(...): the widest routable step
    if (by_slots == 0) {
        return 0; // even one token's top_k does not fit: no step may take the decode path
    }
    return (uint32_t) (by_slots < (uint64_t) cap ? by_slots : (uint64_t) cap);
}

// Minimum width that prefers the PREFILL graph even when the decode graph could route the step.
// Default 512 (the roadmap's starting point). The intent is a performance preference: above it the
// whole-layer slab is a sequential read against a fresh expert set rather than a pool-shaped
// shuffling of the same bytes.
//
// MEASURED STATUS (scripts/check/phase_split_selftest.cpp asserts it): with the shipped ceiling
// (cgc_pool_max_tokens() clamps cap to [2, 64]) the decode width can never reach 512, so this
// threshold is currently NON-BINDING -- the phase boundary is fully determined by the pool's
// routable geometry + cap. It is kept, and asserted to be non-binding, so that raising the cap
// ceiling above it makes the gate speak up instead of silently changing which graph serves which
// step. A knob that cannot be reached is exactly the trap this project keeps paying for; the
// difference here is that the gate says so out loud.
static inline uint32_t cgc_prefill_threshold() {
    static const uint32_t v = [] {
        const char * e = getenv("CGC_PREFILL_THRESHOLD");
        long x = 512;
        if (e != nullptr && *e != '\0') {
            x = strtol(e, nullptr, 10);
        }
        if (x < 2) {
            x = 2;
        }
        if (x > (1L << 20)) {
            x = 1L << 20;
        }
        return (uint32_t) x;
    }();
    return v;
}

// NAMING, because this tree already has a `cgc_phase_t` and it means something else: it is the
// STEP KIND used by the MTP/API layer (CGC_PHASE_PREFILL / VERIFY / DRAFT / CATCHUP /
// NORMAL_DECODE, llama-ext.h) and it is set per request by the caller. This one is the GRAPH SHAPE
// derived from the step's width. They are correlated but not the same question: a DEFAULT-context
// step of 4 tokens is CGC_PHASE_VERIFY-shaped to the API layer while its graph is whichever of the
// two below its width selects. Keeping the names apart is deliberate -- the first collision attempt
// (CGC_PHASE_PREFILL for both) did not compile, which is the good outcome; two vocabularies that
// silently agreed would have been the bad one.
enum cgc_graph_phase_t {
    CGC_GRAPH_DECODE  = 0, // pool path: remap leaf / S1 table, FFN reads pool regions by slot
    CGC_GRAPH_PREFILL = 1, // whole-layer slab, 256 experts, RAW ids, pool not involved
};

// The decode graph's actual width: the routable bound, then the T_prefill preference, folded in
// HERE and only here. Folding it in once is what keeps the clamp and the predicate consistent: the
// n_batch clamp uses this value, and the predicate is then a plain `n_tokens <= width`. An earlier
// revision instead had the predicate re-check the threshold, and the two disagreed for
// CGC_PREFILL_THRESHOLD <= cap -- the clamp allowed a step that the predicate then sent to the
// prefill graph, i.e. a step with no slab armed and raw ids against shrunk tensors. Measured by
// scripts/check/phase_split_selftest.cpp (`CGC_PREFILL_THRESHOLD=4`).
static inline uint32_t cgc_decode_width(uint64_t slots, uint32_t top_k, uint32_t cap) {
    const uint32_t bound = cgc_decode_bound(slots, top_k, cap);
    const uint32_t thr   = cgc_prefill_threshold();
    const uint32_t pref  = thr > 1 ? thr - 1 : 1; // the widest step the threshold still calls DECODE
    return bound < pref ? bound : pref;
}

// THE phase decision. `decode_width` must be the value produced by cgc_decode_width() from the SAME
// pool geometry on both sides (the graph gets it through llm_graph_params, the hook and the clamp
// from the context member) -- that is the whole contract.
static inline enum cgc_graph_phase_t cgc_select_graph_phase(int64_t n_tokens, uint32_t decode_width) {
    if (n_tokens <= 0) {
        return CGC_GRAPH_PREFILL; // no tokens: nothing to route; do not claim the decode path
    }
    return (uint64_t) n_tokens <= (uint64_t) decode_width ? CGC_GRAPH_DECODE : CGC_GRAPH_PREFILL;
}

// Convenience for the many call sites that only ask "does this step take the pool path?".
static inline bool cgc_is_decode_graph(int64_t n_tokens, uint32_t decode_width) {
    return cgc_select_graph_phase(n_tokens, decode_width) == CGC_GRAPH_DECODE;
}
