// M1 work item 2 (phase split): offline gate for the phase rule.
//
// Spec: docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md §M1 work item 2/3 --
//   "graph 按 phase 分流，在建圖前決定（依 request 的 token 數）"; "T <= T_decode（16 =
//    floor(slots/top_k) 的上界）-> DECODE GRAPH"; "cap ... 退化成 decode graph 的寬度上限".
//
// The rule itself lives in src/llama.cpp/src/llama-cgc-phase.h (header-only on purpose), so this
// gate needs no GPU, no server and no project build.
//
// Build & run (seconds; safe while another session owns the GPU):
//     clang++ -std=c++17 -O2 -o /tmp/phase_split_selftest scripts/check/phase_split_selftest.cpp
//     /tmp/phase_split_selftest                      # default threshold (512)
//     CGC_PREFILL_THRESHOLD=4 /tmp/phase_split_selftest   # threshold honoured, phase sets move
//
// The property that matters is not "the phases are disjoint". It is that a step the rule puts in
// the DECODE graph can actually be ROUTED by the pool: n_tokens * top_k <= usable_slots, because
// the step's expert union is at most one expert per (token, top-k) slot. If that fails the pool
// cannot hold the union, and the failure mode is not an abort -- the step silently falls to the
// gathering path, which is the shape that produced `tensor buffer is nil` during work item 1.

#include "../../src/llama.cpp/src/llama-cgc-phase.h"

#include <cstdio>
#include <cstdlib>
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

int main() {
    const uint32_t thr = cgc_prefill_threshold();
    const char * thr_env = getenv("CGC_PREFILL_THRESHOLD");

    std::printf("  threshold = %u tokens (env%s)\n", thr,
                thr_env != nullptr ? " set" : " unset -> default 512");
    check(thr >= 2, "threshold is at least 2 (a width of 1 is always the decode graph)");
    if (thr_env != nullptr) {
        check(thr == (uint32_t) strtoul(thr_env, nullptr, 10) || strtoul(thr_env, nullptr, 10) < 2,
              "an explicit CGC_PREFILL_THRESHOLD is honoured (clamped at the low end)");
    } else {
        check(thr == 512, "the default threshold is the roadmap's starting point, 512");
    }

    // --- 1. soundness of the decode bound, over a wide sweep ------------------------------------
    {
        uint64_t checked_cases = 0;
        bool     sound = true, cap_only_lowers = true, prefix = true;
        for (uint64_t slots = 0; slots <= 8192; slots += 1) {
            for (uint32_t top_k = 1; top_k <= 16; ++top_k) {
                for (uint32_t cap = 2; cap <= 64; cap *= 2) {
                    const uint32_t bound = cgc_decode_bound(slots, top_k, cap);
                    const uint32_t width = cgc_decode_width(slots, top_k, cap);
                    if (bound > cap) {
                        cap_only_lowers = false;
                    }
                    for (uint32_t n = 1; n <= cap; ++n) {
                        const bool decode = cgc_is_decode_graph((int64_t) n, width);
                        checked_cases++;
                        // soundness: everything the rule calls DECODE is routable
                        if (decode && slots > 0 && (uint64_t) n * top_k > slots) {
                            sound = false;
                        }
                        // the DECODE set must be a prefix of 1..cap (no holes), or a caller that
                        // clamps n_batch to the width would leave a step with no valid phase
                        if (decode && n > width) {
                            prefix = false;
                        }
                        // and the width must never exceed the routable bound
                        if (width > bound) {
                            prefix = false;
                        }
                    }
                }
            }
        }
        std::printf("      (%llu (slots, top_k, cap, n) cases swept)\n", (unsigned long long) checked_cases);
        check(sound, "every step the rule calls DECODE has n_tokens*top_k <= usable_slots (union routable)");
        check(cap_only_lowers, "cap never raises the bound -- it is a ceiling, not the switch (work item 3)");
        check(prefix, "the DECODE set is a prefix [1..width] and width <= bound: clamping n_batch leaves no phase gap");
    }

    // --- 2. the bound, on this engine's real geometries -----------------------------------------
    {
        struct c { uint64_t slots; uint32_t top_k; uint32_t cap; uint32_t want; const char * why; };
        const c cases[] = {
            { 143,  8,  8,  8, "8 GiB pool (143 slots): floor(143/8)=17 > cap -> cap wins" },
            { 143,  8, 16, 16, "cap raised to 16: the router's width is now the binding constraint" },
            {  70,  8,  8,  8, "4 GiB pool (70 slots): floor(70/8)=8 -> exactly cap" },
            {  36,  8,  8,  4, "2 GiB pool (36 slots): floor(36/8)=4 -> pool binds, NOT cap" },
            {  36,  8, 64,  4, "even with cap 64 a 36-slot pool may not take an 8-token step" },
            { 256,  8,  8,  8, "one full layer resident: cap still wins" },
            { 2560, 8,  8,  8, "whole model resident: width is chosen by T_prefill, not by cap" },
            { 256,  1, 64, 64, "top_k=1: 256 routable slots allow 64 tokens" },
            {   4,  8,  8,  0, "4 slots with top_k=8: nothing is routable; caller clamps up to 1" },
            {   0,  8,  8,  8, "no pool: the bound is not meaningful, cap is returned unchanged" },
        };
        for (const c & t : cases) {
            const uint32_t got = cgc_decode_bound(t.slots, t.top_k, t.cap);
            if (got != t.want) {
                std::printf("FAIL  bound(slots=%llu, top_k=%u, cap=%u) = %u, want %u  [%s]\n",
                            (unsigned long long) t.slots, t.top_k, t.cap, got, t.want, t.why);
                g_fail++;
            } else {
                std::printf("ok    bound(slots=%llu, top_k=%u, cap=%u) = %u  [%s]\n",
                            (unsigned long long) t.slots, t.top_k, t.cap, got, t.why);
            }
        }
    }

    // --- 3. the phase split itself ---------------------------------------------------------------
    {
        const uint32_t width = cgc_decode_width(143, 8, 8); // 8 GiB pool, the shipped config
        check(cgc_select_graph_phase(1, width) == CGC_GRAPH_DECODE, "T=1 (decode step) -> DECODE graph");
        check(cgc_select_graph_phase((int64_t) width, width) == CGC_GRAPH_DECODE,
              "T=width (MTP verify batch at cap) -> DECODE graph");
        check(cgc_select_graph_phase((int64_t) width + 1, width) == CGC_GRAPH_PREFILL,
              "T=width+1 -> PREFILL graph (the decode graph may not route it)");
        check(cgc_select_graph_phase(0, width) == CGC_GRAPH_PREFILL,
              "T=0 -> not the decode graph (nothing to route; do not claim the pool path)");

        // T_prefill versus the shipped ceiling. cap is clamped to [2, 64] by cgc_pool_max_tokens(),
        // so the decode width can never reach 512 -- the threshold is non-binding today. Asserting
        // that is the point: if someone raises the cap ceiling above 512, this line turns RED and
        // says the phase behaviour just changed, instead of the knob staying invisible.
        const uint32_t widest = cgc_decode_width(1u << 20, 1, 64); // most permissive pool + cap
        if (thr > 64) {
            check(widest < thr, "with cap clamped to 64, T_prefill (512) is non-binding -- and asserted so");
            std::printf("      (widest reachable decode width = %u < T_prefill = %u)\n", widest, thr);
        } else {
            check(cgc_select_graph_phase((int64_t) thr, widest) == CGC_GRAPH_PREFILL,
                  "with T_prefill <= cap, the threshold does bind: T>=thr takes the prefill graph");
            check(cgc_select_graph_phase((int64_t) thr - 1, widest) == CGC_GRAPH_DECODE,
                  "and T=thr-1 stays on the decode graph (the clamp and the predicate agree)");
        }
    }

    // --- 4. the case that actually shipped: clamping n_batch to the bound ------------------------
    {
        // 2 GiB pool: cap says 8, the pool says 4. A clamp to cap would let an 8-token step be built
        // whose union (up to 64 experts) cannot fit 36 slots -- the pre-work-item-2 behaviour.
        const uint32_t width = cgc_decode_width(36, 8, 8);
        bool all_decode = true;
        for (uint32_t n = 1; n <= width; ++n) {
            if (!cgc_is_decode_graph((int64_t) n, width)) {
                all_decode = false;
            }
        }
        // threshold-aware on purpose: with a small T_prefill the width follows the threshold down,
        // so the expectation is min(routable, thr-1), not the routable bound alone
        const uint32_t want36 = (thr - 1 < 4u) ? thr - 1 : 4u;
        check(width == want36, "36-slot pool: the clamp lands on the routable width, not on cap's 8");
        check(all_decode, "every width the clamp still allows is served by the decode graph");
        check(!cgc_is_decode_graph(8, width),
              "and the width cap alone would have allowed (8) is refused -- the union would not fit");

        // And the same with a small threshold: the clamp must follow the threshold down, otherwise it
        // would allow a step the predicate sends to a prefill graph that is not armed.
        const uint32_t w4 = cgc_decode_width(143, 8, 8);
        if (thr <= 8) {
            check(w4 == thr - 1, "a small T_prefill pulls the clamp down with it (no phase gap)");
        } else {
            check(w4 == 8, "with T_prefill above cap the clamp is the cap, as before");
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
