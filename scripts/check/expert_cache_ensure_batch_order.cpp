// [CGC 2026-09-16 Blocker B] Order-dependence test for llama_expert_cache_ensure_batch.
//
// Why this exists
// ---------------
// docs/M1_POOL_SPLIT_COST_2026-09-14.md §4 reported Blocker B as "expert 0 and expert 1 share
// slot 1", quoting the 09-14 capture as
//
//     CGC-PRE:  il=1 st[0]=-1 st[1]=1 st[2]=2 ... st[7]=7
//     CGC-POST: il=1 st[0]=1  st[1]=1 ...            <- two experts own slot 1
//     CGC-SLOT: il=1 remap=[1 2 3 4 5 6 7 8]
//
// The raw capture says st[1]=2, not st[1]=1 (Backup/cgc_logs/llama_server_20260914_094821.log
// line 324), which makes the three lines mutually consistent (remap[i] == st[ids[i]]) and the
// quoted contradiction impossible. The actual anomaly is a UNIFORM +1 shift of the whole layer's
// expert->slot map, and this test reproduces it exactly, on the real function:
//
//   setup = the state the identity prepopulate leaves behind (slot_owner[s] = s, slot_table[e] = e,
//   slot_last_use = 0 for every slot: prepopulate's own stamp, "oldest: first eviction candidate
//   when full"), plus ONE cold member whose slot is not free (slot 0 held by an in-flight fill:
//   owner set, slot_queued set, table entry not yet published). The pool is therefore exhausted,
//   which is the one precondition the cascade needs.
//
//   expected (fixed)   : experts 1..15 keep their own slots; the single cold expert takes a slot
//                        no member is reading.
//   observed (broken)  : expert e lands in slot e+1 for EVERY member -- one cold expert turns a
//                        16-member hits-only batch into 16 misses, because batch_owned was filled
//                        incrementally and each miss evicted the NEXT member's slot.
//
// Runs with no model, no server, no I/O: every fill is a no-op because each key_segs entry has an
// empty position list (fill_pool_direct returns early on segs.empty()).
//
// Build (from src/llama.cpp; the dylib must be built from the SAME header — see below):
//
//   clang++ -std=c++17 -O1 -I include -I ggml/include -I src \
//     ../../scripts/check/expert_cache_ensure_batch_order.cpp \
//     -L build/bin -lllama -Wl,-rpath,$PWD/build/bin \
//     -o ../../scripts/check/expert_cache_ensure_batch_order
//   DYLD_LIBRARY_PATH=build/bin ../../scripts/check/expert_cache_ensure_batch_order
//
// ABI WARNING, and it cost one debugging round: llama_expert_cache is passed BY POINTER across the
// dylib boundary and both sides must agree on every member offset. Adding n_hit_adopted_queued to
// the struct shifted ever_loaded from 0x608 to 0x610; the stale dylib in build/bin (left behind by
// the A/B stash) then read n_slot_table_unchanged — zero — as ever_loaded.data() and the test died
// with SIGSEGV at 0x0 inside ensure_batch. A stale-header/stale-dylib mismatch presents as a crash
// in the LIBRARY, not as a link error, so rebuild the library whenever this header changes and
// rebuild this test from the same tree state it was linked against.
#include "llama-expert-cache.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int32_t slot_of(const llama_expert_cache & c, uint32_t layer, uint32_t expert) {
    return c.slot_table[(size_t) layer * c.n_expert + expert];
}

int main() {
    // zero_slot_enabled() is env-gated (CGC_VERIFY_DECODE / CGC_DRAFT_DECODE); the 09-14 probe ran
    // without either, so usable_slots == n_slots and the identity prepopulate fills every slot.
    if (getenv("CGC_VERIFY_DECODE") != nullptr || getenv("CGC_DRAFT_DECODE") != nullptr) {
        fprintf(stderr, "this test reproduces the un-reserved-slot configuration; unset "
                        "CGC_VERIFY_DECODE / CGC_DRAFT_DECODE\n");
        return 2;
    }

    const uint32_t NL = 1, NE = 256, NS = 24, NMEM = 16; // ntok=2 x n_expert_used=8 (the 09-14 batch)

    llama_expert_cache c;
    c.n_expert = NE;
    c.n_slots  = NS;
    c.pool_active = false; // the adoption path needs a bg thread; not under test here
    c.slot_owner.assign(NL, std::vector<int32_t>(NS, -1));
    c.slot_last_use.assign(NL, std::vector<uint64_t>(NS, 0));
    c.slot_queued.assign(NL, std::vector<uint8_t>(NS, 0));
    c.slot_loading.assign(NL, std::vector<uint8_t>(NS, 0));
    c.slot_decode_reserved.assign(NL, std::vector<uint8_t>(NS, 0));
    c.slot_pinned.assign(NL, std::vector<uint8_t>(NS, 0));
    c.slot_pinned_static.assign(NL, std::vector<uint8_t>(NS, 0));
    c.slot_table.assign((size_t) NL * NE, -1);
    c.ever_loaded.assign(NL, std::vector<uint8_t>(NE, 0));
    c.n_distinct_demanded.assign(NL, 0);
    c.spac_util.assign(NL, {});

    // prepopulate's end state: slot s holds expert s, last_use 0 on every slot.
    for (uint32_t s = 0; s < NS; ++s) {
        c.slot_owner[0][s] = (int32_t) s;
        c.slot_table[s]     = (int32_t) s;
    }
    // the 09-14 state: expert 0 cold while its slot is NOT free (a queued fill holds it), which is
    // what makes pick_slot skip slot 0 in both the free scan and the eviction scan, so the first
    // miss evicts slot 1 instead -- the next member's slot.
    c.slot_table[0]     = -1;
    c.slot_queued[0][0] = 1;

    for (uint32_t e = 0; e < NMEM; ++e) {
        c.key_segs[((uint64_t) 0 << 32) | e] = {}; // empty -> fills are no-ops
    }
    std::vector<uint32_t> experts(NMEM);
    for (uint32_t i = 0; i < NMEM; ++i) {
        experts[i] = i;
    }

    llama_expert_cache_ensure_batch(&c, 0, experts.data(), experts.size(), /*defer_decode_protect=*/false);

    printf("post :");
    for (uint32_t e = 0; e < NMEM; ++e) {
        printf(" st[%u]=%d", e, slot_of(c, 0, e));
    }
    printf("\n");

    // expert 0 was cold by construction: it may take any slot. Every OTHER member was resident
    // before the call, so a correct ensure_batch must leave it where it was.
    int violations = 0;
    for (uint32_t e = 1; e < NMEM; ++e) {
        const int32_t s = slot_of(c, 0, e);
        if (s != (int32_t) e) {
            printf("VIOLATION: expert %u was resident at slot %u, now reads slot %d "
                   "(its slot was evicted by another member's miss)\n", e, e, s);
            if (++violations >= 4) {
                break;
            }
        }
    }
    // and the assignment must be self-consistent: distinct slots, owner agrees
    for (uint32_t e = 0; e < NMEM; ++e) {
        const int32_t s = slot_of(c, 0, e);
        if (s < 0 || s >= (int32_t) NS || c.slot_owner[0][s] != (int32_t) e) {
            printf("VIOLATION: expert %u -> slot %d, owner says %d\n", e, s,
                   (s >= 0 && s < (int32_t) NS) ? c.slot_owner[0][s] : -9);
            violations++;
        }
        for (uint32_t f = e + 1; f < NMEM; ++f) {
            if (slot_of(c, 0, f) == s) {
                printf("VIOLATION: experts %u and %u share slot %d\n", e, f, s);
                violations++;
            }
        }
    }

    printf("misses: %zu  hits: %zu\n", c.n_misses, c.n_hits);
    if (violations > 0) {
        printf("FAIL: %d violation(s) -- a member's slot was evicted by this batch's own misses\n",
               violations);
        return 1;
    }
    printf("PASS: every resident member kept its slot; the one cold expert took an unread slot\n");
    return 0;
}
