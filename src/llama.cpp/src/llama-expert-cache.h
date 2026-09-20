#pragma once

#include "llama.h"

#include <sys/uio.h> // struct iovec (pread_job merge-read form)

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Value semantics for CGC's boolean env knobs -- the ONE definition of "this flag is on".
//
// Why this exists (CONVENTIONS A9, and it cost six days): the readers of
// LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0 used to test `getenv(...) != nullptr`, under which `"0"` is a
// NON-NULL pointer, i.e. writing `0` ENABLED the flag. run_server.sh's profile wrote
// `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0` with a comment saying "layer 0 goes back into the pool",
// and the 2026-09-09 quality fix was therefore inert while every cap_*.json and llama-bench env
// block recorded the intent rather than the behaviour. The three readers live in three files, so
// the fix is not three copies of a comparison -- it is one predicate that they all call and
// cannot drift apart again.
//
// Off: unset, empty, or a leading '0'. Anything else is on (so "1", "true", "yes" all work).
inline bool cgc_env_on(const char * name) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return v[0] != '0';
}

// LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0: keep blk.0's expert FFN out of the L4 Metal pool.
//
// This is a *layout-matching* knob, not a quality knob, and the two historical verdicts on it are
// both right because they were taken against different bases:
//   - `-ngl 30` (run_n30cache.sh): the base keeps blk.0's FFN on the CPU, so pooling blk.0 moves
//     layer 0 to Metal -> float divergence vs base. skip0=1 is the bit-identical setting. (§8.36)
//   - full offload (ALLOW_NGL=1, all layers on Metal): the base has blk.0 on Metal, so skip0=1
//     turns layer 0 into a full-weight CPU tensor read by a Metal graph -> 4/10 on the 15+27
//     probe. skip0=0 is the correct setting. (2026-09-09)
// Set it to match the base layout of the configuration you are running, and expect the numerics
// to move when you flip it: it changes the pooled-layer count (39 vs 40) and layer 0's FFN
// consumer path. Any M1/M2/M3 reference is only comparable to the same value.
inline bool cgc_l4_skip_layer0_on() {
    return cgc_env_on("LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0");
}

// L2: bounded resident cache for MoE expert weights (expert streaming).
//
// Cache unit = (layer, expert). A slot holds a blob with all present kinds (gate/up/down, or the
// merged gate_up for per-layer layouts) concatenated in L1 index order; each segment is pread from
// the GGUF file at the absolute file offset recorded by the L1 index (segment-aware addressing,
// byte-verified against gguf-py on qwen36 IQ2 + gemma4 IQ3).
//
// Lifecycle / threading model (mirrors the Swift hot pool + C streamer design):
//   - slots are created either synchronously (llama_expert_cache_ensure miss -> this thread preads)
//     or by the background thread (llama_expert_cache_prefetch -> queued, filled off the critical
//     path at low priority).
//   - a slot is marked loading/queued while being filled; ensure() on a loading slot waits on its
//     condition variable. fill() refuses to copy from a loading slot.
//   - eviction is global LRU (last_use tick); only non-loading, non-queued slots are evictable.
//     total_bytes is kept <= budget when possible (a single blob larger than the budget is allowed).
//
// Not yet supported: split (multi-file) GGUFs (init fails and returns NULL — the loader still
// records file_idx in the index for future use).

struct llama_expert_cache {
    struct segment {
        int32_t  kind;        // 0=gate 1=up 2=down 3=gate_up
        uint32_t file_idx;    // source GGUF file index
        uint64_t file_offset; // absolute offset in that file
        uint32_t off;         // offset within the slot blob
        uint32_t bytes;
    };

    struct slot {
        uint64_t key; // (layer << 32) | expert
        std::vector<uint8_t> blob;
        std::vector<segment> segs;
        uint64_t last_use = 0;
        bool     loading = false;
        bool     queued  = false; // in the bg queue, not yet filled
        std::condition_variable cv;
    };

    const llama_expert_index_entry * index = nullptr;
    size_t index_size = 0;

    std::vector<FILE *> files; // per file_idx; all must be open (else cache disabled)
    // [CGC V1 verify 2026-09-05] files_path mirrors files[]. We need the path string (not just
    // the open FILE *) to re-open the GGUF through a FRESH fd for byte-identity verification
    // (so any stdio buffering on cache->files cannot mask an off-by-N read). Populated in
    // llama_expert_cache_init alongside the fopen() call. Empty string = split-model / unset.
    std::vector<std::string> files_path;
    std::unordered_map<uint64_t, std::vector<uint32_t>> key_segs; // key -> positions in index

    size_t budget = 0;
    size_t total_bytes = 0;
    uint64_t tick = 0;

    std::unordered_map<uint64_t, std::unique_ptr<slot>> map;

    // background pread thread
    std::thread bg;
    std::vector<uint64_t> bg_queue;
    mutable std::mutex m; // guards map / bg_queue / telemetry / total_bytes
    std::condition_variable bg_cv;
    bool bg_stop = false;

    // L3 Option A persistent pread worker pool (batch fill, LLAMA_EXPERT_CACHE_WORKERS=N,
    // default 8). One job = one segment pread; the batch submits ALL of a layer's misses at
    // once and blocks on pool_outstanding == 0, replacing the per-step spawn-per-expert +
    // join churn. Workers serve the decode critical path so they run at USER_INITIATED QoS
    // (like bg). pool_m must be released while pread runs (may block on IO).
    struct pread_job {
        FILE *    f;
        off_t     offset;
        size_t    bytes;
        uint8_t * dst;
        int *     ok;
        // [CGC 2026-08-29 merge-read] contiguous-file scattered-dst run, read with ONE preadv
        // (file side contiguous, memory side scattered — exactly preadv's iovec semantics).
        // Same experts, same file ranges, same dsts as the per-segment preads it replaces ->
        // pool contents bit-identical by construction; only the syscall count changes.
        // iovs/oks are heap arrays owned by the job (submitted by fill_segments_pool, freed by
        // the worker after the read). nullptr iovs = legacy single pread(dst, bytes).
        struct iovec * iovs = nullptr;
        int          ** oks = nullptr; // pointers to each run member's caller ok flag
        int            niov = 0;
    };
    std::vector<std::thread> workers;
    std::deque<pread_job>    jobs;
    std::mutex               pool_m;
    std::condition_variable  pool_cv;      // workers: new work / stop
    std::condition_variable  pool_done_cv; // submitter: outstanding == 0
    size_t                   pool_outstanding = 0;
    bool                     pool_stop = false;
    void pool_loop();

    // L3 Option A: static per-layer slot pool. Each layer has n_slots fixed slots; the FFN expert
    // tensors (src0 of mul_mat_id) point directly into this pool, and the eval hook remaps the
    // selected expert ids to their slot indices (a small int32 write, no per-step gather memcpy).
    // slot_table[layer * n_expert + expert] = slot index (-1 = not resident).
    uint32_t n_expert = 0;       // experts per layer (max expert id + 1)
    uint32_t n_slots  = 0;       // slots per layer
    bool pool_active = false;    // LLAMA_EXPERT_CACHE_POOL=1: pool allocated, Option A on
    std::vector<int32_t> slot_table;  // [n_layer * n_expert] ACTIVE: decode reads this
    // [CGC DBUF2 full double-buffer 2026-09-06] scratch slot table: bg fills write here when
    // CGC_DBUF2=1, while decode reads slot_table (active). At the per-step swap point, the two
    // vectors are swapped (O(1) internal-pointer swap) and the new scratch is reset to -1, so
    // fills never tear a buffer the decode is reading. CGC_DBUF2=0: scratch is unused, fills
    // write slot_table directly (legacy byte-identical behavior).
    std::vector<int32_t> slot_table_scratch;  // [n_layer * n_expert] SCRATCH: fills write this (DBUF2)
    std::vector<std::vector<std::vector<uint8_t>>> pool; // [layer][kind][slot*stride ..] (malloc path)
    // L4 zero-copy (-ngl>0 + ALLOW_NGL): pool regions adopted from the expert tensors' Metal
    // storage (non-owning). When pool_ext[layer][kind] != nullptr it takes precedence over pool.
    std::vector<std::vector<const uint8_t *>> pool_ext;       // [layer][kind] base
    std::vector<std::vector<size_t>>          pool_ext_stride; // [layer][kind] bytes per slot
    std::vector<std::vector<uint32_t>>        pool_ext_slots;  // [layer][kind] capacity
    // CGC M1 work item 1 (CGC_POOL_SPLIT): when the pool is its OWN allocation rather than the
    // expert tensor's storage, the graph must resolve the FFN weight through the pool's buffer (not
    // the tensor's) -- Metal looks a tensor up via ITS OWN buffer, so moving `data` alone leaves
    // nil. pool_ext_buf[layer][kind] is that buffer (non-owning); pool_owned_bufs owns the one
    // allocation per kind and is freed with the cache.
    std::vector<std::vector<ggml_backend_buffer_t>> pool_ext_buf; // [layer][kind] owner buffer
    std::vector<ggml_backend_buffer_t> pool_owned_bufs;
    // CGC M1 work item 1: the LOAD-TIME geometry of each expert tensor, captured ONCE while it is
    // still the loader's full-width tensor. With a split pool the graph has to alternate between two
    // geometries (pool slots vs the full-width weights), and deriving both sides from stored values
    // is what keeps it deterministic: the earlier attempt recorded the state it saw on each build,
    // and once a repoint had happened the record became the POOL state -- the next restore then
    // wrote a mixed tensor (pool data + full-width ne[2] + model buffer), which Metal reports as
    // `buffer is nil` (measured: exactly that operand triple, docs/M1_POOL_SPLIT_COST_2026-09-14.md).
    std::vector<std::vector<void *>>         pool_wide_data;  // [layer][kind] full-width data ptr
    std::vector<std::vector<ggml_backend_buffer_t>> pool_wide_buf; // [layer][kind] full-width buffer
    std::vector<std::vector<int64_t>>        pool_wide_ne2;   // [layer][kind] full-width ne[2]
    // prefill hot prewarm (LLAMA_EXPERT_CACHE_PREWARM_HOT=1): per-layer expert route frequency
    // accumulated during prefill; the first decode step fills the pool with the top-K hot set
    // (instead of the loader's experts-0..n prewarm, which ignores actual routing).
    std::vector<std::vector<uint64_t>> freq;      // [layer][expert] prefill route counts
    // [CGC M5 prerouter 2026-09-17] The experts predicted FOR this layer (so the scoring at the
    // layer's own hook can compare prediction against the actual selection). Only written when
    // CGC_PREROUTER=1; empty in every default run, so nothing downstream can observe it.
    std::vector<std::vector<uint32_t>> prerouter_pred;  // [layer] = predicted expert ids
    // [CGC mass-coverage measurement 2026-09-06] accumulate the actual softmax MASS carried by
    // every expert selection (not just counts) so a post-run report can answer "would the
    // resident top-K cover enough routing MASS?" — the metric that matters for renorm quality
    // (selection-count coverage overstates harm: tail experts carry little mass). Filled from
    // expert_cache_on_topk (env CGC_MASSCOV=1). Sized max_layer x n_expert.
    std::vector<std::vector<double>> massc_mass;  // [layer][expert] summed selection mass
    std::vector<double> massc_total;              // [layer] total mass of all selections
    std::vector<double> massc_cur_cov;            // [layer] mass of selections resident NOW (slot>=0)
    std::vector<uint64_t> massc_sel_total;        // [layer] total SELECTED expert ids seen (count, not mass)
    std::vector<uint64_t> massc_sel_cold;         // [layer] selected expert ids that were COLD (slot<0)
    bool hot_prewarm_done = false;                // prewarm_hot runs once, before the 1st decode
    // [CGC 2026-09-19 slab→pool handoff] Set by the slab prefill path (expert_cache_on_topk's
    // non-decode-graph branch) and consumed once by the next decode step: that path repoints the
    // FFN weights at a per-layer slab and never writes the pool, so without this the only publish is
    // the one-shot prewarm above -- which a server consumes on its first request. Only set when
    // CGC_SLAB_HANDOFF > 0, so the default path leaves it at 0 and does nothing with it.
    int handoff_pending = 0;
    std::vector<std::vector<int32_t>> slot_owner;        // [layer][slot] = expert (-1 free)
    std::vector<std::vector<uint64_t>> slot_last_use;    // [layer][slot]
    std::vector<std::vector<uint8_t>>  slot_queued;      // [layer][slot] 1 = prefetch queued to bg, fill not started yet
    std::vector<std::vector<uint8_t>>  slot_loading;     // [layer][slot] 1 = bg thread filling (prefetch in flight)
    std::vector<std::vector<uint8_t>>  slot_decode_reserved;
    uint64_t n_prefill_defer_yield = 0;  // prefill fills that had to evict a deferred slot anyway (overflow)
    uint64_t n_defer_skip = 0;           // victim choices the defer rule actually changed (engagement proof)

    std::vector<std::vector<uint8_t>>  slot_pinned;      // [layer][slot] 1 = LRU-exempt (decode tail-union prewarm, TAILPIN)
    std::vector<std::vector<uint8_t>>  slot_pinned_static; // [layer][slot] 1 = LRU-exempt static profile pin (LLAMA_EXPERT_CACHE_PIN_PROFILE, never unpinned)
    // [CGC P1 prefill-protect 2026-09-12] [layer][slot] 1 = this slot's current owner was filled
    // during a DECODE phase (VERIFY / DRAFT / NORMAL_DECODE). With CGC_PREFILL_PROTECT=1 a fill
    // issued during PREFILL defers evicting these slots to the overflow pass, so a long prompt's
    // union (<= n_batch*topk = 64 slots/layer at the default L4 chunk) cannot churn away the
    // decode-converged working set (~45 slots/layer) against 143 available. This changes only
    // WHICH slot an expert lands in -- the fill, the remap and the maths are untouched, so the
    // oracle M1/M2 gate must still come back bit-identical. Default OFF = eviction untouched.
    // [CGC Hybrid 2026-09-05] Soft Pool tier partition (env CGC_SOFT_POOL_L0 / CGC_SOFT_POOL_L1):
    // slot indices [0, soft_pool_l0) are L0 hot (no LRU eviction), [soft_pool_l0, soft_pool_l0+l1)
    // are L1 warm (LRU). soft_pool_l0 + soft_pool_l1 <= slots_l(layer) — slots beyond L1 are
    // spare (currently unused; reserved for future L2 spillover). When both tiers are 0 the
    // pool is unpartitioned (legacy uniform behavior, byte-identical to pre-Soft-Pool).
    uint32_t soft_pool_l0 = 0;
    uint32_t soft_pool_l1 = 0;
    // [CGC prefetch v2] recently-evicted experts per layer (ring). When pick_slot evicts an
    // expert to make room, that expert is a prime "will-be-needed-again" candidate (miss
    // analysis: 1062 misses across 126 steps = only 80 distinct experts, 72 of them repeated —
    // i.e. recurring hot experts that LRU evicted between uses, then miss again). The B-section
    // prefetch re-residents these so the next ensure is a HIT instead of a synchronous pread.
    std::vector<std::vector<uint32_t>> evicted_recent;   // [layer] ring of recently-evicted experts (newest at back)
    std::vector<uint32_t> evicted_ring_size;             // [layer] per-layer ring capacity
    // [CGC Step-2 weighted cold guard 2026-09-06] per-layer weighted cold ratio from the
    // logits eval callback (sum of cold-expert softmax weights / sum of all weights), consumed
    // by expert_cache_on_topk for the CGC_FAST_COLD_MAX decision. Overwritten each step.
    // 1e9 = unset (logits callback hasn't fired for this layer yet this step).
    std::vector<double> cgc_weighted_cold_ratio;         // [layer], 1e9 = unset
    // [CGC §8.101 A/B] per-layer slot capacity (LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap,...";
    // default = n_slots for all layers). Sized max_layer; layer 0 (skip) unused.
    std::vector<uint32_t> n_slots_l;
    // [CGC WIN_PIN] rolling per-layer step-union window (last K ensure_batch calls): resident
    // members get slot_pinned (LRU-exempt) so recurring hot experts are not evicted between
    // uses. Miss analysis (steady MTP, 4GiB pool): 12291 misses are mostly REPEATS — hot
    // experts LRU-evicted then needed again. Pure replacement-policy change, runs on the
    // synchronous path under cache->m (never writes pool bytes) → no MTP bg-thread race.
    // LLAMA_EXPERT_CACHE_WIN_PIN=K enables (0 = default off = old pure-LRU behavior).
    std::vector<std::deque<std::vector<uint32_t>>> win_union; // [layer] last K step unions
    std::vector<std::vector<uint32_t>> pin_profile;        // [layer] experts pinned by the static profile (load-time filled + marked)
    // [CGC routing-aware placement 2026-08-29] pin_profile lookup set: when a listed expert is
    // filled (ensure_batch), its slot gets slot_pinned_static = 1. pick_slot skips static pins
    // in passes 0/1; overflow pass 2 may evict the LRU static pin when a fill needs a slot and
    // nothing else is available (a skipped fill would leave table == -1 -> strict catch-up
    // remap reads OOB -> NaN cascade; waiting would hang — see pick_slot).
    std::vector<std::unordered_set<uint32_t>> pin_set;     // [layer] O(1) membership for pin_profile
    std::deque<std::tuple<uint32_t, int32_t, uint32_t>> pool_queue; // (layer, slot, expert) queued pool fills (FIFO)
    // [CGC MTP fast path] reserved ZERO-slot: 1 when the layer's last slot region has been
    // zeroed (guarded by m). Only touched when CGC_VERIFY_DECODE / CGC_DRAFT_DECODE is set.
    std::vector<uint8_t> zero_slot_done; // [layer] 1 = reserved slot zeroed once

    // telemetry
    size_t n_requests = 0;
    size_t n_hits     = 0;
    size_t n_misses   = 0;
    // [CGC 2026-09-16 Blocker B] hits that had to ADOPT an in-flight bg fill: the expert already
    // owned a slot (slot_owner set, slot_queued/loading set) but had not been published to the
    // slot table yet, so the miss test alone would have given it a second slot. Nonzero here is
    // the count of double-ownership events that used to happen silently.
    size_t n_hit_adopted_queued = 0;
    // [CGC 2026-09-16 Blocker B -- is the fix even REACHABLE where we ship?] The two-pass split
    // changes nothing unless a batch has to EVICT a resident slot while assigning its misses: with
    // no eviction during assignment, pass 1/pass 2 and the old interleaved loop produce identical
    // assignments. The claim "in the non-split shipping configuration this is a no-op" was measured
    // on WARMUP (docs/M1_POOL_SPLIT_COST_2026-09-14.md §4.1), and warmup never fills the pool --
    // but a served prefill chunk against the 8 GiB pool may. This counts the batches where it did,
    // so "no-op there" stops being an assumption carried over from a different workload.
    // Nonzero = the assignment ORDER is load-bearing in that run.
    size_t n_batch_evict_batches = 0;
    // [CGC 2026-09-16 Blocker B] Totals for the LLAMA_EXPERT_CACHE_BATCH_INVARIANT gate. Without
    // them, "no VIOLATIONS line in the log" is the only signal available, and an instrument that
    // reports health by printing nothing cannot be told apart from one that never ran (B12). The
    // gate prints a per-layer OK line only for il<=2, so n_batch_inv_checks is what proves it ran
    // on all 40 layers rather than that it ran on three.
    size_t n_batch_inv_checks     = 0;
    size_t n_batch_inv_violations = 0;
    // [CGC miss attribution 2026-09-13] Split every pooled miss by whether this (layer, expert)
    // has ever been DEMANDED before in this session:
    //   compulsory = first demand touch. No amount of slots removes it; only a workload with
    //                less expert diversity, or a resident set chosen a priori (pin/LoRA), can.
    //   capacity   = demanded once already, so it was resident and then evicted -> this is what
    //                slots / LRU policy / prefetch can remove.
    // The ratio is what decides whether the lever is pool capacity or routing locality, and the
    // existing counters cannot see it: n_evictions counts evictions, but an evicted expert that
    // is never re-demanded costs nothing, and one that IS re-demanded is exactly a capacity miss.
    // (The MTP fast path's ZERO-slot path is a third class: it never fills, so its "cold" count
    // lives in n_fast_cold and is deliberately NOT part of this split.)
    size_t n_miss_compulsory = 0;
    size_t n_miss_capacity   = 0;
    size_t n_evictions       = 0;
    // [CGC verify-strict 2026-09-13] Ground-truth quality counters. Both must stay 0 in a healthy
    // run. n_verify_strict_refused counts fast-path steps that were REFUSED because a selected
    // expert was still cold after the fill attempt (the exact path ran instead), and
    // n_zero_mapped_selected counts selected experts that were actually read from the reserved
    // ZERO slot — i.e. whose weight contribution was silently dropped. That second number is
    // exactly the pool-size-dependent quality leak (7/39 divergence class); it is now counted and
    // printed instead of being invisible.
    size_t n_verify_strict_refused = 0;
    size_t n_zero_mapped_selected  = 0;
    // [CGC 2026-09-15 §8.3 gate quantity] S1 slot-table health, counted on EVERY publish (the
    // counting costs one integer add, so it is unconditional; only the printing is gated).
    //
    // n_slot_table_clamped is the landmine §8.3 identified and it MUST stay 0. The host leaf path
    // and the GPU table are only equivalent while the layer has a reserved ZERO slot: with
    // zero_slot < 0, a selected expert that is not resident makes the leaf write -1 (loud: the
    // consumer reports an out-of-range id) while the table write clamps to 0 (SILENT: index 0 is a
    // legal index, so mul_mat_id reads a DIFFERENT expert's weights and the answer is quietly
    // wrong). Every S1 arm measured so far is exactly that configuration, because zero_slot is
    // gated on the MTP fast path and every S1 arm runs CGC_SERVER_MTP=0. The count is currently 0
    // only because verify-strict keeps every selected expert resident -- an accident of the profile,
    // not an invariant of the code. Promoting it to a gate quantity is the repair that preserves
    // bit-identity (reserving a ZERO slot would change pick_slot's arithmetic and therefore the
    // pool layout, i.e. it would move the very numbers the S1 gate compares).
    size_t n_slot_table_publishes = 0;
    size_t n_slot_table_clamped   = 0;
    // [CGC 2026-09-15 take 2] n_slot_table_clamped is the WHOLE-TABLE count and is 44.1% on this
    // profile by construction (143 slots, 256 experts -> most experts are non-resident at any
    // instant), so a nonzero value there is not a finding. n_slot_table_clamped_selected counts the
    // clamp among the ids the consumer ACTUALLY reads this step -- that is the one that must be 0,
    // because it means a selected expert silently read slot 0 (someone else's weights) instead of
    // the -1 the host leaf would have written.
    size_t n_slot_table_clamped_selected = 0;
    // The same split for the churn question: n_slot_table_changed counts whole-table entry moves
    // (informational), while n_slot_table_consumed_changed counts publishes in which an id the
    // consumer reads differs from the same layer's previous publish. Premise B turns on the latter:
    // if the consumed mapping never moves between steps, republishing is redundant work and no
    // per-step host->GPU ordering requirement exists.
    size_t n_slot_table_consumed_changed = 0;
    size_t n_slot_table_consumed_same    = 0;
    // [CGC 2026-09-20 §G1-B] The consumed-subset counts, SPLIT BY THE STEP'S n_tokens. Premise B
    // asks one question -- does the mapping of an id the consumer reads move between two steps --
    // but `consumed_total = n_tokens * n_expert_used`, so a single scalar ANSWERS IT FOR A MIXTURE
    // whose weights the reader cannot recover. Measured on the delivery configuration: 73.4% overall
    // while only 19% of that rate was decided by the delivery decode step (ntok=4); the rest was
    // ntok=8 prompt chunks carrying twice the ids per publish. Shaping the run instead of the
    // instrument does NOT work here: the only tool with a prompt knob (http_duo.py) cannot drive
    // prod25 at all (its prompt does not fit ctx 4096 -- the attempt produced one step and its own
    // `NOT QUOTABLE: only 1 kept rep(s)`).
    std::map<int64_t, size_t> n_slot_table_consumed_changed_by_ntok;
    std::map<int64_t, size_t> n_slot_table_consumed_same_by_ntok;
    // [CGC 2026-09-20 §EN-317] THE ENTRY-LEVEL COUNTER THE ROUTE DECISION NEEDS. The two pairs above
    // are PUBLISH-LEVEL EVENTS: `n_slot_table_consumed_changed` goes up at most ONCE per publish, when
    // AT LEAST ONE of the consumed ids differs from the same layer's previous publish. Measured on the
    // delivery ntok=4 step that is 42.3% -- but a publish in which 1 of 32 ids moved and one in which
    // 32 of 32 moved are INDISTINGUISHABLE inside it, while the criterion the code states one screen
    // up (`the question is whether it is 0/N or N/N`) IS that distinction. Premise B, and with it the
    // S3-vs-S2 route, was therefore decided on a counter that cannot answer its own question.
    // These four accumulate the SAME comparison, over the SAME ids, at entry granularity: the
    // denominator is `sum over publishes of (# consumed ids)`, NOT `# publishes`. By construction
    // entry_rate <= publish_rate, and the ratio between the two is the mean number of ids that moved
    // inside a publish that moved at least one.
    size_t n_slot_table_consumed_moved_entries = 0;
    size_t n_slot_table_consumed_total_entries = 0;
    std::map<int64_t, size_t> n_slot_table_consumed_moved_entries_by_ntok;
    std::map<int64_t, size_t> n_slot_table_consumed_total_entries_by_ntok;
    // [CGC 2026-09-15 premise B] How often the published table actually CHANGES. §9.17(d) asked for
    // "publish_slot_table call count per decode step", which is degenerate by construction (both of
    // its call sites are inside the S1 hook itself, so the answer is 0 on the baseline arm and once
    // per served layer on S1 -- mechanism presence, not state velocity). The question premise B
    // really needs is whether the table's CONTENT moves between steps: if it does not, publication
    // is redundant work and no per-step host->GPU ordering requirement exists, which is exactly the
    // condition that lets D3 collapse n_segs. n_slot_table_changed counts entries that differed
    // from the same layer's previous publish; n_slot_table_unchanged counts publishes with no
    // change at all.
    size_t n_slot_table_changed   = 0;
    size_t n_slot_table_unchanged = 0;
    std::vector<std::vector<uint8_t>> ever_loaded;   // [layer][expert] 1 = demanded at least once
    std::vector<uint32_t>             n_distinct_demanded; // [layer] distinct experts ever demanded
    // [CGC MTP fast-path telemetry] decode fast path (touch + ZERO-slot): union members examined
    // vs COLD members (slot table == -1 -> ZERO-mapped: that expert's real weight contribution
    // is lost for the step). The final-stats decode/pool hit rate counts only the
    // ensure_slot/ensure_batch paths (prefill / catch-up); the fast path never fills and its
    // cold rate was unmeasured — this is the REAL steady decode miss rate (it drives
    // verify/draft hidden quality -> MTP accept rate). Draft (ctx MTP) split kept separately.
    size_t n_fast_calls        = 0; // touch() invocations (steps x layers)
    size_t n_fast_union        = 0; // union members examined
    size_t n_fast_cold         = 0; // of those, ZERO-mapped
    size_t n_fast_draft_calls  = 0;
    size_t n_fast_draft_union  = 0;
    size_t n_fast_draft_cold   = 0;
    // [CGC RSL-MTP instrument 2026-09-18] p_route = P(top8_{t+i} subset-of top8_t), accumulated
    // over (layer, verify step). RSL-MTP (docs/MOE_MTP_FEASIBILITY_2026-09-18.md §4) locks the
    // verify step's per-layer union to the anchor's own top-8 and truncates the draft at the first
    // token whose top-8 is NOT a subset of it. Its entire value depends on this one number and the
    // plan is falsifiable on it (p_route < 0.5 => do not build it). Only verify steps feed it: a
    // verify batch IS t..t+k, so both sides of the comparison are already in one routes array
    // (llama-context.cpp builds the per-token top-k flatten). Index i-1 holds offset i = 1..7.
    // `n_proute_anchor_shrunk` counts anchors whose top-k deduped to fewer than n_expert_used
    // experts (top-k is not top-unique); without it those would read as containment failures.
    size_t n_proute_hit[8]        = {0};  // top8_{t+i} subset-of top8_t
    size_t n_proute_tot[8]        = {0};  // comparisons at offset i
    size_t n_proute_steps         = 0;    // verify steps observed (counted per layer)
    size_t n_proute_anchor_shrunk = 0;    // anchors whose deduped set < n_expert_used
    size_t n_map_requests = 0;  // L3-B ensure() path (prefill / multi-token)
    size_t n_map_hits     = 0;
    // [CGC §8.99-2] loader prewarm fills (llama_model_loader: experts 0..n at load) are kept
    // out of the runtime hit-rate counters — a prewarm cold fill is NOT a runtime miss, and
    // counting it as one understated every published hit rate by ~15pt (81.3% vs real 96%).
    size_t n_prewarm_requests = 0;
    size_t n_prewarm_hits     = 0;
    size_t n_prewarm_misses   = 0;
    size_t n_prefetch = 0;          // pool prefetches queued to the bg thread
    size_t n_prefetch_dropped = 0;  // skipped (queue full / no free slot / already resident) — TOTAL of all drop reasons below
    // [CGC M5 prerouter 2026-09-17] Prefetch-only predictor counters. The roadmap's M5 exit
    // criterion is stated in terms of OVERLAP (ms), not hit rate — so the two numbers here are the
    // ones that decide whether a queued prefetch could have helped at all:
    //   n_prerouter_hit / n_prerouter_pred_total = the fraction of predicted experts that the layer
    //   actually selected. A prefetch of an expert nobody selects buys nothing; a prefetch of one
    //   that IS selected is the only case where the synchronous pread could have been avoided.
    // Whether that saving shows up as wall-clock is decided by the existing miss attribution
    // (compulsory vs capacity) plus CGC-PHASE's fill_wait, NOT by these counters.
    size_t n_prerouter_calls       = 0;  // predict() invocations with the predictor ON
    size_t n_prerouter_queued      = 0;  // prefetch_slot returned 0 (queued for the bg thread)
    size_t n_prerouter_nodata      = 0;  // no freq data for that layer -> nothing to predict
    size_t n_prerouter_scored      = 0;  // score() calls that had a prediction to score
    size_t n_prerouter_pred_total  = 0;  // predicted experts that were scored
    size_t n_prerouter_hit         = 0;  // of those, how many the layer actually selected
    // [CGC Prefetch Drop Audit 2026-09-07] per-reason drop counters. The legacy n_prefetch_dropped
    // only counted 2 of 12 drop points (drain_layer + prefetch_slot no-evictable). This struct
    // classifies EVERY silent drop so replay/database/precommit can see the real loss breakdown.
    // Drop points #1-#12 correspond to the audit inventory; each counter is incremented at the
    // exact site where the expert is discarded. n_prefetch_dropped is kept as the TOTAL (sum of
    // all per-reason counters) for backward compatibility with existing logs and dashboards.
    struct prefetch_drop_stats {
        // #1: prefetch_slot — no free slot AND no evictable slot (pool full, LRU all protected)
        size_t no_free_slot = 0;
        // #2: dbuf_refill — backlog full (busy >= CGC_DBUF_CAP → break, skip remaining layers)
        size_t dbuf_cap_skip = 0;
        // #3: drain_layer — clears queued-but-not-started fills (context teardown / layer reset)
        size_t drain_cleared = 0;
        // #4: Draft never blocks — cold → ZERO slot at verify read (fast path doesn't wait)
        //     NOTE: this is a structural drop, not a prefetch_slot failure; counted separately
        //     because it's the downstream consequence of #1-#3, #6-#9, not an independent drop.
        size_t zero_slot_fallback = 0;
        // #5: Pool capacity / LRU evicts predicted-correct experts between steps
        //     (pick_slot batch-mask protects only the current step's union)
        size_t lru_evicted_predicted = 0;
        // #6: bg_loop reassign-race — synchronous fill grabs the slot between queue time and bg pickup
        //     (slot_queued=0, slot_loading=0 → continue, silently discard)
        size_t bg_reassign_race = 0;
        // #7: guard-reject — expert not in key_segs / layer OOR / pool inactive → return -1
        //     (only logged when PFDBG is on; invisible by default)
        size_t guard_reject = 0;
        // #8: DBUF2 scratch invisibility — bg fill completes but publishes to scratch table;
        //     if dbuf2_swap doesn't land before verify reads, completed fill is still invisible
        //     (the fill "succeeded" but the expert reads as cold → ZERO)
        size_t dbuf2_scratch_invisible = 0;
        // #9: One-shot consumption — trigger sets draft_prefetch_valid[layer]=false after queueing;
        //     experts that failed to queue (cap/slot) are never retried — prediction consumed anyway
        size_t one_shot_consumed = 0;
        // #10: Fast-Path Wait expiry (default OFF) — bounded wait_loading deadline expires → ZERO
        size_t fast_wait_expired = 0;
        // #11: Trigger timing gap — predictions for ALL layers submitted only when verify's il==1
        //     hook fires; layers 0-1 of that same verify step have zero lead time from this cycle
        //     (structural: counted as "prediction arrived too late for this step")
        size_t trigger_too_late = 0;
        // #12: Collect skip — draft collection only fires when ctx==MTP && n_tokens==1;
        //     any batched draft step silently produces no prediction
        size_t collect_skipped = 0;
        // Helper: total of all per-reason counters (should equal n_prefetch_dropped + structural)
        size_t total() const {
            return no_free_slot + dbuf_cap_skip + drain_cleared + zero_slot_fallback +
                   lru_evicted_predicted + bg_reassign_race + guard_reject +
                   dbuf2_scratch_invisible + one_shot_consumed + fast_wait_expired +
                   trigger_too_late + collect_skipped;
        }
    } drop_stats;
    // [CGC SpAc 2026-09-06] per-(layer, expert) EMA utility estimator (CGC_SPAC=1), ported from
    // the turbo-fieldfare MoE-SpAc scheduler (MoESpAcEstimator.swift): utility[L][e] =
    // alpha*utility[L][e] + (1-alpha)*routed(e) with alpha=CGC_SPAC_ALPHA (default 0.85). Every
    // routed step feeds the layer's expert ids from expert_cache_on_topk (spac_update: full-EMA
    // decay of all entries then bump routed). Every CGC_SPAC_REFRESH between-step B-sections,
    // spac_prefetch re-targets each layer's pool toward the utility top-K (spac_k non-resident
    // members via prefetch_slot, which LRU-evicts when the layer is full) — a drifting hot
    // profile that follows decode routing instead of the loader's static experts-0..n prewarm.
    // Default 0.5 = unseeded (profile-like warm start, mirrors SpAc's 0.5 seed so an expert
    // that has never routed this session still outranks a 0-utility newcomer on refresh #1).
    std::vector<std::vector<double>> spac_util;   // [layer][expert] EMA utility, seeded 0.5
    uint64_t spac_feeds = 0;                      // routed-feed counter (refresh cadence)
    // [CGC MTP Draft Prefetch 2026-09-07] exact next-step expert prefetch from MTP draft ctx.
    // The MTP draft ctx (ctx_type == MTP, 1-token decode) computes the top-8 expert ids for the
    // NEXT token one step ahead of the trunk verify ctx (ctx_type == DEFAULT, multi-token). Since
    // draft_accept is 92-98%, these draft-predicted expert ids are highly likely to be selected by
    // the verify step. We collect them during draft decode and queue them for prefetch before the
    // verify step starts, so the verify decode finds them resident (no ZERO-slot contamination).
    // CGC_DRAFT_PREFETCH=1 enables; default OFF = byte-identical legacy behavior.
    std::vector<std::vector<uint32_t>> draft_prefetch_ids;  // [layer] draft-predicted top-8 expert ids
    std::vector<bool> draft_prefetch_valid;                   // [layer] true if draft collected this round
    size_t n_draft_prefetch_queued = 0;    // experts queued for prefetch from draft predictions
    size_t n_draft_prefetch_hit = 0;       // draft-predicted experts actually selected by verify
    size_t n_draft_prefetch_miss = 0;      // draft-predicted experts NOT selected by verify (wasted)
    // [CGC prev-token prefetch 2026-09-08] Use the PREVIOUS token's per-layer expert ids to prefetch
    // the CURRENT token. Adjacent tokens have strong routing correlation (~70-90% overlap in MoE
    // expert selection), so this provides a simple, high-hit-rate prediction that covers ALL layers
    // (unlike MTP draft ctx which only computes the last layer). Double-buffered: curr collects the
    // current token's ids while prev is used for prefetch; swapped at the il==1 trigger boundary.
    std::vector<std::vector<uint32_t>> prev_token_expert_ids;  // [layer] previous token's top-8 expert ids (for prefetch)
    std::vector<std::vector<uint32_t>> curr_token_expert_ids;  // [layer] current token's top-8 expert ids (being collected)
    std::vector<bool> prev_token_valid;                          // [layer] true if prev has valid data
    size_t n_prev_token_prefetch_queued = 0;  // experts queued from prev-token prediction
    size_t n_prev_token_prefetch_hit = 0;     // prev-token predicted experts actually selected
    size_t n_prev_token_prefetch_miss = 0;    // prev-token predicted experts NOT selected
    // [CGC routing-aware placement 2026-08-29] static-pin telemetry: how many fills landed on
    // pin_profile members (got slot_pinned_static) and how many static pins were evicted by
    // pick_slot's overflow pass 2 (a fill needed the slot and nothing else was available —
    // the pin is a preference, not a capacity reservation).
    size_t n_pin_marked = 0;
    size_t n_pin_yield  = 0;
    // [CGC 2026-09-15 SpAc victim FIX] engagement counters for the two new guards in pick_slot.
    // n_spac_nonfinite > 0 means a spac_util row held a non-finite value (a corrupt or
    // unbounded EMA) and the slot fell back to pure-LRU victim choice; n_spac_lru_fallback > 0
    // means the SpAc branch selected nothing at all for a pass and the flag-filtered LRU
    // candidate was used instead of returning -1. Both are expected to be 0 on a healthy run;
    // either being non-zero is the fingerprint of the pre-fix abort.
    size_t n_spac_nonfinite    = 0;
    size_t n_spac_lru_fallback = 0;
    std::atomic<size_t> n_reads{0};
    // [CGC miss-path cost audit 2026-09-13] bytes actually fetched from the file. n_reads counts
    // preadv JOBS (one per file-contiguous run after merge-read), so bytes/jobs is the mean run
    // size -- without it there is no way to convert the reported MB/s into a device-vs-cache
    // verdict.
    std::atomic<uint64_t> n_read_bytes{0};
    // [CGC M2 pool reuse 2026-09-14] bytes the whole-layer slab path took from the RESIDENT pool
    // (a memcpy) vs from the file (a pread). The M2 exit condition is stated in bytes/token, and
    // until these counters existed there was no way to tell "the fill read 100% of the layer" from
    // "it read only the non-resident share": the CGC-PREFILL-STREAM line printed the slab's SIZE,
    // which is identical in both cases. Together with n_read_bytes these make the I/O budget
    // measurable instead of inferred.
    std::atomic<uint64_t> n_slab_bytes_pool{0};
    std::atomic<uint64_t> n_slab_bytes_disk{0};
    std::atomic<uint64_t> pread_usec{0}; // accumulated pread wall time (us)
    std::atomic<uint64_t> fill_batch_usec{0}; // hook-thread elapsed per fill batch (us; comparable across serial/parallel)

    // [CGC decode phase decomposition 2026-09-15] Wall time this thread spent BLOCKED inside the
    // expert-cache fill paths -- both the `bg_cv.wait` for an in-flight prefetch and the
    // synchronous `fill_pool_direct` pread on a miss. This is the missing term that CGC-PHASE
    // cannot see: `CGC-PHASE ... compute=` measures all of graph_compute, which contains both the
    // Metal layer encoding AND this fill wait, so a step whose entire cost sits in `compute` looks
    // identical whether the time went to the GPU or to the disk.
    //
    // Semantics differ from pread_usec on purpose. pread_usec is an AGGREGATE across worker
    // threads (it can exceed wall time by the worker count), so it cannot be compared against a
    // step's wall clock. This one is accumulated by the CALLING thread only, so
    // `fill_wait_us(step) / n_tokens` IS comparable to the per-token wall time.
    std::atomic<uint64_t> fill_wait_us{0};


    ~llama_expert_cache();

    void bg_loop();
};

// [CGC §8.101 A/B] per-layer slot capacity: LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap;..."
// (e.g. "0-3:256;4-39:180" = layers 0..3 get 256 slots, 4..39 get 180). Later segments override
// earlier ones; layers not covered keep `def`. Read once per process. Used by BOTH the loader
// (expert tensor ne[2] per layer -> Metal buffer sizing) and the cache (slot vectors + pool
// regions). Without the env every layer keeps the uniform n_slots (byte-identical behavior).
static inline uint32_t cgc_layer_cap(uint32_t layer, uint32_t def) {
    static const char * env = getenv("LLAMA_EXPERT_CACHE_LAYER_CAPS");
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    uint32_t cur = def;
    const char * p = env;
    while (*p != '\0') {
        uint32_t start = 0, end = 0, cap = 0;
        if (sscanf(p, "%u-%u:%u", &start, &end, &cap) == 3 && layer >= start && layer <= end) {
            cur = cap;
        }
        while (*p != '\0' && *p != ';' && *p != ',') {
            ++p;
        }
        if (*p == ';' || *p == ',') {
            ++p;
        }
    }
    return cur;
}

// [CGC DBUF spike 2026-09-06] step-ahead cold-expert refill (CGC_DBUF=1). Default OFF =
// byte-identical stock fast path. When on, the decode fast-path hook refills the step's
// ZERO-mapped cold experts (the ones whose weight was silently dropped this step) through
// prefetch_slot, so the NEXT decode step finds them resident (step-to-step routing stability
// ~87%). The hook runs between GPU segments (layer il's FFN seg is submitted only after the
// hook returns and its remap references union slots only), and prefetch_slot evicts only
// non-union LRU slots + publishes slot_table only after bytes land -> race-free (see
// llama-context.cpp call site + spike doc).
//
// NOTE 2026-09-16: the predicate here used to be `getenv(...) != nullptr && [0] != '\0'`, i.e. "set
// to anything at all". Under it a literal `CGC_DBUF=0` was ON, and run_p1_mmap_stream.sh writes
// exactly that literal on the line under a comment saying "in P1 mode turn the pool optimisations
// off" -- so the knob's own value in P1 mode said ON while the comment next to it said OFF. Whether
// the P1 (mmap identity-mapping) path actually REACHES the call site (llama-context.cpp:5780, gated
// on `verify_fast`) is NOT established here: a knob's value states intent, not behaviour
// (CONVENTIONS A9). What was wrong, and is now fixed, is the value itself.
// (run_server.sh escaped this only by omission: it pushes nothing when CGC_DBUF=0, so the wrapper's
// documented disable path worked by luck of the shell rather than by the reader.) Value semantics
// now come from cgc_env_on(), so "0" means off wherever it is written.
static inline bool cgc_dbuf_on() {
    static const bool v = cgc_env_on("CGC_DBUF");
    return v;
}

// [CGC DBUF2 full double-buffer 2026-09-06] slot_table active/scratch double-buffering. When
// CGC_DBUF2=1, bg fills write slot_table_scratch while decode reads slot_table (active); at the
// per-step swap point the two are O(1)-swapped and the new scratch reset to -1. This removes the
// spike's restriction that prefetch_slot can only evict non-union slots — fills can target ANY
// slot in scratch because decode never reads scratch. Default OFF = legacy single-buffer behavior
// (fills write slot_table directly, byte-identical).
// NOTE 2026-09-16: converted with the rest of the family. No script sets CGC_DBUF2 at all, so this
// is behaviour-preserving today; it is converted so that the family cannot drift back apart and so
// that `CGC_DBUF2=0` will mean what it says the first time someone writes it.
static inline bool cgc_dbuf2_on() {
    static const bool v = cgc_env_on("CGC_DBUF2");
    return v;
}

// [CGC Fast-Path Wait 2026-09-06] when a cold expert has an in-flight fill (loading/queued by
// DBUF step-ahead refill or prefetch), the fast path briefly waits instead of ZERO-mapping it.
// pread fills take ~1-3ms; the decode step window is ~30-60ms, so a <=2ms wait catches most
// in-flight fills without materially slowing decode. Default OFF = byte-identical (immediate
// ZERO-slot). CGC_FAST_WAIT_US default 1000 (1ms); CGC_FAST_WAIT_MAX default 4 = at most N
// experts waited per step (bounds total step latency <= MAX * wait_us).
// NOTE 2026-09-16: converted with the rest of the family (see cgc_dbuf_on). No script sets
// CGC_FAST_WAIT, so behaviour-preserving today. run_server.sh does forward it verbatim when set
// (run_server.sh:1079), which is the path where a literal `=0` would previously have meant ON.
static inline bool cgc_fast_wait_on() {
    static const bool v = cgc_env_on("CGC_FAST_WAIT");
    return v;
}
static inline uint64_t cgc_fast_wait_us() {
    static const uint64_t v = []() {
        const char * s = getenv("CGC_FAST_WAIT_US");
        int x = (s != nullptr && s[0] != '\0') ? atoi(s) : 1000;
        if (x < 50) x = 50;
        if (x > 5000) x = 5000;
        return (uint64_t) x;
    }();
    return v;
}
static inline uint32_t cgc_fast_wait_max() {
    static const uint32_t v = []() {
        const char * s = getenv("CGC_FAST_WAIT_MAX");
        int x = (s != nullptr && s[0] != '\0') ? atoi(s) : 4;
        if (x < 1) x = 1;
        if (x > 16) x = 16;
        return (uint32_t) x;
    }();
    return v;
}

// [CGC SpAc 2026-09-06] config helpers (read once per process). CGC_SPAC=1 enables the EMA
// utility prefetch source (replaces the default step/hist source in the B-section; default OFF
// = stock behavior, bit-identical). CGC_SPAC_ALPHA default 0.85 (SpAc's measured inertia),
// CGC_SPAC_K default 8 = top-K per layer re-targeted each refresh, CGC_SPAC_REFRESH default 1
// = run the refresh every Nth B-section (1 = every step, the Swift's per-step prefetch cadence).
//
// NOTE 2026-09-16: see cgc_dbuf_on -- run_p1_mmap_stream.sh writes `CGC_SPAC=0` under a comment
// declaring the pool optimisations off, so this knob's own value in P1 mode also said ON while its
// neighbour said OFF. Same caveat as there: the call sites (llama-context.cpp:1942 source selection
// and :4739 utility update) were not traced through the P1 identity-mapping path. Converted to
// cgc_env_on() so the value means what it says.
static inline bool cgc_spac_on() {
    static const bool v = cgc_env_on("CGC_SPAC");
    return v;
}
static inline double cgc_spac_alpha() {
    const char * s = getenv("CGC_SPAC_ALPHA");
    double v = 0.75;  // [2026-09-06 tuned] 11-point sweep (0.5-0.9): alpha=0.75 gives best
                      // median decode (25.87 t/s) + best peak (26.19 t/s) + 3/3 full-512
                      // stability + 2/3 runs above 25 t/s on 16GB M4 Max, 8GB pool, DBUF+SPAC.
                      // Override via CGC_SPAC_ALPHA env for HarmonyOS / other platform tuning.
    if (s != nullptr && s[0] != '\0') {
        v = atof(s);
        if (v < 0.0) v = 0.0;
        if (v > 0.999) v = 0.999;
    }
    return v;
}
static inline uint32_t cgc_spac_k() {
    const char * s = getenv("CGC_SPAC_K");
    uint32_t v = 8;
    if (s != nullptr && s[0] != '\0') {
        v = (uint32_t) atoi(s);
        if (v < 1) v = 1;
        if (v > 128) v = 128;
    }
    return v;
}
static inline uint32_t cgc_spac_refresh() {
    const char * s = getenv("CGC_SPAC_REFRESH");
    uint32_t v = 1;
    if (s != nullptr && s[0] != '\0') {
        v = (uint32_t) atoi(s);
        if (v < 1) v = 1;
        if (v > 256) v = 256;
    }
    return v;
}

// [CGC MTP Draft Prefetch 2026-09-07] exact next-step expert prefetch from MTP draft ctx.
// When CGC_DRAFT_PREFETCH=1, the draft ctx's top-8 expert ids are collected per layer and
// queued for prefetch before the verify step starts. This attacks the count-cold problem
// (15% of selected experts are non-resident) by prefetching the EXACT experts the draft
// predicts will be needed next step (draft_accept 92-98% = prediction accuracy). Default
// OFF = byte-identical legacy behavior (no draft prefetch).
static inline bool cgc_draft_prefetch_on() {
    static const bool v = getenv("CGC_DRAFT_PREFETCH") != nullptr &&
                          getenv("CGC_DRAFT_PREFETCH")[0] != '\0';
    return v;
}

// [CGC Hybrid 2026-09-05] Soft Pool tier capacities: CGC_SOFT_POOL_L0 / CGC_SOFT_POOL_L1
// partition the per-layer slot pool into L0 (fixed hot, no LRU eviction) and L1 (warm, LRU
// evict). Default 32/32 matches the doc's recommended 64-slot per-layer configuration
// (saving 27% resident memory vs 87 slots). L0+L1=0 disables the partition (legacy uniform
// n_slots behavior, byte-identical to pre-Soft-Pool). Read once per process via
// cgc_soft_pool_tier() and recorded on cache->soft_pool_{l0,l1} for runtime introspection.
static inline uint32_t cgc_soft_pool_tier(const char * env_name, uint32_t fallback) {
    const char * env = getenv(env_name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    long v = atol(env);
    if (v < 0) {
        return 0; // explicit 0 (or negative treated as 0) = "no slots in this tier"
    }
    return (uint32_t) std::min<long>(v, 256); // cap at 256 (= one slot per expert)
}

// L3 Option A: per-layer static slot pool API. Returns the slot index holding (layer, expert)
// after ensuring it is resident (blocking pread on miss, LRU eviction within the layer), or -1
// on error. fill()/ensure() remain available for the L3-B gather path.
int32_t llama_expert_cache_ensure_slot(llama_expert_cache * cache, uint32_t layer, uint32_t expert,
                                  bool count = true); // count=false: loader-prewarm accounting (n_prewarm_*)
// L3 Option A batch (cross-expert parallel fill, LLAMA_EXPERT_CACHE_BATCH=1): ensure ALL
// experts of one layer, filling every miss CONCURRENTLY (one thread per missed expert; each
// fill is itself 3-segment parallel). The hook's serial per-expert loop pays sum(miss fills)
// on the critical path; the batch collapses it to max(miss fills) — SSD queue depth permitting.
// Same semantics as ensure_slot per expert (blocking on miss, LRU within the layer, slot table
// + last_use updated on return). Must be called WITHOUT cache->m held.
// `defer_decode_protect`: true when this fill is issued by a PREFILL step (see CGC_PREFILL_PROTECT).
// It makes pick_slot defer evicting slots whose owner was filled during decode, and it stamps the
// slots this call fills as non-decode. Default false = byte-identical legacy eviction order.
void llama_expert_cache_ensure_batch(llama_expert_cache * cache, uint32_t layer,
                                     const uint32_t * experts, size_t n,
                                     bool defer_decode_protect = false);
// [CGC PREFILL_PROTECT A/B 2026-09-13] Runtime override of the CGC_PREFILL_PROTECT victim rule
// so both arms of an A/B can share ONE server process (one machine state). -1 = env default,
// 0 = force off, 1 = force on. `refresh` re-reads the CGC_PREFILL_PROTECT_FILE toggle file and
// is called once per batch; without that env var it is a no-op and behaviour is unchanged.
void llama_expert_cache_set_prefill_protect(int mode);
void llama_expert_cache_refresh_prefill_protect();
// Prefill hot prewarm: accumulate (layer, expert) route frequencies (prefill only; repeated
// across tokens counts multiple times). Then llama_expert_cache_prewarm_hot fills the pool with
// each layer's top-K most-routed experts at the first decode step. Returns 0 when skipped.
void llama_expert_cache_record_routes(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n);
// [CGC 2026-09-19 slab→pool handoff] Fill each layer's pool with its top-`cap` most-routed experts
// from the recorded prefill routing, WITHOUT the one-shot guard, evicting when the layer is full
// (cap == 0 = legacy prewarm_hot: one-shot, whole slot budget, what the default path calls).
// Called at the first decode step of a request whose prefill went through the slab path.
// Synchronous; returns the number of experts ensured. See the definition for the cost model.
size_t llama_expert_cache_prewarm_hot_capped(llama_expert_cache * cache, size_t cap);

// [CGC RSL-MTP instrument 2026-09-18] Accumulate p_route over verify steps. `experts` is the
// per-token top-k flatten ([n_tokens][n_expert_used], token-major) exactly as built in
// llama-context.cpp; `n_tokens` is the step width (1 for decode, 1+k for MTP verify).
void llama_expert_cache_record_proute(llama_expert_cache * cache,
                                      const uint32_t * experts, size_t n_tokens,
                                      size_t n_expert_used);
// [CGC SpAc 2026-09-06] EMA utility update: decay all of layer's utilities by alpha, then bump
// the routed experts by (1-alpha). Called from expert_cache_on_topk with the step's route list
// (CGC_SPAC=1). Cheap (n_expert mults per layer); lock is brief.
void llama_expert_cache_spac_update(llama_expert_cache * cache, uint32_t layer,
                                    const uint32_t * experts, size_t n);
// [CGC SpAc 2026-09-06] between-step pool re-target: per layer, prefetch the top-K non-resident
// experts by EMA utility (prefetch_slot = bg fill; LRU-evicts a slot when the layer is full, so
// the pool drifts toward the utility hot set). Call from the B-section prefetch site every
// CGC_SPAC_REFRESH feeds when CGC_SPAC=1. Non-blocking; returns the number of prefetches queued.
size_t llama_expert_cache_spac_prefetch(llama_expert_cache * cache);
// [CGC DBUF spike 2026-09-06] queue a bg fill for every COLD (slot_table == -1) member of the
// step's union for `layer`. Called from the decode fast-path hook (CGC_DBUF=1). Non-blocking;
// returns the number of fills queued. See cgc_dbuf_on() header comment for the safety model.
size_t llama_expert_cache_dbuf_refill(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n);
// [CGC DBUF2 full double-buffer 2026-09-06] atomically swap active and scratch slot tables,
// then reset the new scratch (old active) to -1. Must be called at a GPU-idle point (after the
// current step's remap leaves are written, before the next step's FFN is dispatched) so decode
// never reads a mid-swap table. Holds cache->m internally. CGC_DBUF2=0 = no-op. Call once per
// decode step (e.g. at the last layer's hook or at the step boundary), not per-layer.
void llama_expert_cache_dbuf2_swap(llama_expert_cache * cache);
// [CGC Fast-Path Wait 2026-09-06] For cold experts that have an in-flight fill (loading or
// queued by DBUF refill / prefetch), wait up to cgc_fast_wait_us() for the fill to land so the
// fast path uses the real expert weight instead of ZERO-mapping it. At most cgc_fast_wait_max()
// experts are waited per step (bounds total added latency). Experts that become resident during
// the wait have their slot_table updated by the bg thread; callers re-read slot_table after this
// returns. Must NOT be called holding cache->m (takes and releases it internally). Default OFF
// (cgc_fast_wait_on()=false) = no-op, byte-identical fast path. Returns the number of experts
// that became resident during the wait.
size_t llama_expert_cache_wait_loading(llama_expert_cache * cache, uint32_t layer,
                                       const uint32_t * experts, size_t n);
// [CGC mass-coverage 2026-09-06] accumulate selection softmax mass per expert (massc_mass)
// plus the mass covered by the CURRENT residency (slot_table>=0) — see header comment. Called
// from expert_cache_on_topk when CGC_MASSCOV=1. experts/w_sel: the step's selected expert ids
// and their softmax weights; n = n_tokens * n_expert_used (same layout as the ids tensor).
void llama_expert_cache_masscov_record(llama_expert_cache * cache, uint32_t layer,
                                       const uint32_t * experts, const float * w_sel, size_t n);
size_t llama_expert_cache_prewarm_hot(llama_expert_cache * cache);
// Tail-union prewarm (LLAMA_EXPERT_CACHE_TAILPIN=1, 2026-08-15): pin the RESIDENT experts of
// the last K prefill tokens' union so decode's LRU eviction cannot hand their slots out during
// the first decode step (no fill — only protects what prefill already loaded). Returns the
// number of experts actually pinned (resident only; non-resident are ignored).
size_t llama_expert_cache_pin_experts(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n);
// Clear all pin flags (called at the end of the first decode step: the protection window is
// one step, after which the pinned slots go back to normal LRU).
void llama_expert_cache_unpin_all(llama_expert_cache * cache);
// Returns the base pointer of the pool region for (layer, kind), or nullptr.
const uint8_t * llama_expert_cache_pool_data(const llama_expert_cache * cache, uint32_t layer, int kind);
// Per-kind stride within the pool (bytes per slot). 0 if kind absent for the layer.
size_t llama_expert_cache_pool_stride(const llama_expert_cache * cache, uint32_t layer, int kind);
uint32_t llama_expert_cache_slots_per_layer(const llama_expert_cache * cache);
// [CGC decode phase decomposition 2026-09-15] Calling-thread wall time blocked in the fill paths
// (monotonic, in us). Take a delta around one decode step to get that step's fill cost; this is
// the only counter on this path whose units are comparable to a step's wall clock. Returns 0 when
// no cache is active, which is exactly the "no fill to wait for" case.
uint64_t llama_expert_cache_fill_wait_us(const llama_expert_cache * cache);
uint32_t llama_expert_cache_slots_per_layer_l(const llama_expert_cache * cache, uint32_t layer); // [CGC] per-layer cap (LAYER_CAPS)
// Static profile pin (LLAMA_EXPERT_CACHE_PIN_PROFILE=<file>, 2026-08-17): read a per-layer
// top-N expert list (one line per layer, space-separated ids; missing/empty lines = no pins)
// and return the number of pinned experts. The cache records it for load-time fill; pick_slot
// treats static-pinned slots as LRU-exempt forever (unlike TAILPIN they are never unpinned).
size_t llama_expert_cache_load_pin_profile(llama_expert_cache * cache, const char * path);
// L4 zero-copy: adopt (layer, kind) pool region from an expert tensor's own storage (a
// Metal-visible shared buffer at -ngl>0). Returns false if out of range / null. The LRU/fill
// machinery then writes into the tensor's buffer; the Metal FFN reads it directly (zero copy).
bool llama_expert_cache_adopt_pool_region(llama_expert_cache * cache, uint32_t layer, int kind,
        const uint8_t * base, int64_t n_slots, size_t stride);
// CGC M1 work item 1 (CGC_POOL_SPLIT): the pool as its own buffer set. Allocate one buffer per kind
// (sized to every layer's slots x stride, so the whole pool is 3 allocations instead of one per
// (layer, kind) -- 117 Metal buffers is what made the earlier full-width attempt hang on residency
// registration), then register each layer's region inside it. Returns false on any failure and
// leaves the cache unchanged.
bool llama_expert_cache_pool_split_alloc(llama_expert_cache * cache, int kind,
        ggml_backend_buffer_type_t buft, size_t total_bytes, ggml_backend_buffer_t * out_buf);
bool llama_expert_cache_pool_split_adopt(llama_expert_cache * cache, uint32_t layer, int kind,
        const uint8_t * base, uint32_t slots, size_t stride, ggml_backend_buffer_t buf);
// The buffer that owns (layer, kind)'s pool region, or nullptr when the pool is the tensor's own
// storage (the legacy zero-copy path) -- the repoint must only move `buffer` in the former case.
ggml_backend_buffer_t llama_expert_cache_pool_buffer(const llama_expert_cache * cache, uint32_t layer, int kind);
// Record (once, at load) and read back the expert tensor's full-width geometry. The graph sets the
// pool geometry for pool steps and this geometry for wide steps, so neither depends on what the
// previous step happened to leave behind.
void llama_expert_cache_pool_set_wide(llama_expert_cache * cache, uint32_t layer, int kind,
        const void * data, ggml_backend_buffer_t buf, int64_t ne2);
bool llama_expert_cache_pool_get_wide(const llama_expert_cache * cache, uint32_t layer, int kind,
        void ** data, ggml_backend_buffer_t * buf, int64_t * ne2);
// True when the pool owns its allocations (CGC_POOL_SPLIT), i.e. the pool is NOT the expert
// tensors' storage. Callers use it to decide whether the graph must also repoint a tensor's
// `buffer` (and the batch clamp, which only exists because the tensors are shrunk).
bool llama_expert_cache_pool_owned(const llama_expert_cache * cache);
// L4 cold-start (2026-08-15): the loader pre-reads the FIRST n_slots experts of every layer into
// the adopted Metal pool regions (the expert tensor buffers). Mark those slots resident so the
// first accesses are pool HITS instead of re-preading the same bytes — otherwise every short
// prompt pays a cold refill window (measured: L4@4GiB short-prompt decode 122ms vs base 88ms).
void llama_expert_cache_prepopulate(llama_expert_cache * cache, uint32_t layer, uint32_t n_slots);

// [CGC identity-slot verify 2026-09-09] load-time identity fill byte check vs the GGUF file.
// Env-gated: LLAMA_EXPERT_CACHE_VERIFY_IDENTITY=1. Aborts on mismatch.
void llama_expert_cache_verify_identity(llama_expert_cache * cache);
// [CGC 2026-09-06] load-time pin prefill: fill every PIN_PROFILE member (hot experts) into the
// pool and static-pin it, BEFORE any request. Requires load_pin_profile to have run first.
size_t llama_expert_cache_pin_prefill(llama_expert_cache * cache);
// True when the static per-layer pool is active (LLAMA_EXPERT_CACHE_POOL=1 at init).
bool llama_expert_cache_pool_active(const llama_expert_cache * cache);
// Per-layer slot table: expert id -> slot index (-1 = not resident). nullptr when pool inactive.
const int32_t * llama_expert_cache_slot_table(const llama_expert_cache * cache, uint32_t layer);
// [CGC M2 whole-layer streaming 2026-09-14] Read an entire layer's experts (0..n_expert-1) for
// one kind straight from the GGUF file into a caller-provided slab. Expert e lands at
// dst + e * stride. Returns total bytes read, or -1 on short read. Used by the prefill
// streaming path (CGC_PREFILL_STREAM=1) where the union is provably saturated.
int64_t llama_expert_cache_fill_layer_slab(llama_expert_cache * cache, uint32_t layer,
        int kind, void * dst, size_t stride);
// L3 Option A prefetch: queue (layer, expert) for the background thread to fill into the pool.
// Non-blocking; uses only FREE slots (never evicts for a prediction). Returns 0 if queued, -1 if
// skipped (pool inactive / already resident or queued / no free slot / queue full).
int32_t llama_expert_cache_prefetch_slot(llama_expert_cache * cache, uint32_t layer, uint32_t expert);
// [CGC M5 prerouter 2026-09-17] PREFETCH-ONLY expert predictor (env CGC_PREROUTER=1).
//
// Roadmap M5: "only decode, only L+1, only the 8, only time". This ranks the layer's recorded
// route counts (cache->freq, filled by LLAMA_EXPERT_CACHE_ROUTE_RECORD=1) and queues the top-`top_k`
// experts through llama_expert_cache_prefetch_slot(), which is free-slot-only BY CONSTRUCTION
// ("never evicts for a prediction") — i.e. exactly the PREFETCH_ONLY semantics.
//
// WHAT THIS FUNCTION CANNOT DO, stated so a later reader does not have to re-derive it:
//   * it never reserves, allocates or pins a slot;
//   * it never writes slot_owner / slot_table;
//   * it does not consult the current routing decision, so a misprediction costs one queued pread
//     (or nothing at all), never a resident eviction and never a different expert.
// The WIP it is ported from (Backup/wip_p1prime_20260911/) also had a `prerouter_staged_reserve`
// that DID reserve slots before the real routing decision. That half is deliberately NOT ported:
// it makes predictions decide routing, which is the one thing M5's work item forbids.
//
// NOT the trained per-layer linear head the roadmap sketches ("full implementation would require a
// trained linear projection per layer") — that head does not exist and is not trained here. This is
// the frequency fallback the WIP actually shipped, so the honest statement is "a route-frequency
// predictor", and its ceiling is whatever route frequency can predict.
//
// Returns the number of experts queued (0 = nothing queued), or -1 when the predictor is off / has
// no data for that layer. Call it with layer+1 from layer's hook to get the L+1 prefetch.
int32_t llama_expert_cache_prerouter_predict(llama_expert_cache * cache, uint32_t layer, uint32_t top_k);
// Score the prediction that was made FOR `layer` against the experts it actually selected.
// Returns the number of predicted experts that were selected (increments the counters either way).
// No-op (returns 0) when the predictor is off, so a default run pays one predictable-branch test.
uint32_t llama_expert_cache_prerouter_score(llama_expert_cache * cache, uint32_t layer,
                                            const uint32_t * selected, size_t n_selected);
// L3 Option A: stabilize a layer's pool region before its FFN dispatches. The Metal backend's
// async tensor copy is region-wide (all n_slots), so ANY in-flight bg fill for the layer would
// tear it. Drops the layer's still-queued fills (ensure_slot re-fills if actually needed) and
// waits for in-flight ones to complete. The hook calls this after the union ensure loop, right
// before it points the FFN tensors at the pool.
void llama_expert_cache_drain_layer(llama_expert_cache * cache, uint32_t layer);

// [CGC MTP fast path] ZERO-slot mechanism (CGC_VERIFY_DECODE / CGC_DRAFT_DECODE): the last
// slot of every layer is reserved as a zero-initialized "ZERO slot". Cold experts (not resident,
// slot table == -1) map to it, so the decode fast path (which skips the blocking ensure_batch)
// reads a finite zero contribution instead of an OOB pool row (NaN cascade). pick_slot and
// prefetch_slot skip the reserved slot so no real expert is ever assigned to it.
// True when either fast-path env is set (base/non-MTP runs never set them -> byte-identical).
bool llama_expert_cache_zero_slot_enabled(const llama_expert_cache * cache);
// Slot index of the reserved zero slot for `layer` (slots_l(layer) - 1 when enabled, else -1).
int32_t llama_expert_cache_zero_slot(const llama_expert_cache * cache, uint32_t layer);
// Number of slots actually usable for real experts (slots_l - 1 when the ZERO slot is reserved).
uint32_t llama_expert_cache_usable_slots(const llama_expert_cache * cache, uint32_t layer);
// Zero the reserved slot's pool region (all 4 kinds) once per layer (guarded). Must be called
// before the first GPU read of the pool for the layer on the decode fast path.
void llama_expert_cache_zero_reserved_slot(llama_expert_cache * cache, uint32_t layer);
// Decode fast path LRU touch: bump last_use for RESIDENT experts only (no fill, no wait). The
// decode fast path calls this instead of ensure_batch so resident hot experts keep their LRU
// freshness (the next step's pick_slot cannot hand their slots out for a cold fill).
// draft_path: telemetry tag — the caller is the draft context (ctx MTP). Counted separately
// (n_fast_draft_*) so verify vs draft cold rates are distinguishable; defaults false = verify.
void llama_expert_cache_touch(llama_expert_cache * cache, uint32_t layer,
                              const uint32_t * experts, size_t n, bool draft_path = false);
// Slot-table lookup mapping -1 (not resident) to the ZERO slot. Same value as the raw slot
// table for resident experts, so using it on the exact-load path is a no-op (bit-identical).
int32_t llama_expert_cache_slot_table_safe(const llama_expert_cache * cache, uint32_t layer,
                                           uint32_t expert);
// [CGC 2026-09-15 S1 slot-table] Materialise the WHOLE layer's expert->slot map into `dst`
// (n_expert int32), for the GPU-side gather that replaces the host-written remap leaf
// (CGC_SLOT_TABLE_GPU=1). Per expert this produces exactly what
// llama_expert_cache_slot_table_safe returns for it -- resident -> its slot, everything else ->
// the ZERO slot -- so a GPU-computed id vector is the same sequence the host would have written.
//
// It exists as one function because the invariant only holds if EVERY remap-leaf write site
// publishes the table under the same mapping; a per-site reimplementation is how those sites would
// drift apart. Returns the number of entries that had to be clamped to 0 because the layer has no
// reserved ZERO slot (the map would otherwise contain -1 and the gather would read out of bounds);
// callers report a nonzero count instead of treating it as normal.
// [CGC 2026-09-17 S1 clamp accounting]
// `sel_ids` (n_sel_ids entries, the step's raw expert ids) is OPTIONAL and exists so the count can
// be taken INSIDE the publish loop. The caller used to re-derive it afterwards by re-reading the
// live table (`st[e] < 0 && dst[e] == 0`), which is a race: a background fill can land between the
// two loops and turn st[e] non-negative, so the very entries that were published as 0 are then
// counted as fine. Measured 2026-09-17: that counter read 0 while the ids the GPU consumed were
// provably not the table's values, i.e. it reported "nothing wrong" about the bug it exists for.
// With the ids passed in, membership is decided from the same loop that wrote the entries.
//
// out_sel_clamped: consumed ids published as 0 because the layer has no reserved ZERO slot
//                  (the gather then reads slot 0 -- another expert's weights).
// out_sel_wrong:   consumed ids whose published slot is not owned by that expert (superset of the
//                  above; >0 means the consumer reads weights that are not its own).
// Both may be nullptr.
//
// CGC_S1_TAG=<v> (debug only, default off): add v to every published entry and tag the clamped
// ones with 1000+v instead of 0. It exists to answer one question the ring capture cannot: is a
// value the GPU read *published 0* or *never written at all*. Never use it for quality or speed
// numbers -- it deliberately corrupts the mapping.
int64_t llama_expert_cache_publish_slot_table(const llama_expert_cache * cache, uint32_t layer,
                                             int32_t * dst, uint32_t n_expert,
                                             const int32_t * sel_ids = nullptr,
                                             int64_t n_sel_ids = 0,
                                             int64_t * out_sel_clamped = nullptr,
                                             int64_t * out_sel_wrong = nullptr);
