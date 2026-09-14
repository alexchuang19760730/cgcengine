#pragma once

#include "llama.h"

#include <sys/uio.h> // struct iovec (pread_job merge-read form)

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    std::atomic<size_t> n_reads{0};
    // [CGC miss-path cost audit 2026-09-13] bytes actually fetched from the file. n_reads counts
    // preadv JOBS (one per file-contiguous run after merge-read), so bytes/jobs is the mean run
    // size -- without it there is no way to convert the reported MB/s into a device-vs-cache
    // verdict.
    std::atomic<uint64_t> n_read_bytes{0};
    std::atomic<uint64_t> pread_usec{0}; // accumulated pread wall time (us)
    std::atomic<uint64_t> fill_batch_usec{0}; // hook-thread elapsed per fill batch (us; comparable across serial/parallel)

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
static inline bool cgc_dbuf_on() {
    static const bool v = getenv("CGC_DBUF") != nullptr && getenv("CGC_DBUF")[0] != '\0';
    return v;
}

// [CGC DBUF2 full double-buffer 2026-09-06] slot_table active/scratch double-buffering. When
// CGC_DBUF2=1, bg fills write slot_table_scratch while decode reads slot_table (active); at the
// per-step swap point the two are O(1)-swapped and the new scratch reset to -1. This removes the
// spike's restriction that prefetch_slot can only evict non-union slots — fills can target ANY
// slot in scratch because decode never reads scratch. Default OFF = legacy single-buffer behavior
// (fills write slot_table directly, byte-identical).
static inline bool cgc_dbuf2_on() {
    static const bool v = getenv("CGC_DBUF2") != nullptr && getenv("CGC_DBUF2")[0] != '\0';
    return v;
}

// [CGC Fast-Path Wait 2026-09-06] when a cold expert has an in-flight fill (loading/queued by
// DBUF step-ahead refill or prefetch), the fast path briefly waits instead of ZERO-mapping it.
// pread fills take ~1-3ms; the decode step window is ~30-60ms, so a <=2ms wait catches most
// in-flight fills without materially slowing decode. Default OFF = byte-identical (immediate
// ZERO-slot). CGC_FAST_WAIT_US default 1000 (1ms); CGC_FAST_WAIT_MAX default 4 = at most N
// experts waited per step (bounds total step latency <= MAX * wait_us).
static inline bool cgc_fast_wait_on() {
    static const bool v = getenv("CGC_FAST_WAIT") != nullptr && getenv("CGC_FAST_WAIT")[0] != '\0';
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
static inline bool cgc_spac_on() {
    static const bool v = getenv("CGC_SPAC") != nullptr && getenv("CGC_SPAC")[0] != '\0';
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
