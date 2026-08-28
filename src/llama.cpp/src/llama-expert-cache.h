#pragma once

#include "llama.h"

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
    std::vector<int32_t> slot_table;  // [n_layer * n_expert]
    std::vector<std::vector<std::vector<uint8_t>>> pool; // [layer][kind][slot*stride ..] (malloc path)
    // L4 zero-copy (-ngl>0 + ALLOW_NGL): pool regions adopted from the expert tensors' Metal
    // storage (non-owning). When pool_ext[layer][kind] != nullptr it takes precedence over pool.
    std::vector<std::vector<const uint8_t *>> pool_ext;       // [layer][kind] base
    std::vector<std::vector<size_t>>          pool_ext_stride; // [layer][kind] bytes per slot
    std::vector<std::vector<uint32_t>>        pool_ext_slots;  // [layer][kind] capacity
    // prefill hot prewarm (LLAMA_EXPERT_CACHE_PREWARM_HOT=1): per-layer expert route frequency
    // accumulated during prefill; the first decode step fills the pool with the top-K hot set
    // (instead of the loader's experts-0..n prewarm, which ignores actual routing).
    std::vector<std::vector<uint64_t>> freq;      // [layer][expert] prefill route counts
    bool hot_prewarm_done = false;                // prewarm_hot runs once, before the 1st decode
    std::vector<std::vector<int32_t>> slot_owner;        // [layer][slot] = expert (-1 free)
    std::vector<std::vector<uint64_t>> slot_last_use;    // [layer][slot]
    std::vector<std::vector<uint8_t>>  slot_queued;      // [layer][slot] 1 = prefetch queued to bg, fill not started yet
    std::vector<std::vector<uint8_t>>  slot_loading;     // [layer][slot] 1 = bg thread filling (prefetch in flight)
    std::vector<std::vector<uint8_t>>  slot_pinned;      // [layer][slot] 1 = LRU-exempt (decode tail-union prewarm, TAILPIN)
    std::vector<std::vector<uint8_t>>  slot_pinned_static; // [layer][slot] 1 = LRU-exempt static profile pin (LLAMA_EXPERT_CACHE_PIN_PROFILE, never unpinned)
    // [CGC prefetch v2] recently-evicted experts per layer (ring). When pick_slot evicts an
    // expert to make room, that expert is a prime "will-be-needed-again" candidate (miss
    // analysis: 1062 misses across 126 steps = only 80 distinct experts, 72 of them repeated —
    // i.e. recurring hot experts that LRU evicted between uses, then miss again). The B-section
    // prefetch re-residents these so the next ensure is a HIT instead of a synchronous pread.
    std::vector<std::vector<uint32_t>> evicted_recent;   // [layer] ring of recently-evicted experts (newest at back)
    std::vector<uint32_t> evicted_ring_size;             // [layer] per-layer ring capacity
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
    std::deque<std::tuple<uint32_t, int32_t, uint32_t>> pool_queue; // (layer, slot, expert) queued pool fills (FIFO)
    // [CGC MTP fast path] reserved ZERO-slot: 1 when the layer's last slot region has been
    // zeroed (guarded by m). Only touched when CGC_VERIFY_DECODE / CGC_DRAFT_DECODE is set.
    std::vector<uint8_t> zero_slot_done; // [layer] 1 = reserved slot zeroed once

    // telemetry
    size_t n_requests = 0;
    size_t n_hits     = 0;
    size_t n_misses   = 0;
    size_t n_map_requests = 0;  // L3-B ensure() path (prefill / multi-token)
    size_t n_map_hits     = 0;
    // [CGC §8.99-2] loader prewarm fills (llama_model_loader: experts 0..n at load) are kept
    // out of the runtime hit-rate counters — a prewarm cold fill is NOT a runtime miss, and
    // counting it as one understated every published hit rate by ~15pt (81.3% vs real 96%).
    size_t n_prewarm_requests = 0;
    size_t n_prewarm_hits     = 0;
    size_t n_prewarm_misses   = 0;
    size_t n_prefetch = 0;          // pool prefetches queued to the bg thread
    size_t n_prefetch_dropped = 0;  // skipped (queue full / no free slot / already resident)
    std::atomic<size_t> n_reads{0};
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
void llama_expert_cache_ensure_batch(llama_expert_cache * cache, uint32_t layer,
                                     const uint32_t * experts, size_t n);
// Prefill hot prewarm: accumulate (layer, expert) route frequencies (prefill only; repeated
// across tokens counts multiple times). Then llama_expert_cache_prewarm_hot fills the pool with
// each layer's top-K most-routed experts at the first decode step. Returns 0 when skipped.
void llama_expert_cache_record_routes(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n);
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
// L4 cold-start (2026-08-15): the loader pre-reads the FIRST n_slots experts of every layer into
// the adopted Metal pool regions (the expert tensor buffers). Mark those slots resident so the
// first accesses are pool HITS instead of re-preading the same bytes — otherwise every short
// prompt pays a cold refill window (measured: L4@4GiB short-prompt decode 122ms vs base 88ms).
void llama_expert_cache_prepopulate(llama_expert_cache * cache, uint32_t layer, uint32_t n_slots);
// True when the static per-layer pool is active (LLAMA_EXPERT_CACHE_POOL=1 at init).
bool llama_expert_cache_pool_active(const llama_expert_cache * cache);
// Per-layer slot table: expert id -> slot index (-1 = not resident). nullptr when pool inactive.
const int32_t * llama_expert_cache_slot_table(const llama_expert_cache * cache, uint32_t layer);
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
void llama_expert_cache_touch(llama_expert_cache * cache, uint32_t layer,
                              const uint32_t * experts, size_t n);
// Slot-table lookup mapping -1 (not resident) to the ZERO slot. Same value as the raw slot
// table for resident experts, so using it on the exact-load path is a no-op (bit-identical).
int32_t llama_expert_cache_slot_table_safe(const llama_expert_cache * cache, uint32_t layer,
                                           uint32_t expert);
