#include "llama-expert-cache.h"

#include "llama-model.h" // llama_model::expert_cache_path / expert_index

#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/uio.h> // struct iovec / preadv (merge-read jobs)

#ifdef __APPLE__
#include <pthread.h>
#include <sys/fcntl.h> // F_RDADVISE / struct radvisory (Darwin read-ahead advisory)
#endif

static uint64_t make_key(uint32_t layer, uint32_t expert) {
    return ((uint64_t) layer << 32) | expert;
}


// One pread job: read seg.bytes from (file_idx, file_offset) into dst. Accumulates wall time
// into cache->pread_usec and counts one file read. Thread-safe: each job writes a DISTINCT dst,
// and the telemetry counters are atomic. Result via *ok (1 = read in full).
static void fill_job(llama_expert_cache * cache, const llama_expert_cache::segment * seg,
                     uint8_t * dst, int * ok) {
    FILE * f = cache->files.at(seg->file_idx);
    const auto t0 = std::chrono::steady_clock::now();
    const ssize_t rd = pread(fileno(f), dst, seg->bytes, (off_t) seg->file_offset);
    if (getenv("LLAMA_EXPERT_CACHE_PREAD_DBG") != nullptr && rd != (ssize_t) seg->bytes) {
        struct stat st;
        fstat(fileno(f), &st);
        fprintf(stderr, "PREADDBG off=%llu want=%u got=%zd errno=%d fsize=%lld\n",
                (unsigned long long) seg->file_offset, seg->bytes, rd, errno, (long long) st.st_size);
    }
    const auto t1 = std::chrono::steady_clock::now();
    cache->pread_usec.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    cache->n_reads.fetch_add(1, std::memory_order_relaxed);
    *ok = rd == (ssize_t) seg->bytes;
}

// Issue all segments of one (layer, expert) concurrently — one thread per segment, joined before
// return. The segments are SCATTERED across the file (gate/up/down tensors are separate GGUF
// tensors), so a single readv/preadv (contiguous ranges only) cannot express them; measured on
// M4: serial ~204us cold / 90us warm vs threads ~163/84 (SSD queue depth limits the overlap,
// spawn overhead eats most of the rest). Synchronous: the hook joins before returning, so the
// pool/blob is stable before the Metal copy (no async-write race). LLAMA_EXPERT_CACHE_SERIAL_FILL
// forces the old serial loop for A/B. Returns false on any short read (caller zeroes).
static bool fill_segments_concurrent(llama_expert_cache * cache,
                                     const std::vector<llama_expert_cache::segment> & segs,
                                     const std::vector<uint8_t *> & dsts) {
    const size_t n = segs.size();
    if (n == 0) {
        return true;
    }
    static const bool serial_fill = getenv("LLAMA_EXPERT_CACHE_SERIAL_FILL") != nullptr;
    const auto tb0 = std::chrono::steady_clock::now();
    std::vector<int> ok(n, 0);
    if (serial_fill || n == 1) {
        for (size_t i = 0; i < n; ++i) {
            fill_job(cache, &segs[i], dsts[i], &ok[i]);
        }
    } else {
        std::vector<std::thread> ths;
        ths.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            ths.emplace_back(fill_job, cache, &segs[i], dsts[i], &ok[i]);
        }
        for (auto & t : ths) {
            t.join();
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!ok[i]) {
            return false;
        }
    }
    const auto tb1 = std::chrono::steady_clock::now();
    cache->fill_batch_usec.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tb1 - tb0).count());
    return true;
}

// Fills a slot's blob from the file(s). Must be called WITHOUT holding cache->m
// (pread may block on IO). Returns the blob size filled.
// (defined after bg_loop; batch uses it before that point)
static void fill_segments_pool(llama_expert_cache * cache,
                               const std::vector<llama_expert_cache::segment> & segs,
                               const std::vector<uint8_t *> & dsts,
                               std::vector<int> & ok);
static size_t fill_slot(llama_expert_cache * cache, llama_expert_cache::slot * s) {
    const uint64_t key = s->key;
    const auto & positions = cache->key_segs.at(key);

    std::vector<llama_expert_cache::segment> segs;
    size_t total = 0;
    for (uint32_t pos : positions) {
        const auto & e = cache->index[pos];
        llama_expert_cache::segment seg;
        seg.kind        = e.kind;
        seg.file_idx    = e.file_idx;
        seg.file_offset = e.file_offset;
        seg.off         = (uint32_t) total;
        seg.bytes       = (uint32_t) e.bytes;
        total += e.bytes;
        segs.push_back(seg);
    }
    s->segs = std::move(segs);
    s->blob.assign(total, 0);

    std::vector<uint8_t *> dsts;
    dsts.reserve(s->segs.size());
    for (const auto & seg : s->segs) {
        dsts.push_back(s->blob.data() + seg.off);
    }
    if (!fill_segments_concurrent(cache, s->segs, dsts)) {
        fprintf(stderr, "llama_expert_cache: short read for key=%llu — zeroing blob\n",
                (unsigned long long) key);
        s->blob.assign(total, 0);
    }
    return total;
}

// Must be called holding cache->m. Evicts LRU slots (not loading/queued) until total_bytes + need <= budget.
static void evict_lru(llama_expert_cache * cache, size_t need) {
    while (cache->total_bytes + need > cache->budget) {
        uint64_t best_key  = 0;
        uint64_t best_tick = UINT64_MAX;
        bool found = false;
        for (const auto & kv : cache->map) {
            const auto & s = *kv.second;
            if (s.loading || s.queued) {
                continue;
            }
            if (s.last_use < best_tick) {
                best_tick = s.last_use;
                best_key  = kv.first;
                found = true;
            }
        }
        if (!found) {
            break; // cannot evict anything; allow the over-budget slot (single blob > budget)
        }
        auto it = cache->map.find(best_key);
        cache->total_bytes -= it->second->blob.size();
        cache->map.erase(it);
    }
}

// [CGC §8.101 A/B] per-layer slot count: n_slots_l[layer] when LAYER_CAPS is set, else the
// uniform n_slots. Every slot loop must iterate slots_l, never n_slots, so a per-layer cap
// cannot walk past a layer's shorter slot vectors.
static uint32_t slots_l(const llama_expert_cache * cache, uint32_t layer) {
    if (!cache->n_slots_l.empty() && layer < cache->n_slots_l.size()) {
        return cache->n_slots_l[layer];
    }
    return cache->n_slots;
}

// defined below; forward-declared so llama_expert_cache_zero_reserved_slot can zero the
// reserved slot's pool region (the ZERO-slot fast path runs before pool_region's definition).
static inline const uint8_t * pool_region(const llama_expert_cache * cache, uint32_t layer,
        int kind, size_t * stride_out, uint32_t * slots_out);

// [CGC MTP fast path] ZERO-slot. Only enabled when a decode fast-path env is set (MTP
// verify/draft). Base/non-MTP runs never set them, so zero_slot_enabled() == false there and
// every helper below degrades to the exact original behavior (byte-identical).
static bool zero_slot_enabled() {
    return getenv("CGC_VERIFY_DECODE") != nullptr || getenv("CGC_DRAFT_DECODE") != nullptr;
}

bool llama_expert_cache_zero_slot_enabled(const llama_expert_cache * cache) {
    (void) cache;
    return zero_slot_enabled();
}

int32_t llama_expert_cache_zero_slot(const llama_expert_cache * cache, uint32_t layer) {
    if (cache == nullptr || !zero_slot_enabled() || layer >= cache->slot_owner.size()) {
        return -1;
    }
    return (int32_t) slots_l(cache, layer) - 1;
}

uint32_t llama_expert_cache_usable_slots(const llama_expert_cache * cache, uint32_t layer) {
    if (cache == nullptr) {
        return 0;
    }
    const uint32_t ns = slots_l(cache, layer);
    return zero_slot_enabled() && ns > 1 ? ns - 1 : ns;
}

void llama_expert_cache_zero_reserved_slot(llama_expert_cache * cache, uint32_t layer) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || !zero_slot_enabled()) {
        return;
    }
    if (cache->zero_slot_done.size() <= layer) {
        cache->zero_slot_done.resize(layer + 1, 0);
    }
    if (cache->zero_slot_done[layer]) {
        return;
    }
    const int32_t zs = llama_expert_cache_zero_slot(cache, layer);
    if (zs < 0) {
        return;
    }
    cache->zero_slot_done[layer] = 1;
    for (int kind = 0; kind < 4; ++kind) {
        size_t stride = 0;
        uint32_t slots = 0;
        const uint8_t * region = pool_region(cache, layer, kind, &stride, &slots);
        if (region == nullptr || stride == 0 || (uint32_t) zs >= slots) {
            continue;
        }
        memset((void *) (region + (size_t) zs * stride), 0, stride);
    }
}

void llama_expert_cache_touch(llama_expert_cache * cache, uint32_t layer,
                              const uint32_t * experts, size_t n, bool draft_path) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || n == 0) {
        return;
    }
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::lock_guard<std::mutex> lk(cache->m);
    // [CGC MTP fast-path telemetry] count union members vs cold (ZERO-mapped) BEFORE the LRU
    // loop. Only when the fast path is active (base runs never call touch / stay identical).
    if (zero_slot_enabled()) {
        size_t cold = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t e = experts[i];
            if (e < cache->n_expert && table[e] < 0) {
                cold++;
            }
        }
        cache->n_fast_calls++;
        cache->n_fast_union += n;
        cache->n_fast_cold  += cold;
        if (draft_path) {
            cache->n_fast_draft_calls++;
            cache->n_fast_draft_union += n;
            cache->n_fast_draft_cold  += cold;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = experts[i];
        if (e >= cache->n_expert) {
            continue;
        }
        const int32_t slot = table[e];
        if (slot >= 0) {
            cache->slot_last_use[layer][slot] = ++cache->tick;
        }
    }
}

int32_t llama_expert_cache_slot_table_safe(const llama_expert_cache * cache, uint32_t layer,
                                           uint32_t expert) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || expert >= cache->n_expert) {
        return -1;
    }
    const int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    const int32_t slot = table[expert];
    if (slot >= 0) {
        return slot;
    }
    // not resident: map to the ZERO slot (finite zero contribution) when it is reserved,
    // else fall back to the raw -1 (exact-load path never hits this).
    return llama_expert_cache_zero_slot(cache, layer);
}

// Finds a free slot in the layer, else evicts the LRU slot. Never evicts a slot whose prefetch
// fill is in flight (slot_loading) — the bg thread would otherwise write the old expert's bytes
// into a slot already reassigned to a new expert (silent corruption). Returns -1 when every slot
// is loading (caller waits for a fill to finish). Must be called holding cache->m.
// [CGC WIN_PIN fix] min_tick: slots whose last_use >= min_tick were assigned EARLIER IN THE
// CALLER'S BATCH and must never be evicted mid-batch — with WIN_PIN pinning the previous
// batches' residents, the LRU scan would otherwise see this batch's own fresh assignments as
// the "stalest non-pinned" and hand the SAME slot to two experts (concurrent fills overwrite
// each other → corrupted FFN + near-hang). 0 = no restriction (single-expert ensure path).
// Overflow pass: when every non-pinned candidate is exhausted, evict the LRU PINNED slot
// instead of returning -1 / deadlocking (a pin is a preference, not a hard guarantee).
static int32_t pick_slot(llama_expert_cache * cache, uint32_t layer, uint64_t min_tick = 0) {
    auto & owner  = cache->slot_owner[layer];
    auto & last   = cache->slot_last_use[layer];
    auto & load   = cache->slot_loading[layer];
    auto & queued = cache->slot_queued[layer];
    // [CGC MTP fast path] the reserved ZERO slot is never handed to a real expert.
    const uint32_t ns = llama_expert_cache_usable_slots(cache, layer);
    for (uint32_t i = 0; i < ns; ++i) {
        if (owner[i] < 0 && !load[i] && !queued[i]) {
            return (int32_t) i;
        }
    }
    // all occupied (or loading/queued): evict LRU. Pass 0 skips pinned; pass 1 (overflow)
    // allows evicting a DYNAMICALLY pinned slot (slot_pinned — window/tail pins are a
    // preference) when nothing else is available. slot_pinned_static (PIN_PROFILE) stays a
    // hard guarantee in both passes, matching the pre-WIN_PIN behavior. Neither pass touches
    // this-batch assignments (min_tick) or in-flight fills.
    for (int pass = 0; pass < 2; ++pass) {
        uint64_t best_tick = UINT64_MAX;
        int32_t  best_slot = -1;
        for (uint32_t i = 0; i < ns; ++i) {
            if (load[i] || queued[i] || cache->slot_pinned_static[layer][i]) {
                continue;
            }
            if (min_tick != 0 && last[i] >= min_tick) {
                continue; // assigned earlier in the caller's batch
            }
            if (pass == 0 && cache->slot_pinned[layer][i]) {
                continue;
            }
            if (last[i] < best_tick) {
                best_tick = last[i];
                best_slot = (int32_t) i;
            }
        }
        if (best_slot >= 0) {
            const int32_t evicted = owner[best_slot];
            if (evicted >= 0 && evicted < (int32_t) cache->n_expert) {
                cache->slot_table[(size_t) layer * cache->n_expert + evicted] = -1;
            }
            owner[best_slot] = -1;
            return best_slot;
        }
    }
    return -1;
}

// L3 Option A: pread one (layer, expert) straight into its pool slot regions (per kind).
// Must be called WITHOUT holding cache->m. On short read the slot is zeroed (detectable, never
// silently stale). Pool must be active and the slot already assigned.
// Pool region base + stride + slots for (layer, kind): L4 adopted Metal regions first, then
// the malloc'd pool. Returns nullptr when the kind has no region.
static inline const uint8_t * pool_region(const llama_expert_cache * cache, uint32_t layer, int kind,
        size_t * stride_out, uint32_t * slots_out) {
    if (layer < cache->pool_ext.size() && cache->pool_ext[layer][kind] != nullptr) {
        if (stride_out) *stride_out = cache->pool_ext_stride[layer][kind];
        if (slots_out)  *slots_out  = cache->pool_ext_slots[layer][kind];
        return cache->pool_ext[layer][kind];
    }
    if (layer < cache->pool.size() && !cache->pool[layer][kind].empty()) {
        if (stride_out) *stride_out = cache->pool[layer][kind].size() / slots_l(cache, layer);
        if (slots_out)  *slots_out  = slots_l(cache, layer);
        return cache->pool[layer][kind].data();
    }
    return nullptr;
}

// Build the (segment, dst) list that fill_pool_direct would pread for one (layer, expert) into
// its pool slot — shared by the spawn fill (fill_pool_direct) and the persistent-pool batch.
static void fill_pool_direct_collect(llama_expert_cache * cache, uint32_t layer, int32_t slot_idx,
                                     uint32_t expert, std::vector<llama_expert_cache::segment> & segs,
                                     std::vector<uint8_t *> & dsts) {
    const uint64_t key = make_key(layer, expert);
    const auto & positions = cache->key_segs.at(key);
    for (uint32_t pos : positions) {
        const auto & e = cache->index[pos];
        if (e.kind < 0 || e.kind >= 4 || layer >= cache->pool.size()) {
            continue;
        }
        size_t stride = 0;
        uint32_t slots = 0;
        const uint8_t * region = pool_region(cache, layer, e.kind, &stride, &slots);
        if (region == nullptr || slots == 0 || slot_idx < 0 || (uint32_t) slot_idx >= slots) {
            continue;
        }
        llama_expert_cache::segment seg;
        seg.kind        = e.kind;
        seg.file_idx    = e.file_idx;
        seg.file_offset = e.file_offset;
        seg.off         = 0;
        seg.bytes       = (uint32_t) e.bytes;
        dsts.push_back((uint8_t *) region + (size_t) slot_idx * stride);
        segs.push_back(seg);
    }
}

static void fill_pool_direct(llama_expert_cache * cache, uint32_t layer, int32_t slot_idx, uint32_t expert) {
    std::vector<llama_expert_cache::segment> segs;
    std::vector<uint8_t *> dsts;
    fill_pool_direct_collect(cache, layer, slot_idx, expert, segs, dsts);
    if (segs.empty()) {
        return;
    }
    const uint64_t key = make_key(layer, expert);
    if (!fill_segments_concurrent(cache, segs, dsts)) {
        fprintf(stderr, "llama_expert_cache: fill_pool_direct short read key=%llu — zeroing slot\n",
                (unsigned long long) key);
        for (size_t i = 0; i < segs.size(); ++i) {
            memset(dsts[i], 0, segs[i].bytes);
        }
    }
}

int32_t llama_expert_cache_ensure_slot(llama_expert_cache * cache, uint32_t layer, uint32_t expert,
                                  bool count) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || expert >= cache->n_expert) {
        return -1;
    }
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::unique_lock<std::mutex> lk(cache->m);
    (count ? cache->n_requests : cache->n_prewarm_requests)++;
    if (table[expert] >= 0) {
        const int32_t slot = table[expert];
        if (cache->slot_loading[layer][slot] || cache->slot_queued[layer][slot]) {
            // prefetch queued/in flight: wait for the bg thread to finish the fill (it clears
            // the flags and notifies). The pread is hidden behind the current layer's FFN.
            cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
        }
        (count ? cache->n_hits : cache->n_prewarm_hits)++;
        cache->slot_last_use[layer][slot] = ++cache->tick;
        return slot;
    }
    (count ? cache->n_misses : cache->n_prewarm_misses)++;

    // miss: find the slot (free or evict LRU), fill it from the file synchronously.
    // Option A: the pool IS the storage — pread straight into the slot's regions (no
    // map/blob intermediate, no budget accounting; the pool is statically bounded).
    int32_t slot;
    for (;;) {
        slot = pick_slot(cache, layer);
        if (slot >= 0) {
            break;
        }
        cache->bg_cv.wait(lk); // all slots loading: wait for a fill to finish, then retry
    }
    cache->slot_owner[layer][slot] = (int32_t) expert;
    lk.unlock();

    fill_pool_direct(cache, layer, slot, expert);

    lk.lock();
    cache->slot_last_use[layer][slot] = ++cache->tick;
    table[expert] = slot;
    return slot;
}

// L3 Option A batch: cross-expert parallel fill (see header). One lock for the whole batch
// (slot assignment is therefore atomic across the layer's experts — no interleaving between
// them like the serial loop), then one thread per missed expert, joined before return so the
// pool is stable before the FFN dispatches (same synchronous guarantee as ensure_slot).
void llama_expert_cache_ensure_batch(llama_expert_cache * cache, uint32_t layer,
                                     const uint32_t * experts, size_t n) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || n == 0) {
        return;
    }
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::vector<int32_t>  slots(n, -1);
    std::vector<uint32_t> miss_exps;   // (expert) to fill concurrently
    std::vector<int32_t>  miss_slots;  // slot assigned to each miss
    {
        std::unique_lock<std::mutex> lk(cache->m);
        cache->n_requests += n;
        // [CGC WIN_PIN fix] batch_tick: taken ONCE before the per-expert loop — every
        // last_use bump done at/after this tick (hits re-touched + misses assigned) belongs
        // to THIS batch, so pick_slot's LRU must never evict them mid-batch. Taking it inside
        // the loop would re-raise the floor each expert and miss the earlier assignments.
        const uint64_t batch_tick = cache->tick + 1;
        if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
            unsigned busy0 = 0;
            for (uint32_t i = 0; i < slots_l(cache, layer); ++i) if (cache->slot_owner[layer][i] >= 0) busy0++;
            fprintf(stderr, "PFDBG batch l=%u n=%zu busy_before=%u\n", layer, n, busy0);
        }
        for (size_t i = 0; i < n; ++i) {
            const uint32_t e = experts[i];
            if (e >= cache->n_expert) {
                continue;
            }
            if (table[e] >= 0) {
                const int32_t slot = table[e];
                if (cache->slot_loading[layer][slot] || cache->slot_queued[layer][slot]) {
                    // prefetch queued/in flight: wait for the bg thread (same as ensure_slot)
                    cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
                }
                cache->n_hits++;
                cache->slot_last_use[layer][slot] = ++cache->tick;
                slots[i] = slot;
                continue;
            }
            cache->n_misses++;
            if (getenv("LLAMA_EXPERT_CACHE_MISS_DUMP") != nullptr) {
                static FILE * fmiss = nullptr;
                static const char * miss_path = getenv("LLAMA_EXPERT_CACHE_MISS_DUMP");
                if (fmiss == nullptr) {
                    fmiss = fopen(miss_path, "a");
                }
                if (fmiss != nullptr) {
                    fprintf(fmiss, "%u %u\n", layer, e);
                    fflush(fmiss);
                }
            }
            int32_t slot;
            for (;;) {
                slot = pick_slot(cache, layer, batch_tick);
                if (slot >= 0) {
                    break;
                }
                cache->bg_cv.wait(lk); // all slots loading: wait for a fill to finish
            }
            cache->slot_owner[layer][slot] = (int32_t) e;
            // Bump last_use NOW (not after the fill like ensure_slot): the batch assigns all
            // slots under one lock, and the next pick_slot's LRU eviction would otherwise see
            // this slot as the stalest and hand it out AGAIN (two experts on one slot ->
            // concurrent fills overwrite the same region -> nondeterministic FFN garbage).
            // ensure_slot gets away without this because its next pick_slot happens only after
            // the fill + last_use bump (serial per expert).
            cache->slot_last_use[layer][slot] = ++cache->tick;
            slots[i] = slot;
            miss_exps.push_back(e);
            miss_slots.push_back(slot);
        }
    }
    if (getenv("LLAMA_EXPERT_CACHE_BATCH_DBG") != nullptr && !miss_exps.empty()) {
        fprintf(stderr, "BATCHDBG layer=%u misses=%zu slots:", layer, miss_exps.size());
        for (size_t i = 0; i < miss_exps.size(); ++i) {
            fprintf(stderr, " e%u->s%d", miss_exps[i], miss_slots[i]);
        }
        fprintf(stderr, "\n");
    }
    // Fill across the layer's misses. Default: the persistent worker pool (LLAMA_EXPERT_CACHE_
    // WORKERS=N, default 8) — flatten ALL of the layer's misses into one job list (each fill is
    // itself 3 segments), submit, wait for outstanding == 0. No lock held: pread may block on
    // IO. LLAMA_EXPERT_CACHE_BATCH_SPAWN=1 reverts to the old spawn-per-expert + join (A/B).
    if (!miss_exps.empty()) {
        static const bool spawn_fill = getenv("LLAMA_EXPERT_CACHE_BATCH_SPAWN") != nullptr;
        if (spawn_fill) {
            std::vector<std::thread> ths;
            ths.reserve(miss_exps.size());
            for (size_t i = 0; i < miss_exps.size(); ++i) {
                ths.emplace_back(fill_pool_direct, cache, layer, miss_slots[i], miss_exps[i]);
            }
            for (auto & t : ths) {
                t.join();
            }
        } else {
            std::vector<llama_expert_cache::segment> all_segs;
            std::vector<uint8_t *> all_dsts;
            all_segs.reserve(miss_exps.size() * 3);
            all_dsts.reserve(miss_exps.size() * 3);
            for (size_t i = 0; i < miss_exps.size(); ++i) {
                fill_pool_direct_collect(cache, layer, miss_slots[i], miss_exps[i],
                                         all_segs, all_dsts);
            }
            std::vector<int> ok;
            fill_segments_pool(cache, all_segs, all_dsts, ok);
            bool bad = false;
            for (size_t i = 0; i < ok.size(); ++i) {
                if (!ok[i]) {
                    memset(all_dsts[i], 0, all_segs[i].bytes);
                    bad = true;
                }
            }
            if (bad) {
                fprintf(stderr, "llama_expert_cache: pool short read layer=%u — zeroed %zu segment(s)\n",
                        layer, (size_t) std::count(ok.begin(), ok.end(), 0));
            }
        }
        std::unique_lock<std::mutex> lk(cache->m);
        for (size_t i = 0; i < miss_exps.size(); ++i) {
            const uint32_t e = miss_exps[i];
            const int32_t  s = miss_slots[i];
            cache->slot_last_use[layer][s] = ++cache->tick;
            table[e] = s;
        }
        // [CGC WIN_PIN] roll this batch's union into the layer's window and repin the LRU-exempt
        // set. Runs AFTER table[] is updated so this step's union (hits + just-filled misses) is
        // fully resident and gets pinned. pick_slot skips pinned slots, so evictions now only
        // touch cold/non-window experts — the recurring hot experts stay resident and the next
        // step's ensure HITS instead of paying a synchronous pread. LLAMA_EXPERT_CACHE_WIN_PIN=K
        // (default 0 = off = old pure-LRU, bit-identical). Recycles slot_pinned (TAILPIN's field;
        // decode-time repin supersedes the one-shot prefill tail pin by design). Note: the roll
        // happens on miss-steps only (inside this if) — a hits-only step keeps the window as-is,
        // which stretches its TIME span across clean steps: recurring experts reused every N
        // steps stay covered even when N exceeds K.
        {
            static const int win_pin_k = []() {
                const char * s = getenv("LLAMA_EXPERT_CACHE_WIN_PIN");
                return (s != nullptr && s[0] != '\0') ? atoi(s) : 0;
            }();
            if (win_pin_k > 0 && layer < cache->win_union.size()) {
                auto & dq = cache->win_union[layer];
                dq.push_back(std::vector<uint32_t>(experts, experts + n));
                while ((int) dq.size() > win_pin_k) {
                    dq.pop_front();
                }
                const uint32_t ns = slots_l(cache, layer);
                auto & pin = cache->slot_pinned[layer];
                std::fill(pin.begin(), pin.end(), 0);
                for (const auto & u : dq) {
                    for (uint32_t e : u) {
                        if (e >= cache->n_expert) {
                            continue;
                        }
                        const int32_t s = table[e];
                        if (s >= 0 && (uint32_t) s < ns) {
                            pin[s] = 1;
                        }
                    }
                }
            }
        }
    }
}

// L3 Option A: stabilize a layer's pool region before its FFN dispatches (see header).
// With the page-cache-warm prefetch design the pool is ONLY written by ensure_slot on the hook
// thread (serialized, always completes before the FFN dispatches), so no draining is needed.
// Kept as a no-op safety gate in case a future prefetch variant writes the pool directly.
// L3 Option A double-buffer: stabilize a layer's pool region before its FFN dispatches. The
// Metal backend's region copy is region-wide (all n_slots) and the kernel reads slots through
// the remap, so ANY in-flight bg fill for THIS layer would tear it. Drop the layer's still-
// queued fills (the union ensure below already waited on whatever it needs; a dropped fill
// just re-fills synchronously on the next ensure) and wait for in-flight ones to complete.
// Layers other than `layer` are untouched — their bg fills keep hiding behind the FFN window.
void llama_expert_cache_drain_layer(llama_expert_cache * cache, uint32_t layer) {
    if (cache == nullptr || layer >= cache->slot_queued.size()) {
        return;
    }
    std::unique_lock<std::mutex> lk(cache->m);
    for (uint32_t slot = 0; slot < slots_l(cache, layer); ++slot) {
        if (cache->slot_queued[layer][slot]) {
            cache->slot_queued[layer][slot] = 0;
            cache->slot_owner[layer][slot]  = -1; // free for the sync fill path
            cache->n_prefetch_dropped++;
        }
    }
    // wait for in-flight bg fills on this layer (slot_loading) to land
    cache->bg_cv.wait(lk, [&]{
        for (uint32_t slot = 0; slot < slots_l(cache, layer); ++slot) {
            if (cache->slot_loading[layer][slot]) {
                return false;
            }
        }
        return true;
    });
}

// L3 Option A double-buffer: queue (layer, expert) for the bg thread to fill into a FREE pool
// slot. The hook calls this during layer il's hook — the bg fill then runs behind the layer-il
// FFN + attention-il+1 GPU window, so the NEXT layer's ensure finds the bytes already resident
// (a hit) instead of paying the synchronous pread. Free-slot-only: never evicts for a prediction
// (pick_slot is not consulted), so a misprediction costs one pread of queue work, never a
// resident eviction. slot_queued marks the slot so pick_slot / ensure_* treat it as busy, and
// the bg loop re-validates owner under the lock before writing. Non-blocking; returns 0 if
// queued, -1 if skipped (pool inactive / already resident or queued / no free slot).
int32_t llama_expert_cache_prefetch_slot(llama_expert_cache * cache, uint32_t layer, uint32_t expert) {
    if (cache == nullptr || !cache->pool_active || layer >= cache->slot_queued.size() ||
            expert >= cache->n_expert || cache->key_segs.find(make_key(layer, expert)) == cache->key_segs.end()) {
        if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
            fprintf(stderr, "PFDBG guard-reject l=%u e=%u pool=%d layer_ok=%d expert_ok=%d key_ok=%d\n",
                    layer, expert, cache ? (int) cache->pool_active : -1,
                    cache ? (int) (layer < cache->slot_queued.size()) : -1,
                    cache ? (int) (expert < cache->n_expert) : -1,
                    cache ? (int) (cache->key_segs.find(make_key(layer, expert)) != cache->key_segs.end()) : -1);
        }
        return -1;
    }
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::unique_lock<std::mutex> lk(cache->m);
    if (table[expert] >= 0) {
        if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
            fprintf(stderr, "PFDBG drop: resident l=%u e=%u\n", layer, expert);
        }
        return -1; // already resident: nothing to prefetch
    }
    int32_t slot = -1;
    // [CGC MTP fast path] skip the reserved ZERO slot (never assigned to a real expert).
    const uint32_t ns = llama_expert_cache_usable_slots(cache, layer);
    for (uint32_t i = 0; i < ns; ++i) {
        if (cache->slot_owner[layer][i] < 0 && !cache->slot_queued[layer][i] && !cache->slot_loading[layer][i]) {
            slot = (int32_t) i;
            break;
        }
    }
    if (slot < 0) {
        // No free slot: evict the layer's global LRU slot (the union ensure just bumped every
        // member of the CURRENT step's union, so the LRU is by construction a non-union slot —
        // evicting it for a prediction cannot turn a current-step hit into a miss). Queued and
        // in-flight slots are never touched. This covers both load-time prewarmed slots (never
        // re-touched, stalest) and decode-time leftovers.
        uint64_t best_tick = UINT64_MAX;
        for (uint32_t i = 0; i < ns; ++i) {
            if (cache->slot_queued[layer][i] || cache->slot_loading[layer][i]) {
                continue;
            }
            if (cache->slot_last_use[layer][i] < best_tick) {
                best_tick = cache->slot_last_use[layer][i];
                slot = (int32_t) i;
            }
        }
        if (slot >= 0) {
            const int32_t evicted = cache->slot_owner[layer][slot];
            if (evicted >= 0 && evicted < (int32_t) cache->n_expert) {
                cache->slot_table[(size_t) layer * cache->n_expert + evicted] = -1;
            }
            cache->slot_owner[layer][slot] = -1;
        } else {
            if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
                unsigned cold = 0, warm = 0;
                uint64_t min_use = UINT64_MAX;
                for (uint32_t i = 0; i < slots_l(cache, layer); ++i) {
                    if (cache->slot_queued[layer][i] || cache->slot_loading[layer][i]) continue;
                    if (cache->slot_last_use[layer][i] == 0) cold++;
                    else warm++;
                    min_use = std::min(min_use, cache->slot_last_use[layer][i]);
                }
                fprintf(stderr, "PFDBG drop: no-evictable l=%u e=%u cold=%u warm=%u min_use=%llu tick=%llu\n",
                        layer, expert, cold, warm, (unsigned long long) min_use, (unsigned long long) cache->tick);
            }
            cache->n_prefetch_dropped++;
            return -1;
        }
    }
    cache->slot_queued[layer][slot]  = 1;
    cache->slot_owner[layer][slot]   = (int32_t) expert;
    cache->pool_queue.emplace_back(layer, slot, (uint32_t) expert);
    cache->n_prefetch++;
    lk.unlock();
    cache->bg_cv.notify_one();
    return 0;
}

const uint8_t * llama_expert_cache_pool_data(const llama_expert_cache * cache, uint32_t layer, int kind) {
    if (cache == nullptr || kind < 0 || kind >= 4) {
        return nullptr;
    }
    size_t stride = 0;
    uint32_t slots = 0;
    return pool_region(cache, layer, kind, &stride, &slots);
}

size_t llama_expert_cache_pool_stride(const llama_expert_cache * cache, uint32_t layer, int kind) {
    if (cache == nullptr || kind < 0 || kind >= 4) {
        return 0;
    }
    size_t stride = 0;
    uint32_t slots = 0;
    pool_region(cache, layer, kind, &stride, &slots);
    return stride;
}

bool llama_expert_cache_adopt_pool_region(llama_expert_cache * cache, uint32_t layer, int kind,
        const uint8_t * base, int64_t n_slots, size_t stride) {
    if (cache == nullptr || layer >= cache->pool_ext.size() || kind < 0 || kind >= 4 ||
            base == nullptr || n_slots <= 0 || stride == 0) {
        return false;
    }
    cache->pool_ext[layer][kind]       = base;
    cache->pool_ext_stride[layer][kind] = stride;
    cache->pool_ext_slots[layer][kind]  = (uint32_t) n_slots;
    return true;
}

// Prefill hot prewarm: accumulate (layer, expert) route frequencies. Called from the hook for
// every prefill/multi-token batch (repetition across tokens counts multiple times, so the
// top-K reflects true routing frequency). Lock is brief (one increment per expert).
void llama_expert_cache_record_routes(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n) {
    if (cache == nullptr || layer >= cache->freq.size() || n == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    auto & f = cache->freq[layer];
    for (size_t i = 0; i < n; ++i) {
        if (experts[i] < cache->n_expert) {
            f[experts[i]]++;
        }
    }
}

// Prefill hot prewarm: at the first decode step, fill each layer's pool with its top-K most-
// routed prefill experts (instead of the loader's experts-0..n prewarm). The preads are the
// same cold-start cost as the loader prewarm, but land in the slots decode actually uses.
// Runs once (hot_prewarm_done); a no-op on later steps. Returns the number of experts ensured.
size_t llama_expert_cache_prewarm_hot(llama_expert_cache * cache) {
    if (cache == nullptr || !cache->pool_active || cache->freq.empty()) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(cache->m);
        if (cache->hot_prewarm_done) {
            return 0;
        }
        cache->hot_prewarm_done = true;
    }
    // per-layer top-K by route frequency (tie-break by expert id for determinism)
    std::vector<std::vector<uint32_t>> top(cache->freq.size());
    {
        std::lock_guard<std::mutex> lk(cache->m);
        for (uint32_t l = 0; l < cache->freq.size(); ++l) {
            std::vector<uint32_t> order(cache->n_expert);
            for (uint32_t i = 0; i < cache->n_expert; ++i) {
                order[i] = i;
            }
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b){
                const uint64_t fa = cache->freq[l][a], fb = cache->freq[l][b];
                return fa != fb ? fa > fb : a < b;
            });
            const size_t n = std::min<size_t>(slots_l(cache, l), cache->n_expert);
            top[l].assign(order.begin(), order.begin() + n);
        }
    }
    size_t warmed = 0;
    for (uint32_t l = 0; l < top.size(); ++l) {
        if (top[l].empty()) {
            continue; // no prefill routes for this layer
        }
        if (getenv("LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0") != nullptr && l == 0) {
            continue; // blk.0 is not pooled (full-weight CPU skip-load tensor)
        }
        for (uint32_t e : top[l]) {
            if (llama_expert_cache_ensure_slot(cache, l, e) >= 0) {
                warmed++;
            }
        }
    }
    if (getenv("LLAMA_EXPERT_CACHE_GATE_DBG") != nullptr) {
        fprintf(stderr, "PREWARMHOT warmed %zu slots from prefill hot set\n", warmed);
    }
    return warmed;
}

// Tail-union prewarm (TAILPIN): pin resident experts so the first decode step's LRU eviction
// cannot hand their slots out. No fill: non-resident experts are ignored (they are not in the
// pool — pinning cannot conjure them without the §8.15 falsified critical-path cold start).
size_t llama_expert_cache_pin_experts(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n) {
    if (cache == nullptr || layer >= cache->slot_pinned.size() || n == 0) {
        return 0;
    }
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::lock_guard<std::mutex> lk(cache->m);
    size_t pinned = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = experts[i];
        if (e >= cache->n_expert) {
            continue;
        }
        const int32_t slot = table[e];
        if (slot >= 0) {
            cache->slot_pinned[layer][slot] = 1;
            cache->slot_last_use[layer][slot] = ++cache->tick;
            pinned++;
        }
    }
    return pinned;
}

void llama_expert_cache_unpin_all(llama_expert_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    for (auto & lv : cache->slot_pinned) {
        std::fill(lv.begin(), lv.end(), 0);
    }
}

// Static profile pin (LLAMA_EXPERT_CACHE_PIN_PROFILE). Read per-layer top-N expert list:
// one line per layer, space-separated expert ids (line i = layer i). Missing/empty lines are
// skipped (no pins for that layer). Unknown ids are ignored. The list is stored for the
// load-time fill; this function does NOT fill or mark slots (that happens in the loader so the
// fill runs through the same bit-identical ensure_slot path at load time).
size_t llama_expert_cache_load_pin_profile(llama_expert_cache * cache, const char * path) {
    if (cache == nullptr || path == nullptr) {
        return 0;
    }
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        fprintf(stderr, "llama_expert_cache: PIN_PROFILE open failed: %s\n", path);
        return 0;
    }
    size_t total = 0;
    char line[4096];
    uint32_t layer = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (layer >= cache->n_expert) {
            break; // no more layers
        }
        char * save = nullptr;
        char * tok = strtok_r(line, " \t\r\n", &save);
        if (tok != nullptr) {
            // ensure the pin_profile vector has an entry for this layer
            if (cache->pin_profile.size() <= layer) {
                cache->pin_profile.resize(layer + 1);
            }
            while (tok != nullptr) {
                char * end = nullptr;
                const long v = strtol(tok, &end, 10);
                if (end != tok && v >= 0 && (uint32_t) v < cache->n_expert) {
                    cache->pin_profile[layer].push_back((uint32_t) v);
                    total++;
                }
                tok = strtok_r(nullptr, " \t\r\n", &save);
            }
        }
        layer++;
    }
    fclose(f);
    if (getenv("LLAMA_EXPERT_CACHE_GATE_DBG") != nullptr) {
        fprintf(stderr, "PINPROFILE: %zu experts across %zu layers\n", total, layer);
    }
    return total;
}

void llama_expert_cache_prepopulate(llama_expert_cache * cache, uint32_t layer, uint32_t n_slots) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || cache->n_expert == 0 || n_slots == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    // [CGC MTP fast path] never hand the reserved ZERO slot to a real expert.
    const uint32_t n = std::min(n_slots, llama_expert_cache_usable_slots(cache, layer));
    if (getenv("LLAMA_EXPERT_CACHE_PREPOP_DBG") != nullptr && layer == 0 && n > 2) {
        for (int k = 0; k < 4; ++k) {
            size_t stride = 0; uint32_t slots = 0;
            const uint8_t * region = pool_region(cache, 0, k, &stride, &slots);
            if (region == nullptr || stride == 0) { continue; }
            fprintf(stderr, "PREPOP layer0 kind%d stride=%zu: slot0[0..7]=%02x %02x %02x %02x %02x %02x %02x %02x | slot1[0..7]=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                k, stride, region[0],region[1],region[2],region[3],region[4],region[5],region[6],region[7],
                region[stride],region[stride+1],region[stride+2],region[stride+3],region[stride+4],region[stride+5],region[stride+6],region[stride+7]);
        }
    }
    for (uint32_t e = 0; e < n && e < cache->n_expert; ++e) {
        if (table[e] < 0 && cache->slot_owner[layer][e] < 0) {
            table[e] = (int32_t) e;             // slot e holds expert e (loader pre-read order)
            cache->slot_owner[layer][e] = (int32_t) e;
            cache->slot_last_use[layer][e] = 0; // oldest: first eviction candidate when full
        }
    }
}

uint32_t llama_expert_cache_slots_per_layer(const llama_expert_cache * cache) {
    return cache == nullptr ? 0 : cache->n_slots;
}

// [CGC §8.101 A/B] per-layer cap: the slot count actually usable for `layer`. Without the env
// this equals slots_per_layer (uniform n_slots).
uint32_t llama_expert_cache_slots_per_layer_l(const llama_expert_cache * cache, uint32_t layer) {
    return cache == nullptr ? 0 : slots_l(cache, layer);
}

// CGC L3 Option A: kernel slot-table registry (defined in ggml-cpu.c)
extern "C" void ggml_cpu_clear_mmid_slot_tables_all(void);

llama_expert_cache::~llama_expert_cache() {
    ggml_cpu_clear_mmid_slot_tables_all(); // pool buffers are about to die
    {
        std::lock_guard<std::mutex> lk(m);
        fprintf(stderr, "llama_expert_cache: final stats: runtime requests=%zu hits=%zu misses=%zu (hit rate %.1f%%)  prewarm req=%zu hit=%zu miss=%zu  resident=%.2f MiB file_reads=%zu pread_usec=%llu fill_batch_usec=%llu prefetch=%zu/%zu\n",
                n_requests, n_hits, n_misses,
                n_requests ? 100.0 * (double) n_hits / (double) n_requests : 0.0,
                n_prewarm_requests, n_prewarm_hits, n_prewarm_misses,
                total_bytes / 1024.0 / 1024.0, n_reads.load(std::memory_order_relaxed),
                (unsigned long long) pread_usec.load(std::memory_order_relaxed),
                (unsigned long long) fill_batch_usec.load(std::memory_order_relaxed),
                n_prefetch, n_prefetch_dropped);
        const size_t n_dec_req = n_requests - n_map_requests;
        const size_t n_dec_hit = n_hits - n_map_hits;
        fprintf(stderr, "llama_expert_cache: decode/pool (ensure_slot+batch) hits=%zu/%zu (%.1f%%)  gather (ensure) hits=%zu/%zu\n",
                n_dec_hit, n_dec_req, n_dec_req ? 100.0 * (double) n_dec_hit / (double) n_dec_req : 0.0,
                n_map_hits, n_map_requests);
        // [CGC MTP fast-path telemetry] the REAL steady decode miss rate: cold = read the ZERO
        // slot (weight contribution lost) on the touch+ZERO fast path (prefill/catch-up fills
        // excluded — those are the ensure_slot+batch line above). verify = ctx_tgt multi-token,
        // draft = ctx_dft 1-token. STEP_DBG timeline (LLAMA_EXPERT_CACHE_STEP_DBG) shows the
        // split is structural churn (~65% steady), not cold-start concentration.
        if (n_fast_calls > 0) {
            const size_t v_union = n_fast_union - n_fast_draft_union;
            const size_t v_cold  = n_fast_cold  - n_fast_draft_cold;
            const size_t v_calls = n_fast_calls - n_fast_draft_calls;
            fprintf(stderr, "llama_expert_cache: MTP fast path: calls=%zu union=%zu cold(ZERO)=%zu (%.1f%%)   verify: calls=%zu union=%zu cold=%zu (%.1f%%)   draft: calls=%zu union=%zu cold=%zu (%.1f%%)\n",
                    n_fast_calls, n_fast_union, n_fast_cold,
                    n_fast_union ? 100.0 * (double) n_fast_cold / (double) n_fast_union : 0.0,
                    v_calls, v_union, v_cold,
                    v_union ? 100.0 * (double) v_cold / (double) v_union : 0.0,
                    n_fast_draft_calls, n_fast_draft_union, n_fast_draft_cold,
                    n_fast_draft_union ? 100.0 * (double) n_fast_draft_cold / (double) n_fast_draft_union : 0.0);
        }
        bg_stop = true;
    }
    bg_cv.notify_all();
    if (bg.joinable()) {
        bg.join();
    }
    {
        std::lock_guard<std::mutex> lk(pool_m);
        pool_stop = true;
    }
    pool_cv.notify_all();
    for (auto & w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }
    for (FILE * f : files) {
        if (f) {
            fclose(f);
        }
    }
}

void llama_expert_cache::bg_loop() {
#ifdef __APPLE__
    // USER_INITIATED, not BACKGROUND: the bg fills serve the decode critical path (ensure_slot
    // waits on them); a background-priority thread gets starved during heavy compute and the
    // wait-on-prefetch becomes slower than a synchronous pread.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    for (;;) {
        uint64_t key = 0;
        bool has_pool = false;
        std::tuple<uint32_t, int32_t, uint32_t> pk; // (layer, slot, expert)
        {
            std::unique_lock<std::mutex> lk(m);
            while (bg_queue.empty() && pool_queue.empty() && !bg_stop) {
                bg_cv.wait(lk);
            }
            if (bg_stop && bg_queue.empty() && pool_queue.empty()) {
                return; // drain both queues before exiting (destructor path)
            }
            // Pool fills first, FIFO (layer 0's fill must land before layer 0's hook fires).
            if (!pool_queue.empty()) {
                pk = pool_queue.front();
                pool_queue.pop_front();
                has_pool = true;
            } else {
                key = bg_queue.back();
                bg_queue.pop_back();
            }
        }

        if (has_pool) {
            const uint32_t layer  = std::get<0>(pk);
            const int32_t  slot   = std::get<1>(pk);
            const uint32_t expert = std::get<2>(pk);
            {
                std::unique_lock<std::mutex> lk(m);
                // [CGC deadlock fix] prefetch_slot marks the slot with slot_queued (NOT
                // slot_loading — that flag is set here, just before the pread). The old check
                // `!slot_loading` was never true, so every queued prefetch was dropped as
                // "stale" while leaving slot_queued=1 + owner set: each prefetch leaked a
                // permanently-busy slot until pick_slot found nothing evictable and
                // ensure_batch waited on bg_cv forever (deadlock). Validate slot_queued.
                if (slot < 0 || slot >= (int32_t) slots_l(this, layer) || !slot_queued[layer][slot]) {
                    continue; // stale (dropped by drain_layer / cancelled defensively)
                }
                if (slot_owner[layer][slot] != (int32_t) expert) {
                    // slot reassigned by a synchronous fill racing the queue: drop the fill
                    slot_queued[layer][slot]  = 0;
                    slot_loading[layer][slot] = 0;
                    continue;
                }
                // mark in-flight so drain_layer waits for us (its predicate checks slot_loading)
                // and pick_slot treats the slot as protected while the bg thread writes the bytes
                slot_loading[layer][slot] = 1;
            }
            fill_pool_direct(this, layer, (uint32_t) slot, expert);
            std::unique_lock<std::mutex> lk(m);
            slot_loading[layer][slot] = 0;
            slot_queued[layer][slot]  = 0;
            slot_last_use[layer][slot] = ++tick; // fresh BEFORE the layer's hook consumes it
            // publish to the slot table so the next ensure for this expert is a HIT (the whole
            // point of double-buffer: the fill completed behind the previous layer's FFN window)
            if (expert < n_expert) {
                slot_table[(size_t) layer * n_expert + expert] = slot;
            }
            bg_cv.notify_all();
            continue;
        }

        slot * s = nullptr;
        {
            std::unique_lock<std::mutex> lk(m);
            auto it = map.find(key);
            if (it == map.end()) {
                continue; // evicted while queued (should not happen; defensive)
            }
            s = it->second.get();
        }

        const size_t filled = fill_slot(this, s);

        std::unique_lock<std::mutex> lk(m);
        s->loading = false;
        s->queued  = false;
        s->last_use = ++tick; // mark fresh BEFORE eviction so the just-filled slot survives
        total_bytes += filled;
        evict_lru(this, 0);
        s->cv.notify_all();
    }
}

void llama_expert_cache::pool_loop() {
#ifdef __APPLE__
    // USER_INITIATED, not BACKGROUND: the pool fills serve the decode critical path (the batch
    // waits on outstanding == 0 before the FFN dispatches); a background-priority worker gets
    // starved during heavy compute and the wait becomes slower than a synchronous pread.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    for (;;) {
        pread_job job;
        {
            std::unique_lock<std::mutex> lk(pool_m);
            pool_cv.wait(lk, [&]{ return pool_stop || !jobs.empty(); });
            if (pool_stop && jobs.empty()) {
                return; // drain remaining jobs before exiting (destructor path)
            }
            job = jobs.front();
            jobs.pop_front();
        }
        // pread OUTSIDE the lock (may block on IO)
        const auto t0 = std::chrono::steady_clock::now();
        if (job.iovs != nullptr) {
            // [CGC 2026-08-29 merge-read] one preadv covers the whole contiguous-file run
            // (memory scattered across pool slots). Verdict shared by all run members: a short
            // read cannot tell WHICH member is missing, so the whole run is failed and the
            // caller zeroes every member dst (same conservatively-safe semantics as the
            // per-segment short-read path).
#if defined(__APPLE__) || defined(__linux__)
            const ssize_t rd = preadv(fileno(job.f), job.iovs, job.niov, job.offset);
#else
            ssize_t rd = 0;
            for (int i = 0; i < job.niov; ++i) { // fallback: per-member pread into the iov dst
                const ssize_t r = pread(fileno(job.f), job.iovs[i].iov_base, job.iovs[i].iov_len,
                                        job.offset + rd);
                if (r != (ssize_t) job.iovs[i].iov_len) { rd = -1; break; }
                rd += r;
            }
#endif
            const int okv = rd == (ssize_t) job.bytes;
            for (int i = 0; i < job.niov; ++i) {
                *job.oks[i] = okv;
            }
            if (getenv("LLAMA_EXPERT_CACHE_PREAD_DBG") != nullptr && !okv) {
                fprintf(stderr, "PREADVDBG off=%llu want=%zu got=%zd errno=%d niov=%d\n",
                        (unsigned long long) job.offset, job.bytes, rd, errno, job.niov);
            }
            const auto t1 = std::chrono::steady_clock::now();
            pread_usec.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            n_reads.fetch_add(1, std::memory_order_relaxed);
            delete[] job.iovs;
            delete[] job.oks;
        } else {
            const ssize_t rd = pread(fileno(job.f), job.dst, job.bytes, job.offset);
            if (getenv("LLAMA_EXPERT_CACHE_PREAD_DBG") != nullptr && rd != (ssize_t) job.bytes) {
                struct stat st;
                fstat(fileno(job.f), &st);
                fprintf(stderr, "PREADDBG off=%llu want=%zu got=%zd errno=%d fsize=%lld\n",
                        (unsigned long long) job.offset, job.bytes, rd, errno, (long long) st.st_size);
            }
            const auto t1 = std::chrono::steady_clock::now();
            pread_usec.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            n_reads.fetch_add(1, std::memory_order_relaxed);
            *job.ok = rd == (ssize_t) job.bytes;
        }
        {
            std::lock_guard<std::mutex> lk(pool_m);
            --pool_outstanding;
            if (pool_outstanding == 0) {
                pool_done_cv.notify_all();
            }
        }
    }
}

// Submit segment jobs to the persistent pool and block until ALL complete. The ok flags are
// written by the workers; caller zeroes any failed dst (short-read path, same as the spawn
// fill). Single submitter at a time (the hook thread's per-layer batch) — no interleaving.
#ifdef __APPLE__
// [CGC 2026-08-29 RDADVISE] Darwin read-ahead advisory: queues an ASYNC page-cache populate
// for the range without blocking. Issued at submit time for the whole batch, it raises the
// effective read queue depth beyond the worker count — workers' preads then mostly hit pages
// already in flight. Purely advisory: the workers still pread the exact same ranges, so the
// pool bytes are bit-identical. LLAMA_EXPERT_CACHE_NO_RDADVISE disables (A/B; default on).
static void rdadvise_range(FILE * f, off_t off, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    struct radvisory ra;
    ra.ra_offset = off;
    ra.ra_count  = (int) (bytes > (size_t) INT_MAX ? (size_t) INT_MAX : bytes);
    (void) fcntl(fileno(f), F_RDADVISE, &ra);
}
#endif
static void fill_segments_pool(llama_expert_cache * cache,
                               const std::vector<llama_expert_cache::segment> & segs,
                               const std::vector<uint8_t *> & dsts,
                               std::vector<int> & ok) {
    const size_t n = segs.size();
    ok.assign(n, 0);
    if (n == 0) {
        return;
    }
    // [CGC 2026-08-29 merge-read] sort the batch's segments by (file_idx, file_offset) and
    // submit each file-contiguous RUN as ONE preadv job. Within one expert tensor (kind) the
    // segments of adjacent expert ids are adjacent in the file, so a layer's miss set forms
    // multi-expert runs — e.g. 72 scattered 90KB preads collapse into a handful of large
    // sequential reads (measured: 25470 reads / 15.3s pread_usec for 2.3GB vs 3.2GB/s
    // sequential). Same file ranges, same dsts -> pool contents bit-identical by construction;
    // only the syscall pattern changes. LLAMA_EXPERT_CACHE_NO_MERGE reverts to the per-segment
    // jobs (A/B; legacy aggregate init keeps that path byte-identical).
    static const bool no_merge = getenv("LLAMA_EXPERT_CACHE_NO_MERGE") != nullptr;
#ifdef __APPLE__
    static const bool no_rdadvise = getenv("LLAMA_EXPERT_CACHE_NO_RDADVISE") != nullptr;
    std::vector<std::tuple<FILE *, off_t, size_t>> advises; // read-ahead hints, issued post-submit
    advises.reserve(n);
#endif
    {
        std::lock_guard<std::mutex> lk(cache->pool_m);
        if (no_merge) {
            for (size_t i = 0; i < n; ++i) {
                cache->jobs.push_back({ cache->files.at(segs[i].file_idx),
                                        (off_t) segs[i].file_offset, segs[i].bytes,
                                        dsts[i], &ok[i] });
                cache->pool_outstanding++;
#ifdef __APPLE__
                if (!no_rdadvise) {
                    advises.emplace_back(cache->files.at(segs[i].file_idx),
                                         (off_t) segs[i].file_offset, segs[i].bytes);
                }
#endif
            }
        } else {
            std::vector<uint32_t> order(n);
            for (size_t i = 0; i < n; ++i) {
                order[i] = (uint32_t) i;
            }
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                if (segs[a].file_idx != segs[b].file_idx) {
                    return segs[a].file_idx < segs[b].file_idx;
                }
                return segs[a].file_offset < segs[b].file_offset;
            });
            size_t i = 0;
            while (i < n) {
                // extend the run while the next sorted segment is file-contiguous with the current
                size_t j = i;
                while (j + 1 < n
                       && segs[order[j + 1]].file_idx == segs[order[i]].file_idx
                       && segs[order[j]].file_offset + segs[order[j]].bytes == segs[order[j + 1]].file_offset) {
                    ++j;
                }
                if (j == i) {
                    // single segment: legacy job (iovs == nullptr -> plain pread)
                    cache->jobs.push_back({ cache->files.at(segs[order[i]].file_idx),
                                            (off_t) segs[order[i]].file_offset, segs[order[i]].bytes,
                                            dsts[order[i]], &ok[order[i]] });
                    cache->pool_outstanding++;
#ifdef __APPLE__
                    if (!no_rdadvise) {
                        advises.emplace_back(cache->files.at(segs[order[i]].file_idx),
                                             (off_t) segs[order[i]].file_offset, segs[order[i]].bytes);
                    }
#endif
                } else {
                    const int cnt = (int) (j - i + 1);
                    auto * iovs = new struct iovec[cnt];
                    auto * oks  = new int *[cnt];
                    size_t total = 0;
                    for (size_t k = i; k <= j; ++k) {
                        const uint32_t s = order[k];
                        iovs[k - i].iov_base = (void *) dsts[s];
                        iovs[k - i].iov_len  = segs[s].bytes;
                        oks[k - i]           = &ok[s];
                        total += segs[s].bytes;
                    }
                    llama_expert_cache::pread_job job;
                    job.f      = cache->files.at(segs[order[i]].file_idx);
                    job.offset = (off_t) segs[order[i]].file_offset;
                    job.bytes  = total;
                    job.dst    = nullptr; // unused on the iov path (worker branches on iovs)
                    job.ok     = nullptr;
                    job.iovs   = iovs;
                    job.oks    = oks;
                    job.niov   = cnt;
                    cache->jobs.push_back(job);
                    cache->pool_outstanding++;
#ifdef __APPLE__
                    if (!no_rdadvise) {
                        // one hint covers the whole contiguous run
                        advises.emplace_back(job.f, job.offset, total);
                    }
#endif
                }
                i = j + 1;
            }
        }
        cache->pool_cv.notify_all();
    }
#ifdef __APPLE__
    // [CGC 2026-08-29 RDADVISE] issue the read-ahead hints OUTSIDE pool_m (fcntl is cheap but
    // never under the workers' lock). Workers already drain the queue; these hints let the
    // kernel populate pages for the not-yet-started jobs concurrently, raising queue depth.
    if (!no_rdadvise) {
        for (const auto & a : advises) {
            rdadvise_range(std::get<0>(a), std::get<1>(a), std::get<2>(a));
        }
    }
#endif
    {
        std::unique_lock<std::mutex> lk(cache->pool_m);
        cache->pool_done_cv.wait(lk, [&]{ return cache->pool_outstanding == 0; });
    }
}

llama_expert_cache * llama_expert_cache_init(const llama_model * model, size_t budget_bytes) {
    if (model == nullptr || budget_bytes == 0) {
        return nullptr;
    }
    const size_t nidx = llama_model_expert_index_size(model);
    const llama_expert_index_entry * idx = llama_model_expert_index(model);
    if (nidx == 0 || idx == nullptr) {
        fprintf(stderr, "llama_expert_cache_init: model has no expert index (load with expert_cache_bytes > 0)\n");
        return nullptr;
    }
    if (model->expert_cache_path.empty()) {
        fprintf(stderr, "llama_expert_cache_init: model path not available (metadata-only load?)\n");
        return nullptr;
    }

    auto * cache = new llama_expert_cache();
    cache->index      = idx;
    cache->index_size = nidx;
    cache->budget     = budget_bytes;

    // open one file handle per distinct file_idx (split models not yet supported)
    uint32_t max_idx = 0;
    for (size_t i = 0; i < nidx; ++i) {
        max_idx = std::max(max_idx, idx[i].file_idx);
    }
    cache->files.assign(max_idx + 1, nullptr);
    std::vector<int> opened(max_idx + 1, 0);
    for (size_t i = 0; i < nidx; ++i) {
        const uint32_t f = idx[i].file_idx;
        if (!opened[f]) {
            opened[f] = 1;
            if (f == 0) {
                cache->files[f] = fopen(model->expert_cache_path.c_str(), "rb");
            }
        }
    }
    for (size_t f = 0; f < cache->files.size(); ++f) {
        if (!cache->files[f]) {
            fprintf(stderr, "llama_expert_cache_init: cannot open file idx %zu (split models not yet supported) — cache disabled\n", f);
            llama_expert_cache_free(cache);
            return nullptr;
        }
    }

    // key -> index positions (immutable after init)
    for (size_t i = 0; i < nidx; ++i) {
        cache->key_segs[make_key(idx[i].layer, idx[i].expert)].push_back((uint32_t) i);
    }

    // L3 Option A: build the static per-layer slot pool. n_expert = max expert id + 1 across
    // layers; n_slots = clamp(budget / per-expert-bytes, 8 .. 256). Each layer/kind region holds
    // n_slots * stride bytes, contiguous per kind so the FFN tensor can point into it.
    {
        uint32_t max_expert = 0;
        uint32_t max_layer  = 0;
        uint64_t exp_bytes  = 0; // bytes for one (layer,expert) blob (all kinds)
        for (size_t i = 0; i < nidx; ++i) {
            max_expert = std::max(max_expert, idx[i].expert + 1);
            max_layer  = std::max(max_layer,  idx[i].layer + 1);
        }
        for (const auto & kv : cache->key_segs) {
            uint64_t sum = 0;
            for (uint32_t pos : kv.second) {
                sum += cache->index[pos].bytes;
            }
            exp_bytes = std::max(exp_bytes, sum);
        }
        cache->n_expert = max_expert;
        // Bound the WHOLE pool (max_layer layers x per-slot bytes across all kinds) by the budget.
        // exp_bytes here is the per-slot size (sum over kinds). n_slots per layer = budget / (layers * per-slot).
        {
            uint64_t per_slot = 0;
            for (const auto & kv : cache->key_segs) {
                uint64_t sum = 0;
                for (uint32_t pos : kv.second) {
                    sum += cache->index[pos].bytes;
                }
                per_slot = std::max(per_slot, sum);
            }
            const uint64_t denom = (uint64_t) max_layer * (per_slot ? per_slot : 1);
            if (model->expert_cache_pool_capacity > 0) {
                // L4: single source of truth is the loader's capacity (budget-derived slots;
                // the pool occupies exactly expert_cache_bytes).
                cache->n_slots = (uint32_t) model->expert_cache_pool_capacity;
            } else {
                cache->n_slots  = (uint32_t) std::max<uint64_t>(8, std::min<uint64_t>(256, denom ? budget_bytes / denom : 256));
            }
        }
        if (max_layer == 0 || max_expert == 0) {
            llama_expert_cache_free(cache);
            return nullptr;
        }

        cache->slot_table.assign((size_t) max_layer * max_expert, -1);
        // [CGC §8.101 A/B] per-layer capacity: parse LAYER_CAPS (default uniform n_slots), then
        // size every per-layer slot vector to its own cap. Without the env n_slots_l stays empty
        // and slots_l == n_slots everywhere (byte-identical to the old uniform behavior).
        if (getenv("LLAMA_EXPERT_CACHE_LAYER_CAPS") != nullptr &&
                getenv("LLAMA_EXPERT_CACHE_LAYER_CAPS")[0] != '\0') {
            cache->n_slots_l.assign(max_layer, cache->n_slots);
            for (uint32_t l = 0; l < max_layer; ++l) {
                cache->n_slots_l[l] = cgc_layer_cap(l, cache->n_slots);
            }
        }
        cache->slot_owner.resize(max_layer);
        cache->slot_last_use.resize(max_layer);
        cache->slot_queued.resize(max_layer);
        cache->slot_loading.resize(max_layer);
        cache->slot_pinned.resize(max_layer);
        cache->slot_pinned_static.resize(max_layer);
        cache->win_union.resize(max_layer);
        // [CGC prefetch v2] recently-evicted expert ring (per layer): CGC_EVICTED_RING=N sets
        // capacity (default 16; 0 = off, keeps the old pure-LRU behavior for A/B).
        {
            const char * er = getenv("CGC_EVICTED_RING");
            uint32_t cap = 16;
            if (er != nullptr && er[0] != '\0') {
                long v = atol(er);
                cap = v <= 0 ? 0 : (uint32_t) std::min<long>(v, 256);
            }
            cache->evicted_recent.resize(max_layer);
            cache->evicted_ring_size.assign(max_layer, cap);
        }
        for (uint32_t l = 0; l < max_layer; ++l) {
            const uint32_t ns = slots_l(cache, l);
            cache->slot_owner[l].assign(ns, -1);
            cache->slot_last_use[l].assign(ns, 0);
            cache->slot_queued[l].assign(ns, 0);
            cache->slot_loading[l].assign(ns, 0);
            cache->slot_pinned[l].assign(ns, 0);
            cache->slot_pinned_static[l].assign(ns, 0);
        }
        if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
            unsigned busy0 = 0;
            for (uint32_t l = 0; l < max_layer; ++l)
                for (uint32_t i = 0; i < slots_l(cache, l); ++i)
                    if (cache->slot_owner[l][i] >= 0) busy0++;
            fprintf(stderr, "PFDBG init: n_layer=%u n_slots=%u busy=%u\n", max_layer, cache->n_slots, busy0);
        }
        cache->freq.assign(max_layer, std::vector<uint64_t>(cache->n_expert, 0));
        cache->pool.assign(max_layer, std::vector<std::vector<uint8_t>>(4));
        cache->pool_ext.assign(max_layer, std::vector<const uint8_t *>(4, nullptr));
        cache->pool_ext_stride.assign(max_layer, std::vector<size_t>(4, 0));
        cache->pool_ext_slots.assign(max_layer, std::vector<uint32_t>(4, 0));
        // "1" enables the pool; any other value (including "0") leaves it off. The L3-B gather
        // path is always available regardless. L4 (-ngl>0 + ALLOW_NGL) forces the pool on: the
        // Metal-visible pool is the only correct FFN source for a Metal-buft expert tensor.
        const bool pool_on = ((getenv("LLAMA_EXPERT_CACHE_POOL") != nullptr &&
                               getenv("LLAMA_EXPERT_CACHE_POOL")[0] == '1') ||
                              model->expert_cache_pool_capacity > 0);
        cache->pool_active = pool_on;

        // L3 Option A static per-layer pool: only allocated when explicitly requested
        // (LLAMA_EXPERT_CACHE_POOL=1). The wired L3-B path (ensure + fill into the per-step
        // gather buffer) never touches the pool, so allocating it unconditionally would pin
        // ~9.7 GiB of RAM for nothing and undermine bounded residency on 16 GB machines.
        if (pool_on && model->expert_cache_pool_capacity > 0) {
            fprintf(stderr, "llama_expert_cache: L4 metal pool: %u slots/layer, regions adopted from expert tensors\n",
                    cache->n_slots);
        } else if (pool_on) {
            // per (layer, kind) contiguous region sized n_slots * stride
            for (const auto & kv : cache->key_segs) {
                const uint32_t layer = (uint32_t) (kv.first >> 32);
                for (uint32_t pos : kv.second) {
                    const auto & e = cache->index[pos];
                    const int k = (int) e.kind;
                    if (k >= 0 && k < 4 && cache->pool[layer][k].empty()) {
                        cache->pool[layer][k].assign((size_t) slots_l(cache, layer) * e.bytes, 0);
                    }
                }
            }
            fprintf(stderr, "llama_expert_cache: L3 Option A slot pool: layers=%u experts=%u slots/layer=%u pool=%.1f MiB\n",
                    max_layer, max_expert, cache->n_slots,
                    (double) (max_layer * cache->n_slots * exp_bytes) / 1048576.0);
        } else {
            fprintf(stderr, "llama_expert_cache: L3 Option A slot pool skipped (set LLAMA_EXPERT_CACHE_POOL=1 to enable)\n");
        }
        if (!cache->n_slots_l.empty()) {
            uint64_t tot_slots = 0;
            for (uint32_t l = 0; l < max_layer; ++l) {
                tot_slots += slots_l(cache, l);
            }
            fprintf(stderr, "llama_expert_cache: LAYER_CAPS per-layer caps: total %llu slots (avg %.1f/layer)\n",
                    (unsigned long long) tot_slots, (double) tot_slots / max_layer);
        }
    }

    cache->bg = std::thread([cache]() { cache->bg_loop(); });

    // persistent pread worker pool (batch fill). LLAMA_EXPERT_CACHE_WORKERS=N, default 8,
    // clamped 1..64.
    {
        int n_workers = 8;
        if (getenv("LLAMA_EXPERT_CACHE_WORKERS") != nullptr) {
            n_workers = atoi(getenv("LLAMA_EXPERT_CACHE_WORKERS"));
            if (n_workers < 1) {
                n_workers = 1;
            }
            if (n_workers > 64) {
                n_workers = 64;
            }
        }
        for (int i = 0; i < n_workers; ++i) {
            cache->workers.emplace_back([cache]() { cache->pool_loop(); });
        }
    }
    return cache;
}

size_t llama_expert_cache_ensure(llama_expert_cache * cache, const uint32_t * layers, const uint32_t * experts, size_t n) {
    if (cache == nullptr) {
        return n;
    }
    size_t misses = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t key = make_key(layers[i], experts[i]);
        std::unique_lock<std::mutex> lk(cache->m);
        cache->n_requests++;
        cache->n_map_requests++;

        auto it = cache->map.find(key);
        if (it != cache->map.end()) {
            auto & s = *it->second;
            if (s.loading) {
                while (s.loading) {
                    s.cv.wait(lk);
                }
            }
            cache->n_hits++;
            cache->n_map_hits++;
            s.last_use = ++cache->tick;
            continue;
        }

        // miss: create the slot and fill synchronously (blocking IO on this thread)
        cache->n_misses++;
        misses++;
        auto new_slot = std::make_unique<llama_expert_cache::slot>();
        new_slot->key = key;
        new_slot->loading = true;
        new_slot->queued  = false;
        cache->map.emplace(key, std::move(new_slot));
        lk.unlock();

        llama_expert_cache::slot * s = cache->map.find(key)->second.get();
        const size_t filled = fill_slot(cache, s);

        lk.lock();
        s->loading = false;
        s->last_use = ++cache->tick; // mark fresh BEFORE eviction so the just-filled slot survives
        cache->total_bytes += filled;
        evict_lru(cache, 0);
        s->cv.notify_all();
    }
    return misses;
}

void llama_expert_cache_prefetch(llama_expert_cache * cache, const uint32_t * layers, const uint32_t * experts, size_t n) {
    if (cache == nullptr) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(cache->m);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t key = make_key(layers[i], experts[i]);
            if (cache->map.find(key) != cache->map.end()) {
                continue; // already resident or already queued
            }
            auto new_slot = std::make_unique<llama_expert_cache::slot>();
            new_slot->key = key;
            new_slot->loading = true;
            new_slot->queued  = true;
            cache->map.emplace(key, std::move(new_slot));
            cache->bg_queue.push_back(key);
        }
    }
    cache->bg_cv.notify_one();
}

int64_t llama_expert_cache_fill(llama_expert_cache * cache, uint32_t layer,
        const uint32_t * experts, size_t k, int kind, void * dst, size_t dst_stride) {
    if (cache == nullptr || dst == nullptr) {
        return -1;
    }
    uint8_t * out = (uint8_t *) dst;
    int64_t total = 0;
    std::unique_lock<std::mutex> lk(cache->m);
    for (size_t i = 0; i < k; ++i) {
        const uint64_t key = make_key(layer, experts[i]);
        auto it = cache->map.find(key);
        if (it == cache->map.end() || it->second->loading) {
            return -1;
        }
        auto & s = *it->second;
        const llama_expert_cache::segment * seg = nullptr;
        for (const auto & sg : s.segs) {
            if (sg.kind == kind) {
                seg = &sg;
                break;
            }
        }
        if (seg == nullptr) {
            return -1;
        }
        const size_t stride = dst_stride ? dst_stride : seg->bytes;
        memcpy(out + i * stride, s.blob.data() + seg->off, seg->bytes);
        total += seg->bytes;
        s.last_use = ++cache->tick;
    }
    return total;
}

void llama_expert_cache_get_stats(const llama_expert_cache * cache,
        size_t * requests, size_t * hits, size_t * misses, size_t * resident_bytes, size_t * file_reads) {
    if (cache == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    if (requests)       *requests       = cache->n_requests;
    if (hits)           *hits           = cache->n_hits;
    if (misses)         *misses         = cache->n_misses;
    if (resident_bytes) *resident_bytes = cache->total_bytes;
    if (file_reads)     *file_reads     = cache->n_reads.load(std::memory_order_relaxed);
}

bool llama_expert_cache_pool_active(const llama_expert_cache * cache) {
    return cache != nullptr && cache->pool_active;
}

const int32_t * llama_expert_cache_slot_table(const llama_expert_cache * cache, uint32_t layer) {
    if (cache == nullptr || !cache->pool_active || cache->n_expert == 0 || layer >= cache->slot_table.size() / cache->n_expert) {
        return nullptr;
    }
    return cache->slot_table.data() + (size_t) layer * cache->n_expert;
}

void llama_expert_cache_free(llama_expert_cache * cache) {
    delete cache;
}
