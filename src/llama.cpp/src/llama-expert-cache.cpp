#include "llama-expert-cache.h"

#include "llama-model.h" // llama_model::expert_cache_path / expert_index

#include <algorithm>
#include <limits>
#include <cstring>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <fcntl.h>      // O_RDONLY for cgc_exact_cache_verify_post_fill (open fresh fd)
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

int64_t llama_expert_cache_publish_slot_table(const llama_expert_cache * cache, uint32_t layer,
                                             int32_t * dst, uint32_t n_expert) {
    if (cache == nullptr || dst == nullptr || layer >= cache->slot_owner.size()) {
        return 0;
    }
    // [CGC 2026-09-15 S1 cleanup] The CGC_S1_TAG (1000+layer) and CGC_S1_IDENT (constant 0)
    // deliberately-wrong publishes used to sit here. They asked "does the host write reach the very
    // buffer the GPU reads, and in what order" and answered it indirectly, by whether an
    // out-of-range assertion fired. The kernel-side ids capture (CGC_IDS_CAPTURE, ggml-metal-ops.cpp)
    // answers the same question directly -- it reads the value mul_mat_id consumed -- so the two
    // invalid-mapping knobs were removed rather than kept as a standby. Git history has them.
    const uint32_t ne = n_expert < cache->n_expert ? n_expert : cache->n_expert;
    const int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    const int32_t zs = llama_expert_cache_zero_slot(cache, layer);
    int64_t clamped = 0;
    for (uint32_t e = 0; e < ne; ++e) {
        const int32_t slot = table[e];
        if (slot >= 0) {
            dst[e] = slot;
        } else if (zs >= 0) {
            dst[e] = zs;
        } else {
            // No reserved ZERO slot on this layer. The host path would write -1 here; the gather
            // must not (index -1 reads out of bounds). Clamp to 0 and let the caller report it.
            dst[e] = 0;
            clamped++;
        }
    }
    // Anything past the cache's own expert count would otherwise keep the previous step's value.
    for (uint32_t e = ne; e < n_expert; ++e) {
        dst[e] = 0;
    }
    return clamped;
}

// Finds a free slot in the layer, else evicts the LRU slot. Never evicts a slot whose prefetch
// fill is in flight (slot_loading) — the bg thread would otherwise write the old expert's bytes
// into a slot already reassigned to a new expert (silent corruption). Returns -1 when every slot
// is loading (caller waits for a fill to finish). Must be called holding cache->m.
// [CGC WIN_PIN fix -> 2026-08-30 exact-mask fix] batch_mask: slots the CALLER'S CURRENT BATCH
// already owns (hits + assigned misses) and that must never be evicted mid-batch — with WIN_PIN
// pinning the previous batches' residents, the LRU scan would otherwise see this batch's own
// fresh assignments as the "stalest non-pinned" and hand the SAME slot to two experts
// (concurrent fills overwrite each other → corrupted FFN + near-hang). nullptr = no restriction
// (single-expert ensure path).
// HISTORY (deadlock root cause, 2026-08-30): the mask used to be approximated by
// "last_use >= batch_tick" (min_tick). That approximation is ONLY valid when nothing else
// bumps the global tick during the batch. With CGC_PREFETCH_SRC=hist the bg thread completes
// flood fills DURING ensure_batch's assignment loop and stamps slot_last_use = ++tick on
// every completed slot — i.e. >= batch_tick — so pick_slot mistook bg-completed slots for
// "assigned by this batch" and skipped them in ALL passes. Once the bg drained the flood
// queue, every slot had last_use >= batch_tick, pick_slot returned -1 forever and the main
// thread waited on bg_cv with the bg thread asleep (queue empty) = PERMANENT DEADLOCK
// (repro: non-MTP llama-simple, CGC_PREFETCH_SRC=hist, prefill chunk il=1). The exact mask
// fixes it: only the batch's OWN hits/assignments are protected; bg-completed slots stay
// evictable (their freshness still wins LRU when there is no pressure — prefetch yields to
// actual demand only under pressure, which is the correct priority). In the OFF path (no
// prefetch → bg idle → the only tick bumps during the batch are the batch's own hits/
// assignments) mask == {last_use >= batch_tick} exactly, so the eviction choice sequence is
// bit-identical to the old min_tick behavior.
// Overflow pass: when every non-pinned candidate is exhausted, evict the LRU PINNED slot
// instead of returning -1 / deadlocking (a pin is a preference, not a hard guarantee).
// [CGC P1 prefill-protect 2026-09-12] When CGC_PREFILL_PROTECT=1, a fill issued during the
// PREFILL phase must not evict a slot whose current owner was filled during a decode phase.
// Rationale (measured on this box, 8 GiB pool, IQ3_XXS-denseIQ4X): a 67-token prefill churns
// the pool and the next 32 decode tokens must re-read ~1850 experts (~2 GB, ~2.4 s) to
// re-converge. Steady-state decode is 22.2 t/s but drops to 8.1-8.9 t/s for the first
// generation after any prefill. Prefill's per-chunk union is at most n_batch(8) x topk(8) = 64
// slots/layer against 143 usable, so there is room to keep the ~45 slots/layer that decode
// converges on. Purely a victim-choice change: same fills, same remap, same maths.
// Default OFF = eviction order untouched.
//
// [CGC PREFILL_PROTECT A/B 2026-09-13] This used to be a function-local static, i.e. decided
// once at process start, so an A/B needed one full server reload per arm. On this 16 GB box a
// reload is not a neutral act: 12.7 GB model (--no-mmap) + 8 GiB pool over-commits RAM, swap
// grows by GBs across arms, and the after-prefill decode metric then moves by up to 4.7x
// (116 -> 541 ms/tok) for reasons that have nothing to do with the victim rule. Measured
// across 5 interleaved blocks, that drift was larger than the effect under test.
// `CGC_PREFILL_PROTECT_FILE` makes the switch runtime-togglable so both arms can share ONE
// server (and therefore one machine state). The file is re-read once per batch from
// llama_context::set_cgc_phase - one open()/read() per decode step, not per fill. Contents:
// "1" = force on, "0" = force off, empty/missing = fall back to the env default.
static int g_prefill_protect_override = -1;  // -1 = env default, 0 = off, 1 = on
// [A/B rig] the cache the snapshot below belongs to. Set in llama_expert_cache_init, cleared
// in the destructor. Only used by the rig print, which costs nothing when no rig is attached.
static llama_expert_cache * g_rig_cache = nullptr;

void llama_expert_cache_set_prefill_protect(int mode) {
    g_prefill_protect_override = mode;
}

// Cumulative-counter snapshot at every arm boundary. This is what makes the A/B identifiable:
// the wall-clock metric on this box is dominated by how deep into swap the machine currently
// is (measured: 116 -> 541 ms/tok purely from swap growth inside one run), whereas read/miss
// COUNTS are a property of the routing and the pool policy alone, so they do not move when the
// machine thrashes. Differencing consecutive snapshots gives per-arm counts.
static void rig_snapshot(int mode) {
    llama_expert_cache * c = g_rig_cache;
    if (c == nullptr) {
        return;
    }
    const size_t reads  = c->n_reads.load(std::memory_order_relaxed);
    const size_t reqs   = c->n_requests;
    const size_t hits   = c->n_hits;
    const size_t misses = c->n_misses;
    // pread_usec/fill_batch_usec are here (not only in the teardown) because the teardown line is
    // unreliable: on SIGINT the process can abort in ggml_metal_device_free (GGML_ASSERT
    // [rsets->data count] == 0) before ~llama_expert_cache runs, and a SECOND interrupt makes it
    // worse ("terminating immediately"). Mid-run snapshots do not depend on shutdown at all.
    fprintf(stderr,
            "CGC-RIG-SNAPSHOT mode=%d reads=%zu read_bytes=%llu pread_usec=%llu fill_usec=%llu "
            "fill_wait_us=%llu "
            "reqs=%zu hits=%zu misses=%zu "
            "defer_skip=%llu defer_yield=%llu fast_calls=%zu fast_union=%zu fast_cold=%zu\n",
            mode, reads, (unsigned long long) c->n_read_bytes.load(std::memory_order_relaxed),
            (unsigned long long) c->pread_usec.load(std::memory_order_relaxed),
            (unsigned long long) c->fill_batch_usec.load(std::memory_order_relaxed),
            (unsigned long long) c->fill_wait_us.load(std::memory_order_relaxed),
            reqs, hits, misses,
            (unsigned long long) c->n_defer_skip,
            (unsigned long long) c->n_prefill_defer_yield,
            c->n_fast_calls, c->n_fast_union, c->n_fast_cold);
}

void llama_expert_cache_refresh_prefill_protect() {
    static const char * path = getenv("CGC_PREFILL_PROTECT_FILE");
    if (path == nullptr || path[0] == '\0') {
        return;  // no rig attached: env default decides, permanently
    }
    // Report on any CONTENT change, not just a mode change: the rig appends a sequence number
    // after the mode, so two consecutive same-arm requests still get one snapshot each. That
    // makes requesting k's counters = snapshot[k+1] - snapshot[k], i.e. per-request resolution.
    static char last_buf[8] = { 0 };
    static size_t last_n = 0;
    char buf[8] = { 0 };
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        g_prefill_protect_override = -1;
        return;
    }
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    g_prefill_protect_override = (n == 0) ? -1 : (buf[0] == '1' ? 1 : (buf[0] == '0' ? 0 : -1));
    if (n != last_n || memcmp(buf, last_buf, n) != 0) {
        memcpy(last_buf, buf, n);
        last_n = n;
        rig_snapshot(g_prefill_protect_override);
    }
}

static bool cgc_prefill_protect_on() {
    if (g_prefill_protect_override >= 0) {
        return g_prefill_protect_override != 0;
    }
    static const bool env_on = getenv("CGC_PREFILL_PROTECT") != nullptr;
    return env_on;
}

static int32_t pick_slot(llama_expert_cache * cache, uint32_t layer, const uint8_t * batch_mask = nullptr,
                         bool defer_decode = false) {
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
    // preference) when nothing else is available. Pass 2 ([CGC routing-aware placement]
    // 2026-08-29) extends the overflow to STATIC pins (slot_pinned_static — the PIN_PROFILE
    // hot set): the pin is a strong preference, but a fill that needs a slot and finds only
    // pinned ones must NOT skip (a skipped expert leaves table == -1 and the strict catch-up
    // remap writes -1 -> ggml_get_rows reads an OOB pool row -> NaN cascade) and must NOT
    // wait forever (no in-flight fill can free a static pin). So under direct pressure the
    // LRU static pin yields. Neither pass touches this-batch-owned slots (batch_mask) or
    // in-flight fills — pick_slot still returns -1 only while a fill is in flight, which the
    // caller's bg_cv.wait handles correctly.
    // [CGC SpAc EMA-victim 2026-09-06] when CGC_SPAC=1, choose the victim by the slot
    // owner's EMA utility (spac_util) instead of pure LRU. Lower utility = more likely
    // evicted; tie-break by last_use so behavior is deterministic and degrades to LRU
    // when utilities are equal (e.g. before the first feed). Default OFF = bit-identical
    // pure-LRU. The utility vector is seeded 0.5 for all experts, so pre-feed tie-break
    // by last_use preserves the old eviction order exactly.
    //
    // [CGC 2026-09-15 FIX] `best_util` used to be initialised to 2.0 with the comment
    // "> max possible (1.0), so any real utility wins". That upper bound is NOT a property of
    // spac_util: `spac_update` bumps `u[e] += bump` once per OCCURRENCE in the routed union, so
    // an expert selected k times in one step reaches the fixed point k (measured: 4.375 for the
    // layer-1 owner of slot 0). Once any slot's utility exceeded 2.0 the comparison
    // `util < best_util` was false for every candidate, all three passes returned best_slot == -1
    // and the caller aborted with "no usable slot and no fill in flight" -- a hard crash on a
    // full pool, reachable only with CGC_SPAC=1 and only once the EMA had accumulated past 2.0
    // (which is why it survived the 09-14/09-15 SPAC sweeps). Initialising to +infinity makes
    // every finite utility win, which is exactly what the old code did whenever all utilities
    // were < 2.0 -- so the SPAC-off and SPAC-normal paths stay byte-identical.
    const bool spac_victim = cgc_spac_on() && layer < cache->spac_util.size() &&
                             !cache->spac_util[layer].empty();
    for (int pass = 0; pass < 3; ++pass) {
        uint64_t best_tick = UINT64_MAX;
        double   best_util = std::numeric_limits<double>::infinity();
        int32_t  best_slot = -1;
        // [CGC 2026-09-15 FIX] flag-filtered LRU candidate tracked independently of the SpAc
        // branch. It is only ever consulted when the SpAc branch selected nothing, so it cannot
        // change any victim choice that used to succeed; it converts the "every utility is
        // infinite/NaN" abort into the LRU eviction the callers' contract already assumes
        // ("pick_slot returns -1 only while a fill is in flight").
        uint64_t lru_tick = UINT64_MAX;
        int32_t  lru_slot = -1;
        for (uint32_t i = 0; i < ns; ++i) {
            if (load[i] || queued[i]) {
                continue;
            }
            if (pass < 2 && cache->slot_pinned_static[layer][i]) {
                continue;
            }
            if (batch_mask != nullptr && batch_mask[i]) {
                continue; // owned by the caller's current batch (hit or assigned miss)
            }
            if (pass == 0 && cache->slot_pinned[layer][i]) {
                continue;
            }
            // [CGC P1 prefill-protect] defer decode-filled victims to the overflow pass. Pass 2
            // already carries the rule that a fill under direct pressure must yield rather than
            // leave table[e] == -1 (the abort / OOB class documented above).
            if (defer_decode && pass < 2 && cache->slot_decode_reserved[layer][i]) {
                cache->n_defer_skip++;  // engagement proof: the rule changed this victim choice
                continue;
            }
            if (last[i] < lru_tick) {
                lru_tick = last[i];
                lru_slot = (int32_t) i;
            }
            if (spac_victim && owner[i] >= 0 && owner[i] < (int32_t) cache->n_expert) {
                // can only be false when spac_util holds a non-finite value; log once so a
                // corrupt utility row is visible instead of silently disabling SpAc victims.
                const double util = cache->spac_util[layer][owner[i]];
                if (!(util < std::numeric_limits<double>::infinity())) {
                    if (cache->n_spac_nonfinite == 0) {
                        fprintf(stderr, "CGC-SPAC: non-finite util=%.17g at layer=%u expert=%u — "
                                        "falling back to pure-LRU victims for this slot\n",
                                util, layer, owner[i]);
                    }
                    cache->n_spac_nonfinite++;
                }
                if (util < best_util || (util == best_util && last[i] < best_tick)) {
                    best_util = util;
                    best_tick = last[i];
                    best_slot = (int32_t) i;
                }
            } else {
                if (last[i] < best_tick) {
                    best_tick = last[i];
                    best_slot = (int32_t) i;
                }
            }
        }
        if (best_slot < 0 && lru_slot >= 0) {
            best_slot = lru_slot;
            best_tick = lru_tick;
            cache->n_spac_lru_fallback++;
        }
        if (best_slot >= 0) {
            if (pass == 2) {
                // the pin yielded its slot: clear the flag so the slot's NEW owner does not
                // inherit pin protection it never earned (static pins are never rebuilt
                // wholesale the way WIN_PIN's dynamic pins are).
                cache->slot_pinned_static[layer][best_slot] = 0;
                cache->n_pin_yield++;
            }
            if (defer_decode && cache->slot_decode_reserved[layer][best_slot]) {
                cache->n_prefill_defer_yield++;
                cache->slot_decode_reserved[layer][best_slot] = 0;
            }
            const int32_t evicted = owner[best_slot];
            if (evicted >= 0) {
                cache->n_evictions++; // [CGC miss attribution] a resident expert is losing its slot
            }
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

// [CGC V1 verify 2026-09-05] Byte-identity verifier for the expert cache (port from
// flashkv0516/expert-cache-fork). After every fill (spawn or pool-worker), re-open the GGUF
// through a FRESH file descriptor and pread the same (file_offset, bytes) range into a side
// buffer, then byte-compare against dst. Catches three classes of §5 violations: (a) wrong
// file_offset math in ensure_batch / pick_slot, (b) wrong dst pointer in fill_segments_pool,
// (c) cross-layer file_offset aliasing. The INDEPENDENT fd rules out any stdio buffering on
// cache->files[file_idx] masking an off-by-N read.
//
// Env-gated: only runs when CGC_EXACT_CACHE_VERIFY=1 (prod default off; CI / ngl=99 reproduce
// must set it). CGC_EXACT_CACHE_VERIFY_FIRST_N (default 256) caps the number of segments
// verified per fill batch — the verifier is O(bytes) and would otherwise dominate decode under
// ngl=99 full offload. The first batch always verifies everything; subsequent batches are
// sample-bounded so a single rogue segment is found fast without paying the check on every
// token.
static void cgc_exact_cache_verify_post_fill(llama_expert_cache * cache,
                                              const std::vector<llama_expert_cache::segment> & segs,
                                              const std::vector<uint8_t *> & dsts) {
    if (getenv("CGC_EXACT_CACHE_VERIFY") == nullptr) {
        return;
    }
    static int cgc_exact_remain = -1;
    if (cgc_exact_remain < 0) {
        const char * e = getenv("CGC_EXACT_CACHE_VERIFY_FIRST_N");
        cgc_exact_remain = e ? atoi(e) : 256;
    }
    static std::unordered_set<uint64_t> cgc_exact_seen; // (file,off,len) — already verified
    for (size_t i = 0; i < segs.size(); ++i) {
        if (cgc_exact_remain <= 0) {
            return;
        }
        const auto & s = segs[i];
        if (s.file_idx >= cache->files_path.size() || cache->files_path[s.file_idx].empty()) {
            // split-model or unset path: skip silently (the cache couldn't fill either)
            continue;
        }
        const uint64_t key = ((uint64_t) s.file_idx << 40) ^ s.file_offset ^ s.bytes;
        if (cgc_exact_seen.count(key)) {
            continue;
        }
        cgc_exact_seen.insert(key);
        cgc_exact_remain--;

        const int fd = ::open(cache->files_path[s.file_idx].c_str(), O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "CGC-EXACT-VERIFY: open(%s) failed errno=%d — verifier disabled for this segment\n",
                    cache->files_path[s.file_idx].c_str(), errno);
            continue;
        }
        std::vector<uint8_t> ref(s.bytes);
        size_t off = 0;
        while (off < s.bytes) {
            const ssize_t r = ::pread(fd, ref.data() + off, s.bytes - off,
                                       (off_t) s.file_offset + (off_t) off);
            if (r <= 0) {
                fprintf(stderr, "CGC-EXACT-VERIFY: pread failed off=%zu errno=%d — aborting\n",
                        off, errno);
                ::close(fd);
                abort();
            }
            off += (size_t) r;
        }
        ::close(fd);

        size_t diff_off = SIZE_MAX;
        for (size_t k = 0; k < s.bytes; ++k) {
            if (dsts[i][k] != ref[k]) { diff_off = k; break; }
        }
        if (diff_off != SIZE_MAX) {
            fprintf(stderr,
                "CGC-EXACT-MISMATCH: file_idx=%u file_off=%lu bytes=%u "
                "diff_off=%zu pool=0x%02x ref=0x%02x dst=%p path=%s — §5 violated, aborting\n",
                s.file_idx, (unsigned long) s.file_offset, s.bytes, diff_off,
                dsts[i][diff_off], ref[diff_off], (const void *) dsts[i],
                cache->files_path[s.file_idx].c_str());
            abort();
        }
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
    // [CGC V1 verify 2026-09-05] byte-identity check after the spawn fill
    cgc_exact_cache_verify_post_fill(cache, segs, dsts);
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
            const int64_t fw0 = ggml_time_us();
            cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
            cache->fill_wait_us.fetch_add((uint64_t) (ggml_time_us() - fw0), std::memory_order_relaxed);
        }
        (count ? cache->n_hits : cache->n_prewarm_hits)++;
        cache->slot_last_use[layer][slot] = ++cache->tick;
        return slot;
    }
    (count ? cache->n_misses : cache->n_prewarm_misses)++;
    if (count && expert < cache->n_expert) {
        // [CGC miss attribution] first demand touch of this (layer, expert) = compulsory; a
        // second touch means it was resident once and got evicted = capacity.
        if (cache->ever_loaded[layer][expert] == 0) {
            cache->ever_loaded[layer][expert] = 1;
            cache->n_distinct_demanded[layer]++;
            cache->n_miss_compulsory++;
        } else {
            cache->n_miss_capacity++;
        }
    }

    // miss: find the slot (free or evict LRU), fill it from the file synchronously.
    // Option A: the pool IS the storage — pread straight into the slot's regions (no
    // map/blob intermediate, no budget accounting; the pool is statically bounded).
    int32_t slot;
    for (;;) {
        slot = pick_slot(cache, layer);
        if (slot >= 0) {
            break;
        }
        // [CGC 2026-08-30 hang watchdog] same as ensure_batch: -1 is recoverable only while a
        // fill is in flight; with nothing loading/queued the wait would sleep forever.
        bool in_flight = false;
        for (uint32_t i = 0; i < slots_l(cache, layer); ++i) {
            if (cache->slot_loading[layer][i] || cache->slot_queued[layer][i]) {
                in_flight = true;
                break;
            }
        }
        if (!in_flight) {
            // [CGC 2026-09-15] the bare message below was not actionable: `pick_slot` returning -1
            // and this loop seeing no in-flight fill can only BOTH be true when the two loops are
            // not scanning the same set of slots -- i.e. when the per-layer vectors are shorter
            // than `slots_l(cache, layer)`, or `slots_l` itself is 0. Dump the shape so the next
            // occurrence names the culprit instead of requiring another bisect.
            const uint32_t ns_l      = slots_l(cache, layer);
            const uint32_t ns_usable = llama_expert_cache_usable_slots(cache, layer);
            size_t n_owned = 0, n_load = 0, n_queued = 0, n_pin = 0, n_dres = 0, n_pin_static = 0;
            uint64_t last_min = UINT64_MAX, last_max = 0;
            for (uint32_t i = 0; i < ns_l && i < cache->slot_owner[layer].size(); ++i) {
                if (cache->slot_owner[layer][i] >= 0)    { n_owned++;  }
                if (cache->slot_loading[layer][i])       { n_load++;   }
                if (cache->slot_queued[layer][i])        { n_queued++; }
                if (cache->slot_pinned[layer][i])        { n_pin++;    }
                if (cache->slot_pinned_static[layer][i]) { n_pin_static++; }
                if (cache->slot_decode_reserved[layer][i]) { n_dres++; }
                const uint64_t lu = cache->slot_last_use[layer][i];
                last_min = std::min(last_min, lu);
                last_max = std::max(last_max, lu);
            }
            // The eviction passes can only return -1 when every owned slot is rejected, and the
            // ONLY rejection rule that depends on data rather than flags is the SpAc victim branch
            // (`util < best_util || util == best_util`), which is false for every comparison once
            // `util` is NaN. Dump the utility row so the next occurrence is self-explanatory.
            const bool spac_victim = cgc_spac_on() && layer < cache->spac_util.size();
            const size_t spac_row = layer < cache->spac_util.size() ? cache->spac_util[layer].size() : 0;
            double u0 = -1.0, u1 = -1.0, ow0 = -1.0;
            if (spac_row > 0 && cache->slot_owner[layer][0] >= 0) {
                ow0 = (double) cache->slot_owner[layer][0];
                if ((size_t) cache->slot_owner[layer][0] < spac_row) {
                    u0 = cache->spac_util[layer][cache->slot_owner[layer][0]];
                }
            }
            if (spac_row > 0) { u1 = cache->spac_util[layer][0]; }
            fprintf(stderr,
                    "llama_expert_cache: FATAL ensure_slot layer=%u: no usable slot and no fill in flight — cannot assign; aborting\n"
                    "llama_expert_cache:   shape: slots_l=%u usable=%u n_slots=%u n_slots_l_size=%zu zero_slot=%d\n"
                    "llama_expert_cache:   vectors: owner=%zu loading=%zu queued=%zu pinned=%zu pinned_static=%zu decode_reserved=%zu\n"
                    "llama_expert_cache:   occupancy: owned=%zu loading=%zu queued=%zu pinned=%zu pinned_static=%zu decode_reserved=%zu\n"
                    "llama_expert_cache:   last_use: min=%llu max=%llu tick=%llu\n"
                    "llama_expert_cache:   spac: on=%d victim=%d rows=%zu row_size=%zu util[layer][0]=%.17g util[layer][owner[0]]=%.17g owner[0]=%.0f\n"
                    "llama_expert_cache:   cache=%p n_expert=%u n_layer=%zu slot_owner_size=%zu n_slots_l_nonzero=%zu\n",
                    layer,
                    ns_l, ns_usable, cache->n_slots, cache->n_slots_l.size(),
                    (int) (zero_slot_enabled() ? 1 : 0),
                    cache->slot_owner[layer].size(), cache->slot_loading[layer].size(),
                    cache->slot_queued[layer].size(), cache->slot_pinned[layer].size(),
                    cache->slot_pinned_static[layer].size(), cache->slot_decode_reserved[layer].size(),
                    n_owned, n_load, n_queued, n_pin, n_pin_static, n_dres,
                    (unsigned long long) last_min, (unsigned long long) last_max,
                    (unsigned long long) cache->tick,
                    (int) (cgc_spac_on() ? 1 : 0), (int) (spac_victim ? 1 : 0),
                    cache->spac_util.size(), spac_row, u1, u0, ow0,
                    (const void *) cache, cache->n_expert,
                    cache->n_expert ? cache->slot_table.size() / cache->n_expert : (size_t) 0,
                    cache->slot_owner.size(),
                    (size_t) std::count_if(cache->n_slots_l.begin(), cache->n_slots_l.end(),
                                           [](uint32_t v) { return v > 0; }));
            abort();
        }
        cache->bg_cv.wait(lk); // all slots loading: wait for a fill to finish, then retry
    }
    cache->slot_owner[layer][slot] = (int32_t) expert;
    lk.unlock();

    const int64_t fill_t0 = ggml_time_us();
    fill_pool_direct(cache, layer, slot, expert);
    cache->fill_wait_us.fetch_add((uint64_t) (ggml_time_us() - fill_t0), std::memory_order_relaxed);

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
                                     const uint32_t * experts, size_t n,
                                     bool defer_decode_protect) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || n == 0) {
        return;
    }
    const bool defer_decode = defer_decode_protect && cgc_prefill_protect_on();
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::vector<int32_t>  slots(n, -1);
    std::vector<uint32_t> miss_exps;   // (expert) to fill concurrently
    std::vector<int32_t>  miss_slots;  // slot assigned to each miss
    {
        std::unique_lock<std::mutex> lk(cache->m);
        cache->n_requests += n;
        // [CGC WIN_PIN fix -> 2026-08-30 exact-mask fix] batch_owned: slots this batch has
        // already claimed (hit slots + assigned miss slots) and that pick_slot must never
        // hand to a second expert mid-batch. This EXACTLY replaces the old batch_tick
        // (min_tick) heuristic, which approximated this set as "last_use >= batch_tick" —
        // an approximation the bg prefetch thread violates (its fill completions stamp
        // ++tick on unrelated slots during our assignment loop, over-protecting them until
        // pick_slot starves and the bare bg_cv.wait below deadlocks; see pick_slot's
        // HISTORY comment). With the bg idle (prefetch OFF) the two sets coincide exactly,
        // so the OFF path eviction sequence is bit-identical.
        std::vector<uint8_t> batch_owned(llama_expert_cache_usable_slots(cache, layer), 0);
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
                    const int64_t fw0 = ggml_time_us();
                    cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
                    cache->fill_wait_us.fetch_add((uint64_t) (ggml_time_us() - fw0), std::memory_order_relaxed);
                }
                cache->n_hits++;
                cache->slot_last_use[layer][slot] = ++cache->tick;
                // [CGC P1 prefill-protect] a decode hit promotes the slot to protected.
                if (!defer_decode_protect) {
                    cache->slot_decode_reserved[layer][slot] = 1;
                }
                batch_owned[slot] = 1; // this batch reads it via the remap: never evict mid-batch
                slots[i] = slot;
                continue;
            }
            cache->n_misses++;
            // [CGC miss attribution] same split as ensure_slot: first demand touch of this
            // (layer, expert) is compulsory, a repeat means it was evicted = capacity.
            if (cache->ever_loaded[layer][e] == 0) {
                cache->ever_loaded[layer][e] = 1;
                cache->n_distinct_demanded[layer]++;
                cache->n_miss_compulsory++;
            } else {
                cache->n_miss_capacity++;
            }
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
                slot = pick_slot(cache, layer, batch_owned.data(), defer_decode);
                if (slot >= 0) {
                    break;
                }
                // [CGC 2026-08-30 hang watchdog] pick_slot == -1 is recoverable ONLY while a
                // fill is in flight on this layer (loading/queued — bg will complete it and
                // notify). If nothing is in flight, every usable slot is already owned by THIS
                // batch (union larger than the layer's slot count) and the wait below would
                // sleep forever: the 2026-08-30 hist-prefetch deadlock class. Fail loudly with
                // the diagnosis instead of hanging silently.
                bool in_flight = false;
                for (uint32_t i = 0; i < slots_l(cache, layer); ++i) {
                    if (cache->slot_loading[layer][i] || cache->slot_queued[layer][i]) {
                        in_flight = true;
                        break;
                    }
                }
                if (!in_flight) {
                    // [CGC 2026-09-15] the headline numbers in this message were unreadable on
                    // their own: it printed "8 distinct experts exceed the 143 usable pool slots",
                    // where 8 > 143 is impossible. `pick_slot` can only return -1 here when every
                    // slot in [0, usable) is load/queued/batch_mask-pinned, and this loop just
                    // proved none is load/queued -- so the batch mask itself is the suspect. Dump
                    // the mask and the vector shapes instead of another bisect.
                    const uint32_t ns_l      = slots_l(cache, layer);
                    const uint32_t ns_usable = llama_expert_cache_usable_slots(cache, layer);
                    size_t mask_set = 0, n_owned = 0, n_pin_static = 0, n_dres = 0;
                    for (uint32_t i = 0; i < ns_usable && i < batch_owned.size(); ++i) {
                        if (batch_owned[i]) { mask_set++; }
                    }
                    for (uint32_t i = 0; i < ns_l && i < cache->slot_owner[layer].size(); ++i) {
                        if (cache->slot_owner[layer][i] >= 0)        { n_owned++;      }
                        if (cache->slot_pinned_static[layer][i])     { n_pin_static++; }
                        if (cache->slot_decode_reserved[layer][i])   { n_dres++;       }
                    }
                    fprintf(stderr,
                            "llama_expert_cache: FATAL ensure_batch layer=%u: %zu distinct experts exceed the %u usable pool slots and no fill in flight — cannot assign; aborting\n"
                            "llama_expert_cache:   shape: slots_l=%u usable=%u n_slots=%u n_slots_l_size=%zu zero_slot=%d\n"
                            "llama_expert_cache:   mask: batch_owned_size=%zu mask_set=%zu (must be <= the %zu experts in this batch)\n"
                            "llama_expert_cache:   vectors: owner=%zu loading=%zu queued=%zu pinned_static=%zu decode_reserved=%zu\n"
                            "llama_expert_cache:   occupancy: owned=%zu pinned_static=%zu decode_reserved=%zu -- batch_owned.data()=%p\n"
                            "llama_expert_cache:   cache=%p n_expert=%u slot_owner_size=%zu n_slots_l_nonzero=%zu defer_decode=%d\n",
                            layer, n, ns_usable,
                            ns_l, ns_usable, cache->n_slots, cache->n_slots_l.size(),
                            (int) (zero_slot_enabled() ? 1 : 0),
                            batch_owned.size(), mask_set, n,
                            cache->slot_owner[layer].size(), cache->slot_loading[layer].size(),
                            cache->slot_queued[layer].size(), cache->slot_pinned_static[layer].size(),
                            cache->slot_decode_reserved[layer].size(),
                            n_owned, n_pin_static, n_dres, (const void *) batch_owned.data(),
                            (const void *) cache, cache->n_expert, cache->slot_owner.size(),
                            (size_t) std::count_if(cache->n_slots_l.begin(), cache->n_slots_l.end(),
                                                   [](uint32_t v) { return v > 0; }),
                            (int) (defer_decode ? 1 : 0));
                    abort();
                }
                cache->bg_cv.wait(lk); // all slots loading: wait for a fill to finish
            }
            cache->slot_owner[layer][slot] = (int32_t) e;
            // [CGC P1 prefill-protect] stamp the phase this slot was filled in: a decode fill
            // marks the slot protected, a prefill fill clears the mark so it is the first thing
            // the NEXT prefill evicts. NOTE: a variant that stamped ONLY on a decode HIT (on the
            // theory that "reuse" is the right protection key) measured WORSE -- 9.97-11.19 t/s
            // after a prefill vs 12.73-14.91 t/s for this one -- so protection is keyed on "a
            // decode step chose this slot", not on "decode has hit it at least once".
            // Claim the slot for this batch BEFORE the fill: the batch assigns all slots under
            // one lock, and the next pick_slot's LRU eviction would otherwise see this slot as
            // the stalest (fresh assignments bump last_use) and hand it out AGAIN (two experts
            // on one slot -> concurrent fills overwrite the same region -> nondeterministic FFN
            // garbage). batch_owned is the exact mid-batch protection; the last_use bump below
            // keeps the LRU ordering correct for FUTURE batches.
            // ensure_slot gets away without this because its next pick_slot happens only after
            // the fill + last_use bump (serial per expert).
            cache->slot_last_use[layer][slot] = ++cache->tick;
            batch_owned[slot] = 1;
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
            // [CGC routing-aware placement] the fill landed on a PIN_PROFILE member: mark the
            // slot static-pinned so pick_slot never hands it to a tail expert again. The pin
            // takes effect only after the fill (this loop) — the slot table entry is already
            // valid here, so the expert is usable from this step on.
            if (layer < cache->pin_set.size() && cache->pin_set[layer].count(e)) {
                if (!cache->slot_pinned_static[layer][s]) {
                    cache->slot_pinned_static[layer][s] = 1;
                    cache->n_pin_marked++;
                }
            }
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
            cache->drop_stats.drain_cleared++;  // #3: drain_layer clears queued-but-not-started fills
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
        cache->n_prefetch_dropped++;
        cache->drop_stats.guard_reject++;  // #7: guard-reject — expert not in key_segs / layer OOR / pool inactive
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
        // [CGC DBUF spike 2026-09-06] EMA-aware victim: when spac_util is populated (CGC_SPAC
        // feeds), prefer evicting the LOWEST-utility owner (tie-break LRU) — pure-LRU churns
        // recurring hot experts (measured: DBUF+LRU degraded longform/coding quality vs the
        // 8GB baseline). Static profile pins are skipped in the first pass and only yield as a
        // last resort (mirrors pick_slot's pass structure).
        const bool spac_v = cgc_spac_on() && layer < cache->spac_util.size() &&
                            !cache->spac_util[layer].empty();
        uint64_t best_tick = UINT64_MAX;
        double   best_util = 2.0; // > max possible (1.0), so any real utility wins
        int32_t  best_pin  = -1;  // fallback: LRU static pin (pass 2), only if nothing else
        uint64_t best_pin_tick = UINT64_MAX;
        for (uint32_t i = 0; i < ns; ++i) {
            if (cache->slot_queued[layer][i] || cache->slot_loading[layer][i]) {
                continue;
            }
            const int32_t own = cache->slot_owner[layer][i];
            if (cache->slot_pinned_static[layer][i]) {
                if (cache->slot_last_use[layer][i] < best_pin_tick) {
                    best_pin_tick = cache->slot_last_use[layer][i];
                    best_pin = (int32_t) i;
                }
                continue; // pass 0: static pins are skipped
            }
            if (spac_v && own >= 0 && own < (int32_t) cache->n_expert) {
                const double util = cache->spac_util[layer][own];
                if (util < best_util || (util == best_util && cache->slot_last_use[layer][i] < best_tick)) {
                    best_util = util;
                    best_tick = cache->slot_last_use[layer][i];
                    slot = (int32_t) i;
                }
            } else if (cache->slot_last_use[layer][i] < best_tick) {
                best_tick = cache->slot_last_use[layer][i];
                slot = (int32_t) i;
            }
        }
        if (slot < 0 && best_pin >= 0) {
            // everything else busy/queued: yield the LRU static pin (a skipped fill would leave
            // the expert cold -> guard/ensure re-fills synchronously anyway)
            slot = best_pin;
            cache->slot_pinned_static[layer][slot] = 0;
            cache->n_pin_yield++;
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
            cache->drop_stats.no_free_slot++;  // #1: prefetch_slot — no free slot AND no evictable slot
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

// [CGC DBUF spike 2026-09-06] step-ahead cold-expert refill (see cgc_dbuf_on() in the header).
// Called from the decode fast-path hook AFTER the remap leaf was written (GPU idle between
// segments; seg il+1 = this layer's FFN is submitted only after the hook returns and its remap
// references union slots only, and prefetch_slot evicts by LRU which touch() just made
// non-union) -> the bg fill of a non-union slot cannot tear any in-flight read of this decode.
// Prefetch_slot publishes slot_table only after the bytes land, so a still-loading fill at the
// next decode's hook degrades to the existing cold (ZERO/guard) handling for that one token
// instead of racing the region. Returns the number of fills queued.
// [CGC DBUF spike 2026-09-06] per-process cap on outstanding step-ahead fills. Each pread
// fill takes ~1-3ms; the decode step window is ~30-60ms, so only ~20-30 fills can land before
// the NEXT step's hooks need them. Without a cap, an early-decode cold burst (12+ cold/layer ×
// 41 layers) queues 500+ fills that never drain -> late layers' fills are perpetually in flight
// -> the next step ZERO-maps exactly the experts being refilled (quality collapse, measured:
// longform 0.882 -> 0.3/empty under un-capped DBUF). CGC_DBUF_CAP (default 24) bounds it.
static inline size_t cgc_dbuf_cap() {
    static const size_t v = []() {
        const char * s = getenv("CGC_DBUF_CAP");
        return (s != nullptr && s[0] != '\0') ? (size_t) std::max(1, atoi(s)) : 24;
    }();
    return v;
}

size_t llama_expert_cache_dbuf_refill(llama_expert_cache * cache, uint32_t layer,
                                      const uint32_t * experts, size_t n) {
    if (cache == nullptr || !cache->pool_active || experts == nullptr || n == 0 ||
        cache->n_expert == 0 || layer >= cache->slot_owner.size()) {
        return 0;
    }
    const int32_t * st = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    // outstanding = queued pool fills + in-flight (loading) fills across ALL layers
    size_t busy = 0;
    {
        std::lock_guard<std::mutex> lk(cache->m);
        busy = cache->pool_queue.size();
        for (size_t l = 0; l < cache->slot_loading.size() && busy < cgc_dbuf_cap(); ++l) {
            for (size_t s = 0; s < cache->slot_loading[l].size() && busy < cgc_dbuf_cap(); ++s) {
                busy += cache->slot_loading[l][s] ? 1u : 0u;
            }
        }
    }
    size_t cold = 0, queued = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = experts[i];
        if (e >= cache->n_expert) {
            continue;
        }
        if (st[e] >= 0) {
            continue; // resident (touch just refreshed it) -> nothing to refill
        }
        cold++;
        if (busy >= cgc_dbuf_cap()) {
            // #2: dbuf_refill — backlog full (busy >= CGC_DBUF_CAP → break, skip remaining experts)
            // Count ALL remaining cold experts in this layer's union that get skipped due to cap.
            size_t skipped = 1; // this expert
            for (size_t j = i + 1; j < n; ++j) {
                const uint32_t e2 = experts[j];
                if (e2 < cache->n_expert && st[e2] < 0) {
                    skipped++;
                }
            }
            cache->n_prefetch_dropped += skipped;
            cache->drop_stats.dbuf_cap_skip += skipped;
            break; // backlog full: skip the rest of this step (existing cold handling applies)
        }
        if (llama_expert_cache_prefetch_slot(cache, layer, e) == 0) {
            queued++;
            busy++;
        }
    }
    static int dbg_n = 0;
    if (getenv("CGC_DBUF_DBG") != nullptr && dbg_n < 40) {
        dbg_n++;
        fprintf(stderr, "CGC-DBUF: l=%u union=%zu cold=%zu queued=%zu\n", layer, n, cold, queued);
    }
    return queued;
}

// [CGC DBUF2 full double-buffer 2026-09-06] atomically swap active and scratch slot tables at
// the GPU-idle step boundary. After the swap, the new active (old scratch) contains all async
// fills that completed since the last swap PLUS the accumulated mappings from before; the new
// scratch (old active) retains its full mappings and will be incrementally updated by the next
// round of bg fills. CRITICAL: do NOT reset the new scratch to -1 — that would force every
// expert to be re-filled every step (measured: 3 t/s instead of 17+). Both buffers are full
// mirrors; fills overwrite only the experts they touch, so the scratch stays consistent.
// Must be called when decode is NOT reading slot_table (i.e. after the current step's remap
// leaves are written and before the next step's FFN is dispatched). Holds cache->m internally.
// CGC_DBUF2=0 = no-op.
void llama_expert_cache_dbuf2_swap(llama_expert_cache * cache) {
    if (cache == nullptr || !cache->pool_active || !cgc_dbuf2_on()) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    // O(1) swap: exchanges the vectors' internal data pointers (no element copy, no reset).
    cache->slot_table.swap(cache->slot_table_scratch);
    static int dbg_n = 0;
    if (getenv("CGC_DBUF2_DBG") != nullptr && dbg_n < 20) {
        dbg_n++;
        fprintf(stderr, "CGC-DBUF2: swapped active<->scratch (size=%zu)\n", cache->slot_table.size());
    }
}

// [CGC Fast-Path Wait 2026-09-06] wait for in-flight fills of cold experts so the fast path uses
// real weights instead of ZERO-mapping. Default OFF (cgc_fast_wait_on()=false) = no-op.
// Safety: does not hold cache->m on entry; takes/releases it internally. bg_cv is notified on
// every fill completion (bg_loop), so wait_until wakes promptly. Shared deadline bounds total
// added latency to cgc_fast_wait_us() regardless of how many experts are waited.
size_t llama_expert_cache_wait_loading(llama_expert_cache * cache, uint32_t layer,
                                       const uint32_t * experts, size_t n) {
    if (!cgc_fast_wait_on() || cache == nullptr || !cache->pool_active ||
        experts == nullptr || n == 0 || cache->n_expert == 0 ||
        layer >= cache->slot_owner.size()) {
        return 0;
    }
    const int32_t * st = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    const uint32_t nsl = (uint32_t) cache->slot_owner[layer].size();
    // Phase 1: find cold experts with in-flight fills (loading or queued). O(n*nsl); n<=8,
    // nsl~100 -> ~800 checks, trivial. At most cgc_fast_wait_max() experts (bounds latency).
    struct wait_ent { uint32_t expert; };
    std::vector<wait_ent> waiting;
    waiting.reserve(n);
    {
        std::lock_guard<std::mutex> lk(cache->m);
        for (size_t i = 0; i < n && waiting.size() < cgc_fast_wait_max(); ++i) {
            const uint32_t e = experts[i];
            if (e >= cache->n_expert) continue;
            if (st[e] >= 0) continue; // already resident
            for (uint32_t s = 0; s < nsl; ++s) {
                if (cache->slot_owner[layer][s] == (int32_t) e &&
                    (cache->slot_loading[layer][s] || cache->slot_queued[layer][s])) {
                    waiting.push_back({e});
                    break;
                }
            }
        }
    }
    if (waiting.empty()) return 0;
    // Phase 2: wait for each fill to land. Shared deadline = now + cgc_fast_wait_us(), so the
    // total added step latency is bounded regardless of expert count. bg_cv.notify_all() fires
    // on every fill completion (bg_loop), so wait_until wakes promptly (no polling).
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(cgc_fast_wait_us());
    size_t became_resident = 0;
    for (const auto & w : waiting) {
        std::unique_lock<std::mutex> lk(cache->m);
        const bool ok = cache->bg_cv.wait_until(lk, deadline, [&]{ return st[w.expert] >= 0; });
        if (ok && st[w.expert] >= 0) {
            became_resident++;
            // touch the just-resident slot so the next pick_slot doesn't evict it immediately
            cache->slot_last_use[layer][st[w.expert]] = ++cache->tick;
        }
    }
    static int dbg_n = 0;
    if (getenv("CGC_FAST_WAIT_DBG") != nullptr && dbg_n < 40) {
        dbg_n++;
        fprintf(stderr, "CGC-WAIT: l=%u waited=%zu became_resident=%zu\n",
                layer, waiting.size(), became_resident);
    }
    return became_resident;
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

// CGC M1 work item 1 (CGC_POOL_SPLIT) -- see the header for why this exists.
bool llama_expert_cache_pool_split_alloc(llama_expert_cache * cache, int kind,
        ggml_backend_buffer_type_t buft, size_t total_bytes, ggml_backend_buffer_t * out_buf) {
    if (cache == nullptr || kind < 0 || kind >= 4 || buft == nullptr || total_bytes == 0) {
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total_bytes);
    if (buf == nullptr) {
        fprintf(stderr, "CGC-POOL-SPLIT: allocation of %.2f MiB failed kind=%d on %s\n",
                (double) total_bytes / (1024.0 * 1024.0), kind, ggml_backend_buft_name(buft));
        return false;
    }
    // WEIGHTS usage so the scheduler prefers the backend that holds the pool (same as the model's
    // own weight buffers).
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    cache->pool_owned_bufs.push_back(buf);
    fprintf(stderr, "CGC-POOL-SPLIT: kind=%d size=%.2f MiB base=%p buft=%s\n",
            kind, (double) total_bytes / (1024.0 * 1024.0), ggml_backend_buffer_get_base(buf),
            ggml_backend_buft_name(buft));
    if (out_buf != nullptr) {
        *out_buf = buf;
    }
    return true;
}

bool llama_expert_cache_pool_split_adopt(llama_expert_cache * cache, uint32_t layer, int kind,
        const uint8_t * base, uint32_t slots, size_t stride, ggml_backend_buffer_t buf) {
    if (buf == nullptr) {
        return false;
    }
    const bool ok = llama_expert_cache_adopt_pool_region(cache, layer, kind, base, (int64_t) slots, stride);
    if (getenv("CGC_POOL_SPLIT_DBG") != nullptr) {
        fprintf(stderr, "CGC-POOL-SPLIT-ADOPT: layer=%u kind=%d slots=%u stride=%zu base=%p buf=%p "
                "pool_ext.size=%zu pool_ext_buf.size=%zu ok=%d\n",
                layer, kind, slots, stride, (const void *) base, (void *) buf,
                cache->pool_ext.size(), cache->pool_ext_buf.size(), (int) ok);
    }
    if (!ok) {
        return false;
    }
    if (layer < cache->pool_ext_buf.size()) {
        cache->pool_ext_buf[layer][kind] = buf;
    }
    return true;
}

ggml_backend_buffer_t llama_expert_cache_pool_buffer(const llama_expert_cache * cache, uint32_t layer, int kind) {
    if (cache == nullptr || layer >= cache->pool_ext_buf.size() || kind < 0 || kind >= 4) {
        return nullptr;
    }
    return cache->pool_ext_buf[layer][kind];
}

bool llama_expert_cache_pool_owned(const llama_expert_cache * cache) {
    return cache != nullptr && !cache->pool_owned_bufs.empty();
}

void llama_expert_cache_pool_set_wide(llama_expert_cache * cache, uint32_t layer, int kind,
        const void * data, ggml_backend_buffer_t buf, int64_t ne2) {
    if (cache == nullptr || layer >= cache->pool_wide_data.size() || kind < 0 || kind >= 4) {
        return;
    }
    cache->pool_wide_data[layer][kind]  = const_cast<void *>(data);
    cache->pool_wide_buf[layer][kind]   = buf;
    cache->pool_wide_ne2[layer][kind]   = ne2;
}

bool llama_expert_cache_pool_get_wide(const llama_expert_cache * cache, uint32_t layer, int kind,
        void ** data, ggml_backend_buffer_t * buf, int64_t * ne2) {
    if (cache == nullptr || layer >= cache->pool_wide_data.size() || kind < 0 || kind >= 4) {
        return false;
    }
    if (cache->pool_wide_data[layer][kind] == nullptr) {
        return false;
    }
    if (data != nullptr) { *data = cache->pool_wide_data[layer][kind]; }
    if (buf  != nullptr) { *buf  = cache->pool_wide_buf[layer][kind]; }
    if (ne2  != nullptr) { *ne2  = cache->pool_wide_ne2[layer][kind]; }
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

void llama_expert_cache_masscov_record(llama_expert_cache * cache, uint32_t layer,
                                       const uint32_t * experts, const float * w_sel, size_t n) {
    if (cache == nullptr || layer >= cache->massc_mass.size() || n == 0 ||
        layer >= cache->massc_total.size() || layer >= cache->massc_cur_cov.size() ||
        layer >= cache->massc_sel_total.size() || layer >= cache->massc_sel_cold.size()) {
        return;
    }
    std::lock_guard<std::mutex> lk(cache->m);
    const int32_t * st = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    auto & mm = cache->massc_mass[layer];
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = experts[i];
        if (e >= cache->n_expert) {
            continue;
        }
        const double w = (double) w_sel[i];
        mm[e] += w;
        cache->massc_total[layer] += w;
        // [CGC count-cold 2026-09-07] per-step SELECTED count-cold: how many of the step's
        // selected expert ids were non-resident (slot<0) — the count-based tail metric that
        // drives the cold guard / ZERO-slot contamination. Accumulated per selection so the
        // MASSCOV dump can report sel_cold/sel_total per layer (count ratio, not mass).
        cache->massc_sel_total[layer]++;
        if (st[e] < 0) {
            cache->massc_sel_cold[layer]++;
        }
        if (st[e] >= 0) {
            cache->massc_cur_cov[layer] += w;
        }
    }
}

// [CGC SpAc 2026-09-06] EMA utility update (MoESpAcEstimator port). Full-EMA decay: every
// entry of the layer decays by alpha, then the routed experts bump by (1-alpha) — an expert
// decays toward 0 when it stops being routed, while a recurring one stays near 1. Called from
// expert_cache_on_topk for every routed step (CGC_SPAC=1). The layer is checked against
// spac_util.size() so unpooled layers (e.g. L4 skip layer 0 — no spac_util row) are no-ops.
void llama_expert_cache_spac_update(llama_expert_cache * cache, uint32_t layer,
                                    const uint32_t * experts, size_t n) {
    if (cache == nullptr || layer >= cache->spac_util.size() || n == 0 || experts == nullptr) {
        return;
    }
    const double alpha = cgc_spac_alpha();
    const double bump  = 1.0 - alpha;
    std::lock_guard<std::mutex> lk(cache->m);
    auto & u = cache->spac_util[layer];
    const uint32_t ne = cache->n_expert;
    if (u.size() < ne) {
        return;
    }
    for (uint32_t e = 0; e < ne; ++e) {
        u[e] *= alpha;
    }
    for (size_t i = 0; i < n; ++i) {
        if (experts[i] < ne) {
            u[experts[i]] += bump;
        }
    }
    cache->spac_feeds++;
}

// [CGC SpAc 2026-09-06] between-step pool re-target toward the EMA utility top-K. Per layer,
// collect the NON-resident experts that exist in the GGUF index, partial-sort by utility desc,
// and prefetch the top spac_k. prefetch_slot fills via the bg thread and LRU-evicts a slot when
// the layer is full, so over refreshes the pool drifts from the loader's static experts-0..n
// prewarm to the experts decode actually routes. Read-mostly under the lock (slot_table snapshot
// must match what prefetch_slot re-checks under the same lock — it re-validates before queueing,
// so a stale snapshot here only costs a dropped prefetch).
// Cadence: the caller (process_ubatch B-section) invokes this every CGC_SPAC_REFRESH routed
// feeds; spac_feeds is the global feed counter (not per layer) so all layers refresh together.
size_t llama_expert_cache_spac_prefetch(llama_expert_cache * cache) {
    if (cache == nullptr || !cache->pool_active || cache->spac_util.empty()) {
        return 0;
    }
    const uint32_t K   = cgc_spac_k();
    const uint32_t ne  = cache->n_expert;
    size_t queued = 0;
    // [CGC EMA membership 2026-09-08] per-layer membership probe (CGC_SPAC_DBG=1): how many of
    // the EMA top-K are ALREADY resident vs queued now — the count-cold lever for ZERO-slot reads.
    static const bool mem_dbg = getenv("CGC_SPAC_DBG") != nullptr;
    for (size_t layer = 0; layer < cache->spac_util.size(); ++layer) {
        if (llama_expert_cache_slots_per_layer_l(cache, (uint32_t) layer) == 0) {
            continue; // unpooled layer (skip-layer0 / no slots): nothing to re-target
        }
        const auto & u = cache->spac_util[layer];
        if (u.size() < ne) {
            continue;
        }
        const int32_t * table = cache->slot_table.data() + layer * ne;
        std::vector<uint32_t> cand;
        cand.reserve(ne);
        // full candidate set (resident or not) so the membership probe counts overlap;
        // prefetch only the non-resident top-K members below.
        for (uint32_t e = 0; e < ne; ++e) {
            if (cache->key_segs.find(make_key((uint32_t) layer, e)) != cache->key_segs.end()) {
                cand.push_back(e);
            }
        }
        if (cand.empty()) {
            continue;
        }
        const size_t ntake = std::min<size_t>(K, cand.size());
        std::partial_sort(cand.begin(), cand.begin() + ntake, cand.end(),
                          [&u](uint32_t a, uint32_t b) { return u[a] > u[b]; });
        size_t n_resident = 0;
        for (size_t i = 0; i < ntake; ++i) {
            if (table[cand[i]] >= 0) {
                n_resident++;
                continue;
            }
            if (llama_expert_cache_prefetch_slot(cache, (uint32_t) layer, cand[i]) == 0) {
                queued++;
            }
        }
        if (mem_dbg && layer < 4) {
            static int mem_dbg_n = 0;
            if (mem_dbg_n++ < 60) {
                fprintf(stderr, "CGC-SPAC-MEM: layer=%zu topK=%zu resident=%zu (%.0f%%) queued=%zu feeds=%llu\n",
                        layer, ntake, n_resident,
                        ntake > 0 ? (double) n_resident / ntake * 100.0 : 0.0,
                        queued, (unsigned long long) cache->spac_feeds);
            }
        }
    }
    return queued;
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
        if (cgc_l4_skip_layer0_on() && l == 0) {
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
    // [CGC routing-aware placement 2026-08-29] O(1) membership set for the ensure_batch pin
    // marking (pin_profile stays the dump-format source of truth).
    cache->pin_set.assign(cache->pin_profile.size(), {});
    for (uint32_t l = 0; l < cache->pin_profile.size(); ++l) {
        cache->pin_set[l].insert(cache->pin_profile[l].begin(), cache->pin_profile[l].end());
    }
    if (getenv("LLAMA_EXPERT_CACHE_GATE_DBG") != nullptr) {
        fprintf(stderr, "PINPROFILE: %zu experts across %zu layers\n", total, layer);
    }
    return total;
}

void llama_expert_cache_prepopulate(llama_expert_cache * cache, uint32_t layer, uint32_t n_slots) {
    if (cache == nullptr || layer >= cache->slot_owner.size() || cache->n_expert == 0 || n_slots == 0) {
        return;
    }
    // [CGC MTP fast path] never hand the reserved ZERO slot to a real expert.
    const uint32_t n = std::min(n_slots, llama_expert_cache_usable_slots(cache, layer));
    // [CGC M1 work item 1 CGC_POOL_SPLIT] With an OWNED pool the identity slots hold NOTHING yet.
    // The legacy path gets their bytes for free: the loader pre-read experts 0..n-1 into the tensor's
    // own storage, and that storage IS the adopted pool region (which is why the loop below can say
    // "slot e holds expert e" without copying anything). A separate pool allocation has no such
    // loader, so the identity mapping would mark uninitialised Metal memory as resident; the first
    // routed batch then mixes filled slots (experts >= n) with empty ones (< n) and the layer emits
    // NaN (measured: hook ids at il=2 are a NaN bit pattern, then SIGSEGV in the Metal encoder).
    // Fill them explicitly instead - same bytes, same identity order, and exactly the read the
    // loader used to perform. Must run WITHOUT cache->m (fill_pool_direct preads and may block).
    if (llama_expert_cache_pool_owned(cache)) {
        for (uint32_t e = 0; e < n && e < cache->n_expert; ++e) {
            fill_pool_direct(cache, layer, (int32_t) e, e);
        }
    }
    std::lock_guard<std::mutex> lk(cache->m);
    int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
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

// [CGC decode phase decomposition 2026-09-15] see the field comment in the header: this is the
// only counter on the fill path accumulated by the CALLING thread, so a per-step delta is
// directly comparable to that step's wall clock. pread_usec is an aggregate over worker threads
// and therefore cannot be compared to a step at all.
uint64_t llama_expert_cache_fill_wait_us(const llama_expert_cache * cache) {
    return cache == nullptr ? 0 : cache->fill_wait_us.load(std::memory_order_relaxed);
}

// [CGC identity-slot verify 2026-09-09] Load-time identity fill check. The loader pre-reads
// experts 0..n_slots-1 into the adopted pool regions (identity order) and prepopulate marks them
// resident WITHOUT the runtime fill path — so CGC_EXACT_CACHE_VERIFY (which only fires post-fill)
// never checks these bytes. If the load-time pread landed at the wrong file offset (stride or
// alignment mismatch), the identity slots silently hold garbage and EVERY layer computes with
// wrong weights from the first chunk — matches the L4 oracle: diverges from control at prefill
// chunk 1 while V1 reports 0 mismatch. Env-gated: LLAMA_EXPERT_CACHE_VERIFY_IDENTITY=1.
// Verifies layers 0..2 in full; deeper layers sample slots 0..1 (same loader path, spot check).
void llama_expert_cache_verify_identity(llama_expert_cache * cache) {
    if (cache == nullptr || getenv("LLAMA_EXPERT_CACHE_VERIFY_IDENTITY") == nullptr) {
        return;
    }
    size_t checked = 0, mismatched = 0;
    const uint32_t n_layers = (uint32_t) cache->slot_owner.size();
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (int kind = 0; kind < 4; ++kind) {
            size_t stride = 0;
            uint32_t slots = 0;
            const uint8_t * region = pool_region(cache, layer, kind, &stride, &slots);
            if (region == nullptr || stride == 0 || slots == 0) {
                continue;
            }
            const uint32_t n_check = (layer < 3) ? slots : (slots < 2 ? slots : 2);
            for (uint32_t s = 0; s < n_check; ++s) {
                // locate the index entry for (layer, kind, expert=s)
                const llama_expert_index_entry * ent = nullptr;
                for (size_t i = 0; i < cache->index_size; ++i) {
                    const auto & e = cache->index[i];
                    if (e.layer == layer && (uint32_t) e.kind == (uint32_t) kind && e.expert == s) {
                        ent = &e;
                        break;
                    }
                }
                if (ent == nullptr || ent->file_idx >= cache->files_path.size() ||
                    cache->files_path[ent->file_idx].empty()) {
                    continue;
                }
                const uint8_t * slot = region + (size_t) s * stride;
                const int fd = ::open(cache->files_path[ent->file_idx].c_str(), O_RDONLY);
                if (fd < 0) {
                    fprintf(stderr, "CGC-IDENT-VERIFY: open(%s) failed errno=%d\n",
                            cache->files_path[ent->file_idx].c_str(), errno);
                    continue;
                }
                std::vector<uint8_t> ref((size_t) ent->bytes);
                size_t off = 0;
                while (off < ref.size()) {
                    const ssize_t r = ::pread(fd, ref.data() + off, ref.size() - off,
                                              (off_t) ent->file_offset + (off_t) off);
                    if (r <= 0) {
                        fprintf(stderr, "CGC-IDENT-VERIFY: pread failed off=%zu errno=%d — aborting\n",
                                off, errno);
                        ::close(fd);
                        abort();
                    }
                    off += (size_t) r;
                }
                ::close(fd);
                checked++;
                size_t diff = SIZE_MAX;
                for (size_t k = 0; k < ref.size(); ++k) {
                    if (slot[k] != ref[k]) {
                        diff = k;
                        break;
                    }
                }
                if (diff != SIZE_MAX) {
                    mismatched++;
                    fprintf(stderr,
                            "CGC-IDENT-MISMATCH: layer=%u kind=%d slot=%u expert=%u "
                            "file_off=%llu bytes=%u diff_off=%zu pool=0x%02x ref=0x%02x "
                            "stride=%zu path=%s — load-time identity fill corrupted\n",
                            layer, kind, s, s, (unsigned long long) ent->file_offset,
                            (unsigned) ent->bytes, diff, slot[diff], ref[diff], stride,
                            cache->files_path[ent->file_idx].c_str());
                    if (mismatched >= 8) {
                        fprintf(stderr, "CGC-IDENT-VERIFY: %zu checked, %zu mismatched (first 8 shown) — aborting\n",
                                checked, mismatched);
                        abort();
                    }
                }
            }
        }
    }
    fprintf(stderr, "CGC-IDENT-VERIFY: %zu identity slots byte-compared vs GGUF, %zu mismatched\n",
            checked, mismatched);
    if (mismatched > 0) {
        abort();
    }
}

// [CGC §8.101 A/B] per-layer cap: the slot count actually usable for `layer`. Without the env
// this equals slots_per_layer (uniform n_slots).
uint32_t llama_expert_cache_slots_per_layer_l(const llama_expert_cache * cache, uint32_t layer) {
    return cache == nullptr ? 0 : slots_l(cache, layer);
}

// [CGC 2026-09-06 load-time pin prefill] Fill + static-pin every PIN_PROFILE member at load
// time. Previously the pin profile only marked slots pinned AFTER a runtime ensure_batch filled
// them, so the first requests still paid cold preads for the (measured) hottest experts and the
// pool started with identity-ordered prepopulated slots instead of routing-weight-ordered ones.
// Uses ensure_slot (count=false => prewarm accounting, bit-identical fill path); the pin marking
// then runs here so pick_slot can never hand these slots to tail experts. Returns filled count.
size_t llama_expert_cache_pin_prefill(llama_expert_cache * cache) {
    if (cache == nullptr || cache->pin_profile.empty()) {
        return 0;
    }
    size_t filled = 0;
    for (uint32_t layer = 0; layer < cache->pin_profile.size(); ++layer) {
        const uint32_t usable = llama_expert_cache_usable_slots(cache, layer);
        uint32_t done = 0;
        for (uint32_t e : cache->pin_profile[layer]) {
            if (done >= usable) {
                break; // profile lists more than the layer can hold; keep the head (hot) ones
            }
            const int32_t slot = llama_expert_cache_ensure_slot(cache, layer, e, /*count=*/false);
            if (slot < 0) {
                continue;
            }
            std::lock_guard<std::mutex> lk(cache->m);
            if (!cache->slot_pinned_static[layer][slot]) {
                cache->slot_pinned_static[layer][slot] = 1;
                cache->n_pin_marked++;
            }
            done++;
            filled++;
        }
    }
    fprintf(stderr, "llama_expert_cache: pin prefill: %zu experts filled+static-pinned at load\n", filled);
    return filled;
}

// CGC L3 Option A: kernel slot-table registry (defined in ggml-cpu.c)
extern "C" void ggml_cpu_clear_mmid_slot_tables_all(void);

llama_expert_cache::~llama_expert_cache() {
    g_rig_cache = nullptr;  // [A/B rig] stop snapshotting a cache that is going away
    ggml_cpu_clear_mmid_slot_tables_all(); // pool buffers are about to die
    {
        std::lock_guard<std::mutex> lk(m);
        // [CGC 2026-09-13] `resident` used to print total_bytes, which is the L3 blob-map
        // accounting. The L4 path adopts its pool regions from the expert tensors and never
        // touches the blob map, so every L4 run reported resident=0.00 MiB while an 8 GiB pool
        // was fully in use -- precisely the number you need when reasoning about RAM. Use the
        // blob accounting when it is live, otherwise the L4 pool's real occupancy
        // (filled slots x that layer's per-slot stride over the FFN kinds).
        double resident_mib = total_bytes / 1024.0 / 1024.0;
        if (resident_mib == 0.0 && !pool_ext.empty()) {
            for (size_t l = 0; l < slot_owner.size(); ++l) {
                size_t occupied = 0;
                for (size_t s = 0; s < slot_owner[l].size(); ++s) {
                    if (slot_owner[l][s] >= 0) {
                        ++occupied;
                    }
                }
                if (occupied == 0 || l >= pool_ext_stride.size()) {
                    continue;
                }
                size_t per_slot = 0;
                for (size_t k = 0; k < pool_ext_stride[l].size(); ++k) {
                    per_slot += pool_ext_stride[l][k];
                }
                resident_mib += (double) occupied * (double) per_slot / 1024.0 / 1024.0;
            }
        }
        fprintf(stderr, "llama_expert_cache: final stats: runtime requests=%zu hits=%zu misses=%zu (hit rate %.1f%%)  prewarm req=%zu hit=%zu miss=%zu  resident=%.2f MiB file_reads=%zu pread_usec=%llu fill_batch_usec=%llu fill_wait_us=%llu prefetch=%zu/%zu\n",
                n_requests, n_hits, n_misses,
                n_requests ? 100.0 * (double) n_hits / (double) n_requests : 0.0,
                n_prewarm_requests, n_prewarm_hits, n_prewarm_misses,
                resident_mib, n_reads.load(std::memory_order_relaxed),
                (unsigned long long) pread_usec.load(std::memory_order_relaxed),
                (unsigned long long) fill_batch_usec.load(std::memory_order_relaxed),
                (unsigned long long) fill_wait_us.load(std::memory_order_relaxed),
                n_prefetch, n_prefetch_dropped);
        // [CGC 2026-09-15] Pool integrity at teardown. The always-on mul_mat_id assertion in
        // ggml-metal-ops fires ~8x/run and ALWAYS as a gate+up pair on il=1 (pool slot 14 / slab
        // expert 214). That assertion reads at graph-BUILD time, so on its own it cannot separate
        //   (a) a benign transient -- the fill for that slot was still in flight when the graph
        //       was built, and the GPU read the completed region a moment later, from
        //   (b) a real fill bug -- the region was never written, or was written to a different
        //       slot, so the GPU really did multiply by zeros.
        // At teardown nothing is in flight and nothing is mid-remap, so a zero region here is (b)
        // by construction. Walking owner-SET slots only makes this cheap (no env gate needed).
        //
        // [CGC 2026-09-16] The "(b) by construction" inference rested on an unstated premise --
        // that a correctly filled region is never all-zero in its first 4 KiB. That premise does
        // NOT hold for this checkpoint: 10 of 31488 expert rows (measure them with
        // scripts/check/gguf_dead_expert_census.py) begin with 4592-13120 B of zeros while being
        // 17.8-47.0% non-zero overall, so the 4 KiB probe sits entirely inside a legitimately
        // zero prefix. Every `zero-regions` this scan has ever printed is one of those rows.
        // Hence the two-tier probe: 4 KiB first (cheap), and the FULL stride only when that
        // comes back all-zero. `zero-regions` now means "the region holds nothing, the fill did
        // not land" -- the property the scan was written to detect -- and the checkpoint's zero
        // prefixes are reported separately instead of being folded into it.
        if (!slot_owner.empty() && (!pool_ext.empty() || !pool.empty())) {
            size_t zero_slots = 0, checked_slots = 0, checked_bytes = 0;
            size_t prefix_only = 0;
            int    first_l = -1, first_s = -1, first_k = -1, first_e = -1;
            int    pref_l  = -1, pref_s  = -1, pref_k  = -1, pref_e  = -1;
            std::vector<size_t> zero_per_layer(slot_owner.size(), 0);
            std::vector<size_t> pref_per_layer(slot_owner.size(), 0);
            std::vector<size_t> chk_per_layer(slot_owner.size(), 0);
            for (size_t l = 0; l < slot_owner.size(); ++l) {
                const size_t nk = (l < pool_ext.size()) ? pool_ext[l].size()
                                 : (l < pool.size() ? pool[l].size() : 0);
                for (size_t s = 0; s < slot_owner[l].size(); ++s) {
                    const int32_t e = slot_owner[l][s];
                    if (e < 0) {
                        continue;
                    }
                    ++checked_slots;
                    ++chk_per_layer[l];
                    for (size_t k = 0; k < nk; ++k) {
                        const uint8_t * base = nullptr;
                        size_t stride = 0;
                        if (l < pool_ext.size() && k < pool_ext[l].size() && pool_ext[l][k] != nullptr) {
                            base   = pool_ext[l][k];
                            stride = (l < pool_ext_stride.size() && k < pool_ext_stride[l].size())
                                     ? pool_ext_stride[l][k] : 0;
                        } else if (l < pool.size() && k < pool[l].size() && !pool[l][k].empty()) {
                            stride = pool[l][k].size() / std::max<size_t>(slot_owner[l].size(), 1);
                            base   = pool[l][k].data();
                        }
                        if (base == nullptr || stride < 64) {
                            continue;
                        }
                        const uint8_t * row = base + (size_t) s * stride;
                        const size_t probe = std::min<size_t>(stride, 4096);
                        bool allzero = true;
                        for (size_t b = 0; b + 8 <= probe; b += 8) {
                            uint64_t w;
                            memcpy(&w, row + b, 8);
                            if (w != 0) { allzero = false; break; }
                        }
                        checked_bytes += probe;
                        if (!allzero) {
                            continue;
                        }
                        // All-zero over the probe. Confirm over the whole stride before calling it
                        // a dropped fill; a prefix-only region is the checkpoint talking.
                        bool whole_zero = true;
                        for (size_t b = probe; b + 8 <= stride; b += 8) {
                            uint64_t w;
                            memcpy(&w, row + b, 8);
                            if (w != 0) { whole_zero = false; break; }
                        }
                        checked_bytes += stride - probe;
                        if (whole_zero) {
                            ++zero_slots;
                            ++zero_per_layer[l];
                            if (first_l < 0) { first_l = (int) l; first_s = (int) s; first_k = (int) k; first_e = (int) e; }
                        } else {
                            ++prefix_only;
                            ++pref_per_layer[l];
                            if (pref_l < 0) { pref_l = (int) l; pref_s = (int) s; pref_k = (int) k; pref_e = (int) e; }
                        }
                    }
                }
            }
            fprintf(stderr, "llama_expert_cache: pool integrity: owner-set slots=%zu zero-regions=%zu "
                    "zero-prefix-only=%zu (probe<=4KiB, full stride on zero, %.1f MiB scanned)%s\n",
                    checked_slots, zero_slots, prefix_only, (double) checked_bytes / 1048576.0,
                    (zero_slots || prefix_only) ? "" : "  [OK: every resident slot holds non-zero bytes]");
            if (zero_slots) {
                fprintf(stderr, "llama_expert_cache:   first zero region: layer=%d slot=%d kind=%d owner_expert=%d\n",
                        first_l, first_s, first_k, first_e);
                for (size_t l = 0; l < zero_per_layer.size(); ++l) {
                    if (zero_per_layer[l]) {
                        fprintf(stderr, "llama_expert_cache:   layer=%zu zero=%zu of %zu resident slots\n",
                                l, zero_per_layer[l], chk_per_layer[l]);
                    }
                }
            }
            if (prefix_only) {
                fprintf(stderr, "llama_expert_cache:   zero-PREFIX only (not a defect; the file row starts with zeros): first layer=%d slot=%d kind=%d owner_expert=%d\n",
                        pref_l, pref_s, pref_k, pref_e);
                for (size_t l = 0; l < pref_per_layer.size(); ++l) {
                    if (pref_per_layer[l]) {
                        fprintf(stderr, "llama_expert_cache:   layer=%zu zero-prefix=%zu of %zu resident slots\n",
                                l, pref_per_layer[l], chk_per_layer[l]);
                    }
                }
            }
        }
        // [CGC M2 pool reuse 2026-09-14] What the whole-layer slab path actually read. The M2 exit
        // condition is "bytes/token @ chunk 2048 <= 3.0 MB (i.e. only the non-resident share)";
        // report it as the ratio so a claim about it can be checked instead of inferred.
        {
            const uint64_t sb_pool = n_slab_bytes_pool.load(std::memory_order_relaxed);
            const uint64_t sb_disk = n_slab_bytes_disk.load(std::memory_order_relaxed);
            if (sb_pool + sb_disk > 0) {
                fprintf(stderr, "llama_expert_cache: slab fills: pool=%.1f MiB disk=%.1f MiB "
                        "(non-resident share %.1f%%)\n",
                        (double) sb_pool / 1048576.0, (double) sb_disk / 1048576.0,
                        100.0 * (double) sb_disk / (double) (sb_pool + sb_disk));
            }
        }
        // [CGC miss attribution 2026-09-13] compulsory vs capacity, plus the per-layer check that
        // actually decides the lever. If a layer's distinct demanded experts fit inside its slot
        // count, no capacity miss is possible there once warm -> only locality / routing can
        // help and growing the pool is pure RSS cost. If distinct >> slots, the LRU is thrashing
        // and capacity IS the lever.
        if (!n_distinct_demanded.empty()) {
            const double miss_total = (double) (n_miss_compulsory + n_miss_capacity);
            uint32_t layers_over = 0, worst_l = 0, worst_d = 0, worst_ns = 0;
            for (size_t li = 0; li < n_distinct_demanded.size(); ++li) {
                const uint32_t d  = n_distinct_demanded[li];
                const uint32_t ns = li < slot_owner.size() ? (uint32_t) slot_owner[li].size() : 0;
                if (ns > 0 && d > ns) {
                    ++layers_over;
                }
                if (d > worst_d) { worst_d = d; worst_l = (uint32_t) li; worst_ns = ns; }
            }
            fprintf(stderr, "llama_expert_cache: miss attribution: compulsory=%zu capacity=%zu (%.1f%% / %.1f%% of %.0f)  evictions=%zu  layers_distinct_over_slots=%u  worst=layer %u distinct=%u slots=%u\n",
                    n_miss_compulsory, n_miss_capacity,
                    miss_total > 0.0 ? 100.0 * (double) n_miss_compulsory / miss_total : 0.0,
                    miss_total > 0.0 ? 100.0 * (double) n_miss_capacity / miss_total : 0.0,
                    miss_total, n_evictions, layers_over, worst_l, worst_d, worst_ns);
            if (getenv("LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS") != nullptr) {
                for (size_t li = 0; li < n_distinct_demanded.size(); ++li) {
                    fprintf(stderr, "llama_expert_cache:   miss-attr layer=%zu distinct=%u slots=%zu\n",
                            li, n_distinct_demanded[li], slot_owner[li].size());
                }
            }
        }
        // [CGC miss-path cost audit 2026-09-13] the number that decides device-vs-cache: the
        // mean run size actually fetched, and the effective rate it was fetched at.
        {
            const size_t rd_n = n_reads.load(std::memory_order_relaxed);
            const uint64_t rd_b = n_read_bytes.load(std::memory_order_relaxed);
            const uint64_t rd_u = pread_usec.load(std::memory_order_relaxed);
            if (rd_n > 0 && rd_u > 0) {
                fprintf(stderr, "llama_expert_cache: read shape: jobs=%zu bytes=%llu (%.2f MiB/job as one contiguous run)  "
                        "us/job=%.0f  effective_rate=%.0f MiB/s  total_bytes=%.2f GiB\n",
                        rd_n, (unsigned long long) rd_b, rd_n ? (double) rd_b / rd_n / 1048576.0 : 0.0,
                        (double) rd_u / rd_n, (double) rd_b / (double) rd_u,
                        (double) rd_b / 1073741824.0);
            }
        }
        // [CGC Prefetch Drop Audit 2026-09-07] per-reason drop breakdown. The legacy
        // n_prefetch_dropped only counted 2 of 12 drop points; this line shows the full
        // classification so replay/database/precommit can see the real loss breakdown.
        // Drop reasons #1-#12 correspond to the audit inventory; zeros mean that path
        // wasn't exercised (or the counter isn't instrumented yet — see header for status).
        if (n_prefetch_dropped > 0 || drop_stats.total() > 0) {
            fprintf(stderr,
                    "llama_expert_cache: prefetch drop breakdown (total=%zu): "
                    "#1 no_free_slot=%zu  #2 dbuf_cap_skip=%zu  #3 drain_cleared=%zu  "
                    "#4 zero_slot_fallback=%zu  #5 lru_evicted_predicted=%zu  "
                    "#6 bg_reassign_race=%zu  #7 guard_reject=%zu  "
                    "#8 dbuf2_scratch_invisible=%zu  #9 one_shot_consumed=%zu  "
                    "#10 fast_wait_expired=%zu  #11 trigger_too_late=%zu  #12 collect_skipped=%zu\n",
                    n_prefetch_dropped,
                    drop_stats.no_free_slot, drop_stats.dbuf_cap_skip, drop_stats.drain_cleared,
                    drop_stats.zero_slot_fallback, drop_stats.lru_evicted_predicted,
                    drop_stats.bg_reassign_race, drop_stats.guard_reject,
                    drop_stats.dbuf2_scratch_invisible, drop_stats.one_shot_consumed,
                    drop_stats.fast_wait_expired, drop_stats.trigger_too_late,
                    drop_stats.collect_skipped);
        }
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
        // [CGC verify-strict 2026-09-13] Both must be 0 in a healthy run. A nonzero
        // zero_mapped_selected means a selected expert was read from the reserved ZERO slot, i.e.
        // its weight contribution was dropped — the pool-size-dependent quality leak. Printed
        // unconditionally (not env-gated) so it can never be silent again.
        fprintf(stderr, "llama_expert_cache: verify-strict: refused=%zu  zero_mapped_selected=%zu%s\n",
                n_verify_strict_refused, n_zero_mapped_selected,
                n_zero_mapped_selected ? "  <-- QUALITY LEAK: selected expert read as zeros" : "");
        // [CGC 2026-09-15 §8.3 gate quantity] S1 slot-table health. Printed whenever the table was
        // published at all, so it is absent from runs that do not use S1 and can never be silent on
        // runs that do -- and "computed then never observed" is precisely the failure this line
        // exists to prevent: both publish sites used to discard the return value, and the site that
        // every MTP-off S1 arm actually takes was one of them.
        //
        // clamped MUST be 0. Nonzero means the GPU table and the host leaf have stopped being the
        // same mapping: the leaf writes -1 for a non-resident expert (loud -- the consumer reports
        // an out-of-range id) while the table clamps it to 0 (SILENT -- index 0 is legal, so
        // mul_mat_id reads another expert's weights and the answer is quietly wrong). S1 is
        // admissible only while this is 0; CGC_S1_CLAMP_ABORT=1 promotes it from a report to a hard
        // precondition.
        if (n_slot_table_publishes > 0) {
            char cgc_s1_note[192] = "";
            if (n_slot_table_clamped_selected) {
                snprintf(cgc_s1_note, sizeof(cgc_s1_note),
                        "  <-- CONSUMED IDS WERE CLAMPED: silent expert substitution");
            } else if (n_slot_table_consumed_changed + n_slot_table_consumed_same == 0) {
                snprintf(cgc_s1_note, sizeof(cgc_s1_note),
                        "  (consumed-subset churn not instrumented: set CGC_S1_TABLE_CHURN=1)");
            }
            fprintf(stderr, "llama_expert_cache: S1 slot-table: publishes=%zu clamped_selected=%zu"
                            " clamped_table=%zu changed_entries=%zu consumed_changed=%zu"
                            " consumed_unchanged_publishes=%zu%s\n",
                    n_slot_table_publishes, n_slot_table_clamped_selected, n_slot_table_clamped,
                    n_slot_table_changed, n_slot_table_consumed_changed,
                    n_slot_table_consumed_same, cgc_s1_note);
        }
        // [CGC MTP Draft Prefetch 2026-09-07] final stats: how many experts were queued for
        // prefetch from draft predictions, and how many of those predictions were actually selected
        // by the verify step (hit) vs wasted (miss). Hit rate should track draft_accept (~92-98%).
        // This is the exact-prediction counterpart to SpAc's frequency-based prefetch.
        if (n_draft_prefetch_queued > 0 || n_draft_prefetch_hit > 0 || n_draft_prefetch_miss > 0) {
            const size_t total = n_draft_prefetch_hit + n_draft_prefetch_miss;
            fprintf(stderr, "llama_expert_cache: MTP draft prefetch: queued=%zu experts  hit=%zu miss=%zu hitrate=%.1f%%\n",
                    n_draft_prefetch_queued, n_draft_prefetch_hit, n_draft_prefetch_miss,
                    total ? 100.0 * (double) n_draft_prefetch_hit / (double) total : 0.0);
        }
        // [CGC routing-aware placement §2/2 2026-08-29] dump per-layer top-K route-frequency
        // lists in PIN_PROFILE format (line i = layer i, space-separated expert ids, K = usable
        // slots for that layer, LAYER_CAPS-aware) so the next run can feed it back via
        // LLAMA_EXPERT_CACHE_PIN_PROFILE. Also prints the coverage verdict — the fraction of
        // route mass the top-K set covers. Capacity baseline for uniform routing is K/n_expert
        // (e.g. 71/128 = 55.5%); coverage far above baseline = heavy-tailed routing = the
        // placement lever is live; coverage near baseline = diffuse routing = lever closed.
        // Requires LLAMA_EXPERT_CACHE_ROUTE_RECORD=1 during the run (freq only fills then).
        {
            static const char * dump_path = []() {
                const char * p = getenv("LLAMA_EXPERT_CACHE_ROUTE_DUMP");
                return (p && p[0]) ? p : nullptr;
            }();
            if (dump_path != nullptr && !freq.empty()) {
                FILE * f = fopen(dump_path, "w");
                if (f != nullptr) {
                    double cov_sum = 0.0, cov_min = 1.0e9, cov_max = -1.0;
                    uint32_t cov_n = 0;
                    // [CGC 2026-09-13] K varies per layer once LAYER_CAPS is set, so the
                    // uniform-routing baseline is mean(K)/n_expert -- not a constant. The old
                    // text hard-coded "K/128", which on a 256-expert model halves the stated
                    // baseline and can make heavy-tailed routing read as diffuse (exactly the
                    // wrong conclusion, and the one that decides whether to keep optimising
                    // placement at all).
                    uint64_t slots_sum = 0;
                    for (uint32_t l = 0; l < freq.size(); ++l) {
                        uint64_t total = 0;
                        for (uint32_t e = 0; e < n_expert; ++e) {
                            total += freq[l][e];
                        }
                        if (total == 0) {
                            fprintf(f, "\n"); // keep line numbering = layer index
                            continue;
                        }
                        std::vector<uint32_t> order(n_expert);
                        for (uint32_t e = 0; e < n_expert; ++e) {
                            order[e] = e;
                        }
                        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                            const uint64_t fa = freq[l][a], fb = freq[l][b];
                            return fa != fb ? fa > fb : a < b;
                        });
                        const uint32_t k = std::min<uint32_t>(llama_expert_cache_usable_slots(this, l), n_expert);
                        uint64_t top = 0;
                        for (uint32_t i = 0; i < k; ++i) {
                            top += freq[l][order[i]];
                            fprintf(f, "%u%c", order[i], (i + 1 == k) ? '\n' : ' ');
                        }
                        if (k == 0) {
                            fprintf(f, "\n");
                        }
                        const double cov = (double) top / (double) total;
                        cov_sum += cov;
                        cov_n++;
                        slots_sum += k;
                        cov_min = std::min(cov_min, cov);
                        cov_max = std::max(cov_max, cov);
                    }
                    fclose(f);
                    const double base_pct = (cov_n && n_expert)
                        ? 100.0 * (double) slots_sum / (double) cov_n / (double) n_expert : 0.0;
                    fprintf(stderr, "llama_expert_cache: ROUTE-DUMP: %s written (%u layers with routes) coverage: mean=%.1f%% min=%.1f%% max=%.1f%%  (uniform-routing baseline = mean K/n_expert = %llu/%u = %.1f%%; coverage far above baseline = heavy-tailed routing, placement lever live)\n",
                            dump_path, cov_n,
                            cov_n ? 100.0 * cov_sum / cov_n : 0.0,
                            cov_n ? 100.0 * cov_min : 0.0,
                            cov_n ? 100.0 * cov_max : 0.0,
                            (unsigned long long) (cov_n ? slots_sum / cov_n : 0), (unsigned) n_expert,
                            base_pct);
                } else {
                    fprintf(stderr, "llama_expert_cache: ROUTE-DUMP: open failed: %s\n", dump_path);
                }
            }
        }
        // [CGC mass-coverage 2026-09-06] report when massc accumulation ran (env CGC_MASSCOV=1).
        // Prints (per layer + aggregates): (1) CURRENT-membership mass coverage (the live
        // slot_table) — how much routing mass the pool as-is serves resident; (2) counterfactual
        // top-K-by-mass coverage at several capacities — the ceiling if prewarm picked the K most
        // massive experts. n_slots = real per-layer capacity used by the current run.
        {
            const bool mc_on = getenv("CGC_MASSCOV") != nullptr;
            if (mc_on && !massc_mass.empty()) {
                const uint32_t k_run = llama_expert_cache_slots_per_layer(this);
                fprintf(stderr, "llama_expert_cache: MASSCOV summary (per-layer routing MASS, not counts):\n");
                static const uint32_t ks[] = { k_run, 96u, 128u, 192u, 256u };
                const uint32_t n_ex = n_expert;
                double cur_sum = 0.0, cur_min = 1.0, cur_max = 0.0;
                uint32_t cur_n = 0;
                // [CGC count-cold 2026-09-07] selected-id count-cold aggregates (sel_cold/sel_total)
                double sc_sum = 0.0, sc_min = 1.0, sc_max = 0.0;
                uint32_t sc_n = 0;
                double cov[5] = {0,0,0,0,0}, cmin[5] = {1,1,1,1,1}, cmax[5] = {0,0,0,0,0};
                uint32_t cov_n[5] = {0,0,0,0,0};
                for (uint32_t l = 0; l < massc_mass.size() && l < massc_total.size(); ++l) {
                    if (massc_total[l] <= 0.0) {
                        continue;
                    }
                    const double cur = massc_cur_cov[l] / massc_total[l];
                    cur_sum += cur; cur_min = std::min(cur_min, cur); cur_max = std::max(cur_max, cur); cur_n++;
                    if (l < massc_sel_total.size() && massc_sel_total[l] > 0) {
                        const double sc = (double) massc_sel_cold[l] / (double) massc_sel_total[l];
                        sc_sum += sc; sc_min = std::min(sc_min, sc); sc_max = std::max(sc_max, sc); sc_n++;
                    }
                    std::vector<uint32_t> order(n_ex);
                    for (uint32_t e = 0; e < n_ex; ++e) {
                        order[e] = e;
                    }
                    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                        return massc_mass[l][a] != massc_mass[l][b] ? massc_mass[l][a] > massc_mass[l][b] : a < b;
                    });
                    for (int ki = 0; ki < 5; ++ki) {
                        const uint32_t k = std::min(ks[ki], n_ex);
                        double top = 0.0;
                        for (uint32_t i = 0; i < k; ++i) {
                            top += massc_mass[l][order[i]];
                        }
                        const double c = top / massc_total[l];
                        cov[ki] += c; cmin[ki] = std::min(cmin[ki], c); cmax[ki] = std::max(cmax[ki], c); cov_n[ki]++;
                    }
                }
                if (cur_n > 0) {
                    fprintf(stderr, "  CURRENT membership (run slots=%u): mass coverage mean=%.1f%% min=%.1f%% max=%.1f%%  -> the pool TODAY serves this much of the model's routing mass resident\n",
                            k_run, 100.0 * cur_sum / cur_n, 100.0 * cur_min, 100.0 * cur_max);
                    if (sc_n > 0) {
                        fprintf(stderr, "  SELECTED count-cold (of top-k expert ids, not mass): mean=%.1f%% min=%.1f%% max=%.1f%%  -> this share of the step's SELECTED experts was non-resident (ZERO-mapped / guard-tripped)\n",
                                100.0 * sc_sum / sc_n, 100.0 * sc_min, 100.0 * sc_max);
                    }
                    for (int ki = 0; ki < 5; ++ki) {
                        fprintf(stderr, "  COUNTERFACTUAL top-K by mass: K=%3u  mass coverage mean=%.1f%% min=%.1f%% max=%.1f%%\n",
                                ks[ki], 100.0 * cov[ki] / cov_n[ki], 100.0 * cmin[ki], 100.0 * cmax[ki]);
                    }
                    // [CGC 2026-09-13] print the number, not just the formula: this run's live
                    // per-layer capacity over the model's real expert count.
                    fprintf(stderr, "  (uniform-routing mass baseline = K/n_expert = %u/%u = %.1f%%; per-layer detail in MASSCOV file)\n",
                            (unsigned) k_run, (unsigned) n_ex,
                            n_ex ? 100.0 * (double) k_run / (double) n_ex : 0.0);
                }
                const char * mp = getenv("CGC_MASSCOV_DUMP");
                if (mp && mp[0]) {
                    FILE * mf = fopen(mp, "w");
                    if (mf != nullptr) {
                        for (uint32_t l = 0; l < massc_mass.size(); ++l) {
                            fprintf(mf, "layer %u total=%.4f cur=%.4f", l, massc_total[l],
                                    massc_total[l] > 0 ? massc_cur_cov[l] / massc_total[l] : 0.0);
                            if (l < massc_sel_total.size() && massc_sel_total[l] > 0) {
                                fprintf(mf, " selcold=%.4f sel=%llu",
                                        (double) massc_sel_cold[l] / (double) massc_sel_total[l],
                                        (unsigned long long) massc_sel_total[l]);
                            }
                            if (massc_total[l] > 0.0) {
                                std::vector<uint32_t> order(n_ex);
                                for (uint32_t e = 0; e < n_ex; ++e) {
                                    order[e] = e;
                                }
                                std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                                    return massc_mass[l][a] != massc_mass[l][b] ? massc_mass[l][a] > massc_mass[l][b] : a < b;
                                });
                                for (int ki = 0; ki < 5; ++ki) {
                                    const uint32_t k = std::min(ks[ki], n_ex);
                                    double top = 0.0;
                                    for (uint32_t i = 0; i < k; ++i) {
                                        top += massc_mass[l][order[i]];
                                    }
                                    fprintf(mf, " k%d=%.4f", ks[ki], massc_total[l] > 0 ? top / massc_total[l] : 0.0);
                                }
                            }
                            fprintf(mf, "\n");
                        }
                        fclose(mf);
                        fprintf(stderr, "llama_expert_cache: MASSCOV file: %s\n", mp);
                    }
                }
            }
        }
        // [CGC routing-aware placement] static-pin telemetry (only printed when a profile ran)
        if (n_pin_marked > 0 || n_pin_yield > 0) {
            fprintf(stderr, "llama_expert_cache: routing-aware placement: pin_marked=%zu pin_yield(evicted)=%zu\n",
                    n_pin_marked, n_pin_yield);
        }
        // [CGC P1 prefill-protect 2026-09-13] engagement telemetry, printed at teardown.
        // defer_skip = victim choices the rule changed; defer_yield = fills that still had to
        // evict a decode-stamped slot (overflow pass 2, i.e. the pool was genuinely too small).
        if (n_defer_skip > 0 || n_prefill_defer_yield > 0) {
            fprintf(stderr, "llama_expert_cache: prefill-protect: defer_skip=%llu defer_yield(overflow)=%llu\n",
                    (unsigned long long) n_defer_skip, (unsigned long long) n_prefill_defer_yield);
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
    // CGC M1 work item 1 (CGC_POOL_SPLIT): the pool owns its allocations here (one per kind). Empty
    // on every other path, where the pool regions are the expert tensors' own storage.
    for (ggml_backend_buffer_t b : pool_owned_bufs) {
        if (b != nullptr) {
            ggml_backend_buffer_free(b);
        }
    }
    pool_owned_bufs.clear();
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
                    n_prefetch_dropped++;
                    drop_stats.bg_reassign_race++;  // #6: bg_loop reassign-race — sync fill grabbed slot between queue and pickup
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
                // [CGC DBUF2 2026-09-06] when full double-buffering is ON, async bg fills publish
                // to the SCRATCH table (never active), so they can never tear a buffer the decode
                // step is reading. The scratch becomes active at the next dbuf2_swap() (called at
                // the GPU-idle step boundary). CGC_DBUF2=0 = legacy: publish directly to active.
                if (cgc_dbuf2_on()) {
                    slot_table_scratch[(size_t) layer * n_expert + expert] = slot;
                } else {
                    slot_table[(size_t) layer * n_expert + expert] = slot;
                }
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
            n_read_bytes.fetch_add((uint64_t) job.bytes, std::memory_order_relaxed);
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
            n_read_bytes.fetch_add((uint64_t) job.bytes, std::memory_order_relaxed);
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
    // [CGC V1 verify 2026-09-05] byte-identity check after the pool-worker batch finishes
    cgc_exact_cache_verify_post_fill(cache, segs, dsts);
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
    g_rig_cache = cache;  // [A/B rig] snapshot target (see refresh_prefill_protect)
    cache->index      = idx;
    cache->index_size = nidx;
    cache->budget     = budget_bytes;

    // open one file handle per distinct file_idx (split models not yet supported)
    uint32_t max_idx = 0;
    for (size_t i = 0; i < nidx; ++i) {
        max_idx = std::max(max_idx, idx[i].file_idx);
    }
    cache->files.assign(max_idx + 1, nullptr);
    cache->files_path.assign(max_idx + 1, std::string());
    std::vector<int> opened(max_idx + 1, 0);
    for (size_t i = 0; i < nidx; ++i) {
        const uint32_t f = idx[i].file_idx;
        if (!opened[f]) {
            opened[f] = 1;
            if (f == 0) {
                cache->files[f] = fopen(model->expert_cache_path.c_str(), "rb");
                if (cache->files[f] != nullptr) {
                    cache->files_path[f] = model->expert_cache_path;
                }
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
        // [CGC DBUF2 2026-09-06] scratch slot table (same size, -1 init). Unused when
        // CGC_DBUF2=0; when ON, bg fills write here while decode reads slot_table (active).
        cache->slot_table_scratch.assign((size_t) max_layer * max_expert, -1);
        // [CGC Hybrid 2026-09-05] Soft Pool tier partition: read CGC_SOFT_POOL_L0 / L1 once,
        // store on cache for runtime introspection. Clamped to slots_l(layer) per layer
        // (see for-loop below). Default 32/32 per docs/CGC_EXPERT_CACHE_HYBRID_DESIGN §2.2.
        // L0+L1=0 keeps the legacy uniform n_slots behavior (no L0/L1 split in pick_slot).
        cache->soft_pool_l0 = cgc_soft_pool_tier("CGC_SOFT_POOL_L0", 32);
        cache->soft_pool_l1 = cgc_soft_pool_tier("CGC_SOFT_POOL_L1", 32);
        if (cache->soft_pool_l0 + cache->soft_pool_l1 > cache->n_slots) {
            fprintf(stderr, "CGC Soft Pool: L0(%u) + L1(%u) = %u > n_slots(%u); clamping to n_slots\n",
                    cache->soft_pool_l0, cache->soft_pool_l1,
                    cache->soft_pool_l0 + cache->soft_pool_l1, cache->n_slots);
            const uint64_t total = cache->soft_pool_l0 + cache->soft_pool_l1;
            if (total > 0) {
                // preserve ratio when clamping
                cache->soft_pool_l0 = (uint32_t) ((uint64_t) cache->soft_pool_l0 * cache->n_slots / total);
                cache->soft_pool_l1 = cache->n_slots - cache->soft_pool_l0;
            } else {
                cache->soft_pool_l0 = 0;
                cache->soft_pool_l1 = 0;
            }
        }
        fprintf(stderr, "CGC Soft Pool init: L0=%u L1=%u (n_slots=%u, partition %s)\n",
                cache->soft_pool_l0, cache->soft_pool_l1, cache->n_slots,
                (cache->soft_pool_l0 + cache->soft_pool_l1) > 0 ? "active" : "disabled");
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
        cache->cgc_weighted_cold_ratio.assign(max_layer, 1e9); // [CGC Step-2] sentinel = unset (logits callback hasn't fired this step). Consumed by expert_cache_on_topk CGC_FAST_COLD_MAX: >0 = weighted, ==1e9 = use count fallback.
        cache->slot_owner.resize(max_layer);
        cache->slot_last_use.resize(max_layer);
        cache->slot_queued.resize(max_layer);
        cache->slot_loading.resize(max_layer);
        cache->slot_pinned.resize(max_layer);
        cache->slot_pinned_static.resize(max_layer);
        cache->slot_decode_reserved.resize(max_layer);
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
            cache->slot_decode_reserved[l].assign(ns, 0);
        }
        // [CGC miss attribution 2026-09-13] session-long demand bitmaps (41 x 256 bytes here).
        cache->ever_loaded.assign(max_layer, std::vector<uint8_t>(cache->n_expert, 0));
        cache->n_distinct_demanded.assign(max_layer, 0);
        if (getenv("LLAMA_EXPERT_CACHE_PREFETCH_DBG") != nullptr) {
            unsigned busy0 = 0;
            for (uint32_t l = 0; l < max_layer; ++l)
                for (uint32_t i = 0; i < slots_l(cache, l); ++i)
                    if (cache->slot_owner[l][i] >= 0) busy0++;
            fprintf(stderr, "PFDBG init: n_layer=%u n_slots=%u busy=%u\n", max_layer, cache->n_slots, busy0);
        }
        cache->freq.assign(max_layer, std::vector<uint64_t>(cache->n_expert, 0));
        cache->massc_mass.assign(max_layer, std::vector<double>(cache->n_expert, 0.0));
        cache->massc_total.assign(max_layer, 0.0);
        cache->massc_cur_cov.assign(max_layer, 0.0);
        cache->massc_sel_total.assign(max_layer, 0);
        cache->massc_sel_cold.assign(max_layer, 0);
        // [CGC SpAc 2026-09-06] EMA utility, seeded 0.5 (profile-like warm start — an expert
        // that never routed yet still outranks a decayed-to-0 newcomer on refresh #1, mirroring
        // MoESpAcEstimator's seed so the first pool re-target does not degenerate to expert-id
        // ordering). Sized eagerly (cheap: ~41 x 256 doubles) so spac_update needs no resize.
        cache->spac_util.assign(max_layer, std::vector<double>(cache->n_expert, 0.5));
        // [CGC MTP Draft Prefetch 2026-09-07] draft-predicted expert ids per layer, collected
        // during MTP draft decode (ctx_type == MTP, 1-token) and consumed during trunk verify
        // decode (ctx_type == DEFAULT, multi-token) at il==1. draft_prefetch_valid marks layers
        // that have fresh predictions from the most recent draft step. Sized eagerly (cheap:
        // ~41 x 8 uint32 + 41 bool) so the collect phase needs no resize.
        cache->draft_prefetch_ids.assign(max_layer, std::vector<uint32_t>());
        cache->draft_prefetch_valid.assign(max_layer, false);
        // [CGC prev-token prefetch 2026-09-08] double-buffered per-layer expert ids for prev-token
        // prediction. prev is used for prefetch at il==1; curr collects the current token's ids;
        // swapped at the trigger boundary. Sized eagerly (~41 x 8 uint32 x 2 + 41 bool).
        cache->prev_token_expert_ids.assign(max_layer, std::vector<uint32_t>());
        cache->curr_token_expert_ids.assign(max_layer, std::vector<uint32_t>());
        cache->prev_token_valid.assign(max_layer, false);
        cache->pool.assign(max_layer, std::vector<std::vector<uint8_t>>(4));
        cache->pool_ext.assign(max_layer, std::vector<const uint8_t *>(4, nullptr));
        cache->pool_ext_stride.assign(max_layer, std::vector<size_t>(4, 0));
        cache->pool_ext_slots.assign(max_layer, std::vector<uint32_t>(4, 0));
        cache->pool_ext_buf.assign(max_layer, std::vector<ggml_backend_buffer_t>(4, nullptr));
        cache->pool_wide_data.assign(max_layer, std::vector<void *>(4, nullptr));
        cache->pool_wide_buf.assign(max_layer, std::vector<ggml_backend_buffer_t>(4, nullptr));
        cache->pool_wide_ne2.assign(max_layer, std::vector<int64_t>(4, 0));
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
            // [CGC per-layer min 2026-09-11] The DECODE hook takes the pool path only when the
            // per-ubatch expert union fits THAT LAYER's cap, so the binding number is the
            // minimum over layers -- not n_slots and not the average. Reporting it lets the
            // evaluation harness decide arithmetically whether a given CGC_POOL_MAX_TOKENS can
            // even reach the L3-B gather path (cap*topk <= min_slots - 1), instead of relying
            // on a particular prompt happening to route into an overflowing layer and print
            // `buffer is nil`. min is only meaningful over layers that own expert regions.
            uint32_t min_slots_l = UINT32_MAX;
            for (uint32_t l = 0; l < max_layer; ++l) {
                const uint32_t s = slots_l(cache, l);
                tot_slots += s;
                if (s > 0 && s < min_slots_l) {
                    min_slots_l = s;
                }
            }
            fprintf(stderr, "llama_expert_cache: LAYER_CAPS per-layer caps: total %llu slots (avg %.1f/layer, min %u/layer)\n",
                    (unsigned long long) tot_slots, (double) tot_slots / max_layer,
                    min_slots_l == UINT32_MAX ? 0 : min_slots_l);
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

// [CGC M2 whole-layer streaming 2026-09-14] Read an entire layer's experts (0..n_expert-1) for
// one kind straight from the GGUF file into a caller-provided slab. Unlike fill() (which copies
// from already-resident cache blobs), this reads directly from disk — the slab is transient and
// the pool is bypassed entirely. Used by the prefill streaming path: at chunk 2048 the union is
// provably saturated (2048 * top_k 8 = 16384 draws over 256 experts -> ~100% coverage), so
// reading ALL experts is both necessary and predictable (no prediction needed).
//
// [CGC M2 pool reuse 2026-09-14] byte accounting for the slab path (defined below the function).
static void record_slab_bytes(llama_expert_cache * cache, uint32_t layer, int kind, uint32_t ne,
        int64_t pool_bytes, int64_t disk_bytes, size_t stride, int fast);

// Layout: expert e's bytes for `kind` land at dst + e * stride. The caller sets ne[2] = n_expert
// and the remap ids are raw expert ids (0..n_expert-1), so mul_mat_id indexes directly into this
// slab with no slot-indirection layer.
//
// [CGC M2 pool reuse 2026-09-14] Experts the pool already holds are memcpy'd from the pool region
// instead of being read from the file again. This is the difference between 13.9 GiB per ubatch
// and the 44% that is genuinely non-resident, and the slab path is I/O-bound (measured: fill
// ~125 ms/layer = 287 MB at ~2.3 GB/s, i.e. the whole per-ubatch fixed cost), so the saving is
// close to the byte ratio. The bytes are the same bytes: the pool is filled from these very file
// segments, so a pool-served slab is bit-identical by construction -- which is also what the M2
// oracle gate asserts after this change.
//
// Returns total bytes read+served, or -1 on any short read (the caller should then fall back to
// the pool path rather than trust a partial slab).
int64_t llama_expert_cache_fill_layer_slab(llama_expert_cache * cache, uint32_t layer,
        int kind, void * dst, size_t stride) {
    if (cache == nullptr || dst == nullptr || layer >= cache->slot_owner.size()) {
        return -1;
    }
    if (kind < 0 || kind >= 4) {
        return -1;
    }
    const uint32_t ne = cache->n_expert;
    if (ne == 0) {
        return 0;
    }
    std::vector<llama_expert_cache::segment> segs;
    std::vector<uint8_t *> dsts;
    segs.reserve((size_t) ne);
    dsts.reserve((size_t) ne);
    uint8_t * out = (uint8_t *) dst;
    int64_t total = 0;
    // [CGC M2 pool reuse 2026-09-14] Read the residency state BEFORE taking the cache lock: these
    // accessors are pure readers of immutable-after-init geometry, and calling them inside would
    // re-enter cache->m.
    const int32_t * resident    = llama_expert_cache_slot_table(cache, layer);
    const uint32_t  usable      = llama_expert_cache_usable_slots(cache, layer);
    const uint8_t * pool_base   = llama_expert_cache_pool_data(cache, layer, kind);
    const size_t    pool_stride = llama_expert_cache_pool_stride(cache, layer, kind);
    int64_t pool_bytes = 0;
    int64_t disk_bytes = 0;
    // Collect every expert's segment for this kind. One lock for the whole traversal — the
    // index/key_segs are immutable after init, so this is a read-only walk and could be
    // lock-free, but taking the lock matches every other index reader and costs nothing vs IO.
    {
        std::lock_guard<std::mutex> lk(cache->m);
        for (uint32_t e = 0; e < ne; ++e) {
            const uint64_t key = make_key(layer, e);
            auto it = cache->key_segs.find(key);
            if (it == cache->key_segs.end()) {
                continue; // expert has no tensor for this kind (e.g. kind 3 absent)
            }
            const llama_expert_index_entry * idx = nullptr;
            for (uint32_t pos : it->second) {
                if (cache->index[pos].kind == kind) {
                    idx = &cache->index[pos];
                    break;
                }
            }
            if (idx == nullptr) {
                continue;
            }
            const uint32_t nb = (uint32_t) idx->bytes;
            // Resident? Then the pool already holds this expert's file bytes verbatim and the copy
            // is the cheapest possible fill. Three guards, each load-bearing:
            //   * slot < usable: the reserved ZERO slot (last slot, MTP mode) contains zeros, not
            //     this expert's bytes -- copying it would silently zero an expert out.
            //   * pool_stride == nb: source pitch must equal the expert's size, or `slot*stride`
            //     walks into the neighbouring slot's bytes.
            //   * nb == stride: the destination pitch must equal the expert's size, or the reader
            //     (mul_mat_id walks nb[2] = the expert's own size) reads a different pitch. When
            //     any guard fails we fall through to the file, which is always correct.
            const int32_t slot = resident != nullptr ? resident[e] : -1;
            if (pool_base != nullptr && slot >= 0 && (uint32_t) slot < usable &&
                    pool_stride == (size_t) nb && (size_t) nb == stride) {
                memcpy(out + (size_t) e * stride, pool_base + (size_t) slot * pool_stride, nb);
                pool_bytes += nb;
                total += nb;
                continue;
            }
            llama_expert_cache::segment s;
            s.kind        = idx->kind;
            s.file_idx    = idx->file_idx;
            s.file_offset = idx->file_offset;
            s.off         = 0;
            s.bytes       = nb;
            segs.push_back(s);
            dsts.push_back(out + (size_t) e * stride);
            disk_bytes += nb;
            total += nb;
        }
    }
    if (segs.empty()) {
        return 0;
    }
    // [M2 optimization 2026-09-14] Fast path: if all experts for this (layer, kind) are
    // contiguous in the file (same file_idx, consecutive offsets, same bytes), do ONE large
    // pread instead of N small ones. This is the case for standard GGUF MoE layouts and
    // cuts slab fill from ~150ms to ~10ms (15x).
    // [CGC M2 pool reuse 2026-09-14] `segs.size() == ne` is required as well: this path reads the
    // file's contiguous run STRAIGHT into dst (it never uses `stride`), which is only the correct
    // layout when every one of the ne experts came from the file in order. With pool reuse some
    // experts are already in place via memcpy, so the linear assumption has to be given up.
    bool contiguous = (segs.size() == ne && segs.size() > 1);
    if (contiguous) {
        const uint32_t fidx = segs[0].file_idx;
        const uint32_t bsz  = segs[0].bytes;
        for (size_t i = 1; i < segs.size(); ++i) {
            if (segs[i].file_idx != fidx ||
                segs[i].bytes != bsz ||
                segs[i].file_offset != segs[i-1].file_offset + segs[i-1].bytes) {
                contiguous = false;
                break;
            }
        }
    }
    // [M2 profiling 2026-09-14] measure slab fill time
    static const bool m2_profile = getenv("CGC_M2_PROFILE") != nullptr;
    uint64_t t0 = 0;
    if (m2_profile) {
        t0 = ggml_time_us();
    }
    if (contiguous) {
        // One big read: segs[0].file_offset .. segs[0].file_offset + total
        FILE * f = cache->files.at(segs[0].file_idx);
        const ssize_t rd = pread(fileno(f), dst, (size_t) total, (off_t) segs[0].file_offset);
        if (rd != (ssize_t) total) {
            fprintf(stderr, "CGC-M2-FAST-READ: short read layer=%u kind=%d expected=%lld got=%zd\n",
                    layer, kind, (long long) total, rd);
            return -1;
        }
        if (m2_profile) {
            uint64_t dt = ggml_time_us() - t0;
            static int fast_cnt = 0;
            if (fast_cnt < 200) {
                fast_cnt++;
                fprintf(stderr, "CGC-M2-PROF fill_slab_FAST layer=%u kind=%d experts=%zu bytes=%lld time_ms=%.2f\n",
                        layer, kind, segs.size(), (long long) total, (double) dt / 1000.0);
            }
        }
        record_slab_bytes(cache, layer, kind, ne, pool_bytes, disk_bytes, stride, 1);
        return total;
    }
    if (!fill_segments_concurrent(cache, segs, dsts)) {
        return -1;
    }
    if (m2_profile) {
        uint64_t dt = ggml_time_us() - t0;
        static int prof_cnt = 0;
        if (prof_cnt < 200) {
            prof_cnt++;
            fprintf(stderr, "CGC-M2-PROF fill_slab layer=%u kind=%d experts=%zu bytes=%lld time_ms=%.2f\n",
                    layer, kind, segs.size(), (long long) total, (double) dt / 1000.0);
        }
    }
    record_slab_bytes(cache, layer, kind, ne, pool_bytes, disk_bytes, stride, 0);
    return total;
}

// [CGC M2 pool reuse 2026-09-14] One line per fill under CGC_M2_PROFILE, plus process-wide
// counters that the stats printer reports at shutdown. bytes/token is the M2 exit condition and
// the only way to evaluate it is to know which half of the layer came from the pool.
static void record_slab_bytes(llama_expert_cache * cache, uint32_t layer, int kind, uint32_t ne,
        int64_t pool_bytes, int64_t disk_bytes, size_t stride, int fast) {
    cache->n_slab_bytes_pool.fetch_add((uint64_t) pool_bytes, std::memory_order_relaxed);
    cache->n_slab_bytes_disk.fetch_add((uint64_t) disk_bytes, std::memory_order_relaxed);
    static const bool m2_fill_log = getenv("CGC_M2_PROFILE") != nullptr;
    static std::atomic<int> logged{0};
    if (m2_fill_log && logged.fetch_add(1) < 400) {
        fprintf(stderr, "CGC-M2-FILL layer=%u kind=%d experts=%u pool_bytes=%lld disk_bytes=%lld "
                "stride=%zu fast=%d\n",
                layer, kind, ne, (long long) pool_bytes, (long long) disk_bytes, stride, fast);
    }
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
