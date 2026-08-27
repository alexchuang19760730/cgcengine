// L2 expert cache (bounded residency) verification:
//   - byte-identity: cache-fill bytes == direct pread from the file at the L1 index's absolute offset
//   - warm hits / miss accounting
//   - background prefetch (ensure() after prefetch() must return 0 misses)
//   - LRU eviction with a tight budget (evicted entry re-reads, then hits)
//
// usage: test-expert-cache <model.gguf> [budget_mb]

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unistd.h>

static int g_fail = 0;

static void check(bool ok, const char * msg) {
    fprintf(stderr, "%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) {
        g_fail++;
    }
}

// reference: direct pread from the file at the index's absolute offset
static bool read_ref(const char * path, const llama_expert_index_entry & e, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    out.resize(e.bytes);
    const ssize_t rd = pread(fileno(f), out.data(), e.bytes, (off_t) e.file_offset);
    fclose(f);
    return rd == (ssize_t) e.bytes;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [budget_mb]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const size_t budget = argc > 2 ? (size_t) atoi(argv[2]) * 1024 * 1024 : 256ull * 1024 * 1024;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.expert_cache_bytes = 1; // L1: build the per-expert offset index
    mparams.n_gpu_layers = 0;       // CPU-only load (fast; no Metal dependency for this test)

    llama_model * model = llama_model_load_from_file(path, mparams);
    check(model != nullptr, "model load");
    if (!model) {
        return 1;
    }

    const size_t nidx = llama_model_expert_index_size(model);
    const llama_expert_index_entry * idx = llama_model_expert_index(model);
    check(nidx > 0, "expert index built");
    fprintf(stderr, "  index entries: %zu\n", nidx);
    if (nidx == 0) {
        llama_model_free(model);
        return 1;
    }

    bool has_kind[4] = { false, false, false, false };
    for (size_t i = 0; i < nidx; ++i) {
        if (idx[i].kind >= 0 && idx[i].kind < 4) {
            has_kind[idx[i].kind] = true;
        }
    }

    llama_expert_cache * cache = llama_expert_cache_init(model, budget);
    check(cache != nullptr, "cache init");
    if (!cache) {
        llama_model_free(model);
        return 1;
    }

    // ---- 1. byte-identity: ensure + fill == direct file read ----
    const uint32_t n_layer = (uint32_t) llama_model_n_layer(model);
    std::vector<uint32_t> layers  = { 0, 0, 0, 0, 0, 0, 0, 0, n_layer - 1, n_layer - 1, n_layer - 1, n_layer - 1 };
    std::vector<uint32_t> experts = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
    const size_t nsel = layers.size();
    const size_t k0   = 8; // first group: layer 0, experts 0..7

    size_t misses = llama_expert_cache_ensure(cache, layers.data(), experts.data(), nsel);
    check(misses == nsel, "ensure: all misses on cold cache");

    for (int kind = 0; kind < 4; ++kind) {
        if (!has_kind[kind]) {
            continue;
        }
        size_t seg_bytes = 0;
        for (size_t i = 0; i < nidx; ++i) {
            if (idx[i].kind == kind) {
                seg_bytes = idx[i].bytes;
                break;
            }
        }
        if (seg_bytes == 0) {
            continue;
        }

        std::vector<uint8_t> dst(k0 * seg_bytes, 0xAB);
        const int64_t n = llama_expert_cache_fill(cache, 0, experts.data(), k0, kind, dst.data(), 0);
        char msg[128];
        snprintf(msg, sizeof msg, "fill: %zu experts of kind %d (bytes=%zd)", k0, kind, n);
        check(n == (int64_t) k0 * (int64_t) seg_bytes, msg);

        bool ok_all = true;
        for (size_t i = 0; i < k0; ++i) {
            const llama_expert_index_entry * e = nullptr;
            for (size_t j = 0; j < nidx; ++j) {
                if (idx[j].layer == 0 && idx[j].expert == experts[i] && idx[j].kind == kind) {
                    e = &idx[j];
                    break;
                }
            }
            if (!e) {
                ok_all = false;
                continue;
            }
            std::vector<uint8_t> ref;
            if (!read_ref(path, *e, ref)) {
                ok_all = false;
                continue;
            }
            if (memcmp(dst.data() + i * seg_bytes, ref.data(), seg_bytes) != 0) {
                ok_all = false;
            }
        }
        snprintf(msg, sizeof msg, "byte-identity kind=%d (layer 0, experts 0..%zu)", kind, k0 - 1);
        check(ok_all, msg);
    }

    // ---- 2. warm cache: all hits ----
    size_t misses2 = llama_expert_cache_ensure(cache, layers.data(), experts.data(), nsel);
    check(misses2 == 0, "ensure: warm cache all hits");

    size_t req, hits, mis, res, reads;
    llama_expert_cache_get_stats(cache, &req, &hits, &mis, &res, &reads);
    fprintf(stderr, "  stats: requests=%zu hits=%zu misses=%zu resident=%zuB reads=%zu\n", req, hits, mis, res, reads);

    // ---- 3. background prefetch: ensure() right after prefetch() must block-wait and return 0 misses ----
    const size_t nsel2 = 6;
    std::vector<uint32_t> layers2(nsel2, 1);
    std::vector<uint32_t> experts2 = { 100, 101, 102, 103, 104, 105 };
    llama_expert_cache_prefetch(cache, layers2.data(), experts2.data(), nsel2);
    size_t misses3 = llama_expert_cache_ensure(cache, layers2.data(), experts2.data(), nsel2);
    check(misses3 == 0, "prefetch: ensure after prefetch is a hit (bg fill)");

    // ---- 4. LRU eviction with a tight budget (room for ~1 blob) ----
    size_t blob0 = 0;
    for (size_t j = 0; j < nidx; ++j) {
        if (idx[j].layer == 0 && idx[j].expert == 0) {
            blob0 += idx[j].bytes;
        }
    }
    llama_expert_cache * small = llama_expert_cache_init(model, blob0 + 4096);
    check(small != nullptr, "small-budget cache init");
    if (small) {
        std::vector<uint32_t> l1 = { 0 }, e1 = { 0 };
        std::vector<uint32_t> l2 = { 1 }, e2 = { 0 };
        llama_expert_cache_ensure(small, l1.data(), e1.data(), 1); // miss (fills blob0)
        llama_expert_cache_ensure(small, l2.data(), e2.data(), 1); // miss (evicts layer-0 blob)
        size_t m4 = llama_expert_cache_ensure(small, l1.data(), e1.data(), 1); // evicted -> miss again
        check(m4 == 1, "LRU: evicted entry re-reads (miss)");
        size_t m5 = llama_expert_cache_ensure(small, l1.data(), e1.data(), 1); // now resident
        check(m5 == 0, "LRU: after re-fill, hit");
        llama_expert_cache_free(small);
    }

    llama_expert_cache_free(cache);
    llama_model_free(model);

    if (g_fail == 0) {
        fprintf(stderr, "ALL TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "%d TEST(S) FAILED\n", g_fail);
    return 1;
}
