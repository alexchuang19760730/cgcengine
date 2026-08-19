#include "cgc_expert_streamer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#ifdef _WIN32
#pragma comment(lib, "kernel32.lib")
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>
#endif

static uint64_t now_nanos(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int freq_init = 0;
    if (!freq_init) {
        QueryPerformanceFrequency(&freq);
        freq_init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static void* cgc_aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

static void cgc_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int find_slot(cgc_expert_streamer_t* s, int expert_id) {
    for (int i = 0; i < s->slot_count; i++) {
        if (s->slot_expert[i] == expert_id) return i;
    }
    return -1;
}

static int evict_slot(cgc_expert_streamer_t* s) {
    int victim = -1;
    int min_use = 0x7FFFFFFF;

    for (int i = 0; i < s->slot_count; i++) {
        if (s->slot_pinned[i]) continue;
        if (s->slot_expert[i] == -1) return i;
        if (s->slot_last_use[i] < min_use) {
            min_use = s->slot_last_use[i];
            victim = i;
        }
    }

    if (victim >= 0) {
        s->slot_expert[victim] = -1;
        s->slot_owner_phase[victim] = CGC_CACHE_SLOT_UNASSIGNED;
        s->slot_hit_count[victim] = 0;
        s->slot_last_use[victim] = 0;
        s->total_evictions++;
    }
    return victim;
}

static int allocate_slot(cgc_expert_streamer_t* s, const cgc_cache_access_ctx_t* ctx) {
    int slot = evict_slot(s);
    if (slot >= 0 && ctx) {
        s->slot_owner_phase[slot] = ctx->owner_phase;
    }
    return slot;
}

static uint64_t read_expert(cgc_expert_streamer_t* s, int expert_id, void* buffer) {
    uint64_t seg_offs[3] = {0}, seg_sizes[3] = {0};
    int n_seg = cgc_expert_segments(&s->layout, 0, expert_id, seg_offs, seg_sizes);
    if (n_seg <= 0) {
        fprintf(stderr, "[ExpertStreamer] invalid expert=%d\n", expert_id);
        return 0;
    }

    uint64_t total_read_size = 0;
    uint64_t t0 = now_nanos();
    char* dst = (char*)buffer;

    for (int k = 0; k < n_seg; k++) {
        uint64_t file_offset = seg_offs[k];
        uint64_t read_size = seg_sizes[k];
        if (read_size == 0) continue;

#ifdef _WIN32
        OVERLAPPED ov = {0};
        ov.Offset = (DWORD)(file_offset & 0xFFFFFFFF);
        ov.OffsetHigh = (DWORD)(file_offset >> 32);

        DWORD bytes_read = 0;
        BOOL ok = ReadFile(s->file_handle, dst, (DWORD)read_size, &bytes_read, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) {
                fprintf(stderr, "[ExpertStreamer] ReadFile failed: expert=%d offset=%llu err=%lu\n",
                        expert_id, (unsigned long long)file_offset, err);
                return now_nanos() - t0;
            }
            if (!GetOverlappedResult(s->file_handle, &ov, &bytes_read, TRUE)) {
                fprintf(stderr, "[ExpertStreamer] GetOverlappedResult failed: expert=%d err=%lu\n",
                        expert_id, GetLastError());
                return now_nanos() - t0;
            }
        }
        if (bytes_read != read_size) {
            fprintf(stderr, "[ExpertStreamer] short read: expert=%d expected=%llu got=%lu\n",
                    expert_id, (unsigned long long)read_size, bytes_read);
        }
#else
        ssize_t n = pread(s->fd, dst, read_size, (off_t)file_offset);
        if (n != (ssize_t)read_size) {
            fprintf(stderr, "[ExpertStreamer] pread failed: expert=%d seg=%d n=%zd\n", expert_id, k, n);
        }
#endif
        dst += read_size;
        total_read_size += read_size;
    }

    uint64_t elapsed = now_nanos() - t0;
    s->total_read_wall_nanos += elapsed;
    s->total_read_bytes += total_read_size;
    s->total_loads++;
    return elapsed;
}

static void prefetch_expert(cgc_expert_streamer_t* s, int expert_id) {
    if (s->use_mmap && s->mapped_base && !s->layout.has_segments) {
#ifdef _WIN32
        uint64_t expert_off = cgc_expert_offset(&s->layout, 0, expert_id);
        void* region = (char*)s->mapped_base + expert_off;

        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = region;
        entry.NumberOfBytes = (SIZE_T)s->layout.expert_stride;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#elif defined(__linux__)
        uint64_t expert_off = cgc_expert_offset(&s->layout, 0, expert_id);
        void* region = (char*)s->mapped_base + expert_off;
        madvise(region, s->layout.expert_stride, MADV_WILLNEED);
#endif
    }
}

cgc_expert_streamer_t* cgc_expert_streamer_create(const cgc_stream_layout_t* layout,
                                                   int slot_count,
                                                   bool use_mmap,
                                                   const int* hot_pool_experts,
                                                   int hot_pool_count) {
    if (!layout || slot_count <= 0 || slot_count > CGC_MAX_SLOT_COUNT) {
        return NULL;
    }

    cgc_expert_streamer_t* s = (cgc_expert_streamer_t*)calloc(1, sizeof(cgc_expert_streamer_t));
    if (!s) return NULL;

    memcpy(&s->layout, layout, sizeof(cgc_stream_layout_t));
    s->slot_count = slot_count;
    s->use_mmap = use_mmap;

    // llama.cpp 3-tensor 佈局（has_segments）: 專家分散於非連續區段，
    // mmap 單一視圖無法覆蓋 → 強制走 pread 路徑。
    if (layout->has_segments && s->use_mmap) {
        fprintf(stderr, "[ExpertStreamer] layout has_segments: forcing pread (mmap unsupported)\n");
        s->use_mmap = false;
    }

#ifdef _WIN32
    s->file_handle = INVALID_HANDLE_VALUE;
    s->mapping_handle = NULL;
    s->mapped_base = NULL;
#else
    s->fd = -1;
    s->mapped_base = NULL;
#endif

#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, layout->path, -1, NULL, 0);
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wpath) { free(s); return NULL; }
    MultiByteToWideChar(CP_UTF8, 0, layout->path, -1, wpath, wlen);
    s->file_handle = CreateFileW(
        wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
        use_mmap ? FILE_ATTRIBUTE_READONLY : FILE_ATTRIBUTE_NORMAL, NULL);
    free(wpath);

    if (s->file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ExpertStreamer] CreateFileW failed: %s (err=%lu)\n",
                layout->path, GetLastError());
        free(s);
        return NULL;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(s->file_handle, &file_size)) {
        CloseHandle(s->file_handle);
        free(s);
        return NULL;
    }
    uint64_t required = layout->stream_offset + layout->stream_size;
    if ((uint64_t)file_size.QuadPart < required) {
        fprintf(stderr, "[ExpertStreamer] file size mismatch: expected %llu, got %llu\n",
                required, (uint64_t)file_size.QuadPart);
        CloseHandle(s->file_handle);
        free(s);
        return NULL;
    }

    if (use_mmap) {
        s->mapping_handle = CreateFileMappingW(
            s->file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!s->mapping_handle) {
            CloseHandle(s->file_handle);
            free(s);
            return NULL;
        }
        s->mapped_base = MapViewOfFile(
            s->mapping_handle, FILE_MAP_READ,
            (DWORD)(layout->stream_offset >> 32),
            (DWORD)(layout->stream_offset & 0xFFFFFFFF),
            (SIZE_T)layout->stream_size);
        if (!s->mapped_base) {
            CloseHandle(s->mapping_handle);
            CloseHandle(s->file_handle);
            free(s);
            return NULL;
        }
    }
#else
    s->fd = open(layout->path, O_RDONLY);
    if (s->fd < 0) {
        fprintf(stderr, "[ExpertStreamer] open failed: %s (errno=%d)\n",
                layout->path, errno);
        free(s);
        return NULL;
    }
    struct stat st;
    if (fstat(s->fd, &st) == 0) {
        uint64_t required = layout->stream_offset + layout->stream_size;
        if ((uint64_t)st.st_size < required) {
            close(s->fd);
            free(s);
            return NULL;
        }
    }
    if (use_mmap) {
        s->mapped_base = mmap(NULL, layout->stream_size, PROT_READ, MAP_PRIVATE,
                              s->fd, (off_t)layout->stream_offset);
        if (s->mapped_base == MAP_FAILED) {
            close(s->fd);
            free(s);
            return NULL;
        }
    }
#endif

    for (int i = 0; i < slot_count; i++) {
        s->slot_buffers[i] = cgc_aligned_alloc((size_t)layout->expert_stride, CGC_DEFAULT_ALIGN);
        if (!s->slot_buffers[i]) {
            for (int j = 0; j < i; j++) cgc_aligned_free(s->slot_buffers[j]);
#ifdef _WIN32
            if (s->mapped_base) UnmapViewOfFile(s->mapped_base);
            if (s->mapping_handle) CloseHandle(s->mapping_handle);
            if (s->file_handle != INVALID_HANDLE_VALUE) CloseHandle(s->file_handle);
#else
            if (s->mapped_base) munmap(s->mapped_base, layout->stream_size);
            if (s->fd >= 0) close(s->fd);
#endif
            free(s);
            return NULL;
        }
        memset(s->slot_buffers[i], 0, (size_t)layout->expert_stride);
        s->slot_expert[i] = -1;
        s->slot_owner_phase[i] = CGC_CACHE_SLOT_UNASSIGNED;
        s->slot_hit_count[i] = 0;
        s->slot_last_use[i] = 0;
        s->slot_pinned[i] = false;
    }

    s->hot_pool_count = 0;
    if (hot_pool_experts && hot_pool_count > 0) {
        int seen[CGC_MAX_EXPERTS_PER_LAYER];
        int seen_count = 0;

        for (int i = 0; i < hot_pool_count && i < CGC_MAX_EXPERTS_PER_LAYER; i++) {
            int eid = hot_pool_experts[i];
            if (eid < 0 || eid >= layout->experts_per_layer) continue;

            bool dup = false;
            for (int j = 0; j < seen_count; j++) {
                if (seen[j] == eid) { dup = true; break; }
            }
            if (dup) continue;
            seen[seen_count++] = eid;
            s->hot_pool_experts[s->hot_pool_count++] = eid;
        }

        int slot_idx = 0;
        for (int i = 0; i < s->hot_pool_count && slot_idx < slot_count; i++) {
            int eid = s->hot_pool_experts[i];
            s->slot_expert[slot_idx] = eid;
            s->slot_owner_phase[slot_idx] = CGC_CACHE_SLOT_SHARED_RESIDENT;
            s->slot_pinned[slot_idx] = true;
            read_expert(s, eid, s->slot_buffers[slot_idx]);
            s->slot_hit_count[slot_idx] = 1;
            s->slot_last_use[slot_idx] = ++s->use_clock;
            slot_idx++;
        }
    }

    s->initialized = 1;
    fprintf(stderr, "[ExpertStreamer] %s: %d slots, mmap=%d, hotPool=%d, stride=%llu\n",
            layout->path, slot_count, use_mmap ? 1 : 0,
            s->hot_pool_count,
            (unsigned long long)layout->expert_stride);
    return s;
}

void cgc_expert_streamer_destroy(cgc_expert_streamer_t* s) {
    if (!s) return;

    for (int i = 0; i < s->slot_count; i++) {
        if (s->slot_buffers[i]) {
            cgc_aligned_free(s->slot_buffers[i]);
            s->slot_buffers[i] = NULL;
        }
    }

#ifdef _WIN32
    if (s->mapped_base) { UnmapViewOfFile(s->mapped_base); s->mapped_base = NULL; }
    if (s->mapping_handle) { CloseHandle(s->mapping_handle); s->mapping_handle = NULL; }
    if (s->file_handle != INVALID_HANDLE_VALUE) { CloseHandle(s->file_handle); s->file_handle = INVALID_HANDLE_VALUE; }
#else
    if (s->mapped_base) { munmap(s->mapped_base, s->layout.stream_size); s->mapped_base = NULL; }
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
#endif

    free(s);
}

cgc_cache_result_t cgc_expert_streamer_load_experts(cgc_expert_streamer_t* s,
                                                     const int* expert_ids,
                                                     int count,
                                                     const cgc_cache_access_ctx_t* ctx) {
    cgc_cache_result_t result;
    memset(&result, 0, sizeof(result));

    if (!s || !expert_ids || count <= 0 || count > CGC_MAX_EXPERTS_PER_LAYER) {
        return result;
    }

    result.count = count;

    uint64_t total_read_nanos = 0;
    uint64_t total_read_bytes = 0;

    for (int i = 0; i < count; i++) {
        int expert_id = expert_ids[i];
        s->total_requests++;

        int slot = find_slot(s, expert_id);

        if (slot >= 0) {
            s->total_hits++;
            result.hits++;
            s->slot_hit_count[slot]++;
            s->slot_last_use[slot] = ++s->use_clock;
            result.buffers[i] = s->slot_buffers[slot];
            result.sizes[i] = s->layout.expert_stride;
        } else {
            s->total_misses++;
            result.misses++;

            slot = allocate_slot(s, ctx);
            if (slot < 0) {
                result.buffers[i] = NULL;
                result.sizes[i] = 0;
                continue;
            }

            uint64_t read_nanos = read_expert(s, expert_id, s->slot_buffers[slot]);
            total_read_nanos += read_nanos;
            total_read_bytes += s->layout.expert_stride;

            s->slot_expert[slot] = expert_id;
            s->slot_hit_count[slot] = 1;
            s->slot_last_use[slot] = ++s->use_clock;
            result.buffers[i] = s->slot_buffers[slot];
            result.sizes[i] = s->layout.expert_stride;
        }

        if (s->use_mmap && s->mapped_base && !s->layout.has_segments) {
            uint64_t expert_off = cgc_expert_offset(&s->layout, 0, expert_id);
            result.buffers[i] = (char*)s->mapped_base + expert_off;
            result.offsets[i] = 0;
            result.sizes[i] = s->layout.expert_stride;
        }
    }

    result.read_wall_nanos = total_read_nanos;
    result.read_bytes = total_read_bytes;

    return result;
}

void cgc_expert_streamer_prefetch(cgc_expert_streamer_t* s,
                                   const int* expert_ids,
                                   int count) {
    if (!s || !expert_ids) return;

    for (int i = 0; i < count; i++) {
        if (find_slot(s, expert_ids[i]) < 0) {
            prefetch_expert(s, expert_ids[i]);
        }
    }
}

cgc_cache_telemetry_t cgc_expert_streamer_telemetry(const cgc_expert_streamer_t* s) {
    cgc_cache_telemetry_t t;
    memset(&t, 0, sizeof(t));

    if (!s) return t;

    t.slot_count = s->slot_count;
    t.occupied_slots = 0;
    for (int i = 0; i < s->slot_count; i++) {
        if (s->slot_expert[i] != -1) t.occupied_slots++;
    }
    t.total_requests = s->total_requests;
    t.total_hits = s->total_hits;
    t.total_misses = s->total_misses;
    t.total_loads = s->total_loads;
    t.total_evictions = s->total_evictions;
    t.total_read_wall_nanos = s->total_read_wall_nanos;
    t.total_read_bytes = s->total_read_bytes;

    return t;
}

void cgc_expert_streamer_release_slot(cgc_expert_streamer_t* s, int slot) {
    if (!s || slot < 0 || slot >= s->slot_count || s->slot_pinned[slot]) return;
    s->slot_expert[slot] = -1;
    s->slot_owner_phase[slot] = CGC_CACHE_SLOT_UNASSIGNED;
    s->slot_hit_count[slot] = 0;
    s->slot_last_use[slot] = 0;
}

cgc_streamer_pool_t* cgc_streamer_pool_create(void) {
    cgc_streamer_pool_t* pool = (cgc_streamer_pool_t*)calloc(1, sizeof(cgc_streamer_pool_t));
    return pool;
}

void cgc_streamer_pool_destroy(cgc_streamer_pool_t* pool) {
    if (!pool) return;
    free(pool);
}

bool cgc_streamer_pool_add(cgc_streamer_pool_t* pool,
                            int layer_idx,
                            cgc_expert_streamer_t* streamer) {
    if (!pool || !streamer || pool->count >= 1024) return false;
    pool->layer_indices[pool->count] = layer_idx;
    pool->streamers[pool->count] = streamer;
    pool->count++;
    return true;
}

cgc_expert_streamer_t* cgc_streamer_pool_get(cgc_streamer_pool_t* pool, int layer_idx) {
    if (!pool) return NULL;
    for (int i = 0; i < pool->count; i++) {
        if (pool->layer_indices[i] == layer_idx) return pool->streamers[i];
    }
    return NULL;
}

cgc_cache_result_t cgc_streamer_pool_load_experts(cgc_streamer_pool_t* pool,
                                                   int layer_idx,
                                                   const int* expert_ids,
                                                   int count,
                                                   const cgc_cache_access_ctx_t* ctx) {
    cgc_cache_result_t empty;
    memset(&empty, 0, sizeof(empty));
    if (!pool) return empty;

    cgc_expert_streamer_t* s = cgc_streamer_pool_get(pool, layer_idx);
    if (!s) return empty;
    return cgc_expert_streamer_load_experts(s, expert_ids, count, ctx);
}

void cgc_streamer_pool_prefetch(cgc_streamer_pool_t* pool,
                                  int layer_idx,
                                  const int* expert_ids,
                                  int count) {
    if (!pool) return;
    cgc_expert_streamer_t* s = cgc_streamer_pool_get(pool, layer_idx);
    if (s) cgc_expert_streamer_prefetch(s, expert_ids, count);
}

uint64_t cgc_stream_layout_compute_offset(const cgc_stream_layout_t* layout,
                                            int layer, int expert) {
    return cgc_expert_offset(layout, layer, expert);
}
