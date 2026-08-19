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
#endif

// ---------------------------------------------------------------------------
// 共享 mmap: 同一 GGUF 文件被多层 streamer 复用时只映射一次,
// 避免每层独立 MapViewOfFile 整个 13GB 文件 (30 层 = 390GB 虚拟地址空间导致失败)
// ---------------------------------------------------------------------------
#ifdef _WIN32
static char  g_mmap_path[CGC_MAX_PATH_LEN];
static void* g_mmap_base   = NULL;
static HANDLE g_mapping_handle = NULL;
static HANDLE g_file_handle = INVALID_HANDLE_VALUE;
static int   g_mmap_refcount = 0;
#endif

#ifdef _WIN32
// 建立或复用全局共享映射; 返回 1=成功, 0=失败
static int shared_mmap_ensure(const char* path, cgc_expert_streamer_t* s) {
    // 同路径已映射: 直接复用
    if (g_mmap_base && strcmp(g_mmap_path, path) == 0) {
        s->mapped_base    = g_mmap_base;
        s->mapping_handle = g_mapping_handle;
        s->file_handle    = g_file_handle;   // mmap 读路径不用 ReadFile, 共享句柄即可
        g_mmap_refcount++;
        return 1;
    }
    // 不同路径: 释放旧映射后重建
    if (g_mmap_refcount > 0) {
        if (g_mmap_base)    UnmapViewOfFile(g_mmap_base);
        if (g_mapping_handle) CloseHandle(g_mapping_handle);
        if (g_file_handle != INVALID_HANDLE_VALUE) CloseHandle(g_file_handle);
        g_mmap_base = NULL; g_mapping_handle = NULL; g_file_handle = INVALID_HANDLE_VALUE;
        g_mmap_refcount = 0; g_mmap_path[0] = 0;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wpath) return 0;
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);

    g_file_handle = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
    free(wpath);
    if (g_file_handle == INVALID_HANDLE_VALUE) return 0;

    g_mapping_handle = CreateFileMappingW(g_file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!g_mapping_handle) {
        CloseHandle(g_file_handle); g_file_handle = INVALID_HANDLE_VALUE;
        return 0;
    }
    // dwNumberOfBytesToMap=0 → 映射整个文件
    g_mmap_base = MapViewOfFile(g_mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (!g_mmap_base) {
        CloseHandle(g_mapping_handle); g_mapping_handle = NULL;
        CloseHandle(g_file_handle); g_file_handle = INVALID_HANDLE_VALUE;
        return 0;
    }
    strncpy(g_mmap_path, path, CGC_MAX_PATH_LEN - 1);
    g_mmap_path[CGC_MAX_PATH_LEN - 1] = 0;

    s->mapped_base    = g_mmap_base;
    s->mapping_handle = g_mapping_handle;
    s->file_handle    = g_file_handle;
    g_mmap_refcount = 1;
    return 1;
}
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

static void* aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

static void aligned_free(void* ptr) {
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

// 从文件指定偏移读入 buffer (Windows OVERLAPPED / POSIX pread)
static uint64_t read_from_file(cgc_expert_streamer_t* s, uint64_t file_offset,
                               void* buffer, uint64_t read_size) {
#ifdef _WIN32
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(file_offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(file_offset >> 32);

    DWORD bytes_read = 0;
    BOOL ok = ReadFile(s->file_handle, buffer, (DWORD)read_size, &bytes_read, &ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            fprintf(stderr, "[ExpertStreamer] ReadFile failed: offset=%llu err=%lu\n",
                    (unsigned long long)file_offset, err);
            return 0;
        }
        if (!GetOverlappedResult(s->file_handle, &ov, &bytes_read, TRUE)) {
            fprintf(stderr, "[ExpertStreamer] GetOverlappedResult failed: offset=%llu err=%lu\n",
                    (unsigned long long)file_offset, GetLastError());
            return 0;
        }
    }
    if (bytes_read != (DWORD)read_size) {
        fprintf(stderr, "[ExpertStreamer] short read: offset=%llu expected=%llu got=%lu\n",
                (unsigned long long)file_offset, (unsigned long long)read_size, bytes_read);
        return bytes_read;
    }
#else
    size_t filled = 0;
    while (filled < read_size) {
        ssize_t n = pread(s->fd, (char*)buffer + filled, read_size - filled,
                          (off_t)(file_offset + filled));
        if (n < 0) {
            fprintf(stderr, "[ExpertStreamer] pread failed: offset=%llu errno=%d\n",
                    (unsigned long long)file_offset, errno);
            return filled;
        }
        if (n == 0) break;
        filled += (size_t)n;
    }
    return filled;
#endif
    return read_size;
}

static uint64_t read_expert(cgc_expert_streamer_t* s, int expert_id, void* buffer) {
    uint64_t t0 = now_nanos();
    uint64_t total = 0;

    if (s->use_mmap && s->mapped_base) {
        // mmap 路径: 直接从映射内存 memcpy (省去 ReadFile/pread 系统调用 + 内核拷贝),
        // 首次访问触发 page fault, 之后走 OS page cache
        if (s->layout.has_layer_offsets && s->layout.region_count > 0) {
            char* dst = (char*)buffer;
            for (int r = 0; r < s->layout.region_count && r < CGC_MAX_EXPERT_REGIONS; r++) {
                uint64_t off = cgc_stream_region_offset(&s->layout, s->layer_index, r, expert_id);
                uint64_t size = s->layout.region_stride[r];
                memcpy(dst, (const char*)s->mapped_base + off, size);
                total += size;
                dst += size;
            }
        } else {
            uint64_t file_offset = cgc_expert_offset(&s->layout, s->layer_index, expert_id);
            memcpy(buffer, (const char*)s->mapped_base + file_offset, s->layout.expert_stride);
            total = s->layout.expert_stride;
        }
    } else if (s->layout.has_layer_offsets && s->layout.region_count > 0) {
        // per-layer 多 region: 把每个 region 的专家切片顺序读入 slot buffer
        char* dst = (char*)buffer;
        for (int r = 0; r < s->layout.region_count && r < CGC_MAX_EXPERT_REGIONS; r++) {
            uint64_t off = cgc_stream_region_offset(&s->layout, s->layer_index, r, expert_id);
            uint64_t size = s->layout.region_stride[r];
            uint64_t n = read_from_file(s, off, dst, size);
            total += n;
            dst += size;
        }
    } else {
        uint64_t file_offset = cgc_expert_offset(&s->layout, s->layer_index, expert_id);
        uint64_t read_size = s->layout.expert_stride;
        total = read_from_file(s, file_offset, buffer, read_size);
    }

    uint64_t elapsed = now_nanos() - t0;
    s->total_read_wall_nanos += elapsed;
    s->total_read_bytes += total;
    s->total_loads++;
    return elapsed;
}

// 把专家数据写入槽位: 紧凑模式下按 region 基址 + slot*stride 分散写入
// (region_bases[r] + slot*region_stride[r]); 否则沿用 per-slot 连续 buffer。
// 返回读取耗时; 失败返回 0 但不破坏槽位状态 (调用方按 miss 处理)。
static uint64_t load_expert_to_slot(cgc_expert_streamer_t* s, int expert_id, int slot) {
    if (!s || slot < 0 || slot >= s->slot_count) {
        return 0;
    }

    if (!s->compact_regions) {
        return read_expert(s, expert_id, s->slot_buffers[slot]);
    }

    uint64_t t0 = now_nanos();
    uint64_t total = 0;
    const int n_regions = (s->layout.region_count > 0) ? s->layout.region_count : 1;
    for (int r = 0; r < n_regions && r < CGC_MAX_EXPERT_REGIONS; r++) {
        const uint64_t rstride = (s->layout.region_count > 0)
                                     ? s->layout.region_stride[r]
                                     : s->layout.expert_stride;
        if (rstride == 0 || s->region_bases[r] == NULL) {
            continue;
        }
        char* dst = (char*)s->region_bases[r] + (size_t)slot * (size_t)rstride;
        uint64_t off;
        uint64_t size;
        if (s->layout.has_layer_offsets && s->layout.region_count > 0) {
            off = cgc_stream_region_offset(&s->layout, s->layer_index, r, expert_id);
            size = rstride;
        } else {
            off = cgc_expert_offset(&s->layout, s->layer_index, expert_id);
            size = s->layout.expert_stride;
        }
        if (s->use_mmap && s->mapped_base) {
            memcpy(dst, (const char*)s->mapped_base + off, (size_t)size);
        } else {
            uint64_t n = read_from_file(s, off, dst, size);
            if (n != size) {
                // 部分读取: 剩余清零, 避免脏数据被 mul_mat_id 读到
                memset(dst + n, 0, (size_t)(size - n));
            }
        }
        total += size;
    }

    uint64_t elapsed = now_nanos() - t0;
    s->total_read_wall_nanos += elapsed;
    s->total_read_bytes += total;
    s->total_loads++;
    return elapsed;
}

static void prefetch_expert(cgc_expert_streamer_t* s, int expert_id) {
    if (s->use_mmap && s->mapped_base) {
#ifdef _WIN32
        // per-layer 多 region: 专家切片分散在多个 region, 逐 region 构造 hint
        WIN32_MEMORY_RANGE_ENTRY entries[CGC_MAX_EXPERT_REGIONS];
        int n = 0;
        if (s->layout.has_layer_offsets && s->layout.region_count > 0) {
            for (int r = 0; r < s->layout.region_count && r < CGC_MAX_EXPERT_REGIONS; r++) {
                uint64_t off = cgc_stream_region_offset(&s->layout, s->layer_index, r, expert_id);
                entries[n].VirtualAddress = (char*)s->mapped_base + off;
                entries[n].NumberOfBytes = (SIZE_T)s->layout.region_stride[r];
                n++;
            }
        } else {
            uint64_t expert_off = cgc_expert_offset(&s->layout, s->layer_index, expert_id);
            entries[0].VirtualAddress = (char*)s->mapped_base + expert_off;
            entries[0].NumberOfBytes = (SIZE_T)s->layout.expert_stride;
            n = 1;
        }
        if (n > 0) PrefetchVirtualMemory(GetCurrentProcess(), n, entries, 0);
#elif defined(__linux__)
        uint64_t expert_off = cgc_expert_offset(&s->layout, s->layer_index, expert_id);
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
    return cgc_expert_streamer_create_ex(layout, slot_count, use_mmap,
                                         hot_pool_experts, hot_pool_count, false);
}

cgc_expert_streamer_t* cgc_expert_streamer_create_ex(const cgc_stream_layout_t* layout,
                                                      int slot_count,
                                                      bool use_mmap,
                                                      const int* hot_pool_experts,
                                                      int hot_pool_count,
                                                      bool compact_regions) {
    if (!layout || slot_count <= 0 || slot_count > CGC_MAX_SLOT_COUNT) {
        return NULL;
    }

    cgc_expert_streamer_t* s = (cgc_expert_streamer_t*)calloc(1, sizeof(cgc_expert_streamer_t));
    if (!s) return NULL;

    memcpy(&s->layout, layout, sizeof(cgc_stream_layout_t));
    s->slot_count = slot_count;
    s->use_mmap = use_mmap;
    s->compact_regions = compact_regions;

#ifdef _WIN32
    s->file_handle = INVALID_HANDLE_VALUE;
    s->mapping_handle = NULL;
    s->mapped_base = NULL;
#else
    s->fd = -1;
    s->mapped_base = NULL;
#endif

    int wlen = MultiByteToWideChar(CP_UTF8, 0, layout->path, -1, NULL, 0);
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wpath) { free(s); return NULL; }
    MultiByteToWideChar(CP_UTF8, 0, layout->path, -1, wpath, wlen);

#ifdef _WIN32
    s->file_handle = CreateFileW(
        wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
        use_mmap ? FILE_ATTRIBUTE_READONLY
                 : (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED), NULL);
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
        // 共享 mmap: 同文件只映射一次, 避免 30 层 × 13GB 虚拟地址空间耗尽
        HANDLE own_handle = s->file_handle;   // 每层独立打开的句柄 (仅用于大小校验)
        if (!shared_mmap_ensure(layout->path, s)) {
            fprintf(stderr, "[ExpertStreamer] mmap failed, fallback to ReadFile mode: %s (err=%lu)\n",
                    layout->path, GetLastError());
            s->use_mmap = false;
            s->file_handle = own_handle;      // 回退恢复独立句柄
        } else {
            CloseHandle(own_handle);          // 共享映射就绪, 关闭独立句柄
        }
    }
#else
    free(wpath);
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

    // per-layer 多 region 布局下 expert_stride 可能为 0: slot buffer 大小取
    // cgc_stream_layout_expert_size (所有 region 合计), 否则 aligned_alloc(0)
    // 得到 0 字节 buffer, read_expert 按 region 大小 memcpy 越界 -> access violation
    if (s->compact_regions) {
        // 紧凑模式: 每 region 一块连续内存 (容量 = region_stride * slot_count),
        // 槽数据按 region_bases[r] + slot*region_stride[r] 排布。
        const int n_regions = (layout->region_count > 0) ? layout->region_count : 1;
        for (int r = 0; r < n_regions && r < CGC_MAX_EXPERT_REGIONS; r++) {
            const uint64_t rstride = (layout->region_count > 0)
                                         ? layout->region_stride[r]
                                         : layout->expert_stride;
            if (rstride == 0) {
                s->region_bases[r] = NULL;
                continue;
            }
            const size_t rsize = (size_t)(rstride * (uint64_t)slot_count);
            s->region_bases[r] = aligned_alloc(rsize, CGC_DEFAULT_ALIGN);
            if (!s->region_bases[r]) {
                for (int j = 0; j < r; j++) {
                    if (s->region_bases[j]) aligned_free(s->region_bases[j]);
                }
                goto alloc_fail;
            }
            memset(s->region_bases[r], 0, rsize);
        }
    } else {
        uint64_t expert_bytes = cgc_stream_layout_expert_size(layout);
        for (int i = 0; i < slot_count; i++) {
            s->slot_buffers[i] = aligned_alloc((size_t)expert_bytes, CGC_DEFAULT_ALIGN);
            if (!s->slot_buffers[i]) {
                for (int j = 0; j < i; j++) aligned_free(s->slot_buffers[j]);
                goto alloc_fail;
            }
            memset(s->slot_buffers[i], 0, (size_t)expert_bytes);
        }
    }

    for (int i = 0; i < slot_count; i++) {
        s->slot_expert[i] = -1;
        s->slot_owner_phase[i] = CGC_CACHE_SLOT_UNASSIGNED;
        s->slot_hit_count[i] = 0;
        s->slot_last_use[i] = 0;
        s->slot_pinned[i] = false;
    }

    goto alloc_ok;

alloc_fail:
#ifdef _WIN32
    // 共享映射: 只减引用计数, 不释放共享句柄 (其他 streamer 可能仍在使用)
    if (s->use_mmap && s->mapped_base == g_mmap_base) {
        if (g_mmap_refcount > 0) g_mmap_refcount--;
        s->mapped_base = NULL; s->mapping_handle = NULL; s->file_handle = INVALID_HANDLE_VALUE;
    } else {
        if (s->mapped_base) UnmapViewOfFile(s->mapped_base);
        if (s->mapping_handle) CloseHandle(s->mapping_handle);
        if (s->file_handle != INVALID_HANDLE_VALUE) CloseHandle(s->file_handle);
    }
#else
    if (s->mapped_base) munmap(s->mapped_base, layout->stream_size);
    if (s->fd >= 0) close(s->fd);
#endif
    free(s);
    return NULL;

alloc_ok:
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
            load_expert_to_slot(s, eid, slot_idx);
            s->slot_hit_count[slot_idx] = 1;
            s->slot_last_use[slot_idx] = ++s->use_clock;
            slot_idx++;
        }
    }

    s->initialized = 1;
    const uint64_t expert_bytes = cgc_stream_layout_expert_size(layout);
    fprintf(stderr, "[ExpertStreamer] %s: %d slots, mmap=%d, hotPool=%d, expert_bytes=%llu, compact=%d\n",
            layout->path, slot_count, use_mmap ? 1 : 0,
            s->hot_pool_count,
            (unsigned long long)expert_bytes,
            s->compact_regions ? 1 : 0);
    return s;
}

void cgc_expert_streamer_destroy(cgc_expert_streamer_t* s) {
    if (!s) return;

    if (s->compact_regions) {
        for (int r = 0; r < CGC_MAX_EXPERT_REGIONS; r++) {
            if (s->region_bases[r]) {
                aligned_free(s->region_bases[r]);
                s->region_bases[r] = NULL;
            }
        }
    } else {
        for (int i = 0; i < s->slot_count; i++) {
            if (s->slot_buffers[i]) {
                aligned_free(s->slot_buffers[i]);
                s->slot_buffers[i] = NULL;
            }
        }
    }

#ifdef _WIN32
    if (s->use_mmap && s->mapped_base == g_mmap_base) {
        // 共享映射: 引用计数释放
        if (g_mmap_refcount > 0) g_mmap_refcount--;
        if (g_mmap_refcount <= 0) {
            if (g_mmap_base)        UnmapViewOfFile(g_mmap_base);
            if (g_mapping_handle)   CloseHandle(g_mapping_handle);
            if (g_file_handle != INVALID_HANDLE_VALUE) CloseHandle(g_file_handle);
            g_mmap_base = NULL; g_mapping_handle = NULL; g_file_handle = INVALID_HANDLE_VALUE;
            g_mmap_path[0] = 0; g_mmap_refcount = 0;
        }
        s->mapped_base = NULL; s->mapping_handle = NULL; s->file_handle = INVALID_HANDLE_VALUE;
    } else {
        if (s->mapped_base) { UnmapViewOfFile(s->mapped_base); s->mapped_base = NULL; }
        if (s->mapping_handle) { CloseHandle(s->mapping_handle); s->mapping_handle = NULL; }
        if (s->file_handle != INVALID_HANDLE_VALUE) { CloseHandle(s->file_handle); s->file_handle = INVALID_HANDLE_VALUE; }
    }
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
            result.sizes[i] = cgc_stream_layout_expert_size(&s->layout);
        } else {
            s->total_misses++;
            result.misses++;

            slot = allocate_slot(s, ctx);
            if (slot < 0) {
                result.buffers[i] = NULL;
                result.sizes[i] = 0;
                continue;
            }

            load_expert_to_slot(s, expert_id, slot);
            total_read_nanos += (s->compact_regions ? cgc_stream_layout_expert_size(&s->layout) : 0);
            total_read_bytes += cgc_stream_layout_expert_size(&s->layout);

            s->slot_expert[slot] = expert_id;
            s->slot_hit_count[slot] = 1;
            s->slot_last_use[slot] = ++s->use_clock;
            // 紧凑模式下专家数据分散在 region_bases, 无单一 buffer 可返回
            result.buffers[i] = s->compact_regions ? NULL : s->slot_buffers[slot];
            result.sizes[i] = s->compact_regions ? 0 : cgc_stream_layout_expert_size(&s->layout);
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

// 主动预取加载: 把尚未缓存且非热池的专家真正读入缓存槽 (不同于 mmap hint)
// 返回实际读入的专家数。路由控制器用它实现 miss-only 预取。
int cgc_expert_streamer_prefetch_load(cgc_expert_streamer_t* s,
                                      const int* expert_ids,
                                      int count,
                                      const cgc_cache_access_ctx_t* ctx) {
    if (!s || !expert_ids || count <= 0 || count > CGC_MAX_EXPERTS_PER_LAYER) {
        return 0;
    }

    int loaded = 0;
    uint64_t t0 = now_nanos();

    for (int i = 0; i < count; i++) {
        int expert_id = expert_ids[i];
        if (expert_id < 0 || expert_id >= s->layout.experts_per_layer) continue;
        if (find_slot(s, expert_id) >= 0) continue;   // miss-only

        int slot = allocate_slot(s, ctx);
        if (slot < 0) break;                          // 无可用槽, 停止 (避免干扰主线程)

        read_expert(s, expert_id, s->slot_buffers[slot]);
        s->slot_expert[slot] = expert_id;
        s->slot_hit_count[slot] = 1;
        s->slot_last_use[slot] = ++s->use_clock;
        loaded++;
    }

    uint64_t elapsed = now_nanos() - t0;
    s->total_prefetch_wall_nanos += elapsed;
    return loaded;
}

void cgc_expert_streamer_set_layer(cgc_expert_streamer_t* streamer, int layer) {
    if (!streamer) return;
    streamer->layer_index = layer;
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
