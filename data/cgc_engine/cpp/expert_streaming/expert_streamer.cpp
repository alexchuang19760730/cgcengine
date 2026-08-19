// expert_streamer.cpp — Windows 实现 (ported from turbo-fieldfare PreadExpertStreamer.swift)
//
// 平台映射:
//   open()/O_RDONLY   -> CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING)
//   pread()           -> ReadFile() + OVERLAPPED (异步定位读取)
//   mmap(MAP_PRIVATE) -> CreateFileMapping(PAGE_READONLY) + MapViewOfFile(FILE_MAP_READ)
//   F_RDADVISE        -> PrefetchVirtualMemory() (Win8+, 需 SeProfileSingleProcessPrivilege)
//   posix_memalign    -> _aligned_malloc / VirtualAlloc
//   MTLBuffer         -> void* (CPU buffer)
//
// 设计原则:
// 1. 语义 1:1 对应 turbo-fieldfare,只换系统调用
// 2. 先做 pread 模式 (最小可行),mmap 模式可选
// 3. LRU 驱逐 (slotLastUse_),后续可扩展 LFU
// 4. hot pool expert pin 在 slot 里不驱逐

#include "expert_streamer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "kernel32.lib")
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#endif

namespace cgc_moe {

// ============================================================================
// 计时工具
// ============================================================================

static inline uint64_t nowNanos() {
#ifdef _WIN32
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<uint64_t>(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
}

// ============================================================================
// 对齐分配
// ============================================================================

void* ExpertStreamer::allocateAligned(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

void ExpertStreamer::freeAligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// 构造 / 析构
// ============================================================================

ExpertStreamer::ExpertStreamer(const StreamLayout& layout,
                               int slotCount,
                               bool useMmap,
                               std::vector<int> hotPoolExperts)
    : layout_(layout)
    , slotCount_(slotCount)
    , useMmap_(useMmap)
    , hotPoolExperts_(std::move(hotPoolExperts))
{
    // 去重 hot pool
    std::sort(hotPoolExperts_.begin(), hotPoolExperts_.end());
    hotPoolExperts_.erase(
        std::unique(hotPoolExperts_.begin(), hotPoolExperts_.end()),
        hotPoolExperts_.end());

    // 过滤无效 expert
    hotPoolExperts_.erase(
        std::remove_if(hotPoolExperts_.begin(), hotPoolExperts_.end(),
            [&](int e) { return e < 0 || e >= layout_.expertsPerLayer; }),
        hotPoolExperts_.end());

#ifdef _WIN32
    // ---- Windows: 打开文件 ----
    // 将 UTF-8 path 转 UTF-16
    int wlen = MultiByteToWideChar(CP_UTF8, 0, layout_.path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, layout_.path.c_str(), -1, wpath.data(), wlen);

    fileHandle_ = CreateFileW(
        wpath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        useMmap_ ? FILE_ATTRIBUTE_READONLY : FILE_ATTRIBUTE_NORMAL,
        nullptr);

    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[ExpertStreamer] CreateFileW failed: %s (errno=%lu)\n",
                layout_.path.c_str(), GetLastError());
        throw std::runtime_error("ExpertStreamer: CreateFileW failed");
    }

    // 验证文件大小
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(fileHandle_, &fileSize)) {
        CloseHandle(fileHandle_);
        throw std::runtime_error("ExpertStreamer: GetFileSizeEx failed");
    }
    uint64_t required = layout_.streamOffset + layout_.streamSize;
    if (static_cast<uint64_t>(fileSize.QuadPart) < required) {
        CloseHandle(fileHandle_);
        fprintf(stderr, "[ExpertStreamer] file size mismatch: expected %llu, got %llu\n",
                required, static_cast<uint64_t>(fileSize.QuadPart));
        throw std::runtime_error("ExpertStreamer: file size mismatch");
    }

    if (useMmap_) {
        // ---- mmap 模式: 映射整个 stream ----
        mappingHandle_ = CreateFileMappingW(
            fileHandle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mappingHandle_) {
            CloseHandle(fileHandle_);
            throw std::runtime_error("ExpertStreamer: CreateFileMappingW failed");
        }
        mappedBase_ = MapViewOfFile(
            mappingHandle_, FILE_MAP_READ,
            static_cast<DWORD>(layout_.streamOffset >> 32),
            static_cast<DWORD>(layout_.streamOffset & 0xFFFFFFFF),
            static_cast<SIZE_T>(layout_.streamSize));
        if (!mappedBase_) {
            CloseHandle(mappingHandle_);
            CloseHandle(fileHandle_);
            throw std::runtime_error("ExpertStreamer: MapViewOfFile failed");
        }
    }
#else
    // ---- POSIX: open ----
    fd_ = open(layout_.path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "[ExpertStreamer] open failed: %s (errno=%d)\n",
                layout_.path.c_str(), errno);
        throw std::runtime_error("ExpertStreamer: open failed");
    }
    struct stat st;
    if (fstat(fd_, &st) == 0) {
        uint64_t required = layout_.streamOffset + layout_.streamSize;
        if (static_cast<uint64_t>(st.st_size) < required) {
            close(fd_);
            throw std::runtime_error("ExpertStreamer: file size mismatch");
        }
    }
    if (useMmap_) {
        mappedBase_ = mmap(nullptr, layout_.streamSize, PROT_READ, MAP_PRIVATE,
                           fd_, static_cast<off_t>(layout_.streamOffset));
        if (mappedBase_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("ExpertStreamer: mmap failed");
        }
    }
#endif

    // ---- 分配 slot buffer ----
    slotBuffers_.resize(slotCount_);
    slotExpert_.resize(slotCount_, -1);
    slotOwnerPhase_.resize(slotCount_, ExpertCacheSlotOwnerPhase::Unassigned);
    slotHitCount_.resize(slotCount_, 0);
    slotLastUse_.resize(slotCount_, 0);
    slotPinned_.resize(slotCount_, false);

    for (int i = 0; i < slotCount_; ++i) {
        slotBuffers_[i] = allocateAligned(static_cast<size_t>(layout_.expertStride));
        if (!slotBuffers_[i]) {
            // 清理已分配的
            for (int j = 0; j < i; ++j) freeAligned(slotBuffers_[j]);
#ifdef _WIN32
            if (mappedBase_) { UnmapViewOfFile(mappedBase_); mappedBase_ = nullptr; }
            if (mappingHandle_) { CloseHandle(mappingHandle_); mappingHandle_ = nullptr; }
            if (fileHandle_ != INVALID_HANDLE_VALUE) { CloseHandle(fileHandle_); fileHandle_ = INVALID_HANDLE_VALUE; }
#else
            if (mappedBase_) { munmap(mappedBase_, layout_.streamSize); mappedBase_ = nullptr; }
            if (fd_ >= 0) { close(fd_); fd_ = -1; }
#endif
            throw std::runtime_error("ExpertStreamer: allocateAligned failed");
        }
    }

    // ---- 预加载 hot pool experts ----
    int hotSlotIdx = 0;
    for (int expertId : hotPoolExperts_) {
        if (hotSlotIdx >= slotCount_) break;
        slotExpert_[hotSlotIdx] = expertId;
        slotOwnerPhase_[hotSlotIdx] = ExpertCacheSlotOwnerPhase::SharedResident;
        slotPinned_[hotSlotIdx] = true;
        readExpert(expertId, slotBuffers_[hotSlotIdx]);
        slotHitCount_[hotSlotIdx] = 1;
        slotLastUse_[hotSlotIdx] = ++useClock_;
        ++hotSlotIdx;
        ++totalLoads_;
    }

    fprintf(stderr, "[ExpertStreamer] %s: %d slots, mmap=%d, hotPool=%zu, stride=%llu\n",
            layout_.path.c_str(), slotCount_, useMmap_ ? 1 : 0,
            hotPoolExperts_.size(),
            static_cast<unsigned long long>(layout_.expertStride));
}

ExpertStreamer::~ExpertStreamer() {
    for (int i = 0; i < slotCount_; ++i) {
        if (slotBuffers_[i]) freeAligned(slotBuffers_[i]);
    }
#ifdef _WIN32
    if (mappedBase_) UnmapViewOfFile(mappedBase_);
    if (mappingHandle_) CloseHandle(mappingHandle_);
    if (fileHandle_ != INVALID_HANDLE_VALUE) CloseHandle(fileHandle_);
#else
    if (mappedBase_) munmap(mappedBase_, layout_.streamSize);
    if (fd_ >= 0) close(fd_);
#endif
}

// ============================================================================
// findSlot — 查找 expert 在 cache 中的 slot
// ============================================================================

int ExpertStreamer::findSlot(int expertId) {
    for (int i = 0; i < slotCount_; ++i) {
        if (slotExpert_[i] == expertId) return i;
    }
    return -1;
}

// ============================================================================
// evictSlot — LRU 驱逐 (跳过 pinned slot)
// ============================================================================

int ExpertStreamer::evictSlot() {
    int victim = -1;
    int minUse = INT32_MAX;

    for (int i = 0; i < slotCount_; ++i) {
        if (slotPinned_[i]) continue;
        if (slotExpert_[i] == -1) return i;  // 空闲 slot 直接用
        if (slotLastUse_[i] < minUse) {
            minUse = slotLastUse_[i];
            victim = i;
        }
    }

    if (victim >= 0) {
        slotExpert_[victim] = -1;
        slotOwnerPhase_[victim] = ExpertCacheSlotOwnerPhase::Unassigned;
        slotHitCount_[victim] = 0;
        slotLastUse_[victim] = 0;
        ++totalEvictions_;
    }
    return victim;
}

// ============================================================================
// allocateSlot — 分配 slot (cache miss 时)
// ============================================================================

int ExpertStreamer::allocateSlot(const ExpertCacheAccessContext& ctx) {
    int slot = evictSlot();
    if (slot >= 0) {
        slotOwnerPhase_[slot] = ctx.ownerPhase;
    }
    return slot;
}

// ============================================================================
// readExpert — 从文件读取 expert 到 buffer
//   Windows: ReadFile + OVERLAPPED (异步定位读取,等价 POSIX pread)
//   POSIX:   pread
// ============================================================================

uint64_t ExpertStreamer::readExpert(int expertId, void* buffer) {
    uint64_t fileOffset = layout_.streamOffset +
                          layout_.expertOffset(0, expertId);  // layer 由 streamer 管理
    uint64_t readSize = layout_.expertStride;

    uint64_t t0 = nowNanos();

#ifdef _WIN32
    OVERLAPPED ov = {};
    ov.Offset = static_cast<DWORD>(fileOffset & 0xFFFFFFFF);
    ov.OffsetHigh = static_cast<DWORD>(fileOffset >> 32);

    DWORD bytesRead = 0;
    BOOL ok = ReadFile(fileHandle_, buffer, static_cast<DWORD>(readSize),
                       &bytesRead, &ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            fprintf(stderr, "[ExpertStreamer] ReadFile failed: expert=%d offset=%llu err=%lu\n",
                    expertId, static_cast<unsigned long long>(fileOffset), err);
            return nowNanos() - t0;
        }
        // 等待异步完成
        if (!GetOverlappedResult(fileHandle_, &ov, &bytesRead, TRUE)) {
            fprintf(stderr, "[ExpertStreamer] GetOverlappedResult failed: expert=%d err=%lu\n",
                    expertId, GetLastError());
            return nowNanos() - t0;
        }
    }
    if (bytesRead != readSize) {
        fprintf(stderr, "[ExpertStreamer] short read: expert=%d expected=%llu got=%lu\n",
                expertId, static_cast<unsigned long long>(readSize), bytesRead);
    }
#else
    ssize_t n = pread(fd_, buffer, readSize, static_cast<off_t>(fileOffset));
    if (n != static_cast<ssize_t>(readSize)) {
        fprintf(stderr, "[ExpertStreamer] pread failed: expert=%d n=%zd\n", expertId, n);
    }
#endif

    uint64_t elapsed = nowNanos() - t0;
    totalReadWallNanos_ += elapsed;
    totalReadBytes_ += readSize;
    ++totalLoads_;
    return elapsed;
}

// ============================================================================
// prefetchExpert — Windows PrefetchVirtualMemory / POSIX readahead
// ============================================================================

void ExpertStreamer::prefetchExpert(int expertId) {
    if (useMmap_ && mappedBase_) {
#ifdef _WIN32
        // PrefetchVirtualMemory (Win8+)
        uint64_t expertOff = layout_.expertOffset(0, expertId);
        void* region = static_cast<char*>(mappedBase_) + expertOff;

        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = region;
        entry.NumberOfBytes = static_cast<SIZE_T>(layout_.expertStride);

        // PrefetchVirtualMemory 需要 SeProfileSingleProcessPrivilege
        // 如果失败,静默忽略 (不阻塞主路径)
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#elif defined(__linux__)
        // POSIX: madvise(MADV_WILLNEED) 或 readahead
        uint64_t expertOff = layout_.expertOffset(0, expertId);
        void* region = static_cast<char*>(mappedBase_) + expertOff;
        madvise(region, layout_.expertStride, MADV_WILLNEED);
#endif
    }
    // pread 模式下预取由 ReadFile 的 OS 缓存处理
}

// ============================================================================
// loadExperts — 核心:加载 experts,返回 buffer 指针
// ============================================================================

ExpertCacheResult ExpertStreamer::loadExperts(
    const std::vector<int>& expertIds,
    const ExpertCacheAccessContext& ctx)
{
    ExpertCacheResult result;
    result.buffers.resize(expertIds.size());
    result.offsets.resize(expertIds.size(), 0);
    result.sizes.resize(expertIds.size(), layout_.expertStride);

    std::lock_guard<std::mutex> lock(cacheLock_);

    uint64_t totalReadNanos = 0;
    uint64_t totalReadBytes = 0;

    for (size_t i = 0; i < expertIds.size(); ++i) {
        int expertId = expertIds[i];
        ++totalRequests_;

        // 1. 查 cache
        int slot = findSlot(expertId);

        if (slot >= 0) {
            // ---- Cache Hit ----
            ++totalHits_;
            ++result.hits;
            ++slotHitCount_[slot];
            slotLastUse_[slot] = ++useClock_;
            result.buffers[i] = slotBuffers_[slot];
        } else {
            // ---- Cache Miss ----
            ++totalMisses_;
            ++result.misses;

            // 分配 slot
            slot = allocateSlot(ctx);
            if (slot < 0) {
                // 无可用 slot (不应该发生)
                result.buffers[i] = nullptr;
                continue;
            }

            // 读取 expert 到 slot buffer
            uint64_t readNanos = readExpert(expertId, slotBuffers_[slot]);
            totalReadNanos += readNanos;
            totalReadBytes += layout_.expertStride;

            slotExpert_[slot] = expertId;
            slotHitCount_[slot] = 1;
            slotLastUse_[slot] = ++useClock_;
            result.buffers[i] = slotBuffers_[slot];
        }

        // mmap 模式:直接返回映射区域的指针
        if (useMmap_ && mappedBase_) {
            uint64_t expertOff = layout_.expertOffset(0, expertId);
            result.buffers[i] = static_cast<char*>(mappedBase_) + expertOff;
            result.offsets[i] = 0;
        }
    }

    result.readWallNanos = totalReadNanos;
    result.readBytes = totalReadBytes;

    return result;
}

// ============================================================================
// prefetch — 预取 experts (非阻塞)
// ============================================================================

void ExpertStreamer::prefetch(const std::vector<int>& expertIds) {
    std::lock_guard<std::mutex> lock(cacheLock_);
    for (int expertId : expertIds) {
        // 只预取不在 cache 里的
        if (findSlot(expertId) < 0) {
            prefetchExpert(expertId);
        }
    }
}

// ============================================================================
// releaseSlot — 释放 slot
// ============================================================================

void ExpertStreamer::releaseSlot(int slot) {
    std::lock_guard<std::mutex> lock(cacheLock_);
    if (slot >= 0 && slot < slotCount_ && !slotPinned_[slot]) {
        slotExpert_[slot] = -1;
        slotOwnerPhase_[slot] = ExpertCacheSlotOwnerPhase::Unassigned;
        slotHitCount_[slot] = 0;
        slotLastUse_[slot] = 0;
    }
}

// ============================================================================
// telemetry — 获取遥测
// ============================================================================

ExpertCacheTelemetry ExpertStreamer::telemetry() const {
    ExpertCacheTelemetry t;
    t.slotCount = slotCount_;
    t.occupiedSlots = 0;
    for (int i = 0; i < slotCount_; ++i) {
        if (slotExpert_[i] != -1) ++t.occupiedSlots;
    }
    t.totalRequests = totalRequests_;
    t.totalHits = totalHits_;
    t.totalMisses = totalMisses_;
    t.totalLoads = totalLoads_;
    t.totalEvictions = totalEvictions_;
    t.totalReadWallNanos = totalReadWallNanos_;
    t.totalReadBytes = totalReadBytes_;
    return t;
}

// ============================================================================
// ExpertStreamerPool — 多 layer 管理
// ============================================================================

void ExpertStreamerPool::addStreamer(int layerIdx, std::unique_ptr<ExpertStreamer> streamer) {
    std::lock_guard<std::mutex> lock(poolLock_);
    streamers_.emplace_back(layerIdx, std::move(streamer));
}

ExpertStreamer* ExpertStreamerPool::getStreamer(int layerIdx) {
    std::lock_guard<std::mutex> lock(poolLock_);
    for (auto& [idx, s] : streamers_) {
        if (idx == layerIdx) return s.get();
    }
    return nullptr;
}

ExpertCacheResult ExpertStreamerPool::loadExperts(
    int layerIdx,
    const std::vector<int>& expertIds,
    const ExpertCacheAccessContext& ctx)
{
    auto* s = getStreamer(layerIdx);
    if (!s) {
        ExpertCacheResult empty;
        return empty;
    }
    return s->loadExperts(expertIds, ctx);
}

void ExpertStreamerPool::prefetch(int layerIdx, const std::vector<int>& expertIds) {
    auto* s = getStreamer(layerIdx);
    if (s) s->prefetch(expertIds);
}

std::vector<std::pair<int, ExpertCacheTelemetry>>
ExpertStreamerPool::allTelemetry() const {
    std::lock_guard<std::mutex> lock(poolLock_);
    std::vector<std::pair<int, ExpertCacheTelemetry>> result;
    for (const auto& [idx, s] : streamers_) {
        result.emplace_back(idx, s->telemetry());
    }
    return result;
}

} // namespace cgc_moe
