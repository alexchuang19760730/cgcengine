/**
 * @file expert_streamer_gpu.h
 * @brief GPU Zero-Copy Integration for Expert Streaming
 *
 * Provides Vulkan/OpenCL buffer management for expert weights,
 * enabling zero-copy transfer from mmap'd GGUF data to GPU memory.
 */

#ifndef EXPERT_STREAMER_GPU_H
#define EXPERT_STREAMER_GPU_H

#include "expert_streamer.h"

// Forward declarations for Vulkan types
struct VkDevice_T;
struct VkBuffer_T;
struct VkDeviceMemory_T;

namespace expert_streamer {
namespace gpu {

/**
 * @brief GPU buffer for expert weights
 */
struct ExpertGpuBuffer {
    VkDevice_T* device = nullptr;
    VkBuffer_T* buffer = nullptr;
    VkDeviceMemory_T* memory = nullptr;
    uint64_t size = 0;
    bool mapped = false;
    void* mapped_ptr = nullptr;
};

/**
 * @brief GPU Expert Cache Manager
 *
 * Manages GPU-resident expert buffers with LRU eviction.
 * Supports both per-expert and per-layer layouts.
 */
class GpuExpertCache {
public:
    /**
     * @brief Initialize GPU cache
     * @param device Vulkan device
     * @param max_total_bytes Maximum GPU memory for expert cache
     */
    void init(VkDevice_T* device, uint64_t max_total_bytes);

    /**
     * @brief Upload expert weights to GPU (zero-copy from mmap)
     * @param slice Expert slice with mmap'd data pointers
     * @param out_gpu_buffer Resulting GPU buffer handle
     * @return true on success
     */
    bool upload_expert(const ExpertSlice& slice, ExpertGpuBuffer& out_gpu_buffer);

    /**
     * @brief Get or upload expert (cache-aware)
     * @return Pointer to GPU buffer, or nullptr if not available
     */
    ExpertGpuBuffer* get_or_upload(int layer, int expert_id, const ExpertSlice& slice);

    /**
     * @brief Evict oldest GPU buffer
     */
    void evict();

    /**
     * @brief Get current GPU memory usage
     */
    uint64_t get_memory_used() const { return total_used_; }

    /**
     * @brief Get cache statistics
     */
    struct GpuCacheStats {
        int cached_experts;
        uint64_t memory_used_bytes;
        uint64_t max_memory_bytes;
        double gpu_utilization_percent;
    };
    GpuCacheStats get_stats() const;

private:
    VkDevice_T* device_ = nullptr;
    uint64_t max_total_bytes_ = 0;
    uint64_t total_used_ = 0;

    struct GpuCacheEntry {
        ExpertGpuBuffer buffer;
        int layer;
        int expert_id;
        uint64_t last_used;
    };

    std::unordered_map<std::string, GpuCacheEntry> cache_;
    uint64_t clock_ = 0;

    std::string make_key(int layer, int expert_id) const;
};

// ============================================================================
// PD (Prefill-Decode) Separation Scheduler
// ============================================================================

/**
 * @brief PD Phase
 */
enum class PDPhase {
    PREFILL,  // Processing prompt tokens
    DECODE,   // Generating output tokens
    IDLE
};

/**
 * @brief PD Separation Configuration
 */
struct PDConfig {
    int prefill_gpu_id = 0;      // GPU for prefill phase
    int decode_gpu_id = 1;       // GPU for decode phase
    int context_switch_threshold = 512;  // Tokens to switch from prefill to decode
    bool enable_dynamic_switch = true;   // Auto-switch based on token count
};

/**
 * @brief PD Separation Scheduler
 *
 * Orchestrates expert loading across prefill/decode phases.
 * Supports multi-GPU setup with dynamic load balancing.
 */
class PDScheduler {
public:
    /**
     * @brief Initialize PD scheduler
     * @param streamer Expert streamer (on CPU)
     * @param gpu_cache0 GPU cache for prefill GPU
     * @param gpu_cache1 GPU cache for decode GPU
     */
    void init(UnifiedExpertStreamer* streamer,
              GpuExpertCache* gpu_cache0 = nullptr,
              GpuExpertCache* gpu_cache1 = nullptr);

    /**
     * @brief Start prefill phase
     * @param tokens Number of tokens in prompt
     * @param expert_ids Expert IDs needed for first token
     */
    void start_prefill(int tokens, const std::vector<int>& expert_ids);

    /**
     * @brief Process single token during prefill
     * @param token_id Current token
     * @param expert_ids Expert IDs needed for this token
     */
    void process_prefill_token(int token_id, const std::vector<int>& expert_ids);

    /**
     * @brief Switch to decode phase
     * @param last_prefill_token Last token processed in prefill
     */
    void switch_to_decode(int last_prefill_token);

    /**
     * @brief Process decode token
     * @param token_id Current decode token
     * @param expert_ids Expert IDs needed
     */
    void process_decode_token(int token_id, const std::vector<int>& expert_ids);

    /**
     * @brief Get current phase
     */
    PDPhase get_phase() const { return current_phase_; }

    /**
     * @brief Get performance statistics
     */
    struct PDStats {
        int prefill_tokens_processed;
        int decode_tokens_processed;
        int expert_switches;
        double avg_expert_load_time_ms;
        double prefill_time_ms;
        double decode_time_ms;
        std::vector<double> expert_load_times;
    };
    PDStats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

private:
    UnifiedExpertStreamer* streamer_ = nullptr;
    GpuExpertCache* gpu_cache_prefill_ = nullptr;
    GpuExpertCache* gpu_cache_decode_ = nullptr;

    PDPhase current_phase_ = PDPhase::IDLE;
    PDConfig config_;

    // Per-layer expert tracking
    struct LayerState {
        int current_expert = -1;
        std::vector<int> cached_experts;
        uint64_t last_switch_time;
    };
    std::unordered_map<int, LayerState> layer_states_;

    // Statistics
    PDStats stats_;

    void load_experts_for_layer(int layer, const std::vector<int>& expert_ids);
    void update_layer_state(int layer, int expert_id);
};

// ============================================================================
// Dynamic Scheduler
// ============================================================================

/**
 * @brief Dynamic expert scheduler based on token routing
 */
class DynamicExpertScheduler {
public:
    struct SchedulingDecision {
        int layer;
        int expert_id;
        bool should_prefetch;
        int target_gpu;
    };

    /**
     * @brief Schedule expert loading for next token
     * @param token_router Token → expert mapping (from router)
     * @param current_layers Currently active layers
     * @return Scheduling decisions
     */
    std::vector<SchedulingDecision> schedule(
        const std::vector<std::pair<int, int>>& token_router,  // (layer, expert_id) pairs
        const std::vector<int>& current_layers);

    /**
     * @brief Get prefetch predictions
     * @param recent_experts Recently used expert IDs
     * @return Predicted experts for next token
     */
    std::vector<int> predict_prefetch(const std::vector<int>& recent_experts);

private:
    // Simple prediction: cache most-recently-used experts
    std::vector<int> recent_experts_;
    static const int MAX_HISTORY = 32;
};

} // namespace gpu
} // namespace expert_streamer

// C API for GPU integration
extern "C" {

typedef struct GpuExpertCacheOpaque* gpu_expert_cache_t;
typedef struct PDSchedulerOpaque* pd_scheduler_t;

gpu_expert_cache_t gpu_expert_cache_create(void* vk_device, uint64_t max_bytes);
void gpu_expert_cache_destroy(gpu_expert_cache_t cache);

pd_scheduler_t pd_scheduler_create(void* streamer, void* gpu0, void* gpu1);
void pd_scheduler_destroy(pd_scheduler_t scheduler);

int pd_scheduler_start_prefill(pd_scheduler_t scheduler, int tokens, int* expert_ids, int n_experts);
int pd_scheduler_process_token(pd_scheduler_t scheduler, int token_id, int* expert_ids, int n_experts);
int pd_scheduler_switch_to_decode(pd_scheduler_t scheduler, int last_token);

int pd_scheduler_get_phase(pd_scheduler_t scheduler);

} // extern "C"

#endif // EXPERT_STREAMER_GPU_H
