/**
 * @file expert_streamer_gpu.cpp
 * @brief GPU Zero-Copy Integration & PD Separation Implementation
 */

#include "expert_streamer_gpu.h"
#include <algorithm>
#include <numeric>

namespace expert_streamer {
namespace gpu {

// ============================================================================
// GpuExpertCache Implementation
// ============================================================================

void GpuExpertCache::init(VkDevice_T* device, uint64_t max_total_bytes) {
    device_ = device;
    max_total_bytes_ = max_total_bytes;
    total_used_ = 0;
    cache_.clear();
    printf("[GpuExpertCache] Initialized: max=%llu bytes (%llu MB)\n",
           (unsigned long long)max_total_bytes_,
           (unsigned long long)(max_total_bytes_ / (1024 * 1024)));
}

bool GpuExpertCache::upload_expert(const ExpertSlice& slice, ExpertGpuBuffer& out_buffer) {
    // Calculate total size
    uint64_t total_size = 0;
    if (slice.has_gate) total_size += slice.gate.size;
    if (slice.has_up) total_size += slice.up.size;
    if (slice.has_down) total_size += slice.down.size;

    if (total_size == 0) return false;

    // In a real implementation, this would:
    // 1. Create VkBuffer with VK_BUFFER_USAGE_TRANSFER_DST_BIT
    // 2. Allocate VkDeviceMemory with VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    // 3. Copy mmap'd data to GPU via vkCmdUpdateBuffer or staging buffer
    //
    // For now, we use a simplified CPU-side simulation for testing:

    out_buffer.size = total_size;
    out_buffer.device = device_;
    out_buffer.buffer = nullptr;  // Would be actual VkBuffer
    out_buffer.memory = nullptr;  // Would be actual VkDeviceMemory
    out_buffer.mapped = false;
    out_buffer.mapped_ptr = nullptr;

    // Track memory usage
    total_used_ += total_size;

    return true;
}

ExpertGpuBuffer* GpuExpertCache::get_or_upload(int layer, int expert_id, const ExpertSlice& slice) {
    std::string key = make_key(layer, expert_id);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.last_used = ++clock_;
        return &it->second.buffer;
    }

    // Evict if needed
    if (total_used_ >= max_total_bytes_) {
        evict();
    }

    // Upload
    GpuCacheEntry entry;
    entry.layer = layer;
    entry.expert_id = expert_id;
    entry.last_used = ++clock_;

    if (!upload_expert(slice, entry.buffer)) {
        return nullptr;
    }

    cache_[key] = entry;
    return &cache_[key].buffer;
}

void GpuExpertCache::evict() {
    if (cache_.empty()) return;

    // Find LRU entry
    std::string lru_key;
    uint64_t min_time = clock_ + 1;

    for (const auto& [key, entry] : cache_) {
        if (entry.last_used < min_time) {
            min_time = entry.last_used;
            lru_key = key;
        }
    }

    if (!lru_key.empty()) {
        total_used_ -= cache_[lru_key].buffer.size;
        cache_.erase(lru_key);
    }
}

GpuExpertCache::GpuCacheStats GpuExpertCache::get_stats() const {
    GpuCacheStats stats;
    stats.cached_experts = cache_.size();
    stats.memory_used_bytes = total_used_;
    stats.max_memory_bytes = max_total_bytes_;
    stats.gpu_utilization_percent = max_total_bytes_ > 0
        ? (double)total_used_ / max_total_bytes_ * 100.0 : 0.0;
    return stats;
}

std::string GpuExpertCache::make_key(int layer, int expert_id) const {
    return "L" + std::to_string(layer) + "_E" + std::to_string(expert_id);
}

// ============================================================================
// PDScheduler Implementation
// ============================================================================

void PDScheduler::init(UnifiedExpertStreamer* streamer,
                        GpuExpertCache* gpu_cache0,
                        GpuExpertCache* gpu_cache1) {
    streamer_ = streamer;
    gpu_cache_prefill_ = gpu_cache0;
    gpu_cache_decode_ = gpu_cache1;
    current_phase_ = PDPhase::IDLE;
    reset_stats();
    printf("[PDScheduler] Initialized\n");
}

void PDScheduler::start_prefill(int tokens, const std::vector<int>& expert_ids) {
    current_phase_ = PDPhase::PREFILL;
    stats_.prefill_tokens_processed = 0;

    // Get all layers
    const auto& layers = streamer_->get_layers();

    // Prewarm prefill layers (first half)
    int mid = layers.size() / 2;
    auto prefill_layers = std::vector<int>(layers.begin(), layers.begin() + mid);

    printf("[PDScheduler] Starting prefill: %d tokens, %zu layers\n",
           tokens, prefill_layers.size());

    // Load initial experts for all prefill layers
    for (int layer : prefill_layers) {
        load_experts_for_layer(layer, expert_ids);
    }

    stats_.prefill_time_ms = 0;
}

void PDScheduler::process_prefill_token(int token_id, const std::vector<int>& expert_ids) {
    if (current_phase_ != PDPhase::PREFILL) return;

    const auto& layers = streamer_->get_layers();
    int mid = layers.size() / 2;

    // For prefill, load experts for all prefill layers
    for (int i = 0; i < mid; i++) {
        load_experts_for_layer(layers[i], expert_ids);
    }

    stats_.prefill_tokens_processed++;
}

void PDScheduler::switch_to_decode(int last_prefill_token) {
    if (current_phase_ != PDPhase::PREFILL) return;

    current_phase_ = PDPhase::DECODE;
    stats_.expert_switches++;

    const auto& layers = streamer_->get_layers();
    int mid = layers.size() / 2;

    // Invalidate prefill layer caches
    for (int i = 0; i < mid; i++) {
        streamer_->invalidate_layer(layers[i]);
    }

    // Prewarm decode layers (second half)
    auto decode_layers = std::vector<int>(layers.begin() + mid, layers.end());
    printf("[PDScheduler] Switching to decode: %zu layers\n", decode_layers.size());

    // Load default experts for decode layers
    std::vector<int> default_experts = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int layer : decode_layers) {
        load_experts_for_layer(layer, default_experts);
    }
}

void PDScheduler::process_decode_token(int token_id, const std::vector<int>& expert_ids) {
    if (current_phase_ != PDPhase::DECODE) return;

    const auto& layers = streamer_->get_layers();
    int mid = layers.size() / 2;

    // For decode, only load experts for decode layers
    for (int i = mid; i < (int)layers.size(); i++) {
        load_experts_for_layer(layers[i], expert_ids);
    }

    stats_.decode_tokens_processed++;
}

PDScheduler::PDStats PDScheduler::get_stats() const {
    return stats_;
}

void PDScheduler::reset_stats() {
    stats_.prefill_tokens_processed = 0;
    stats_.decode_tokens_processed = 0;
    stats_.expert_switches = 0;
    stats_.avg_expert_load_time_ms = 0;
    stats_.prefill_time_ms = 0;
    stats_.decode_time_ms = 0;
    stats_.expert_load_times.clear();
}

void PDScheduler::load_experts_for_layer(int layer, const std::vector<int>& expert_ids) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int eid : expert_ids) {
        ExpertSlice slice;
        if (streamer_->load_expert(layer, eid, slice)) {
            update_layer_state(layer, eid);

            // Upload to GPU cache if available
            GpuExpertCache* gpu_cache = (current_phase_ == PDPhase::PREFILL)
                ? gpu_cache_prefill_ : gpu_cache_decode_;
            if (gpu_cache) {
                gpu_cache->get_or_upload(layer, eid, slice);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    stats_.expert_load_times.push_back(elapsed);

    // Update average
    if (!stats_.expert_load_times.empty()) {
        double sum = std::accumulate(stats_.expert_load_times.begin(),
                                      stats_.expert_load_times.end(), 0.0);
        stats_.avg_expert_load_time_ms = sum / stats_.expert_load_times.size();
    }
}

void PDScheduler::update_layer_state(int layer, int expert_id) {
    LayerState& state = layer_states_[layer];
    if (state.current_expert != expert_id) {
        state.current_expert = expert_id;
        state.last_switch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Add to cached experts if not present
        if (std::find(state.cached_experts.begin(), state.cached_experts.end(),
                       expert_id) == state.cached_experts.end()) {
            state.cached_experts.push_back(expert_id);
        }
    }
}

// ============================================================================
// DynamicExpertScheduler Implementation
// ============================================================================

std::vector<DynamicExpertScheduler::SchedulingDecision>
DynamicExpertScheduler::schedule(
    const std::vector<std::pair<int, int>>& token_router,
    const std::vector<int>& current_layers) {

    std::vector<SchedulingDecision> decisions;

    for (const auto& [layer, expert_id] : token_router) {
        SchedulingDecision decision;
        decision.layer = layer;
        decision.expert_id = expert_id;
        decision.should_prefetch = true;

        // Simple heuristic: use current layer's GPU
        // In real implementation, this would be more sophisticated
        decision.target_gpu = (layer < 15) ? 0 : 1;  // First half on GPU 0, second on GPU 1

        decisions.push_back(decision);
    }

    return decisions;
}

std::vector<int> DynamicExpertScheduler::predict_prefetch(
    const std::vector<int>& recent_experts) {

    // Update history
    for (int eid : recent_experts) {
        recent_experts_.push_back(eid);
        if ((int)recent_experts_.size() > MAX_HISTORY) {
            recent_experts_.erase(recent_experts_.begin());
        }
    }

    // Predict: return most frequent experts
    // In real implementation, this would use a learned predictor
    if (recent_experts_.empty()) {
        return {0, 1, 2, 3};  // Default to first 4
    }

    // Count frequency
    std::unordered_map<int, int> freq;
    for (int eid : recent_experts_) {
        freq[eid]++;
    }

    // Sort by frequency
    std::vector<std::pair<int, int>> freq_vec(freq.begin(), freq.end());
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top 8 predictions
    std::vector<int> predictions;
    for (size_t i = 0; i < std::min((size_t)8, freq_vec.size()); i++) {
        predictions.push_back(freq_vec[i].first);
    }

    return predictions;
}

// ============================================================================
// C API Implementation
// ============================================================================

struct GpuExpertCacheOpaque {
    GpuExpertCache impl;
};

struct PDSchedulerOpaque {
    PDScheduler impl;
};

extern "C" {

gpu_expert_cache_t gpu_expert_cache_create(void* vk_device, uint64_t max_bytes) {
    auto* cache = new (std::nothrow) GpuExpertCacheOpaque();
    if (cache) {
        cache->impl.init((VkDevice_T*)vk_device, max_bytes);
    }
    return cache;
}

void gpu_expert_cache_destroy(gpu_expert_cache_t cache) {
    delete cache;
}

pd_scheduler_t pd_scheduler_create(void* streamer, void* gpu0, void* gpu1) {
    auto* scheduler = new (std::nothrow) PDSchedulerOpaque();
    if (scheduler) {
        scheduler->impl.init((UnifiedExpertStreamer*)streamer,
                              (GpuExpertCache*)gpu0,
                              (GpuExpertCache*)gpu1);
    }
    return scheduler;
}

void pd_scheduler_destroy(pd_scheduler_t scheduler) {
    delete scheduler;
}

int pd_scheduler_start_prefill(pd_scheduler_t scheduler, int tokens, int* expert_ids, int n_experts) {
    if (!scheduler || !expert_ids) return -1;

    std::vector<int> ids(expert_ids, expert_ids + n_experts);
    scheduler->impl.start_prefill(tokens, ids);
    return 0;
}

int pd_scheduler_process_token(pd_scheduler_t scheduler, int token_id, int* expert_ids, int n_experts) {
    if (!scheduler || !expert_ids) return -1;

    std::vector<int> ids(expert_ids, expert_ids + n_experts);

    auto phase = scheduler->impl.get_phase();
    if (phase == PDPhase::PREFILL) {
        scheduler->impl.process_prefill_token(token_id, ids);
    } else if (phase == PDPhase::DECODE) {
        scheduler->impl.process_decode_token(token_id, ids);
    }

    return 0;
}

int pd_scheduler_switch_to_decode(pd_scheduler_t scheduler, int last_token) {
    if (!scheduler) return -1;
    scheduler->impl.switch_to_decode(last_token);
    return 0;
}

int pd_scheduler_get_phase(pd_scheduler_t scheduler) {
    if (!scheduler) return -1;
    switch (scheduler->impl.get_phase()) {
        case PDPhase::IDLE: return 0;
        case PDPhase::PREFILL: return 1;
        case PDPhase::DECODE: return 2;
        default: return -1;
    }
}

} // extern "C"

} // namespace gpu
} // namespace expert_streamer
