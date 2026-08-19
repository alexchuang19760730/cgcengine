/**
 * @file expert_streamer.h
 * @brief Unified MoE Expert Streamer for llama.cpp (Header-Only)
 *
 * Supports both:
 * - Per-Expert layout (Qwen3.6 style): blk.X.expert.Y.role.weight
 * - Per-Layer layout (Gemma4 style): ffn_down_exps.weight [inter, hidden, num_experts]
 *
 * Integration with llama.cpp:
 *   1. Use existing gguf_context and mmap'd data pointer
 *   2. Compute expert slice offsets directly in mmap buffer
 *   3. Zero-copy GPU transfer via Vulkan buffer mapping
 *
 * Usage:
 *   gguf_context* ctx = ...; // already loaded
 *   ExpertStreamer streamer(ctx);
 *   ExpertSlice slice = streamer.load_expert(layer, expert_id);
 *   // slice.gate, slice.up, slice.down point into mmap'd memory
 */

#ifndef EXPERT_STREAMER_H
#define EXPERT_STREAMER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <algorithm>

// Forward declarations for gguf types
struct gguf_context;
struct gguf_tensor_info;
struct ggml_tensor;

namespace expert_streamer {

/**
 * @brief Layout types for MoE expert storage
 */
enum class ExpertLayout {
    PER_EXPERT,   // Qwen3.6: blk.X.expert.Y.role.weight
    PER_LAYER,    // Gemma4: ffn_down_exps.weight [inter, hidden, num_experts]
    UNKNOWN
};

/**
 * @brief Single expert weight slice result
 */
struct ExpertWeight {
    void* data;           // Pointer into mmap'd buffer (zero-copy)
    uint64_t size;        // Size in bytes
    int64_t dims[2];      // Logical dimensions [rows, cols]
    ggml_tensor* tensor;  // Associated ggml_tensor (optional, for dequant)
};

/**
 * @brief Complete expert weights (gate + up + down)
 */
struct ExpertSlice {
    int layer;
    int expert_id;
    ExpertLayout layout;

    ExpertWeight gate;      // [hidden, inter] for Gemma4, or per-expert
    ExpertWeight up;        // [hidden, inter]
    ExpertWeight down;      // [inter, hidden]
    ExpertWeight gate_inp;  // Shared gate input projection (Gemma4 only)

    bool has_gate;
    bool has_up;
    bool has_down;
    bool has_gate_inp;
};

/**
 * @brief Cache entry for loaded expert
 */
struct CacheEntry {
    ExpertSlice slice;
    uint64_t last_access;  // For LRU tracking
};

/**
 * @brief Per-Layer Adapter (Gemma4 style)
 *
 * Handles sliced expert tensors where all experts are packed
 * into a single tensor with shape [dim0, dim1, num_experts].
 */
class PerLayerAdapter {
public:
    /**
     * @brief Initialize adapter from gguf context
     * @param ctx Already loaded gguf context with mmap'd data
     */
    void init(const gguf_context* ctx);

    /**
     * @brief Load expert weights by slicing from packed tensor
     * @param layer Layer index
     * @param expert_id Expert ID within layer
     * @param out_slice Output slice structure
     */
    void load_expert(const gguf_context* ctx, int layer, int expert_id, ExpertSlice& out_slice);

    /**
     * @brief Get list of MoE layers
     */
    const std::vector<int>& get_layers() const { return moe_layers_; }

    /**
     * @brief Get number of experts in layer
     */
    int get_num_experts(int layer = 0) const;

private:
    struct LayerInfo {
        const gguf_tensor_info* down_info = nullptr;
        const gguf_tensor_info* gate_up_info = nullptr;
        const gguf_tensor_info* gate_inp_info = nullptr;
    };

    std::unordered_map<int, LayerInfo> layer_info_;
    std::vector<int> moe_layers_;
    int hidden_ = 0;
    int expert_inter_ = 0;
    int num_experts_ = 128;
    int top_k_ = 8;
};

/**
 * @brief Per-Expert Adapter (Qwen3.6 style)
 *
 * Each expert has separate tensors: blk.X.expert.Y.gate/up/down.weight
 */
class PerExpertAdapter {
public:
    void init(const gguf_context* ctx);
    void load_expert(const gguf_context* ctx, int layer, int expert_id, ExpertSlice& out_slice);
    const std::vector<int>& get_layers() const { return moe_layers_; }
    int get_num_experts(int layer = 0) const;

private:
    // Map: (layer, expert_id, role) -> tensor info pointer
    struct TensorKey {
        int layer;
        int expert_id;
        std::string role;

        bool operator==(const TensorKey& other) const {
            return layer == other.layer && expert_id == other.expert_id && role == other.role;
        }
    };

    struct TensorKeyHash {
        size_t operator()(const TensorKey& k) const {
            size_t h = std::hash<int>()(k.layer);
            h ^= std::hash<int>()(k.expert_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>()(k.role);
            return h;
        }
    };

    std::unordered_map<TensorKey, const gguf_tensor_info*, TensorKeyHash> offsets_;
    std::vector<int> moe_layers_;
    std::vector<int> all_experts_;
};

/**
 * @brief Unified Expert Streamer (main entry point)
 *
 * Auto-detects layout and provides unified load_expert() interface.
 * Operates on existing gguf_context and mmap'd data (zero-copy).
 */
class UnifiedExpertStreamer {
public:
    /**
     * @brief Initialize from existing gguf context
     * @param ctx Already loaded gguf context (with mmap'd data)
     */
    explicit UnifiedExpertStreamer(const gguf_context* ctx);

    /**
     * @brief Load expert weights (with LRU cache)
     * @param layer Layer index
     * @param expert_id Expert ID
     * @param out_slice Output (zero-copy pointers into mmap buffer)
     * @return true if expert was found and loaded
     */
    bool load_expert(int layer, int expert_id, ExpertSlice& out_slice);

    /**
     * @brief Load multiple experts for a layer
     * @param layer Layer index
     * @param expert_ids Vector of expert IDs to load
     * @param out_slices Output vector (will be resized)
     */
    void load_layer(int layer, const std::vector<int>& expert_ids,
                    std::vector<ExpertSlice>& out_slices);

    /**
     * @brief Prewarm prefill phase (first half of layers)
     * @return Number of experts loaded
     */
    int prewarm_prefill();

    /**
     * @brief Prewarm decode phase (second half of layers)
     * @return Number of experts loaded
     */
    int prewarm_decode();

    /**
     * @brief Invalidate cache for specific layer
     */
    void invalidate_layer(int layer);

    /**
     * @brief Get cache statistics
     */
    struct CacheStats {
        int hits;
        int misses;
        int cached_count;
        double hit_rate;
    };
    CacheStats get_cache_stats() const;

    /**
     * @brief Get memory savings estimate
     */
    struct MemoryEstimate {
        double full_model_mb;
        double streaming_mb;
        double saving_percent;
        std::string layout_name;
    };
    MemoryEstimate get_memory_estimate() const;

    // Accessors
    ExpertLayout get_layout() const { return layout_; }
    const std::vector<int>& get_layers() const;
    int get_num_experts(int layer = 0) const;

    void set_max_cache_size(int size) { max_cache_size_ = size; }

private:
    const gguf_context* ctx_;
    ExpertLayout layout_;

    PerLayerAdapter per_layer_adapter_;
    PerExpertAdapter per_expert_adapter_;

    // LRU Cache
    std::unordered_map<std::string, CacheEntry> cache_;
    int max_cache_size_ = 64;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t clock_ = 0;

    std::string make_key(int layer, int expert_id) const;
    void evict_if_needed();
};

// Type aliases for C compatibility
using ExpertStreamer = UnifiedExpertStreamer;

} // namespace expert_streamer

// ============================================================================
// C API for llama.cpp integration
// ============================================================================

extern "C" {

typedef struct ExpertStreamerOpaque* expert_streamer_t;

/**
 * @brief Create expert streamer from gguf context
 * @param ctx Already loaded gguf context
 * @return Opaque handle
 */
expert_streamer_t expert_streamer_create(const gguf_context* ctx);

/**
 * @brief Destroy expert streamer
 */
void expert_streamer_destroy(expert_streamer_t streamer);

/**
 * @brief Load expert weights (zero-copy)
 * @param out_slice Caller-owned buffer to receive slice info
 * @return 0 on success, -1 if not found
 */
int expert_streamer_load_expert(
    expert_streamer_t streamer,
    int layer,
    int expert_id,
    expert_streamer::ExpertSlice* out_slice
);

/**
 * @brief Get layout type
 * @return 0=PER_EXPERT, 1=PER_LAYER, -1=UNKNOWN
 */
int expert_streamer_get_layout(expert_streamer_t streamer);

/**
 * @brief Get number of MoE layers
 */
int expert_streamer_get_num_layers(expert_streamer_t streamer);

/**
 * @brief Get number of experts in layer
 */
int expert_streamer_get_num_experts(expert_streamer_t streamer, int layer);

/**
 * @brief Prewarm prefill phase
 */
int expert_streamer_prewarm_prefill(expert_streamer_t streamer);

/**
 * @brief Prewarm decode phase
 */
int expert_streamer_prewarm_decode(expert_streamer_t streamer);

/**
 * @brief Invalidate layer cache
 */
void expert_streamer_invalidate_layer(expert_streamer_t streamer, int layer);

} // extern "C"

#endif // EXPERT_STREAMER_H
