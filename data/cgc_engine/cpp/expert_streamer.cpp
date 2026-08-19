/**
 * @file expert_streamer.cpp
 * @brief Unified MoE Expert Streamer Implementation
 *
 * Uses ONLY public gguf API for integration with llama.cpp.
 * Compile standalone for testing, then integrate into llama.cpp build.
 */

#include "expert_streamer.h"

// Public gguf API (from gguf.h)
// These functions are the ONLY valid way to access gguf_context
extern "C" {
    size_t gguf_get_n_tensors(const gguf_context* ctx);
    const char* gguf_get_tensor_name(const gguf_context* ctx, int tensor_id);
    int64_t gguf_get_tensor_ne(const gguf_context* ctx, int tensor_id);
    const int64_t* gguf_get_tensor_nb(const gguf_context* ctx, int tensor_id);
    int gguf_get_tensor_type(const gguf_context* ctx, int tensor_id);
    uint64_t gguf_get_tensor_offset(const gguf_context* ctx, int tensor_id);
    uint64_t gguf_get_tensor_size(const gguf_context* ctx, int tensor_id);
    void* gguf_get_data(const gguf_context* ctx);

    const gguf_kv* gguf_find_kv(const gguf_context* ctx, const char* key);
}

namespace expert_streamer {

// ============================================================================
// Helper Functions
// ============================================================================

static int64_t tensor_rows(const gguf_context* ctx, int tensor_id) {
    // nb[0] is the slowest-varying dimension (rows)
    const int64_t* nb = gguf_get_tensor_nb(ctx, tensor_id);
    return nb ? nb[0] : 0;
}

static int64_t tensor_cols(const gguf_context* ctx, int tensor_id) {
    // nb[1] is the next dimension (cols)
    const int64_t* nb = gguf_get_tensor_nb(ctx, tensor_id);
    return nb ? nb[1] : 0;
}

static int tensor_ndims(const gguf_context* ctx, int tensor_id) {
    // We need to read this from the tensor info
    // Since gguf_get_tensor_ne only returns total elements,
    // we track n_dims ourselves
    return 2;  // Most expert tensors are 2D or 3D
}

// ============================================================================
// PerLayerAdapter Implementation
// ============================================================================

void PerLayerAdapter::init(const gguf_context* ctx) {
    if (!ctx) return;

    int n_tensors = (int)gguf_get_n_tensors(ctx);
    void* data_ptr = gguf_get_data(ctx);

    if (!data_ptr) {
        fprintf(stderr, "[PerLayerAdapter] No data pointer in context\n");
        return;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;

        std::string tensor_name(name);

        // Parse blk.X.role format
        if (tensor_name.substr(0, 4) != "blk.") continue;

        size_t dot_pos1 = tensor_name.find('.');
        if (dot_pos1 == std::string::npos) continue;

        size_t dot_pos2 = tensor_name.find('.', dot_pos1 + 1);
        if (dot_pos2 == std::string::npos) continue;

        std::string layer_str = tensor_name.substr(dot_pos1 + 1, dot_pos2 - dot_pos1 - 1);
        int layer = std::atoi(layer_str.c_str());
        std::string role = tensor_name.substr(dot_pos2 + 1);

        LayerInfo& li = layer_info_[layer];

        // Store tensor ID instead of pointer (for safe API usage)
        if (role == "ffn_down_exps") {
            li.down_tensor_id = i;
        } else if (role == "ffn_gate_up_exps") {
            li.gate_up_tensor_id = i;
        } else if (role == "ffn_gate_inp") {
            li.gate_inp_tensor_id = i;
        }

        moe_layers_.push_back(layer);
    }

    // Remove duplicates and sort
    std::sort(moe_layers_.begin(), moe_layers_.end());
    moe_layers_.erase(std::unique(moe_layers_.begin(), moe_layers_.end()), moe_layers_.end());

    // Read architecture params from KV
    const gguf_kv* kv_hidden = gguf_find_kv(ctx, "gemma4.embedding_length");
    if (kv_hidden) {
        hidden_ = (int)kv_hidden->value.uint32;
    }

    const gguf_kv* kv_inter = gguf_find_kv(ctx, "gemma4.expert_feed_forward_length");
    if (kv_inter) {
        expert_inter_ = (int)kv_inter->value.uint32;
    }

    const gguf_kv* kv_num = gguf_find_kv(ctx, "gemma4.expert_count");
    if (kv_num) {
        num_experts_ = (int)kv_num->value.uint32;
    }

    const gguf_kv* kv_topk = gguf_find_kv(ctx, "gemma4.expert_used_count");
    if (kv_topk) {
        top_k_ = (int)kv_topk->value.uint32;
    }

    printf("[PerLayerAdapter] Initialized: %zu layers, hidden=%d, inter=%d, experts=%d, top_k=%d\n",
           moe_layers_.size(), hidden_, expert_inter_, num_experts_, top_k_);
}

void PerLayerAdapter::load_expert(const gguf_context* ctx, int layer, int expert_id,
                                   ExpertSlice& out_slice) {
    out_slice.layer = layer;
    out_slice.expert_id = expert_id;
    out_slice.layout = ExpertLayout::PER_LAYER;
    out_slice.has_gate = false;
    out_slice.has_up = false;
    out_slice.has_down = false;
    out_slice.has_gate_inp = false;

    auto it = layer_info_.find(layer);
    if (it == layer_info_.end()) return;

    const LayerInfo& li = it->second;
    uint8_t* base_ptr = (uint8_t*)gguf_get_data(ctx);
    if (!base_ptr) return;

    // Load down projection
    if (li.down_tensor_id >= 0 && expert_id < num_experts_) {
        uint64_t total_size = gguf_get_tensor_size(ctx, li.down_tensor_id);
        uint64_t per_expert_size = total_size / num_experts_;
        uint64_t offset = gguf_get_tensor_offset(ctx, li.down_tensor_id)
                        + (uint64_t)expert_id * per_expert_size;

        out_slice.down.data = base_ptr + offset;
        out_slice.down.size = per_expert_size;

        int64_t rows = tensor_rows(ctx, li.down_tensor_id);
        int64_t cols = tensor_cols(ctx, li.down_tensor_id);
        out_slice.down.dims[0] = rows;
        out_slice.dims[1] = cols;
        out_slice.down.tensor = nullptr;
        out_slice.has_down = true;
    }

    // Load gate+up (packed, need to split)
    if (li.gate_up_tensor_id >= 0 && expert_id < num_experts_) {
        uint64_t total_size = gguf_get_tensor_size(ctx, li.gate_up_tensor_id);
        uint64_t per_expert_size = total_size / num_experts_;
        uint64_t offset = gguf_get_tensor_offset(ctx, li.gate_up_tensor_id)
                        + (uint64_t)expert_id * per_expert_size;

        uint8_t* expert_ptr = base_ptr + offset;

        // Calculate gate/up split
        // For IQ4_XS/Q4_K, we split at the midpoint of elements
        int64_t gate_rows = hidden_;
        int64_t gate_cols = expert_inter_;

        // Determine bytes per element from total size
        uint64_t total_elements = (uint64_t)tensor_rows(ctx, li.gate_up_tensor_id)
                                * (uint64_t)tensor_cols(ctx, li.gate_up_tensor_id);
        if (total_elements > 0) {
            double bpe = (double)per_expert_size / total_elements;
            uint64_t gate_elements = (uint64_t)gate_rows * gate_cols;
            uint64_t gate_bytes = (uint64_t)(gate_elements * bpe);

            // Gate part: first half
            out_slice.gate.data = expert_ptr;
            out_slice.gate.size = gate_bytes;
            out_slice.gate.dims[0] = gate_rows;
            out_slice.gate.dims[1] = gate_cols;
            out_slice.gate.tensor = nullptr;
            out_slice.has_gate = true;

            // Up part: second half
            out_slice.up.data = expert_ptr + gate_bytes;
            out_slice.up.size = per_expert_size - gate_bytes;
            out_slice.up.dims[0] = gate_rows;
            out_slice.up.dims[1] = gate_cols;
            out_slice.up.tensor = nullptr;
            out_slice.has_up = true;
        }
    }

    // Load shared gate input projection
    if (li.gate_inp_tensor_id >= 0) {
        uint64_t offset = gguf_get_tensor_offset(ctx, li.gate_inp_tensor_id);
        uint64_t size = gguf_get_tensor_size(ctx, li.gate_inp_tensor_id);

        out_slice.gate_inp.data = base_ptr + offset;
        out_slice.gate_inp.size = size;
        out_slice.gate_inp.dims[0] = tensor_rows(ctx, li.gate_inp_tensor_id);
        out_slice.gate_inp.dims[1] = tensor_cols(ctx, li.gate_inp_tensor_id);
        out_slice.gate_inp.tensor = nullptr;
        out_slice.has_gate_inp = true;
    }
}

int PerLayerAdapter::get_num_experts(int layer) const {
    return num_experts_;
}

// ============================================================================
// PerExpertAdapter Implementation
// ============================================================================

void PerExpertAdapter::init(const gguf_context* ctx) {
    if (!ctx) return;

    int n_tensors = (int)gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;

        std::string tensor_name(name);

        // Parse blk.X.expert.Y.role.weight
        if (tensor_name.find(".expert.") == std::string::npos) continue;
        if (tensor_name.substr(0, 4) != "blk.") continue;

        // Split by '.'
        std::vector<std::string> parts;
        size_t start = 0;
        size_t end;
        while ((end = tensor_name.find('.', start)) != std::string::npos) {
            parts.push_back(tensor_name.substr(start, end - start));
            start = end + 1;
        }
        parts.push_back(tensor_name.substr(start));

        if (parts.size() < 5 || parts[0] != "blk" || parts[2] != "expert") continue;

        int layer = std::atoi(parts[1].c_str());
        int expert_id = std::atoi(parts[3].c_str());
        std::string role = parts[4];
        // Remove ".weight" suffix if present
        if (role.find(".weight") != std::string::npos) {
            role = role.substr(0, role.find(".weight"));
        }

        TensorKey key{layer, expert_id, role};
        offsets_[key] = i;  // Store tensor ID

        moe_layers_.push_back(layer);
        all_experts_.push_back(expert_id);
    }

    std::sort(moe_layers_.begin(), moe_layers_.end());
    moe_layers_.erase(std::unique(moe_layers_.begin(), moe_layers_.end()), moe_layers_.end());

    std::sort(all_experts_.begin(), all_experts_.end());
    all_experts_.erase(std::unique(all_experts_.begin(), all_experts_.end()), all_experts_.end());

    printf("[PerExpertAdapter] Initialized: %zu layers, %zu experts total\n",
           moe_layers_.size(), all_experts_.size());
}

void PerExpertAdapter::load_expert(const gguf_context* ctx, int layer, int expert_id,
                                    ExpertSlice& out_slice) {
    out_slice.layer = layer;
    out_slice.expert_id = expert_id;
    out_slice.layout = ExpertLayout::PER_EXPERT;
    out_slice.has_gate = false;
    out_slice.has_up = false;
    out_slice.has_down = false;
    out_slice.has_gate_inp = false;

    uint8_t* base_ptr = (uint8_t*)gguf_get_data(ctx);
    if (!base_ptr) return;

    // Load gate, up, down separately
    for (const std::string& role : {"gate", "up", "down"}) {
        TensorKey key{layer, expert_id, role};
        auto it = offsets_.find(key);
        if (it == offsets_.end()) continue;

        int tensor_id = it->second;
        uint64_t offset = gguf_get_tensor_offset(ctx, tensor_id);
        uint64_t size = gguf_get_tensor_size(ctx, tensor_id);

        ExpertWeight* ew = nullptr;
        if (role == "gate") {
            ew = &out_slice.gate;
            out_slice.has_gate = true;
        } else if (role == "up") {
            ew = &out_slice.up;
            out_slice.has_up = true;
        } else if (role == "down") {
            ew = &out_slice.down;
            out_slice.has_down = true;
        }

        if (ew) {
            ew->data = base_ptr + offset;
            ew->size = size;
            ew->dims[0] = tensor_rows(ctx, tensor_id);
            ew->dims[1] = tensor_cols(ctx, tensor_id);
            ew->tensor = nullptr;
        }
    }
}

int PerExpertAdapter::get_num_experts(int layer) const {
    return (int)all_experts_.size();
}

// ============================================================================
// UnifiedExpertStreamer Implementation
// ============================================================================

UnifiedExpertStreamer::UnifiedExpertStreamer(const gguf_context* ctx)
    : ctx_(ctx), layout_(ExpertLayout::UNKNOWN) {

    if (!ctx) {
        fprintf(stderr, "UnifiedExpertStreamer: null context\n");
        return;
    }

    // Auto-detect layout
    int n_tensors = (int)gguf_get_n_tensors(ctx);
    bool found_per_layer = false;
    bool found_per_expert = false;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;
        std::string tensor_name(name);

        if (tensor_name.find("ffn_down_exps") != std::string::npos ||
            tensor_name.find("ffn_gate_up_exps") != std::string::npos) {
            found_per_layer = true;
        }

        if (tensor_name.find(".expert.") != std::string::npos) {
            found_per_expert = true;
        }
    }

    if (found_per_layer) {
        layout_ = ExpertLayout::PER_LAYER;
        per_layer_adapter_.init(ctx);
        printf("[UnifiedExpertStreamer] Layout: PER_LAYER (Gemma4 style)\n");
    } else if (found_per_expert) {
        layout_ = ExpertLayout::PER_EXPERT;
        per_expert_adapter_.init(ctx);
        printf("[UnifiedExpertStreamer] Layout: PER_EXPERT (Qwen3.6 style)\n");
    } else {
        layout_ = ExpertLayout::UNKNOWN;
        fprintf(stderr, "[UnifiedExpertStreamer] Unknown layout\n");
    }
}

bool UnifiedExpertStreamer::load_expert(int layer, int expert_id, ExpertSlice& out_slice) {
    std::string key = make_key(layer, expert_id);

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        hits_++;
        out_slice = it->second.slice;
        it->second.last_access = ++clock_;
        return true;
    }

    // Load from adapter
    misses_++;

    if (layout_ == ExpertLayout::PER_LAYER) {
        per_layer_adapter_.load_expert(ctx_, layer, expert_id, out_slice);
    } else if (layout_ == ExpertLayout::PER_EXPERT) {
        per_expert_adapter_.load_expert(ctx_, layer, expert_id, out_slice);
    } else {
        return false;
    }

    if (!out_slice.has_gate && !out_slice.has_up && !out_slice.has_down) {
        return false;
    }

    // Store in cache
    CacheEntry entry;
    entry.slice = out_slice;
    entry.last_access = ++clock_;
    cache_[key] = entry;

    evict_if_needed();
    return true;
}

void UnifiedExpertStreamer::load_layer(int layer, const std::vector<int>& expert_ids,
                                        std::vector<ExpertSlice>& out_slices) {
    out_slices.clear();
    out_slices.reserve(expert_ids.size());

    for (int eid : expert_ids) {
        ExpertSlice slice;
        if (load_expert(layer, eid, slice)) {
            out_slices.push_back(slice);
        }
    }
}

int UnifiedExpertStreamer::prewarm_prefill() {
    const std::vector<int>& layers = get_layers();
    int mid = layers.size() / 2;
    int warmed = 0;

    for (int i = 0; i < mid; i++) {
        int layer = layers[i];
        int top_k = (layout_ == ExpertLayout::PER_LAYER) ? per_layer_adapter_.top_k_ : 8;

        for (int e = 0; e < top_k; e++) {
            ExpertSlice slice;
            if (load_expert(layer, e, slice)) {
                warmed++;
            }
        }
    }

    printf("[UnifiedExpertStreamer] Prewarm prefill: %d experts loaded\n", warmed);
    return warmed;
}

int UnifiedExpertStreamer::prewarm_decode() {
    const std::vector<int>& layers = get_layers();
    int mid = layers.size() / 2;
    int warmed = 0;

    // Invalidate prefill layers first
    for (int i = 0; i < mid; i++) {
        invalidate_layer(layers[i]);
    }

    for (int i = mid; i < (int)layers.size(); i++) {
        int layer = layers[i];
        int top_k = (layout_ == ExpertLayout::PER_LAYER) ? per_layer_adapter_.top_k_ : 8;

        for (int e = 0; e < top_k; e++) {
            ExpertSlice slice;
            if (load_expert(layer, e, slice)) {
                warmed++;
            }
        }
    }

    printf("[UnifiedExpertStreamer] Prewarm decode: %d experts loaded\n", warmed);
    return warmed;
}

void UnifiedExpertStreamer::invalidate_layer(int layer) {
    std::string prefix = "L" + std::to_string(layer) + "_";
    for (auto it = cache_.begin(); it != cache_.end(); ) {
        if (it->first.find(prefix) == 0) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

UnifiedExpertStreamer::CacheStats UnifiedExpertStreamer::get_cache_stats() const {
    CacheStats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.cached_count = cache_.size();
    uint64_t total = hits_ + misses_;
    stats.hit_rate = total > 0 ? (double)hits_ / total * 100.0 : 0.0;
    return stats;
}

UnifiedExpertStreamer::MemoryEstimate UnifiedExpertStreamer::get_memory_estimate() const {
    MemoryEstimate est;
    est.full_model_mb = 0.0;
    est.streaming_mb = 0.0;
    est.saving_percent = 0.0;

    if (layout_ == ExpertLayout::PER_LAYER) {
        int top_k = per_layer_adapter_.top_k_;
        int num_experts = per_layer_adapter_.get_num_experts();
        int n_tensors = (int)gguf_get_n_tensors(ctx_);

        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(ctx_, i);
            if (!name) continue;
            std::string tensor_name(name);

            if (tensor_name.find("ffn_down_exps") != std::string::npos ||
                tensor_name.find("ffn_gate_up_exps") != std::string::npos) {
                uint64_t size = gguf_get_tensor_size(ctx_, i);
                est.full_model_mb += (double)size / (1024.0 * 1024.0);
                est.streaming_mb += (double)size * top_k / num_experts / (1024.0 * 1024.0);
            }
        }

        if (est.full_model_mb > 0) {
            est.saving_percent = (1.0 - est.streaming_mb / est.full_model_mb) * 100.0;
        }
        est.layout_name = "PER_LAYER (Gemma4)";
    } else {
        est.layout_name = "PER_EXPERT (Qwen3.6)";
    }

    return est;
}

const std::vector<int>& UnifiedExpertStreamer::get_layers() const {
    if (layout_ == ExpertLayout::PER_LAYER) {
        return per_layer_adapter_.get_layers();
    } else {
        return per_expert_adapter_.get_layers();
    }
}

int UnifiedExpertStreamer::get_num_experts(int layer) const {
    if (layout_ == ExpertLayout::PER_LAYER) {
        return per_layer_adapter_.get_num_experts(layer);
    } else {
        return per_expert_adapter_.get_num_experts(layer);
    }
}

std::string UnifiedExpertStreamer::make_key(int layer, int expert_id) const {
    return "L" + std::to_string(layer) + "_E" + std::to_string(expert_id);
}

void UnifiedExpertStreamer::evict_if_needed() {
    if ((int)cache_.size() > max_cache_size_) {
        // Find LRU entry
        std::string lru_key;
        uint64_t min_time = clock_ + 1;

        for (const auto& [key, entry] : cache_) {
            if (entry.last_access < min_time) {
                min_time = entry.last_access;
                lru_key = key;
            }
        }

        if (!lru_key.empty()) {
            cache_.erase(lru_key);
        }
    }
}

} // namespace expert_streamer
