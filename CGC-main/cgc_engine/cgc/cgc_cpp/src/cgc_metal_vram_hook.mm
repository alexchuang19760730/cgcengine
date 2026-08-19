#include <ggml-metal.h>
#include <ggml-backend-impl.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <cstring>

static ggml_backend_buffer_t (*g_original_metal_alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size) = nullptr;
static void (*g_original_set_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) = nullptr;

static void* g_intercepted_kv_base = nullptr;
static size_t g_intercepted_kv_size = 0;
static bool g_interception_enabled = true;
static bool g_skip_tensor_set = false;

extern "C" void cgc_set_skip_tensor_set(bool skip) {
    g_skip_tensor_set = skip;
}

static void cgc_metal_set_tensor_hook(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (g_skip_tensor_set && buffer->iface.get_base && buffer->iface.get_base(buffer) == g_intercepted_kv_base) {
        // [UMA 0-copy] Skip CPU memcpy because VRAM is already directly overwritten!
        return;
    }
    if (g_original_set_tensor) {
        g_original_set_tensor(buffer, tensor, data, offset, size);
    }
}

// Pre-allocation globals
static ggml_backend_buffer_type_t g_metal_buft = nullptr;
static ggml_backend_buffer_t g_preallocated_buf = nullptr;
static void* g_preallocated_base = nullptr;
static size_t g_preallocated_size = 0;
static bool g_use_prealloc = false;

// Tensor Direct Mapping globals
struct TensorMapInfo {
    std::string name;
    void* ptr;
    size_t size;
    size_t offset;
};
static std::vector<TensorMapInfo> g_intercepted_tensors;
static bool g_tensor_hook_enabled = false;

extern "C" void cgc_set_vram_interception_enabled(bool enabled) {
    g_interception_enabled = enabled;
}

extern "C" void cgc_enable_vram_preallocation(bool enable) {
    g_use_prealloc = enable;
}

extern "C" void cgc_enable_tensor_hook(bool enable) {
    g_tensor_hook_enabled = enable;
    if (enable) {
        g_intercepted_tensors.clear();
        printf("[CGC Tensor Hook] Enabled. Ready to intercept K/V cache tensors.\n");
    }
}

extern "C" void cgc_preallocate_vram(size_t size) {
    if (g_metal_buft && g_original_metal_alloc_buffer) {
        if (g_preallocated_buf) {
            if (size <= g_preallocated_size) return;
        }
        g_preallocated_buf = g_original_metal_alloc_buffer(g_metal_buft, size);
        g_preallocated_size = size;
        if (g_preallocated_buf && g_preallocated_buf->iface.get_base) {
            g_preallocated_base = g_preallocated_buf->iface.get_base(g_preallocated_buf);
        }
        printf("[CGC VRAM Hook] Pre-allocated VRAM Pool: %zu bytes at %p\n", size, g_preallocated_base);
    } else {
        printf("[CGC VRAM Hook] Error: Cannot preallocate, metal_buft is null.\n");
    }
}

// Hook for ggml_backend_tensor_alloc (This is a simplified conceptual hook, as actual ggml hooking requires modifying ggml.c or advanced dyld hooking)
// In a production environment, we intercept the buffer initialization instead.
extern "C" void cgc_register_tensor_offset(const char* name, void* ptr, size_t size, size_t offset) {
    if (g_tensor_hook_enabled) {
        g_intercepted_tensors.push_back({name ? name : "unknown", ptr, size, offset});
    }
}

extern "C" size_t cgc_get_tensor_count() {
    return g_intercepted_tensors.size();
}

extern "C" void cgc_direct_write_tensor(const char* name, void* src_data, size_t size) {
    if (!name || !src_data) return;
    std::string s_name(name);
    for (const auto& t : g_intercepted_tensors) {
        if (t.name == s_name && t.size >= size) {
            // Direct memory overwrite (RDMA style simulation)
            std::memcpy(t.ptr, src_data, size);
            printf("[CGC UMA 0-copy] Directly overwrote tensor %s (%zu bytes) at %p\n", name, size, t.ptr);
            return;
        }
    }
    printf("[CGC UMA 0-copy] Warning: Tensor %s not found for direct write.\n", name);
}

extern "C" void cgc_direct_write_all_vram(void* src_data, size_t size) {
    if (g_intercepted_kv_base && size <= g_intercepted_kv_size) {
        std::memcpy(g_intercepted_kv_base, src_data, size);
        printf("[CGC UMA 0-copy] 🚀 DIRECT VRAM OVERWRITE SUCCESS: %zu bytes at %p\n", size, g_intercepted_kv_base);
    } else {
        printf("[CGC UMA 0-copy] ❌ Failed: VRAM base is null or size mismatch.\n");
    }
}

static ggml_backend_buffer_t cgc_metal_alloc_buffer_hook(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buf = nullptr;
    
    if (g_interception_enabled && g_use_prealloc && g_preallocated_buf && size > 10 * 1024 * 1024 && size <= g_preallocated_size) {
        printf("[CGC VRAM Hook] Intercepted! Using Pre-allocated VRAM Pool for size: %zu bytes\n", size);
        buf = g_preallocated_buf;
        
        g_intercepted_kv_base = g_preallocated_base;
        g_intercepted_kv_size = size; // We use requested size for tracking
        
        // Reset so it's not reused by another tensor
        g_preallocated_buf = nullptr; 
    } else {
        // Call the original allocator to get physical VRAM (MTLBuffer)
        buf = g_original_metal_alloc_buffer(buft, size);
        
        // Only intercept if enabled from Python
        if (g_interception_enabled && size > 10 * 1024 * 1024) { 
            printf("[CGC VRAM Hook] Intercepted KV Cache allocation! Size: %zu bytes\n", size);
            if (buf && buf->iface.get_base) {
                g_intercepted_kv_base = buf->iface.get_base(buf);
                g_intercepted_kv_size = size;
                printf("[CGC VRAM Hook] Successfully captured physical MTLBuffer pointer: %p\n", g_intercepted_kv_base);
            }
        }
    }
    
    if (buf) {
        if (!g_original_set_tensor) {
            g_original_set_tensor = buf->iface.set_tensor;
        }
        buf->iface.set_tensor = cgc_metal_set_tensor_hook;
    }
    
    return buf;
}

extern "C" void cgc_install_vram_interception_hook() {
    if (g_original_metal_alloc_buffer != nullptr) {
        return; // Already installed
    }
    
    typedef ggml_backend_t (*ggml_backend_metal_init_t)(void);
    typedef ggml_backend_buffer_type_t (*ggml_backend_get_default_buffer_type_t)(ggml_backend_t);

    ggml_backend_metal_init_t p_ggml_backend_metal_init = (ggml_backend_metal_init_t)dlsym(RTLD_DEFAULT, "ggml_backend_metal_init");
    ggml_backend_get_default_buffer_type_t p_ggml_backend_get_default_buffer_type = (ggml_backend_get_default_buffer_type_t)dlsym(RTLD_DEFAULT, "ggml_backend_get_default_buffer_type");

    if (!p_ggml_backend_metal_init || !p_ggml_backend_get_default_buffer_type) {
        printf("[CGC VRAM Hook] Error: Failed to resolve ggml symbols via dlsym.\n");
        return;
    }

    ggml_backend_t metal_backend = p_ggml_backend_metal_init();
    if (!metal_backend) {
        printf("[CGC VRAM Hook] Error: ggml_backend_metal_init() failed.\n");
        return;
    }
    
    ggml_backend_buffer_type_t metal_buft = p_ggml_backend_get_default_buffer_type(metal_backend);
    if (metal_buft) {
        struct ggml_backend_buffer_type * buft = (struct ggml_backend_buffer_type *) metal_buft;
        g_original_metal_alloc_buffer = buft->iface.alloc_buffer;
        buft->iface.alloc_buffer = cgc_metal_alloc_buffer_hook;
        g_metal_buft = metal_buft; // Save for pre-allocation
        printf("[CGC VRAM Hook] Successfully installed Metal alloc_buffer hook for RDMA/PCIe direct write.\n");
    } else {
        printf("[CGC VRAM Hook] Error: ggml_backend_metal_buffer_type() returned null.\n");
    }
}

extern "C" void* cgc_get_intercepted_kv_cache_ptr(size_t* out_size) {
    if (out_size) {
        *out_size = g_intercepted_kv_size;
    }
    return g_intercepted_kv_base;
}
