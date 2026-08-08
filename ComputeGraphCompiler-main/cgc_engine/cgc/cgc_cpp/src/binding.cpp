#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cgc_cpp.h"

namespace py = pybind11;

#ifdef CGC_METAL_ENABLED
extern "C" void cgc_install_vram_interception_hook();
extern "C" void* cgc_get_intercepted_kv_cache_ptr(size_t* out_size);
extern "C" void cgc_set_vram_interception_enabled(bool enabled);
extern "C" void cgc_preallocate_vram(size_t size);
extern "C" void cgc_enable_vram_preallocation(bool enable);
extern "C" void cgc_direct_write_all_vram(void* src_data, size_t size);
extern "C" void cgc_set_skip_tensor_set(bool skip);
#endif

PYBIND11_MODULE(cgc_cpp, m) {
    m.doc() = "CGC C++ SIMD Engine";

    m.def("init", []() { cgc_init(); }, "Initialize CGC C++ engine");
    m.def("destroy", []() { cgc_destroy(); }, "Destroy CGC C++ engine");
    m.def("has_opcode", &cgc_has_opcode, "Check if opcode is supported");

    m.def(
        "execute_opcode",
        [](
            int opcode,
            std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> inputs,
            py::dict params
        ) -> std::vector<py::array_t<float>> {
            std::vector<const float*> input_ptrs;
            std::vector<int64_t> input_dims;
            std::vector<int> input_ndims;

            for (const auto& arr : inputs) {
                auto buf = arr.request();
                input_ptrs.push_back(static_cast<const float*>(buf.ptr));
                for (ssize_t d = 0; d < buf.ndim; d++) {
                    input_dims.push_back(static_cast<int64_t>(buf.shape[d]));
                }
                input_ndims.push_back(buf.ndim);
            }

            if (!cgc_has_opcode(opcode)) {
                throw py::value_error("Opcode 0x" + py::str("{:02x}").format(opcode).cast<std::string>() + " not supported in C++ engine");
            }

            std::vector<py::array_t<float>> outputs;
            std::vector<float*> output_ptrs;
            std::vector<int64_t> output_dims(16);
            std::vector<int> output_ndims(4);

            // Linear/GEMM
            if (opcode == 0x20) { // LINEAR_GEMM
                int64_t m = input_dims[0];
                int64_t n = input_dims[2];
                auto out = py::array_t<float>({ m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x21) { // LINEAR_BIAS
                int64_t m = input_dims[0];
                int64_t n = input_dims[1];
                auto out = py::array_t<float>({ m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x22) { // GEMM_BATCHED
                int64_t batch = input_dims[0];
                int64_t m = input_dims[1];
                int64_t n = input_dims[3];
                auto out = py::array_t<float>({ batch, m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // Attention
            } else if (opcode == 0x10) { // ATTENTION_SDPA
                int64_t b = input_dims[0];
                int64_t h = input_dims[1];
                int64_t s = input_dims[2];
                int64_t d = input_dims[3];
                auto out = py::array_t<float>({ b, h, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x12 || opcode == 0x13) { // ATTENTION_PAGED, ATTENTION_FLASH
                int64_t b = input_dims[0];
                int64_t h = input_dims[1];
                int64_t s = input_dims[2];
                int64_t d = input_dims[3];
                auto out = py::array_t<float>({ b, h, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // Norm
            } else if (opcode == 0x30) { // LAYER_NORM
                int64_t b = input_dims[0];
                int64_t s = input_dims[1];
                int64_t d = input_dims[2];
                auto out = py::array_t<float>({ b, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x31) { // RMS_NORM
                int64_t b = input_dims[0];
                int64_t s = input_dims[1];
                int64_t d = input_dims[2];
                auto out = py::array_t<float>({ b, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x32) { // GROUP_NORM
                int64_t b = input_dims[0];
                int64_t c = input_dims[1];
                int64_t h = input_dims[2];
                int64_t w = input_dims[3];
                auto out = py::array_t<float>({ b, c, h, w });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // RoPE
            } else if (opcode == 0x40 || opcode == 0x41 || opcode == 0x42) {
                auto out = py::array_t<float>(inputs[0]);
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // Activation
            } else if (opcode >= 0x50 && opcode <= 0x54) { // SILU, GELU, GELU_TANH, RELU, SIGMOID
                auto out = py::array_t<float>(inputs[0]);
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // Sampling
            } else if (opcode == 0x60) { // SOFTMAX
                int64_t b = input_dims[0];
                int64_t s = input_dims[1];
                int64_t d = input_dims[2];
                auto out = py::array_t<float>({ b, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x61) { // LOG_SOFTMAX
                int64_t b = input_dims[0];
                int64_t s = input_dims[1];
                int64_t d = input_dims[2];
                auto out = py::array_t<float>({ b, s, d });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x62) { // TOP_K
                int64_t b = input_dims[0];
                auto out = py::array_t<float>(b * 10);
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x63) { // TOP_P
                int64_t b = input_dims[0];
                int64_t vocab_size = input_dims[1];
                auto out = py::array_t<float>({ b, vocab_size });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x64) { // TEMPERATURE
                int64_t b = input_dims[0];
                int64_t vocab_size = input_dims[1];
                auto out = py::array_t<float>({ b, vocab_size });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // Memory/KV Cache
            } else if (opcode == 0x70) { // KV_CACHE_LOAD
                int64_t num_indices = input_dims[1];
                int64_t elem_size = input_dims[2];
                auto out = py::array_t<float>({ num_indices, elem_size });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x71) { // KV_CACHE_STORE
                // 不需要输出
            } else if (opcode == 0x72) { // KV_CACHE_UPDATE
                int64_t cache_size = input_dims[0];
                int64_t elem_size = input_dims[2];
                auto out = py::array_t<float>({ cache_size, elem_size });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x73) { // EMBEDDING_LOOKUP
                int64_t num_indices = input_dims[2];
                int64_t embed_dim = input_dims[1];
                auto out = py::array_t<float>({ num_indices, embed_dim });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0x74 || opcode == 0x75) { // KV_CACHE_STATIC_LAYOUT, KV_CACHE_COMMIT
                // 不需要输出
            
            // Quantization
            } else if (opcode == 0xA0) { // QUANTIZE_W8A16
                int64_t m = input_dims[0];
                int64_t n = input_dims[1];
                auto out = py::array_t<float>({ m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0xA1) { // QUANTIZE_W4A16
                int64_t m = input_dims[0];
                int64_t n = input_dims[1];
                auto out = py::array_t<float>({ m, n / 2 });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0xA2) { // DEQUANTIZE
                int64_t size = input_dims[0];
                auto out = py::array_t<float>(size);
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else if (opcode == 0xA3 || opcode == 0xA4) { // GPTQ_KERNEL, AWQ_KERNEL
                int64_t m = input_dims[0];
                int64_t n = input_dims[1];
                auto out = py::array_t<float>({ m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            
            // LLAMA.CPP
            } else if (opcode == 0xC3) { // LLAMA_Q4_K_MATMUL
                int64_t m = input_dims[0];
                int64_t n = input_dims[2];
                auto out = py::array_t<float>({ m, n });
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            } else {
                auto out = py::array_t<float>(inputs[0]);
                outputs.push_back(out);
                output_ptrs.push_back(static_cast<float*>(out.request().ptr));
            }

            // KDA params (used for opcode 0x11)
            void* actual_params = nullptr;

            cgc_error_t err = cgc_execute_opcode(
                opcode,
                input_ptrs.data(), input_dims.data(), input_ndims.data(), inputs.size(),
                output_ptrs.data(), output_dims.data(), output_ndims.data(), outputs.size(),
                actual_params
            );

            if (err != CGC_OK) {
                throw std::runtime_error("CGC C++ execution failed");
            }

            return outputs;
        },
        py::arg("opcode"),
        py::arg("inputs"),
        py::arg("params") = py::dict(),
        "Execute CGC opcode with C++ SIMD engine"
    );

    m.def("inject_strategy",
        [](int backend, int tile_m, int tile_n, int tile_k, int attn_block, bool enable_fusion) -> bool {
            cgc_strategy_t strategy = {};
            strategy.backend = static_cast<cgc_backend_t>(backend);
            strategy.tile_config.tile_m = tile_m;
            strategy.tile_config.tile_n = tile_n;
            strategy.tile_config.tile_k = tile_k;
            strategy.tile_config.attn_block = attn_block;
            strategy.enable_op_fusion = enable_fusion;
            cgc_error_t err = cgc_inject_strategy(&strategy);
            return err == CGC_OK;
        },
        py::arg("backend"),
        py::arg("tile_m"),
        py::arg("tile_n"),
        py::arg("tile_k"),
        py::arg("attn_block"),
        py::arg("enable_fusion"),
        "Inject compilation strategy into C++ engine"
    );

    m.def("get_strategy",
        []() -> std::tuple<int, int, int, int, int, bool> {
            cgc_strategy_t strategy = {};
            cgc_error_t err = cgc_get_strategy(&strategy);
            if (err != CGC_OK) {
                return std::make_tuple(0, 128, 128, 128, 128, true);
            }
            return std::make_tuple(
                static_cast<int>(strategy.backend),
                strategy.tile_config.tile_m,
                strategy.tile_config.tile_n,
                strategy.tile_config.tile_k,
                strategy.tile_config.attn_block,
                strategy.enable_op_fusion
            );
        },
        "Get current strategy from C++ engine"
    );

    m.def("reset_strategy",
        []() -> bool {
            cgc_error_t err = cgc_reset_strategy();
            return err == CGC_OK;
        },
        "Reset strategy to default"
    );

    m.def("set_backend",
        [](int backend) -> bool {
            return cgc_set_backend(static_cast<cgc_backend_t>(backend));
        },
        py::arg("backend"),
        "Set C++ backend (0=auto, 1=cpu, 2=cuda, 3=metal)"
    );

    m.def("get_current_backend",
        []() -> int {
            return static_cast<int>(cgc_get_current_backend());
        },
        "Get current backend ID"
    );

    // Hardware Bus API
    m.def("mmap_file", [](std::string filepath) -> py::tuple {
        size_t size = 0;
        void* ptr = cgc_mmap_file(filepath.c_str(), &size);
        return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
    }, py::arg("filepath"), "Memory map a file returning pointer and size");

    m.def("munmap_file", [](uintptr_t ptr, size_t size) {
        cgc_munmap_file(reinterpret_cast<void*>(ptr), size);
    }, py::arg("ptr"), py::arg("size"), "Unmap a memory mapped file");

    m.def("install_vram_interception_hook", []() {
#ifdef CGC_METAL_ENABLED
        cgc_install_vram_interception_hook();
#endif
    }, "Install Custom Backend hook to intercept KV cache allocation");

    m.def("set_vram_interception_enabled", [](bool enabled) {
#ifdef CGC_METAL_ENABLED
        cgc_set_vram_interception_enabled(enabled);
#endif
    }, py::arg("enabled"), "Enable or disable KV cache interception during model load");

    m.def("get_intercepted_kv_cache_ptr", []() -> py::tuple {
#ifdef CGC_METAL_ENABLED
        size_t size = 0;
        void* ptr = cgc_get_intercepted_kv_cache_ptr(&size);
        return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
#else
        return py::make_tuple(0, 0);
#endif
    }, "Get the intercepted KV cache physical memory pointer and size");

    m.def("preallocate_vram", [](size_t size) {
#ifdef CGC_METAL_ENABLED
        cgc_preallocate_vram(size);
#endif
    }, py::arg("size"), "Pre-allocate a Metal buffer pool");

    m.def("enable_vram_preallocation", [](bool enable) {
#ifdef CGC_METAL_ENABLED
        cgc_enable_vram_preallocation(enable);
#endif
    }, py::arg("enable"), "Enable using the pre-allocated Metal buffer pool");

    m.def("direct_write_vram", [](uintptr_t src_ptr, size_t size) {
#ifdef CGC_METAL_ENABLED
        cgc_direct_write_all_vram(reinterpret_cast<void*>(src_ptr), size);
#endif
    }, py::arg("src_ptr"), py::arg("size"), "Directly overwrite VRAM bypassing llama.cpp serialization");

    m.def("set_skip_tensor_set", [](bool skip) {
#ifdef CGC_METAL_ENABLED
        cgc_set_skip_tensor_set(skip);
#endif
    }, py::arg("skip"), "Skip CPU memcpy during set_tensor for 0-copy VRAM injection");

    printf("[CGC C++] Module loaded with %d opcodes!\n", 40);
}