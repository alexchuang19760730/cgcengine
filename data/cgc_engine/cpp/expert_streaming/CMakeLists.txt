# CMakeLists.txt — Expert Streaming + Repack + IQ3_M (Windows port of turbo-fieldfare)
cmake_minimum_required(VERSION 3.16)
project(cgc_expert_streaming C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(CGC_LINK_GGML "Link llama.cpp ggml for IQ3_M quantization" OFF)
option(CGC_BUILD_TESTS "Build test programs" ON)
option(CGC_EXPERT_STREAMING_WITH_PYTORCH "Enable PyTorch integration (expert_compute_torch)" OFF)
option(CGC_EXPERT_STREAMING_WITH_MOE_ENGINE "Link cgc_moe_engine for full MoE forward" OFF)

# ----------------------------------------------------------------------------
# Expert Streaming Library (C implementation - primary)
# ----------------------------------------------------------------------------
add_library(cgc_expert_streaming STATIC
    cgc_expert_streamer.c         # C 版核心流式加载器
    cgc_expert_streamer_gguf.c    # C 版 GGUF layout 解析
    cgc_pd_scheduler.c            # C 版 PD 分离调度器
    cgc_expert_compute.c          # C 版计算桥接
    cgc_gguf_lite.c               # 轻量 GGUF v3 header 解析器
    cgc_routing_controller.c      # C 版路由驱动专家预取控制器
)

# 兼容层: C++ wrapper (保留原有接口以支持现有 C++ 代码)
add_library(cgc_expert_streaming_cpp STATIC
    expert_streamer.cpp
    expert_streamer_gguf.cpp
    expert_compute.cpp
    pd_expert_scheduler.cpp
)
target_link_libraries(cgc_expert_streaming_cpp PUBLIC cgc_expert_streaming)

target_include_directories(cgc_expert_streaming PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
    target_link_libraries(cgc_expert_streaming PUBLIC kernel32)
    target_compile_definitions(cgc_expert_streaming PRIVATE _WIN32_WINNT=0x0602)
endif()

# ----------------------------------------------------------------------------
# Repack Library (safetensors → per-layer GGUF + IQ3_M)
# ----------------------------------------------------------------------------
add_library(cgc_repack STATIC
    cgc_safetensors.c
    cgc_arch_info.c
    cgc_repack_planner.c
    cgc_repack.c
    cgc_quantize.c
)

target_include_directories(cgc_repack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
    target_link_libraries(cgc_repack PUBLIC kernel32)
endif()

# ----------------------------------------------------------------------------
# GGML integration (IQ3_M quantization)
# ----------------------------------------------------------------------------
if(CGC_LINK_GGML)
    set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../Backend/Llama.cpp/llama.cpp)

    if(EXISTS ${LLAMA_CPP_DIR}/ggml/CMakeLists.txt)
        message(STATUS "Linking llama.cpp ggml for IQ3_M quantization")

        # 只编译 ggml CPU backend (不需要 CUDA/Metal/Vulkan)
        set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
        set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
        set(GGML_NATIVE ON CACHE BOOL "" FORCE)

        add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

        target_compile_definitions(cgc_repack PUBLIC CGC_LINK_GGML)
        target_link_libraries(cgc_repack PUBLIC ggml)
        target_include_directories(cgc_repack PUBLIC
            ${LLAMA_CPP_DIR}/ggml/include
        )
    else()
        message(WARNING "llama.cpp ggml not found at ${LLAMA_CPP_DIR}/ggml, IQ3_M disabled")
    endif()
endif()

# ----------------------------------------------------------------------------
# PyTorch integration (expert_compute_torch.cpp)
#   编译条件: CGC_EXPERT_STREAMING_WITH_PYTORCH=ON + 找到 PyTorch
#   用途: ExpertWeightsView → torch::Tensor → grouped_gemm_silu_bf16_forward
# ----------------------------------------------------------------------------
if(CGC_EXPERT_STREAMING_WITH_PYTORCH)
    find_package(Torch REQUIRED)

    add_library(cgc_expert_streaming_torch STATIC
        expert_compute_torch.cpp
    )
    target_include_directories(cgc_expert_streaming_torch PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(cgc_expert_streaming_torch PUBLIC
        cgc_expert_streaming
        ${TORCH_LIBRARIES}
    )
    target_compile_definitions(cgc_expert_streaming_torch PUBLIC CGC_HAS_PYTORCH)

    # 可选: 链接 cgc_moe_engine (启用 moeForward 端到端)
    if(CGC_EXPERT_STREAMING_WITH_MOE_ENGINE)
        set(CGC_MOE_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cgc_moe_engine)
        if(EXISTS ${CGC_MOE_ENGINE_DIR}/cgc_moe_engine.h)
            target_include_directories(cgc_expert_streaming_torch PUBLIC ${CGC_MOE_ENGINE_DIR})
            target_compile_definitions(cgc_expert_streaming_torch PUBLIC CGC_HAS_MOE_ENGINE)
            # 若 cgc_moe_engine 已编译为 lib, 取消注释:
            # target_link_libraries(cgc_expert_streaming_torch PUBLIC cgc_moe_engine)
            message(STATUS "expert_streaming_torch: linked cgc_moe_engine headers (moeForward enabled)")
        else()
            message(WARNING "cgc_moe_engine not found at ${CGC_MOE_ENGINE_DIR}, moeForward disabled")
        endif()
    endif()

    message(STATUS "expert_streaming_torch: PyTorch integration enabled")
endif()

# ----------------------------------------------------------------------------
# Test program: C 版测试
# ----------------------------------------------------------------------------
if(CGC_BUILD_TESTS)
    # C 版测试 (推荐)
    add_executable(test_cgc_streamer
        test_cgc_expert_streamer.c
    )
    target_link_libraries(test_cgc_streamer PRIVATE cgc_expert_streaming)

    add_executable(test_cgc_pd_scheduler
        test_cgc_pd_scheduler.c
    )
    target_link_libraries(test_cgc_pd_scheduler PRIVATE cgc_expert_streaming)

    add_executable(test_cgc_gguf_integration
        test_cgc_gguf_integration.c
    )
    target_link_libraries(test_cgc_gguf_integration PRIVATE cgc_expert_streaming)

    add_executable(test_cgc_routing_controller
        test_cgc_routing_controller.c
    )
    target_link_libraries(test_cgc_routing_controller PRIVATE cgc_expert_streaming)

    # C 版 llama-bench: 真实 GGUF 上测量 prefill/decode tok/s + memory
    add_executable(cgc_llama_bench
        cgc_llama_bench.c
    )
    target_link_libraries(cgc_llama_bench PRIVATE cgc_expert_streaming)

    if(WIN32)
        target_link_libraries(cgc_llama_bench PRIVATE psapi)
    endif()

    # C++ 版测试 (兼容保留)
    add_executable(test_expert_streamer
        test_expert_streamer.cpp
    )
    target_link_libraries(test_expert_streamer PRIVATE cgc_expert_streaming_cpp)

    add_executable(test_expert_streamer_gguf
        test_expert_streamer_gguf.cpp
    )
    target_link_libraries(test_expert_streamer_gguf PRIVATE cgc_expert_streaming_cpp)

    add_executable(test_pd_scheduler
        test_pd_scheduler.cpp
    )
    target_link_libraries(test_pd_scheduler PRIVATE cgc_expert_streaming_cpp)
endif()

# ----------------------------------------------------------------------------
# Repack executable: safetensors → per-layer GGUF (+ IQ3_M if enabled)
# ----------------------------------------------------------------------------
add_executable(cgc_repack_exe
    cgc_repack_main.c
)
set_target_properties(cgc_repack_exe PROPERTIES OUTPUT_NAME cgc_repack)
target_link_libraries(cgc_repack_exe PRIVATE cgc_repack)

if(WIN32)
    target_link_libraries(cgc_repack_exe PRIVATE kernel32)
endif()
