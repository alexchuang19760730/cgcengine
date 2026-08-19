# CMakeLists.txt — Expert Streaming + Repack + IQ3_M
# 原位於 cgc_engine/cpp/expert_streaming（Windows port of turbo-fieldfare），
# 2026-08-14 整理進 cgc_engine/cgc/expert_streaming，並已在 macOS (M4, gcc/clang) 驗證可編譯。
cmake_minimum_required(VERSION 3.16)
project(cgc_expert_streaming C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(ES_SRC ${CMAKE_CURRENT_SOURCE_DIR})

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(CGC_LINK_GGML "Link llama.cpp ggml for IQ3_M quantization" OFF)
option(CGC_BUILD_TESTS "Build test programs" ON)
option(CGC_EXPERT_STREAMING_WITH_PYTORCH "Enable PyTorch integration (expert_compute_torch)" OFF)
option(CGC_EXPERT_STREAMING_WITH_MOE_ENGINE "Link cgc_moe_engine for full MoE forward" OFF)
option(CGC_REAL_GGUF_TESTS "Build real-GGUF segment validation tests (need model files)" OFF)

# ----------------------------------------------------------------------------
# Expert Streaming Library (C implementation - primary)
# ----------------------------------------------------------------------------
add_library(cgc_expert_streaming STATIC
    ${ES_SRC}/src/cgc_expert_streamer.c         # C 版核心流式加载器
    ${ES_SRC}/src/cgc_expert_streamer_gguf.c    # C 版 GGUF layout 解析
    ${ES_SRC}/src/cgc_pd_scheduler.c            # C 版 PD 分离调度器
    ${ES_SRC}/src/cgc_expert_compute.c          # C 版计算桥接
    ${ES_SRC}/src/cgc_gguf_lite.c               # 轻量 GGUF v3 header 解析器
)

# 兼容层: C++ wrapper (保留原有接口以支持现有 C++ 代码)
add_library(cgc_expert_streaming_cpp STATIC
    ${ES_SRC}/src/expert_streamer.cpp
    ${ES_SRC}/src/expert_streamer_gguf.cpp
    ${ES_SRC}/src/expert_compute.cpp
    ${ES_SRC}/src/pd_expert_scheduler.cpp
)
target_link_libraries(cgc_expert_streaming_cpp PUBLIC cgc_expert_streaming)

target_include_directories(cgc_expert_streaming PUBLIC
    ${ES_SRC}/include
)

if(WIN32)
    target_link_libraries(cgc_expert_streaming PUBLIC kernel32)
    target_compile_definitions(cgc_expert_streaming PRIVATE _WIN32_WINNT=0x0602)
endif()

# ----------------------------------------------------------------------------
# Repack Library (safetensors → per-layer GGUF + IQ3_M)
# ----------------------------------------------------------------------------
add_library(cgc_repack STATIC
    ${ES_SRC}/src/cgc_safetensors.c
    ${ES_SRC}/src/cgc_arch_info.c
    ${ES_SRC}/src/cgc_repack_planner.c
    ${ES_SRC}/src/cgc_repack.c
    ${ES_SRC}/src/cgc_quantize.c
)

target_include_directories(cgc_repack PUBLIC
    ${ES_SRC}/include
)

if(WIN32)
    target_link_libraries(cgc_repack PUBLIC kernel32)
endif()

# ----------------------------------------------------------------------------
# GGML integration (IQ3_M quantization)
# ----------------------------------------------------------------------------
if(CGC_LINK_GGML)
    set(LLAMA_CPP_DIR ${ES_SRC}/../../../Backend/Llama.cpp/llama.cpp)

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
#   注意: torch/DeepEP 路徑為 CUDA-only（Windows/Linux+NVIDIA），macOS 上請改用 llama.cpp+Metal compute
# ----------------------------------------------------------------------------
if(CGC_EXPERT_STREAMING_WITH_PYTORCH)
    find_package(Torch REQUIRED)

    add_library(cgc_expert_streaming_torch STATIC
        ${ES_SRC}/src/expert_compute_torch.cpp
    )
    target_include_directories(cgc_expert_streaming_torch PUBLIC
        ${ES_SRC}/include
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(cgc_expert_streaming_torch PUBLIC
        cgc_expert_streaming
        ${TORCH_LIBRARIES}
    )
    target_compile_definitions(cgc_expert_streaming_torch PUBLIC CGC_HAS_PYTORCH)

    # 可选: 链接 cgc_moe_engine (启用 moeForward 端到端)
    if(CGC_EXPERT_STREAMING_WITH_MOE_ENGINE)
        set(CGC_MOE_ENGINE_DIR ${ES_SRC}/../cgc_moe_engine)
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
enable_testing()
if(CGC_BUILD_TESTS)
    # C 版测试 (推荐)
    add_executable(test_cgc_streamer
        ${ES_SRC}/tests/test_cgc_expert_streamer.c
    )
    target_link_libraries(test_cgc_streamer PRIVATE cgc_expert_streaming)
    add_test(NAME cgc_streamer COMMAND test_cgc_streamer)

    add_executable(test_cgc_pd_scheduler
        ${ES_SRC}/tests/test_cgc_pd_scheduler.c
    )
    target_link_libraries(test_cgc_pd_scheduler PRIVATE cgc_expert_streaming)
    add_test(NAME cgc_pd_scheduler COMMAND test_cgc_pd_scheduler)

    add_executable(test_cgc_gguf_integration
        ${ES_SRC}/tests/test_cgc_gguf_integration.c
    )
    target_link_libraries(test_cgc_gguf_integration PRIVATE cgc_expert_streaming)
    add_test(NAME cgc_gguf_integration COMMAND test_cgc_gguf_integration)

    # 真檔 segment 驗證（qwen36 3 張量 / gemma4 合併 2 張量 byte-identical）
    # CGC_REAL_GGUF_TESTS=ON 才建；執行需提供真實 GGUF 路徑：
    #   -DCGC_QWEN36_GGUF=<qwen35moe IQ2/IQ3 gguf>
    #   -DCGC_GEMMA4_GGUF=<gemma4 gguf>
    # 未提供路徑時只建 binary 不註冊 ctest；提供後 ctest 直接跑。
    if(CGC_REAL_GGUF_TESTS)
        add_executable(test_real_qwen35moe
            ${ES_SRC}/tests/test_real_qwen35moe.c
        )
        target_link_libraries(test_real_qwen35moe PRIVATE cgc_expert_streaming)

        add_executable(test_real_gemma4
            ${ES_SRC}/tests/test_real_gemma4.c
        )
        target_link_libraries(test_real_gemma4 PRIVATE cgc_expert_streaming)

        # 消費端端到端：create + load_experts 的 slot bytes vs 直接 pread 絕對定址
        add_executable(test_streamer_real_gemma4
            ${ES_SRC}/tests/test_streamer_real_gemma4.c
        )
        target_link_libraries(test_streamer_real_gemma4 PRIVATE cgc_expert_streaming)

        # fill 頻寬 bench（泛化雙家族）：data_start 修復的 layer-0 段表 vs parser 驗證
        # 也註冊為回歸：-for-synth 合成 trace 下若段表/parser 不一致會 abort（防定址回歸）
        add_executable(bench_gguf_fill
            ${ES_SRC}/tools/bench_gguf_fill.c
        )
        target_link_libraries(bench_gguf_fill PRIVATE cgc_expert_streaming)
        if(NOT WIN32)
            target_link_libraries(bench_gguf_fill PRIVATE m)
        endif()

        set(CGC_QWEN36_GGUF "" CACHE FILEPATH "Path to qwen35moe GGUF for test_real_qwen35moe")
        set(CGC_GEMMA4_GGUF "" CACHE FILEPATH "Path to gemma4 GGUF for test_real_gemma4")
        if(CGC_QWEN36_GGUF)
            add_test(NAME real_qwen35moe COMMAND test_real_qwen35moe ${CGC_QWEN36_GGUF})
            add_test(NAME bench_fill_qwen36 COMMAND bench_gguf_fill ${CGC_QWEN36_GGUF} -for-synth tokens=8)
        endif()
        if(CGC_GEMMA4_GGUF)
            add_test(NAME real_gemma4 COMMAND test_real_gemma4 ${CGC_GEMMA4_GGUF})
            add_test(NAME streamer_real_gemma4 COMMAND test_streamer_real_gemma4 ${CGC_GEMMA4_GGUF})
            add_test(NAME bench_fill_gemma4 COMMAND bench_gguf_fill ${CGC_GEMMA4_GGUF} -for-synth tokens=8)
        endif()
    endif()

    # C++ 版测试 (兼容保留, Windows-only: 依赖 windows.h/GetTempPathA, macOS 用 C 版测试)
    if(WIN32)
        add_executable(test_expert_streamer
            ${ES_SRC}/tests/test_expert_streamer.cpp
        )
        target_link_libraries(test_expert_streamer PRIVATE cgc_expert_streaming_cpp)

        add_executable(test_expert_streamer_gguf
            ${ES_SRC}/tests/test_expert_streamer_gguf.cpp
        )
        target_link_libraries(test_expert_streamer_gguf PRIVATE cgc_expert_streaming_cpp)

        add_executable(test_pd_scheduler
            ${ES_SRC}/tests/test_pd_scheduler.cpp
        )
        target_link_libraries(test_pd_scheduler PRIVATE cgc_expert_streaming_cpp)
    endif()
endif()

# ----------------------------------------------------------------------------
# Repack executable: safetensors → per-layer GGUF (+ IQ3_M if enabled)
# ----------------------------------------------------------------------------
add_executable(cgc_repack_exe
    ${ES_SRC}/src/cgc_repack_main.c
)
set_target_properties(cgc_repack_exe PROPERTIES OUTPUT_NAME cgc_repack)
target_link_libraries(cgc_repack_exe PRIVATE cgc_repack)

if(WIN32)
    target_link_libraries(cgc_repack_exe PRIVATE kernel32)
endif()
