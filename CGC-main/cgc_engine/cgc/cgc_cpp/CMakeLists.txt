cmake_minimum_required(VERSION 3.20)
project(cgc_cpp)

option(CGC_ENABLE_CUDA "Enable CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_BUILD_TYPE Release)

if(CGC_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_definitions(-DCGC_CUDA_ENABLED)
endif()

# ------------------------------------------------------------------------------
# 🔥 Core Dependencies
# ------------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ------------------------------------------------------------------------------
# 📦 Include Directories
# ------------------------------------------------------------------------------
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Backend/Llama.cpp/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Backend/Llama.cpp/llama.cpp/ggml/src
)

# ------------------------------------------------------------------------------
# 🧠 GGUF from llama.cpp
# ------------------------------------------------------------------------------
set(GGUF_SOURCES
    src/gguf.c
)

# ------------------------------------------------------------------------------
# ⚙️ CGC C++ Core Engine
# ------------------------------------------------------------------------------
set(CGC_SOURCES
    src/cgc_cpp.cpp
    src/cgc_platform.cpp
    src/cgc_backend.cpp
    src/cgc_cpu_backend.cpp
    src/cgc_gguf.cpp
    src/magi_backend_unified.cpp
    src/anti_fraud.cpp
    src/binding.cpp
    src/kernels/linear.cpp
    src/kernels/attention.cpp
    src/kernels/quant.cpp
    src/kernels/quant_full.cpp
    src/kernels/rope.cpp
    src/kernels/rope_full.cpp
    src/kernels/norm.cpp
    src/kernels/norm_full.cpp
    src/kernels/ortho_kda_v4.cpp
    src/kernels/ortho_kda_v4_binding.cpp
    src/kernels/activation.cpp
    src/kernels/activation_full.cpp
    src/kernels/kv_cache.cpp
    src/kernels/sampling.cpp
    src/kernels/gemm_cpu.cpp
    ${GGUF_SOURCES}
)

if(CGC_ENABLE_CUDA)
    list(APPEND CGC_SOURCES src/cgc_cuda_backend.cpp)
    set_source_files_properties(
        ${CGC_SOURCES}
        PROPERTIES LANGUAGE CUDA
    )
endif()

# ------------------------------------------------------------------------------
# 🍎 Metal Support (Apple only)
# ------------------------------------------------------------------------------
if(APPLE AND NOT DISABLE_METAL)
    list(APPEND CGC_SOURCES src/cgc_metal_backend.mm src/cgc_metal_vram_hook.mm)
    add_definitions(-DCGC_METAL_ENABLED)

    # Metal Shader compilation
    set(METAL_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal/kda.metal)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

    if(NOT SKIP_METAL_COMPILE)
        add_custom_command(
            OUTPUT ${METAL_LIB}
            COMMAND xcrun -sdk macosx metal ${METAL_SHADERS} -o ${METAL_LIB}
            DEPENDS ${METAL_SHADERS}
        )
        add_custom_target(metal_lib ALL DEPENDS ${METAL_LIB})
    endif()

    find_library(METAL_FRAMEWORK Metal)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    set(METAL_LIBS ${METAL_FRAMEWORK} ${QUARTZCORE_FRAMEWORK} ${MPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

    message(STATUS "🔧 Metal support enabled")
endif()

# ------------------------------------------------------------------------------
# 🚀 pybind11 Module Output -> cgc_cpp.so
# ------------------------------------------------------------------------------
add_library(cgc_cpp MODULE ${CGC_SOURCES})

# Link Python
link_directories(${Python3_LIBRARY_DIRS})
target_link_libraries(cgc_cpp PRIVATE ${Python3_LIBRARIES})

# Link Metal libraries (Apple only)
if(APPLE AND METAL_LIBS)
    target_link_libraries(cgc_cpp PRIVATE ${METAL_LIBS})
    if(NOT SKIP_METAL_COMPILE)
        add_dependencies(cgc_cpp metal_lib)
    endif()
endif()

# Link NVML libraries (Linux/NVIDIA only)
if(UNIX AND NOT APPLE)
    find_library(NVML_LIBRARY nvml)
    if(NVML_LIBRARY)
        target_link_libraries(cgc_cpp PRIVATE ${NVML_LIBRARY})
        add_definitions(-DNVML_AVAILABLE=1)
        message(STATUS "🔧 NVML support enabled")
    else()
        add_definitions(-DNVML_AVAILABLE=0)
        message(STATUS "🔧 NVML not found, using engine估算值")
    endif()
endif()

if(CGC_ENABLE_CUDA)
    link_directories(/usr/local/cuda/lib64)
    target_link_libraries(cgc_cpp PRIVATE CUDA::cudart /root/flashkv0516/venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcublas.so.13)
endif()

# Set output properties
set_target_properties(cgc_cpp PROPERTIES 
    PREFIX "" 
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cgc_cpp PRIVATE -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|ARM64")
        target_compile_options(cgc_cpp PRIVATE -mcpu=apple-m1)
    endif()
endif()

# ------------------------------------------------------------------------------
# 📦 Expert Streaming Submodule (兄弟目錄: cgc/expert_streaming)
#    一次建置即可產出 cgc_expert_streaming / cgc_expert_streaming_cpp / cgc_repack 庫
#    （GGUF per-expert pread streaming + segment-aware layout + repack）
# ------------------------------------------------------------------------------
set(CGC_BUILD_TESTS ON CACHE BOOL "Build expert_streaming tests" FORCE)
add_subdirectory(../expert_streaming ${CMAKE_CURRENT_BINARY_DIR}/expert_streaming)

message(STATUS "✅ CGC C++ Engine configured")
message(STATUS "📦 Sources: ${CGC_SOURCES}")
