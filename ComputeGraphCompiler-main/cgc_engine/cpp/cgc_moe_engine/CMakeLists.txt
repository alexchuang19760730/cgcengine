# CGC MoE Engine - CMake 建置
#
# 編譯選項：
#   -DCGC_WITH_DEEPEP=ON      啟用 DeepEP（需要 deep_ep 庫）
#   -DCGC_WITH_DEEPGEMM=ON    啟用 DeepGEMM（需要 deep_gemm 庫）
#   -DCGC_BUILD_METAL=ON      啟用 Metal 後端（macOS）
#
# 使用方式：
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
#   make -j

cmake_minimum_required(VERSION 3.20)
project(cgc_moe_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 選項
option(CGC_WITH_DEEPEP "Enable DeepEP integration" OFF)
option(CGC_WITH_DEEPGEMM "Enable DeepGEMM integration" OFF)
option(CGC_BUILD_METAL "Build Metal backend (macOS)" OFF)
option(CGC_BUILD_TESTS "Build tests" OFF)

# 尋找 PyTorch
find_package(Torch REQUIRED)

# 尋找 Python（提供 Python.h 標頭）
find_package(Python3 COMPONENTS Development REQUIRED)

# 編譯定義
add_compile_definitions(${TORCH_CXX_FLAGS})
# 定義 TORCH_EXTENSION_NAME（bindings.cpp 的 PYBIND11_MODULE 依賴此巨集）
# 使用 torch.utils.cpp_extension 時會自動定義，裸 CMake 需手動指定
add_compile_definitions(TORCH_EXTENSION_NAME=cgc_moe_engine)
if(CGC_WITH_DEEPEP)
    add_compile_definitions(WITH_DEEPEP=1)
    message(STATUS "DeepEP integration: ON")
endif()
if(CGC_WITH_DEEPGEMM)
    add_compile_definitions(WITH_DEEPGEMM=1)
    message(STATUS "DeepGEMM integration: ON")
endif()

# 來源檔案
set(SOURCES
    cgc_moe_engine.cpp
    bindings.cpp
)

# CUDA 來源
if(CGC_WITH_DEEPEP OR CGC_WITH_DEEPGEMM)
    set(CUDA_SOURCES
        deepep_kernels.cu
        deepgemm_kernels.cu
    )
endif()

# 建立 extension 模組
add_library(cgc_moe_engine MODULE ${SOURCES} ${CUDA_SOURCES})

# 連結（包含 torch_python：提供 pybind11 的 at::Tensor type_caster）
# libtorch_python.so 位於 torch/lib/，不在系統連結路徑，需用完整路徑
find_library(TORCH_PYTHON_LIB torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH
)
target_link_libraries(cgc_moe_engine PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIB}")

# Python 標頭 include 路徑（PyTorch C++ API 依賴 Python.h）
target_include_directories(cgc_moe_engine PRIVATE
    "${Python3_INCLUDE_DIRS}"
    "${TORCH_INCLUDE_DIRS}"
)

# 設定輸出
set_target_properties(cgc_moe_engine PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Metal 後端（macOS）
if(CGC_BUILD_METAL AND APPLE)
    add_subdirectory(metal_backend)
endif()

# 測試
if(CGC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "CGC MoE Engine configured")
message(STATUS "  DeepEP: ${CGC_WITH_DEEPEP}")
message(STATUS "  DeepGEMM: ${CGC_WITH_DEEPGEMM}")
message(STATUS "  Backend: ${TORCH_DEVICE_TYPE}")
