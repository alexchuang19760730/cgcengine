#!/bin/bash
# =============================================================================
# rswaengine 统一构建脚本（纯 CPU / C++17，无需 CUDA / torch）
# 产出：
#   build/libcgc_unified.so   —— 供 C/C++ demo 与 Python ctypes 桥接共用
#   build/cgc_unified_demo    —— R-SWA + 窗口 OrthoKDA 功能演示
#   build/cgc_engine_demo     —— cgc_inject_strategy <-> 统一 IR 联动演示
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
INC_DIR="${SCRIPT_DIR}/include"
EX_DIR="${SCRIPT_DIR}/examples"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -fPIC -I${INC_DIR} -I${INC_DIR}/kernels"

echo "=============================================="
echo " rswaengine build (R-SWA + Window OrthoKDA)"
echo " CXX=${CXX}"
echo "=============================================="

echo "[1/3] 编译共享库 libcgc_unified.so ..."
"${CXX}" ${CXXFLAGS} -shared \
    "${SRC_DIR}/cgc_cpp.cpp" \
    "${SRC_DIR}/unified_ir.cpp" \
    "${SRC_DIR}/kernels/rswa_manager.cpp" \
    "${SRC_DIR}/kernels/window_ortho_kv_compressor.cpp" \
    -o "${BUILD_DIR}/libcgc_unified.so"
echo "  ✅ ${BUILD_DIR}/libcgc_unified.so"

echo "[2/3] 编译 demo: cgc_unified_demo ..."
"${CXX}" ${CXXFLAGS} \
    "${EX_DIR}/main_unified.cpp" \
    -L"${BUILD_DIR}" -lcgc_unified \
    -o "${BUILD_DIR}/cgc_unified_demo"

echo "[3/3] 编译 demo: cgc_engine_demo ..."
"${CXX}" ${CXXFLAGS} \
    "${EX_DIR}/main_engine.cpp" \
    -L"${BUILD_DIR}" -lcgc_unified \
    -o "${BUILD_DIR}/cgc_engine_demo"

echo ""
echo "▶ 运行 cgc_unified_demo ..."
echo "----------------------------------------------"
LD_LIBRARY_PATH="${BUILD_DIR}" "${BUILD_DIR}/cgc_unified_demo"

echo ""
echo "▶ 运行 cgc_engine_demo (联动验证) ..."
echo "----------------------------------------------"
LD_LIBRARY_PATH="${BUILD_DIR}" "${BUILD_DIR}/cgc_engine_demo"

echo ""
echo "✅ 全部构建并运行完成。libcgc_unified.so 可经 Python ctypes 调用："
echo "   from cgc_unified_injection import inject_unified_ir_for_role"
