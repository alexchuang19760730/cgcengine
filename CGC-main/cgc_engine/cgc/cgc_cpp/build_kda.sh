#!/bin/bash
# =============================================================================
# CGC Kimi KDA C++ 编译脚本
# =============================================================================
# 用途: 编译 KDA C++ SIMD 实现 + pybind11 binding
# 不需要 Xcode，只需要 clang++ 和 pybind11
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SRC_DIR="$SCRIPT_DIR/src"
INCLUDE_DIR="$SCRIPT_DIR/include"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     CGC Kimi KDA C++ 编译脚本（免 Xcode）                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# 创建 build 目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 检测编译器
if command -v clang++ &> /dev/null; then
    CXX=clang++
    echo "🔧 使用 clang++"
elif command -v g++ &> /dev/null; then
    CXX=g++
    echo "🔧 使用 g++"
else
    echo "❌ 未找到 C++ 编译器 (clang++ 或 g++)"
    exit 1
fi

# 检测 pybind11
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())" 2>/dev/null || echo "")
if [ -z "$PYBIND11_INCLUDE" ]; then
    echo "❌ 未找到 pybind11，请安装: pip install pybind11"
    exit 1
fi
echo "✅ pybind11: $PYBIND11_INCLUDE"

# 获取 Python include 路径
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "")
if [ ! -d "$PYTHON_INCLUDE" ] || [ ! -f "$PYTHON_INCLUDE/Python.h" ]; then
    PYTHON_INCLUDE="/opt/homebrew/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/include/python3.13"
fi
echo "✅ Python include: $PYTHON_INCLUDE"

# 获取 Python framework 路径
PYTHON_FRAMEWORK="/opt/homebrew/Frameworks/Python.framework/Versions/3.13/Python"
if [ ! -f "$PYTHON_FRAMEWORK" ]; then
    PYTHON_FRAMEWORK="/opt/homebrew/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/Python"
fi
echo "✅ Python framework: $PYTHON_FRAMEWORK"

# 编译选项
CXXFLAGS="-O3 -std=c++17 -fPIC -Wall"
INCLUDES="-I$INCLUDE_DIR -I$PYBIND11_INCLUDE -I$PYTHON_INCLUDE"

# ARM NEON 支持
if [ "$(uname -m)" = "arm64" ]; then
    echo "🔧 检测到 ARM64，启用 NEON SIMD"
    CXXFLAGS="$CXXFLAGS -D KDA_USE_NEON=1"
fi

# 源文件
SOURCES="$SRC_DIR/kda_binding.cpp"

# 输出文件
PYTHON_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null || echo ".so")
OUTPUT="kda_cpp$PYTHON_EXT"

echo ""
echo "🔨 编译中..."
echo "   源文件: $SOURCES"
echo "   输出: $OUTPUT"
echo ""

# 编译
$CXX $CXXFLAGS $INCLUDES $SOURCES -o $OUTPUT -dynamiclib "$PYTHON_FRAMEWORK"

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "✅ 编译成功！"
    echo "   输出文件: $BUILD_DIR/$OUTPUT"
    echo ""
    echo "💡 测试方法:"
    echo "   cd $BUILD_DIR"
    echo "   python3 -c \"import kda_cpp; kda = kda_cpp.KDA(); kda.init(1, 4, 64); print('✅ KDA C++ 加载成功')\""
else
    echo ""
    echo "❌ 编译失败！"
    exit 1
fi