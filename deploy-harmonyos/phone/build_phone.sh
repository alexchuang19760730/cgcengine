#!/bin/bash
# Build llama.cpp CGC fork for HarmonyOS NEXT phone (Mate 70 Pro)
# 需要 DevEco Studio 5.0+ 安装的 HarmonyOS NEXT NDK
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
SRC_DIR="${1:-$REPO_ROOT/src/llama.cpp}"
BUILD_DIR="$SRC_DIR/build-harmony-phone"

echo "=== Building llama.cpp CGC fork for HarmonyOS NEXT phone ==="
echo "Source: $SRC_DIR"
echo "Build:  $BUILD_DIR"

# 自动检测 HarmonyOS NEXT NDK
if [ -z "$HARMONY_NDK" ]; then
    for candidate in \
        "$HOME/AppData/Local/Huawei/Sdk/openharmony/5.0.3.500/toolchains" \
        "$HOME/AppData/Local/Huawei/Sdk/openharmony/5.0.3.xxx/toolchains" \
        "/opt/HarmonyOS/ndk/toolchains" \
        "$HARMONYOS_NDK_HOME"; do
        if [ -d "$candidate/llvm/bin" ]; then
            HARMONY_NDK="$candidate"
            break
        fi
    done
fi

if [ -z "$HARMONY_NDK" ] || [ ! -d "$HARMONY_NDK/llvm/bin" ]; then
    echo "Error: HarmonyOS NEXT NDK not found!"
    echo ""
    echo "请安装 DevEco Studio 5.0+:"
    echo "  https://developer.huawei.com/consumer/cn/download/"
    echo ""
    echo "安装后设置环境变量:"
    echo "  export HARMONY_NDK=~/AppData/Local/Huawei/Sdk/openharmony/<version>/toolchains"
    exit 1
fi

CC="$HARMONY_NDK/llvm/bin/aarch64-unknown-linux-ohos-clang"
CXX="$HARMONY_NDK/llvm/bin/aarch64-unknown-linux-ohos-clang++"

if [ ! -f "$CC" ]; then
    echo "Error: clang not found at $CC"
    ls "$HARMONY_NDK/llvm/bin/" | grep -i clang | head -10
    exit 1
fi

echo "Compiler: $CC"
echo ""

CORES=$(nproc 2>/dev/null || echo 8)
JOBS=${JOBS:-$CORES}

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake "$SRC_DIR" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -march=armv8.2-a -mtune=cortex-a720" \
    -DCMAKE_CXX_FLAGS="-O3 -march=armv8.2-a -mtune=cortex-a720" \
    -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENCL=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DMTP_SUPPORT=ON

cmake --build . -j"$JOBS" --target llama-simple llama-server llama-speculative-simple

echo ""
echo "=== Build complete! ==="
ls -la bin/llama-simple bin/llama-server bin/llama-speculative-simple 2>/dev/null
echo ""
echo "Deploy to phone:"
echo "  hdc file send bin/llama-server /data/local/tmp/"
echo "  hdc file send bin/llama-simple /data/local/tmp/"
