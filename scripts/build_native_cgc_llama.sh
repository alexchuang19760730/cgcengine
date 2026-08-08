#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 CGC Engine: Compiling native libllama.dylib with UMA Hook"
echo "=========================================================="

LLAMA_DIR="/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Backend/Llama.cpp/llama.cpp"
METAL_FILE="$LLAMA_DIR/ggml/src/ggml-metal/ggml-metal-context.m"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "[Error] Directory not found: $LLAMA_DIR"
    exit 1
fi

cd "$LLAMA_DIR"

# 1. Inject UMA Hook into ggml-metal-context.m
echo "[1/3] Injecting UMA 0-copy hook into ggml-metal-context.m..."

if [ ! -f "$METAL_FILE" ]; then
    echo "[Error] Could not find ggml-metal-context.m at $METAL_FILE"
    exit 1
fi

if grep -q "cgc_metal_set_tensor_hook" "$METAL_FILE"; then
    echo "Hook already injected in $METAL_FILE."
else
    # 建立一個暫存檔來進行修改
    cat > patch_hook.txt << 'EOF'
    // --- CGC Engine UMA 0-copy Hook ---
    extern bool cgc_skip_tensor_set;
    if (cgc_skip_tensor_set) {
        // KV Cache already injected into UMA via cgc_metal_vram_hook.mm
        // Bypassing CPU-to-GPU memcpy entirely!
        return;
    }
    // ----------------------------------
EOF

    # 找到 ggml_metal_set_tensor_async 並在下一行插入
    sed -i '' -e '/void ggml_metal_set_tensor_async(/r patch_hook.txt' "$METAL_FILE"

    # 在檔案開頭加入外部變數宣告，移除 extern "C" 因為這是 .m 檔案
    cat > patch_header.txt << 'EOF'
#include <stdbool.h>

// CGC Engine: Global flag for UMA 0-copy bypass
bool cgc_skip_tensor_set = false;

void cgc_metal_set_tensor_hook(bool skip) {
    cgc_skip_tensor_set = skip;
}
EOF

    # 將 header 插入到第一行
    sed -i '' -e '1r patch_header.txt' "$METAL_FILE"

    # 清理暫存檔
    rm patch_hook.txt patch_header.txt

    echo "Successfully patched $METAL_FILE."
fi

# 2. Configure with CMake
echo "[2/3] Configuring CMake (Metal = ON, BUILD_SHARED_LIBS = ON)..."
mkdir -p build
cd build
cmake .. -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON

# 3. Compile
echo "[3/3] Compiling libllama.dylib..."
make llama -j$(sysctl -n hw.ncpu)

echo "=========================================================="
echo "✅ Compilation complete!"
echo "   Shared library is ready at: $LLAMA_DIR/build/src/libllama.dylib"
echo "=========================================================="
