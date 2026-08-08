#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 CGC Engine: Compiling cgc_llama_cpp with Metal VRAM Hook"
echo "=========================================================="

CGC_BACKEND_DIR="/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cgc_llama_cpp"

# 1. Clone llama-cpp-python
if [ ! -d "$CGC_BACKEND_DIR" ]; then
    echo "[1/4] Cloning llama-cpp-python..."
    mkdir -p /Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Backend/CGC
    cd /Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Backend/CGC
    git clone --recursive https://github.com/abetlen/llama-cpp-python.git cgc_llama_cpp
else
    echo "[1/4] Source directory already exists. Pulling latest submodules..."
    cd $CGC_BACKEND_DIR
    git submodule update --init --recursive
fi

cd $CGC_BACKEND_DIR

# 2. Patch ggml-metal.m to inject cgc_metal_set_tensor_hook
echo "[2/4] Injecting UMA 0-copy hook into ggml-metal.m..."
METAL_FILE="vendor/llama.cpp/ggml/src/ggml-metal.m"

if grep -q "cgc_metal_set_tensor_hook" "$METAL_FILE"; then
    echo "Hook already injected in $METAL_FILE."
else
    # 找到 ggml_metal_set_tensor 函數的開頭，注入我們的攔截邏輯
    # 這裡使用 sed 在函數宣告後插入我們硬核的 UMA 0-copy bypass
    sed -i '' '/void ggml_metal_set_tensor(/a\
\    // --- CGC Engine UMA 0-copy Hook ---\n\
\    extern bool cgc_skip_tensor_set;\n\
\    if (cgc_skip_tensor_set) {\n\
\        // KV Cache already injected into UMA via cgc_metal_vram_hook.mm\n\
\        // Bypassing CPU-to-GPU memcpy entirely!\n\
\        return;\n\
\    }\n\
\    // ----------------------------------\n' "$METAL_FILE"

    # 在檔案開頭加入外部變數宣告
    sed -i '' '1i\
// CGC Engine: Global flag for UMA 0-copy bypass\n\
bool cgc_skip_tensor_set = false;\n\
\n\
extern "C" void cgc_metal_set_tensor_hook(bool skip) {\n\
    cgc_skip_tensor_set = skip;\n\
}\n' "$METAL_FILE"

    echo "Successfully patched $METAL_FILE."
fi

# 3. Rename python package to cgc_llama_cpp to avoid conflicts
echo "[3/4] Renaming package to cgc_llama_cpp..."
if [ -d "llama_cpp" ]; then
    mv llama_cpp cgc_llama_cpp
    # Replace references in setup.py
    sed -i '' 's/name="llama_cpp_python"/name="cgc_llama_cpp"/g' pyproject.toml || true
    sed -i '' 's/packages=\["llama_cpp", "llama_cpp.server"\]/packages=["cgc_llama_cpp", "cgc_llama_cpp.server"]/g' setup.py || true
    # Find and replace all import llama_cpp with import cgc_llama_cpp
    find . -type f -name "*.py" -exec sed -i '' 's/import llama_cpp/import cgc_llama_cpp/g' {} +
    find . -type f -name "*.py" -exec sed -i '' 's/from llama_cpp/from cgc_llama_cpp/g' {} +
fi

# 4. Compile and Install with Metal Acceleration
echo "[4/4] Compiling and installing with Metal Support..."
# Force uninstall existing
pip uninstall -y cgc_llama_cpp llama_cpp_python || true

# Build with Metal enabled
CMAKE_ARGS="-DGGML_METAL=on" pip install -e .

echo "=========================================================="
echo "✅ cgc_llama_cpp successfully compiled and installed!"
echo "   UMA 0-copy VRAM injection is now fully armed."
echo "=========================================================="
