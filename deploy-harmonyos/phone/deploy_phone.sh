#!/bin/bash
# Deploy CGC engine to HarmonyOS NEXT phone (Mate 70 Pro)
# 通过 HDC 推送 binary + 模型到手机
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

# 默认路径
LLAMA_BIN="${LLAMA_BIN:-$REPO_ROOT/src/llama.cpp/build-harmony-phone/bin}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/gguf}"
PHONE_DIR="/data/local/tmp/cgc"

# 模型文件
MODEL_FILE="${MODEL_FILE:-Qwen3-14B-IQ4_XS.gguf}"

echo "=== Deploying CGC Engine to Mate 70 Pro ==="

# 检查 HDC 连接
echo "Checking HDC connection..."
if ! hdc list targets 2>/dev/null | grep -q "HarmonyOS\|HUAWEI\|device"; then
    echo "Warning: No HDC device found. Trying WiFi..."
    echo ""
    echo "请确保:"
    echo "1. 手机开启 USB 调试 或 无线调试"
    echo "2. DevEco Studio 5.0+ 已安装 (提供 HDC 驱动)"
    echo "3. 手机弹窗点 '允许'"
    echo ""
    echo "WiFi HDC 连接:"
    echo "  hdc tconn <phone-ip>:<port>"
    echo ""
    exit 1
fi

# 创建手机端目录
echo "Creating phone directory..."
hdc shell "mkdir -p $PHONE_DIR/models"

# 推送 binary
echo "Pushing binaries..."
for bin in llama-simple llama-server llama-speculative-simple; do
    if [ -f "$LLAMA_BIN/$bin" ]; then
        echo "  $bin -> $PHONE_DIR/"
        hdc file send "$LLAMA_BIN/$bin" "$PHONE_DIR/$bin"
        hdc shell "chmod +x $PHONE_DIR/$bin"
    else
        echo "  $bin: NOT FOUND (skipped)"
    fi
done

# 推送模型
echo ""
echo "Pushing model..."
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "  $MODEL_FILE -> $PHONE_DIR/models/"
    hdc file send "$MODEL_DIR/$MODEL_FILE" "$PHONE_DIR/models/$MODEL_FILE"
else
    echo "  $MODEL_FILE: NOT FOUND"
    echo ""
    echo "请先下载模型:"
    echo "  huggingface-cli download bartowski/Qwen_Qwen3-14B-GGUF \\"
    echo "      Qwen3-14B-IQ4_XS.gguf --local-dir $MODEL_DIR"
    echo ""
    exit 1
fi

# 推送运行脚本
echo ""
echo "Pushing run script..."
hdc file send "$SCRIPT_DIR/run_phone.sh" "$PHONE_DIR/run_phone.sh"
hdc shell "chmod +x $PHONE_DIR/run_phone.sh"

echo ""
echo "=== Deployment complete! ==="
echo ""
echo "在手机上运行:"
echo "  hdc shell"
echo "  cd $PHONE_DIR"
echo "  ./run_phone.sh -m models/$MODEL_FILE -n 128 -p 'Hello'"
echo ""
echo "或启动 HTTP server (PD 分离):"
echo "  ./llama-server -m models/$MODEL_FILE --host 0.0.0.0 --port 8080 -ngl 0 -t 6 -c 2048 --no-mmap -expert-cache 1073741824"
