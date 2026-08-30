#!/bin/bash
# Deploy CGC engine to HarmonyOS NEXT phone (Mate 70 Pro)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

LLAMA_BIN="${LLAMA_BIN:-$REPO_ROOT/src/llama.cpp/build-harmony-phone/bin}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/gguf}"
MODEL_FILE="${MODEL_FILE:-Qwen3-14B-IQ4_XS.gguf}"
PHONE_DIR="/data/local/tmp/cgc"

echo "=== Deploying CGC Engine to Mate 70 Pro ==="

if ! hdc list targets 2>/dev/null | grep -q "device"; then
    echo "Warning: No HDC device found."
    echo "请确保:"
    echo "1. DevEco Studio 5.0+ 已安装"
    echo "2. 手机开启 USB 调试 或 无线调试"
    echo "3. hdc tconn <phone-ip>:<port>"
    exit 1
fi

echo "Creating phone directory..."
hdc shell "mkdir -p $PHONE_DIR/models"

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

echo ""
echo "Pushing model..."
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "  $MODEL_FILE -> $PHONE_DIR/models/"
    hdc file send "$MODEL_DIR/$MODEL_FILE" "$PHONE_DIR/models/$MODEL_FILE"
else
    echo "  $MODEL_FILE: NOT FOUND"
    echo "请先下载: huggingface-cli download bartowski/Qwen_Qwen3-14B-GGUF Qwen3-14B-IQ4_XS.gguf --local-dir $MODEL_DIR"
    exit 1
fi

echo ""
echo "Pushing run script..."
hdc file send "$SCRIPT_DIR/run_phone.sh" "$PHONE_DIR/run_phone.sh"
hdc shell "chmod +x $PHONE_DIR/run_phone.sh"

echo ""
echo "=== Deployment complete! ==="
echo "在手机上运行:"
echo "  hdc shell"
echo "  cd $PHONE_DIR"
echo "  ./run_phone.sh -m models/$MODEL_FILE -n 128 -p 'Hello'"
