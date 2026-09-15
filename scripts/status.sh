#!/bin/bash
# 系统状态检查脚本

PROJECT_ROOT="/Users/alexchuang/Documents/flashkv-devserver"

echo "=== 系统状态检查 ==="
echo ""

# Git 状态
echo "--- Git ---"
cd "$PROJECT_ROOT"
echo "当前分支: $(git branch --show-current)"
echo "最新 commit: $(git log --oneline -1)"
echo ""

# 模型文件
echo "--- 模型文件 ---"
echo "主模型:"
if [ -f "$PROJECT_ROOT/models/gguf/Wan2.2-TI2V-5B-Q4_K_M.gguf" ]; then
    ls -lh "$PROJECT_ROOT/models/gguf/Wan2.2-TI2V-5B-Q4_K_M.gguf" | awk '{print "  "$5"  "$9}'
else
    echo "  未下载"
fi

echo "VAE:"
if [ -f "$PROJECT_ROOT/models/vae/Wan2.1_VAE.safetensors" ]; then
    ls -lh "$PROJECT_ROOT/models/vae/Wan2.1_VAE.safetensors" | awk '{print "  "$5"  "$9}'
else
    echo "  未下载"
fi

echo "Text Encoder:"
if [ -f "$PROJECT_ROOT/models/text_encoder/umt5_xxl_fp8_e4m3fn_scaled.safetensors" ]; then
    ls -lh "$PROJECT_ROOT/models/text_encoder/umt5_xxl_fp8_e4m3fn_scaled.safetensors" | awk '{print "  "$5"  "$9}'
else
    echo "  未下载"
fi
echo ""

# ComfyUI
echo "--- ComfyUI ---"
if [ -f "$PROJECT_ROOT/video-gen/ComfyUI/main.py" ]; then
    echo "  已安装"
    if [ -d "$PROJECT_ROOT/video-gen/ComfyUI/custom_nodes/ComfyUI-GGUF" ]; then
        echo "  GGUF 插件: 已安装"
    else
        echo "  GGUF 插件: 未安装"
    fi
else
    echo "  未安装/克隆中"
fi
echo ""

# 服务状态
echo "--- 服务状态 ---"
LLM_PID=$(lsof -ti:8080 2>/dev/null || echo "")
VIDEO_PID=$(lsof -ti:8188 2>/dev/null || echo "")

if [ -n "$LLM_PID" ]; then
    echo "  LLM 服务 (8080): 运行中 (PID: $LLM_PID)"
else
    echo "  LLM 服务 (8080): 未运行"
fi

if [ -n "$VIDEO_PID" ]; then
    echo "  视频生成 (8188): 运行中 (PID: $VIDEO_PID)"
else
    echo "  视频生成 (8188): 未运行"
fi
echo ""

# 内存状态
echo "--- 内存 ---"
vm_stat | head -5
echo ""

echo "=== 检查完成 ==="
