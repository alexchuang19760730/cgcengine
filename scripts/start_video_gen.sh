#!/bin/bash
# Wan 2.2 TI2V-5B 视频生成服务启动脚本
# 基于 ComfyUI + GGUF 量化模型

set -e

# 配置
PROJECT_ROOT="/Users/alexchuang/Documents/flashkv-devserver"
COMFYUI_DIR="$PROJECT_ROOT/video-gen/ComfyUI"
PORT=8188
VENV_DIR="$PROJECT_ROOT/video-gen/venv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Wan 2.2 TI2V-5B 视频生成服务启动 ===${NC}"
echo ""

# 检查 ComfyUI 是否存在
if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo -e "${RED}错误: ComfyUI 未找到 at $COMFYUI_DIR${NC}"
    echo "请先运行: git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_DIR"
    exit 1
fi

# 检查模型文件
MODEL_FILE="$PROJECT_ROOT/models/gguf/Wan2.2-TI2V-5B-Q4_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${YELLOW}警告: 主模型未找到 at $MODEL_FILE${NC}"
else
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo -e "${GREEN}✓ 主模型: $MODEL_SIZE${NC}"
fi

# 检查 VAE
VAE_FILE="$PROJECT_ROOT/models/vae/Wan2.1_VAE.safetensors"
if [ ! -f "$VAE_FILE" ]; then
    echo -e "${YELLOW}警告: VAE 未找到 at $VAE_FILE${NC}"
else
    echo -e "${GREEN}✓ VAE 已就绪${NC}"
fi

# 检查 Text Encoder
TE_FILE="$PROJECT_ROOT/models/text_encoder/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
if [ ! -f "$TE_FILE" ]; then
    echo -e "${YELLOW}警告: Text Encoder 未找到 at $TE_FILE${NC}"
else
    echo -e "${GREEN}✓ Text Encoder 已就绪${NC}"
fi

echo ""

# 检查 Python 虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}创建 Python 虚拟环境...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
pip install -q -r "$COMFYUI_DIR/requirements.txt" 2>/dev/null || true

# 检查 ComfyUI-GGUF 插件
GGUF_PLUGIN="$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF_PLUGIN" ]; then
    echo -e "${YELLOW}安装 ComfyUI-GGUF 插件...${NC}"
    git clone https://github.com/city96/ComfyUI-GGUF.git "$GGUF_PLUGIN"
    pip install -q -r "$GGUF_PLUGIN/requirements.txt" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}启动 ComfyUI 服务 (端口 $PORT)...${NC}"
echo -e "${YELLOW}Web UI: http://127.0.0.1:$PORT${NC}"
echo -e "${YELLOW}API: http://127.0.0.1:$PORT/api${NC}"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动 ComfyUI
cd "$COMFYUI_DIR"
python main.py --listen 127.0.0.1 --port $PORT --preview-method auto
