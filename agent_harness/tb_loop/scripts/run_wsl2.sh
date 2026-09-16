#!/usr/bin/env bash
# ============================================================
# WSL2 启动脚本：在 Linux bash 环境下跑 local_rehearsal.py
# 用法:
#   bash scripts/run_wsl2.sh hello-world
#   bash scripts/run_wsl2.sh hello-world,grid-pattern-transform
#   TASKS=hello-world bash scripts/run_wsl2.sh
# ============================================================
set -euo pipefail

# 从 Windows 侧调用: wsl -d Ubuntu-24.04 -e bash scripts/run_wsl2.sh hello-world
# 或者直接在 WSL2 里: bash scripts/run_wsl2.sh hello-world

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AH_DIR="$(dirname "$SCRIPT_DIR")"

# venv 路径
VENV="$HOME/ah"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: python3 -m venv ~/ah && source ~/ah/bin/activate && pip install terminal-bench pyyaml pytest requests"
    exit 1
fi

source "$VENV/bin/activate"

# PYTHONPATH
export PYTHONPATH="$AH_DIR:$AH_DIR/.."

# 模型 API（默认走 Mac 1240，可通过环境变量覆盖）
export SFT_API_BASE_URL="${SFT_API_BASE_URL:-http://192.168.101.87:1240/v1}"
export SFT_API_KEY="${SFT_API_KEY:-sk-local}"
export SFT_MODEL="${SFT_MODEL:-Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf}"

# 任务列表
TASKS="${1:-${TASKS:-hello-world}}"
RUN_ID="${RUN_ID:-wsl2_$(date +%Y%m%d_%H%M)}"
MAX_STEPS="${MAX_STEPS:-12}"

echo "=== WSL2 local_rehearsal ==="
echo "  Tasks: $TASKS"
echo "  Model: $SFT_MODEL"
echo "  API: $SFT_API_BASE_URL"
echo "  Run ID: $RUN_ID"
echo ""

cd "$AH_DIR"

python scripts/local_rehearsal.py \
    --data-dir datasets/terminal-bench-core-0.1.1 \
    --tasks "$TASKS" \
    --run-id "$RUN_ID" \
    --out-root results \
    --max-steps "$MAX_STEPS" \
    --rewrite-app \
    2>&1
