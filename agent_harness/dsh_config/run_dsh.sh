#!/usr/bin/env bash
# ============================================================
# DSH + agent_harness 整合启动脚本
# 用 DSH Minimal/Standard mode 跑 Terminal-Bench 任务
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_HARNESS_DIR="$(dirname "$SCRIPT_DIR")"
DSH_DIR="${AGENT_HARNESS_DIR}/../dsh"
CONFIG_FILE="${SCRIPT_DIR}/cordis.yml"

# 默认配置
DSH_MODE="${DSH_MODE:-minimal}"          # minimal | standard | multi-model
TB_TASK="${TB_TASK:-}"                    # Terminal-Bench 任务 ID (留空=交互模式)
TB_TASKS_DIR="${AGENT_HARNESS_DIR}/datasets/terminal-bench/tasks"
RESULTS_DIR="${AGENT_HARNESS_DIR}/results/dsh_$(date +%Y%m%d_%H%M%S)"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[DSH]${NC} $*"; }
warn() { echo -e "${YELLOW}[DSH]${NC} $*"; }
err() { echo -e "${RED}[DSH]${NC} $*" >&2; }

# ---- 检查 ----
if [[ ! -d "$DSH_DIR" ]]; then
    err "DSH 目录不存在: $DSH_DIR"
    err "请先运行: git clone --depth 1 https://github.com/deepseek-ai/deepseek-harness.git $DSH_DIR"
    exit 1
fi

if ! command -v node &>/dev/null; then
    err "需要 Node.js >= 22.19.0"
    exit 1
fi

# ---- 构建 DSH (如果需要) ----
if [[ ! -d "$DSH_DIR/node_modules" ]]; then
    log "安装 DSH 依赖..."
    cd "$DSH_DIR"
    if command -v pnpm &>/dev/null; then
        pnpm install
        pnpm run build
    else
        npm install
        npm run build
    fi
    cd "$SCRIPT_DIR"
fi

# ---- 创建结果目录 ----
mkdir -p "$RESULTS_DIR"

# ---- 选择预设 ----
case "$DSH_MODE" in
    minimal)
        PRESET="minimal-qwen36"
        log "使用 Minimal mode (bash + str_replace_editor)"
        ;;
    standard)
        PRESET="standard-qwen36"
        log "使用 Standard mode (完整工具集)"
        ;;
    multi-model)
        PRESET="multi-model"
        log "使用 Multi-model mode (Qwen3.6 + Gemma4)"
        ;;
    *)
        err "未知模式: $DSH_MODE (可选: minimal, standard, multi-model)"
        exit 1
        ;;
esac

# ---- 运行 DSH ----
log "启动 DSH with preset: $PRESET"
log "配置文件: $CONFIG_FILE"
log "结果目录: $RESULTS_DIR"

if [[ -n "$TB_TASK" ]]; then
    # 单任务模式
    log "运行任务: $TB_TASK"
    TASK_FILE="${TB_TASKS_DIR}/${TB_TASK}/task.md"
    if [[ ! -f "$TASK_FILE" ]]; then
        err "任务文件不存在: $TASK_FILE"
        exit 1
    fi
    
    cd "$DSH_DIR"
    npx @deepseek-ai/dsh run \
        --preset "$PRESET" \
        --config "$CONFIG_FILE" \
        --cwd "$RESULTS_DIR" \
        "$(cat "$TASK_FILE")" \
        2>&1 | tee "${RESULTS_DIR}/dsh_output.log"
else
    # 交互模式
    log "进入 DSH 交互模式 (Ctrl+C 退出)"
    cd "$DSH_DIR"
    npx @deepseek-ai/dsh web \
        --preset "$PRESET" \
        --config "$CONFIG_FILE" \
        --port 3080
fi

log "完成。结果保存在: $RESULTS_DIR"
