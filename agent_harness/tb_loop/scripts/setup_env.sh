#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# M4 一次性环境准备
# 前置：Docker Desktop（手动安装并启动）、gemma4 的 OpenAI 兼容 server 已在跑
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TB_LOOP_DIR"

echo "==> 1/3 Python venv + terminal-bench"
if [[ ! -x .venv/bin/python ]]; then
    python3.13 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet "terminal-bench>=0.2.0"
.venv/bin/python -c "import terminal_bench; print('   terminal-bench', terminal_bench.__file__)"

echo "==> 2/3 host prime-agent（用于 /refine 学习）"
if ! command -v prime-agent >/dev/null 2>&1; then
    curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
fi
command -v prime-agent || { echo "   prime-agent 安装失败，手动装"; exit 1; }

echo "==> 3/3 检查 Docker"
docker info >/dev/null 2>&1 || {
    echo "   WARN: Docker 不可用——请先启动 Docker Desktop（Terminal-Bench 需要）"
}

mkdir -p harness results
echo
echo "环境就绪。下一步："
echo "  1. 改 config.env（gemma4 endpoint / key / model / 任务数）"
echo "  2. ./run_round.sh"
