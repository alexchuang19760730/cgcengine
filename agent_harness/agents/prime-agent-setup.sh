#!/usr/bin/env bash
# 在 Terminal-Bench 任务容器里安装 prime-agent。
# 由 tb 的 installed-agent 流程复制到容器 /installed-agent/install-agent.sh 后执行。
set -e

export PATH="$HOME/.local/bin:$PATH"

# 1) Node（prime-agent 的 release 二进制通常自带，但保留兜底）
if ! command -v node >/dev/null 2>&1; then
    (curl -fsSL https://deb.nodesource.com/setup_22.x | bash -) 2>/dev/null || true
    apt-get update -y >/dev/null 2>&1 && apt-get install -y nodejs >/dev/null 2>&1 || true
fi

# 2) prime-agent CLI（官方安装脚本，下载 versioned release 到 ~/.local/bin）
if ! command -v prime-agent >/dev/null 2>&1; then
    curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
fi

# 3) 验证
export PATH="$HOME/.local/bin:$PATH"
if ! command -v prime-agent >/dev/null 2>&1; then
    echo "prime-agent install failed"
    exit 1
fi
echo "prime-agent ready: $(command -v prime-agent)"
