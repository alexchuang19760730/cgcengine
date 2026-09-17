#!/usr/bin/env bash
# 在 Terminal-Bench 任务容器里安装 prime-agent。
#
# ★ 2026-09-17 实读：tb 的 installed-agent 流程把它复制到 /installed-agent/install-agent.sh 后，
#   是以 `source /installed-agent/install-agent.sh || echo 'INSTALL_FAIL_STATUS'` 执行的
#   —— 也就是**被 source**，不是当独立程序跑。两件事因此成立：
#     ① **不能用 `exit`。** exit 会把 tmux shell 一起结束，于是同一行后面的 `tmux wait -S done`
#        永远不会执行，tb 就在 `min_timeout_sec: 0.0 max_timeout_sec: inf` 上**无限等待**。
#        2026-09-17 实测：安装失败后 hang 4 分钟以上，而 `INSTALL_FAIL_STATUS` 一次都没被印出来
#        —— 它在 pane 与 run.log 里各出现 1 次，但那两次都是**被印出来的那行命令本身**，不是输出。
#     ② 失败必须让 source 的返回值非零，tb 才会走到 `||` 那一支。
#   所以：不用 exit、不用 `set -e`（sourced script 的 `set -e` 会留在调用者的 shell 里），
#   每一步显式判错，失败时用 `return 1 2>/dev/null || exit 1`（被 source 时走 return、
#   当独立程序跑时走 exit）。
#
# ★ 0) curl 是硬前置：任务映像是 `ubuntu-24-04` 精简版，**不带 curl**
#   （2026-09-17 实测 curl/wget/nc 皆无，python3 有），而 prime-agent 的官方安装脚本要 curl。
#   实测该映像 apt 可达、`apt-get install -y curl` 成功，所以这里先把它补上。
#   注：原版把 `curl … | sh` 放在 `set -e` 下却没开 pipefail —— 没有 curl 时管线的状态取 `sh`
#   的（空 stdin → 0），于是 `set -e` 不会中止，真正的错因只剩 stderr 那一行。

export PATH="$HOME/.local/bin:$PATH"

if ! command -v curl >/dev/null 2>&1; then
    echo "curl missing -- installing via apt"
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y curl >/dev/null 2>&1 || true
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "prime-agent install failed: curl unavailable (apt could not provide it)"
    return 1 2>/dev/null || exit 1
fi

# 1) Node（prime-agent 的 release 二进制通常自带，但保留兜底）
if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - || true
    { apt-get update -y && apt-get install -y nodejs; } >/dev/null 2>&1 || true
fi

# 2) prime-agent CLI（官方安装脚本，下载 versioned release 到 ~/.local/bin）
if ! command -v prime-agent >/dev/null 2>&1; then
    # ★ 2026-09-17：官方安装器会问 `Install Prime Agent v<ver>? [Y/n]`，而它优先开 /dev/tty
    #   （安装脚本 :888 `if ( : <>/dev/tty )`）—— tmux 里 /dev/tty 是有的，所以它会**等一个
    #   永远不来的 Enter**（实测卡在这个提示上，第二次 hang）。官方给的开关是
    #   PRIME_AGENT_INSTALLER_NONINTERACTIVE=1（安装脚本 :2232）；PLAIN=1 顺带关掉花屏 TUI，
    #   让容器日志可读（:263）。
    #   注：必须用 `export`。写成 `VAR=1 curl … | sh` 时那个赋值只作用于 **curl**，
    #   而安装器是在管道右侧的 sh 里跑的 —— 它会看不到这些变量（一个很容易写错、
    #   且症状与「开关没用」完全一样的地方）。
    #
    # ★★ 但 NONINTERACTIVE 只盖得住 5 个提示里的 1 个。官方安装器一共问 5 次
    #   （Native 安装 :2233 / Node.js 安装 :1078 / 加 PATH :1485 / npm 全局装 :1607 /
    #   Python runtime :1642），**只有 :2233 那一个认这个开关**，其余四个没有任何开关。
    #   实测：装了 NONINTERACTIVE 之后仍然卡在「Install Node.js and npm with
    #   standalone Node.js?」（进程只剩一个 sh 在等，~/.local/bin 空）。
    #   决定性的办法是**不给它控制终端**：用 setsid 跑，/dev/tty 打不开、stdin 也不是 tty
    #   ⇒ prime_agent_prompt_yes_no 走 `return 2`（:892），每个提示都自动「继续」。
    #   实测同一颗有 tty 的容器：`curl … | setsid sh` 一路装到
    #   `Installed Prime Agent 0.9.5 at /root/.local/share/prime-agent/bin/prime-agent`。
    export PRIME_AGENT_INSTALLER_NONINTERACTIVE=1
    export PRIME_AGENT_INSTALLER_PLAIN=1
    if command -v setsid >/dev/null 2>&1; then
        curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | setsid sh || true
    else
        curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh || true
    fi
fi

# 3) 验证
if ! command -v prime-agent >/dev/null 2>&1; then
    echo "prime-agent install failed"
    return 1 2>/dev/null || exit 1
fi
echo "prime-agent ready: $(command -v prime-agent)"
