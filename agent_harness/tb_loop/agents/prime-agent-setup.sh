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
#
# ★★ 2026-09-17：优先用**宿主注入的离线镜像**，让安装器从 loopback 取档。
#   为什么非这样不可（数字都是当天实测）：
#     同一个 release tarball，host 是 12.5 MB/s（59,607,437 B / 4.55s），容器内只有
#     164 KB/s（300s 只收到 49,090,240 B 就逾时）—— 差 80 倍，瓶颈是 colima NAT。
#     而官方安装器把 `--max-time 300` **写死**（install.sh 的
#     `prime_agent_curl_download -fsSL --connect-timeout 10 --max-time 300`），
#     全档没有可调该值的环境变数 ⇒ 59.6 MB 永远抓不完。
#   后果（这才是它值得写这么长的原因）：整段安装 421.89s 后逾时、印出 `INSTALL_FAIL_STATUS`、
#     agent 从未被启动、`total_input_tokens = 0`；而 tb 把这个结果报成 `agent_timeout`
#     ⇒ 那个 Unresolved 与模型／agent 路径**毫无关系**，却长得像模型判决。
#   上游明文支持这种 feed（不是 hack）：install.sh 的
#     prime_agent_validate_download_base_url 对 `http://*` 只在
#     PRIME_AGENT_ALLOW_INSECURE_HTTP_FOR_TESTS=1 **且** 是 loopback
#     （prime_agent_is_loopback_test_base_url：只接受 http://127.0.0.1:<数字埠>）时放行。
#   实测同一颗容器：安装 421.89s → **4s**（rc=0，`prime-agent --version` = 0.9.5）。
#   镜像由 host 侧 scripts/fetch-prime-agent-mirror.sh 产生，adapter 的 _ship_mirror 送进来。
if ! command -v prime-agent >/dev/null 2>&1; then
    PA_MIRROR_TAR=/installed-agent/prime-agent-mirror.tar.gz
    PA_MIRROR_DIR=/installed-agent/prime-agent-mirror
    PA_MIRROR_PORT="${TB_PA_MIRROR_PORT:-8899}"
    pa_mirror_ok=0
    if [ -s "${PA_MIRROR_TAR}" ] && command -v python3 >/dev/null 2>&1; then
        mkdir -p "${PA_MIRROR_DIR}"
        tar -xzf "${PA_MIRROR_TAR}" -C "${PA_MIRROR_DIR}" 2>/dev/null || true
        if [ -s "${PA_MIRROR_DIR}/stable" ]; then
            # 用子壳包住 cd（不要让它外泄到后续指令），并在子壳内背景起服务
            ( cd "${PA_MIRROR_DIR}" && nohup python3 -m http.server "${PA_MIRROR_PORT}" \
                --bind 127.0.0.1 >/tmp/prime-agent-mirror-http.log 2>&1 & ) || true
            sleep 1
            if curl -sf -o /dev/null "http://127.0.0.1:${PA_MIRROR_PORT}/stable"; then
                pa_mirror_ok=1
                export PRIME_AGENT_DOWNLOAD_BASE_URL="http://127.0.0.1:${PA_MIRROR_PORT}"
                export PRIME_AGENT_ALLOW_INSECURE_HTTP_FOR_TESTS=1
                echo "prime-agent: 用宿主注入的离线镜像（loopback:${PA_MIRROR_PORT}）"
            fi
        fi
    fi
    if [ "${pa_mirror_ok}" != 1 ]; then
        echo "prime-agent: 没有可用镜像 —— 退回直接下载（容器网络慢时会逾时）"
    fi

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
    # ★★ 2026-09-17：**不要**设 PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0 来「省时间」。
    #   我试过，被实验推翻：跳过后 python kernel 起不来，而 `--autonomous` 的 bash() 是
    #   **通过 Python REPL 暴露的** ⇒ 没有 kernel 就等于**没有任何工具**。agent 自己报：
    #     Status: blocked — no executable tool is available.
    #     - Every `ipython` call returns: `Failed to set up the Python kernel runtime.
    #       uv is required to set up the Python kernel.`
    #     - `bash()` is exposed only through the Python REPL, so the broken kernel
    #       also removes shell access.
    #   而这一段确实慢：实测 **>418s**（900s 上限下跑到 6:58 仍未完成；先下 uv 0.12.15
    #   约 20MB，再下 Python 与一批 wheel，全程走那个 164 KB/s 的网络）。
    #   ⇒ 结论：kernel 是**必需品**，只是它也需要和 release 一样的「镜像／预热」待遇。
    #     在那之前，安装这一步会慢（但正确）——**不要**用关掉 kernel 来换速度。

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
