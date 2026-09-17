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

# 宿主注入的兩個檔案（見 adapter 的 _ship_mirror／_ship_uvbundle）：
#   release 鏡像      —— 讓官方安裝器從 loopback 取檔（不走走不通的容器外網）
#   uv bundle         —— 讓步驟 3 能**離線**把 Python kernel 建起來
PA_MIRROR_TAR=/installed-agent/prime-agent-mirror.tar.gz
PA_UVBUNDLE_TAR=/installed-agent/prime-agent-uvbundle.tar.gz

if ! command -v curl >/dev/null 2>&1; then
    # ★ 这一步是**整条安装路径上唯一剩下的外网依赖**，也是唯一会「看网络脸色」的一步。
    #   实测耗时在 4s 到 ~330s 之间跳动（`apt-get update` 要抓约 10 MB 索引；而
    #   `apt-get install -y curl` **不先 update 会失败**：`E: Unable to locate package curl`，
    #   所以 update 去不掉）。
    #   ⇒ 把它计时并印出来。理由：它会直接吃掉 agent 的时间预算，而症状（0 token、timeout）
    #     与「模型解不出来」长得一样 —— 日志里有这个数字，归因就不必靠猜。
    #   （预算那一半的修法在 run_model_smoke.sh 的 --global-agent-timeout-sec。）
    echo "curl missing -- installing via apt"
    pa_apt_t0=${SECONDS}
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y curl >/dev/null 2>&1 || true
    echo "curl via apt 完成: $((SECONDS - pa_apt_t0))s"
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "prime-agent install failed: curl unavailable (apt could not provide it)"
    return 1 2>/dev/null || exit 1
fi

# 1) ★★ 2026-09-17 实测：**不要装 Node**（这一段原本是 npm 安装法的兜底，现在是纯负担）
#
#   ① 根本不需要：官方安装器的入口（install.sh:76-88）在 `PRIME_AGENT_INSTALL_METHOD=auto`
#      且 `prime_agent_native_platform` 成功时，**直接 `prime_agent_install_native` 并 return**
#      —— Node.js 那条分支（:1078 的 `Install Node.js and npm with …?` 提示）**一行都不会执行**。
#      我们用的是 native release 二进制（linux-arm64），不是 npm 全局安装。
#
#   ② 它是**安装阶段唯一剩下的外网依赖**，而且是最慢的那条路。实测（smoke6，卡死现场）：
#         Get:4 https://deb.nodesource.com/node_22.x nodistro InRelease [12.1 kB]
#         Fetched 24.1 kB in 2s (**10.7 kB/s**)
#      而 nodejs 的 deb 约 25 MB ⇒ 按这个速率要 **约 40 分钟**。tb 的 agent 预算是 360s，
#      所以安装**永远跑不完** ⇒ agent 从未被启动 ⇒ `total_input_tokens = 0`。
#
#   ③ 这件事最值得记的地方是**它的不确定性**：同一段程式、同一个任务，smoke5 装了（43.26s）
#      而 smoke6 卡了 10 分钟以上 —— 差别只在那一刻的网络。**「偶发」比「必错」更难发现**：
#      如果 smoke5 是唯一一次运行，这一段会被记成「可以工作」。
#      ⇒ 判准：安装阶段**不该有任何**「能不能跑完取决于网络」的步骤。
#
#   ⇒ 删掉整个 node 步骤，并把安装方法**钉成 `binary`**：万一哪天平台检测失败，
#     它会**当场报错退出**（`no compatible compiled archive is available`），
#     而不是静默回退到那条会挂 40 分钟的 node 路径。
export PRIME_AGENT_INSTALL_METHOD=binary

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
    # ★★ Python kernel 是**必需品**：`--autonomous` 的 bash() 是**透过 Python REPL 暴露的**，
    #   没有 kernel 就等于**没有任何工具** —— agent 自己会报：
    #     Status: blocked — no executable tool is available.
    #     - Every `ipython` call returns: `Failed to set up the Python kernel runtime.`
    #     - `bash()` is exposed only through the Python REPL, so the broken kernel
    #       also removes shell access.
    #   但**不要**让 prime-agent 自己的 bootstrap 去准备它：那段会去下 uv 0.12.15 ＋ Python ＋
    #   一批 wheel，而容器对 PyPI 的吞吐实测只有 **~0.7 MB/min**（20 分钟都跑不完；而且
    #   `UV_OFFLINE=1` 也挡不住那个停顿 —— 它根本不走 uv 的离线逻辑）。
    #   改成**下面步骤 3 自己用 uv 离线建**（host 已备好料）。
    #   ⇒ 判准：**有 uvbundle 才关它**（我们接得住）；没有 bundle 时只能让它自己做（慢但正确）。
    #   （我一度无条件设 0 来「省时间」，被实验推翻 —— 那会静默地做出一个没有工具的 agent。）
    if [ -s "${PA_UVBUNDLE_TAR}" ]; then
        export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0
    fi

    if command -v setsid >/dev/null 2>&1; then
        curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | setsid sh || true
    else
        curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh || true
    fi
fi

# 3) Python kernel（离线建 —— 见上面步骤 2 的说明：它是必需品，但不能让 prime-agent 自己做）
#
#    ★★ 2026-09-17 第二版：**改用 bundle 自带的 CPython 3.11**，并装上 prime-agent 的
#       default Python packages 清单。第一版（用映像自己的 python ＋ 只装 pure-python wheel）
#       被实验推翻了 —— prime-agent 对 KERNEL_PYTHON 有一道**硬检查**，缺清单就拒绝，
#       实测（smoke9）agent 拿到的原文：
#         PRIME_AGENT_KERNEL_PYTHON points to a Python missing default Python packages
#         (requests, httpx, yaml, tomli, dotenv, pandas, numpy, scipy, bs4, lxml, pydantic, tyro):
#         /root/.prime/agent/kernel-venv/bin/python
#       那份清单里有 numpy/scipy/pandas/lxml/pydantic-core（原生扩充）⇒ native wheel 绑 ABI
#       ⇒ 「用映像的 python ＋ 只装 pure-python」在这道要求下不成立。
#       自带 CPython 3.11 同时解决两件事：清单齐得起来，且与任务映像的 Python 版本无关
#       （顺带：3.11 正是执行档里 `pPn = "3.11"` 期望的版本）。
#
#    位置仍是官方文件写的「解析路径 2」＋ 我们显式设 KERNEL_PYTHON（见 adapter 的 agent 命令）。
#      1. PRIME_AGENT_KERNEL_PYTHON，当它有一个 current prime-agent-runtime；
#      2. ~/.prime/agent/kernel-venv/bin/python，用 uv bootstrap 的；
#      3. ~/.prime 不可写时的 XDG 位置。
#    ★ 路径 2 需要执行档里那个 bootstrap 标記（`xPn = ".bootstrap-version"` / 常数 `bPn = 9`）
#      才算 current；我们离线建出来的没有那個標記 ⇒ 一定会被判 stale ⇒ 走 bootstrap ⇒
#      因为没网而失败。所以**必须**走路径 1。
#
#    ★ 刻意**不装 mcp**（用 --no-deps 装 runtime）：rlm/mcp.py 是 MCP **client** registry，
#      只 import 标准库与 .mcp_base，而 mcp_base 对 mcp SDK 的 import 全在函式内部
#      ⇒ 实测 `import rlm.mcp` 在没有 mcp 的情况下 **OK**（这一点很重要：kernel shim 把
#      `import rlm.mcp` 放在 `bash` 的同一个 try 里，import 不过就连 bash 一起废）。
#      而装了反而坏：mcp → pyjwt[crypto] → cryptography，其原生扩充在这台 VM 会 SIGILL。
PA_KERNEL_VENV="$HOME/.prime/agent/kernel-venv"
PA_KERNEL_PY="$HOME/py311/bin/python3.11"
pa_kernel_ok=0
if [ -s "${PA_UVBUNDLE_TAR}" ]; then
    tar -xzf "${PA_UVBUNDLE_TAR}" -C "$HOME" 2>/dev/null || true
fi
if [ -x "$HOME/.local/bin/uv" ] && [ -x "${PA_KERNEL_PY}" ] && [ -d "$HOME/wheels" ]; then
    pa_runtime="$(find "$HOME/.local/share/prime-agent/releases" -maxdepth 2 -name prime-agent-runtime 2>/dev/null | head -1)"
    if [ -n "${pa_runtime}" ]; then
        export UV_OFFLINE=1
        export UV_FIND_LINKS="$HOME/wheels"
        # ① runtime + dill（--no-deps：把 mcp/cryptography 挡在外面）
        # ② prime-agent 硬检查要求的那 12 个 default packages
        # ③ 验收：rlm / dill / rlm.mcp ＋ 清单里几个关键的真实 import
        if "$HOME/.local/bin/uv" venv "${PA_KERNEL_VENV}" --python "${PA_KERNEL_PY}" >/dev/null 2>&1 &&
           "$HOME/.local/bin/uv" pip install --offline --find-links "$HOME/wheels" \
               --python "${PA_KERNEL_VENV}/bin/python" -q --no-deps dill "${pa_runtime}" >/dev/null 2>&1 &&
           "$HOME/.local/bin/uv" pip install --offline --find-links "$HOME/wheels" \
               --python "${PA_KERNEL_VENV}/bin/python" -q \
               requests httpx pyyaml tomli python-dotenv pandas numpy scipy beautifulsoup4 lxml pydantic tyro >/dev/null 2>&1 &&
           "${PA_KERNEL_VENV}/bin/python" -c 'import rlm, rlm.mcp, dill, numpy, scipy, pandas, pydantic, tyro' >/dev/null 2>&1; then
            pa_kernel_ok=1
            echo "prime-agent kernel ready: ${PA_KERNEL_VENV} ($("${PA_KERNEL_VENV}/bin/python" -V 2>&1))"
        fi
    fi
fi
if [ "${pa_kernel_ok}" != 1 ]; then
    # 不是致命错误（安装本身可能仍然成功），但表现是「agent 自己说 blocked」而不是安装失败，
    # 所以刻意大声印出来 —— 这一行是判「为什么 0 token」时最该先看的东西。
    echo "WARN: prime-agent kernel 没建起来 —— --autonomous 会没有工具（见步骤 2 的说明）" >&2
fi

# 4) 验证
if ! command -v prime-agent >/dev/null 2>&1; then
    echo "prime-agent install failed"
    return 1 2>/dev/null || exit 1
fi
echo "prime-agent ready: $(command -v prime-agent)"
