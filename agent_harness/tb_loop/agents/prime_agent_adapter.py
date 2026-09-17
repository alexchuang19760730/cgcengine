"""
Prime Agent × Terminal-Bench 适配器（tb 的 installed-agent 扩展点）。

工作原理：
- 通过 `tb run --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent`
  注册进 tb 的 AgentFactory；
- 本类继承 AbstractInstalledAgent：安装脚本把 prime-agent 装进任务容器，然后在
  容器 tmux 里以 headless 模式跑 prime-agent（-p --autonomous），模型指向 M4
  宿主机上 gemma4 的 OpenAI 兼容端点；
- 学习循环：perform_task 先把 host 侧 harness 状态（tb_loop/harness/，含
  extensions/gemma4-provider.ts 以及前几轮 /refine 产生的 skill/记忆）打包打进
  容器，任务跑完再回传，实现跨轮持续学习（agent 级 TTT）。

用法（由 run_round.sh 调用，也可手动）：
  tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent \
    -m openai/<model> \
    -k model_name=<model> -k api_key=<key> -k base_url=<url> \
    -k harness_dir=<tb_loop>/harness -k max_turns=12 -k max_tokens=30000 \
    --n-tasks 10 --output-path results/round_1 --run-id round_1
"""

from __future__ import annotations

import os
import shlex
import tarfile
import tempfile
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

# 容器内 prime-agent 的工作目录（harness 状态 + provider extension 都在这里）
CONTAINER_HARNESS_DIR = "/prime-agent-harness"
CONTAINER_ENV_SCRIPT = "/prime-agent-harness-env.sh"
CONTAINER_HARNESS_TAR = "/installed-agent/harness.tar.gz"
# host 上快取的 prime-agent release 鏡像（由 scripts/fetch-prime-agent-mirror.sh 產生）。
# 為什麼要送它進容器：同一個 URL，host 是 ~12.5 MB/s、容器內只有 ~164 KB/s（差 80 倍，
# 瓶頸是 colima NAT），而官方安裝器把 `--max-time 300` 寫死 ⇒ 59.6 MB 的 release 永遠
# 抓不完。實測後果：整段安裝 421.89s 逾時、`INSTALL_FAIL_STATUS`、agent 從未被啟動、
# `total_input_tokens = 0`，而 tb 把這個結果報成 `agent_timeout`（與模型無關）。
CONTAINER_MIRROR_TAR = "/installed-agent/prime-agent-mirror.tar.gz"
DEFAULT_MIRROR_TAR = Path.home() / ".cache" / "prime-agent-mirror.tar.gz"
# uv bundle（`fetch-prime-agent-uvbundle.sh` 產生，約 19 MB）：linux 的 uv 二進位 ＋ 一批
# pure-python wheel。容器內的 setup 腳本用它**離線**把 Python kernel 建起來 ——
# kernel 是必需品（`--autonomous` 的 `bash()` 是透過 Python REPL 暴露的），
# 而讓 prime-agent 自己 bootstrap 會卡在容器那條 ~0.7 MB/min 的 PyPI 通道上。
CONTAINER_UVBUNDLE_TAR = "/installed-agent/prime-agent-uvbundle.tar.gz"
DEFAULT_UVBUNDLE_TAR = (
    Path.home() / ".cache" / "prime-agent-uvbundle" / "prime-agent-uvbundle.tar.gz"
)


class PrimeAgentAgent(AbstractInstalledAgent):
    """在 Terminal-Bench 任务容器里跑 prime-agent，模型走宿主 M4 的 gemma4。"""

    @staticmethod
    def name() -> str:
        return "prime-agent"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_prefix: str = "local-gemma4",
        harness_dir: str | None = None,
        max_turns: int = 12,
        max_tokens: int = 30000,
        timeout_ms: int = 600000,
        max_continuations: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name or os.environ.get("TB_GEMMA4_MODEL", "gemma-4-26b-a4b-it")
        # ★ tb 會把 `-m` 的**值**當成 model_name 傳進來，而那個值傳統寫法帶 provider 前綴
        #   （`openai/<model>`）。那個前綴不該變成送給端點的 model id —— 2026-09-17 單變數實測：
        #   `local-gemma4/openai/deepseek/deepseek-flash` ⇒ 400「model or service ID … does
        #   not exist」；拿掉 `openai/` 之後同一顆容器 rc=0、正常回覆。所以在這裡剝掉它，
        #   讓本 adapter 對 tb 的兩種寫法（有／無前綴）都成立。
        if self._model_name.startswith("openai/"):
            self._model_name = self._model_name[len("openai/"):]
        self._api_key = api_key or os.environ.get("TB_GEMMA4_API_KEY", "sk-local")
        self._base_url = base_url or os.environ.get(
            "TB_GEMMA4_BASE_URL", "http://host.docker.internal:1234/v1"
        )
        self._model_prefix = model_prefix or os.environ.get("TB_MODEL_PREFIX", "local-gemma4")
        self._harness_dir = Path(harness_dir or os.environ.get("TB_HARNESS_DIR", "harness"))
        self._max_turns = int(max_turns)
        self._max_tokens = int(max_tokens)
        self._timeout_ms = int(timeout_ms)
        self._max_continuations = int(max_continuations)

    # ------------------------------------------------------------------
    # AbstractInstalledAgent 接口
    # ------------------------------------------------------------------
    @property
    def _env(self) -> dict[str, str]:
        return {
            # 把 harness 目录指到容器内注入的位置（provider extension 自动被发现）
            "PRIME_AGENT_CODING_AGENT_DIR": CONTAINER_HARNESS_DIR,
            # 模型端点（provider extension 里读取这些变量）
            "TB_GEMMA4_BASE_URL": self._base_url,
            "TB_GEMMA4_API_KEY": self._api_key,
            "TB_GEMMA4_MODEL": self._model_name,
            "TB_MODEL_PREFIX": self._model_prefix,
            # 兜底：部分 OpenAI 兼容 server 也认标准 env
            "OPENAI_API_KEY": self._api_key,
            "OPENAI_BASE_URL": self._base_url,
        }

    @property
    def _install_agent_script_path(self) -> Path:
        return Path(__file__).parent / "prime-agent-setup.sh"

    @property
    def _mirror_tar(self) -> Path:
        return Path(os.environ.get("TB_PA_MIRROR_TAR", str(DEFAULT_MIRROR_TAR)))

    @property
    def _uvbundle_tar(self) -> Path:
        return Path(os.environ.get("TB_PA_UVBUNDLE_TAR", str(DEFAULT_UVBUNDLE_TAR)))

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        model = f"{self._model_prefix}/{self._model_name}"
        cmd = (
            f"source {CONTAINER_ENV_SCRIPT} 2>/dev/null || true; "
            f"export PATH=\"$HOME/.local/bin:$PATH\"; "
            # ★★ 把 Python kernel 指到安裝腳本**離線建好**的那個 venv。
            #   不指的話 prime-agent 會走自己的 bootstrap（`uv python install 3.11`）——
            #   那需要網路，而容器對 PyPI 的吞吐實測只有 ~0.7 MB/min ⇒ 必然失敗。
            #   實測（smoke8）agent 拿到的錯誤原文：
            #     Failed to set up the Python kernel runtime.
            #     /root/.local/bin/uv python install 3.11 failed with exit code 1
            #     ... Set PRIME_AGENT_KERNEL_PYTHON to a Python with a current
            #         prime-agent-runtime and default Python packages installed
            #         to skip auto-bootstrap.
            #   为什么光「把 venv 建在 ~/.prime/agent/kernel-venv」不够：官方文件的
            #   Kernel Lifecycle 写的是
            #     1. PRIME_AGENT_KERNEL_PYTHON，当它有一个 current prime-agent-runtime；
            #     2. ~/.prime/agent/kernel-venv/bin/python，用 uv bootstrap 的；
            #   而执行档里有 `xPn = ".bootstrap-version"` / `E$s = ".bootstrap.lock"` 与
            #   常數 `bPn = 9` —— 路径 2 **要有那个 bootstrap 標記**才算 current。
            #   我们离线建出来的 venv 没有那个標記 ⇒ 被判 stale ⇒ 走 bootstrap ⇒ 失败。
            #   路径 1 不需要標記，它验证的是「venv 里有没有可用的 prime-agent-runtime」
            #   （协议版本比對；我们装的就是同一个 release 的 runtime ⇒ 相符）。
            #   ★ 代价（诚实记录）：venv 里没有官方那串 default Python packages
            #     （numpy/scipy/pandas…）⇒ prime-agent 会印
            #     `Warning: Python skills unavailable … will be disabled` —— 是 warning，
            #     `ipython`／`bash` 这两个工具仍可用（shim 只要求 `import rlm.mcp` 成功，
            #     而 rlm/mcp.py 并不 import `mcp` 那个 PyPI 包：mcp_base 是函式内惰性载入）。
            f"export PRIME_AGENT_KERNEL_PYTHON=\"$HOME/.prime/agent/kernel-venv/bin/python\"; "
            f"prime-agent -p --offline --model {shlex.quote(model)} "
            f"--autonomous "
            f"--autonomous-max-turns {self._max_turns} "
            f"--autonomous-max-tokens {self._max_tokens} "
            f"--autonomous-timeout-ms {self._timeout_ms} "
            f"--autonomous-max-continuations {self._max_continuations} "
            f"{shlex.quote(instruction)}"
        )
        return [
            TerminalCommand(
                command=cmd,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                # ★★ 必須是 True。出處（原始碼，不是推論）：
                #   terminal_bench/terminal/models.py:13  `TerminalCommand.block: bool = False`
                #   terminal_bench/terminal/tmux_session.py:296-307
                #       `send_command()` → `send_keys(block=command.block, …)`
                #   terminal_bench/agents/installed_agents/abstract_installed_agent.py:173-179
                #       `for command in run_agent_commands: session.send_command(command)`
                #       `return AgentResult(total_input_tokens=0, total_output_tokens=0)`
                #   ⇒ block=False 时 send_command 送完就返回，`perform_task` **立刻**回一个
                #     0/0 的 AgentResult，tb 随即进入测试阶段 —— **agent 从未被等待过**。
                #     实测（smoke7）：安装 9.59s 成功，agent 阶段只有 **10 秒**、
                #     `total_input_tokens = 0`、`failure_mode = test_timeout`；
                #     run.log 里那条 agent 命令之后**没有** "Blocking command completed"。
                #   ⇒ 上游的既有写法都是 True：claude_code_agent.py:64、codex_agent.py:46
                #     （都配 max_timeout_sec=inf，超时交给 harness 的 asyncio.wait_for）。
                #   ★ 这一条与「安装慢」是两个独立的缺陷，症状完全一样（0 token、timeout）：
                #     安装慢 ⇒ 预算被吃光；block=False ⇒ agent 根本没跑。修好前者不会自动修后者。
                block=True,
                append_enter=True,
            )
        ]

    # ------------------------------------------------------------------
    # 学习循环：harness 状态注入
    # ------------------------------------------------------------------
    def perform_task(self, instruction, session, logging_dir=None):
        self._ship_mirror(session)
        self._ship_uvbundle(session)
        self._ship_harness(session)
        return super().perform_task(instruction, session, logging_dir)

    def _ship_uvbundle(self, session) -> None:
        """把 host 上備好的 uv bundle（約 19 MB）送進容器。

        容器內的 setup 腳本用它**離線**建 Python kernel（`~/.prime/agent/kernel-venv`）——
        那是 `--autonomous` 的必需品：它的 `bash()` 是透過 Python REPL 暴露的，
        沒有 kernel 就等於沒有工具。而讓 prime-agent 自己 bootstrap 會卡在容器那條
        ~0.7 MB/min 的 PyPI 通道上（實測 20 分鐘跑不完）。

        沒有 bundle 時只印一行就繼續：setup 腳本會讓 prime-agent 自己做 bootstrap
        （慢，而且可能逾時）—— 但不會靜默地做出一個沒有工具的 agent。
        """
        tar_path = self._uvbundle_tar
        if not tar_path.is_file():
            print(
                f"[prime-agent] 沒有 uv bundle {tar_path} —— kernel 會由 prime-agent "
                "自己 bootstrap（容器網路慢時會逾時）。\n"
                "              產生它: bash agent_harness/tb_loop/scripts/"
                "fetch-prime-agent-uvbundle.sh"
            )
            return
        session.copy_to_container(
            tar_path,
            container_dir="/installed-agent",
            container_filename="prime-agent-uvbundle.tar.gz",
        )
        print(
            f"[prime-agent] 已注入 uv bundle {tar_path.name}（{tar_path.stat().st_size} B）"
        )

    def _ship_mirror(self, session) -> None:
        """把 host 上快取的 prime-agent release 鏡像送進容器。

        送檔走 `copy_to_container`（Docker API），**不經過容器那個 164 KB/s 的網路**；
        送進去之後由 prime-agent-setup.sh 在容器內起一個 loopback http server 供檔，
        讓官方安裝器從 `127.0.0.1` 取 —— 那是上游明文支援的 feed（見該腳本的說明）。

        沒有鏡像時只印一行就繼續：安裝會退回「直接從外網下載」的老路徑，
        容器網路慢時那條路徑會逾時失敗（可接受，比讓 agent 靜默 0 token 好讀）。
        """
        tar_path = self._mirror_tar
        if not tar_path.is_file():
            print(
                f"[prime-agent] 沒有離線鏡像 {tar_path} —— 安裝會退回直接下載。\n"
                "              產生它: bash agent_harness/tb_loop/scripts/"
                "fetch-prime-agent-mirror.sh"
            )
            return
        session.copy_to_container(
            tar_path,
            container_dir="/installed-agent",
            container_filename="prime-agent-mirror.tar.gz",
        )
        print(
            f"[prime-agent] 已注入離線鏡像 {tar_path.name}"
            f"（{tar_path.stat().st_size} B）"
        )

    def _ship_harness(self, session) -> None:
        """把 host 侧 harness（skills/memories/provider extension）打包进容器。"""
        src = self._harness_dir
        if not src.is_dir():
            return

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as fh:
            tar_path = Path(fh.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for p in sorted(src.rglob("*")):
                    if p.is_file() and p.name != ".gitkeep":
                        tar.add(p, arcname=p.relative_to(src))

            session.copy_to_container(
                tar_path,
                container_dir="/installed-agent",
                container_filename="harness.tar.gz",
            )
            session.container.exec_run(
                [
                    "sh",
                    "-c",
                    (
                        "mkdir -p /prime-agent-harness && "
                        "tar -xzf /installed-agent/harness.tar.gz -C /prime-agent-harness 2>/dev/null || true; "
                        "echo 'export PRIME_AGENT_CODING_AGENT_DIR=/prime-agent-harness' > "
                        "/prime-agent-harness-env.sh"
                    ),
                ]
            )
        finally:
            tar_path.unlink(missing_ok=True)
